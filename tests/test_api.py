import importlib
import io
import json
import os
import tempfile
import time
import unittest
import zipfile
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from smart_signals import runtime, storage


@unittest.skipUnless(find_spec('fastapi'), 'La API requiere el entorno de Studio.')
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(storage, 'HOME', self.home))
        self.enterContext(patch.object(runtime, 'HOME', self.home))
        app_module = importlib.import_module('app')
        self.manager = runtime.RunManager()
        self.enterContext(patch.object(app_module, 'HOME', self.home))
        self.enterContext(patch.object(app_module, 'manager', self.manager))
        self.client = self.enterContext(TestClient(app_module.app))
        self.config = dict(source='0', roi=[[0, 0], [1, 0], [1, 1]])

    def test_invalid_configuration_cannot_start_a_capture(self):
        for invalid in ({'sample_fps': 0}, {'image_size': 641}, {'confidence': 1.1}):
            with self.subTest(invalid=invalid):
                response = self.client.post('/api/start', json=dict(self.config, **invalid))
                self.assertEqual(response.status_code, 422)
        crossed_roi = [[0, 0], [1, 1], [0, 1], [1, 0]]
        response = self.client.post('/api/start', json=dict(self.config, roi=crossed_roi))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.client.get('/api/status').json()['state'], 'idle')
        self.assertIsNone(self.manager.worker)

    def test_machine_endpoint_uses_fresh_zero_and_expires_old_readings(self):
        self.manager._set(state='running', source={'kind': 'live'}, run_id='a' * 32,
                          observation={'vehicles_present': 0, 'status': 'valid', 'by_class': {'car': 0}})
        self.manager.last_observation_at = time.monotonic()
        response = self.client.get('/api/integration/latest')
        fresh = response.json()
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertEqual(fresh['source_mode'], 'live')
        self.assertTrue(fresh['usable'])
        self.assertEqual(fresh['vehicles_present'], 0)

        self.manager._set(observation={'vehicles_present': 2, 'status': 'degraded', 'by_class': {'car': 2}})
        warned = self.client.get('/api/integration/latest').json()
        self.assertFalse(warned['usable'])
        self.assertIsNone(warned['vehicles_present'])
        self.assertEqual(warned['observation']['vehicles_present'], 2)
        self.manager._set(observation={'vehicles_present': 0, 'status': 'valid', 'by_class': {'car': 0}})

        self.manager.last_observation_at = time.monotonic() - 2.5
        expired = self.client.get('/api/integration/latest').json()
        self.assertFalse(expired['usable'])
        self.assertIsNone(expired['vehicles_present'])
        self.assertEqual(expired['observation']['status'], 'unavailable')

        self.manager._set(state='completed', source={'kind': 'file'})
        complete = self.client.get('/api/integration/latest').json()
        self.assertFalse(complete['usable'])
        self.assertIsNone(complete['vehicles_present'])
        historical = self.client.get('/api/status').json()['observation']
        self.assertEqual(historical['vehicles_present'], 0)
        self.assertEqual(historical['status'], 'valid')

    def test_profile_and_archive_redact_credentials_and_preserve_blind_annotations(self):
        source = 'rtsp://operator:camera-pass@camera.example:8554/live?token=private-token#fragment'
        profile_response = self.client.post('/api/profiles', json={
            'name': 'Acceso',
            'config': dict(self.config, source=source),
        })
        self.assertEqual(profile_response.status_code, 200)
        profile = profile_response.json()
        safe_url = 'rtsp://camera.example:8554/live'
        self.assertEqual(profile['config']['source'], safe_url)
        persisted = self.client.get('/api/profiles').json()['profiles']
        self.assertEqual(persisted[0]['config']['source'], safe_url)

        run_id = 'b' * 32
        directory = self.home / 'runs' / run_id
        storage.write_json(directory / 'manifest.json', {
            'id': run_id, 'state': 'completed', 'config': profile['config'],
        })
        storage.write_json(directory / 'samples.json', [
            {'id': '0000', 'time_s': 0.0, 'manual_count': None, 'predicted_count': 0, 'condition': 'day_dry'},
            {'id': '0001', 'time_s': 1.0, 'manual_count': None, 'predicted_count': None, 'condition': 'day_dry'},
        ])
        endpoint = f'/api/runs/{run_id}'
        samples = self.client.get(f'{endpoint}/samples').json()['samples']
        self.assertEqual(len(samples), 2)
        self.assertTrue(all('predicted_count' not in sample for sample in samples))
        response = self.client.post(f'{endpoint}/annotations', json={
            'sample_id': '0000', 'manual_count': 2, 'condition': '   ',
        })
        self.assertEqual(response.status_code, 422)
        self.assertIsNone(storage.read_json(directory / 'samples.json')[0]['manual_count'])
        for sample in samples:
            response = self.client.post(f'{endpoint}/annotations', json={
                'sample_id': sample['id'], 'manual_count': 0, 'condition': 'day_dry',
            })
            self.assertEqual(response.status_code, 200)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(self.client.get, f'{endpoint}/download')
            second = pool.submit(self.client.get, f'{endpoint}/download')
            responses = (first.result(timeout=5), second.result(timeout=5))
        for response in responses:
            self.assertEqual(response.status_code, 200)
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                self.assertIsNone(archive.testzip())
                manifest = json.loads(archive.read('manifest.json'))
                self.assertEqual(manifest['config']['source'], safe_url)
                metrics = json.loads(archive.read('validation.json'))
                self.assertEqual(metrics['exact_agreement'], 0.5)
                self.assertEqual(metrics['unavailable'], 1)
                files = b'\n'.join(archive.read(name) for name in archive.namelist())
                for secret in (b'operator:', b'camera-pass', b'private-token', b'#fragment'):
                    self.assertNotIn(secret, files)
        self.assertEqual(list(self.home.glob('smart-signals-*.zip')), [])

    def test_cross_origin_control_requests_and_unsolicited_view_confirmation_are_rejected(self):
        response = self.client.post('/api/stop', headers={'Origin': 'https://example.invalid'})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.manager.stop_event.is_set())
        response = self.client.post('/api/confirm-view')
        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.manager.confirm_event.is_set())

    def test_run_with_live_owner_cannot_be_modified_or_exported_by_another_manager(self):
        run_id = 'c' * 32
        directory = self.home / 'runs' / run_id
        storage.write_json(directory / 'manifest.json', {
            'id': run_id, 'state': 'running', 'process_id': os.getpid(),
        })
        samples = [
            {'id': '0000', 'time_s': 0.0, 'manual_count': None, 'predicted_count': 0, 'condition': 'day_dry'},
        ]
        storage.write_json(directory / 'samples.json', samples)
        endpoint = f'/api/runs/{run_id}'
        self.assertEqual(self.client.get('/api/status').json()['state'], 'idle')
        self.assertEqual(self.client.get(f'{endpoint}/download').status_code, 409)
        response = self.client.post(f'{endpoint}/annotations', json={
            'sample_id': '0000', 'manual_count': 0, 'condition': 'day_dry',
        })
        self.assertEqual(response.status_code, 409)
        self.assertEqual(storage.read_json(directory / 'samples.json'), samples)


if __name__ == '__main__':
    unittest.main()
