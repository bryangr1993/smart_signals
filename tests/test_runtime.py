import csv
import hashlib
import json
import os
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from smart_signals import runtime, storage
from smart_signals.capture import Captured


class FileCapture:
    def __init__(self, timestamps, fps):
        self.timestamps = timestamps
        self.fps = fps
        self.index = -1
        self.released = False
        self.frame = np.zeros((24, 36, 3), dtype=np.uint8)

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps[self.index]
        return {
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_COUNT: len(self.timestamps),
            cv2.CAP_PROP_FRAME_WIDTH: 36,
            cv2.CAP_PROP_FRAME_HEIGHT: 24,
        }.get(property_id, 0)

    def read(self):
        self.index += 1
        if self.index >= len(self.timestamps):
            return False, None
        return True, self.frame

    def release(self):
        self.released = True


class Camera:
    def __init__(self):
        self.frames = queue.Queue()
        self.state = 'reconnecting'
        self.closed = False
        self.info = dict(kind='live', width=0, height=0, fps=10, frame_count=0, duration_s=None)

    def take(self, last_index):
        try:
            return self.frames.get_nowait()
        except queue.Empty:
            return None

    def push(self, index, epoch, source_time_s, received_age_s=0):
        self.state = 'connected'
        frame = np.zeros((24, 36, 3), dtype=np.uint8)
        received_at = time.monotonic() - received_age_s
        self.frames.put(Captured(frame, source_time_s, storage.utc_now(), received_at, index, epoch))

    def close(self):
        self.closed = True


class Engine:
    def __init__(self, counts):
        self.counts = iter(counts)
        self.received = []
        self.reset_count = 0

    def process(self, frame, source_time_s):
        self.received.append(source_time_s)
        count = next(self.counts)
        if count == 'inference_error':
            result = runtime.unavailable('No se pudo ejecutar la detección. Dispositivo inválido.')
        elif count is None:
            result = runtime.unavailable('Imagen sin contraste')
        else:
            result = dict(vehicles_present=count, by_class={'car': count}, detections=[],
                          status='valid', warnings=[])
        return dict(result, inference_ms=1.0, brightness=80.0, sharpness=20.0), frame

    def reset(self):
        self.reset_count += 1


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(storage, 'HOME', self.home))
        self.enterContext(patch.object(runtime, 'HOME', self.home))
        self.manager = runtime.RunManager()
        model = self.home / 'fixture.pt'
        model.write_bytes(b'test model')
        self.video = self.home / 'test.mp4'
        self.video.write_bytes(b'fixed test video')
        self.enterContext(patch.object(runtime, 'ensure_model', return_value=model))
        self.enterContext(patch.object(runtime.importlib.metadata, 'version', return_value='test'))

    def start(self, engine, **overrides):
        config = dict(name='Prueba', source=str(self.video), model='yolov8n.pt',
                      roi=[[0, 0], [1, 0], [1, 1]], sample_fps=1, realtime=False,
                      condition='unknown', validation_interval_s=1)
        config.update(overrides)
        self.enterContext(patch('smart_signals.engine.VisionEngine', return_value=engine))
        self.addCleanup(self.manager.close)
        return self.home / 'runs' / self.manager.start(config)

    def wait_for(self, condition):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            snapshot = self.manager.snapshot()
            if condition(snapshot):
                return snapshot
            time.sleep(.01)
        self.fail(f'Estado inesperado {self.manager.snapshot()}')

    def finished(self):
        self.manager.worker.join(timeout=4)
        self.assertFalse(self.manager.worker.is_alive())
        return self.manager.snapshot()

    @staticmethod
    def observations(directory):
        return [json.loads(line) for line in (directory / 'observations.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_video_sampling_persists_zero_and_null_and_uses_pts(self):
        engine = Engine([0, None, 3])
        capture = FileCapture([0, 400, 1000, 1200, 2000], fps=1000)
        self.enterContext(patch.object(runtime, 'open_capture', return_value=capture))
        directory = self.start(engine)
        snapshot = self.finished()
        manifest = storage.read_json(directory / 'manifest.json')
        rows = self.observations(directory)
        with (directory / 'counts.csv').open(encoding='utf-8-sig', newline='') as handle:
            csv_rows = list(csv.DictReader(handle))

        self.assertEqual(snapshot['state'], 'completed')
        self.assertTrue(capture.released)
        self.assertEqual([row['source_time_s'] for row in rows], [0.0, 1.0, 2.0])
        self.assertEqual([row['vehicles_present'] for row in rows], [0, None, 3])
        self.assertEqual([row['vehicles_present'] for row in csv_rows], ['0', '', '3'])
        self.assertEqual(snapshot['summary']['mean_present'], 1.5)
        self.assertEqual(snapshot['summary']['valid_counts'], 2)
        self.assertTrue(manifest['source']['fps_fallback'])
        self.assertIsNone(manifest['source']['duration_s'])
        self.assertEqual(manifest['source']['sha256'], hashlib.sha256(b'fixed test video').hexdigest())
        self.assertEqual(manifest['source']['size_bytes'], len(b'fixed test video'))
        self.assertEqual(rows[-1]['progress'], 1.0)

    def test_broken_pts_uses_previous_clock_without_rewinding(self):
        capture = FileCapture([1000, float('nan'), 1200], fps=10)
        self.enterContext(patch.object(runtime, 'open_capture', return_value=capture))
        directory = self.start(Engine([1, 1, 1]), sample_fps=10)
        self.assertEqual(self.finished()['state'], 'completed')
        self.assertEqual([row['source_time_s'] for row in self.observations(directory)], [1.0, 1.1, 1.2])
        self.assertTrue(storage.read_json(directory / 'manifest.json')['timestamp_fallback'])

    def test_reconnection_keeps_cut_null_and_waits_for_view_confirmation(self):
        camera = Camera()
        engine = Engine([0, 2])
        self.enterContext(patch.object(runtime, 'LiveCapture', return_value=camera))
        directory = self.start(engine, source=' rtsp://operator:secret@camera.example/feed?token=private ', sample_fps=5)
        self.wait_for(lambda status: status['state'] == 'reconnecting')
        camera.push(1, 1, 0.25, received_age_s=1)
        fresh = self.wait_for(lambda status: (status['observation'] or {}).get('vehicles_present') == 0)
        self.assertGreaterEqual(fresh['observation']['age_ms'], 1000)
        self.assertEqual((fresh['source']['width'], fresh['source']['height']), (36, 24))
        camera.state = 'reconnecting'
        self.wait_for(lambda status: status['state'] == 'reconnecting')
        camera.push(2, 2, 2.0)
        pending = self.wait_for(lambda status: status['state'] == 'awaiting_confirmation')
        self.assertIsNone(pending['observation']['vehicles_present'])
        self.assertEqual(engine.received, [0.25])
        self.manager.confirm_event.set()
        camera.push(3, 2, 2.25)
        self.wait_for(lambda status: (status['observation'] or {}).get('vehicles_present') == 2)
        self.manager.stop()
        snapshot = self.finished()
        self.assertEqual(snapshot['state'], 'stopped')
        self.assertTrue(camera.closed)
        self.assertGreaterEqual(engine.reset_count, 1)
        rows = self.observations(directory)
        self.assertEqual([row['vehicles_present'] for row in rows], [None, 0, None, None, 2])
        self.assertEqual([row['sequence'] for row in rows], list(range(1, len(rows) + 1)))
        with (directory / 'counts.csv').open(encoding='utf-8-sig', newline='') as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertTrue(json.loads(csv_rows[2]['warnings']))
        manifest = (directory / 'manifest.json').read_text(encoding='utf-8')
        self.assertIn('rtsp://camera.example/feed', manifest)
        self.assertNotIn('secret', manifest)
        self.assertNotIn('token=private', manifest)
        source_metadata = json.loads(manifest)['source']
        self.assertEqual((source_metadata['width'], source_metadata['height'], source_metadata['fps']), (36, 24, 10))

    def test_missing_dependency_finishes_as_error(self):
        self.enterContext(patch.object(runtime.importlib.metadata, 'version',
                                        side_effect=runtime.importlib.metadata.PackageNotFoundError('torch')))
        directory = self.start(Engine([]))
        snapshot = self.finished()
        self.assertEqual(snapshot['state'], 'error')
        self.assertIsNone(snapshot['observation']['vehicles_present'])
        self.assertIn('torch', snapshot['message'])
        self.assertEqual(storage.read_json(directory / 'manifest.json')['state'], 'error')

    def test_repeated_inference_errors_abort_without_misclassifying_black_frames(self):
        capture = FileCapture([0, 1000, 2000, 3000, 4000, 5000, 6000], fps=1)
        self.enterContext(patch.object(runtime, 'open_capture', return_value=capture))
        failure = 'inference_error'
        engine = Engine([failure, failure, None, failure, failure, failure, 1])
        directory = self.start(engine)
        snapshot = self.finished()
        self.assertEqual(snapshot['state'], 'error')
        self.assertIn('Dispositivo inválido', snapshot['message'])
        self.assertEqual(len(self.observations(directory)), 6)
        self.assertEqual(engine.received[-1], 5)

    def test_opening_another_manager_keeps_the_live_process_manifest(self):
        active = self.home / 'runs' / 'active' / 'manifest.json'
        orphan = self.home / 'runs' / 'orphan' / 'manifest.json'
        storage.write_json(active, {'state': 'running', 'process_id': os.getpid()})
        storage.write_json(orphan, {'state': 'running'})
        runtime.RunManager()
        self.assertEqual(storage.read_json(active)['state'], 'running')
        self.assertEqual(storage.read_json(orphan)['state'], 'interrupted')

    def test_age_limit_hides_late_live_observation(self):
        self.manager._set(state='running', source={'kind': 'live'},
                          observation={'vehicles_present': 4, 'status': 'valid'})
        self.manager.last_observation_at = time.monotonic() - 2.5
        observation = self.manager.snapshot()['observation']
        self.assertGreater(observation['age_ms'], 2000)
        self.assertIsNone(observation['vehicles_present'])
        self.assertEqual(observation['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
