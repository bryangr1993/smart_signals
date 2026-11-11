"""Single local analysis worker, independent of the browser lifecycle."""
from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import os
import tempfile
import threading
import time
import urllib.request
import uuid
from collections import deque
from math import isfinite
from pathlib import Path

import cv2
import psutil

from .capture import Captured, LiveCapture, encode_jpeg, metadata, open_capture
from .storage import HOME, code_fingerprint, folders, public_source, read_json, resolve_source, source_kind, utc_now, write_json

CSV_FIELDS = ['sequence', 'source_time_s', 'acquired_at_utc', 'processed_at_utc', 'vehicles_present',
              'status', 'inference_ms', 'latency_from_receive_ms', 'brightness', 'sharpness', 'by_class', 'warnings']
MAX_LIVE_AGE_S = 2.0
LATENCY_WINDOW = 1024


def unavailable(reason):
    return dict(vehicles_present=None, by_class={}, detections=[], status='unavailable', warnings=[reason])


def write_observation(observation, json_file, csv_writer):
    json_file.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + '\n')
    row = {key: observation.get(key) for key in CSV_FIELDS}
    for key in ('by_class', 'warnings'):
        row[key] = json.dumps(row[key], ensure_ascii=False)
    csv_writer.writerow(row)


def live_run_owner(manifest):
    process_id = manifest.get('process_id')
    if type(process_id) is not int or process_id <= 0 or not psutil.pid_exists(process_id):
        return False
    started_at = manifest.get('process_started_at')
    if not isinstance(started_at, (float, int)):
        return True
    try:
        return abs(psutil.Process(process_id).create_time() - started_at) < 1
    except psutil.AccessDenied:
        return True
    except psutil.NoSuchProcess:
        return False


def ensure_model(name):
    folders()
    if name not in ('yolov8n.pt', 'yolov8s.pt'):
        path = Path(name)
        if not path.is_file():
            raise ValueError('Modelo no encontrado')
        return path
    path = HOME / 'models' / name
    if not path.exists():
        url = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/' + name
        temp = None
        try:
            with urllib.request.urlopen(url, timeout=60) as response, tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=path.stem, suffix='.download', delete=False,
            ) as target:
                temp = Path(target.name)
                while data := response.read(1024 * 1024):
                    target.write(data)
            if temp.stat().st_size < 1_000_000:
                raise ValueError('Descarga del modelo incompleta')
            if not path.exists():
                try:
                    temp.replace(path)
                except PermissionError:
                    if not path.is_file():
                        raise
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
    return path


class RunManager:
    def __init__(self):
        folders()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.confirm_event = threading.Event()
        self.worker = None
        self.frame = None
        self.preview_frame = None
        self.last_observation_at = None
        self.history = deque(maxlen=600)
        self.state = {'state': 'idle', 'message': 'Listo para configurar una fuente',
                      'run_id': None, 'observation': None, 'source': None, 'config': None}
        # A prior terminated process must never appear to still be operating.
        for path in (HOME / 'runs').glob('*/manifest.json'):
            doc = read_json(path)
            if doc and doc.get('state') in ('loading', 'running', 'reconnecting', 'awaiting_confirmation') and not live_run_owner(doc):
                doc.update(state='interrupted', ended_at=utc_now(), message='La aplicación se cerró durante la sesión')
                write_json(path, doc)

    def preview(self, source):
        source = source.strip()
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError('Detenga la sesión actual antes de cambiar la fuente')
        cap = open_capture(source)
        try:
            info = metadata(cap, source)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError('La fuente se abrió pero no entregó una imagen')
            info.update(width=frame.shape[1], height=frame.shape[0], preview_url='/api/preview.jpg', source=public_source(source))
            with self.lock:
                self.preview_frame = encode_jpeg(frame, max_width=1920)
            return info
        finally:
            cap.release()

    def snapshot(self):
        with self.lock:
            result = copy.deepcopy(self.state)
            result['frame_available'] = self.frame is not None
            result['history'] = list(self.history)
            obs = result.get('observation')
            if obs and self.last_observation_at is not None:
                obs['age_ms'] = round((time.monotonic() - self.last_observation_at) * 1000)
                source = result.get('source') or {}
                if result['state'] == 'running' and source.get('kind') == 'live' and obs['age_ms'] > MAX_LIVE_AGE_S * 1000:
                    obs.update(unavailable('Lectura vencida: esperando una imagen reciente'))
            if obs and result['state'] == 'running' and self.stop_event.is_set():
                obs.update(unavailable('La sesión se está deteniendo'))
            return result

    def start(self, config):
        from .geometry import validate_roi
        config = dict(config, source=config['source'].strip())
        validate_roi(config['roi'])
        with self.lock:
            if self.worker and self.worker.is_alive():
                raise ValueError('Ya hay una sesión en curso')
            self.stop_event.clear()
            self.confirm_event.clear()
            self.history.clear()
            self.frame = None
            self.last_observation_at = None
            run_id = uuid.uuid4().hex
            safe = dict(config, source=public_source(config['source']))
            self.state = dict(state='loading', message='Preparando modelo y fuente…', run_id=run_id,
                              config=safe, observation=None, source={})
            self.worker = threading.Thread(target=self._run, args=(run_id, dict(config)), daemon=True, name='vision-worker')
            self.worker.start()
            return run_id

    def stop(self):
        self.stop_event.set()
        self.confirm_event.set()
        return {'stopping': bool(self.worker and self.worker.is_alive())}

    def close(self):
        self.stop()
        if self.worker:
            self.worker.join(timeout=12)

    def _set(self, **changes):
        with self.lock:
            self.state.update(changes)

    def _run(self, run_id, config):
        directory = HOME / 'runs' / run_id
        manifest = dict(schema_version=1, id=run_id, name=config['name'], created_at=utc_now(), state='loading')
        samples = []
        valid_counts = count_sum = 0
        max_present = None
        latencies = deque(maxlen=LATENCY_WINDOW)
        observed = 0
        extra_records = signal_losses = 0
        detection_failures = 0
        cap = live = writer_video = None
        wall_start = time.monotonic()
        state, message = 'completed', 'Procesamiento finalizado'
        try:
            from .engine import VisionEngine

            directory.mkdir()
            (directory / 'samples').mkdir()
            source = config['source']
            is_live = source_kind(source) == 'live'
            manifest.update(
                config=dict(config, source=public_source(source)),
                process_id=os.getpid(),
                process_started_at=psutil.Process().create_time(),
                code_sha256=code_fingerprint(),
                clock='local_receipt_monotonic' if is_live else 'video_pts_with_fps_fallback',
                note='Conteo presente, no aforo acumulado. Calidad de captura no equivale a precisión.',
            )
            try:
                manifest['versions'] = {p: importlib.metadata.version(p) for p in ('ultralytics', 'torch', 'opencv-python', 'numpy')}
            except importlib.metadata.PackageNotFoundError as error:
                raise ValueError(f'Falta la dependencia {error.name}.') from error
            write_json(directory / 'manifest.json', manifest)
            model = ensure_model(config['model'])
            manifest['model_sha256'] = hashlib.sha256(model.read_bytes()).hexdigest()
            engine = VisionEngine(dict(config, model=str(model)))
            if is_live:
                live = LiveCapture(source)
                info = dict(kind='live', width=0, height=0, fps=0, duration_s=None)
            else:
                cap = open_capture(source)
                info = metadata(cap, source)
                video_path = Path(resolve_source(source))
                before = video_path.stat()
                self._set(message='Registrando la huella del video local…')
                with video_path.open('rb') as video_file:
                    info['sha256'] = hashlib.file_digest(video_file, 'sha256').hexdigest()
                after = video_path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('El video cambió mientras se preparaba. Termine de copiarlo y vuelva a iniciar.')
                info['size_bytes'] = after.st_size
            manifest.update(state='running', source=info)
            write_json(directory / 'manifest.json', manifest)
            self._set(state='loading', message='Preparando el detector y la primera imagen…', source=info)
            next_sample_time = 0.0
            next_frame_time = 0.0
            next_health_time = 0.0
            last_index, epoch = -1, 0
            last_timestamp = -1.0
            frame_index = 0
            dimensions = None
            replay_origin = None
            next_preview_at = 0.0
            last_annotated = None
            interval = 1 / config['sample_fps']
            with (directory / 'counts.csv').open('w', newline='', encoding='utf-8-sig') as csv_file, (directory / 'observations.jsonl').open('w', encoding='utf-8') as json_file:
                csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
                csv_writer.writeheader()
                while not self.stop_event.is_set():
                    if live:
                        item = live.take(last_index)
                        if item is None:
                            now = time.monotonic()
                            if now >= next_health_time and live.state != 'connected':
                                extra_records += 1
                                signal_losses += 1
                                health = unavailable('Sin señal de cámara')
                                health.update(source_time_s=None, sequence=observed+extra_records,
                                              processed_at_utc=utc_now(), processed_frames=observed, progress=None)
                                write_observation(health, json_file, csv_writer)
                                csv_file.flush(); json_file.flush()
                                next_health_time = now + 1
                                with self.lock:
                                    self.last_observation_at = now
                                    self.state.update(state='reconnecting', message='Reconectando cámara…', observation=health)
                                    if self.history:
                                        self.history.append(dict(t=self.history[-1]['t'], count=None, status='unavailable'))
                            self.stop_event.wait(.025)
                            continue
                        last_index = item.index
                        if item.source_time_s + 1e-6 < next_frame_time:
                            continue
                        next_frame_time = item.source_time_s + interval
                        if epoch and item.epoch != epoch:
                            engine.reset()
                            self.confirm_event.clear()
                            epoch = item.epoch
                            extra_records += 1
                            health = unavailable('Verificar encuadre')
                            health.update(source_time_s=item.source_time_s, sequence=observed+extra_records,
                                          acquired_at_utc=item.acquired_at, processed_at_utc=utc_now(),
                                          processed_frames=observed, progress=None)
                            write_observation(health, json_file, csv_writer)
                            csv_file.flush(); json_file.flush()
                            with self.lock:
                                self.frame = encode_jpeg(item.frame)
                                self.last_observation_at = item.received_monotonic
                                self.history.append(dict(t=item.source_time_s, count=None, status='unavailable'))
                                self.state.update(
                                    state='awaiting_confirmation', observation=health,
                                    message='Cámara recuperada: confirme que el encuadre coincide con el polígono',
                                )
                        else:
                            epoch = item.epoch
                        if self.state['state'] == 'awaiting_confirmation':
                            with self.lock:
                                self.frame = encode_jpeg(item.frame)
                            if not self.confirm_event.is_set():
                                continue
                            engine.reset()
                            self._set(state='running', message='Encuadre confirmado; análisis reanudado')
                        if live.info:
                            info = live.info
                    else:
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            if frame_index == 0:
                                raise ValueError('No se pudo decodificar el video')
                            if info['frame_count'] and frame_index < info['frame_count'] - max(2, int(info['fps'])):
                                raise ValueError('El video terminó antes de lo esperado; posible archivo dañado')
                            break
                        timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
                        if not isfinite(timestamp) or timestamp < 0 or (frame_index and timestamp <= last_timestamp):
                            timestamp = last_timestamp + 1 / info['fps'] if frame_index else 0.0
                            manifest['timestamp_fallback'] = True
                        last_timestamp = timestamp
                        item = Captured(frame, timestamp, utc_now(), time.monotonic(), frame_index)
                        frame_index += 1
                        if timestamp + 1e-6 < next_frame_time:
                            continue
                        next_frame_time += interval
                        if next_frame_time < timestamp:
                            next_frame_time = timestamp + interval
                        if config['realtime']:
                            if replay_origin is None:
                                replay_origin = time.monotonic() - timestamp
                            self.stop_event.wait(max(0, replay_origin + timestamp - time.monotonic()))
                            if self.stop_event.is_set():
                                break
                    current_dimensions = item.frame.shape[:2]
                    if dimensions and current_dimensions != dimensions:
                        raise ValueError('Cambió la resolución de la fuente. Revise el encuadre y vuelva a configurar la zona.')
                    dimensions = current_dimensions
                    if live and time.monotonic() - item.received_monotonic > MAX_LIVE_AGE_S:
                        continue
                    height, width = dimensions
                    if (info.get('width'), info.get('height')) != (width, height):
                        info = dict(info, width=width, height=height)
                    manifest['source'] = info
                    result, annotated = engine.process(item.frame, item.source_time_s)
                    observed += 1
                    detection_error = next(
                        (warning for warning in result.get('warnings', [])
                         if warning.startswith('No se pudo ejecutar la detección.')),
                        None,
                    )
                    detection_failures = detection_failures + 1 if detection_error else 0
                    latency = (time.monotonic() - item.received_monotonic) * 1000
                    if live and latency > MAX_LIVE_AGE_S * 1000:
                        result.update(vehicles_present=None, by_class={}, detections=[], status='unavailable')
                        result.setdefault('warnings', []).append('La lectura superó 2 segundos de antigüedad')
                    duration = info.get('duration_s')
                    frame_count = info.get('frame_count')
                    progress = None
                    if duration:
                        progress = min(1.0, item.source_time_s / duration)
                    elif not live and frame_count:
                        progress = min(1.0, frame_index / frame_count)
                    result.update(sequence=observed+extra_records, source_time_s=round(item.source_time_s, 6),
                                  acquired_at_utc=item.acquired_at, processed_at_utc=utc_now(),
                                  latency_from_receive_ms=round(latency, 2), processed_frames=observed,
                                  progress=progress,
                                  processing_fps=round(observed / max(.001, time.monotonic() - wall_start), 2),
                                  age_ms=0)
                    count = result.get('vehicles_present')
                    if count is not None:
                        valid_counts += 1
                        count_sum += count
                        max_present = count if max_present is None else max(max_present, count)
                    latencies.append(latency)
                    write_observation(result, json_file, csv_writer)
                    if detection_failures >= 3:
                        raise ValueError(detection_error)
                    if observed % 10 == 0:
                        csv_file.flush(); json_file.flush()
                    if item.source_time_s >= next_sample_time and len(samples) < 240:
                        sample_id = f'{len(samples):04d}'
                        (directory / 'samples' / (sample_id + '.jpg')).write_bytes(encode_jpeg(item.frame, max_width=1920))
                        samples.append(dict(id=sample_id, time_s=item.source_time_s, manual_count=None,
                                            predicted_count=count, condition=config['condition']))
                        write_json(directory / 'samples.json', samples)
                        next_sample_time += config['validation_interval_s']
                        if next_sample_time <= item.source_time_s:
                            next_sample_time = item.source_time_s + config['validation_interval_s']
                    last_annotated = annotated
                    preview = None
                    if time.monotonic() >= next_preview_at:
                        preview = encode_jpeg(annotated)
                        next_preview_at = time.monotonic() + .2
                    with self.lock:
                        if preview is not None:
                            self.frame = preview
                        self.last_observation_at = item.received_monotonic if live else time.monotonic()
                        self.state.update(state='running', message='Analizando vehículos presentes', observation=result, source=info)
                        self.history.append(dict(t=item.source_time_s, count=count, status=result['status']))
                    if config.get('save_annotated'):
                        if writer_video is None:
                            h, w = annotated.shape[:2]
                            writer_video = cv2.VideoWriter(str(directory / 'annotated.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), config['sample_fps'], (w, h))
                            if not writer_video.isOpened():
                                raise ValueError('No se pudo crear el video de diagnóstico')
                        writer_video.write(annotated)
            if self.stop_event.is_set():
                state, message = 'stopped', 'Sesión detenida; resultados conservados'
            if last_annotated is not None:
                with self.lock:
                    self.frame = encode_jpeg(last_annotated)
        except Exception as exc:
            state = 'error'
            # Avoid exposing source credentials in transport-library errors.
            message = str(exc) if isinstance(exc, ValueError) else f'No se pudo completar el análisis ({type(exc).__name__}). Revise fuente, modelo y dispositivo.'
        finally:
            import numpy as np

            for resource, release_method in ((cap, 'release'), (live, 'close'), (writer_video, 'release')):
                if resource is None:
                    continue
                try:
                    getattr(resource, release_method)()
                except Exception:
                    if state != 'error':
                        state, message = 'error', 'No se pudo cerrar la fuente o el video de diagnóstico.'
            summary = dict(processed_frames=observed, missing_signal_records=signal_losses,
                           unavailable_records=observed-valid_counts+extra_records,
                           valid_counts=valid_counts, mean_present=count_sum / valid_counts if valid_counts else None,
                           max_present=max_present,
                           latency_p95_ms=float(np.percentile(latencies, 95)) if latencies else None,
                           latency_p95_window_frames=len(latencies),
                           latency_p95_window_limit=LATENCY_WINDOW,
                           elapsed_s=round(time.monotonic()-wall_start, 3),
                           validation_samples=len(samples), note='Media y máximo de presencia, no vehículos únicos ni aforo.')
            manifest.update(state=state, message=message, ended_at=utc_now(), summary=summary)
            try:
                write_json(directory / 'summary.json', summary)
                write_json(directory / 'samples.json', samples)
                write_json(directory / 'manifest.json', manifest)
            except (OSError, ValueError):
                state, message = 'error', 'No se pudieron guardar los resultados de la sesión.'
            with self.lock:
                self.state.update(state=state, message=message, summary=summary)
                if state == 'error':
                    self.state['observation'] = unavailable(message)
