"""Video time for files; bounded latest-frame acquisition for cameras."""
from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from math import isfinite
import cv2
from .storage import resolve_source, source_kind, utc_now

@dataclass
class Captured:
    frame: object
    source_time_s: float
    acquired_at: str
    received_monotonic: float
    index: int
    epoch: int = 0

def open_capture(source):
    target = resolve_source(source)
    if isinstance(target, str) and source_kind(source.strip()) == 'live':
        cap = cv2.VideoCapture(target, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 3000])
    else:
        cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        cap.release()
        raise ValueError('No se pudo abrir la fuente. Verifique archivo, conexión y credenciales.')
    return cap

def metadata(cap, source):
    def reported_int(property_id):
        value = cap.get(property_id)
        return int(value) if isfinite(value) and value > 0 else 0

    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps_fallback = not 0.1 <= reported_fps <= 240
    fps = 25.0 if fps_fallback else reported_fps
    count = reported_int(cv2.CAP_PROP_FRAME_COUNT)
    kind = source_kind(source.strip())
    duration = count / fps if kind == 'file' and count and not fps_fallback else None
    return dict(width=reported_int(cv2.CAP_PROP_FRAME_WIDTH), height=reported_int(cv2.CAP_PROP_FRAME_HEIGHT),
                fps=fps, fps_fallback=fps_fallback, frame_count=count, duration_s=duration, kind=kind)

class LiveCapture:
    """One slot, never an unbounded queue. Reconnects create a new epoch."""
    def __init__(self, source):
        self.source = source
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.state = 'connecting'
        self.epoch = 0
        self.cap = None
        self.info = None
        self.thread = threading.Thread(target=self._loop, name='camera-capture', daemon=True)
        self.thread.start()

    def _loop(self):
        origin = time.monotonic()
        index = 0
        while not self.stop_event.is_set():
            cap = None
            try:
                cap = open_capture(self.source)
                self.cap = cap
                self.info = metadata(cap, self.source)
                self.epoch += 1
                self.state = 'connected'
                while not self.stop_event.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    now = time.monotonic()
                    index += 1
                    item = Captured(frame, now - origin, utc_now(), now, index, self.epoch)
                    with self.lock:
                        self.latest = item
            except (ValueError, cv2.error):
                pass
            finally:
                if cap is not None:
                    cap.release()
                self.cap = None
            self.state = 'reconnecting'
            with self.lock:
                self.latest = None
            self.stop_event.wait(2)
        self.state = 'stopped'

    def take(self, last_index):
        with self.lock:
            item = self.latest
            return item if item is not None and item.index != last_index else None

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=9)

def encode_jpeg(frame, max_width=1280):
    if frame.shape[1] > max_width:
        frame = cv2.resize(frame, (max_width, round(frame.shape[0] * max_width / frame.shape[1])))
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('No se pudo generar la vista previa')
    return encoded.tobytes()
