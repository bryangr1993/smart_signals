"""Exercise actual OpenCV network decoding and reconnection on localhost.

No real camera is accessed. A synthetic MJPEG server disconnects after 1.5 s.
This supplements mocked lifecycle tests; it does not certify RTSP/NVR hardware.
"""
from __future__ import annotations
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart_signals.capture import LiveCapture


class Stream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            for i in range(30):
                frame = np.full((120, 160, 3), 70, np.uint8)
                cv2.rectangle(frame, (i * 3 % 120, 30), (i * 3 % 120 + 30, 60), (180, 220, 90), -1)
                cv2.putText(frame, str(i), (4, 110), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
                image = cv2.imencode('.jpg', frame)[1].tobytes()
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(image)).encode() + b'\r\n\r\n' + image + b'\r\n')
                self.wfile.flush()
                time.sleep(.05)
        except ConnectionError:
            pass
        self.close_connection = True


def main():
    server = ThreadingHTTPServer(('127.0.0.1', 0), Stream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    capture = LiveCapture(f'http://127.0.0.1:{server.server_port}/stream')
    origin = time.monotonic()
    epochs = set()
    indexes = []
    saw_reconnecting = False
    last = -1
    try:
        while time.monotonic() - origin < 15:
            item = capture.take(last)
            if item is not None:
                last = item.index
                indexes.append(last)
                epochs.add(item.epoch)
                assert item.frame.shape[:2] == (120, 160)
                assert time.monotonic() - item.received_monotonic < 1
            saw_reconnecting |= capture.state == 'reconnecting'
            if len(epochs) >= 2 and saw_reconnecting:
                break
            time.sleep(.1)
    finally:
        capture.close()
        server.shutdown()
        server.server_close()
    assert len(epochs) >= 2, 'No se observó recuperación de conexión'
    assert saw_reconnecting, 'No se observó estado de reconexión'
    assert all(b > a for a, b in zip(indexes, indexes[1:])), 'Secuencia no monotónica'
    assert not capture.thread.is_alive(), 'El hilo de captura no terminó'
    print(json.dumps(dict(test='local_mjpeg_reconnect', passed=True, samples=len(indexes),
                          epochs=sorted(epochs), elapsed_s=round(time.monotonic()-origin, 2)), indent=2))


if __name__ == '__main__':
    main()
