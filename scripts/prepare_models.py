"""Download official baseline weights into the local data directory."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from smart_signals.runtime import ensure_model

if __name__ == '__main__':
    for name in (sys.argv[1:] or ['yolov8n.pt', 'yolov8s.pt']):
        print(ensure_model(name))
