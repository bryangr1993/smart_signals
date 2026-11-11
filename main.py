"""Command-line entry point using the same engine as the local interface."""
from __future__ import annotations
import argparse
import json
import shutil
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Smart Signals: vehículos presentes en un polígono')
    parser.add_argument('--serve', action='store_true', help='Abrir servidor local en puerto 8765')
    parser.add_argument('--video', help='Video local, URL de cámara o índice de webcam')
    parser.add_argument('--roi', type=Path, help='JSON de zona normalizada o roi.json legado en píxeles')
    parser.add_argument('--model', default='yolov8s.pt', choices=['yolov8n.pt', 'yolov8s.pt'])
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', '0'])
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--sample-fps', type=float, default=5)
    parser.add_argument('--anchor', default='bottom_center', choices=['bottom_center','center'])
    parser.add_argument('--no-tracking', action='store_true')
    parser.add_argument('--crop-to-roi', action='store_true', help='Concentrar análisis en la zona más un margen')
    parser.add_argument('--realtime', action='store_true', help='Reproducir al ritmo original')
    parser.add_argument('--save-annotated', action='store_true')
    parser.add_argument('--output', type=Path, help='Copia adicional del CSV al terminar')
    parser.add_argument('--name', default='Análisis desde terminal')
    args = parser.parse_args()
    if args.serve or not args.video:
        import uvicorn
        uvicorn.run('app:app', host='127.0.0.1', port=8765, access_log=False)
        return
    if not args.roi:
        parser.error('--roi es obligatorio para analizar un video')
    from app import StartRequest, manager
    from smart_signals.storage import HOME
    info = manager.preview(args.video)
    doc = json.loads(args.roi.read_text(encoding='utf-8-sig'))
    roi = doc['roi'] if isinstance(doc, dict) else doc
    if any(max(point) > 1 for point in roi):
        roi = [[x / info['width'], y / info['height']] for x, y in roi]
    config = StartRequest(source=args.video, name=args.name, roi=roi, anchor=args.anchor,
                          model=args.model, device=args.device, image_size=args.imgsz,
                          confidence=args.confidence, sample_fps=args.sample_fps,
                          realtime=args.realtime, tracking=not args.no_tracking,
                          crop_to_roi=args.crop_to_roi,
                          save_annotated=args.save_annotated).model_dump()
    run_id = manager.start(config)
    try:
        while manager.worker.is_alive():
            time.sleep(.5)
    except KeyboardInterrupt:
        manager.close()
    result = manager.snapshot()
    print(json.dumps({k: result.get(k) for k in ('state', 'message', 'run_id', 'summary')}, ensure_ascii=False, indent=2))
    directory = HOME / 'runs' / run_id
    print('Resultados:', directory)
    if args.output and (directory / 'counts.csv').exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            parser.error('La ruta --output ya existe; los resultados originales se conservaron en la sesión')
        shutil.copyfile(directory / 'counts.csv', args.output)
    if result['state'] == 'error':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
