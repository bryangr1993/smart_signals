"""
main.py
=======

Cuenta vehículos PRESENTES dentro de un ROI en tiempo real.

Flags principales
-----------------
--video      Ruta al .mp4 o cámara.
--roi        ROI en JSON (dibujado con roi_selector.py).
--area_min   Filtra detecciones demasiado pequeñas (px²).
--min_hits   Frames que un ID debe estar *dentro* antes de contarlo.
--max_miss   Frames que un ID puede estar *fuera* antes de restarlo.
--skip       Procesa 1 de cada N frames (acelera en CPU).
--debug      Muestra recuadros de diagnóstico (rojo/amarillo/verde).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.counter import VehicleCounter

def iou(box1, box2) -> float:
    """Intersection-over-Union de dos bboxes (x1,y1,x2,y2)."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w, inter_h = max(0, xB - xA), max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(area1 + area2 - inter_area)

def inside_ratio(box_small, box_big) -> float:
    """
    Porción del área de `box_small` que queda dentro de `box_big`.
    Cada box = (x1,y1,x2,y2).
    """
    xA = max(box_small[0], box_big[0])
    yA = max(box_small[1], box_big[1])
    xB = min(box_small[2], box_big[2])
    yB = min(box_small[3], box_big[3])

    inter_w, inter_h = max(0, xB - xA), max(0, yB - yA)
    inter_area = inter_w * inter_h
    small_area = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])
    return inter_area / small_area if small_area else 0


# --------------------------------------------------------------------------- #
# Logging básico                                                              #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s • %(levelname)s • %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vehicle counter inside an ROI")
    p.add_argument("--video", required=True, type=Path, help="Ruta al vídeo .mp4")
    p.add_argument("--roi", required=True, type=Path, help="JSON con polígono ROI")
    p.add_argument("--model", default="yolov8n.pt", help="Modelo YOLOv8")
    p.add_argument("--area_min", type=int, default=400,
                   help="Área mínima (px²) para aceptar bbox")
    p.add_argument("--min_hits", type=int, default=3,
                   help="Frames dentro antes de sumar al conteo")
    p.add_argument("--max_miss", type=int, default=7,
                   help="Frames fuera antes de restar del conteo")
    p.add_argument("--skip", type=int, default=1,
                   help="Procesar 1 de cada N frames (acelera en CPU)")
    p.add_argument("--output", type=Path, default=Path("output/counts.csv"),
                   help="CSV de salida")
    p.add_argument("--debug", action="store_true",
                   help="Muestra recuadros de diagnóstico (rojo/amarillo/verde)")
    p.add_argument("--show_conf", action="store_true",
               help="En modo --debug muestra la confianza junto al ID")
    p.add_argument("--imgsz", type=int, default=640,
               help="Lado (px) al que se redimensiona la imagen antes de YOLO")
    p.add_argument("--tracker", default="config/bytetrack.yaml",
               help="Ruta al YAML del tracker (o 'auto' para usar el default)")

    return p.parse_args()

# --------------------------------------------------------------------------- #
def draw_roi(frame: np.ndarray, poly: np.ndarray) -> None:
    cv2.polylines(frame, [poly], True, (0, 255, 0), 2)

def put_count(frame: np.ndarray, count: int) -> None:
    cv2.putText(frame, f"Vehicles in ROI: {count}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2, lineType=cv2.LINE_AA)

# --------------------------------------------------------------------------- #
def process_video(
    video: Path,
    roi_json: Path,
    model_path: str | Path,
    csv_out: Path,
    area_min: int,
    min_hits: int,
    max_miss: int,
    skip: int,
    tracker: str,
    imgsz: int = 640,
    debug: bool = False,
) -> None:

    model = YOLO(str(model_path))
    counter = VehicleCounter(roi_json, min_hits=min_hits, max_miss=max_miss)
    roi_poly = np.array(counter.roi.exterior.coords, dtype=np.int32)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    log.info("Procesando %s a %.2f FPS", video, fps)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(["timestamp", "vehicles_in_roi"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % skip:
                frame_idx += 1
                continue

            counter.start_frame()


            # construcción flexible de argumentos para el tracker
            track_kwargs = dict(
                imgsz=imgsz,
                persist=True,
                classes=[2, 3, 5, 7],   # car, bus, truck, motorcycle
                verbose=False,
            )
            if tracker != "auto":              # usa tu YAML si no es 'auto'
                track_kwargs["tracker"] = tracker

            results = model.track(frame, **track_kwargs)


            boxes = results[0].boxes
            if boxes.id is not None:
                accepted_boxes = []                       # ← bboxes ya válidos este frame
                for i, (xyxy, tid) in enumerate(
                    zip(boxes.xyxy.cpu().numpy(), boxes.id.int().cpu().numpy())
                ):

                    if tid == -1:
                        continue
                    x1, y1, x2, y2 = map(int, xyxy)
                    area = (x2 - x1) * (y2 - y1)

                    # ───────── filtro de área ─────────
                    if area < area_min:
                        if debug:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # rojo
                        continue
                    # ──────────────────────────────────

                    this_box = (x1, y1, x2, y2)

                    # ─── filtro de solapamiento (IoU) ───
                    if any(iou(this_box, prev) > 0.6 for prev in accepted_boxes):
                        if debug:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # azul
                        continue
                    # ─────────────────────────────────────

                    # --- filtro de contención: box chica casi dentro de box grande ---
                    discard = False
                    for prev in accepted_boxes:
                        r1 = inside_ratio(this_box, prev)
                        r2 = inside_ratio(prev, this_box)
                        if r1 > 0.8 or r2 > 0.8:        # 80 % o más del área se solapa
                            discard = True
                            break

                    if discard:
                        if debug:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # magenta
                        continue
                    # ----------------------------------------------------------------




                    accepted_boxes.append(this_box)
                    counter.update(int(tid), this_box)


                    if debug:
                        is_counting = tid in counter.ids_inside
                        color = (0, 255, 0) if is_counting else (0, 255, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # ---------- ID + confianza ----------
                        conf = float(boxes.conf[i])          # 'i' es el índice del bucle for
                        label_id = f"{tid}" if is_counting else f"{tid}*"
                        cv2.putText(frame, f"{label_id} ({conf:.2f})",
                                    (x1 + 3, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                        # ------------------------------------




            counter.finalize_frame()

            # overlay ROI + conteo
            draw_roi(frame, roi_poly)
            put_count(frame, counter.current_count)

            cv2.imshow("Smart Signals — Debug" if debug else "Smart Signals",
                       frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            # escribir CSV una vez por segundo
            if frame_idx % round(fps) == 0:
                writer.writerow([dt.datetime.now().isoformat(),
                                 counter.current_count])

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    log.info("CSV guardado en %s", csv_out)

# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    process_video(
        video=args.video,
        roi_json=args.roi,
        model_path=args.model,
        csv_out=args.output,
        area_min=args.area_min,
        min_hits=args.min_hits,
        max_miss=args.max_miss,
        skip=max(1, args.skip),
        tracker=args.tracker,     # ← añade esta línea
        imgsz=args.imgsz,
        debug=args.debug,
    )

if __name__ == "__main__":
    main()
