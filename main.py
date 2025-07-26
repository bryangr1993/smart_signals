"""
main.py  ·  Detección, tracking y conteo con visualización.
"""
import numpy as np
from ultralytics import YOLO
import cv2, csv, datetime, pathlib, os
from src.counter import VehicleCounter

VIDEO = "data/raw/traffic_lima.mp4"
ROI    = "config/roi.json"
LINE   = ((0, 400), (1280, 400))        # ajusta Y según tu clip

# ------------------- inicialización -------------------
counter = VehicleCounter(ROI, LINE)
model   = YOLO("yolov8n.pt")
cap     = cv2.VideoCapture(VIDEO)
fps     = cap.get(cv2.CAP_PROP_FPS) or 30

# CSV de salida
os.makedirs("output", exist_ok=True)
csv_path = pathlib.Path("output/counts.csv")
with csv_path.open("w", newline="") as f:
    writer = csv.writer(f); writer.writerow(["timestamp","in","out"])

    # ------------------- bucle principal -------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True,
                              classes=[2,3,5,7], verbose=False)

        # Proceso cada detección
        for b in results[0].boxes.data.cpu().numpy():
            x1,y1,x2,y2,_,trk_id = b.astype(int)
            counter.update(int(trk_id), (x1,y1,x2,y2))

        # ---------- overlay ----------
        # dibuja ROI
        roi_pts = [(int(x),int(y)) for x,y in counter.roi.exterior.coords]
        cv2.polylines(frame, [np.array(roi_pts)], True, (0,255,0), 2)
        # dibuja línea de conteo
        cv2.line(frame, *LINE, (0,0,255), 2)
        # muestra totales
        cv2.putText(frame, f"In: {counter.in_count}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(frame, f"Out: {counter.out_count}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("Smart Signals – Conteo", frame)
        if cv2.waitKey(1) & 0xFF == 27:      # ESC
            break

        # Cada segundo escribimos al CSV
        if cap.get(cv2.CAP_PROP_POS_FRAMES) % fps == 0:
            writer.writerow([datetime.datetime.now(),
                             counter.in_count, counter.out_count])

cap.release()
cv2.destroyAllWindows()
print("Fin del video – resultados guardados en", csv_path)
