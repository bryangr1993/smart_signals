import cv2
import numpy as np                 # ← ¡faltaba!
import json, pathlib, os, sys

pts = []                 # puntos clicados
VIDEO = pathlib.Path(sys.argv[1])

def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))

# cargar primer frame
cap = cv2.VideoCapture(str(VIDEO))
ret, frame = cap.read()
cap.release()
if not ret:
    raise RuntimeError("No se pudo leer el video.")

# ventana e interacción
msg = "Click puntos ROI · ENTER = guardar · r = reset · ESC = salir"
cv2.namedWindow(msg)
cv2.setMouseCallback(msg, on_click)

while True:
    temp = frame.copy()
    if len(pts) > 1:
        cv2.polylines(temp, [np.array(pts)], False, (0, 255, 0), 2)
    cv2.imshow(msg, temp)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:      # ESC
        break
    if key == ord('r'):
        pts = []
    if key == 13 and len(pts) >= 3:   # ENTER
        os.makedirs("config", exist_ok=True)
        with open("config/roi.json", "w") as f:
            json.dump({"roi": pts}, f, indent=2)
        print("ROI guardado en config/roi.json")
        break

cv2.destroyAllWindows()
