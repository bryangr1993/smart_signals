from ultralytics import YOLO
import cv2, sys, pathlib

MODEL = "yolov8n.pt"                   # modelo ligero (se descarga la 1.ª vez)
CLASSES = [2, 3, 5, 7]                 # car, bus, truck, motorcycle

model = YOLO(MODEL)
video_path = pathlib.Path(sys.argv[1])

cap = cv2.VideoCapture(str(video_path))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    res = model.track(frame, persist=True, classes=CLASSES, verbose=False)
    cv2.imshow("Detections", res[0].plot())
    if cv2.waitKey(1) & 0xFF == 27:    # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
