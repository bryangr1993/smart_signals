"""Estimate the current vehicle presence from one frame."""

from __future__ import annotations

from math import ceil, floor
from numbers import Real
from pathlib import Path
from sys import float_info
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np

from .geometry import box_anchor, contains_point, validate_roi


DEFAULT_CLASSES = ("car", "bus", "truck", "motorcycle")
_TRACK_RESET_GAP_S = 1.0


def _is_number(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and -float_info.max <= value <= float_info.max
    )


def _fraction(config: dict, name: str, default: float) -> float:
    value = config.get(name, default)
    if not _is_number(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} debe ser un número entre 0 y 1.")
    return float(value)


def _bgr_frame(frame: np.ndarray) -> np.ndarray | None:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        return None
    if frame.ndim not in (2, 3) or min(frame.shape[:2]) < 2:
        return None
    if frame.ndim == 2 or frame.shape[2] == 1:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame if frame.shape[2] == 3 else None


def _image_health(frame: np.ndarray) -> tuple[float, float, str, list[str]]:
    height, width = frame.shape[:2]
    scale = min(1.0, 320 / max(height, width))
    if scale < 1:
        size = (max(2, round(width * scale)), max(2, round(height * scale)))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    low, high, _, _ = cv2.minMaxLoc(gray)
    if high <= 4 or high - low <= 2:
        return brightness, sharpness, "unavailable", ["Imagen negra o sin contraste observable."]

    warnings = []
    if float((gray > 245).mean()) > 0.10:
        warnings.append('Áreas extensas saturadas: revise deslumbramiento y visibilidad de la zona.')
    if brightness < 22:
        warnings.append("Imagen con poca luz.")
    elif brightness > 240:
        warnings.append("Imagen con luminosidad muy alta.")
    if sharpness < 15:
        warnings.append("Imagen con poco detalle en la muestra de calidad.")
    return brightness, sharpness, "degraded" if warnings else "valid", warnings


class VisionEngine:
    """Run YOLO and optionally associate the current detections with ByteTrack."""

    def __init__(self, config: dict) -> None:
        self.roi = validate_roi(config.get("roi"))
        self.anchor = config.get("anchor", "bottom_center")
        if self.anchor not in ("bottom_center", "center"):
            raise ValueError("El punto de referencia debe ser bottom_center o center.")
        self.model_path = config.get("model", "yolov8n.pt")
        self.device = config.get("device", "auto")
        self.image_size = config.get("image_size", 640)
        self.confidence = _fraction(config, "confidence", 0.25)
        self.iou = _fraction(config, "iou", 0.7)
        self.tracking = config.get("tracking", True)
        self.crop_to_roi = config.get("crop_to_roi", False)
        names = config.get("classes", DEFAULT_CLASSES)
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("Indica un archivo de modelo o yolov8n.pt.")
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("Indica el dispositivo de inferencia.")
        if type(self.image_size) is not int or self.image_size < 32:
            raise ValueError("El tamaño de inferencia debe ser un entero de al menos 32.")
        if not isinstance(self.tracking, bool):
            raise ValueError("El seguimiento debe estar activado o desactivado.")
        if not isinstance(self.crop_to_roi, bool):
            raise ValueError("El recorte de la zona debe estar activado o desactivado.")
        if not isinstance(names, (tuple, list)) or not names or not all(
            isinstance(name, str) and name.strip() for name in names
        ):
            raise ValueError("Selecciona al menos una clase de vehículo por nombre.")

        self.classes = tuple(dict.fromkeys(name.strip().casefold() for name in names))
        roi_x, roi_y = zip(*self.roi)
        self._roi_bounds = min(roi_x), min(roi_y), max(roi_x), max(roi_y)
        self._low_confidence = min(0.1, self.confidence / 2)
        self._model = None
        self._effective_device: str | None = None
        self._class_names: dict[int, str] = {}
        self._missing_classes: list[str] = []
        self._tracker = None
        self._tracking_error: str | None = None
        self._last_time_s: float | None = None
        self._frame_size: tuple[int, int] | None = None

    def reset(self) -> None:
        """Discard source-specific tracking history while retaining the loaded model."""
        self._tracker = None
        self._tracking_error = None
        self._last_time_s = None
        self._frame_size = None

    def process(self, frame: np.ndarray, source_time_s: float) -> tuple[dict, np.ndarray]:
        """Return a JSON-ready observation and an annotated BGR image."""
        frame = _bgr_frame(frame)
        brightness = sharpness = inference_ms = 0.0
        detections: list[dict] = []
        by_class: dict[str, int] = {}
        crop_box = None
        if frame is None:
            self.reset()
            frame = np.zeros((360, 640, 3), dtype=np.uint8)
            status, warnings = "unavailable", ["No se pudo leer el fotograma."]
        elif not _is_number(source_time_s) or source_time_s < 0:
            self.reset()
            status, warnings = "unavailable", ["El tiempo del fotograma no es válido."]
        else:
            brightness, sharpness, status, warnings = _image_health(frame)
            if status == "unavailable":
                self.reset()
            else:
                self._check_source(frame, float(source_time_s))
                crop = self._inference_crop(frame)
                if crop is not None:
                    height, width = frame.shape[:2]
                    left, top, right, bottom = crop
                    crop_box = [left / width, top / height, right / width, bottom / height]
                try:
                    rows, inference_ms, malformed = self._detect(frame, crop)
                except Exception as error:
                    self.reset()
                    status = "unavailable"
                    warnings.append(f"No se pudo ejecutar la detección. {error}")
                else:
                    if malformed:
                        warnings.append("Se omitieron detecciones con datos inválidos.")
                    if self._missing_classes:
                        missing = ", ".join(self._missing_classes)
                        warnings.append(f"Clases sin correspondencia en el modelo ({missing}).")
                    track_ids, duplicates = self._associate(rows)
                    if self._tracking_error:
                        warnings.append(self._tracking_error)
                    status = "degraded" if warnings else "valid"
                    detections, by_class = self._count(frame, rows, track_ids, duplicates)

        count = None if status == "unavailable" else sum(by_class.values())
        observation = {
            "vehicles_present": count,
            "by_class": by_class,
            "detections": detections,
            "status": status,
            "warnings": warnings,
            "inference_ms": inference_ms,
            "brightness": brightness,
            "sharpness": sharpness,
            "crop_box": crop_box,
            "effective_device": self._effective_device,
        }
        return observation, self._annotate(frame, detections, count)

    def _check_source(self, frame: np.ndarray, source_time_s: float) -> None:
        if self._last_time_s is not None:
            gap = source_time_s - self._last_time_s
            if gap < 0 or gap > _TRACK_RESET_GAP_S or frame.shape[:2] != self._frame_size:
                self.reset()
        self._last_time_s = source_time_s
        self._frame_size = frame.shape[:2]

    def _inference_crop(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        if not self.crop_to_roi:
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self._roi_bounds
        margin_x = max(32, 0.1 * (x2 - x1) * width)
        margin_y = max(32, 0.1 * (y2 - y1) * height)
        left = max(0, floor(x1 * width - margin_x))
        top = max(0, floor(y1 * height - margin_y))
        right = min(width, ceil(x2 * width + margin_x))
        bottom = min(height, ceil(y2 * height + margin_y))
        if (right - left) * (bottom - top) >= 0.9 * width * height:
            return None
        return left, top, right, bottom

    def _load_detector(self):
        if self.model_path != "yolov8n.pt" and not Path(self.model_path).exists():
            raise FileNotFoundError(f"No se encontró el modelo local {self.model_path}.")
        from ultralytics import YOLO

        return YOLO(self.model_path, task="detect")

    def _detect(
        self, frame: np.ndarray, crop: tuple[int, int, int, int] | None
    ) -> tuple[np.ndarray, float, bool]:
        if self._model is None:
            model = self._load_detector()
            names = model.names
            items = names.items() if isinstance(names, dict) else enumerate(names)
            labels = {int(key): str(name).strip().casefold() for key, name in items}
            self._class_names = {key: name for key, name in labels.items() if name in self.classes}
            self._missing_classes = [name for name in self.classes if name not in self._class_names.values()]
            self._model = model
        if not self._class_names:
            raise ValueError("El modelo no contiene las clases de vehículo seleccionadas.")

        left = top = 0
        if crop is not None:
            left, top, right, bottom = crop
            frame = frame[top:bottom, left:right]

        started_at = perf_counter()
        result = self._model.predict(
            frame,
            device=None if self.device == "auto" else self.device,
            imgsz=self.image_size,
            conf=self._low_confidence if self.tracking and not self._tracking_error else self.confidence,
            iou=self.iou,
            agnostic_nms=True,
            classes=list(self._class_names),
            verbose=False,
        )[0]
        elapsed = (perf_counter() - started_at) * 1000
        if self._effective_device is None:
            predictor = getattr(self._model, "predictor", None)
            device = getattr(predictor, "device", None)
            if device is None:
                device = getattr(self._model, "device", None)
            if device is not None:
                self._effective_device = str(device)
        if result.boxes is None:
            return np.empty((0, 6), dtype=np.float32), elapsed, False

        rows = np.array(result.boxes.cpu().numpy().data, dtype=np.float32, copy=True)
        if rows.ndim != 2 or rows.shape[1] != 6:
            raise ValueError("El modelo no devolvió cajas de detección compatibles.")
        height, width = frame.shape[:2]
        finite = np.isfinite(rows).all(axis=1)
        scores_valid = (rows[:, 4] >= 0) & (rows[:, 4] <= 1)
        rows[:, :4] = np.clip(rows[:, :4], 0, [width, height, width, height])
        if crop is not None:
            rows[:, :4] += [left, top, left, top]
        boxes_valid = (rows[:, 2] > rows[:, 0]) & (rows[:, 3] > rows[:, 1])
        valid = finite & scores_valid & boxes_valid
        selected = np.isin(rows[:, 5], list(self._class_names))
        return rows[valid & selected], elapsed, not bool(valid.all())

    def _create_tracker(self):
        from importlib.util import find_spec

        if find_spec("lap") is None:
            raise RuntimeError("Falta la dependencia lap.")
        from ultralytics.trackers.byte_tracker import BYTETracker

        config = SimpleNamespace(
            track_high_thresh=self.confidence,
            track_low_thresh=self._low_confidence,
            new_track_thresh=self.confidence,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
        )
        return BYTETracker(config, frame_rate=30)

    def _associate(self, rows: np.ndarray) -> tuple[dict[int, int], set[int]]:
        if not self.tracking or self._tracking_error:
            return {}, set()
        try:
            if self._tracker is None:
                self._tracker = self._create_tracker()
            xywh = np.empty_like(rows[:, :4])
            xywh[:, :2] = (rows[:, :2] + rows[:, 2:4]) / 2
            xywh[:, 2:4] = rows[:, 2:4] - rows[:, :2]
            detections = SimpleNamespace(xywh=xywh, conf=rows[:, 4], cls=rows[:, 5])
            tracks = np.asarray(self._tracker.update(detections))
            if not tracks.size:
                return {}, set()
            if tracks.ndim != 2 or tracks.shape[1] != 8:
                raise ValueError("Formato de ByteTrack incompatible.")

            # ByteTrack v8.3.169 returns the current detection index in the last column.
            # Use its identity only. ROI membership uses the detector's current box.
            ids: dict[int, int] = {}
            seen_ids: set[int] = set()
            duplicates: set[int] = set()
            for track in tracks:
                track_id, index = float(track[4]), float(track[-1])
                if not index.is_integer() or not 0 <= index < len(rows):
                    continue
                if not track_id.is_integer() or track_id < 0:
                    continue
                index, track_id = int(index), int(track_id)
                if index in ids:
                    continue
                if track_id in seen_ids:
                    duplicates.add(index)
                    continue
                ids[index] = track_id
                seen_ids.add(track_id)
                duplicates.discard(index)
            return ids, duplicates
        except Exception as error:
            self._tracker = None
            self._tracking_error = f"Seguimiento no disponible. Se usa solo la detección. {error}"
            return {}, set()

    def _count(
        self,
        frame: np.ndarray,
        rows: np.ndarray,
        track_ids: dict[int, int],
        duplicates: set[int],
    ) -> tuple[list[dict], dict[str, int]]:
        height, width = frame.shape[:2]
        detections = []
        by_class = dict.fromkeys(self._class_names.values(), 0)
        for index, (x1, y1, x2, y2, score, class_id) in enumerate(rows.tolist()):
            track_id = track_ids.get(index)
            if index in duplicates or (track_id is None and score < self.confidence):
                continue
            box = [x1 / width, y1 / height, x2 / width, y2 / height]
            inside = contains_point(self.roi, box_anchor(box, self.anchor))
            name = self._class_names[int(class_id)]
            if inside:
                by_class[name] += 1
            detections.append({
                "box": box,
                "class_name": name,
                "confidence": score,
                "track_id": track_id,
                "inside": inside,
            })
        return detections, by_class

    def _annotate(self, frame: np.ndarray, detections: list[dict], count: int | None) -> np.ndarray:
        annotated = frame.copy()
        height, width = frame.shape[:2]

        def pixel(x: float, y: float) -> tuple[int, int]:
            return min(width - 1, round(x * width)), min(height - 1, round(y * height))

        polygon = np.array([pixel(x, y) for x, y in self.roi], dtype=np.int32)
        cv2.polylines(annotated, [polygon], True, (32, 217, 240), 2, cv2.LINE_AA)
        for detection in detections:
            box = detection["box"]
            x1, y1 = pixel(*box[:2])
            color = (63, 204, 117) if detection["inside"] else (217, 143, 64)
            cv2.rectangle(annotated, (x1, y1), pixel(*box[2:]), color, 2)
            cv2.circle(annotated, pixel(*box_anchor(box, self.anchor)), 3, color, -1)
            label = detection["class_name"]
            if detection["track_id"] is not None:
                label += f" #{detection['track_id']}"
            cv2.putText(annotated, label, (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        label = "Sin dato" if count is None else f"Vehiculos presentes  {count}"
        cv2.putText(annotated, label, (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (10, 10, 10), 4, cv2.LINE_AA)
        cv2.putText(annotated, label, (12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        return annotated
