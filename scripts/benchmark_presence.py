"""Compare the presence engine with the published AAU RainSnow RGB labels."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from importlib.metadata import version
from math import isclose
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from time import perf_counter

import cv2
import numpy as np

from scripts.fetch_datasets import DEFAULT_DIRECTORY
from smart_signals.engine import DEFAULT_CLASSES, VisionEngine
from smart_signals.evaluation import evaluate_samples
from smart_signals.geometry import box_anchor


DEFAULT_SCENES = ("Hadsundvej-2", "Egensevej-5")
SCENES = {
    "Hadsundvej-2": "day_rain",
    "Egensevej-5": "night_snow",
    "Ringvej-1": "night_wet",
}
FULL_FRAME = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
LABEL_SOURCE = "https://doi.org/10.34740/kaggle/dsv/105294"
CODE_FILES = (
    "scripts/benchmark_presence.py",
    "smart_signals/engine.py",
    "smart_signals/geometry.py",
    "smart_signals/evaluation.py",
)


def _sha256(path: Path) -> str:
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest()


def load_references(annotation_path: Path, scene: str) -> dict[int, list[dict]]:
    with annotation_path.open(encoding="utf-8-sig") as reader:
        dataset = json.load(reader)
    classes = {
        row["id"]: {"motorbike": "motorcycle"}.get(row["name"], row["name"])
        for row in dataset["categories"]
    }
    by_image = defaultdict(list)
    for annotation in dataset["annotations"]:
        name = classes[annotation["category_id"]]
        if name in DEFAULT_CLASSES and not annotation.get("iscrowd"):
            by_image[annotation["image_id"]].append((name, annotation["bbox"]))

    frames = {}
    for image in dataset["images"]:
        filename = PurePosixPath(image["file_name"])
        if filename.parent.name != scene or not filename.stem.startswith("cam1-"):
            continue
        # AAU PNG filenames use 1-based frame numbers. Confirmed against the MKVs.
        frame_index = int(filename.stem.removeprefix("cam1-")) - 1
        if frame_index < 0 or frame_index in frames:
            raise ValueError(f"Índice de fotograma de referencia inválido: {filename}")
        width, height = image["width"], image["height"]
        frames[frame_index] = [
            {"class_name": name, "box": [x / width, y / height, (x + w) / width, (y + h) / height]}
            for name, (x, y, w, h) in by_image[image["id"]]
        ]
    if not frames:
        raise ValueError(f"No hay referencias publicadas para {scene}")
    return dict(sorted(frames.items()))


def _iou(a: list[float], b: list[float]) -> float:
    overlap_x = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    overlap_y = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    overlap = overlap_x * overlap_y
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def _box_totals(references: list[dict], predictions: list[dict]) -> Counter:
    matched = set()
    for prediction in sorted(predictions, key=lambda row: row["confidence"], reverse=True):
        candidates = (
            (_iou(prediction["box"], reference["box"]), index)
            for index, reference in enumerate(references)
            if index not in matched and prediction["class_name"] == reference["class_name"]
        )
        overlap, index = max(candidates, default=(0.0, -1))
        if overlap >= 0.5:
            matched.add(index)
    return Counter(reference=len(references), predicted=len(predictions), matched=len(matched))


def _box_summary(totals: Counter) -> dict:
    return {
        **totals,
        "precision": totals["matched"] / totals["predicted"] if totals["predicted"] else None,
        "recall": totals["matched"] / totals["reference"] if totals["reference"] else None,
        "iou_threshold": 0.5,
    }


def _visible_rows(rows: list[dict], mask: np.ndarray, anchor: str) -> list[dict]:
    height, width = mask.shape
    visible = []
    for row in rows:
        x, y = box_anchor(row["box"], anchor)
        column = min(width - 1, max(0, int(x * width)))
        line = min(height - 1, max(0, int(y * height)))
        if mask[line, column] == 255:
            visible.append(row)
    return visible


def _run_scene(
    engine: VisionEngine,
    source: Path,
    condition: str,
    references: dict[int, list[dict]],
    stride: int,
    box_metrics: bool,
) -> tuple[list[dict], dict, Counter]:
    engine.reset()
    mask_path = source.with_name(f"{source.stem}-mask.png")
    mask = cv2.imdecode(np.frombuffer(mask_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or not np.array_equal(np.unique(mask), [0, 255]):
        raise RuntimeError(f"La máscara de {source.name} debe ser binaria (0 y 255)")
    started = perf_counter()
    cap = cv2.VideoCapture(str(source))
    samples = []
    totals = Counter(reference=0, predicted=0, matched=0)
    warning_counts = Counter()
    evaluated = 0
    excluded_reference = excluded_predictions = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir {source}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not isclose(fps, 20.0, abs_tol=0.05):
            raise RuntimeError(f"FPS inesperados en {source.name}: {fps}")

        for frame_index in range(max(references) + 1):
            if not cap.grab():
                raise RuntimeError(f"Faltan fotogramas en {source.name} desde {frame_index}")
            reference = references.get(frame_index)
            if reference is None and frame_index % stride:
                continue
            valid, frame = cap.retrieve()
            if not valid:
                raise RuntimeError(f"No se pudo leer {source.name} en {frame_index}")
            if frame_index == 0 and frame.shape[:2] != mask.shape:
                raise RuntimeError(f"La máscara y el video tienen tamaños distintos: {source.name}")
            source_time = frame_index / 20.0
            observation, _ = engine.process(frame, source_time)
            evaluated += 1
            if reference is None:
                continue

            truth = _visible_rows(reference, mask, engine.anchor)
            detections = [row for row in observation["detections"] if row["inside"]]
            predictions = _visible_rows(detections, mask, engine.anchor)
            excluded_reference += len(reference) - len(truth)
            excluded_predictions += len(detections) - len(predictions)
            available = observation["vehicles_present"] is not None
            samples.append({
                "scene": source.stem,
                "condition": condition,
                "frame_index": frame_index,
                "source_time_s": source_time,
                "manual_count": len(truth),
                "predicted_count": len(predictions) if available else None,
                "status": observation["status"],
                "inference_ms": observation["inference_ms"],
            })
            warning_counts.update(observation["warnings"])
            if box_metrics:
                totals.update(_box_totals(truth, predictions))
    finally:
        cap.release()

    elapsed = perf_counter() - started
    details = {
        "fps_source": fps,
        "frames_processed": evaluated,
        "elapsed_s": elapsed,
        "effective_processing_fps": evaluated / elapsed,
        "mask_coverage_of_frame": float(np.count_nonzero(mask) / mask.size),
        "reference_boxes_outside_mask": excluded_reference,
        "predicted_boxes_outside_mask": excluded_predictions,
        "status_counts_on_labeled_frames": dict(Counter(row["status"] for row in samples)),
        "warnings_on_labeled_frames": dict(warning_counts),
    }
    if box_metrics:
        details["box_metrics"] = _box_summary(totals)
    return samples, details, totals


def _write_report(output: Path, report: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False)
    temporary = Path(writer.name)
    try:
        with writer:
            json.dump(report, writer, ensure_ascii=False, indent=2, allow_nan=False)
            writer.write("\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--model", type=Path, required=True, help="Ruta a pesos locales")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--scene", choices=SCENES, action="append",
                        help="Escena a incluir. Por defecto Hadsundvej-2 y Egensevej-5.")
    parser.add_argument("--max-labels", type=int, help="Primeros N fotogramas etiquetados por escena")
    parser.add_argument("--without-tracking", action="store_true")
    parser.add_argument("--box-metrics", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stride < 1 or (args.max_labels is not None and args.max_labels < 1):
        parser.error("--stride y --max-labels deben ser enteros positivos.")
    if not args.model.is_file():
        parser.error(f"No se encontró el modelo local {args.model}.")
    model_sha256 = _sha256(args.model)
    repo = Path(__file__).resolve().parents[1]
    code_sha256 = {filename: _sha256(repo / filename) for filename in CODE_FILES}

    config = {
        "roi": FULL_FRAME,
        "anchor": "bottom_center",
        "model": str(args.model.resolve()),
        "device": args.device,
        "confidence": args.confidence,
        "image_size": args.image_size,
        "tracking": not args.without_tracking,
    }
    engine = VisionEngine(config)
    samples = []
    scene_results = {}
    box_totals = Counter(reference=0, predicted=0, matched=0)
    directory = args.directory / "aau-rainsnow"
    annotation_path = directory / "aauRainSnow-rgb.json"
    source_sha256 = {annotation_path.name: _sha256(annotation_path)}
    for scene in dict.fromkeys(args.scene or DEFAULT_SCENES):
        source = directory / f"{scene}.mkv"
        mask = directory / f"{scene}-mask.png"
        source_sha256.update({path.name: _sha256(path) for path in (source, mask)})
        references = load_references(annotation_path, scene)
        if args.max_labels is not None:
            references = dict(list(references.items())[:args.max_labels])
        print(f"{scene}: {len(references)} fotogramas etiquetados", flush=True)
        scene_samples, details, scene_boxes = _run_scene(
            engine, source, SCENES[scene], references, args.stride, args.box_metrics,
        )
        samples.extend(scene_samples)
        scene_results[scene] = details
        box_totals.update(scene_boxes)

    report = {
        "reference_source": LABEL_SOURCE,
        "reference_scope": "Región blanca de la máscara RGB de los autores para cada cámara.",
        "reference_limit": "Prueba externa con un video por condición y fotogramas etiquetados dispersos.",
        "source_sha256": source_sha256,
        "code_sha256": code_sha256,
        "environment": {"opencv": cv2.__version__, "torch": version("torch"),
                        "ultralytics": version("ultralytics")},
        "config": {**config, "model_sha256": model_sha256, "stride": args.stride,
                   "max_labels_per_scene": args.max_labels},
        "scenes": scene_results,
        "metrics": evaluate_samples(samples),
        "samples": samples,
    }
    if args.box_metrics:
        report["box_metrics"] = _box_summary(box_totals)
    _write_report(args.output, report)
    print(f"Informe guardado en {args.output}")


if __name__ == "__main__":
    main()
