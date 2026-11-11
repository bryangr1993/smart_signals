"""Evaluate still-image presence in one fixed near-stop lane of AAU MTID."""

import argparse
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np

from scripts.benchmark_presence import _sha256, _write_report
from scripts.fetch_datasets import DEFAULT_DIRECTORY, Asset, _download, _verified
from smart_signals.engine import DEFAULT_CLASSES, VisionEngine
from smart_signals.evaluation import evaluate_samples
from smart_signals.geometry import box_anchor, contains_point, validate_roi


SOURCE = "https://www.kaggle.com/datasets/andreasmoegelmose/multiview-traffic-intersection-dataset"
LICENSE = "https://creativecommons.org/licenses/by/4.0/"
PAPER = "https://doi.org/10.1109/ITSC45102.2020.9294694"
_DOWNLOAD = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "andreasmoegelmose/multiview-traffic-intersection-dataset/"
)
WIDTH, HEIGHT = 1024, 640
ROI_PIXELS = [(0, 468), (121, 380), (181, 486), (0, 623)]
ROI = validate_roi([(x / WIDTH, y / HEIGHT) for x, y in ROI_PIXELS])
ANCHOR = "bottom_center"
SAMPLE_LIMIT = 300
MANIFEST = "mtid/samples-v1.json"
_SOURCE_PATH = re.compile(r"Infrastructure/\d+/seq3-infra_\d{7}\.jpg")


def _url(source_path: str) -> str:
    return f"{_DOWNLOAD}{quote(source_path, safe='')}?datasetVersionNumber=1"


ANNOTATIONS = Asset(
    "mtid/infrastructure-mscoco.json",
    _url("infrastructure-mscoco.json"),
    16_832_831,
    "a5111f1454f5834660135eb585631db9af46971bf6922262f38528a36bfa764f",
    "infrastructure-mscoco.json",
)


def _sample_plan(dataset: dict) -> tuple[list[dict], dict]:
    images = dataset["images"]
    ids = Counter(row["id"] for row in images)
    annotated_ids = {row["image_id"] for row in dataset["annotations"]}
    ordered = sorted(images, key=lambda row: Path(row["file_name"]).name)
    eligible = [
        row for row in ordered
        if ids[row["id"]] == 1 and row["id"] in annotated_ids
    ]
    sample_count = min(SAMPLE_LIMIT, len(eligible))
    if not sample_count:
        raise ValueError("MTID no contiene imágenes elegibles.")
    if sample_count == 1:
        selected = eligible
    else:
        ranks = [round(index * (len(eligible) - 1) / (sample_count - 1)) for index in range(sample_count)]
        selected = [eligible[rank] for rank in ranks]
    result = []
    for row in selected:
        source_path = row["file_name"]
        if not _SOURCE_PATH.fullmatch(source_path) or (row["width"], row["height"]) != (WIDTH, HEIGHT):
            raise ValueError(f"Ruta o dimensiones inesperadas en MTID: {source_path}")
        result.append({
            "image_id": row["id"],
            "source_path": source_path,
            "filename": f"mtid/frames/{Path(source_path).name}",
            "url": _url(source_path),
        })
    details = {
        "published_entries": len(images),
        "excluded_ambiguous_id": [row["file_name"] for row in ordered if ids[row["id"]] > 1],
        "excluded_without_annotations": [
            row["file_name"] for row in ordered
            if ids[row["id"]] == 1 and row["id"] not in annotated_ids
        ],
        "eligible_entries": len(eligible),
        "rule": "Posiciones uniformes ordenadas por número de cuadro entre las entradas elegibles.",
    }
    return result, details


def _decode(data: bytes, filename: str) -> np.ndarray:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError(f"La respuesta no es un JPEG: {filename}")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[:2] != (HEIGHT, WIDTH):
        raise ValueError(f"Cuadro MTID inválido: {filename}")
    return frame


def _fetch_jpeg(entry: dict) -> bytes:
    limit = 2 * 1024 * 1024
    request = Request(entry["url"], headers={"User-Agent": "SmartSignals/1.0 research dataset"})
    with urlopen(request, timeout=25) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Respuesta demasiado grande: {entry['source_path']}")
    stream = io.BytesIO(data)
    if is_zipfile(stream):
        with ZipFile(stream) as archive:
            members = [member for member in archive.infolist() if not member.is_dir()]
            if (
                len(members) != 1 or members[0].file_size > limit
                or Path(members[0].filename).name != Path(entry["filename"]).name
            ):
                raise ValueError(f"ZIP inesperado: {entry['source_path']}")
            data = archive.read(members[0])
    return data


def _save_image(directory: Path, entry: dict, cached: dict | None) -> dict:
    target = directory / entry["filename"]
    if cached and _verified(target, Asset(
        entry["filename"], entry["url"], cached["size"], cached["sha256"],
    )):
        return {**entry, "size": cached["size"], "sha256": cached["sha256"]}
    data = _fetch_jpeg(entry)
    _decode(data, entry["filename"])
    checksum = hashlib.sha256(data).hexdigest()
    if cached and checksum != cached["sha256"]:
        raise ValueError(f"El archivo publicado no coincide con su SHA-256: {entry['source_path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = NamedTemporaryFile(dir=target.parent, delete=False)
    temporary = Path(writer.name)
    try:
        with writer:
            writer.write(data)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {**entry, "size": len(data), "sha256": checksum}


def _prepare(directory: Path, plan: list[dict], details: dict) -> None:
    path = directory / MANIFEST
    cached = {}
    if path.exists():
        previous = json.loads(path.read_bytes())
        cached = {entry["url"]: entry for entry in previous["images"]}
    print(f"Preparando {len(plan)} JPEG oficiales de MTID...", flush=True)
    with ThreadPoolExecutor(max_workers=5) as pool:
        images = list(pool.map(
            lambda entry: _save_image(directory, entry, cached.get(entry["url"])),
            plan,
        ))
    manifest = {
        "source": SOURCE,
        "paper": PAPER,
        "license": LICENSE,
        "dataset_version": 1,
        "annotations": ANNOTATIONS._asdict(),
        "selection": details,
        "roi_pixels": ROI_PIXELS,
        "roi_scope": "Carril cercano inferior izquierdo hacia el cruce, alrededor de su línea de parada.",
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "anchor": ANCHOR,
        "images": images,
    }
    _write_report(path, manifest)
    print(f"Manifiesto verificado en {path}", flush=True)


def _reference_counts(dataset: dict) -> dict[int, int]:
    counts = defaultdict(int)
    aliases = {"motorbike": "motorcycle", "lorry": "truck"}
    classes = {
        row["id"] for row in dataset["categories"]
        if aliases.get(row["name"], row["name"]) in DEFAULT_CLASSES
    }
    for row in dataset["annotations"]:
        if row.get("iscrowd") or row["category_id"] not in classes:
            continue
        x, y, width, height = row["bbox"]
        box = [
            min(1.0, max(0.0, coordinate))
            for coordinate in (x / WIDTH, y / HEIGHT, (x + width) / WIDTH, (y + height) / HEIGHT)
        ]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        if contains_point(ROI, box_anchor(box, ANCHOR)):
            counts[row["image_id"]] += 1
    return counts


def _run(directory: Path, dataset: dict, plan: list[dict], model: Path, output: Path) -> None:
    path = directory / MANIFEST
    manifest = json.loads(path.read_bytes())
    images = manifest["images"]
    if (
        manifest["roi_pixels"] != [list(point) for point in ROI_PIXELS]
        or [{key: row[key] for key in plan[0]} for row in images] != plan
    ):
        raise ValueError("El manifiesto no coincide con el ROI y el plan de muestreo. Ejecuta --prepare.")
    for row in images:
        if not _verified(directory / row["filename"], Asset(
            row["filename"], row["url"], row["size"], row["sha256"],
        )):
            raise ValueError(f"JPEG ausente o inválido: {row['filename']}. Ejecuta --prepare.")
    repo = Path(__file__).resolve().parents[1]
    code = (
        "scripts/benchmark_mtid.py", "scripts/benchmark_presence.py", "scripts/fetch_datasets.py",
        "smart_signals/engine.py", "smart_signals/geometry.py", "smart_signals/evaluation.py",
    )
    code_sha256 = {name: _sha256(repo / name) for name in code}
    model_sha256 = _sha256(model)
    manifest_sha256 = _sha256(path)
    counts = _reference_counts(dataset)
    config = {
        "roi": ROI,
        "anchor": ANCHOR,
        "model": str(model.resolve()),
        "device": "cpu",
        "confidence": 0.25,
        "image_size": 960,
        "tracking": False,
        "crop_to_roi": False,
    }
    engine = VisionEngine(config)
    samples = []
    print(f"Evaluando {len(images)} JPEG sin tracking...", flush=True)
    for row in images:
        frame = _decode((directory / row["filename"]).read_bytes(), row["filename"])
        observation, _ = engine.process(frame, 0.0)
        samples.append({
            "image_id": row["image_id"],
            "filename": row["filename"],
            "condition": "day_clear_view",
            "manual_count": counts.get(row["image_id"], 0),
            "predicted_count": observation["vehicles_present"],
            "status": observation["status"],
            "warnings": observation["warnings"],
            "inference_ms": observation["inference_ms"],
        })
    report = {
        "reference_source": SOURCE,
        "reference_scope": manifest["roi_scope"],
        "reference_box_rule": "Recorte al cuadro antes de calcular el centro inferior.",
        "reference_limit": "Cuadros espaciados de una sola cámara; la observación es por imagen.",
        "manifest": str(path),
        "manifest_sha256": manifest_sha256,
        "annotation_sha256": ANNOTATIONS.sha256,
        "code_sha256": code_sha256,
        "environment": {"opencv": cv2.__version__, "torch": version("torch"),
                        "ultralytics": version("ultralytics")},
        "config": {**config, "model_sha256": model_sha256},
        "status_counts": dict(Counter(row["status"] for row in samples)),
        "metrics": evaluate_samples(samples),
        "samples": samples,
    }
    _write_report(output, report)
    print(f"Informe guardado en {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--prepare", action="store_true", help="Descargar y verificar el subconjunto.")
    parser.add_argument("--model", type=Path, help="Ruta al modelo YOLOv8s preentrenado.")
    parser.add_argument("--output", type=Path, help="Ruta del informe JSON.")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    directory = args.directory.resolve()
    if directory.is_relative_to(repo):
        parser.error("La carpeta de datos debe estar fuera del repositorio.")
    if not args.prepare and (not args.model or not args.model.is_file() or not args.output):
        parser.error("Para evaluar indica --model con un archivo local y --output.")
    annotation_path = directory / ANNOTATIONS.filename
    if not _verified(annotation_path, ANNOTATIONS):
        if not args.prepare:
            parser.error("Las etiquetas están ausentes o dañadas. Ejecuta --prepare.")
        _download(ANNOTATIONS, annotation_path)
    dataset = json.loads(annotation_path.read_bytes())
    plan, details = _sample_plan(dataset)
    if args.prepare:
        _prepare(directory, plan, details)
    else:
        _run(directory, dataset, plan, args.model, args.output)


if __name__ == "__main__":
    main()
