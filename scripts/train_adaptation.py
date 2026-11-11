"""Prepare the AAU RainSnow adaptation split and train a YOLOv8s candidate."""

import argparse
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path, PurePosixPath
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile

import cv2
import numpy as np


CACHE = Path(os.environ.get("SMART_SIGNALS_HOME", Path.home() / ".cache" / "smart-signals"))
SOURCE = "https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow/"
ANNOTATIONS_SHA256 = "a3f90c9b7329e9bfc6ecac59d5f71e7583ea961ecaa19ecc7d7ffdbf91a8284a"
SPLITS = {
    "train": ("Hjorringvej-3", "Hasserisvej-2", "Ostre-3"),
    "val": ("Hobrovej-1",),
}
RESERVED_LOCATIONS = {"Egensevej", "Hadsundvej", "Ringvej"}
CLASSES = ("car", "bus", "truck", "motorcycle")
COCO_VEHICLE_IDS = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
PNG_HEADER = bytes.fromhex("89504e470d0a1a0a")
MAX_RESPONSE = 4 * 1024**2


def sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_png(path: Path, channels: int) -> np.ndarray:
    data = path.read_bytes()
    if not data.startswith(PNG_HEADER):
        raise ValueError(f"No es un PNG {path}")
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), channels)
    if frame is None or frame.shape[:2] != (480, 640):
        raise ValueError(f"PNG RGB AAU ilegible o de tamaño inesperado {path}")
    return frame


def write_png(path: Path, frame: np.ndarray) -> None:
    valid, data = cv2.imencode(".png", frame)
    if not valid:
        raise ValueError(f"No se pudo preparar {path}")
    path.write_bytes(data.tobytes())


def download_png(raw: Path, name: str) -> None:
    target = raw / name
    if target.is_file():
        read_png(target, cv2.IMREAD_COLOR)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        f"{SOURCE}{quote(name, safe='')}?datasetVersionNumber=1",
        headers={"User-Agent": "SmartSignals/1.0 AAU research"},
    )
    for attempt in range(4):
        try:
            with urlopen(request, timeout=40) as response:
                data = response.read(MAX_RESPONSE + 1)
            if len(data) > MAX_RESPONSE:
                raise ValueError(f"Respuesta demasiado grande {name}")
            if is_zipfile(BytesIO(data)):
                with ZipFile(BytesIO(data)) as archive:
                    members = [item for item in archive.infolist() if not item.is_dir()]
                    if len(members) != 1 or members[0].file_size > MAX_RESPONSE:
                        raise ValueError(f"ZIP inesperado {name}")
                    if PurePosixPath(members[0].filename).name != target.name:
                        raise ValueError(f"Nombre de miembro inesperado {name}")
                    data = archive.read(members[0])
            if not data.startswith(PNG_HEADER):
                raise ValueError(f"La fuente no devolvió un PNG {name}")
            temporary = target.with_suffix(".part")
            temporary.write_bytes(data)
            read_png(temporary, cv2.IMREAD_COLOR)
            temporary.replace(target)
            return
        except (OSError, ValueError):
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def mask_and_labels(
    frame: np.ndarray, mask: np.ndarray, annotations: list[dict], class_ids: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    height, width = mask.shape
    visible = mask.copy()
    labels = []
    for annotation in annotations:
        category = annotation["category_id"]
        if category not in class_ids:
            continue
        x, y, w, h = annotation["bbox"]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        x_center = (x1 + x2) / 2
        anchor_x, anchor_y = min(int(x_center), width - 1), min(int(y2), height - 1)
        if not mask[anchor_y, anchor_x] or annotation.get("iscrowd"):
            # Avoid leaving a visible fragment of an excluded box as a negative.
            visible[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = False
            continue
        labels.append((class_ids[category], x_center / width, (y1 + y2) / (2 * height),
                       (x2 - x1) / width, (y2 - y1) / height))
    masked = frame.copy()
    masked[~visible] = 114
    return masked, visible, labels


def prepare(work: Path, annotation_path: Path, weights: Path, args: argparse.Namespace) -> None:
    if sha256(annotation_path) != ANNOTATIONS_SHA256:
        raise ValueError("El JSON de anotaciones no corresponde a AAU RainSnow RGB v1.")
    class_names = CLASSES
    if args.recipe == "v2":
        os.environ.setdefault("YOLO_CONFIG_DIR", str(work / "ultralytics-config"))
        from ultralytics import YOLO

        names = YOLO(str(weights)).names
        if len(names) != 80 or any(names.get(index) != name for name, index in COCO_VEHICLE_IDS.items()):
            raise ValueError("La receta v2 requiere los 80 nombres COCO del peso YOLOv8s oficial.")
        class_names = tuple(names[index] for index in range(80))

    dataset = json.loads(annotation_path.read_text(encoding="utf-8"))
    split_of = {clip: split for split, clips in SPLITS.items() for clip in clips}
    images = [image for image in dataset["images"] if PurePosixPath(image["file_name"]).parent.name in split_of]
    images.sort(key=lambda image: image["file_name"])
    counts = Counter(PurePosixPath(image["file_name"]).parent.name for image in images)
    if counts != Counter({clip: 100 for clip in split_of}):
        raise ValueError(f"Selección incompleta o inesperada {counts}")

    locations = defaultdict(set)
    mask_names = {}
    for image in images:
        location, clip, _ = PurePosixPath(image["file_name"]).parts
        locations[split_of[clip]].add(location)
        mask_names[clip] = f"{location}/{clip}-mask.png"
    if locations["train"] & locations["val"] or (locations["train"] | locations["val"]) & RESERVED_LOCATIONS:
        raise ValueError("Las ubicaciones de ajuste, validación y prueba deben estar separadas.")

    raw = work / "raw" if args.recipe == "v1" else CACHE / "adaptation" / "aau-rain-v1" / "raw"
    assets = sorted(mask_names.values()) + [image["file_name"] for image in images]
    with ThreadPoolExecutor(max_workers=args.download_workers) as pool:
        for index, _ in enumerate(pool.map(lambda asset: download_png(raw, asset), assets), 1):
            if index % 50 == 0 or index == len(assets):
                print(f"PNG comprobados {index}/{len(assets)}", flush=True)

    masks = {}
    for clip, name in mask_names.items():
        image_mask = read_png(raw / name, cv2.IMREAD_GRAYSCALE)
        if set(np.unique(image_mask)) != {0, 255}:
            raise ValueError(f"Máscara oficial inesperada {name}")
        masks[clip] = image_mask == 255

    ids = {}
    for category in dataset["categories"]:
        name = {"motorbike": "motorcycle"}.get(category["name"], category["name"])
        if name in CLASSES:
            ids[category["id"]] = class_names.index(name)
    by_image = defaultdict(list)
    selected_ids = {image["id"] for image in images}
    for annotation in dataset["annotations"]:
        if annotation["image_id"] in selected_ids:
            by_image[annotation["image_id"]].append(annotation)

    prepared = work / "prepared"
    lists = defaultdict(list)
    by_split = defaultdict(Counter)
    provenance = []
    generator = random.Random(args.seed)
    contrast_ids = set(generator.sample(
        [image["id"] for image in images if split_of[PurePosixPath(image["file_name"]).parent.name] == "train"],
        75,
    ))
    for image in images:
        name = image["file_name"]
        path = PurePosixPath(name)
        clip = path.parent.name
        split = split_of[clip]
        frame = read_png(raw / name, cv2.IMREAD_COLOR)
        frame, visible, boxes = mask_and_labels(frame, masks[clip], by_image[image["id"]], ids)
        label = "".join(f"{index} " + " ".join(f"{value:.9f}" for value in coords) + "\n" for index, *coords in boxes)
        variants = [("", frame, None)]
        if image["id"] in contrast_ids:
            contrast = generator.uniform(0.85, 1.15)
            brightness = generator.uniform(-8.0, 8.0)
            lookup = np.clip((np.arange(256) - 128) * contrast + 128 + brightness, 0, 255).astype(np.uint8)
            augmented = cv2.LUT(frame, lookup)
            augmented[~visible] = 114
            variants.append(("-contrast", augmented, dict(contrast=contrast, brightness=brightness)))
        by_split[split].update(class_names[index] for index, *_ in boxes)
        source_hash = sha256(raw / name)
        for suffix, pixels, augmentation in variants:
            stem = f"{clip}-{path.stem}{suffix}"
            image_path = prepared / "images" / split / f"{stem}.png"
            label_path = prepared / "labels" / split / f"{stem}.txt"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            write_png(image_path, pixels)
            label_path.write_text(label, encoding="utf-8", newline="\n")
            lists[split].append(image_path.as_posix())
            provenance.append(dict(
                source=name, source_sha256=source_hash, image_id=image["id"], split=split,
                image=image_path.relative_to(prepared).as_posix(), image_sha256=sha256(image_path),
                label=label_path.relative_to(prepared).as_posix(), label_sha256=sha256(label_path),
                boxes=len(boxes), augmentation=augmentation,
            ))
    for split, paths in lists.items():
        (prepared / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8", newline="\n")
    config = prepared / "data.yaml"
    config.write_text(json.dumps(dict(path=prepared.as_posix(), train="train.txt", val="val.txt", names=class_names),
                                 indent=2), encoding="utf-8")
    manifest = dict(
        source_doi="https://doi.org/10.34740/kaggle/dsv/105294", source_version=1, license="CC BY 4.0",
        annotations_sha256=ANNOTATIONS_SHA256, seed=args.seed, recipe=args.recipe, splits=SPLITS,
        reserved_locations=sorted(RESERVED_LOCATIONS),
        masks={name: sha256(raw / name) for name in sorted(mask_names.values())},
        source_images_by_clip=counts, prepared_images={split: len(paths) for split, paths in lists.items()},
        original_boxes_by_class=by_split, classes=class_names,
        vehicle_label_ids={name: class_names.index(name) for name in CLASSES},
        split_lists={split: sha256(prepared / f"{split}.txt") for split in lists},
        data_yaml_sha256=sha256(config), files=provenance,
    )
    (prepared / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(prepared_images=manifest["prepared_images"], boxes=by_split)), flush=True)


def train(work: Path, weights: Path, args: argparse.Namespace) -> None:
    prepared = work / "prepared"
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = prepared / "data.yaml"
    if (manifest.get("recipe", "v1") != args.recipe or manifest["seed"] != args.seed
            or sha256(config) != manifest["data_yaml_sha256"]):
        raise ValueError("Receta, semilla o archivo de configuración distintos del manifiesto.")
    for split, expected in manifest["split_lists"].items():
        if sha256(prepared / f"{split}.txt") != expected:
            raise ValueError(f"Lista alterada para {split}")
    for record in manifest["files"]:
        for kind in ("image", "label"):
            if sha256(prepared / record[kind]) != record[f"{kind}_sha256"]:
                raise ValueError(f"Archivo preparado alterado {record[kind]}")

    os.environ.setdefault("YOLO_CONFIG_DIR", str(work / "ultralytics-config"))
    os.chdir(work)
    import torch
    from ultralytics import YOLO, __version__

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible.")

    training_options = {}
    transfer_count = None
    if args.recipe == "v2":
        from ultralytics.models.yolo.detect import DetectionTrainer

        class VerifyCocoTransfer(DetectionTrainer):
            def get_model(self, cfg=None, weights=None, verbose=True):
                nonlocal transfer_count
                model = super().get_model(cfg, weights, verbose)
                if weights is None:
                    raise ValueError("No se recibió el peso COCO para la transferencia.")
                original = weights.state_dict()
                initialized = model.state_dict()
                transfer_count = sum(
                    name in initialized and torch.equal(value, initialized[name])
                    for name, value in original.items()
                )
                if transfer_count != 355 or len(original) != len(initialized) or len(initialized) != 355:
                    raise ValueError(f"Transferencia incompleta antes de entrenar ({transfer_count}/355).")
                print("Transferencia COCO verificada 355/355 antes de entrenar", flush=True)
                return model

        training_options.update(trainer=VerifyCocoTransfer, warmup_bias_lr=0.0)

    v2 = args.recipe == "v2"
    result = YOLO(str(weights)).train(
        data=str(config), device=args.device, imgsz=args.image_size or (960 if v2 else 640),
        batch=args.batch or (4 if v2 else 8),
        epochs=args.epochs, patience=5, seed=args.seed, deterministic=True, workers=0,
        freeze=10, optimizer="AdamW", lr0=0.0001 if v2 else 0.0008, lrf=0.1, warmup_epochs=1,
        hsv_h=0.005, hsv_s=0.10, hsv_v=0.18, translate=0.03, scale=0.12, fliplr=0.15,
        mosaic=0, mixup=0, close_mosaic=0, amp=args.device != "cpu",
        project=str(work / "runs"), name="yolov8s-aau-rain", save=True, plots=False,
        **training_options,
    )
    directory = Path(result.save_dir)
    checkpoint = directory / "weights" / "best.pt"
    report = dict(
        weights=str(weights), weights_sha256=sha256(weights), manifest_sha256=sha256(manifest_path),
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), ultralytics=__version__,
        recipe=args.recipe, verified_pretrained_tensors=transfer_count,
        torch=torch.__version__, val_metrics={key: float(value) for key, value in result.results_dict.items()},
    )
    (directory / "adaptation_result.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=("v1", "v2"), default="v1")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--annotations", type=Path, default=CACHE / "datasets" / "aau-rainsnow" / "aauRainSnow-rgb.json")
    parser.add_argument("--weights", type=Path, default=CACHE / "models" / "yolov8s.pt")
    parser.add_argument("--seed", type=int, default=89)
    parser.add_argument("--download-workers", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, choices=(640, 960))
    parser.add_argument("--batch", type=int)
    parser.add_argument("--epochs", type=int, choices=range(10, 21), default=15)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--train-only", action="store_true")
    args = parser.parse_args()
    work = (args.directory or CACHE / "adaptation" / f"aau-rain-{args.recipe}").resolve()
    if work.is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("Guarde imágenes, manifiestos y pesos fuera del repositorio.")
    manifest = work / "prepared" / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("recipe", "v1") != args.recipe:
        parser.error("Cada receta requiere su propia carpeta de resultados.")
    weights = args.weights.resolve()
    if not weights.is_file():
        parser.error("No se encontró el peso inicial de YOLOv8s.")
    if not args.train_only:
        prepare(work, args.annotations.resolve(), weights, args)
    if not args.prepare_only:
        train(work, weights, args)


if __name__ == "__main__":
    main()
