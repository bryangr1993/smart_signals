"""Fetch a small, checksummed subset of public fixed-camera traffic data."""

import argparse
import hashlib
import shutil
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import NamedTuple
from urllib.parse import quote
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile


DEFAULT_DIRECTORY = Path.home() / ".cache" / "smart-signals" / "datasets"
_KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow/"
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024


class Asset(NamedTuple):
    filename: str
    url: str
    size: int
    sha256: str
    member: str | None = None


def _aau(filename: str, source: str, size: int, sha256: str) -> Asset:
    return Asset(
        f"aau-rainsnow/{filename}",
        f"{_KAGGLE}{quote(source, safe='')}?datasetVersionNumber=1",
        size,
        sha256,
        PurePosixPath(source).name,
    )


ASSETS = (
    _aau("Hadsundvej-2.mkv", "Hadsundvej/Hadsundvej-2/cam1.mkv", 22_312_639,
         "d3d66096369ba7fc8cc834e80c715cb4b27ba5dd93ec2922a68a09483b45dec1"),
    _aau("Egensevej-5.mkv", "Egensevej/Egensevej-5/cam1.mkv", 39_445_033,
         "cbe3b3c9bae697b6c0e4381fdc53f24a5f68629f8332ad9e68cbd0b26840e8dc"),
    _aau("aauRainSnow-rgb.json", "aauRainSnow-rgb.json", 5_997_145,
         "a3f90c9b7329e9bfc6ecac59d5f71e7583ea961ecaa19ecc7d7ffdbf91a8284a"),
    _aau("Hadsundvej-2-mask.png", "Hadsundvej/Hadsundvej-2-mask.png", 2_824,
         "3854473ab931148115dc11fcced8157222db926082552cb6f6c65a9ca0cd7ac3"),
    _aau("Egensevej-5-mask.png", "Egensevej/Egensevej-5.png", 2_751,
         "3b310e8ad13643794ebfbe4f4c4bd33bafdd5a3edc2f6fb5fa0e5cd5b3534865"),
    _aau("Ringvej-1.mkv", "Ringvej/Ringvej-1/cam1.mkv", 25_529_405,
         "609142029482ff6151d4c473a36ebc54f9da7aded2ed1d5b787dea9415e66cc9"),
    _aau("Ringvej-1-mask.png", "Ringvej/Ringvej-1-mask.png", 2_734,
         "e1645ef0c2c250a0fb6a78edff44a334a05b1d7a8d0335eb72beacb3383072cc"),
    Asset("wikimedia/Street_traffic.webm",
          "https://upload.wikimedia.org/wikipedia/commons/7/70/Street_traffic.webm",
          26_490_572, "e5facb24baf755f0c1193999d823e99f6032661ad2bdef5cc28cc1cbbfde03e2"),
)


def _verified(path: Path, asset: Asset) -> bool:
    if not path.is_file() or path.stat().st_size != asset.size:
        return False
    with path.open("rb") as reader:
        return hashlib.file_digest(reader, "sha256").hexdigest() == asset.sha256


def _download(asset: Asset, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".download-", dir=target.parent) as temporary:
        response_path = Path(temporary) / "response"
        request = Request(asset.url, headers={"User-Agent": "SmartSignals/1.0 research dataset"})
        received = 0
        with urlopen(request, timeout=60) as response, response_path.open("wb") as writer:
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > _MAX_RESPONSE_BYTES:
                    raise ValueError(f"Respuesta demasiado grande para {asset.filename}")
                writer.write(chunk)

        candidate = response_path
        if asset.member and is_zipfile(response_path):
            candidate = Path(temporary) / "asset"
            with ZipFile(response_path) as archive:
                members = [item for item in archive.infolist() if not item.is_dir()]
                if len(members) != 1 or PurePosixPath(members[0].filename).name != asset.member:
                    raise ValueError(f"Archivo ZIP inesperado para {asset.filename}")
                if members[0].file_size != asset.size:
                    raise ValueError(f"Tamaño inesperado para {asset.filename}")
                with archive.open(members[0]) as reader, candidate.open("wb") as writer:
                    shutil.copyfileobj(reader, writer)

        if not _verified(candidate, asset):
            raise ValueError(f"Tamaño o SHA-256 inesperados para {asset.filename}")
        candidate.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--check", action="store_true", help="Verificar sin descargar")
    args = parser.parse_args()
    directory = args.directory.resolve()
    repo = Path(__file__).resolve().parents[1]
    if directory.is_relative_to(repo):
        parser.error("La carpeta de datos debe estar fuera del repositorio.")

    for asset in ASSETS:
        target = directory / asset.filename
        if not _verified(target, asset):
            if args.check:
                parser.exit(1, f"Archivo ausente o inválido: {target}\n")
            print(f"Descargando {asset.filename}", flush=True)
            _download(asset, target)
        print(f"Verificado: {target} ({asset.size:,} bytes)")


if __name__ == "__main__":
    main()
