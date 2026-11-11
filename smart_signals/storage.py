"""Local, revisioned results. Large data stays out of the source checkout."""
from __future__ import annotations
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get('SMART_SIGNALS_HOME', Path.home() / '.cache' / 'smart-signals'))

def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')

def folders():
    for name in ('runs', 'models', 'uploads', 'profiles', 'datasets'):
        (HOME / name).mkdir(parents=True, exist_ok=True)
    return HOME

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                     prefix=path.name, suffix='.tmp', delete=False) as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = Path(f.name)
    os.replace(temporary, path)

def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default

def run_dir(run_id: str):
    if not re.fullmatch(r'[0-9a-f]{32}', run_id):
        raise ValueError('Identificador de sesión inválido')
    path = HOME / 'runs' / run_id
    if not path.is_dir():
        raise FileNotFoundError('Sesión no encontrada')
    return path

def public_source(source: str):
    """Never persist or export RTSP/HTTP userinfo or query-string secrets."""
    if '://' not in source:
        return source
    parts = urlsplit(source)
    host = parts.hostname or ''
    if ':' in host:
        host = '[' + host + ']'
    if parts.port:
        host += ':' + str(parts.port)
    return urlunsplit((parts.scheme, host, parts.path, '', ''))

def code_fingerprint():
    digest = hashlib.sha256()
    files = sorted((ROOT / 'smart_signals').glob('*.py')) + [ROOT / 'app.py', ROOT / 'requirements.txt']
    for path in files:
        if path.exists():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()

def source_kind(source: str):
    return 'live' if source.isdecimal() or source.lower().startswith(('rtsp://', 'rtsps://', 'http://', 'https://')) else 'file'

def resolve_source(source: str):
    source = source.strip()
    if not source:
        raise ValueError('Elija una cámara o un video')
    if source_kind(source) == 'live':
        return int(source) if source.isdecimal() else source
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise ValueError('No se encontró el archivo de video')
    if path.suffix.lower() not in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'):
        raise ValueError('Formato de video no admitido')
    return str(path.resolve())
