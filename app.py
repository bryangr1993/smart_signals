"""Smart Signals Studio: local operator UI and versioned sensor output."""
from __future__ import annotations

import csv
import io
import re
import tempfile
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.background import BackgroundTask

from smart_signals.evaluation import evaluate_samples
from smart_signals.runtime import RunManager, live_run_owner
from smart_signals.storage import HOME, ROOT, folders, public_source, read_json, resolve_source, run_dir, utc_now, write_json

manager = RunManager()

@asynccontextmanager
async def lifespan(app):
    yield
    manager.close()

app = FastAPI(title='Smart Signals · Presencia vehicular', version='0.2.0', lifespan=lifespan)

@app.middleware('http')
async def local_only(request: Request, call_next):
    allowed = {'127.0.0.1', 'localhost', '::1', 'testserver'}
    if request.url.hostname not in allowed:
        return JSONResponse({'detail': 'Esta aplicación solo admite acceso local'}, status_code=403)
    origin = request.headers.get('origin')
    if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin:
        parts = urlsplit(origin)
        if parts.hostname != request.url.hostname or parts.port != request.url.port:
            return JSONResponse({'detail': 'Origen no autorizado'}, status_code=403)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response

@app.exception_handler(ValueError)
async def invalid(request, exc):
    return JSONResponse({'detail': str(exc)}, status_code=400)

@app.exception_handler(FileNotFoundError)
async def missing(request, exc):
    return JSONResponse({'detail': str(exc)}, status_code=404)

class SourceRequest(BaseModel):
    source: str = Field(min_length=1, max_length=4096)

class StartRequest(SourceRequest):
    name: str = Field(default='Sesión de presencia', min_length=1, max_length=120)
    roi: list[list[float]] = Field(min_length=3, max_length=50)
    anchor: Literal['bottom_center', 'center'] = 'bottom_center'
    model: Literal['yolov8n.pt', 'yolov8s.pt'] = 'yolov8s.pt'
    device: Literal['auto', 'cpu', '0'] = 'auto'
    image_size: int = Field(default=960, ge=320, le=1280, multiple_of=32)
    confidence: float = Field(default=.25, ge=.05, le=.95)
    iou: float = Field(default=.7, ge=.1, le=.95)
    sample_fps: float = Field(default=5, ge=.2, le=30)
    realtime: bool = False
    tracking: bool = True
    crop_to_roi: bool = False
    save_annotated: bool = False
    condition: Literal['day_dry', 'night_dry', 'day_rain', 'night_rain', 'unknown'] = 'unknown'
    validation_interval_s: float = Field(default=5, ge=1, le=300)

class ProfileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    config: StartRequest

class AnnotationRequest(BaseModel):
    sample_id: str = Field(pattern=r'^\d{4}$')
    manual_count: int | None = Field(default=None, ge=0, le=1000)
    condition: str = Field(default='unknown', max_length=50)

    @field_validator('condition')
    @classmethod
    def nonempty_condition(cls, value):
        value = value.strip()
        if not value:
            raise ValueError('La condición no puede estar vacía')
        return value

@app.get('/')
def index():
    return FileResponse(ROOT / 'web' / 'index.html')

@app.get('/api/status')
def status():
    return manager.snapshot()

@app.get('/api/integration/latest')
def integration():
    state = manager.snapshot()
    observation = state.get('observation')
    usable = bool(state['state'] == 'running' and observation and observation.get('vehicles_present') is not None and observation.get('status') == 'valid')
    return dict(schema_version=1, run_id=state.get('run_id'), source_mode=(state.get('source') or {}).get('kind'),
                usable=usable, vehicles_present=observation.get('vehicles_present') if usable else None,
                observation=observation, state=state['state'], generated_at_utc=utc_now())

@app.post('/api/preview')
def preview(body: SourceRequest):
    return manager.preview(body.source)

@app.get('/api/preview.jpg')
def preview_image():
    if manager.preview_frame is None:
        raise HTTPException(404, 'Primero abra una fuente')
    return Response(manager.preview_frame, media_type='image/jpeg')

@app.get('/api/frame.jpg')
def current_image():
    with manager.lock:
        image = manager.frame
    if image is None:
        raise HTTPException(404, 'Esperando el primer fotograma')
    return Response(image, media_type='image/jpeg')

@app.post('/api/start')
def start(body: StartRequest):
    resolve_source(body.source)
    return {'run_id': manager.start(body.model_dump())}

@app.post('/api/stop')
def stop():
    return manager.stop()

@app.post('/api/confirm-view')
def confirm_view():
    if manager.snapshot()['state'] != 'awaiting_confirmation':
        raise HTTPException(409, 'No hay un encuadre pendiente de confirmación')
    manager.confirm_event.set()
    return {'confirmed': True}

@app.get('/api/sources')
def sources():
    paths = []
    for folder in (ROOT / 'data' / 'raw', HOME / 'datasets', HOME / 'uploads'):
        if folder.exists():
            paths.extend(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov', '.webm'))
    rows = [dict(id=str(i), name=p.name, path=str(p)) for i, p in enumerate(sorted(set(paths)))][:300]
    return {'sources': rows, 'default_path': str(ROOT / 'data' / 'raw' / 'traffic_lima.mp4')}

@app.post('/api/upload')
async def upload(file: UploadFile):
    folders()
    name = Path(file.filename or 'video.mp4').name
    if Path(name).suffix.lower() not in ('.mp4', '.avi', '.mkv', '.mov', '.webm', '.m4v'):
        raise HTTPException(400, 'Seleccione un archivo de video')
    safe_name = re.sub(r'[^\w. -]', '_', name)[-150:]
    path = HOME / 'uploads' / (uuid.uuid4().hex[:10] + '_' + safe_name)
    total = 0
    try:
        with path.open('wb') as target:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 2 * 1024**3:
                    raise HTTPException(413, 'Para videos de más de 2 GB utilice su ruta local')
                target.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {'path': str(path), 'name': name}

@app.get('/api/profiles')
def profiles():
    return {'profiles': [read_json(p) for p in sorted((HOME / 'profiles').glob('*.json'))]}

@app.post('/api/profiles')
def save_profile(body: ProfileRequest):
    from smart_signals.geometry import validate_roi
    config = body.config.model_dump()
    validate_roi(config['roi'])
    config['source'] = public_source(config['source'])
    key = uuid.uuid4().hex
    profile = dict(id=key, name=body.name, config=config, saved_at=utc_now())
    write_json(HOME / 'profiles' / (key + '.json'), profile)
    return profile

@app.get('/api/runs')
def runs():
    rows = [read_json(p) for p in (HOME / 'runs').glob('*/manifest.json')]
    return {'runs': sorted(rows, key=lambda x: x.get('created_at', ''), reverse=True)}

@app.get('/api/runs/{run_id}/samples')
def samples(run_id: str):
    directory = run_dir(run_id)
    doc = read_json(directory / 'manifest.json')
    rows = []
    for sample in read_json(directory / 'samples.json', []):
        sample.pop('predicted_count', None)  # Blind annotation.
        sample['image_url'] = f'/api/runs/{run_id}/samples/{sample["id"]}.jpg'
        rows.append(sample)
    return dict(samples=rows, roi=doc['config']['roi'], anchor=doc['config']['anchor'])

@app.get('/api/runs/{run_id}/samples/{sample_id}.jpg')
def sample_image(run_id: str, sample_id: str):
    if not re.fullmatch(r'\d{4}', sample_id):
        raise HTTPException(400, 'Muestra inválida')
    path = run_dir(run_id) / 'samples' / (sample_id + '.jpg')
    if not path.exists():
        raise HTTPException(404, 'Muestra no encontrada')
    return FileResponse(path)


def run_is_active(run_id: str, directory: Path) -> bool:
    with manager.lock:
        if manager.state['run_id'] == run_id and manager.worker and manager.worker.is_alive():
            return True
    manifest = read_json(directory / 'manifest.json') or {}
    return (
        manifest.get('state') in ('loading', 'running', 'reconnecting', 'awaiting_confirmation')
        and live_run_owner(manifest)
    )


@app.post('/api/runs/{run_id}/annotations')
def annotate(run_id: str, body: AnnotationRequest):
    directory = run_dir(run_id)
    if run_is_active(run_id, directory):
        raise HTTPException(409, 'Termine o detenga el análisis antes de validar las muestras')
    with manager.lock:
        records = read_json(directory / 'samples.json', [])
        found = False
        for sample in records:
            if sample['id'] == body.sample_id:
                sample.update(manual_count=body.manual_count, condition=body.condition, annotated_at=utc_now())
                found = True
        if not found:
            raise HTTPException(404, 'Muestra no encontrada')
        metrics = evaluate_samples(records)
        write_json(directory / 'samples.json', records)
        write_json(directory / 'validation.json', metrics)
    return {'metrics': metrics}

@app.get('/api/runs/{run_id}/metrics')
def metrics(run_id: str):
    return evaluate_samples(read_json(run_dir(run_id) / 'samples.json', []))

@app.get('/api/runs/{run_id}/download')
def download(run_id: str):
    directory = run_dir(run_id)
    if run_is_active(run_id, directory):
        raise HTTPException(409, 'Termine o detenga la sesión para exportar un resultado consistente')
    with manager.lock:
        records = read_json(directory / 'samples.json', [])
        write_json(directory / 'validation.json', evaluate_samples(records))
        with (directory / 'validation.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            fields = ['id', 'time_s', 'manual_count', 'predicted_count', 'condition']
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
            writer.writeheader(); writer.writerows(records)
        with tempfile.NamedTemporaryFile(prefix='smart-signals-', suffix='.zip', delete=False, dir=HOME) as handle:
            target = Path(handle.name)
        try:
            with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(directory.rglob('*')):
                    if path.is_file() and path.suffix not in ('.zip', '.tmp'):
                        archive.write(path, path.relative_to(directory))
        except Exception:
            target.unlink(missing_ok=True)
            raise
    return FileResponse(target, filename=f'smart-signals-{run_id[:8]}.zip', media_type='application/zip',
                        background=BackgroundTask(target.unlink, missing_ok=True))

app.mount('/static', StaticFiles(directory=ROOT / 'web'), name='static')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8765, access_log=False)
