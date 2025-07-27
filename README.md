# Sistema de Conteo Vehicular — Smart Signals

> Guía paso a paso para clonar, instalar, probar y calibrar el prototipo en tu PC.

---

## 1 · Requisitos

| Herramienta | Versión mínima | Enlace |
| ----------- | -------------- | ------ |
|             |                |        |

|   |
| - |

|   |
| - |

|   |
| - |

| **Git**     | 2.40        | [https://git-scm.com](https://git-scm.com) |
| ----------- | ----------- | ------------------------------------------ |
| **Git LFS** | 3.4         | [https://git-lfs.com](https://git-lfs.com) |
| **Python**  | 3.11 / 3.12 | [https://python.org](https://python.org)   |
| **FFmpeg**  | opcional    | grabar clips                               |

> Windows 10/11 probado; en Linux los comandos son equivalentes.

---

## 2 · Clonar el repositorio

```bash
cd ~/Documents
# El repositorio ahora es **público**.
# Puedes clonar directamente (HTTPS o SSH):
#  ➜ HTTPS (sin clave SSH):
#      git clone https://github.com/bryangr1993/smart_signals.git
#  ➜ SSH (si ya tienes clave configurada):
#      git clone git@github.com:bryangr1993/smart_signals.git git@github.com:bryangr1993/smart_signals.git
cd smart_signals

# LFS (una sola vez por PC)
git lfs install
```

---

## 3 · Entorno virtual e instalación

```bash
python -m venv .venv
source .venv/Scripts/activate    # PowerShell: .\.venv\Scripts\Activate.ps1

pip install --upgrade pip wheel
pip install -r requirements.txt  # Ultralytics, OpenCV, Shapely, etc.
```

---

## 4 · Estructura de carpetas

```
smart_signals/
├─ src/              # código reutilizable
│  ├─ counter.py
│  └─ …
├─ scripts/          # utilidades one‑shot
│  └─ roi_selector.py
├─ config/           # archivos de configuración
│  ├─ roi.json       # polígono del ROI
│  └─ bytetrack.yaml # tracker personalizado (opcional)
├─ data/
│  └─ raw/           # vídeos .mp4 (LFS)
└─ output/           # CSVs de conteo
```

---

## 5 · Definir el ROI

```bash
python scripts/roi_selector.py data/raw/mi_video.mp4
# → click puntos • ENTER guarda config/roi.json
```

---

## 6 · Ejecución rápida

```bash
python main.py \
  --video  data/raw/mi_video.mp4 \
  --roi    config/roi.json \
  --tracker config/bytetrack.yaml \
  --imgsz  640 \
  --skip   2 \
  --area_min 500 \
  --min_hits 3 \
  --max_miss 12 \
  --output output/mi_video_counts.csv
```

> **Modo diagnóstico** añade `--debug --show_conf` para ver recuadros y confianza.

### Significado de colores (debug)

| Color        | Estado                            |
| ------------ | --------------------------------- |
| **Verde**    | cuenta en ROI                     |
| **Amarillo** | aún no supera `min_hits`          |
| **Rojo**     | descartado por `area_min`         |
| **Azul**     | descartado por solapamiento (IoU) |
| **Magenta**  | descartado por contención (>80 %) |

---

## 7 · Flags importantes

| Flag          | Descripción                          | Por defecto             |
| ------------- | ------------------------------------ | ----------------------- |
| `--imgsz`     | Redimensiona frame antes de YOLO     | 640                     |
| `--skip`      | Procesa 1 de N frames                | 1                       |
| `--area_min`  | Área mínima del bbox (px²)           | 400                     |
| `--min_hits`  | Frames dentro antes de sumar         | 3                       |
| `--max_miss`  | Frames fuera antes de restar         | 7                       |
| `--tracker`   | YAML de ByteTrack (`auto` = default) | `config/bytetrack.yaml` |
| `--debug`     | Activa overlay de recuadros          | off                     |
| `--show_conf` | Muestra confianza junto al ID        | off                     |

---

## 8 · Flujo de calibración en campo

1. **Graba 60 s** con cámara fija.
2. Dibuja ROI → `roi_selector.py`.
3. Ejecuta con `--debug --show_conf`.
4. Ajusta `area_min`, `min_hits`, `max_miss` hasta que el número estable coincida con la realidad.
5. Guarda parámetros en un YAML (próxima versión) y ejecuta sin debug para producción.

---

## 9 · Próximos pasos

- Fine‑tuning rápido con clips reales (mejorar motos y camiones locales).
- Empaquetar en Docker/**exe** para mini‑PC industrial.
- Exportar conteo vía MQTT o HTTP para el módulo de control semafórico.

---
