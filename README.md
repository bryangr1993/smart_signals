# Smart Signals · Presencia vehicular

Aplicación local del proyecto DIPI-089 para contar **vehículos presentes dentro de un polígono**, tanto detenidos como en movimiento. Incluye cámara/video, editor de zona, seguimiento opcional, historial, muestras para revisión manual y exportación reproducible.

El número presente no es aforo acumulado, vehículos únicos ni longitud de cola. Una imagen no evaluable produce **sin dato**, distinto de cero. El sistema es una herramienta de investigación; no acciona semáforos.

## Abrir en este computador

Ejecutar **Abrir Smart Signals.cmd**, o abrir [la interfaz local](http://127.0.0.1:8765) si el servidor ya está iniciado. La aplicación escucha solo en este equipo. Cerrar la pestaña mantiene el análisis; use **Detener** para finalizar la sesión.

En Windows, los modelos, videos descargados, resultados y el entorno Python están por defecto en `%USERPROFILE%/.cache/smart-signals`, fuera de OneDrive. Se puede cambiar con `SMART_SIGNALS_HOME` antes de instalar/iniciar.

## Instalación reproducible

Se requiere Python **3.12**. En PowerShell, desde el repositorio:

```powershell
# CPU, para equipos sin NVIDIA compatible
powershell -ExecutionPolicy Bypass -File scripts/install.ps1 -Device cpu

# Alternativa: NVIDIA CUDA. Descarga varios GB de PyTorch.
powershell -ExecutionPolicy Bypass -File scripts/install.ps1 -Device cuda

.\'Abrir Smart Signals.cmd'
```

El instalador crea un entorno aislado, instala versiones fijadas y descarga los pesos oficiales YOLOv8n y YOLOv8s. Instalar y descargar requieren internet, analizar material local ya preparado no. Las sesiones nuevas usan YOLOv8s a 960 y análisis del cuadro completo; el recorte contextual es opcional y YOLOv8n a 640 queda como alternativa de menor consumo. Los perfiles conservan sus ajustes propios.

El código original está en el historial de Git (`af8e1c9`). El desarrollo actual está en `feature/presence-studio`. Para clonar con el ejemplo original:

```powershell
git clone https://github.com/bryangr1993/smart_signals.git
cd smart_signals
git lfs pull
```

Esta rama local aún debe publicarse para que un clon remoto incluya la aplicación nueva.

## Estado comprobado al 11 de noviembre de 2026

Se comprobó el flujo completo de interfaz, selección del polígono, procesamiento con GPU y CPU, detención, perfiles, anotación manual y exportación ZIP. Pasaron 31 pruebas automatizadas y una prueba real de decodificación/reconexión contra una transmisión HTTP/MJPEG sintética local. La ejecución final por terminal procesó el video completo de Lima a una muestra por segundo y produjo 103 observaciones y un video anotado de 103 cuadros, verificado mediante decodificación. La RTX 4060 Laptop GPU de este computador funciona con PyTorch 2.7.1+cu128. La instantánea de dependencias está en [environment-cuda.lock.txt](docs/environment-cuda.lock.txt).

La exactitud todavía necesita mejora y validación con las cámaras del proyecto. El perfil base obtuvo 102/299 conteos exactos en tres escenas adversas de AAU. En el acceso diurno parcial de MTID acertó 13/22 instantes ocupados; el 95 % global incluía 278 instantes vacíos de 300. Estos resultados **no acreditan el objetivo del 90 % en el piloto**. Dos ajustes experimentales se conservaron con sus resultados; ninguno sustituye al modelo operativo. Consulte [evaluación detallada y fuentes](docs/DATASETS.md), [adaptación experimental](docs/ADAPTACION.md) y las [evidencias conservadas](docs/evidence/README.md).

## Uso

1. Seleccione un video de la biblioteca, importe un archivo o escriba la ruta/URL de una cámara. `0` representa la primera webcam. Pulse abrir/probar fuente.
2. Dibuje al menos tres vértices sobre la imagen y ajuste el polígono. Mantenga el encuadre fijo. El centro inferior de la caja es el punto de referencia predeterminado; la alternativa central se configura en ajustes.
3. Dé nombre a la sesión, indique la condición observada y comience. El conteo, las clases y la curva describen presencia, sin acumular cada fotograma como si fueran vehículos distintos.
4. Finalice la sesión para revisar muestras y exportar. La revisión manual muestra el cuadro sin las detecciones automáticas para reducir sesgo.

La vista de cámara debe mantener **posición, zoom y recorte**. La aplicación pide confirmar el encuadre tras recuperar una conexión; no controla automáticamente el preset PTZ ni puede garantizar que detecte cualquier giro durante una conexión continua. Si mueve la cámara, detenga, revise la imagen y vuelva a definir la zona.

Los estados `degraded` son advertencias de captura/seguimiento. No representan porcentajes de precisión. La alerta por áreas saturadas es una heurística, no un diagnóstico infalible de visibilidad: puede advertir en escenas claras y pasar por alto oclusiones. Una imagen visualmente correcta aún puede contener detecciones erróneas; comprobarlo requiere etiquetas independientes.

## Archivos por sesión

El ZIP incluye:

- `counts.csv`: tiempo de fuente, conteo, clases, estado, tiempos de procesamiento y advertencias.
- `observations.jsonl`: observaciones detalladas y cajas por fotograma analizado. Durante cortes registra ausencia de señal, sin inventar ceros.
- `manifest.json`: configuración sin credenciales, huellas del código/modelo y del video local, versiones, fuente y reloj utilizado.
- `summary.json`: número de observaciones, media/máximo de presencia y rendimiento; no son aforos. La latencia p95 desde recepción usa una ventana de hasta los últimos 1.024 fotogramas procesados, indicada en el archivo.
- `samples/` y `samples.json`: imágenes preseleccionadas para anotación y su correspondencia temporal.
- `validation.csv` y `validation.json`: etiquetas manuales y métricas. También se incluyen sin etiquetas, con métricas no calculables en `null`.
- `annotated.mp4`, si se activó la grabación de diagnóstico. Su velocidad corresponde a la frecuencia de muestreo; para el tiempo exacto use los timestamps del CSV.

Hay un máximo de 240 muestras de revisión por sesión; el conteo continúa registrándose después. La exportación se habilita al terminar o detener para producir archivos consistentes. El registro de larga duración consume disco: gestione la conservación de sesiones según el estudio.

## Motor y eficiencia

La base es YOLOv8 con ByteTrack opcional. La pertenencia se decide sobre las cajas de detección actuales, no sobre predicciones antiguas. Los nuevos vehículos pueden contar aunque aún no tengan ID. No se descartan automáticamente dos vehículos porque sus cajas se superpongan. Las clases se seleccionan por nombre, no por IDs COCO codificados a mano. El recorte contextual opcional concentra la resolución de inferencia en el rectángulo de la zona más un margen; conserva coordenadas del cuadro completo y no recorta exactamente por el borde del polígono. `crop_box` y `effective_device` quedan en las observaciones.

En directo la captura conserva una sola imagen pendiente y limita la antigüedad. En archivos el muestreo se basa en tiempo de video. Si el contenedor no permite conocerlo con fiabilidad, se indica la aproximación usada. Un equipo lento no debe crear una cola ilimitada de imágenes viejas.

No hay reentrenamiento automático, lectura de matrículas ni reconocimiento de personas. Luz, lluvia, reflejos, oclusiones y perspectiva necesitan evaluación por sitio. Una cámara no puede reconstruir por sí sola vehículos nunca visibles.

## Terminal y otras etapas

```powershell
$signalsPython = Join-Path $env:USERPROFILE '.cache/smart-signals/venv/Scripts/python.exe'
& $signalsPython main.py --video data/raw/traffic_lima.mp4 --roi config/roi.json --sample-fps 5 --device auto
& $signalsPython main.py --serve
```

El CLI admite el `roi.json` original en píxeles, convertido según la resolución del video. Las sesiones nuevas de la interfaz usan coordenadas normalizadas con configuración propia.

`GET /api/integration/latest` expone el contrato `schema_version=1`: origen, sesión, observación y `usable`. Una sesión terminada, detenida, con lectura vencida o en estado degradado nunca se anuncia como utilizable. `usable` solo expresa estas condiciones operativas; no certifica exactitud del conteo. La fuente indica `file` o `live`; el receptor debe respetarlo. No incluye órdenes para luces.

El futuro adaptador SUMO debe generar presencia bajo una definición comparable. Los aforos para construir la demanda siguen siendo otra medición. Un CSV reproducido permite probar integración, no demostrar mejoras contrafactuales de tiempos semafóricos.

## Validación y pruebas

```powershell
$signalsPython = Join-Path $env:USERPROFILE '.cache/smart-signals/venv/Scripts/python.exe'
& $signalsPython -m pytest -q
& $signalsPython -m scripts.check_stream
& $signalsPython -m scripts.fetch_datasets --help
& $signalsPython -m scripts.benchmark_presence --help
```

Consulte [fuentes y licencias de datos](docs/DATASETS.md) y [protocolo de validación](docs/VALIDACION.md). Las pruebas sintéticas comprueban reglas y estados; `check_stream.py` comprueba decodificación y reconexión HTTP/MJPEG real contra una fuente sintética local. Las cámaras RTSP/NVR físicas todavía necesitan comprobación en sitio. El benchmark externo comprueba determinados videos. Ninguno acredita automáticamente el indicador institucional del piloto de Riobamba.

El modelo y sus dependencias mantienen sus licencias originales. [Ultralytics](https://github.com/ultralytics/ultralytics) y [ByteTrack](https://github.com/ifzhang/ByteTrack) son componentes externos; conservar sus avisos en las distribuciones. Los datasets no se incluyen en Git: el descargador conserva fuentes y hashes.
