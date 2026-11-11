# Ajuste experimental de YOLOv8s con AAU RainSnow

El candidato se entrena para reconocer vehículos en cámaras fijas con lluvia, reflejos y poca luz. La comparación para decidir si mejora el sensor de presencia se realiza en secuencias de prueba independientes, con el fotograma completo y la misma regla de pertenencia que el modelo de referencia. El peso se guarda como candidato experimental.

## Datos y separación por ubicación

Se usan 400 PNG RGB anotados y cuatro máscaras oficiales de AAU RainSnow Traffic Surveillance Dataset, versión 1. Autores Chris H. Bahnsen y Thomas B. Moeslund, Aalborg University. Las imágenes preparadas modifican los originales con las máscaras y, en un subconjunto, brillo y contraste. [Datos de los autores](https://doi.org/10.34740/kaggle/dsv/105294) · [Artículo](https://doi.org/10.1109/TITS.2018.2872502) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

| Uso | Secuencias | Imágenes originales |
| --- | --- | ---: |
| Entrenamiento | Hjorringvej-3, Hasserisvej-2 y Ostre-3 | 300 |
| Validación para seleccionar el peso | Hobrovej-1 | 100 |
| Prueba independiente | Todas las secuencias de Egensevej, Hadsundvej y Ringvej se reservan por ubicación | 0 en el ajuste |

El proceso convierte las cajas COCO a coordenadas normalizadas de YOLO y mapea `motorbike` a `motorcycle`. Conserva `car`, `bus`, `truck` y `motorcycle`. Aplica el blanco de la máscara oficial al centro inferior de cada caja. Reemplaza con gris neutro el exterior del área anotada y los fragmentos de cajas excluidas que aún caigan dentro de ella. De esta forma, las zonas no anotadas, incluidos los estacionamientos fuera de máscara, no se presentan como negativos. Se revisó una imagen con etiquetas superpuestas por secuencia.

El conjunto original preparado tiene 1.397 cajas de automóvil, 75 de camión, 36 de autobús y 6 de motocicleta en entrenamiento. La validación tiene 80 cajas de automóvil y 18 de camión. Carece de ejemplos válidos de autobuses y motocicletas, por lo que no mide el desempeño de esas clases. La vista de validación presenta reflejos fuertes de faros. La mediana de las cajas es 18 × 16 píxeles y 47 de los 100 instantes están vacíos dentro de la máscara.

## Reproducir

Usar el entorno de Studio con PyTorch CUDA y el peso inicial oficial `yolov8s.pt`. Desde la raíz del repositorio:

```powershell
$signalsRoot = Join-Path $env:USERPROFILE '.cache/smart-signals'
$signalsPython = Join-Path $signalsRoot 'venv/Scripts/python.exe'
& $signalsPython -m scripts.fetch_datasets
& $signalsPython -m scripts.train_adaptation --recipe v1 --prepare-only
& $signalsPython -m scripts.train_adaptation --recipe v1 --train-only --device 0
& $signalsPython -m scripts.train_adaptation --recipe v2 --prepare-only
& $signalsPython -m scripts.train_adaptation --recipe v2 --train-only --device 0
```

El descargador recupera el JSON de anotaciones y prepara los archivos del benchmark público. `--prepare-only` comprueba el SHA-256 del JSON, descarga o comprueba los 404 PNG y escribe `prepared/manifest.json` con semilla 89, ubicación y partición de cada imagen y hashes de originales, máscaras, archivos derivados y etiquetas. `--train-only` verifica los hashes preparados y guarda las opciones efectivas en `args.yaml`. Las recetas guardan sus resultados en `adaptation/aau-rain-v1` y `adaptation/aau-rain-v2` dentro de la caché. Ambas usan los mismos PNG originales.

Ambas recetas usan precisión mixta en CUDA, hasta 15 épocas y parada tras cinco épocas sin mejora en validación. Congelan los primeros diez módulos de YOLOv8s. Varían moderadamente tono, saturación, brillo, escala y traslación. Añaden 75 copias de entrenamiento con contraste entre 0,85 y 1,15 y desplazamiento de brillo entre −8 y +8. Las copias se mantienen en entrenamiento y la validación conserva 100 imágenes. Mosaicos y mezcla de escenas están desactivados. [Parámetros de Ultralytics](https://docs.ultralytics.com/usage/cfg/)

| Receta | Salidas e inicialización | Resolución, lote y optimizador |
| --- | --- | --- |
| v1 | Cuatro clases. Al pasar de 80 a cuatro salidas se transfirieron 349 de 355 tensores. Los seis restantes correspondían a los pesos y sesgos finales de clasificación. | 640, lote 8, AdamW con tasa inicial 0,0008. |
| v2 | 80 salidas COCO, con etiquetas vehiculares `2 car`, `3 motorcycle`, `5 bus` y `7 truck`. Antes de entrenar, comprueba que los 355 tensores coincidan con los del peso oficial. | 960, lote 4, AdamW con tasa inicial 0,0001 y tasa inicial de sesgos durante el calentamiento en cero. |

El mejor peso se guarda en `runs/yolov8s-aau-rain/weights/best.pt`. Cada repetición crea otra carpeta. `adaptation_result.json` registra el peso inicial y el candidato con sus hashes, la versión de PyTorch y Ultralytics, el hash del manifiesto y las métricas internas de cajas.

## Validación para seleccionar los pesos

Las corridas usaron PyTorch 2.7.1+cu128 y Ultralytics 8.3.169. Se comprobó igualdad de hashes en las 475 imágenes preparadas, de las particiones y de las coordenadas de las 1.960 cajas derivadas. En v2 se verificó la transferencia de los 355 tensores COCO antes de entrenar. La resolución de validación interna sigue la receta de cada corrida.

| Receta | Mejor época | Parada temprana | Precisión de cajas | Exhaustividad | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 2 | 7 | 44,09 % | 6,18 % | 7,68 % | 1,14 % |
| v2 | 4 | 9 | 23,21 % | 6,08 % | 4,80 % | 1,58 % |

| Receta | SHA-256 de `best.pt` | SHA-256 del manifiesto preparado |
| --- | --- | --- |
| v1 | `617a41273e01e25d5e1f9cf84e74e0ad5fa495dc8f78fedeb710ab6d0f1c595d` | `f559b41a3bb66ac654bb13d4510b42585a259c10a4d91ce2cbdcdd12b59184ba` |
| v2 | `8b3d3329dfcb43efaac4ca64b1cf2ff9f0b63f276c70509b047c588fb2bf5e3b` | `cf3954469dde624f62c52518f646d5e88bd82010c720af0f8086429a31695d1b` |

La validación interna mide cajas en una sola ubicación con imágenes enmascaradas. La comparación de presencia debe usar secuencias completas reservadas, aplicar la máscara al resultado después de detectar y conservar frecuencia, regla de pertenencia y protocolo entre candidatos. Comparar concordancia exacta, error absoluto medio, sesgo, cobertura y tasas de falso vacío y falsa presencia por condición. La evidencia de AAU permite comparar modelos sobre esas escenas. Para medir el indicador del DIPI-089 hacen falta referencias propias en la cámara piloto.

## Prueba de presencia en ubicaciones reservadas

La comparación de YOLOv8s oficial y v2 usa los mismos 299 fotogramas anotados de tres ubicaciones reservadas, el mismo código y los mismos hashes de videos, anotaciones y máscaras. Ambas corridas usan cuadro completo, 960, confianza 0,25, ByteTrack y supresión entre clases con IoU 0,70. Procesan todos los cuadros anotados y uno de cada cuatro entre esos instantes. La máscara oficial se aplica a las detecciones después de inferir. Las dos corridas produjeron un conteo en todos los fotogramas de referencia.

| Métrica | YOLOv8s oficial | v2 |
| --- | ---: | ---: |
| Coincidencias exactas | 102/299 (34,11 %) | 110/299 (36,79 %) |
| Error absoluto medio, vehículos | 1,562 | 1,318 |
| Sesgo, vehículos | −1,181 | +0,154 |
| Falsos vacíos en cuadros con vehículos | 80/220 | 31/220 |
| Falsas presencias en cuadros vacíos | 1/79 | 11/79 |

| Ubicación y condición | Coincidencias oficial → v2 | Error absoluto medio oficial → v2 |
| --- | ---: | ---: |
| Hadsundvej-2, lluvia diurna | 17/100 → 10/100 | 2,090 → 2,270 |
| Egensevej-5, nieve nocturna | 74/99 → 74/99 | 0,283 → 0,303 |
| Ringvej-1, noche húmeda | 11/100 → 26/100 | 2,300 → 1,370 |

La ganancia se concentró en Ringvej-1. En Egensevej-5, v2 redujo los falsos vacíos de 25 a 5 y pasó de 0 a 11 falsas presencias. Los resultados mixtos respaldan conservar v2 como candidato experimental. Cada condición proviene de una sola ubicación y de fotogramas de una misma secuencia.

La prueba histórica de v1 sobre las mismas referencias obtuvo 36/299 coincidencias y un error absoluto medio de 4,803 vehículos. Esa corrida usó la versión previa del motor con supresión por clase. Los informes completos están en la carpeta de caché `datasets/reports`, con los nombres `aau-final-s960-agnostic.json`, `aau-adapted-v2-s960-agnostic.json` y `aau-adapted-s960.json`.
