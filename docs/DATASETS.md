# Datos de prueba y evaluación de presencia

Los informes JSON de las comparaciones documentadas y dos vistas de apoyo también se conservan en [evidence/](evidence/README.md), dentro del repositorio. Los videos, las imágenes de entrenamiento y los pesos permanecen en la caché local.

Desde la raíz del repositorio, `python -m scripts.fetch_datasets` descarga los archivos en `~/.cache/smart-signals/datasets`. `python -m scripts.fetch_datasets --check` comprueba la copia local. La descarga verifica el tamaño y SHA-256 antes de reemplazar cada archivo. Los archivos ocupan unos 120 MB en total.

## AAU RainSnow RGB

Chris H. Bahnsen y Thomas B. Moeslund, Universidad de Aalborg, publicaron videos de intersecciones de Dinamarca y anotaciones de instancias. La versión 1 del conjunto se distribuye bajo CC BY 4.0. Hadsundvej-2 muestra lluvia a las 15:23 y Egensevej-5 muestra nieve a las 03:00, según la tabla X del artículo. Ambos videos provienen de cámaras fijas y tienen 640 × 480 píxeles a 20 fps. [Datos de los autores](https://doi.org/10.34740/kaggle/dsv/105294), [artículo](https://doi.org/10.1109/TITS.2018.2872502), [texto de la licencia](https://creativecommons.org/licenses/by/4.0/).

| Archivo en `aau-rainsnow/` | Bytes | SHA-256 |
| --- | ---: | --- |
| [Hadsundvej-2.mkv](https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow/Hadsundvej%2FHadsundvej-2%2Fcam1.mkv?datasetVersionNumber=1) | 22 312 639 | `d3d66096369ba7fc8cc834e80c715cb4b27ba5dd93ec2922a68a09483b45dec1` |
| [Egensevej-5.mkv](https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow/Egensevej%2FEgensevej-5%2Fcam1.mkv?datasetVersionNumber=1) | 39 445 033 | `cbe3b3c9bae697b6c0e4381fdc53f24a5f68629f8332ad9e68cbd0b26840e8dc` |
| [aauRainSnow-rgb.json](https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow/aauRainSnow-rgb.json?datasetVersionNumber=1) | 5 997 145 | `a3f90c9b7329e9bfc6ecac59d5f71e7583ea961ecaa19ecc7d7ffdbf91a8284a` |
| [Hadsundvej-2-mask.png](https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow/Hadsundvej%2FHadsundvej-2-mask.png?datasetVersionNumber=1) | 2 824 | `3854473ab931148115dc11fcced8157222db926082552cb6f6c65a9ca0cd7ac3` |
| [Egensevej-5-mask.png](https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow/Egensevej%2FEgensevej-5.png?datasetVersionNumber=1) | 2 751 | `3b310e8ad13643794ebfbe4f4c4bd33bafdd5a3edc2f6fb5fa0e5cd5b3534865` |
| [Ringvej-1.mkv](https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow/Ringvej%2FRingvej-1%2Fcam1.mkv?datasetVersionNumber=1) | 25 529 405 | `609142029482ff6151d4c473a36ebc54f9da7aded2ed1d5b787dea9415e66cc9` |
| [Ringvej-1-mask.png](https://www.kaggle.com/api/v1/datasets/download/aalborguniversity/aau-rainsnow/Ringvej%2FRingvej-1-mask.png?datasetVersionNumber=1) | 2 734 | `e1645ef0c2c250a0fb6a78edff44a334a05b1d7a8d0335eb72beacb3383072cc` |

Los autores numeran los PNG desde 1. La decodificación con OpenCV dio igualdad de píxeles entre `cam1-00247.png` y el índice 246 de Hadsundvej-2, entre `cam1-00014.png` y el índice 13 de Egensevej-5 y entre `cam1-02655.png` y el índice 2654 de Ringvej-1. El tiempo de referencia es el índice dividido para 20. Las máscaras RGB son binarias de 640 × 480. El blanco señala el área evaluable. La máscara de Egensevej-5 se publica con el nombre `Egensevej-5.png` y se guarda localmente con el mismo sufijo de las demás máscaras.

`benchmark_presence` evalúa los instantes etiquetados por los autores. El motor procesa el cuadro completo y se aplica la máscara oficial al centro inferior de cada caja, tanto de referencia como de detección. Se incluyen automóviles, autobuses, camiones y motos según el nombre de la clase. Se omiten cajas de grupos (`iscrowd`). La selección predeterminada evalúa 199 instantes en Hadsundvej-2 y Egensevej-5. Tras la máscara, Hadsundvej-2 aporta 100 muestras con tres o más vehículos. Egensevej-5 aporta 74 muestras vacías y 25 con uno o dos. Por ello, la iluminación y la densidad varían juntas entre ambas escenas. El indicador del piloto requiere conteo de referencia en las condiciones del sitio.

Ringvej-1 añade una vista nocturna con pavimento mojado y vehículos mejor visibles que en Egensevej-5. Cuenta con 100 instantes anotados en un video de 640 × 480 a 20 fps. Dentro de su máscara hay 5 instantes vacíos, 41 con uno o dos vehículos y 54 con tres o más. Se ejecuta con `--scene Ringvej-1` y queda fuera de la comparación original de 199 instantes. Las ubicaciones completas Egensevej, Hadsundvej y Ringvej se reservaron para pruebas externas.

```powershell
$cache = Join-Path $env:USERPROFILE '.cache\smart-signals'
python -m scripts.benchmark_presence --model "$cache\models\yolov8n.pt" --device 0 --confidence 0.25 --image-size 640 --box-metrics --output "$cache\datasets\reports\aau-n640.json"
```

Para una comprobación breve se puede añadir `--max-labels 2`. El valor predeterminado `--stride 4` procesa uno de cada cuatro cuadros entre anotaciones y procesa cada cuadro etiquetado. Para medir otra configuración, usar otro archivo de salida y conservar el mismo conjunto de instantes y la cadencia. `--without-tracking` permite medir solo la detección.

## Evaluación de YOLOv8s con la supresión de duplicados entre clases

Se evaluó YOLOv8s preentrenado a 960 píxeles, confianza 0,25, cuadro completo, ByteTrack y `--stride 4` en una RTX 4060 Laptop GPU. El motor actual suprime cajas superpuestas aunque tengan clases distintas. Se conservaron los mismos 299 índices de video, las máscaras y los conteos de referencia que en las corridas anteriores del mismo perfil.

| Escena y condición | Conteos exactos | MAE (vehículos) | Falsos vacíos entre ocupados | Falsas ocupaciones entre vacíos |
| --- | ---: | ---: | ---: | ---: |
| Hadsundvej-2, lluvia diurna | 17/100 | 2,09 | 0/100 | Sin muestras vacías |
| Egensevej-5, nieve nocturna y deslumbramiento severo | 74/99 | 0,28 | 25/25 | 0/74 |
| Ringvej-1, noche y pavimento mojado | 11/100 | 2,30 | 55/95 | 1/5 |
| Total | 102/299 (34,1 %) | 1,56 | 80/220 | 1/79 |

La concordancia fue 78/79 para instantes vacíos, 4/66 con uno o dos vehículos y 20/154 con tres o más. El sesgo global fue de −1,18 vehículos. La mediana de inferencia fue 16,26 ms y el percentil 95 fue 18,71 ms en los cuadros etiquetados de esta corrida. En Egensevej-5 los 74 aciertos correspondieron a cuadros vacíos. El aviso de saturación marcó 18 de 99 observaciones de esa escena como degradadas. Al exigir estado válido para integración, quedan 281/299 observaciones disponibles y 100/299 concordancias exactas sobre todas las referencias.

Antes de suprimir duplicados entre clases, el mismo perfil tenía 100/299 aciertos, MAE de 1,55 y las mismas cifras globales de falso vacío y falsa ocupación. Los resultados provienen de tres cámaras externas con condiciones adversas, cada una en un solo video. La evaluación de C2 requiere la referencia manual del sitio y del acceso elegidos.

El informe `aau-final-s960-agnostic.json` registra los cuadros, los SHA-256 de pesos, código y fuentes y las versiones de OpenCV, PyTorch y Ultralytics. Para reproducir la corrida desde el repositorio:

```powershell
$cache = Join-Path $env:USERPROFILE '.cache\smart-signals'
python -m scripts.benchmark_presence --model "$cache\models\yolov8s.pt" --device 0 --scene Hadsundvej-2 --scene Egensevej-5 --scene Ringvej-1 --image-size 960 --confidence 0.25 --stride 4 --box-metrics --output "$cache\datasets\reports\aau-final-s960-agnostic.json"
```

## Comparación inicial con los pesos preentrenados

Las corridas de esta sección preceden a la supresión de cajas duplicadas entre clases. Las tres configuraciones de la tabla usan los mismos 199 instantes de Hadsundvej-2 y Egensevej-5, ByteTrack, `--stride 4` y una RTX 4060 Laptop GPU. Las cifras de tiempo son medianas de inferencia en los instantes etiquetados. Los informes JSON están en `~/.cache/smart-signals/datasets/reports` y contienen el SHA-256 de los pesos. Las corridas nuevas registran también SHA-256 del código y las fuentes, y versiones de las dependencias de inferencia.

| Modelo y configuración | Coincidencias globales | MAE global (vehículos) | Lluvia diurna, coincidencias | Nieve nocturna, falsos vacíos si había vehículos | Inferencia mediana |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n, 640, confianza 0,25 | 84/199 (42,2 %) | 1,81 | 10/100 | 25/25 | 13,28 ms |
| YOLOv8s, 960, confianza 0,25 | 89/199 (44,7 %) | 1,17 | 15/100 | 25/25 | 16,31 ms |
| YOLOv8s, 1280, confianza 0,15 | 94/199 (47,2 %) | 1,16 | 18/100 | 23/25 | 22,41 ms |

Las tres configuraciones acertaron los 74 instantes vacíos de Egensevej-5. La configuración de 1280 acertó dos instantes ocupados y recuperó 1 de 28 cajas vehiculares por clase e IoU ≥ 0,5. Las otras dos configuraciones devolvieron cero en los 25 instantes ocupados. Las muestras nocturnas presentan deslumbramiento severo y vehículos pequeños. Las corridas preceden al aviso heurístico por áreas saturadas, por lo que el indicador de calidad de entonces clasificó las 99 muestras de Egensevej-5 como válidas. El aviso actual marca observaciones degradadas y deja visible el conteo para revisión. Su capacidad para detectar fallos por deslumbramiento requiere validación propia. El umbral de C2 se debe evaluar con referencia manual en las condiciones del piloto.

En Ringvej-1, YOLOv8n a 640 y confianza 0,25 en CPU coincidió en 9 de 100 instantes, con MAE de 2,58 vehículos y 68 falsos vacíos de 95 instantes ocupados. YOLOv8s a 960 y confianza 0,25 en GPU coincidió en 11 de 100, con MAE de 2,30 y 55 falsos vacíos de 95. Los informes de ambas corridas se guardaron como `aau-ringvej-n640-cpu.json` y `aau-ringvej-s960.json`.

## Ajuste fino experimental

El candidato se entrenó con Hjorringvej-3, Hasserisvej-2 y Ostre-3 y se validó internamente en Hobrovej-1. Para la comparación externa, tanto el candidato como YOLOv8s preentrenado usaron 960 píxeles, confianza 0,25 y ByteTrack, antes de la supresión de cajas duplicadas entre clases. Se verificó que los 299 índices de video y sus conteos de referencia coincidían entre las dos evaluaciones. Las observaciones degradadas con un conteo disponible conservan su valor numérico en esta comparación.

| Métrica | YOLOv8s preentrenado | Candidato ajustado |
| --- | ---: | ---: |
| Concordancia exacta global | 100/299 (33,4 %) | 36/299 (12,0 %) |
| Concordancia en Hadsundvej-2 | 15/100 | 2/100 |
| Concordancia en Egensevej-5 | 74/99 | 34/99 |
| Concordancia en Ringvej-1 | 11/100 | 0/100 |
| MAE global (vehículos) | 1,55 | 4,80 |
| Falsos vacíos sobre instantes ocupados | 80/220 | 6/220 |
| Falsas ocupaciones sobre instantes vacíos | 1/79 | 53/79 |

El candidato redujo los falsos vacíos y aumentó el sobreconteo. Su sesgo medio fue de +4,68 vehículos y dejó 53 de 79 escenas vacías con un conteo positivo. En su corrida el aviso de saturación marcó 18 de los 99 cuadros de Egensevej-5 como degradados. El informe queda en `aau-adapted-s960.json` y el SHA-256 del peso experimental es `617a41273e01e25d5e1f9cf84e74e0ad5fa495dc8f78fedeb710ab6d0f1c595d`.

Un segundo candidato se evaluó con el motor actualizado, la misma configuración y los mismos 299 instantes y referencias que la corrida preentrenada posterior a la supresión de duplicados entre clases. Los dos informes comparten hashes de código y fuentes.

| Métrica | YOLOv8s preentrenado | Segundo candidato |
| --- | ---: | ---: |
| Concordancia exacta global | 102/299 (34,1 %) | 110/299 (36,8 %) |
| Concordancia en Hadsundvej-2 | 17/100 | 10/100 |
| Concordancia en Egensevej-5 | 74/99 | 74/99 |
| Concordancia en Ringvej-1 | 11/100 | 26/100 |
| MAE global (vehículos) | 1,56 | 1,32 |
| Falsos vacíos sobre instantes ocupados | 80/220 | 31/220 |
| Falsas ocupaciones sobre instantes vacíos | 1/79 | 11/79 |

En Egensevej-5, el segundo candidato tuvo 5 falsos vacíos de 25 instantes ocupados y 11 falsas ocupaciones de 74 instantes vacíos. En Ringvej-1 tuvo 26 falsos vacíos de 95 instantes ocupados. El sesgo global fue de +0,15 vehículos. El informe es `aau-adapted-v2-s960-agnostic.json` y el SHA-256 del peso es `8b3d3329dfcb43efaac4ca64b1cf2ff9f0b63f276c70509b047c588fb2bf5e3b`.

## MTID, acceso cercano a la línea de parada

El conjunto MTID de Jensen, Møgelmose y Moeslund incluye una cámara de poste fijo que observa una intersección semaforizada de día. Se usaron JPG oficiales de esa cámara a 1024 × 640 y sus cajas publicadas. La versión 1 de `infrastructure-mscoco.json` ocupa 16 832 831 bytes y tiene SHA-256 `a5111f1454f5834660135eb585631db9af46971bf6922262f38528a36bfa764f`. [Universidad de Aalborg](https://vap.aau.dk/mtid/), [datos de los autores](https://www.kaggle.com/datasets/andreasmoegelmose/multiview-traffic-intersection-dataset), [artículo](https://doi.org/10.1109/ITSC45102.2020.9294694), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

El polígono se fijó antes de ejecutar el detector. Sus vértices en píxeles son `(0,468), (121,380), (181,486), (0,623)`. Cubre la franja visible alrededor de la línea de parada de un carril del acceso cercano inferior izquierdo, hacia el cruce. El borde de la cámara corta la aproximación y limita la longitud de cola observable. Se incluyeron automóviles, autobuses, camiones y motos con el centro inferior de su caja visible dentro del polígono. Se recortaron al cuadro las cajas de referencia que excedían sus bordes para aplicar el mismo criterio que a las cajas del detector. Las imágenes aisladas se evalúan sin ByteTrack.

El JSON publicado incluye 2 441 registros. Se excluyeron dos con un identificador de imagen compartido y uno sin anotaciones. Se escogieron 300 posiciones a intervalos uniformes en las 2 438 entradas elegibles, ordenadas por número de cuadro. Las etiquetas con cajas recortadas aportan 278 instantes vacíos y 22 con un vehículo en el polígono. Los cuadros vienen de una sola cámara y secuencia, con vehículos parcialmente visibles que aparecen repetidamente junto al borde. El subconjunto carece de referencias de dos o más vehículos en el polígono.

```powershell
$cache = Join-Path $env:USERPROFILE '.cache\smart-signals'
python -m scripts.benchmark_mtid --prepare
python -m scripts.benchmark_mtid --model "$cache\models\yolov8s.pt" --output "$cache\datasets\reports\mtid-near-stop-s960.json"
```

La preparación descarga los JPG individuales, que suman 7 340 513 bytes, y guarda los enlaces, tamaños y hashes en `mtid/samples-v1.json`. La evaluación comprueba el manifiesto, cuyo SHA-256 se fijó en `fd8186b30ab82bce8bde50eb82e0ee44ff6813ffe0008b3936ee5694a55730ee` antes de ejecutar la inferencia.

YOLOv8s preentrenado se ejecutó en CPU, con cuadro completo, 960 píxeles, confianza 0,25 y supresión de duplicados entre clases. Los 300 cuadros tuvieron salida válida.

| Vehículos en la referencia | Conteos exactos | Error de presencia |
| --- | ---: | --- |
| 1 (banda 1–2) | 13/22 (59,1 %) | 9/22 falsos vacíos |
| 0 | 272/278 (97,8 %) | 6/278 falsas ocupaciones |
| 3 o más | Sin muestras | Sin muestras |
| Total | 285/300 (95,0 %) | MAE de 0,05 vehículos |

Los cuadros vacíos representan 278 de los 300 instantes. Un conteo constante de cero acertaría en el 92,7 % global. La concordancia entre los cuadros ocupados describe mejor la detección de un vehículo en esta franja. El resultado tampoco mide el desempeño con colas, con referencias de dos vehículos ni en condiciones nocturnas del mismo acceso. El informe `mtid-near-stop-s960.json` contiene las observaciones y los hashes de la corrida.

## Video diurno para pruebas de interfaz

`wikimedia/Street_traffic.webm` dura 35 segundos y muestra tráfico diurno de Market Street, San Francisco, con vehículos cercanos que bloquean parte de la vista. Es un video de Editor publicado bajo CC BY 3.0. Puede usarse para revisar el ROI y anotar manualmente una muestra. El video publicado carece de etiquetas de presencia. OpenCV decodifica 1050 cuadros y expone tiempos del flujo coherentes con 30 fps. Su valor nominal de `CAP_PROP_FPS` puede ser 1000, por lo que conviene usar el tiempo de video para alinear anotaciones. [Ficha y atribución](https://commons.wikimedia.org/wiki/File:Street_traffic.webm), [original](https://upload.wikimedia.org/wikipedia/commons/7/70/Street_traffic.webm), [licencia](https://creativecommons.org/licenses/by/3.0/).

El original ocupa 26 490 572 bytes y tiene SHA-256 `e5facb24baf755f0c1193999d823e99f6032661ad2bdef5cc28cc1cbbfde03e2`.

## Lectura de métricas

Cada registro de `evaluate_samples` aporta `condition`, `manual_count` y `predicted_count`. Los conteos son enteros no negativos o `null`. Una referencia `null` reduce `coverage`, que es la proporción de instantes con referencia. `exact_agreement` divide coincidencias exactas para todas las referencias observables, incluyendo como fallos las predicciones `null`. `unavailable` cuenta esas predicciones ausentes y `prediction_coverage` indica qué fracción de referencias sí tuvo salida.

`mae` y `bias` usan los pares disponibles. Un sesgo positivo indica exceso de conteo. Las tasas de falso vacío y falsa ocupación se calculan sobre los pares cuya referencia era positiva y cero, respectivamente. Una tasa sin denominador se informa como `null`. Las bandas `0`, `1-2` y `3+` se determinan con el conteo de referencia. Las muestras sin referencia quedan fuera de las bandas. `--box-metrics` agrega precisión y exhaustividad de cajas por clase, con emparejamiento uno a uno e IoU ≥ 0,5. Esas métricas son distintas de la concordancia del número de vehículos presentes.
