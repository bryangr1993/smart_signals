# Evidencias de la entrega local

Copias de los informes originales generados durante el desarrollo del 10–11 de noviembre de 2026. Se conservan dentro del repositorio y de la carpeta del proyecto para que la evidencia no dependa únicamente de la caché local. No contienen videos ni pesos del modelo.

- [AAU, perfil operativo final](aau-final-s960-agnostic.json): tres escenas, 299 referencias; incluye hashes de código, pesos y fuentes. [Figura](aau-final-s960-agnostic.png).
- [AAU, segundo ajuste experimental](aau-adapted-v2-s960-agnostic.json): mismo código, configuración y referencias; cambia el peso.
- [MTID, acceso diurno parcial](mtid-near-stop-s960.json): 300 imágenes, 278 referencias vacías y 22 con un vehículo. [Polígono fijado antes de inferir](mtid-acceso-roi-preview.png).
- Los restantes JSON conservan los experimentos previos descritos en [DATASETS.md](../DATASETS.md), incluyendo el primer ajuste y las comparaciones de resolución/modelo.

Las rutas absolutas de los informes corresponden a este computador. Para repetirlos en otro equipo, instalar el entorno y usar los descargadores y comandos documentados; los hashes permiten verificar los archivos. El código de evaluación y el motor registrados no cambian al corregir documentación o la interfaz.

AAU RainSnow: Chris H. Bahnsen y Thomas B. Moeslund, Aalborg University, [fuente](https://doi.org/10.34740/kaggle/dsv/105294). MTID: Jensen, Møgelmose y Moeslund, Aalborg University, [fuente](https://vap.aau.dk/mtid/). Ambos conjuntos se distribuyen bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). La vista del polígono añade una superposición al fotograma de MTID; la figura AAU representa métricas derivadas. Las condiciones, limitaciones y metodología completas están en [DATASETS.md](../DATASETS.md).
