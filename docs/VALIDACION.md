# Validación del conteo de presencia

El objetivo es medir cuántos vehículos están presentes en una zona en un instante. La evaluación de aforos acumulados, giros y tiempos semafóricos pertenece a otros componentes.

## Referencia independiente

Seleccionar instantes antes de ver los errores del modelo. La interfaz guarda muestras cada cinco segundos por defecto, hasta 240 por sesión. El usuario puede modificar el intervalo antes de comenzar. Para estudios largos, planificar sesiones que cubran todas las franjas, en vez de asumir que las primeras 240 muestras representan todo el día.

Anotar el mismo polígono, tipos de vehículo y punto de referencia configurados. El centro inferior de la caja aproxima contacto con el suelo; no es una calibración física. Incluir vehículos detenidos. No inferir un total oculto a partir de las predicciones del propio modelo. Si no existe referencia verificable, marcar la muestra como no anotada/indeterminada y conservarla en la cobertura.

La pantalla de revisión muestra inicialmente el cuadro sin cajas automáticas, con el polígono. Una segunda persona revisará una muestra y las discrepancias para la evaluación formal. Separar videos/días de ajuste y evaluación: fotogramas vecinos del mismo video no son observaciones independientes para estimar incertidumbre.

## Métricas

`exact_agreement = coincidencias exactas / instantes con referencia manual`.

Una predicción no disponible en un instante con referencia cuenta como fallo de concordancia. El error absoluto medio (`mae`) y el sesgo (`bias`) se calculan entre pares comparables; deben leerse junto a `prediction_coverage`, la proporción de referencias con predicción. `coverage` mide la proporción de muestras guardadas para revisión que tiene referencia manual. Los cortes sin imagen quedan en CSV/JSONL y en el resumen; no generan una imagen anotable ni entran en esa cobertura. Un denominador vacío produce `null`, nunca un cero que parezca buen resultado.

El falso vacío es la proporción de pares con vehículos manuales para los que se predijo cero. La falsa presencia usa pares manualmente vacíos con predicción positiva. Ambos requieren reportar el número de casos y la disponibilidad.

Los reportes separan condición y bandas de referencia 0, 1–2 y 3 o más. No ocultar problemas de un único vehículo tras muchos cuadros vacíos ni atribuir desempeño nocturno a una prueba diurna.

La concordancia exacta ≥90 % es una propuesta técnica para la señal de presencia. Debe conciliarse con la definición institucional del indicador antes de usarla como acreditación formal. No es intercambiable con precisión de cajas ni con uno menos el error porcentual agregado.

## Qué demuestra cada prueba

| Evidencia | Alcance |
| --- | --- |
| Pruebas sintéticas de motor, geometría y estados | Correctitud de reglas, nulidad, identidades, bordes y registros. |
| Videos externos anotados | Desempeño en sus condiciones y regiones evaluables, con clases y anotaciones publicadas. |
| Videos propios anotados de ESPOCH | Desempeño del montaje, encuadre y condiciones efectivamente observadas. |
| Piloto urbano independiente | Transferencia al acceso seleccionado, una vez autorizado y capturado. |

La lluvia o nieve de un dataset extranjero no acredita por sí sola condiciones de Riobamba. Tampoco la respuesta correcta ante una imagen negra demuestra detección de cualquier bloqueo parcial o movimiento de PTZ.

## Rendimiento

Medir frecuencia efectiva, tiempo de inferencia, antigüedad de la observación y recuperación de cortes. La latencia desde recepción local excluye retrasos previos de cámara, red o NVR; medir la cadena completa requiere relojes de adquisición sincronizados o una prueba independiente. El p95 del resumen usa hasta los últimos 1.024 fotogramas procesados; los registros individuales permiten calcular otras ventanas. La generación JPEG se limita a cinco vistas por segundo en el hilo del motor y la interfaz solicita la vista más reciente cada 400 ms; el último cuadro se conserva al finalizar. La vista puede estar reducida respecto del original.

Para comprobar retraso al cambiar el conteo hacen falta segmentos anotados con mayor densidad temporal que la muestra general. La media del conteo no revela cuánto tardó el sistema en detectar que el último vehículo salió.

## Condiciones todavía necesarias

Capturar día, noche, lluvia cuando ocurra, sombras/contraluz, colas y baja demanda con las cámaras existentes y el encuadre de trabajo. Conservar configuración y versiones, etiquetas independientes y registro de limitaciones. Cualquier condición sin datos queda **no evaluada**.
