# Análisis acústico experimental

## Auditoría de `main` en `a43fe2f`

| Área | Comportamiento observado en el código |
| --- | --- |
| Audio | `AudioMonitor` recibe bloques del Dayton iMM-6C USB vía PortAudio, en mono flotante. La configuración de ejemplo usa 44 100 Hz y 4096 muestras (~93 ms). El modo simulado puede leer WAV. |
| Niveles | El monitor aplica ponderación A y calcula RMS por bloque. Suma `calibration_offset` en dB. El pico guardado suele ser el nivel del bloque que disparó el evento y el resumen por minuto usa una media aritmética de dB. La auditoría no encontró LAeq energético, constante Fast, espectro persistido ni una referencia absoluta de presión con ID y ganancia vigentes. |
| Disparo | Pico de 85 dB o sucesión sobre 70 dB durante ~1,5 s en el ejemplo. Umbral de correlación de 65 dB y cooldown de 30 s en el consumidor. La configuración efectiva del nodo es local. |
| YAMNet | Modelo ligero con 521 etiquetas AudioSet, LiteRT en CPU y remuestreo a 16 kHz. El monitor clasifica ~1 s anterior al disparo en un hilo con cola de 8. Si se satura, pasa el evento sin etiqueta. El consumidor de pasajes también usa YAMNet para silenciar clips con habla. |
| Visión | OpenCV, SSD MobileNet V2 en Coral USB y seguimiento por centroides. El ejemplo usa 720p, 30 FPS, inferencia cada dos cuadros y un ROI de calle. Si falla el Coral, el código desactiva detección y reintenta. |
| Sincronización | Audio y video usan `time.time()` del host. El consumidor usa la ventana pre/post, con un fallback legado a detecciones recientes. El vínculo con el pasaje usa clase y proximidad temporal. Este método admite falsos positivos con tráfico simultáneo. |
| SQLite | `noise_events` registra disparos, etiqueta, correlación, clip y estado de audio. `vehicle_passages` registra cruces y `audio_levels`, resúmenes por minuto. La inicialización tolera migraciones previas. |
| Clips | Eventos de hasta ~10 s (5 s a cada lado) y pasajes de hasta 5 s. Se convierte el audio a PCM16 temporal y se guarda solo el MP4 final, con H.264 y AAC 64 kb/s, o sin pista de audio para clases de habla. Retención de ejemplo de 30 días para clips y 400 días para niveles. Los clips fijados están exentos. |
| Calibración | El archivo de respuesta del micrófono y el offset de la instalación son datos distintos de una calibración absoluta de la cadena de grabación. El repositorio contiene la configuración de ejemplo y la propuesta histórica de calibración, con ganancia y calibración operativas sin verificar. |
| Carga | README indica ThinkCentre M910q, i5-7500T y 8 GB. El documento histórico y el brief mencionan i5-6500T. El uso real de CPU/RAM, los FPS y los desbordes de audio requieren una medición del nodo. El buffer de video de 10 s puede ocupar ~790 MiB a 720p y 30 FPS con cuadros BGR. |
| Datos disponibles | El checkout no contiene clips de producción, SQLite, modelo descargado, calibración ni evaluaciones humanas. Las pruebas existentes generan audio y video sintéticos. |

## Plan incremental

1. Ejecutar MoSQITo en un entorno aislado con presión en Pa y registrar resultados, errores, versión y tiempos por clip. Calcular el nivel físico sobre el clip con la ponderación A existente y promedio energético. Mantener los niveles legados con su semántica original.
2. Reutilizar YAMNet LiteRT en CPU en ventanas a lo largo del clip. Guardar etiquetas, scores de modelo y evidencia visual por separado. Reconocer indicios domésticos y declarar incertidumbre ante señales conflictivas o falta de evidencia.
3. Procesar clips existentes desde un proceso separado y optativo. Versionar resultados en una tabla propia, acotar el lote, registrar latencia y permitir reintentos o reprocesamiento por perfil. Preservar la política de clips mudos y mantener el análisis fuera de la ruta de captura.
4. Comparar dBA, métricas perceptivas y etiquetas con clips reales anotados. Ensayar un modelo publicado de PA con sus variables y supuestos completos y compararlo con oyentes. Dejar LAS y el entrenamiento local para una decisión posterior.

## Instalación y ejecución offline

~~~bash
python3.12 -m venv .venv/analysis
.venv/analysis/bin/pip install -r requirements-analysis.txt
bash scripts/download_yamnet.sh
.venv/analysis/bin/python scripts/analyze_audio.py offline data/manifest.json --csv data/metricas.csv --summary data/evaluacion.json
~~~

Se necesitan `ffmpeg` y `ffprobe`. Las opciones globales `--model` y `--labels` permiten usar rutas alternativas. `scripts/download_yamnet.sh /ruta/a/modelos` descarga el modelo y regenera sus etiquetas desde el CSV de TensorFlow. Las etiquetas con comas se conservan completas.

Ejemplo de manifiesto local. La calibración del ejemplo es ficticia. El factor real convierte una unidad de muestra digital decodificada a pascales. Mantén las grabaciones, factores y anotaciones privadas.

~~~json
[
  {
    "event_id": "cocina-01",
    "clip_path": "clips/cocina-01.wav",
    "calibration": {"id": "cadena-calibrada-01", "pa_per_unit": 0.25, "codec": "pcm_s16le"},
    "human_review": {
      "category": "vajilla",
      "expected_yamnet_label": "Dishes, pots, and pans",
      "expected_attribution": "indoor",
      "listening_condition": "original_calibrated",
      "perceived_loudness": 4,
      "annoyance_rating": 3
    }
  }
]
~~~

Las rutas relativas se resuelven desde el manifiesto. Un clip puede indicar `audio_muted: true` y el analizador omite su pista. Para la comparación audiovisual offline, `visual_labels` contiene clases del monitor como `car` o `motorcycle`. `visual_vehicle_confirmed: true` indica que la anotación visual identifica un único pasaje candidato en la ventana. En caso de duda se conserva `uncertain`.

El CSV incluye una fila por clip, SHA-256 del medio, códec, métricas, versiones, parámetros y motivo de los valores ausentes. El JSON opcional resume la primera etiqueta de YAMNet y la atribución con matrices de confusión, precisión, recall y falsos positivos. También mide si la etiqueta esperada apareció entre las emitidas. Para la molestia, compara la correlación de rangos de cada métrica con el LAeq de los mismos eventos, por condición de escucha y categoría.

### Cómo interpretar las métricas

- LAeq es el promedio energético con ponderación A durante el clip decodificado. LAFmax usa una constante de 125 ms que parte de energía cero al inicio del clip. Ambos se referencian a 20 µPa y la ponderación digital se normaliza a 1 kHz. El `db_max` histórico corresponde al disparo y se conserva.
- MoSQITo recibe audio mono en Pa remuestreado a 48 kHz y requiere al menos 0,5 s en este análisis. La sonoridad variable Zwicker ISO 532-1 se resume como N5, el percentil 95 temporal. La agudeza DIN 45692 se promedia donde la sonoridad es positiva y la aspereza Daniel–Weber se promedia con 50 % de solapamiento.
- La tonalidad es el máximo TNR ECMA en dB entre segmentos completos de 0,5 s. La implementación estacionaria se aplica a cada segmento. Cero puede indicar ausencia de tonos detectados por el algoritmo en su banda de búsqueda. Un dato ausente se exporta como campo vacío o `null` según el formato.
- YAMNet corta ventanas de 0,975 s con paso de 0,48 s, incluye una ventana al final y conserva el máximo score por clase. El modelo LiteRT que ya usa el proyecto expone 521 scores, sin una salida de embeddings. Los scores son evidencia de clasificación y pueden reflejar pasajes distintos dentro del mismo clip.
- La atribución espacial es una hipótesis. Una etiqueta específica y un único pasaje vinculado compatible permiten sugerir exterior si falta evidencia competidora. Indicios domésticos sin evidencia de tráfico permiten sugerir interior. Voces, escenas mixtas y coincidencias insuficientes pueden quedar inciertas. Bocina, aceleración y arranque se muestran como comportamientos acústicos.
- El valor de molestia permanece vacío en esta versión. El modelo clásico de PA requiere además fuerza de fluctuación en vacil y una convención de agregación documentada. MoSQITo 1.2.1 carece de esta métrica. El protocolo siguiente permite comparar una futura variante publicada con evaluaciones humanas.

## Procesamiento continuo o histórico

El worker se ejecuta como un proceso separado. Arranca bajo demanda, consulta los eventos ya guardados y toma como máximo cuatro por lote. En modo continuo espera por defecto 45 s desde el timestamp del evento para dar margen al tracker y pausa 15 s entre lotes. Se puede iniciar con prioridad reducida y un lote pequeño para medir el impacto antes de incrementar la capacidad.

~~~bash
nice -n 15 .venv/analysis/bin/python scripts/analyze_audio.py worker --watch --limit 1
.venv/analysis/bin/python scripts/analyze_audio.py worker --since 2026-11-20T00:00:00 --limit 20 --profile comparacion-01
~~~

El worker reconoce `noise_events` y el esquema unificado `events` de la PR #2. Si ambos existen, prioriza el unificado. Para analizar filas legadas explícitamente, usa `--event-table noise_events`. Del esquema unificado se procesan los eventos acústicos. Se guarda un resultado por revisión del clip y perfil en la tabla aditiva `acoustic_analysis`. Un clip reemplazado puede volver a analizarse sin reutilizar el resultado anterior.

El proceso escribe `pending` antes de analizar. Si se interrumpe, retoma ese intento al reiniciar. Los fallos reintentan hasta tres veces, con esperas de 60 s y 120 s. `--retry-failed` reinicia los intentos del perfil y `--profile` permite reprocesar conservando las ejecuciones anteriores. Los estados, intentos y latencia se consultan desde **Ver métricas** en la tabla de eventos o en `/api/events/{id}/analysis`. Detener el worker deja activa la grabación. El worker omite los clips mudos y opera sobre las grabaciones existentes.

Para habilitar métricas absolutas en el worker, registra la calibración aplicable a los formatos y clips de un intervalo verificado. Usa el mismo ajuste de micrófono, ganancia, montaje y codificación de la grabación. Para AAC, verifica el factor con una referencia que atraviese también la codificación y la decodificación. AAC puede ocultar un recorte previo de la señal. Excluye grabaciones saturadas en la cadena original. La calibración requiere sensibilidad absoluta de la cadena. El worker compara el intervalo completo del clip con la vigencia de la calibración. Para archivos históricos cuyos parámetros de pre/post-roll o ganancia cambiaron, usa un manifiesto con calibración por clip.

~~~yaml
acoustic_analysis:
  calibration:
    id: dayton-cadena-aac-verificada-01
    pa_per_unit: 0.25  # Solo ejemplo, sustituir por un valor medido.
    codec: aac
    valid_from: "2026-11-20T10:00:00"
    valid_until: "2026-11-21T10:00:00"
~~~

La fecha de vigencia usa la hora local de la instalación, como los registros de eventos. Si falta calibración, el worker puede producir etiquetas y atribución. El API conserva los códigos de incidencia y la base local guarda el detalle del fallo.

## Validación preliminar

Prueba con Python 3.12, FFmpeg 7.1.5 y el YAMNet LiteRT descargado. Los tonos mono de 2 s se generaron a 44,1 kHz y la amplitud del de 250 Hz se ajustó para igualar el LAeq del de 1 kHz. La presión de referencia del tono de 1 kHz es 0,02 Pa RMS.

| Clip | LAeq dBA | N5 sones | Agudeza acum | Aspereza asper | TNR dB | Tiempo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tono sintético 1 kHz | 60,00 | 4,02 | 1,058 | 0,001 | 94,99 | 4,39 s |
| Tono sintético 250 Hz | 60,00 | 6,38 | 0,421 | 0,001 | 0,00 | 4,00 s |
| Tono 1 kHz sin calibración | — | — | — | — | — | 0,26 s |
| Clip marcado como mudo | — | — | — | — | — | Omitido |
| Clip ausente | — | — | — | — | — | Fallido |

La ejecución independiente de cuatro casos registró un máximo de 373 196 KiB de RSS en este entorno. Los tiempos corresponden a esta máquina. Hace falta medir carga y desbordes en el ThinkCentre.

La segunda prueba usó ocho grabaciones públicas sin calibración, video ni evaluaciones de molestia. La selección corresponde a los conjuntos 1 y 2 del dataset de Karol J. Piczak. Las grabaciones conservan su licencia y atribución original. [ESC-50](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6)

| Anotación | Clips ESC-50 | Primera etiqueta |
| --- | --- | --- |
| Bocina | `1-17124-A-43`, `2-100648-A-43` | Silence, Vehicle |
| Motor | `1-18527-A-44`, `2-106014-A-44` | Vehicle, Vehicle |
| Motosierra | `1-116765-A-41`, `2-50667-A-41` | Chainsaw, Chainsaw |
| Vidrio roto | `1-20133-A-39`, `2-106881-A-39` | Silence, Silence |

Una bocina apareció como etiqueta secundaria (0,41). La primera etiqueta específica acertó dos de ocho clips, ambos de motosierra. Los clips tomaron entre 0,21 y 0,27 s y todos quedaron con atribución incierta. Esta muestra señala la necesidad de revisar etiquetas secundarias, clases genéricas y silencios dentro de las ventanas con ejemplos del entorno.

### Validación local pendiente

- Construir un conjunto por fecha y sesión con tráfico normal y ruidoso, motos, camiones, bocinas, aceleraciones, vajilla, TV, voces, maquinaria y mezclas. Cubrir niveles bajos y altos. Registrar fuente y comportamiento por evidencia independiente y marcar los casos ambiguos.
- Comparar la ubicación actual del Dayton con una prueba exterior conservando el resto de la cadena. Registrar sensibilidad absoluta, ganancia, montaje, fecha y, si aplica, respuesta en frecuencia.
- Presentar los clips en orden aleatorio a varios oyentes. Registrar sonoridad y molestia con escalas consistentes, por ejemplo 0–10, y conservar los ratings individuales antes de calcular un resumen por clip. Correr por separado el nivel original calibrado y el nivel normalizado. Para comparar métricas de la escucha normalizada, analizar el archivo efectivamente reproducido y usar otro ID y `listening_condition`.
- Comparar matriz de confusión, errores por clase y rangos frente a LAeq. Medir CPU/RSS, latencia por evento, cola pendiente, FPS y desbordes de audio antes y después de activar el worker en condiciones comparables.
- Usar SQAT como referencia al evaluar una variante PA con sonoridad, agudeza, aspereza y fluctuación. Probar en clips de referencia antes de cotejar con oyentes. Si un modelo local resulta necesario, separar entrenamiento y prueba por fecha o sesión.

## Compatibilidad con las PR abiertas

La PR #1 cambia la ganancia A, el manejo de silencio y la exportación. La normalización explícita a 1 kHz estabiliza el cálculo offline en ambos estados del filtro. [PR #1](https://github.com/rmichelena/noise-monitor/pull/1)

La PR #2 introduce `events` y puede reemplazar el clip al combinar un pasaje con un disparo. Los resultados conservan el esquema y la revisión que les corresponden. El worker crea su tabla al iniciarse. Al integrar la plantilla unificada, conserva el enlace de análisis para `noise` y `both`. [PR #2](https://github.com/rmichelena/noise-monitor/pull/2)
