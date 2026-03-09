# Plan de Desarrollo — Noise Monitor

Checklist ejecutable para el desarrollo del sistema. Cada item incluye contexto suficiente para que cualquier agente pueda retomarlo sin dependencia del anterior.

**Servidor:** `ssh roberto@10.10.10.137`
**Proyecto en servidor:** `~/noise-monitor/` (venv en `~/noise-monitor/venv/`)
**Docs de referencia:** Doc 3 (spec software), Doc 7 (setup node), Doc 8 (deployment)
**Sync dev → server:** `rsync -avz scripts/ roberto@10.10.10.137:~/noise-monitor/scripts/`

---

## Fase 0 — Infraestructura (COMPLETADA)

- [x] Hardware adquirido e instalado (ThinkCentre M910q, cámara ELP, iMM-6C, Coral USB)
- [x] Ubuntu Server 24.04 LTS instalado y optimizado para 24/7
- [x] Cámara USB estable en puerto trasero (1920x1080 @ 37 FPS)
- [x] Micrófono iMM-6C funcionando (usuario en grupo `audio`, libportaudio2 instalado)
- [x] Coral USB Accelerator funcionando (feranick/libedgetpu TF2.19.1 + ai-edge-litert 2.1.2)
- [x] Python 3.12 venv con todas las dependencias
- [x] Test scripts validados (`scripts/test_all.py` pasa los 3 componentes)
- [x] Git inicializado con docs y scripts

---

## Fase 0.5 — Videos de prueba para desarrollo sin balcón

Mientras la cámara no esté instalada en el balcón, se usan videos de tráfico descargados de YouTube como fuente simulada. El pipeline acepta tanto un índice de dispositivo USB (int) como un path a archivo de video (string) en `config.yaml` → `video.source`, gracias a que `cv2.VideoCapture` soporta ambos.

### 0.5.1 Descarga de videos

- [ ] Instalar `yt-dlp` en el server: `pip install yt-dlp` (fuera del venv, o con pipx).
- [ ] Crear directorio `data/test_videos/` en el server.
- [ ] Descargar 3-4 videos de tráfico con audio real (buscar "traffic camera live" o "urban traffic noise"):
  ```bash
  cd ~/noise-monitor/data/test_videos
  yt-dlp -f "bestvideo[height<=1080]+bestaudio/best[height<=1080]" \
    --merge-output-format mp4 -o "traffic_%(id)s.mp4" "URL_DEL_VIDEO"
  ```
  Criterios de selección: ángulo similar al del balcón (vista cenital o semi-cenital de calle), con audio de tráfico real (motores, bocinas, etc.), variedad de condiciones (día, noche, lluvia si hay).
- [ ] Verificar que los videos tienen audio: `ffprobe -i archivo.mp4` (debe mostrar stream de audio).

### 0.5.2 Configuración para modo simulado

- [ ] En `config.yaml`, cambiar `video.source` de `0` a la ruta del video:
  ```yaml
  video:
    source: "data/test_videos/traffic_xxx.mp4"  # ruta relativa al proyecto
  ```
- [ ] En `src/video_monitor.py`, agregar lógica de loop: cuando `cv2.VideoCapture.read()` retorna `False` (fin del video), hacer `cap.set(cv2.CAP_PROP_POS_FRAMES, 0)` para reiniciar desde el inicio. Esto permite correr el pipeline continuamente con un video finito.
- [ ] Para audio: el video de YouTube ya tiene pista de audio. Dos opciones:
  - **Opción A (recomendada)**: extraer el audio del video con ffmpeg y alimentarlo al pipeline en lugar del micrófono:
    ```bash
    ffmpeg -i traffic_xxx.mp4 -vn -ar 44100 -ac 1 -f wav traffic_xxx_audio.wav
    ```
    Luego en `audio_monitor.py`, agregar modo "file" que lee el .wav en chunks simulando captura en tiempo real.
  - **Opción B (más simple)**: usar el micrófono real (iMM-6C) y reproducir el audio del video por un parlante cerca. Menos reproducible pero no requiere cambios en audio_monitor.
- [ ] Agregar flag en `config.yaml` para modo simulado:
  ```yaml
  debug:
    simulate: true              # true = video de archivo, false = cámara real
    loop_video: true            # reiniciar video al terminar
    audio_from_video: true      # extraer audio del video en lugar de usar micrófono
  ```

### 0.5.3 Revertir para producción

Cuando se instale la cámara en el balcón, solo hay que cambiar:
```yaml
video:
  source: 0                     # volver al dispositivo USB
debug:
  simulate: false
```

---

## Fase 1 — Pipeline central de captura

El objetivo es que el sistema detecte y registre eventos de ruido con correlación visual. Todo el código vive en `src/` y se orquesta desde `main.py`.

### 1.1 Configuración y base de datos

- [x] Crear `config.yaml` real basado en la spec de Doc 3 sección 4. Ajustar `audio.device_index` (actualmente `5` para iMM-6C) y `video.source` (actualmente `0`).
- [x] Crear `schema.sql` según Doc 3 sección 3. Ejecutar para crear `data/events.db`.
- [x] Crear `requirements.txt` con las versiones exactas instaladas en el venv.

### 1.2 Utilidades base

- [x] Implementar `src/utils.py`: filtro A-weighting IEC 61672-1 (ver Doc 3 sección 6), función de conversión RMS→dB, carga de config.yaml, inicialización de SQLite.

### 1.3 Monitor de audio

- [x] Implementar `src/audio_monitor.py`: clase `AudioMonitor` con:
  - Captura continua vía `sounddevice.InputStream` callback
  - Buffer circular de audio (configurable, default 10s)
  - Aplicación de filtro A-weighting en cada bloque
  - Cálculo de dB(A) por bloque (~46ms con block_size=2048)
  - Detección de triggers: (a) pico instantáneo > peak_db, (b) sostenido > sustained_db por sustained_duration_ms
  - Propiedad `current_db` accesible para el dashboard en tiempo real
  - Señal al event_manager cuando se dispara un trigger

### 1.4 Monitor de video

- [x] Implementar `src/video_monitor.py`: clase `VideoMonitor` con:
  - Captura continua vía `cv2.VideoCapture`
  - Buffer circular de frames (configurable, default 10s)
  - Inferencia cada N frames usando Coral Edge TPU (`ai-edge-litert` + delegate `libedgetpu.so.1`)
  - Modelo: `models/ssd_mobilenet_v2_coco_edgetpu.tflite` (300x300 input)
  - Decodificación de detecciones: boxes, classes, scores
  - Propiedad `recent_detections` accesible para correlación
  - Cargar labels desde `models/coco_labels.txt`

### 1.5 Event Manager

- [x] Implementar `src/event_manager.py`: clase `EventManager` con:
  - Recibe triggers del AudioMonitor
  - Consulta VideoMonitor.recent_detections para correlación
  - Lógica de asignación de culpable por jerarquía (truck/bus > motorcycle > car, ver Doc 3 sección 7)
  - Cálculo de confianza (high si 1 vehículo, medium si jerarquía clara, low si ambiguo)
  - Volcado de clip ±5s de los buffers de audio y video a disco (ffmpeg para encoding)
  - Inserción del evento en SQLite con todos los campos del schema

### 1.6 Orquestador principal

- [x] Implementar `main.py`:
  - Carga config, inicializa DB
  - Arranca AudioMonitor, VideoMonitor y EventManager en threads
  - Manejo limpio de señales (SIGINT/SIGTERM → shutdown graceful)
  - Logging estructurado (nivel configurable)

### 1.7 Clasificador de audio (YAMNet)

YAMNet clasifica el tipo de sonido (521 clases AudioSet) para diferenciar vehículos de construcción, animales, alarmas, etc. Corre en CPU, no necesita el Coral. Ver Doc 3 sección 8.

- [ ] Descargar `yamnet.tflite` y `yamnet_labels.txt` del repo TFLite model hub. Guardar en `models/`.
- [ ] Instalar `resampy` en el venv del server (para resampleo 44.1kHz → 16kHz).
- [ ] Implementar `src/audio_classifier.py`: clase `AudioClassifier` con:
  - Carga de YAMNet TFLite vía `ai-edge-litert` (mismo runtime, sin delegate Edge TPU)
  - Resampleo de audio input a 16kHz mono
  - Inferencia → top class + score
  - Mapeo de clase AudioSet → categoría propia (`vehicle`, `construction`, `animal`, `alarm`, `human`, `other`) según `config.yaml` → `audio_classifier.category_map`
  - Score mínimo configurable (`min_score`, default 0.3)
- [ ] Crear `scripts/test_yamnet.py`: grabar 3s de audio, clasificar, imprimir top-5 clases con scores.
- [ ] Actualizar `schema.sql`: agregar columnas `sound_class`, `sound_category`, `sound_confidence` a tabla events. Agregar índice `idx_sound_category`.

### 1.8 Integración de YAMNet en pipeline

- [ ] Refactorizar `src/audio_monitor.py`: cuando se dispara un trigger, clasificar la ventana de audio con `AudioClassifier` antes de encolar el evento.
- [ ] Refactorizar `src/event_manager.py`: bifurcación según `sound_category`:
  - `vehicle` → correlación visual (VideoMonitor) como antes
  - Cualquier otra categoría → registro directo sin video
  - En ambos casos: guardar `sound_class`, `sound_category`, `sound_confidence` en el evento
- [ ] Actualizar `config.yaml` con sección `audio_classifier` (model_path, labels_path, min_score, category_map).
- [ ] Actualizar `main.py` para instanciar `AudioClassifier` y pasarlo al `AudioMonitor`.

### 1.9 Evaluación de modelo de video

- [ ] Descargar `efficientdet_lite0_edgetpu.tflite` pre-compilado de Google Coral model zoo (320x320 input).
- [ ] Comparar MobileNet V2 (~3ms, mAP 22%) vs EfficientDet-Lite0 (~10ms, mAP 26%) usando clips reales de la cámara.
- [ ] Elegir modelo definitivo y actualizar `config.yaml` y `models/`. Ambos corren 100% en Edge TPU; la velocidad no es cuello de botella (procesamos 6 frames/s).

### 1.10 Validación de pipeline

- [ ] Test manual: reproducir video de tráfico frente a la cámara + ruido fuerte → verificar que se genera evento en SQLite con clip y clasificación de sonido.
- [ ] Test con sonidos no vehiculares (ladrido, sirena, herramienta) → verificar que se registra sin correlación de video.
- [ ] Test de 1 hora continua sin crashes ni memory leaks.

---

## Fase 2 — Dashboard web

El objetivo es poder ver qué está pasando desde un browser en la red local (`http://10.10.10.137:8000`).

### 2.1 Estructura FastAPI

- [ ] Implementar `web/app.py`: aplicación FastAPI, montaje de static files y templates Jinja2.
- [ ] Implementar `web/routes.py`: endpoints de páginas y API.

### 2.2 Páginas del dashboard (HTMX)

- [ ] `templates/base.html`: layout base con nav, CSS (Pico CSS o similar para estilo sin build step).
- [ ] `templates/dashboard.html`: página principal con:
  - Indicador de dB actual (polling HTMX cada 1s a `/api/current-db`)
  - Contador de eventos de hoy / esta semana
  - Tabla de últimos 20 eventos con: timestamp, categoría de sonido, clase vehículo (si aplica), dB, confianza, botón reproducir clip
  - Histograma de ruido últimas 24h (imagen generada server-side con matplotlib)
  - Desglose por categoría de sonido (vehículo vs construcción vs animal vs otro)
- [ ] `templates/events.html`: tabla paginada de todos los eventos con filtros (fecha, categoría sonido, clase vehículo, dB mínimo).
- [ ] `templates/reports.html`: generación de reportes on-demand (rango de fechas, formato PDF/HTML).

### 2.3 API REST

- [ ] `GET /api/current-db` → nivel dB actual (JSON)
- [ ] `GET /api/events?limit=N&offset=M&category=X&vehicle_class=Y&min_db=Z` → lista de eventos
- [ ] `GET /api/events/{id}` → detalle de un evento
- [ ] `GET /api/clips/{id}` → streaming del clip de video
- [ ] `GET /api/stats?period=day|week|month` → estadísticas agregadas
- [ ] `POST /api/reports/generate` → genera reporte y retorna URL de descarga

### 2.4 Integración con pipeline

- [ ] Uvicorn se arranca dentro de `main.py` como thread adicional (o subprocess).
- [ ] El web app lee de SQLite (read-only para la mayoría de endpoints).
- [ ] `current_db` se expone desde AudioMonitor vía variable compartida thread-safe.

### 2.5 Validación

- [ ] Acceder a `http://10.10.10.137:8000` desde un browser y verificar que se ven datos en vivo.
- [ ] Reproducir un clip desde la tabla de eventos.
- [ ] Generar un reporte PDF de prueba.

---

## Fase 3 — Calibración y reportes

### 3.1 Calibración del micrófono

- [ ] Implementar `calibrate.py` (ver Doc 3 sección 8):
  - Graba N segundos con iMM-6C
  - Compara dB medido vs referencia de sonómetro
  - Calcula y guarda offset en config.yaml
  - Opcionalmente aplica curva de calibración individual del iMM-6C

### 3.2 Generador de reportes

- [ ] Implementar `src/report_generator.py`:
  - Query a SQLite por rango de fechas
  - Estadísticas: total eventos, por categoría de sonido, por clase vehicular, por hora del día, por día de semana
  - Histogramas con matplotlib
  - Top-10 eventos más ruidosos con links a clips
  - Output: HTML autocontenido y/o PDF
  - Opcionalmente: compilar top-10 clips en un solo MP4 con ffmpeg

---

## Fase 4 — Despliegue físico

### 4.1 Preparación

- [ ] Comprar: caja IP65, espuma antiviento, abrazaderas UV, adaptador USB-C→A, velcro, silicona.
- [ ] Montar cámara y micrófono en la caja con protección contra lluvia y viento.
- [ ] Tender cables al balcón (alimentación + USB si necesario).

### 4.2 Calibración in-situ

- [ ] Ejecutar `calibrate.py` con sonómetro de referencia en el balcón.
- [ ] Verificar ángulo de cámara: que cubra ambos carriles de la calle.
- [ ] Ajustar umbrales de trigger según el ruido ambiental real.

### 4.3 Servicio 24/7

- [ ] Configurar systemd service (ver Doc 8 sección 3).
- [ ] Configurar crons: reportes semanales + purga de disco (ver Doc 8 secciones 5-6).
- [ ] Prueba de 24h continuas: monitorear estabilidad USB, uso de CPU, crecimiento de disco.

### 4.4 Operación continua

- [ ] Revisar dashboard diariamente la primera semana.
- [ ] Ajustar umbrales si hay muchos falsos positivos/negativos.
- [ ] Primer reporte semanal real → revisar manualmente 20 eventos para evaluar precisión.

---

## Notas para el agente

- **Stack Python 3.12:** No usar pycoral ni tflite-runtime. Usar `ai-edge-litert` + `load_delegate('libedgetpu.so.1')` para video (Edge TPU). YAMNet usa el mismo `ai-edge-litert` pero sin delegate (corre en CPU). Ver `scripts/test_coral.py` para ejemplo funcional de video.
- **Sync local ↔ server:** Los docs viven localmente (Dropbox). El código ejecutable se sincroniza al server con rsync. El venv NO se sincroniza (se recrea en el server).
- **Activar venv en server:** `cd ~/noise-monitor && source venv/bin/activate`
- **Git:** El repo está en el directorio local del proyecto. Hacer commit después de cada fase o cambio significativo.
- **Dispositivos USB:** Cámara = índice 0, micrófono iMM-6C = índice 5 (puede cambiar al reboot, verificar con `sounddevice.query_devices()`).
- **Modo simulado:** La cámara no está instalada en el balcón todavía. Para desarrollo se usan videos de tráfico descargados de YouTube (ver Fase 0.5). `config.yaml` → `video.source` acepta int (dispositivo) o string (archivo). Asegurar que `VideoMonitor` implemente loop de video y que `AudioMonitor` soporte lectura de audio desde archivo .wav.
