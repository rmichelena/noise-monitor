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

## Fase 1 — Pipeline central de captura

El objetivo es que el sistema detecte y registre eventos de ruido con correlación visual. Todo el código vive en `src/` y se orquesta desde `main.py`.

### 1.1 Configuración y base de datos

- [ ] Crear `config.yaml` real basado en la spec de Doc 3 sección 4. Ajustar `audio.device_index` (actualmente `5` para iMM-6C) y `video.source` (actualmente `0`).
- [ ] Crear `schema.sql` según Doc 3 sección 3. Ejecutar para crear `data/events.db`.
- [ ] Crear `requirements.txt` con las versiones exactas instaladas en el venv.

### 1.2 Utilidades base

- [ ] Implementar `src/utils.py`: filtro A-weighting IEC 61672-1 (ver Doc 3 sección 6), función de conversión RMS→dB, carga de config.yaml, inicialización de SQLite.

### 1.3 Monitor de audio

- [ ] Implementar `src/audio_monitor.py`: clase `AudioMonitor` con:
  - Captura continua vía `sounddevice.InputStream` callback
  - Buffer circular de audio (configurable, default 10s)
  - Aplicación de filtro A-weighting en cada bloque
  - Cálculo de dB(A) por bloque (~46ms con block_size=2048)
  - Detección de triggers: (a) pico instantáneo > peak_db, (b) sostenido > sustained_db por sustained_duration_ms
  - Propiedad `current_db` accesible para el dashboard en tiempo real
  - Señal al event_manager cuando se dispara un trigger

### 1.4 Monitor de video

- [ ] Implementar `src/video_monitor.py`: clase `VideoMonitor` con:
  - Captura continua vía `cv2.VideoCapture`
  - Buffer circular de frames (configurable, default 10s)
  - Inferencia cada N frames usando Coral Edge TPU (`ai-edge-litert` + delegate `libedgetpu.so.1`)
  - Modelo: `models/ssd_mobilenet_v2_coco_edgetpu.tflite` (300x300 input)
  - Decodificación de detecciones: boxes, classes, scores
  - Propiedad `recent_detections` accesible para correlación
  - Cargar labels desde `models/coco_labels.txt`

### 1.5 Event Manager

- [ ] Implementar `src/event_manager.py`: clase `EventManager` con:
  - Recibe triggers del AudioMonitor
  - Consulta VideoMonitor.recent_detections para correlación
  - Lógica de asignación de culpable por jerarquía (truck/bus > motorcycle > car, ver Doc 3 sección 7)
  - Cálculo de confianza (high si 1 vehículo, medium si jerarquía clara, low si ambiguo)
  - Volcado de clip ±5s de los buffers de audio y video a disco (ffmpeg para encoding)
  - Inserción del evento en SQLite con todos los campos del schema

### 1.6 Orquestador principal

- [ ] Implementar `main.py`:
  - Carga config, inicializa DB
  - Arranca AudioMonitor, VideoMonitor y EventManager en threads
  - Manejo limpio de señales (SIGINT/SIGTERM → shutdown graceful)
  - Logging estructurado (nivel configurable)

### 1.7 Evaluación de modelo

- [ ] Descargar `efficientdet_lite0_edgetpu.tflite` pre-compilado de Google Coral model zoo (320x320 input).
- [ ] Comparar MobileNet V2 (~3ms, mAP 22%) vs EfficientDet-Lite0 (~10ms, mAP 26%) usando clips reales de la cámara.
- [ ] Elegir modelo definitivo y actualizar `config.yaml` y `models/`. Ambos corren 100% en Edge TPU; la velocidad no es cuello de botella (procesamos 6 frames/s).

### 1.8 Validación de pipeline

- [ ] Test manual: reproducir video de tráfico frente a la cámara + ruido fuerte → verificar que se genera evento en SQLite con clip.
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
  - Tabla de últimos 20 eventos con: timestamp, clase vehículo, dB, confianza, botón reproducir clip
  - Histograma de ruido últimas 24h (imagen generada server-side con matplotlib)
- [ ] `templates/events.html`: tabla paginada de todos los eventos con filtros (fecha, clase, dB mínimo).
- [ ] `templates/reports.html`: generación de reportes on-demand (rango de fechas, formato PDF/HTML).

### 2.3 API REST

- [ ] `GET /api/current-db` → nivel dB actual (JSON)
- [ ] `GET /api/events?limit=N&offset=M&class=X&min_db=Y` → lista de eventos
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
  - Estadísticas: total eventos, por clase, por hora del día, por día de semana
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

- **Stack Python 3.12:** No usar pycoral ni tflite-runtime. Usar `ai-edge-litert` + `load_delegate('libedgetpu.so.1')`. Ver `scripts/test_coral.py` para ejemplo funcional.
- **Sync local ↔ server:** Los docs viven localmente (Dropbox). El código ejecutable se sincroniza al server con rsync. El venv NO se sincroniza (se recrea en el server).
- **Activar venv en server:** `cd ~/noise-monitor && source venv/bin/activate`
- **Git:** El repo está en el directorio local del proyecto. Hacer commit después de cada fase o cambio significativo.
- **Dispositivos USB:** Cámara = índice 0, micrófono iMM-6C = índice 5 (puede cambiar al reboot, verificar con `sounddevice.query_devices()`).
