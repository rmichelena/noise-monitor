# Noise Monitor

Nodo de monitoreo de ruido urbano sobre un mini-PC. Captura audio y video 24/7,
detecta eventos ruidosos, clasifica la fuente del sonido, registra el paso de
vehículos y guarda evidencia local (dB, clase, clip y timestamp) para respaldar
reclamos por ruido.

El proyecto nace de un caso concreto: una calle en "cañón urbano" frente a un
balcón. El objetivo es producir evidencia cuantitativa —nivel sonoro, tipo de
fuente y hora— en lugar de percepciones subjetivas.

## Estado

Implementado en `src/`:

- Captura continua de audio con ponderación A (IEC 61672-1, 0 dB a 1 kHz) y
  cálculo de dB SPL; la sensibilidad del micrófono se ajusta con
  `audio.calibration_offset`.
- Captura de video e inferencia de objetos sobre Coral Edge TPU (`ai-edge-litert`).
- `CentroidTracker` para registro continuo de pasajes, con filtros de duración
  mínima y desplazamiento mínimo (descarta parpadeos del modelo y vehículos
  estacionados).
- Estimación de velocidad por desplazamiento horizontal, escalada con el
  tramo de pista configurado (`speed.reference_*`).
- Clip por pasaje: cada vehículo en movimiento guarda 5 s de video, aunque
  no supere el umbral de ruido (`vehicle_passages.clip_path`). Solo clases
  vehiculares; nunca personas sueltas.
- Clips con voz: si YAMNet clasifica habla, el clip se guarda sin audio
  (`audio_muted`), para no publicar conversaciones de terceros. Aplica también
  a los clips de pasajes.
- Triggers por pico instantáneo o por persistencia sobre umbral.
- Clasificación de audio con YAMNet (521 clases AudioSet) agrupadas en
  categorías: vehículo, construcción, animal, alarma, humano, otro. Cada
  evento usa hasta tres ventanas solapadas alrededor del disparo y conserva
  tres etiquetas candidatas con sus scores. El dashboard y `/api/events`
  muestran las candidatas. `Silence` queda fuera del ranking de fuentes y
  el umbral de YAMNet sigue determinando la etiqueta principal.
- Correlación audio-visión: si el sonido es vehicular se busca el pasaje
  reciente más plausible y se registra la clase "culpable".
- Persistencia en SQLite (`vehicle_passages`, `noise_events`), con clips MP4
  ±5 s por evento (buffer circular + `ffmpeg`).

Implementado además:

- Dashboard web (`web/`, FastAPI + Jinja2) servido por el propio pipeline en
  el puerto 8000: nivel en vivo, fondo sonoro por minuto, contadores del día,
  últimos eventos y clips.
- Reporte de turno (`scripts/nightly_report.py`) publicado a diario en Buzz
  mediante `noise-report.timer`.

Pendiente: generación de reportes HTML/PDF y páginas de eventos/reportes.

## Hardware de referencia

| Componente | Modelo |
|---|---|
| Cómputo | Lenovo ThinkCentre M910q (i5-7500T, 8 GB RAM, SSD NVMe 256 GB) |
| Sistema | Ubuntu Server 24.04 LTS (headless) |
| Cámara | USB UVC 1080p, gran angular (~160°) |
| Micrófono | Dayton Audio iMM-6C (USB-C, omnidireccional, calibración individual) |
| Acelerador | Google Coral USB Accelerator |

El software no depende de ese hardware exacto: cualquier x86-64 con Linux,
una cámara UVC y un micrófono USB debería funcionar. Sin Coral, la inferencia
cae a CPU (mucho más lenta).

## Requisitos

- Python 3.12.
- `ffmpeg` y `libportaudio2` a nivel de sistema.
- Runtime del Coral: `ai-edge-litert` (en `requirements.txt`) más `libedgetpu`.
  En Ubuntu 24.04 la vía probada es el paquete de
  [feranick/libedgetpu](https://github.com/feranick/libedgetpu); el repo apt
  oficial de Google no soporta 24.04.

```bash
sudo apt install ffmpeg libportaudio2
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml   # obligatorio: main.py lo carga al arrancar
```

## Configuración

```bash
cp config.example.yaml config.yaml   # hardware real
# o, para desarrollar sin balcón:
# cp config.simulate.yaml.example config.yaml
```

`config.yaml` está en `.gitignore`: cada nodo mantiene su configuración local
(índice de dispositivo, umbrales calibrados, coordenadas de las líneas de
velocidad). Los umbrales por defecto del ejemplo (85 dB pico / 70 dB sostenido)
son un punto de partida, no una calibración.

Al arrancar, `main.py` valida la config (`src.utils.validate_config`): una
clave ausente o un valor imposible detiene el proceso con un mensaje que nombra
`sección.clave`, en vez de morir dentro de un thread. Los avisos (buffer más
corto que la ventana del clip, `speed.enabled` sin `reference_left_x` /
`reference_right_x`) salen por el log sin impedir el arranque. Ambos ejemplos
versionados son configuraciones completas y válidas.

## Modelo de detección de video

El detector es configuración (`video.detector`), no código:

- `ssd` (por defecto): SSD MobileNet V2 COCO 300×300, el histórico. El crop
  de calzada se estira a 300×300.
- `yolo`: YOLO11n INT8 exportado a Edge TPU con entrada **160×640** (4:1),
  sin deformación ni tiles. Requiere `models/yolo11n_160x640_edgetpu.tflite`
  (`video.yolo_model_path`). En runtime solo usa `ai-edge-litert`; no necesita
  Ultralytics ni TensorFlow.

El `.tflite` no se versiona. Se exporta en un entorno aparte:

```bash
python -m venv /tmp/yoloenv
/tmp/yoloenv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
/tmp/yoloenv/bin/pip install ultralytics
/tmp/yoloenv/bin/python -c "from ultralytics import YOLO; \
  YOLO('yolo11n.pt').export(format='edgetpu', imgsz=(160, 640), int8=True, data='coco8.yaml')"
cp yolo11n_saved_model/yolo11n_full_integer_quant_edgetpu.tflite \
   models/yolo11n_160x640_edgetpu.tflite
```

Los pesos YOLO11 son **AGPL-3.0** (el proyecto es MIT): decisión consciente
del operador al activar `video.detector: yolo`.

Para producción, calibrar la cuantización INT8 con frames reales de esta
cámara (día y noche) en vez de `coco8`: la salida esperada del `.tflite` es
`[1, 84, N]` (4 cajas + 80 clases); el detector valida esa forma y el labelmap
COCO-80 al arrancar.

Modelo de clasificación de audio:

```bash
./scripts/download_yamnet.sh   # models/yamnet.tflite + labels
```

## Ejecución

```bash
source venv/bin/activate
python main.py
```

Arranca tres hilos: audio, video y `EventManager`. Los clips y la base quedan en
`data/` (ignorado por git).

Estadísticas rápidas de la base:

```bash
python scripts/stats.py --days 7
```

## Tests

```bash
python -m pytest            # config en pyproject.toml (testpaths + pythonpath)
python -m unittest discover -s tests   # equivalente, sin pytest
```

Los tests no necesitan cámara, micrófono, Coral ni `config.yaml`: parten de
`config.example.yaml`, escriben en un directorio temporal y solo requieren
`ffmpeg` para los dos casos que generan clips (si falta, se saltan). Los de
`AudioMonitor` se saltan el dispositivo de audio real.

## Scripts

| Script | Uso |
|---|---|
| `scripts/download_yamnet.sh` | Descarga YAMNet y sus etiquetas. `--coral` añade el modelo Edge TPU |
| `scripts/benchmark_yamnet.py` | Compara CPU y Coral con los mismos clips. [Protocolo para el ThinkCentre](docs/yamnet-coral.md) |
| `scripts/download_test_videos.sh` | Baja videos de tráfico (`yt-dlp`) para modo simulado |
| `scripts/extract_audio_from_video.sh` | Extrae el WAV de un video de prueba |
| `scripts/setup_simulate_videos.sh` | Prepara `data/test_videos/` y muestra la config simulada |
| `scripts/stats.py` | Consultas agregadas sobre `events.db` |
| `scripts/test_camera.py`, `test_audio.py`, `test_coral.py`, `test_all.py` | Verificación de hardware, componente por componente |

## Estructura

```
main.py                 Orquestador (audio + video + eventos)
config.example.yaml     Configuración de referencia para hardware real
config.simulate.yaml.example  Configuración de referencia para modo simulado
schema.sql              Esquema SQLite
src/
  audio_monitor.py      Captura, A-weighting, triggers
  audio_classifier.py   YAMNet → categorías
  video_monitor.py      Captura, inferencia Edge TPU
  vehicle_tracker.py    CentroidTracker + filtros
  event_manager.py      Correlación, clips, persistencia
  utils.py              A-weighting, dB, config, SQLite
scripts/                Herramientas operativas y tests de hardware
```

## Operación

```bash
sudo cp deploy/noise-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now noise-monitor     # pipeline 24/7, Restart=always
sudo systemctl enable --now noise-report.timer # reporte diario 07:00 a Buzz
```

- `noise-monitor.service` corre el pipeline con reinicio automático. El buffer
  de video (10 s a 1080p30) ocupa ~1.9 GB de RSS; el límite del unit es 4 GB.
- Si el Coral no enumera (cable, puerto, alimentación), `VideoMonitor` registra
  el error y sigue **sin detección visual**: audio, YAMNet y clips continúan.
  Revisar con `lsusb` (`1a6e:089a` o `18d1:9302`) y `dmesg` (`error -62`).
- `scripts/nightly_report.py` resume la ventana nocturna (10 h por defecto) y
  `noise-report.service` lo publica en el canal usando la clave del agente
  cargada como *systemd credential* (`/home/roberto/buzz/credentials/noise.key`).
- Dashboard: `http://<host>:8000` (host/puerto en `config.yaml` → `web`). Lee SQLite
  en solo-lectura y muestra dB instantáneo, pico por minuto de las últimas 2 h,
  contadores del día, sliders de nivel y velocidad, marcas ★ de eventos y
  pasajes fijados (desde el modal del clip) y listas de eventos y pasajes
  con filtros. Exportación CSV de la evidencia para un periodo
  (`/api/export/events.csv` y `/api/export/passages.csv`, parámetros
  `days`, `min_db`, `vehicle`, `direction`, `category`: los mismos filtros
  del dashboard). Incluye una
  vista en vivo MJPEG (`/stream`) con las detecciones dibujadas, que solo se
  codifica mientras la transmisión está activada.
- Retención: `scripts/purge.py` borra clips de más de 30 días (la fila del
  evento y la del pasaje se conservan sin clip) y `audio_levels` de más de 400
  días. Los clips **fijados** (`pinned`) no se borran nunca.
  `noise-purge.timer` lo ejecuta los domingos a las 04:00. Configurable en
  `config.yaml` → `retention`.
- Acceso externo: `noise-tunnel.service` levanta un *quick tunnel* de
  Cloudflare hacia `127.0.0.1:8000` (sin cuenta, sin abrir puertos). La URL
  `https://<aleatoria>.trycloudflare.com` cambia al reiniciar el túnel y no
  tiene garantía de uptime; para una URL fija hace falta un túnel nombrado
  con un dominio en Cloudflare (`cloudflared tunnel login`/`create`).
  Consultar la URL actual: `journalctl -u noise-tunnel | grep trycloudflare`.
- Red: `noise-net-watchdog.timer` comprueba cada 60 s gateway e internet;
  tras 3 fallos reconecta Wi-Fi (networkd + suplicante netplan) y lo deja en
  el journal (`journalctl -t noise-net-watchdog`). DNS de respaldo
  `1.1.1.1`/`8.8.8.8` configurado por drop-in de systemd-networkd.
- Logs: `journalctl -u noise-monitor -f`.

## Privacidad y uso responsable

El sistema graba video y audio de una vía pública. Los clips pueden contener
personas identificables, matrículas y fachadas de terceros.

- Los datos quedan **solo en el nodo**; el proyecto no envía nada a la nube.
- No publiques clips ni bases de datos en este repositorio ni en issues.
- Antes de desplegarlo, verifica la normativa local sobre videovigilancia,
  grabación de audio y tratamiento de datos personales.
- La finalidad declarada es medir ruido, no vigilar personas: conviene limitar
  la retención de clips y enfocar la cámara al espacio público.

## Documentación

- `1- … Visión y Arquitectura` — contexto, objetivos, KPIs y riesgos.
- `2- … Adquisición de Hardware` — equipamiento y justificación.
- `3- … Desarrollo de Software` — especificación técnica del pipeline.
- `6- compras realizadas` — costos reales.
- `9- plan-de-desarrollo` — checklist de fases y estado.
- `historicos/` — decisiones de hardware descartadas (Raspberry Pi, micrófono
  anterior) y por qué se cambiaron.

## Licencia

MIT — ver [`LICENSE`](LICENSE). Copyright (c) 2026 Roberto Michelena.

## Análisis acústico experimental

El analizador offline y el worker opcional de clips guardados se describen en la [guía de análisis acústico](docs/acoustic-analysis.md).
