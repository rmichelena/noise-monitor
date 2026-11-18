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

- Captura continua de audio con ponderación A (IEC 61672-1) y cálculo de dB SPL.
- Captura de video e inferencia de objetos sobre Coral Edge TPU (`ai-edge-litert`).
- `CentroidTracker` para registro continuo de pasajes, con filtros de duración
  mínima y desplazamiento mínimo (descarta parpadeos del modelo y vehículos
  estacionados).
- Triggers por pico instantáneo o por persistencia sobre umbral.
- Clasificación de audio con YAMNet (521 clases AudioSet) agrupadas en
  categorías: vehículo, construcción, animal, alarma, humano, otro.
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

## Scripts

| Script | Uso |
|---|---|
| `scripts/download_yamnet.sh` | Descarga el modelo y las etiquetas de YAMNet |
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
  contadores del día y los últimos eventos con enlace a su clip.
- Retención: `scripts/purge.py` borra clips de más de 30 días (la fila del
  evento se conserva sin clip) y `audio_levels` de más de 400 días.
  `noise-purge.timer` lo ejecuta los domingos a las 04:00. Configurable en
  `config.yaml` → `retention`.
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
