## Documento PRD – Sección 3: Handout de Desarrollo de Software
*Responsable:* Claude
*Versión:* 4.0 - Especificación Técnica (Mini-PC x86, cámara USB, iMM-6C, Python 3.12 + ai-edge-litert + YAMNet)

Este documento detalla la arquitectura de software, dependencias, estructuras de datos y lógica de implementación para el desarrollador encargado de construir el sistema.

---

### 0. Arquitectura de Datos del Sistema

```
             ┌───────────────────────────────────────────┐
             │   Lenovo ThinkCentre M910q                │
             │   Ubuntu Server 24.04 LTS                 │
             │                                           │
             │  ┌────────────────┐                       │
Camera ──────┼─▶│ /dev/video0    │                       │
 (USB 2.0)   │  │ OpenCV capture │                       │
             │  └──────┬─────────┘                       │
             │         │ frames                          │
             │         ▼                                 │
             │  ┌────────────────┐                       │
             │  │ Edge TPU       │◀── Coral (USB 3.0)    │
             │  │ ai-edge-litert │                       │
             │  └──────┬─────────┘                       │
             │         │ vehicle detections              │
             │         ├──────────────────┐              │
             │         │                  ▼              │
             │         │  ┌───────────────────────┐      │
             │         │  │ CentroidTracker       │      │
             │         │  │ (registro continuo)   │      │
             │         │  │ filtros: min_track_s,  │      │
             │         │  │   min_displacement    │      │
             │         │  └──────┬────────────────┘      │
             │         │         │ passage_queue          │
             │         │         ▼                       │
             │  ┌────────────────┐                       │
Mic ─────────┼─▶│ sounddevice    │                       │
 (iMM-6C)   │  │ A-weighting    │                       │
             │  └──────┬─────────┘                       │
             │         │ dB levels + trigger             │
             │         ▼                                 │
             │  ┌────────────────┐                       │
             │  │ YAMNet (CPU)   │                       │
             │  │ audio classif. │                       │
             │  │ 521 classes    │                       │
             │  └──────┬─────────┘                       │
             │         │ trigger_queue                   │
             │         ▼                                 │
             │  ┌────────────────────────────────┐       │
             │  │ Event Manager (2 colas)        │       │
             │  │ passage_queue → vehicle_passages│       │
             │  │ trigger_queue → noise_events   │──┐    │
             │  │ + correlación cruzada          │  │    │
             │  └──────┬─────────────────────────┘  │    │
             │         │                            │    │
             │    ┌────┴────┐                       │    │
             │    ▼         ▼                       │    │
             │  SQLite    clips/                    │    │
             │  events.db  *.mp4 ◀──────────────────┘    │
             │    │                                      │
             │    ▼                                      │
             │  ┌────────────────┐                       │
             │  │ FastAPI + HTMX │ ◀── Browser           │
             │  │ :8000          │    (red local)         │
             │  │ dashboard,     │                       │
             │  │ API REST,      │                       │
             │  │ reportes       │                       │
             │  └────────────────┘                       │
             └───────────────────────────────────────────┘
```

---

### 1. Entorno y Dependencias

*Sistema Operativo:*
```
Ubuntu Server 24.04 LTS (64-bit)
Sin interfaz gráfica (headless)
Hardware: Lenovo ThinkCentre M910q (i5-7500T, 8 GB RAM, 256 GB SSD)
```

*Dependencias Python (requirements.txt):*
```
# Visión (captura vía OpenCV)
opencv-python-headless>=4.9.0
numpy>=2.0.0

# Inferencia visual (Coral Edge TPU vía ai-edge-litert + feranick/libedgetpu)
ai-edge-litert>=2.1.0
# YAMNet usa el mismo runtime (ai-edge-litert) pero corre en CPU

# Clasificación de audio (YAMNet TFLite, corre en CPU)
resampy>=0.4.0

# Audio (Dayton iMM-6C vía USB, captura con sounddevice)
sounddevice>=0.5.0
scipy>=1.13.0

# Datos y Reportes
pandas>=3.0.0
matplotlib>=3.9.0

# Dashboard web
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
jinja2>=3.1.0
python-multipart>=0.0.9

# Utilidades
pyyaml>=6.0
```

**Nota sobre el Coral Edge TPU:** No se usa `pycoral` ni `tflite-runtime` (oficialmente solo soportan hasta Python 3.9). Se usa `ai-edge-litert` (sucesor de tflite-runtime) con `libedgetpu` del fork [feranick/libedgetpu](https://github.com/feranick/libedgetpu), que provee `.deb` pre-compilados para Ubuntu 24.04 compatibles con TensorFlow moderno. Ver Doc 7 sección 8 para instrucciones de instalación.

**Nota sobre elección de modelo de video:** Se usa SSD MobileNet V2 para validación inicial, con plan de migrar a EfficientDet-Lite0 (+4 puntos mAP, ~10ms vs ~3ms, ambos 100% Edge TPU). YOLO no es viable porque el Edge TPU compiler solo soporta un subconjunto de operaciones TFLite — los modelos YOLO compilan parcialmente y terminan corriendo mayormente en CPU, perdiendo la ventaja del acelerador.

**Nota sobre clasificación de audio (YAMNet):** Se usa YAMNet (Google), un modelo TFLite pre-entrenado en AudioSet (521 clases de sonido ambiental). Corre en CPU (~pocos ms por ventana de 0.975s a 16kHz). Las 521 clases se agrupan en categorías propias del proyecto: `vehicle`, `construction`, `animal`, `alarm`, `human`, `other`. No requiere entrenamiento adicional. Se requiere resampleo de 44.1kHz (captura) a 16kHz (YAMNet input), manejado con `resampy`. Modelos: `models/yamnet.tflite` + `models/yamnet_labels.txt`.

*Dependencias del Sistema (apt):*
```bash
sudo apt install -y libatlas-base-dev ffmpeg v4l-utils libportaudio2
# Para Coral Edge TPU (usar .deb de feranick, NO el repo de Google):
# Ver Doc 7, sección 8 para instrucciones detalladas.
```

---

### 2. Estructura del Proyecto

```
noise-monitor/
├── config.yaml              # Parámetros configurables
├── main.py                  # Script principal (hilos audio/video + web server)
├── calibrate.py             # Calibración inicial del micrófono
├── schema.sql               # Definición de base de datos
├── requirements.txt         # Dependencias Python
├── src/
│   ├── audio_monitor.py     # Clase para captura y análisis de audio
│   ├── audio_classifier.py  # Clasificación de sonido con YAMNet
│   ├── video_monitor.py     # Clase para captura y detección
│   ├── vehicle_tracker.py   # CentroidTracker: registro continuo de vehículos
│   ├── event_manager.py     # Correlación, guardado, consumo de 2 colas
│   ├── report_generator.py  # Generación de reportes (PDF/HTML)
│   └── utils.py             # Filtro A-weighting, helpers
├── web/
│   ├── app.py               # FastAPI application
│   ├── routes.py            # Endpoints (API + páginas)
│   ├── templates/           # Templates Jinja2 + HTMX
│   │   ├── base.html
│   │   ├── dashboard.html
│   │   ├── events.html
│   │   └── reports.html
│   └── static/              # CSS, favicon
├── scripts/
│   ├── test_all.py          # Test integrado de hardware
│   ├── test_camera.py
│   ├── test_audio.py
│   └── test_coral.py
├── models/
│   ├── ssd_mobilenet_v2_coco_edgetpu.tflite
│   ├── coco_labels.txt
│   ├── yamnet.tflite          # Clasificador de audio (521 clases AudioSet)
│   └── yamnet_labels.txt
├── data/
│   ├── events.db            # SQLite (tablas: vehicle_passages + noise_events)
│   └── clips/               # Videos de eventos
└── reports/                 # PDFs/HTMLs generados
```

---

### 3. Esquema de Base de Datos (schema.sql)

El sistema usa dos tablas: `vehicle_passages` registra **todos** los vehículos detectados por el tracker (independientemente del ruido), y `noise_events` registra los eventos de ruido con clip y clasificación YAMNet. Ambas se cruzan opcionalmente vía FK.

```sql
-- Registro continuo de vehículos detectados por el CentroidTracker
CREATE TABLE IF NOT EXISTS vehicle_passages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601 del momento en que el vehículo entró al cuadro
    timestamp_exit TEXT,               -- ISO 8601 cuando salió del cuadro
    vehicle_class TEXT NOT NULL,       -- 'car', 'truck', 'bus', 'motorcycle', 'bicycle'
    lane TEXT,                         -- 'left', 'right'
    detection_score REAL,              -- Score promedio del modelo durante el tracking
    db_at_time REAL,                   -- dB(A) al momento de la detección
    noise_event_id INTEGER,            -- FK a noise_events si este vehículo generó un evento ruidoso
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (noise_event_id) REFERENCES noise_events(id)
);

-- Eventos de ruido (vehiculares y no vehiculares) con clip de audio/video
CREATE TABLE IF NOT EXISTS noise_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601
    db_max REAL NOT NULL,              -- Nivel máximo ponderado A
    db_avg REAL,                       -- Nivel promedio en ventana
    trigger_type TEXT,                 -- 'peak', 'sustained'
    sound_class TEXT,                  -- Clase YAMNet raw: 'Vehicle', 'Dog', 'Jackhammer', etc.
    sound_category TEXT,               -- Categoría agrupada: 'vehicle','construction','animal','alarm','human','other'
    sound_confidence REAL,             -- Score de YAMNet (0.0-1.0)
    vehicle_class TEXT,                -- Solo si vehicular: 'motorcycle', 'car', 'truck', 'bus', 'unknown'
    lane TEXT,                         -- 'left', 'right', 'both', 'none'
    confidence TEXT,                   -- 'high', 'medium', 'low' (confianza en atribución visual)
    passage_id INTEGER,                -- FK a vehicle_passages si se correlacionó con un vehículo
    clip_path TEXT,                    -- Ruta relativa al clip
    duration_ms INTEGER,               -- Duración del clip en ms
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (passage_id) REFERENCES vehicle_passages(id)
);
```

**Nota sobre migración:** Si existía la tabla `events` (v1), `schema.sql` incluye un `INSERT OR IGNORE` para migrar datos a `noise_events`.

---

### 4. Archivo de Configuración (config.yaml)

```yaml
# Parámetros de audio
audio:
  device_index: null          # null = dispositivo por defecto
  sample_rate: 44100
  block_size: 2048            # ~46ms por bloque
  calibration_offset: 0.0     # dB a sumar tras calibración
  
# Umbrales de disparo
triggers:
  peak_db: 85                 # Disparo instantáneo
  sustained_db: 70            # Umbral de persistencia
  sustained_duration_ms: 1500 # Tiempo mínimo sobre umbral
  correlation_db: 65          # Umbral si hay vehículo visible

# Parámetros de video (cámara USB ELP vía OpenCV)
video:
  source: 0                     # Índice del dispositivo USB (o "/dev/video0")
  resolution: [1920, 1080]
  framerate: 30
  inference_interval: 5         # Procesar 1 de cada N frames
  detection_threshold: 0.5      # Confianza mínima del modelo

# Buffer circular
buffer:
  audio_seconds: 10
  video_seconds: 10
  pre_event_seconds: 5
  post_event_seconds: 5

# Estimación de velocidad
speed:
  enabled: true
  reference_distance_m: 10.0  # Distancia física entre líneas
  line1_y_pixel: 400          # Coordenada Y de línea 1
  line2_y_pixel: 680          # Coordenada Y de línea 2

# Rutas (SSD interno del Mini-PC)
paths:
  database: /home/roberto/noise-monitor/data/events.db
  clips: /home/roberto/noise-monitor/data/clips
  reports: /home/roberto/noise-monitor/reports

# Dashboard web
web:
  host: 0.0.0.0
  port: 8000

# Centroid tracker para registro continuo de vehículos
tracker:
  max_distance: 80          # px, distancia máxima entre centroides para match
  max_disappeared: 20       # inference frames sin ver antes de marcar como salido (~3.3s @ 6fps)
  min_track_seconds: 2.0    # duración mínima para contar como pasaje real (filtra flickering)
  min_displacement: 60      # px, desplazamiento mínimo del centroide para contar (filtra estacionados)

# Clasificador de audio (YAMNet)
audio_classifier:
  model_path: models/yamnet.tflite
  labels_path: models/yamnet_labels.txt
  min_score: 0.3                    # Score mínimo para aceptar clasificación
  category_map:                     # Mapeo de clases AudioSet → categoría propia
    vehicle:
      - Vehicle
      - Car
      - Truck
      - Motorcycle
      - Bus
      - Engine
      - Accelerating, revving, vroom
      - Tire squeal
      - Car horn, honking
    construction:
      - Jackhammer
      - Chainsaw
      - Power tool
      - Drill
      - Sawing
      - Hammer
    animal:
      - Dog
      - Bark
      - Cat
      - Bird
      - Rooster
    alarm:
      - Siren
      - Alarm
      - Emergency vehicle
      - Buzzer
      - Fire alarm
      - Smoke detector
    human:
      - Speech
      - Shout
      - Screaming
      - Music
      - Singing

# Jerarquía de ruido para inferencia visual
noise_priority:
  truck: 1
  bus: 1
  motorcycle: 2
  car: 3
  bicycle: 4
  person: 5
```

---

### 5. Lógica Principal (Pseudocódigo main.py)

```python
"""
Arquitectura multi-hilo con buffers circulares, tracking continuo de
vehículos (CentroidTracker) y clasificación de audio (YAMNet).
EventManager consume 2 colas: passage_queue y trigger_queue.
"""

import threading
from queue import Queue

# --- COMPONENTES ---
# AudioClassifier: carga YAMNet, clasifica ~1s de audio → (class, category, score)
# AudioMonitor: captura audio, A-weighting, dB, triggers → trigger_queue
#               Si trigger: clasifica con YAMNet antes de encolar
# VideoMonitor: captura video, inferencia Edge TPU cada N frames
#               CentroidTracker actualiza en cada inferencia → passage_queue
# EventManager: consume 2 colas en paralelo:
#   - passage_queue → INSERT en vehicle_passages (registro continuo)
#   - trigger_queue → INSERT en noise_events + guardar clip + correlacionar

# --- FLUJO DE DATOS ---
# 1. VideoMonitor captura frames y corre inferencia cada N frames
# 2. CentroidTracker recibe detecciones, rastrea centroides en px
# 3. Cuando un vehículo "sale" del cuadro (max_disappeared frames sin verlo):
#    - Si duration >= min_track_seconds Y displacement >= min_displacement:
#      → emite pasaje a passage_queue
#    - Si no: descartado (flickering o vehículo estacionado)
# 4. AudioMonitor detecta triggers (peak/sustained)
# 5. Si trigger: clasifica ~1s de audio con YAMNet → encola en trigger_queue
# 6. EventManager._passage_loop() → consume passage_queue → vehicle_passages
# 7. EventManager._trigger_loop() → consume trigger_queue → noise_events
#    - Si vehicular: busca passage reciente del mismo tipo para correlacionar
#    - Guarda clip ±5s de buffers de audio y video

# --- MAIN ---
if __name__ == '__main__':
    config = load_config('config.yaml')
    init_database()
    
    classifier = AudioClassifier(...)  # YAMNet (graceful fallback si no hay modelo)
    audio = AudioMonitor(config, audio_classifier=classifier)
    video = VideoMonitor(config)  # incluye CentroidTracker
    event_mgr = EventManager(audio, video, config)
    
    t_audio = threading.Thread(target=audio.run, daemon=True)
    t_video = threading.Thread(target=video.run, daemon=True)
    t_event = threading.Thread(target=event_mgr.run, daemon=True)
    # event_mgr.run() arranca internamente _passage_loop en thread separado
    
    for t in [t_audio, t_video, t_event]:
        t.start()
    
    # Mantener vivo hasta Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        running = False
```

---

### 6. Función de Filtro A-Weighting (utils.py)

```python
from scipy.signal import bilinear, lfilter

def design_a_weighting(fs):
    """
    Diseña filtro A-weighting según IEC 61672-1.
    Retorna coeficientes (b, a) para lfilter.
    """
    # Frecuencias de polo/cero del estándar
    f1, f2, f3, f4 = 20.6, 107.7, 737.9, 12194.0
    
    # Polos analógicos
    nums = [(2_np.pi_f4)*_2]
    dens = [1, 4_np.pi_f4, (2_np.pi_f4)__2]
    
    # ... (implementación completa de filtro de 4to orden)
    
    b, a = bilinear(nums, dens, fs)
    return b, a

# Pre-calcular coeficientes al inicio
A_WEIGHT_B, A_WEIGHT_A = design_a_weighting(SAMPLE_RATE)

def apply_a_weighting(signal):
    return lfilter(A_WEIGHT_B, A_WEIGHT_A, signal)
```

---

### 7. Lógica de Asignación de Culpable

```python
NOISE_PRIORITY = {
    'truck': 1, 'bus': 1,
    'motorcycle': 2,
    'car': 3, 'van': 3,
    'bicycle': 4, 'person': 5
}

def assign_culprit(detections):
    """
    Dado una lista de detecciones, retorna la clase
    más probable de ser la fuente del ruido.
    """
    if not detections:
        return 'unknown'
    
    # Ordenar por prioridad de ruido
    sorted_dets = sorted(
        detections,
        key=lambda d: NOISE_PRIORITY.get(d['class'], 99)
    )
    return sorted_dets[0]['class']

def calculate_confidence(detections):
    """
    Determina nivel de confianza en la atribución.
    """
    if len(detections) == 0:
        return 'low'
    if len(detections) == 1:
        return 'high'
    
    # Múltiples vehículos: verificar si hay jerarquía clara
    priorities = [NOISE_PRIORITY.get(d['class'], 99) for d in detections]
    if min(priorities) < max(priorities):
        return 'medium'  # Hay un candidato claro
    return 'low'  # Varios del mismo nivel
```

---

### 8. Clasificación de Audio (audio_classifier.py)

```python
import numpy as np
from ai_edge_litert.interpreter import Interpreter
import resampy

YAMNET_SAMPLE_RATE = 16000
YAMNET_EXPECTED_SAMPLES = 15600  # ~0.975s @ 16kHz — tamaño fijo del modelo

class AudioClassifier:
    """
    Clasifica sonidos usando YAMNet (521 clases AudioSet).
    Input: ventana de audio a cualquier sample rate.
    Output: (sound_class, sound_category, score).
    
    IMPORTANTE: YAMNet espera exactamente 15600 samples a 16kHz.
    El interpreter se pre-aloca con ese tamaño fijo en __init__
    para evitar errores de reshape con XNNPACK delegate.
    """
    def __init__(self, model_path, labels_path, category_map, min_score=0.3):
        self.interpreter = Interpreter(model_path=model_path)
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        self._input_index = input_details[0]["index"]
        self._output_index = output_details[0]["index"]
        # Pre-alocar con tamaño fijo (no redimensionar en cada classify)
        self.interpreter.resize_tensor_input(self._input_index, [YAMNET_EXPECTED_SAMPLES])
        self.interpreter.allocate_tensors()
        self.labels = self._load_labels(labels_path)
        self.category_map = category_map
        self.min_score = min_score
        self._build_reverse_map()

    def classify(self, audio_window, source_sr=44100):
        """
        Clasifica una ventana de audio.
        Resamplea a 16kHz y pad/trunca a 15600 samples.
        """
        if source_sr != YAMNET_SAMPLE_RATE:
            audio = resampy.resample(audio_window, source_sr, YAMNET_SAMPLE_RATE)
        else:
            audio = audio_window
        audio = audio.astype(np.float32)

        # Pad o truncar a exactamente 15600 samples
        if len(audio) > YAMNET_EXPECTED_SAMPLES:
            audio = audio[-YAMNET_EXPECTED_SAMPLES:]
        elif len(audio) < YAMNET_EXPECTED_SAMPLES:
            audio = np.pad(audio, (YAMNET_EXPECTED_SAMPLES - len(audio), 0))

        self.interpreter.set_tensor(self._input_index, audio)
        self.interpreter.invoke()

        scores = self.interpreter.get_tensor(self._output_index)
        # YAMNet puede retornar (patches, 521) — promediar patches
        if scores.ndim > 1:
            scores = scores.mean(axis=0)

        top_idx = np.argmax(scores)
        top_score = float(scores[top_idx])
        sound_class = self.labels[top_idx]

        if top_score < self.min_score:
            return ('unknown', 'other', top_score)

        category = self._lookup_category(sound_class)
        return (sound_class, category, top_score)
```

### 8b. Tracking Continuo de Vehículos (vehicle_tracker.py)

```python
class CentroidTracker:
    """
    Rastrea vehículos entre frames usando distancia euclidiana de centroides
    en coordenadas de píxel. Cada vez que un objeto "sale" del cuadro
    (no se ve durante max_disappeared frames de inferencia), se emite un
    pasaje en passage_queue.
    
    Filtros para evitar falsos positivos:
    - min_track_seconds: descarta tracks de duración menor (flickering del modelo)
    - min_displacement: descarta objetos cuyo centroide no se movió lo suficiente
      (vehículos estacionados que el modelo detecta de forma intermitente)
    """
    def __init__(self, max_distance=80.0, max_disappeared=20,
                 min_track_seconds=2.0, min_displacement=60.0,
                 passage_queue=None):
        # max_distance: px máximos entre centroides para considerar match
        # max_disappeared: frames de inferencia sin ver antes de deregistrar
        # min_track_seconds: duración mínima para emitir pasaje
        # min_displacement: px mínimos de desplazamiento start→end

    def update(self, detections, timestamp):
        """Recibe lista de detecciones del frame actual, actualiza tracking."""
        # Filtra solo clases vehiculares
        # Matching húngaro por distancia euclidiana de centroides
        # Registra nuevos, actualiza existentes, incrementa disappeared

    def _deregister(self, object_id):
        """Deregistra un objeto. Si pasó los filtros, emite pasaje a la cola."""
        # 1. Verificar duration >= min_track_seconds
        # 2. Verificar displacement >= min_displacement (parked filter)
        # 3. Si pasa ambos: passage_queue.put({...})
```

---

### 9. Script de Calibración (calibrate.py)

```python
"""
Uso: python calibrate.py --reference-db 75

1. Reproduce un tono de referencia con tu celular/parlante
2. Mide con sonómetro de referencia (ej: 75 dB)
3. El script mide lo que capta el Dayton
4. Calcula y guarda el offset en config.yaml
"""

import argparse
import sounddevice as sd
import numpy as np

def calibrate(reference_db, duration=10):
    print(f"Midiendo durante {duration} segundos...")
    print("Asegúrate de que el sonómetro de referencia marque {reference_db} dB")
    
    recording = sd.rec(int(duration _ 44100), samplerate=44100, channels=1)
    sd.wait()
    
    # Calcular dB medido (sin calibrar)
    rms = np.sqrt(np.mean(recording*_2))
    measured_db = 20 _ np.log10(rms / 20e-6)  # Referencia estándar
    
    offset = reference_db - measured_db
    
    print(f"dB medido (raw): {measured_db:.1f}")
    print(f"dB referencia: {reference_db}")
    print(f"Offset calculado: {offset:+.1f} dB")
    print(f"\nActualiza config.yaml con: calibration_offset: {offset:.1f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-db', type=float, required=True)
    args = parser.parse_args()
    calibrate(args.reference_db)
```

---

### 10. Notas de Implementación

1. *Threading vs Asyncio:* Se recomienda `threading` por simplicidad y porque `sounddevice` usa callbacks síncronos. `asyncio` añadiría complejidad sin beneficio real aquí.

2. *SSD interno:* El ThinkCentre M910q tiene SSD de 256 GB integrado. No se necesita montaje externo. Los datos se almacenan en `~/noise-monitor/data/`.

3. *CentroidTracker — coordenadas y filtros:* El tracker opera en **coordenadas de píxel** (no normalizadas 0-1). Las detecciones del modelo COCO se convierten a centroides en px antes de pasarlas al tracker. Dos filtros evitan falsos positivos: `min_track_seconds` (descarta flickering del modelo) y `min_displacement` (descarta vehículos estacionados cuyo centroide no se desplaza entre primera y última detección).

4. *YAMNet — tamaño de input fijo:* El modelo espera exactamente 15600 samples (0.975s @ 16kHz). El interpreter se pre-aloca una sola vez en `__init__` con ese tamaño. Cada ventana de audio se pad/trunca a 15600 samples antes de la inferencia. Esto evita errores con el delegate XNNPACK que no soporta reshape dinámico.

5. *Dashboard web (FastAPI + HTMX):* Se ejecuta como parte del proceso principal (`main.py` arranca uvicorn en un hilo). No requiere proxy reverso para uso en red local. Funcionalidades:
   - Nivel de dB en tiempo real (polling cada 1s vía HTMX)
   - Tabla de eventos recientes con reproducción de clips
   - Histogramas por hora/día (matplotlib renderizado server-side)
   - Generación de reportes PDF/HTML bajo demanda
   - API REST (`/api/events`, `/api/stats`, `/api/clips/{id}`) para integración externa

4. *Servicio systemd:* Ver Doc 8 (Deployment & Operations) para configuración completa del servicio, watchdog, crons y operación 24/7.

5. *Detección de dispositivos:* Antes de la primera ejecución, identificar índices de cámara y micrófono:
   ```bash
   v4l2-ctl --list-devices
   python3 -c "import sounddevice; print(sounddevice.query_devices())"
   ```
   Ajustar `video.source` y `audio.device_index` en `config.yaml` según los resultados.

6. *Pruebas iniciales:* Antes de desplegar en el balcón, probar en interior con videos de tráfico en un monitor para validar que la detección funciona.