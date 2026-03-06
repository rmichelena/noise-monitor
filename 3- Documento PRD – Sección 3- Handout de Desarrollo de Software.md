## Documento PRD – Sección 3: Handout de Desarrollo de Software
*Responsable:* Claude
*Versión:* 2.0 - Especificación Técnica (actualizada para Mini-PC x86 + cámara USB + iMM-6C)

Este documento detalla la arquitectura de software, dependencias, estructuras de datos y lógica de implementación para el desarrollador encargado de construir el sistema.

---

### 1. Entorno y Dependencias

*Sistema Operativo:*
```
Ubuntu Server 22.04 LTS o Debian 12 (64-bit)
Sin interfaz gráfica (headless)
Hardware: Lenovo ThinkCentre M910q (i5-7500T, 8 GB RAM, 256 GB SSD)
```

*Dependencias Python (requirements.txt):*
```
# Visión (captura vía OpenCV + inferencia Coral)
opencv-python-headless>=4.8.0
tflite-runtime>=2.14.0
pycoral>=2.0.0
numpy>=1.24.0

# Audio (Dayton iMM-6C vía USB, captura con sounddevice)
sounddevice>=0.4.6
scipy>=1.11.0

# Datos y Reportes
pandas>=2.0.0
matplotlib>=3.7.0

# Utilidades
pyyaml>=6.0
```

*Dependencias del Sistema (apt):*
```bash
sudo apt install -y libatlas-base-dev ffmpeg v4l-utils
# Para Coral Edge TPU:
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update && sudo apt install -y libedgetpu1-std python3-pycoral
```

---

### 2. Estructura del Proyecto

```
noise-monitor/
├── config.yaml           # Parámetros configurables
├── main.py               # Script principal (hilos audio/video)
├── calibrate.py          # Calibración inicial del micrófono
├── report.py             # Generador de reportes semanales
├── schema.sql            # Definición de base de datos
├── src/
│   ├── audio_monitor.py  # Clase para captura y análisis de audio
│   ├── video_monitor.py  # Clase para captura y detección
│   ├── event_manager.py  # Lógica de correlación y guardado
│   └── utils.py          # Filtro A-weighting, helpers
├── models/
│   └── efficientdet_lite0_edgetpu.tflite
├── data/
│   ├── events.db         # SQLite (en SSD interno)
│   └── clips/            # Videos de eventos (en SSD interno)
└── reports/              # PDFs/HTMLs generados
```

---

### 3. Esquema de Base de Datos (schema.sql)

```sql
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,           -- ISO 8601: 2026-02-08T14:30:45
    db_max REAL NOT NULL,              -- Nivel máximo ponderado A
    db_avg REAL,                       -- Nivel promedio en ventana
    vehicle_class TEXT,                -- 'motorcycle', 'car', 'truck', 'bus', 'unknown'
    lane TEXT,                         -- 'left', 'right', 'both', 'none'
    confidence TEXT,                   -- 'high', 'medium', 'low'
    speed_estimate REAL,               -- km/h estimados (nullable)
    clip_path TEXT,                    -- Ruta relativa al video
    duration_ms INTEGER,               -- Duración del clip en ms
    trigger_type TEXT,                 -- 'peak', 'sustained', 'correlation'
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_timestamp ON events(timestamp);
CREATE INDEX idx_vehicle ON events(vehicle_class);
CREATE INDEX idx_db ON events(db_max);
```

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
  database: /opt/noise-monitor/data/events.db
  clips: /opt/noise-monitor/data/clips
  reports: /opt/noise-monitor/reports

# Jerarquía de ruido para inferencia
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
Arquitectura de dos hilos con buffers circulares compartidos.
"""

import threading
from collections import deque
from queue import Queue

# Buffers compartidos (thread-safe con locks)
audio_buffer = deque(maxlen=AUDIO_BUFFER_SIZE)
video_buffer = deque(maxlen=VIDEO_BUFFER_SIZE)
event_queue = Queue()  # Comunicación entre hilos

# --- HILO DE AUDIO ---
def audio_thread():
    """
    1. Captura continua con sounddevice
    2. Aplica filtro A-weighting (scipy.signal)
    3. Calcula RMS -> dB
    4. Evalúa condiciones de trigger
    5. Si trigger, encola pre-evento
    """
    sustained_counter = 0
    
    def audio_callback(indata, frames, time, status):
        nonlocal sustained_counter
        
        # Filtro A-weighting
        filtered = apply_a_weighting(indata)
        
        # Calcular dB
        rms = np.sqrt(np.mean(filtered*_2))
        db = 20 _ np.log10(rms / REF_PRESSURE) + CALIBRATION_OFFSET
        
        # Guardar en buffer con timestamp
        audio_buffer.append((time.inputBufferAdcTime, db, indata.copy()))
        
        # Evaluar triggers
        if db >= PEAK_THRESHOLD:
            event_queue.put(('peak', db, time.inputBufferAdcTime))
            sustained_counter = 0
        elif db >= SUSTAINED_THRESHOLD:
            sustained_counter += 1
            if sustained_counter >= SUSTAINED_BLOCKS:
                event_queue.put(('sustained', db, time.inputBufferAdcTime))
                sustained_counter = 0
        else:
            sustained_counter = max(0, sustained_counter - 1)
    
    with sd.InputStream(callback=audio_callback, ...):
        while running:
            time.sleep(0.1)

# --- HILO DE VIDEO ---
def video_thread():
    """
    1. Captura continua con OpenCV (cámara USB vía V4L2)
    2. Cada N frames, ejecuta inferencia con Coral
    3. Mantiene buffer circular de frames + detecciones
    4. Escucha event_queue para confirmar y guardar eventos
    """
    import cv2
    from pycoral.utils.edgetpu import make_interpreter
    
    cap = cv2.VideoCapture(config['video']['source'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config['video']['resolution'][0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config['video']['resolution'][1])
    cap.set(cv2.CAP_PROP_FPS, config['video']['framerate'])
    
    interpreter = make_interpreter('models/efficientdet_lite0_edgetpu.tflite')
    interpreter.allocate_tensors()
    
    frame_count = 0
    recent_detections = []
    
    while running:
        ret, frame = cap.read()
        if not ret:
            continue
        timestamp = time.time()
        
        # Guardar en buffer
        video_buffer.append((timestamp, frame))
        
        # Inferencia cada N frames
        frame_count += 1
        if frame_count % INFERENCE_INTERVAL == 0:
            detections = run_inference(interpreter, frame)
            recent_detections = detections
        
        # Verificar si hay evento pendiente
        if not event_queue.empty():
            trigger_type, db, audio_ts = event_queue.get()
            
            # Verificar correlación visual
            vehicles = [d for d in recent_detections if d['class'] in VEHICLE_CLASSES]
            
            if vehicles or db >= PEAK_THRESHOLD:
                # Determinar culpable por jerarquía
                culprit = assign_culprit(vehicles)
                confidence = calculate_confidence(vehicles)
                
                # Guardar clip
                save_event(
                    audio_buffer, video_buffer,
                    timestamp=audio_ts,
                    db_max=db,
                    vehicle=culprit,
                    confidence=confidence,
                    trigger_type=trigger_type
                )

# --- MAIN ---
if __name__ == '__main__':
    load_config('config.yaml')
    init_database()
    
    t_audio = threading.Thread(target=audio_thread, daemon=True)
    t_video = threading.Thread(target=video_thread, daemon=True)
    
    t_audio.start()
    t_video.start()
    
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

### 8. Script de Calibración (calibrate.py)

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

### 9. Notas de Implementación

1. *Threading vs Asyncio:* Se recomienda `threading` por simplicidad y porque `sounddevice` usa callbacks síncronos. `asyncio` añadiría complejidad sin beneficio real aquí.

2. *SSD interno:* El ThinkCentre M910q tiene SSD de 256 GB integrado. No se necesita montaje externo. Los datos se almacenan en `/opt/noise-monitor/data/`.

3. *Servicio systemd:* Para ejecución automática al boot:
   ```ini
   # /etc/systemd/system/noise-monitor.service
   [Unit]
   Description=Noise Monitor
   After=multi-user.target
   
   [Service]
   Type=simple
   User=noise
   WorkingDirectory=/opt/noise-monitor
   ExecStart=/usr/bin/python3 main.py
   Restart=always
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   ```

4. *Cron para reportes:*
   ```bash
   # crontab -e
   0 6 * * 1 /usr/bin/python3 /opt/noise-monitor/report.py
   ```

5. *Detección de dispositivos:* Antes de la primera ejecución, identificar índices de cámara y micrófono:
   ```bash
   v4l2-ctl --list-devices          # Cámara USB
   python3 -c "import sounddevice; print(sounddevice.query_devices())"  # Micrófono
   ```
   Ajustar `video.source` y `audio.device_index` en `config.yaml` según los resultados.

6. *Pruebas iniciales:* Antes de desplegar en el balcón, probar en interior con videos de tráfico en un monitor para validar que la detección funciona.

---

*Fin del documento PRD completo.*