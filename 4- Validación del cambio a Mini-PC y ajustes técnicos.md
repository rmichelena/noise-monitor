## Validación del cambio a Mini-PC y ajustes técnicos

Roberto, coincido completamente con ChatGPT y Gemini: el ThinkCentre (especialmente el M910q con i5-6500T) es objetivamente mejor opción que la Raspberry Pi al mismo precio. Ganas CPU x86-64 con AVX, SSD nativo, fuente incluida y menos dolores de cabeza con disponibilidad de stock.

### Respuesta a la pregunta de Gemini sobre cambios en software

*Sí, exactamente.* Al migrar a x86 + cámara USB, los cambios son:

*1. requirements.txt actualizado:*
```
# Visión (CAMBIO)
opencv-python-headless>=4.8.0    # Mantiene
tflite-runtime>=2.14.0          # Mantiene (funciona en x86)
pycoral>=2.0.0                  # Mantiene (Coral USB compatible)
numpy>=1.24.0                   # Mantiene

# ELIMINAR: picamera2 (ya no aplica)

# Audio (SIN CAMBIOS)
sounddevice>=0.4.6
scipy>=1.11.0

# Datos y Reportes (SIN CAMBIOS)
pandas>=2.0.0
matplotlib>=3.7.0
pyyaml>=6.0
```

*2. Inicialización de video en main.py:*

```python
# ANTES (Raspberry Pi):
from picamera2 import Picamera2
picam = Picamera2()
picam.configure(picam.create_video_configuration(...))
picam.start()
frame = picam.capture_array()

# DESPUÉS (Mini-PC con USB):
import cv2
cap = cv2.VideoCapture(0)  # Índice del dispositivo USB
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

# En el loop:
ret, frame = cap.read()
if not ret:
    continue  # Manejar error de captura
# frame ya es numpy array BGR, listo para inferencia
```

*3. Ajuste en config.yaml:*
```yaml
video:
  source: 0                    # Índice de dispositivo USB (o "/dev/video0")
  resolution: [1920, 1080]
  framerate: 30
  # ... resto igual
```

---

### Consideraciones adicionales para x86

1. *Rendimiento:* El i5-6500T puede correr inferencia TFLite en CPU a velocidad aceptable (~15-20 FPS con EfficientDet-Lite0). El Coral USB sigue siendo útil para liberar CPU para otras tareas, pero ya no es estrictamente necesario. Podrías incluso probar modelos más grandes si lo deseas.

2. *Drivers Coral en x86:* Instalar con:
   ```bash
   echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
   sudo apt update && sudo apt install libedgetpu1-std python3-pycoral
   ```

3. *Detección del dispositivo de video:* Para identificar la cámara USB:
   ```bash
   v4l2-ctl --list-devices
   ```
   Esto te mostrará `/dev/video0`, `/dev/video1`, etc. Usa el índice correcto en `cv2.VideoCapture()`.

4. *Latencia USB:* Con cables activos de hasta 10m no deberías tener problemas. Para la cámara industrial ELP que sugiere Gemini, verifica que soporte UVC (prácticamente todas lo hacen).

---

### Actualización del Handout de Hardware (Resumen)

| Componente | Especificación | Precio aprox. |
|------------|----------------|---------------|
| Mini-PC | Lenovo M910q i5-6500T, 8GB, 256GB SSD | ~$125-135 |
| Cámara USB | Industrial 1080p 120° (ELP o similar) | $30-40 |
| Coral USB Accelerator | (ya lo tienes) | $0 |
| Micrófono | Dayton iMM-6 + adaptador TRRS + tarjeta USB | $25-35 |
| Extensión USB activa | Solo si distancia >3m al balcón | $15-20 |
| *TOTAL* | | *~$195-230* |

Esto es comparable o ligeramente superior al presupuesto original con Pi, pero con hardware significativamente más capaz y confiable.

