### Documento PRD – Sistema de Monitoreo de Ruido Urbano (Sección 1: Visión y Arquitectura)

*1. Contexto y propósito*
La calle frente al departamento presenta un efecto "cañón urbano" que amplifica el ruido de tráfico y otras fuentes urbanas (construcción, animales, alarmas). El proyecto busca generar evidencia cuantitativa correlacionando tipo de fuente sonora, nivel sonoro y momento del día para sustentar reclamos vecinales o legales. La solución debe ser autónoma, de bajo costo relativo y operable 24/7 en el balcón del tercer piso.

*2. Objetivos del sistema*
- Capturar video y audio sincronizados para cada evento ruidoso.
- Clasificar el tipo de sonido (vehículo, construcción, animal, alarma, etc.) usando un modelo de clasificación de audio (YAMNet).
- Cuando el sonido es vehicular, identificar la clase del vehículo responsable usando visión por computadora en el borde.
- Registrar nivel sonoro (dB ponderación A) y, cuando sea posible, una estimación de velocidad.
- Almacenar eventos en una base de datos local y generar reportes periódicos (CSV/PDF/HTML) con resúmenes semanales (histogramas horarios, desglose por categoría de sonido, etc.).
- Producir clips de video/audio cortos autocontenidos que puedan enviarse como evidencia (incluyendo un "Top 10" de picos semanales como opción).
- Proveer un dashboard web en red local para monitoreo en tiempo real (nivel de dB actual, eventos recientes, clips, histogramas) y generación de reportes bajo demanda.

*3. Alcance funcional*
- *Detección continua*: audio monitoreado en tiempo real para disparar eventos (pico instantáneo o persistencia).
- *Buffer circular*: guarda ±5 s de video y audio para capturar el momento exacto del evento.
- *Clasificación de sonido*: cada trigger pasa por YAMNet (521 clases AudioSet, modelo TFLite en CPU) que identifica el tipo de fuente. Las clases se agrupan en categorías: vehículo, construcción, animal, alarma/sirena, otro.
- *Correlación audio-visión*: si YAMNet indica sonido vehicular, se confirma con detección visual (al menos un vehículo en cuadro). Sonidos no vehiculares se registran directamente sin correlación de video.
- *Reportes automáticos*: script semanal que genera estadísticas y, opcionalmente, compila los clips más ruidosos.
- *Dashboard web*: interfaz accesible desde la red local (`http://10.10.10.137:8000`) con FastAPI + HTMX. Muestra nivel de dB en vivo, tabla de eventos recientes con clips reproducibles, histogramas por hora/día, y generación de reportes on-demand.

*4. Arquitectura propuesta*
- *Hardware base*: Mini-PC Lenovo ThinkCentre M910q (i5-7500T, 8 GB RAM, 256 GB SSD interno) con Coral USB Accelerator, cámara USB industrial ELP 1080p ~160° y micrófono de medición Dayton iMM-6C (USB-C nativo, omnidireccional, calibrado individualmente).
- *Software*: Ubuntu Server 24.04 LTS (headless); Python 3.12; captura de video con `opencv-python-headless` vía V4L2; clasificación de sonido con YAMNet (TFLite en CPU); detección de vehículos con `ai-edge-litert` + `libedgetpu` de feranick (modelo SSD MobileNet V2 / EfficientDet-Lite0 en Edge TPU); audio con `sounddevice` + `scipy` (filtro A-weighting); persistencia en SQLite; dashboard web con `FastAPI` + `HTMX`; reportes con `pandas` + `matplotlib`.
- *Flujo de datos*:
  1. Audio y video se almacenan en buffers circulares independientes.
  2. El hilo de audio aplica ponderación A y evalúa dos condiciones: (a) pico >85 dB, (b) persistencia >70 dB durante 1.5 s.
  3. Al cumplirse una condición, se clasifica la ventana de audio con YAMNet → categoría (vehículo, construcción, animal, alarma, otro).
  4a. Si categoría = vehículo: el hilo de video consulta detecciones recientes y confirma si hay un vehículo en cuadro. Se calcula la clase "culpable" (camión/bus > moto > auto) y se registra con correlación visual.
  4b. Si categoría ≠ vehículo: se registra el evento directamente con la clase de sonido de YAMNet, sin correlación de video.
  5. En ambos casos se vuelca el clip ±5 s y se inserta en SQLite (timestamp, sound_class, sound_category, vehicle_class, dB, carril, confianza, ruta_clip).
  6. Un cron semanal ejecuta el generador de reportes y opcionalmente compone un MP4 con los 10 eventos más ruidosos usando `ffmpeg`.

*5. KPIs y entregables*
- *KPIs operativos*: % de eventos con clase asignada, precisión subjetiva en 20 revisiones manuales (>90%), disponibilidad >95%, retención de datos ≥30 días.
- *Entregables iniciales*: hardware configurado, scripts principales (`main.py`, `calibrate.py`, report scheduler), manual de calibración y guía de operación.

*6. Caveats y riesgos*
- *Calibración acústica*: requiere comparar contra un sonómetro confiable al inicio; sin este paso los dB serán relativos.
- *Ambigüedad de origen del ruido*: al usar un solo micrófono omnidireccional se resuelve por heurística visual para vehículos; los eventos con múltiples fuentes se marcarán como "confianza baja".
- *Precisión de YAMNet*: es un modelo general (521 clases). Puede confundir sonidos similares (e.g., sierra eléctrica vs moto). Las categorías agrupadas mitigan esto, y se puede refinar con fine-tuning futuro.
- *Condiciones ambientales*: viento fuerte, lluvia o vibraciones del balcón pueden introducir ruido en audio/video; se recomienda protección física y filtros software básicos.
- *Disponibilidad del hardware*: mini-PCs refurbished pueden variar en stock; el Coral USB Accelerator tiene disponibilidad intermitente.
- *Escalabilidad*: la solución está pensada para un solo balcón. Extender a múltiples puntos requerirá sincronización entre nodos y backend agregado.
