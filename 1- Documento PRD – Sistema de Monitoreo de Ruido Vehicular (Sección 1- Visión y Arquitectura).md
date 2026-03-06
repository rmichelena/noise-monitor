### Documento PRD – Sistema de Monitoreo de Ruido Vehicular (Sección 1: Visión y Arquitectura)

*1. Contexto y propósito*  
La calle frente al departamento presenta un efecto “cañón urbano” que amplifica el ruido de tráfico. El proyecto busca generar evidencia cuantitativa correlacionando tipo de vehículo, nivel sonoro y momento del día para sustentar reclamos vecinales o legales. La solución debe ser autónoma, de bajo costo relativo y operable 24/7 en el balcón del tercer piso.

*2. Objetivos del sistema*  
- Capturar video y audio sincronizados para cada evento ruidoso.  
- Identificar la clase del vehículo responsable usando visión por computadora en el borde.  
- Registrar nivel sonoro (dB ponderación A) y, cuando sea posible, una estimación de velocidad.  
- Almacenar eventos en una base de datos local y generar reportes periódicos (CSV/PDF/HTML) con resúmenes semanales (“motos >80 dB”, histogramas horarios, etc.).  
- Producir clips de video cortos autocontenidos que puedan enviarse como evidencia (incluyendo un “Top 10” de picos semanales como opción).

*3. Alcance funcional*  
- *Detección continua*: audio monitoreado en tiempo real para disparar eventos (pico instantáneo o persistencia).  
- *Buffer circular*: guarda ±5 s de video y audio para capturar el momento exacto del paso del vehículo.  
- *Correlación audio-visión*: el trigger se confirma sólo si hay al menos un vehículo en cuadro; los eventos guardan clase, dB, carriles ocupados y nivel de confianza en la atribución.  
- *Reportes automáticos*: script semanal que genera estadísticas y, opcionalmente, compila los clips más ruidosos.  

*4. Arquitectura propuesta*  
- *Hardware base*: Mini-PC Lenovo ThinkCentre M910q (i5-7500T, 8 GB RAM, 256 GB SSD interno) con Coral USB Accelerator, cámara USB industrial ELP 1080p ≈160° y micrófono de medición Dayton iMM-6C (USB-C nativo, omnidireccional, calibrado individualmente).  
- *Software*: Ubuntu Server 22.04 LTS o Debian 12 (headless); captura de video con `opencv-python-headless` vía V4L2; inferencia con `tflite-runtime` + `pycoral` (modelo EfficientDet-Lite0 o SSD MobileNet V2 optimizado para Edge TPU); audio con `sounddevice` + `scipy` (filtro A-weighting); persistencia en SQLite; reportes con `pandas` + `matplotlib`.  
- *Flujo de datos*:  
  1. Audio y video se almacenan en buffers circulares independientes.  
  2. El hilo de audio aplica ponderación A y evalúa dos condiciones: (a) pico >85 dB, (b) persistencia >70 dB durante 1.5 s.  
  3. Al cumplirse una condición, se marca un “pre-evento”; el hilo de video consulta detecciones recientes (1 de cada 5 frames) y confirma si existe un vehículo en las regiones izquierda o derecha.  
  4. Si hay coincidencia, se vuelca el clip ±5 s, se calcula la clase “culpable” mediante reglas de prioridad (camión/bus > moto > auto) y se inserta el registro en SQLite (`timestamp, clase, dB, carril, confianza, ruta_clip`).  
  5. Un cron semanal ejecuta el generador de reportes y opcionalmente compone un MP4 con los 10 eventos más ruidosos usando `ffmpeg`.

*5. KPIs y entregables*  
- *KPIs operativos*: % de eventos con clase asignada, precisión subjetiva en 20 revisiones manuales (>90%), disponibilidad >95%, retención de datos ≥30 días.  
- *Entregables iniciales*: hardware configurado, scripts principales (`main.py`, `calibrate.py`, report scheduler), manual de calibración y guía de operación.

*6. Caveats y riesgos*  
- *Calibración acústica*: requiere comparar contra un sonómetro confiable al inicio; sin este paso los dB serán relativos.  
- *Ambigüedad de origen del ruido*: al usar un solo micrófono omnidireccional se resuelve por heurística visual; los eventos con múltiples vehículos similares se marcarán como “confianza baja”.  
- *Condiciones ambientales*: viento fuerte, lluvia o vibraciones del balcón pueden introducir ruido en audio/video; se recomienda protección física y filtros software básicos.  
- *Disponibilidad del hardware*: mini-PCs refurbished pueden variar en stock; el Coral USB Accelerator tiene disponibilidad intermitente.  
- *Escalabilidad*: la solución está pensada para un solo balcón. Extender a múltiples puntos requerirá sincronización entre nodos y backend agregado.  
