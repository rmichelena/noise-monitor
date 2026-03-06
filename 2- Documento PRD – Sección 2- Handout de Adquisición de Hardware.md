### Documento PRD – Sección 2: Handout de Adquisición de Hardware
*Responsable:* Gemini
*Versión:* 2.0 - Hardware Adquirido (actualizado tras migración a Mini-PC e iMM-6C)

Este listado desglosa los componentes físicos del "Nodo de Monitoreo Acústico Vehicular". Se migró de Raspberry Pi a Mini-PC x86 por mejor relación precio/rendimiento, y se simplificó la cadena de audio con el Dayton iMM-6C USB-C nativo.

#### 1. Núcleo de Procesamiento (The Brains)
_   *Item:* Lenovo ThinkCentre M910q (refurbished).
    _   *Especificación:* Intel Core i5-7500T, 8 GB RAM, 256 GB SSD interno.
    _   *Ventajas sobre Raspberry Pi:* CPU x86-64 con AVX2, SSD nativo (sin adaptadores), fuente de alimentación incluida, mayor estabilidad térmica.
    _   *Sistema operativo:* Ubuntu Server 22.04 LTS o Debian 12 (headless, sin interfaz gráfica).
    _   *Nota:* El SSD interno es suficiente para OS, base de datos y clips de video. No se necesita almacenamiento externo adicional inicialmente; si la retención de datos supera la capacidad, se puede agregar un SSD USB 3.0.

#### 2. Aceleración de IA (The Muscle)
_   *Item:* Google Coral USB Accelerator.
    _   *Estado:* *Ya adquirido.*
    _   *Requisito:* Conectar a un puerto *USB 3.0* para máximo ancho de banda.
    _   *Nota:* En x86, el Coral sigue siendo útil para liberar CPU, pero ya no es estrictamente necesario: el i5-7500T puede correr EfficientDet-Lite0 a ~15-20 FPS por software.

#### 3. Subsistema de Visión (The Eyes)
_   *Cámara:* ELP-USBFHD08S-L29 (cámara USB industrial, 1080p, ≈160° FOV).
    _   *Estado:* *Ya adquirida.*
    _   *Interfaz:* USB 2.0 con cable de 1m incluido. Compatible UVC (plug & play en Linux).
    _   *Detección del dispositivo:*
    ```bash
    v4l2-ctl --list-devices
    ```
    _   *Nota:* Si la distancia al balcón supera 3m, usar una extensión USB-A activa.

#### 4. Subsistema de Audio (The Ears)
Cadena simplificada: micrófono USB-C directo, sin tarjeta de sonido ni cable divisor.
_   *Micrófono:* Dayton Audio iMM-6C (USB-C nativo, omnidireccional, calibrado individualmente).
    _   *Estado:* *Ya adquirido.*
    _   *Ventaja:* ADC integrado en el micrófono → mejor relación señal/ruido, menos puntos de falla.
    _   *Calibración:* Dayton incluye archivo de calibración específico por unidad.
_   *Cableado:* Adaptador USB-C hembra → USB-A macho + cable USB-A activo para el tramo largo al balcón.
    _   *Estado:* Cable USB-A activo *ya adquirido.*
_   *Protección:* Espuma antiviento (windscreen) tipo lavalier para exteriores.

#### 5. Montaje y Protección (The Shell)
_   *Caja estanca:* Caja de conexiones eléctricas IP65 (aprox 20x15x10 cm) para alojar el mini-PC o al menos proteger las conexiones exteriores (cámara, micrófono).
_   *Sujeción:* Abrazaderas plásticas UV o soporte de brazo articulado para cámara de seguridad.
_   *Misceláneos:*
    _   Velcro adhesivo para fijar componentes dentro de la caja.
    *   Silicona o cinta aislante para sellar salidas de cables.

#### 6. Resumen de costos

| Componente | Especificación | Estado |
|------------|----------------|--------|
| Mini-PC | Lenovo M910q i5-7500T, 8 GB, 256 GB SSD | Comprado |
| Cámara USB | ELP-USBFHD08S-L29, 1080p, ≈160° | Comprada |
| Coral USB Accelerator | Edge TPU | Ya se tenía |
| Micrófono | Dayton iMM-6C (USB-C) | Comprado |
| Cable USB-A activo | Para tramo largo del micrófono | Comprado |
| Caja estanca + montaje | IP65, abrazaderas, velcro, sellado | Pendiente |
| Espuma antiviento | Tipo lavalier | Pendiente |
