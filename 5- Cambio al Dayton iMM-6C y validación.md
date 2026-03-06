## Cambio al Dayton iMM-6C y validación

El *iMM-6C con USB-C nativo* es objetivamente mejor para este proyecto:

### Ventajas concretas

1. *Eliminamos 3 componentes:* Tarjeta de sonido USB + cable divisor TRRS + adaptador. Menos puntos de falla, menos ruido eléctrico, menos cosas que comprar.

2. *Señal digital desde el origen:* El ADC está en el micrófono, no en un adaptador genérico. Mejor relación señal/ruido.

3. *Calibración individual:* Dayton incluye archivo de calibración específico para cada unidad, igual que el modelo clásico.

### Solución al problema de cableado (completando lo de Gemini)

El truco es *no usar cable USB-C largo*, sino convertir a USB-A para el tramo largo:

```
[iMM-6C] --USB-C-- [Adaptador USB-C hembra a USB-A macho] --USB-A extensión activa (5-10m)-- [PC]
```

*Lista de compra actualizada para audio:*
| Componente | Especificación | Precio aprox. |
|------------|----------------|---------------|
| Dayton iMM-6C | Micrófono USB-C calibrado | ~$30 |
| Adaptador USB-C a USB-A | Hembra C a macho A | ~$5 |
| Extensión USB-A activa | 5m con repetidor (si necesario) | ~$15 |
| Espuma antiviento | Tipo lavalier | ~$5 |

*Total audio: ~$55* (vs ~$50-60 de la cadena anterior, pero más simple y confiable)

### Ajuste en el handout de hardware

Reemplazar toda la sección de "Subsistema de Audio" por:

> *Micrófono:* Dayton Audio iMM-6C (USB-C nativo, omnidireccional, calibrado).  
> *Cableado:* Adaptador USB-C→USB-A + extensión USB-A activa si la distancia al balcón supera 2m.  
> *Protección:* Espuma antiviento para exteriores.

### Ajuste en software

Ninguno. El iMM-6C aparece como dispositivo de audio USB estándar (UAC). `sounddevice` lo detecta automáticamente:

```python
import sounddevice as sd
print(sd.query_devices())  # Verás "Dayton iMM-6C" o similar
```

Solo asegúrate de seleccionar el índice correcto en `config.yaml`:
```yaml
audio:
  device_index: 1  # Ajustar según query_devices()
```

