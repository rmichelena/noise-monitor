# Prueba CPU vs Coral de YAMNet en el ThinkCentre

El backend de YAMNet se aplica al clasificador que ya corre al grabar eventos y pasajes. El disparador dBA o psicoacústico sigue usando su configuración. El modelo oficial para Coral conserva la entrada mono de 15.600 muestras a 16 kHz y las 521 clases. El remuestreo y la preparación del espectrograma aún consumen CPU. [Ejemplo oficial de TensorFlow para Coral USB](https://github.com/tensorflow/examples/blob/master/lite/examples/sound_classification/raspberry_pi/README.md#speed-up-the-inference-time-optional)

## Preparación para el agente residente

1. Trabaja sobre la PR #9 o el commit de `main` que la incorpore. Anota el commit y usa el mismo usuario y entorno Python del servicio. Mantén las pruebas, clips y copias de configuración en el ThinkCentre.
2. Guarda una copia privada de la configuración operativa y anota el estado actual del servicio. Ejecuta `systemctl cat noise-monitor` para confirmar el directorio y el intérprete. Los ejemplos usan el `venv/bin/python` del servicio.
3. Verifica que el detector de video elegido funcione con el Coral. Usa `lsusb`, el journal de `noise-monitor` y el campo `coral_ok` de `/api/live`. Conserva el mismo detector, modelo de video, resolución, criterio de disparo, umbrales y ajustes de grabación durante la comparación.
4. Descarga los dos modelos y genera un único mapa de etiquetas. El instalador comprueba la huella SHA-256 del asset oficial para Coral.

~~~bash
testdir="$(mktemp -d)"
cp -p config.yaml "$testdir/config.original.yaml"
git rev-parse HEAD > "$testdir/commit.txt"
bash scripts/download_yamnet.sh --coral
~~~

La configuración antigua sigue eligiendo CPU. El cambio para el piloto consiste en añadir estas claves bajo la sección `audio_classifier` existente.

~~~yaml
audio_classifier:
  backend: coral
  model_path: models/yamnet.tflite
  coral_model_path: models/yamnet_edgetpu.tflite
  labels_path: models/yamnet_labels.txt
  fallback_to_cpu: true
~~~

Conserva el `min_score` y el `category_map` existentes. Con `fallback_to_cpu: true`, si la carga o el delegado de Coral devuelve un error, YAMNet pasa a CPU hasta el siguiente reinicio. El journal registra la causa. `/api/health` publica `audio_classifier.requested_backend`, `active_backend`, `fallbacks`, `inference_count` e `inference_ms_total`. El tiempo de inferencia incluye la ejecución del modelo y una eventual recuperación. El remuestreo y el ranking se miden en el benchmark de clips. Con `fallback_to_cpu: false`, un fallo de inicialización impide activar el backend pedido.

## 1. Mismos clips con ambos backends

Haz el ensayo en una ventana de mantenimiento con el servicio detenido para aislar YAMNet y evitar que dos procesos compitan por el dispositivo. Ejecuta el benchmark con el usuario del servicio, para que también pruebe acceso al Coral. Incluye clips cercanos al umbral, motores, bocinas, tráfico mezclado, vajilla y habla audible sola y mezclada. Usa datos con permiso de revisión. Las anotaciones de voz necesitan clips que conserven audio donde se sepa si había habla.

Crea un JSON como este en el directorio privado. Las rutas relativas parten de ese archivo. `trigger_seconds` es la posición del disparo dentro del audio. `expected_labels` usa nombres literales del mapa de YAMNet y permite varias etiquetas aceptables. Omite `speech` para clips sin etiqueta fiable. En dBA, anota habla en el entorno del disparo. Para el modo psicoacústico, anota habla en todo el clip.

~~~json
[
  {
    "id": "bocina-01",
    "clip": "clips/bocina-01.wav",
    "trigger_seconds": 2.45,
    "expected_labels": ["Vehicle horn, car horn, honking"],
    "speech": false
  },
  {
    "id": "trafico-con-voz-01",
    "clip": "clips/trafico-con-voz-01.wav",
    "trigger_seconds": 3.0,
    "speech": true
  }
]
~~~

~~~bash
sudo systemctl stop noise-monitor
benchmark_status=0
venv/bin/python scripts/benchmark_yamnet.py "$testdir/manifest.json" \
  --config config.yaml --backend both --repeat 5 > "$testdir/benchmark.json" || benchmark_status=$?
sudo systemctl start noise-monitor
printf 'Benchmark exit code: %s\n' "$benchmark_status"
~~~

Se aceptan WAV y MP4 mono con pista de audio. El script usa el método de clasificación de eventos del monitor, las mismas ventanas y el mismo umbral para los dos backends. `--mode dba` o `--mode psychoacoustic` permite repetir con uno u otro alcance de voz. El script exige el backend pedido y devuelve error si no carga Coral. El informe incluye tiempos de pared y CPU del proceso (mediana y p95), concordancia de etiqueta principal, aciertos entre las tres primeras, `unknown`, voces omitidas y casos en que solo CPU habría silenciado.

## 2. Corriendo junto al detector de video

Vuelve a arrancar el servicio y mide **CPU → Coral → CPU**, con ventanas de observación comparables de al menos cinco minutos y actividad acústica en las tres. Usa primero `backend: cpu`, luego `backend: coral`, y vuelve a CPU. Cambia únicamente ese campo entre corridas y reinicia con `sudo systemctl restart noise-monitor`. Espera a que el servicio y el video estén listos antes de medir. Si la actividad difiere mucho, repite el ensayo. El Coral puede perder rendimiento al alternar entre dos modelos. [Consideraciones de rendimiento de Coral](https://coral.ai/docs/edgetpu/multiple-edgetpu/#performance-considerations)

Adapta `label` por corrida y `base_url` si el servicio usa otro puerto. Captura contadores al principio y al final del mismo proceso.

~~~bash
label=cpu-1
base_url=http://127.0.0.1:8000
since="$(date --iso-8601=seconds)"
pid="$(systemctl show -p MainPID --value noise-monitor)"
test "$pid" -gt 0 || { echo 'El servicio debe estar activo' >&2; exit 1; }
systemctl show -p CPUUsageNSec -p MemoryCurrent noise-monitor > "$testdir/$label.service.start.txt"
curl -fsS "$base_url/api/health" > "$testdir/$label.health.start.json"
curl -fsS "$base_url/api/live" > "$testdir/$label.live.start.json"
curl -fsS "$base_url/api/tracker" > "$testdir/$label.tracker.start.json"
if command -v pidstat >/dev/null; then
  pidstat -u -r -p "$pid" 1 300 > "$testdir/$label.pidstat.txt"
else
  sleep 300
fi
systemctl show -p CPUUsageNSec -p MemoryCurrent noise-monitor > "$testdir/$label.service.end.txt"
curl -fsS "$base_url/api/health" > "$testdir/$label.health.end.json"
curl -fsS "$base_url/api/live" > "$testdir/$label.live.end.json"
curl -fsS "$base_url/api/tracker" > "$testdir/$label.tracker.end.json"
test "$(systemctl show -p MainPID --value noise-monitor)" = "$pid"
journalctl -u noise-monitor --since "$since" --no-pager > "$testdir/$label.journal.txt"
~~~

Para CPU total, divide el delta de `CPUUsageNSec` por la duración observada en nanosegundos. Un valor de 1 equivale a un núcleo ocupado. El contador del servicio incluye el proceso psicoacústico y los subprocesos de codificación. Si systemd no expone el valor, en cgroup v2 usa el delta de `usage_usec` de `cpu.stat` bajo `/sys/fs/cgroup$(systemctl show -p ControlGroup --value noise-monitor)`. `pidstat` pertenece a `sysstat` y aporta CPU y RSS del proceso principal.

Compara también `inference_count` y `frame_count` de `/api/tracker` por segundo, y el delta de `inference_ms_total` dividido por el delta de `inference_count` de audio. Verifica que en Coral haya inferencias de audio, que `active_backend` sea `coral` y que `fallbacks` sea cero. Descarta una corrida si cambia el PID, si falla un endpoint o si hubo fallback. Busca errores de USB, desbordes de audio, clips vacíos y reinicios en los logs.

## 3. Decisión y vuelta a CPU

- Adopta Coral si el informe emparejado conserva una calidad aceptable, sin voces anotadas omitidas ni casos nuevos en que el backend dejaría audio que CPU habría silenciado. Revisa los casos etiquetados y los cambios cerca del umbral.
- Exige una reducción de CPU total que se repita frente a ambas corridas en CPU y que no empeore el video ni la estabilidad del audio. Especifica el margen que importa al nodo antes de comparar. Si faltó carga de audio, repite con más eventos.
- Si adopta Coral, deja `backend: coral` y `fallback_to_cpu: true`. Comprueba el backend efectivo, la detección de video y un clip real tras el reinicio. Para volver al estado anterior, restaura la copia de `config.yaml` y reinicia el servicio. Si se hicieron otros cambios válidos durante el ensayo, cambia solo `backend: cpu`.
- En la ventana de mantenimiento puedes comprobar el fallback con una ruta temporal al modelo Coral inexistente. Tras reiniciar, espera `requested_backend: coral`, `active_backend: cpu` y `fallbacks: 1`. Restaura la ruta y reinicia de nuevo. Así se prueba la recuperación de arranque sin desconectar el Coral del detector de video.
- Entrega un resumen con commit, versiones del entorno, huellas de modelos, cantidad de clips, calidad, latencia y CPU/RSS, FPS de video, incidencias y configuración final. Mantén el JSON por clip y los medios en el nodo.
