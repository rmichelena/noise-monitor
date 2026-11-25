"""
Event Manager: una sola tabla `events`, un clip por evento.

1. Passages (del CentroidTracker via VideoMonitor) → fila `traffic` con su clip.
2. Triggers del AudioMonitor → clasificación YAMNet y fila `noise` con su clip.

Correlación: si un trigger vehicular coincide con un pasaje `traffic`
reciente, esa fila pasa a `both`, recibe los datos acústicos y un único clip
con la ventana unión; el clip previo del pasaje se borra.

La tabla `events` la crea scripts/migrate_unified.py (hasta el cutover no
vive en schema.sql): `run()` se niega a arrancar sin ella.
"""

import json
import logging
import sqlite3
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from uuid import uuid4

import cv2
import numpy as np
from scipy.io import wavfile

from .trigger_policy import TriggerPolicy
from .utils import SPEECH_CLASSES, init_database, load_config, resolve_path
from .vehicle_tracker import VEHICLE_CLASSES

logger = logging.getLogger(__name__)

NOISE_PRIORITY = {
    "truck": 1,
    "bus": 1,
    "motorcycle": 2,
    "car": 3,
    "van": 3,
    "bicycle": 4,
    "person": 5,
}


# Escala del indicador quemado. Hasta que haya calibración con sonómetro
# los valores son relativos; la barra muestra la misma escala que el slider.
DB_BAR_MAX = 75.0
# Bandas (límite superior, color BGR): verde, amarillo, naranja, rojo.
DB_BANDS = [
    (40.0, (80, 200, 80)),
    (50.0, (0, 220, 230)),
    (60.0, (0, 150, 255)),
    (DB_BAR_MAX, (40, 40, 220)),
]


def _iso(timestamp: float) -> str:
    """Epoch → ISO 8601 local, el formato de `events.timestamp`."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%S")


def _db_color(value: float) -> tuple[int, int, int]:
    for limit, color in DB_BANDS:
        if value <= limit:
            return color
    return DB_BANDS[-1][1]


def _dim(color: tuple[int, int, int], factor: float = 0.35):
    return tuple(int(c * factor) for c in color)


def draw_frame_overlay(frame, timestamp: float, db_value: float | None):
    """Copia del frame con fecha/hora, dB y barra de nivel quemados."""
    img = frame.copy()
    h, w = img.shape[:2]
    scale = h / 1080.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(1, int(round(2 * scale)))

    def put(text, x, y, size):
        cv2.putText(img, text, (x, y), font, size, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, size, (255, 255, 255), thick, cv2.LINE_AA)

    stamp = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    put(stamp, int(18 * scale), int(40 * scale), 0.75 * scale)

    numeric = isinstance(db_value, (int, float))
    text = f"{db_value:.1f} dB(A)" if numeric else "— dB(A)"

    # Barra de nivel: fondo segmentado tenue + relleno brillante.
    bar_w = int(260 * scale)
    bar_h = max(6, int(14 * scale))
    x0 = int(18 * scale)
    y0 = h - int(12 * scale) - bar_h
    put(text, x0, y0 - int(10 * scale), 1.0 * scale)

    seg_w = bar_w / len(DB_BANDS)
    for i, (_, color) in enumerate(DB_BANDS):
        cv2.rectangle(
            img,
            (x0 + int(i * seg_w), y0),
            (x0 + int((i + 1) * seg_w), y0 + bar_h),
            _dim(color),
            -1,
        )
    if numeric:
        level = max(0.0, min(float(db_value), DB_BAR_MAX))
        fill_w = int(bar_w * level / DB_BAR_MAX)
        if fill_w > 0:
            cv2.rectangle(img, (x0, y0), (x0 + fill_w, y0 + bar_h), _db_color(level), -1)
    cv2.rectangle(
        img, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), max(1, int(scale))
    )
    return img


def assign_culprit(detections: list[dict]) -> str:
    vehicle_dets = [d for d in detections if d.get("class") in NOISE_PRIORITY]
    if not vehicle_dets:
        return "unknown"
    sorted_dets = sorted(
        vehicle_dets, key=lambda d: NOISE_PRIORITY.get(d["class"], 99)
    )
    return sorted_dets[0]["class"]


def calculate_confidence(detections: list[dict]) -> str:
    """Confianza en la atribución según cuántas clases compiten.

    Se evalúa por clase distinta, no por número de detecciones: ventanas de
    varios frames repiten el mismo vehículo.
    """
    classes = {d["class"] for d in detections if d.get("class") in NOISE_PRIORITY}
    if not classes:
        return "low"
    if len(classes) == 1:
        return "high"
    priorities = {NOISE_PRIORITY[c] for c in classes}
    return "medium" if len(priorities) > 1 else "low"


def get_lane(detections: list[dict]) -> str:
    lanes = [d.get("lane") for d in detections if d.get("lane")]
    if not lanes:
        return "none"
    left = sum(1 for l in lanes if l == "left")
    right = sum(1 for l in lanes if l == "right")
    if left and right:
        return "both"
    return "left" if left else "right"


# Columnas que el reclamo de `_upgrade_to_both` sobrescribe.
_CLAIM_COLUMNS = (
    "kind", "timestamp", "peak_ts", "db_max", "db_avg", "trigger_type", "sound_class",
    "sound_category", "sound_confidence", "sound_labels_json", "confidence",
)


class EventManager:
    """
    Consume dos colas en threads separados:
      - passage_queue  → INSERT events (traffic)
      - trigger_queue  → correlación + clips + INSERT/UPDATE events (noise/both)
    """

    def __init__(
        self,
        audio_monitor,
        video_monitor,
        config: dict | None = None,
        base_dir: Path | None = None,
    ):
        self.audio_monitor = audio_monitor
        self.video_monitor = video_monitor
        self.config = config or load_config()
        self.base_dir = Path(base_dir or Path.cwd())

        buffer_cfg = self.config["buffer"]
        self.pre_seconds = int(buffer_cfg["pre_event_seconds"])
        self.post_seconds = int(buffer_cfg["post_event_seconds"])
        self.peak_db = float(self.config["triggers"].get("peak_db", float("inf")))
        self.correlation_db = float(self.config["triggers"].get("correlation_db", float("inf")))
        # Espaciado mínimo entre eventos: evita guardar un clip por segundo
        # cuando hay tráfico continuo (cada clip cuesta CPU y disco).
        self.cooldown_seconds = float(self.config["triggers"].get("cooldown_seconds", 0))
        self.discarded_stale = 0
        self.discarded_cooldown = 0
        # Habilitación y horarios de las secciones ruido/tráfico.
        self.trigger_policy = TriggerPolicy.from_config(self.config)
        self.passages_not_recorded = 0
        self.discarded_without_traffic = 0
        self.union_clip_fallbacks = 0
        self.events_lost = 0
        self.perceptual_queue_full = 0
        self._last_event_ts = 0.0
        psycho = (self.config.get("triggers") or {}).get("psychoacoustic") or {}
        try:
            qsize = int(psycho.get("queue_size") or 8)
        except (TypeError, ValueError):
            qsize = 8
        self._perceptual_queue: Queue = Queue(maxsize=max(1, min(64, qsize)))
        if audio_monitor is not None:
            audio_monitor.event_manager = self

        self.db_path = str(resolve_path(self.config, "database", self.base_dir))
        self.clips_dir = resolve_path(self.config, "clips", self.base_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        init_database(self.config, self.base_dir)
        self._running = False

        # Cache de passages recientes para correlación rápida (últimos 30s)
        self._recent_passages: list[dict] = []
        self._passages_lock = threading.Lock()
        self._PASSAGE_TTL = 30.0  # seconds
        # Ruidos `both` por un vehículo en escena sin pasaje cerrado (S6).
        # Mientras dure el TTL, el pasaje de ese vehículo se adjunta a su fila
        # en vez de insertar un `traffic` aparte. Protegido por _passages_lock.
        self._in_scene: list[dict] = []
        self.passages_counted = 0
        self.count_errors = 0

    # ------------------------------------------------------------------
    # Clip saving (unchanged logic)
    # ------------------------------------------------------------------
    def _unique_clip_path(self, prefix: str, event_timestamp: float) -> Path:
        """Ruta de clip libre para ese segundo.

        El nombre se deriva al segundo: dos eventos en el mismo segundo (pico
        y sostenido encadenados, o dos pasajes cruzándose) escribían el mismo
        archivo, así que el segundo sobrescribía el clip del primero y su fila
        quedaba apuntando a la evidencia del otro evento. Se añade un sufijo
        `_2`, `_3`... cuando el nombre ya existe.
        """
        stamp = datetime.fromtimestamp(event_timestamp).strftime("%Y%m%d_%H%M%S")
        candidate = self.clips_dir / f"{prefix}_{stamp}.mp4"
        n = 2
        while candidate.exists():
            candidate = self.clips_dir / f"{prefix}_{stamp}_{n}.mp4"
            n += 1
        return candidate

    def _save_clip(
        self,
        audio_blocks: list[tuple[float, np.ndarray]],
        video_frames: list[tuple[float, np.ndarray]],
        sample_rate: int,
        event_timestamp: float,
        mute_audio: bool = False,
        prefix: str = "event",
        unique_name: bool = False,
    ) -> tuple[str | None, int]:
        if not audio_blocks or not video_frames:
            logger.warning("Buffers vacíos, no se puede guardar clip")
            return None, 0

        audio_data = np.concatenate([b[1] for b in audio_blocks], axis=0)
        if audio_data.dtype != np.int16:
            audio_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_audio:
            wavfile.write(f_audio.name, sample_rate, audio_data)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_video:
            video_path = f_video.name

        first_frame = video_frames[0][1]
        frame_h, frame_w = first_frame.shape[:2]
        actual_resolution = (frame_w, frame_h)

        # fps real derivado de los timestamps: si la cámara entrega 60 fps y
        # se escribe a 30, el video duraba el doble que el audio.
        span = video_frames[-1][0] - video_frames[0][0]
        fps_out = (len(video_frames) - 1) / span if span > 0 else 1.0
        fps_out = max(1.0, min(fps_out, 120.0))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps_out, actual_resolution)
        if not out.isOpened():
            logger.error("No se pudo abrir el writer MP4; clip descartado")
            Path(f_audio.name).unlink(missing_ok=True)
            Path(video_path).unlink(missing_ok=True)
            return None, 0

        video_cfg = self.config.get("video", {})
        overlay = bool(video_cfg.get("overlay_db", True))
        every = max(1, int(video_cfg.get("overlay_every_frames", 5)))
        db_val = None
        try:
            for idx, (ts, frame) in enumerate(video_frames):
                if overlay:
                    if idx % every == 0:
                        audio = self.audio_monitor
                        if audio is not None and hasattr(audio, "db_at"):
                            try:
                                db_val = audio.db_at(ts)
                            except Exception:
                                db_val = None
                    frame = draw_frame_overlay(frame, ts, db_val)
                out.write(frame)
        finally:
            out.release()

        duration_ms = int(len(audio_data) / sample_rate * 1000)
        audio_span = audio_blocks[-1][0] - audio_blocks[0][0]
        logger.info(
            "Clip: audio %.2fs (%d bloques), video %.2fs (%d frames, %.1f fps)",
            audio_span, len(audio_blocks), span, len(video_frames), fps_out,
        )
        effective_prefix = f"{prefix}_{uuid4().hex}" if unique_name else prefix
        clip_path = self._unique_clip_path(effective_prefix, event_timestamp)

        try:
            # Todos los -i primero: cualquier opción entre dos entradas la
            # interpreta ffmpeg como opción de la entrada siguiente.
            cmd = ["ffmpeg", "-y", "-i", video_path]
            if not mute_audio:
                cmd += ["-i", f_audio.name]
            # Recompresión: el mp4v intermedio pesa ~5 MB/s a 1080p.
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-pix_fmt", "yuv420p"]
            if mute_audio:
                cmd += ["-an"]  # voz detectada: clip sin audio
            else:
                cmd += ["-c:a", "aac", "-b:a", "64k", "-shortest"]
            cmd.append(str(clip_path))
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg falló: %s", e.stderr.decode() if e.stderr else e)
            return None, 0
        finally:
            Path(f_audio.name).unlink(missing_ok=True)
            Path(video_path).unlink(missing_ok=True)

        # Ruta relativa al proyecto (antes estaba fija a data/clips, que solo
        # coincide por casualidad con la config por defecto).
        try:
            stored = clip_path.relative_to(self.base_dir)
        except ValueError:
            stored = clip_path
        return str(stored), duration_ms

    # ------------------------------------------------------------------
    # Passage processing (from tracker via VideoMonitor.passage_queue)
    # ------------------------------------------------------------------
    def _passage_has_speech(self, audio_list: list) -> bool:
        """Si el audio de un clip debe ir mudo: hay voz, o no se pudo saber.

        Recorre todo el audio con `scan_all_speech`: `classify` sólo mira los
        últimos 0,975 s, y en un clip unión esa cola es el post-roll del
        ruido, no el pasaje.
        """
        classifier = getattr(self.audio_monitor, "_classifier", None)
        if classifier is None or not audio_list:
            return False
        sample_rate = self.config["audio"]["sample_rate"]
        try:
            window = np.concatenate([b for _, b in audio_list], axis=0)
            _, speech = classifier.classify_event(window, sample_rate, 0, scan_all_speech=True)
            return speech
        except Exception:
            logger.warning("Pasaje sin audio porque falló la clasificación", exc_info=True)
            return True

    def _save_passage_clip(self, passage: dict) -> tuple[str, int, bool]:
        """Clip corto (5 s) alrededor del pasaje; audio mudo si hay voz.

        Devuelve (clip_path, duration_ms, audio_muted).
        """
        now = time.time()
        t_end = min(now, passage["timestamp_exit"] + self.post_seconds)
        t_start = max(passage["timestamp_enter"] - self.pre_seconds, t_end - 5.0)
        if t_start >= t_end:
            return "", 0, False

        # Copiar antes de iterar: los buffers los mutan los hilos de captura
        # y un deque modificado durante la iteración lanza RuntimeError.
        audio_list = [
            (ts, block)
            for ts, block in self.audio_monitor.audio_buffer.copy()
            if t_start <= ts <= t_end
        ]
        video_list = [
            (ts, frame)
            for ts, frame in self.video_monitor.video_buffer.copy()
            if t_start <= ts <= t_end
        ]
        if len(video_list) < 10 or not audio_list:
            return "", 0, False

        mute_audio = self._passage_has_speech(audio_list)

        sample_rate = self.config["audio"]["sample_rate"]
        clip_path, duration_ms = self._save_clip(
            audio_list, video_list, sample_rate, passage["timestamp_exit"],
            mute_audio=mute_audio, prefix="passage",
        )
        return clip_path or "", duration_ms or 0, mute_audio

    def _remember_passage(self, passage: dict, event_id: int | None) -> None:
        """Deja el pasaje disponible para correlación con triggers de audio.

        `db_id=None`: el pasaje no tiene fila; si un ruido lo reclama, el
        evento se inserta directamente como `both`.
        """
        passage["db_id"] = event_id
        passage["claimed"] = False
        with self._passages_lock:
            now = time.time()
            self._recent_passages = [
                p for p in self._recent_passages
                if now - p["timestamp_exit"] < self._PASSAGE_TTL
            ]
            self._recent_passages.append(passage)

    def _count_passage(self, passage: dict) -> None:
        """Suma el pasaje a `traffic_counts`, se grabe o no como evento.

        Los flags de `triggers` deciden qué se guarda en `events`; la medición
        de tráfico (gráfico, contadores, stats) no depende de ellos.
        """
        speed = passage.get("speed_kmh")
        speed = float(speed) if isinstance(speed, (int, float)) and not isinstance(speed, bool) else None
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO traffic_counts
                    (minute, vehicle_class, direction, count, speed_n, speed_sum, speed_max)
                VALUES (?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT (minute, vehicle_class, direction) DO UPDATE SET
                    count = count + 1,
                    speed_n = speed_n + excluded.speed_n,
                    speed_sum = speed_sum + excluded.speed_sum,
                    speed_max = CASE
                        WHEN excluded.speed_max IS NULL THEN speed_max
                        WHEN speed_max IS NULL OR excluded.speed_max > speed_max THEN excluded.speed_max
                        ELSE speed_max END
                """,
                (
                    _iso(passage["timestamp_enter"])[:16], passage["vehicle_class"],
                    passage.get("direction") or "",
                    1 if speed is not None else 0, speed or 0.0, speed,
                ),
            )
            conn.commit()
            self.passages_counted += 1
        except sqlite3.Error:
            # Un fallo del conteo no puede costar el evento del pasaje.
            self.count_errors += 1
            logger.exception("No se pudo contar el pasaje %s", passage.get("track_id"))
        finally:
            conn.close()

    def _process_passage(self, passage: dict):
        """Cuenta el pasaje y lo guarda: adjunto a un `both` en escena o como `traffic`."""
        self._count_passage(passage)
        if self._attach_to_in_scene(passage):
            return
        self._record_passage(passage)

    # ------------------------------------------------------------------
    # Vehículo en escena (S6): ruido sin pasaje cerrado, vehículo detectado
    # ------------------------------------------------------------------
    def _open_in_scene(self, vehicle_class: str, start: float, end: float) -> dict | None:
        """Registra un `both` en escena, o None si el vehículo ya tiene atribución.

        Si un pasaje reciente de esa clase (reclamado o fijado) u otro ruido
        en escena se solapa con la ventana, el vehículo detectado es muy
        probablemente ese: un vehículo no se atribuye a dos eventos.
        """
        entry = {"vehicle_class": vehicle_class, "start": start, "end": end,
                 "event_id": None, "passage": None}
        with self._passages_lock:
            now = time.time()
            self._in_scene = [
                e for e in self._in_scene
                if e["event_id"] is None or now - e["end"] < self._PASSAGE_TTL
            ]
            overlapping = [
                item for item in (
                    *((p["timestamp_enter"], p["timestamp_exit"], p["vehicle_class"])
                      for p in self._recent_passages),
                    *((e["start"], e["end"], e["vehicle_class"]) for e in self._in_scene),
                )
                if item[2] == vehicle_class and item[0] <= end and item[1] >= start
            ]
            if overlapping:
                return None
            self._in_scene.append(entry)
        return entry

    def _attach_to_in_scene(self, passage: dict) -> bool:
        """Adjunta el pasaje al único `both` en escena compatible, si lo hay.

        Compatible = misma clase, ventanas que se solapan y sin pasaje. Con
        dos candidatos no se elige: el pasaje sigue el camino normal.
        """
        with self._passages_lock:
            now = time.time()
            candidates = [
                e for e in self._in_scene
                if e["passage"] is None
                and e["vehicle_class"] == passage["vehicle_class"]
                and passage["timestamp_enter"] <= e["end"]
                and passage["timestamp_exit"] >= e["start"]
                and (e["event_id"] is None or now - e["end"] < self._PASSAGE_TTL)
            ]
            if len(candidates) != 1:
                return False
            entry = candidates[0]
            entry["passage"] = passage
            event_id = entry["event_id"]
        # Si el ruido todavía no insertó su fila, la escribe él con el pasaje.
        if event_id is not None and not self._write_in_scene_passage(event_id, passage):
            return False
        logger.info("Pasaje %s adjunto al evento en escena", passage.get("track_id"))
        return True

    def _write_in_scene_passage(self, event_id: int, passage: dict) -> bool:
        """Escribe el pasaje en la fila `both` en escena; False si no pudo."""
        conn = sqlite3.connect(self.db_path)
        try:
            updated = conn.execute(
                """
                UPDATE events SET track_id = ?, direction = ?, speed_kmh = ?,
                    detection_score = ?, lane = COALESCE(?, lane)
                WHERE id = ? AND kind = 'both' AND track_id IS NULL
                    AND COALESCE(pinned, 0) = 0
                """,
                (
                    passage.get("track_id"), passage.get("direction"), passage.get("speed_kmh"),
                    passage.get("detection_score"), passage.get("lane"), event_id,
                ),
            ).rowcount
            conn.commit()
        except sqlite3.Error:
            logger.exception("No se pudo adjuntar el pasaje al evento #%d", event_id)
            return False
        finally:
            conn.close()
        if not updated:
            logger.info("Evento #%d fijado o ya con pasaje: el pasaje va aparte", event_id)
        return bool(updated)

    def _record_passage(self, passage: dict):
        """Inserta un pasaje completo como evento `traffic` con su propio clip."""
        if not self.trigger_policy.records_passage(passage["vehicle_class"], passage["timestamp_enter"]):
            # Ni fila ni clip, pero el tracker sigue sirviendo a la correlación.
            self.passages_not_recorded += 1
            self._remember_passage(passage, None)
            return
        db_now = self.audio_monitor.current_db

        # Clip del pasaje: vehículos en movimiento, aunque no hicieran ruido.
        # Si después un ruido lo reclama, el clip unión lo reemplaza.
        clip_path, duration_ms, muted = "", 0, False
        try:
            clip_path, duration_ms, muted = self._save_passage_clip(passage)
        except Exception:
            logger.exception("No se pudo guardar el clip del pasaje")

        conn = sqlite3.connect(self.db_path)
        try:
            cur = conn.execute(
                """
                INSERT INTO events
                    (kind, timestamp, start_ts, end_ts, db_max, vehicle_class, lane,
                     direction, speed_kmh, detection_score, track_id, audio_muted,
                     duration_ms, clip_path)
                VALUES ('traffic', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _iso(passage["timestamp_enter"]),
                    passage["timestamp_enter"],
                    passage["timestamp_exit"],
                    round(db_now, 1),
                    passage["vehicle_class"],
                    passage.get("lane"),
                    passage.get("direction"),
                    passage.get("speed_kmh"),
                    passage.get("detection_score"),
                    passage.get("track_id"),
                    1 if muted else 0,
                    duration_ms or None,
                    clip_path or None,
                ),
            )
            event_id = cur.lastrowid
            conn.commit()
        finally:
            conn.close()

        # En memoria para la correlación con triggers de audio.
        self._remember_passage(passage, event_id)

        logger.info(
            "Traffic #%d: %s (%s) %.1f dB @ %s",
            event_id,
            passage["vehicle_class"],
            passage.get("lane", "?"),
            db_now,
            _iso(passage["timestamp_enter"]),
        )

    def _find_matching_passage(
        self, vehicle_class: str, timestamp: float, window: tuple[float, float] | None = None,
    ) -> dict | None:
        """Pasaje `traffic` reciente de la misma clase; lo marca como reclamado.

        Un pasaje reclamado ya es (o va a ser) `both` y sale del conjunto de
        candidatos: un pasaje no se asigna a dos ruidos.
        """
        with self._passages_lock:
            if window:
                left, right = window
                # La unicidad se evalúa sobre todos los pasajes de la ventana:
                # que uno ya esté reclamado no vuelve inequívoco al otro.
                candidates = [
                    p for p in self._recent_passages
                    if p["timestamp_enter"] <= right and p["timestamp_exit"] >= left
                ]
                if (
                    len(candidates) == 1
                    and candidates[0]["vehicle_class"] == vehicle_class
                    and not candidates[0].get("claimed")
                ):
                    candidates[0]["claimed"] = True
                    return candidates[0]
                return None
            best = None
            best_dt = float("inf")
            for p in self._recent_passages:
                if p["vehicle_class"] == vehicle_class and not p.get("claimed"):
                    dt = abs(p["timestamp_exit"] - timestamp)
                    if dt < best_dt:
                        best_dt = dt
                        best = p
            if best and best_dt < self._PASSAGE_TTL:
                best["claimed"] = True
                return best
        return None

    def _window_from_buffers(
        self, held: list[tuple[float, object]], buffer, start: float, end: float,
    ) -> list[tuple[float, object]]:
        """Lo retenido más lo que siga en el buffer dentro de [start, end]."""
        merged = {ts: item for ts, item in held if start <= ts <= end}
        for ts, item in buffer.copy():
            if start <= ts <= end and ts not in merged:
                merged[ts] = item
        return sorted(merged.items(), key=lambda pair: pair[0])

    def _delete_replaced_clip(self, stored: str | None, replacement: str) -> None:
        """Borra un clip que ya no referencia ninguna fila (el del pasaje que
        la unión reemplazó, o uno cuyo INSERT no entró)."""
        if not stored or stored == replacement:
            return
        path = Path(stored)
        path = (path if path.is_absolute() else self.base_dir / path).resolve()
        if self.clips_dir.resolve() not in path.parents:
            logger.warning("Clip reemplazado fuera del directorio de clips; no se borra: %s", path)
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("No se pudo borrar el clip reemplazado %s", path, exc_info=True)

    def _union_clip(
        self,
        passage: dict,
        audio_list: list,
        video_list: list,
        noise_window: tuple[float, float],
        peak_ts: float,
        *,
        union_muted: bool,
        unique_name: bool,
        label: str,
    ) -> tuple[str | None, int | None, bool, float, float, bool]:
        """Un solo clip para ruido + pasaje: [min(inicios), min(max(fines), ahora)].

        Si falla, cae al clip del ruido con su ventana. Devuelve
        (clip_path, duration_ms, audio_muted, start_ts, end_ts, union_ok);
        `union_ok` es False si el clip final no es la unión (sólo ruido, o
        ninguno: clip_path None). La unión se clasifica entera salvo que ya
        venga muda: ninguna clasificación previa la cubre (el disparo dBA
        mira ±0,97 s del pico; el clip del pasaje, sus últimos 5 s).
        """
        noise_start, noise_end = noise_window
        union_start = min(noise_start, passage["timestamp_enter"])
        union_end = min(max(noise_end, passage["timestamp_exit"]), time.time())
        union_audio = self._window_from_buffers(
            audio_list, self.audio_monitor.audio_buffer, union_start, union_end)
        union_video = self._window_from_buffers(
            video_list, self.video_monitor.video_buffer, union_start, union_end)
        sample_rate = self.config["audio"]["sample_rate"]
        if not union_muted:
            union_muted = self._passage_has_speech(union_audio)

        clip_path, duration_ms, muted = None, None, union_muted
        start_ts, end_ts = union_start, union_end
        union_ok = False
        try:
            clip_path, duration_ms = self._save_clip(
                union_audio, union_video, sample_rate, peak_ts,
                mute_audio=union_muted, unique_name=unique_name,
            )
        except Exception:
            logger.exception("Falló el clip unión del evento %s", label)
            clip_path = None
        if clip_path and union_audio and union_video:
            covered = [ts for ts, _ in union_audio[:1] + union_video[:1]]
            covered_end = [ts for ts, _ in union_audio[-1:] + union_video[-1:]]
            start_ts = max(union_start, min(covered))
            end_ts = min(union_end, max(covered_end))
            union_ok = True
        else:
            self.union_clip_fallbacks += 1
            logger.warning("Clip unión del evento %s falló; se guarda el clip del ruido", label)
            # Con el muteo de la unión, no el del ruido: el pasaje suele caer dentro
            # de la ventana del ruido, y entonces este clip también lo tiene.
            muted = union_muted
            start_ts, end_ts = noise_window
            try:
                clip_path, duration_ms = self._save_clip(
                    audio_list, video_list, sample_rate, peak_ts,
                    mute_audio=union_muted, unique_name=unique_name,
                )
            except Exception:
                logger.exception("Falló también el clip del ruido del evento %s", label)
                clip_path = None

        return clip_path, duration_ms, muted, start_ts, end_ts, union_ok

    def _upgrade_to_both(
        self,
        passage: dict,
        fields: dict,
        audio_list: list,
        video_list: list,
        noise_window: tuple[float, float],
        mute_audio: bool,
        unique_name: bool,
    ) -> int | None:
        """Convierte el `traffic` del pasaje en `both` con un único clip unión.

        Devuelve el id del evento, o None si el pasaje ya no era `traffic`, está
        fijado o se fijó durante el clip unión: el llamador guarda el ruido como
        `noise` con su propio clip.
        """
        event_id = passage["db_id"]
        conn = sqlite3.connect(self.db_path)
        try:
            # Lo que el reclamo pisa: si la fila se fija durante el clip unión,
            # se restaura tal cual (ver más abajo). Lectura y reclamo en la
            # misma transacción de escritura, para restaurar lo que se pisó.
            conn.execute("BEGIN IMMEDIATE")
            original = conn.execute(
                f"SELECT {', '.join(_CLAIM_COLUMNS)} FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            # Reclamo atómico: sólo un trigger puede pasar este pasaje a both.
            claimed = conn.execute(
                """
                UPDATE events SET kind = 'both', timestamp = ?, peak_ts = ?,
                    db_max = ?, db_avg = ?, trigger_type = ?, sound_class = ?,
                    sound_category = ?, sound_confidence = ?, sound_labels_json = ?,
                    confidence = ?
                WHERE id = ? AND kind = 'traffic' AND COALESCE(pinned, 0) = 0
                """,
                (
                    fields["timestamp"], fields["peak_ts"], fields["db_max"],
                    fields["db_avg"], fields["trigger_type"], fields["sound_class"],
                    fields["sound_category"], fields["sound_confidence"],
                    fields["sound_labels_json"], fields["confidence"], event_id,
                ),
            ).rowcount
            conn.commit()
            if not claimed:
                # El pin va en el mismo WHERE que el reclamo: un pin entre una
                # lectura previa y el UPDATE no puede colarse.
                state = conn.execute(
                    "SELECT kind, pinned FROM events WHERE id = ?", (event_id,)
                ).fetchone()
                if state is not None and state[0] == "traffic" and state[1]:
                    # El dueño fijó el pasaje: su fila no se toca. El ruido se
                    # guarda como `noise` con su propio clip y su atribución.
                    # `claimed` sigue en True: fijado = consumido, así ningún
                    # otro disparo vuelve a elegirlo mientras dure el TTL.
                    logger.info("Pasaje #%d fijado: el ruido se guarda aparte", event_id)
                    return None
                logger.warning("El pasaje #%d ya no es traffic; el ruido queda aparte", event_id)
                return None
            old_clip, passage_muted = conn.execute(
                "SELECT clip_path, audio_muted FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        finally:
            conn.close()

        clip_path, duration_ms, muted, start_ts, end_ts, union_ok = self._union_clip(
            passage, audio_list, video_list, noise_window, fields["peak_ts"],
            union_muted=mute_audio or bool(passage_muted),
            unique_name=unique_name, label=f"#{event_id}",
        )
        if not union_ok and old_clip:
            # El clip del pasaje es la única evidencia con el vehículo: la
            # fila conserva su clip y su ventana, y el fallback sólo-ruido no
            # se usa. Los metadatos del ruido ya quedaron en el reclamo.
            logger.warning(
                "Evento #%d sin clip unión: conserva el del pasaje (%s)", event_id, old_clip,
            )
            if clip_path:
                self._delete_replaced_clip(clip_path, "")
            return event_id
        if not clip_path:
            logger.error(
                "Evento #%d sin clip nuevo: conserva el del pasaje (%s)", event_id, old_clip or "none",
            )
            return event_id

        conn = sqlite3.connect(self.db_path)
        try:
            # `pinned = 0` también acá: el clip unión tarda segundos y el
            # dueño puede fijar la fila mientras tanto.
            updated = conn.execute(
                "UPDATE events SET clip_path = ?, duration_ms = ?, audio_muted = ?,"
                " start_ts = ?, end_ts = ? WHERE id = ? AND COALESCE(pinned, 0) = 0",
                (clip_path, duration_ms, 1 if muted else 0, start_ts, end_ts, event_id),
            ).rowcount
            conn.commit()
            if not updated:
                # Se fijó mientras se generaba la unión. Una fila `both` con los
                # datos del ruido sobre la ventana del pasaje sería incoherente
                # (timestamp > end_ts): se deshace el reclamo y el ruido va
                # aparte como `noise`, igual que con un pasaje ya fijado.
                # `claimed` queda en True: fijado = consumido.
                if original is not None:
                    conn.execute(
                        f"UPDATE events SET {', '.join(f'{c} = ?' for c in _CLAIM_COLUMNS)} WHERE id = ?",
                        (*original, event_id),
                    )
                    conn.commit()
                logger.warning(
                    "Pasaje #%d fijado durante el clip unión: vuelve a traffic, se descarta la "
                    "unión y el ruido se guarda aparte", event_id,
                )
                self._delete_replaced_clip(clip_path, old_clip or "")
                return None
        except sqlite3.Error:
            # El reclamo ya está commiteado: propagar la excepción haría que
            # el llamador inserte un `noise` duplicado. La fila sigue
            # apuntando al clip del pasaje, que se conserva; el clip unión
            # recién guardado no lo referencia nadie y se borra.
            logger.exception(
                "No se pudo actualizar el clip unión del evento #%d: conserva %s",
                event_id, old_clip or "none",
            )
            # Con old_clip como «reemplazo», nunca se borra el archivo que la
            # fila sigue referenciando aunque los nombres coincidieran.
            self._delete_replaced_clip(clip_path, old_clip or "")
            return event_id
        finally:
            conn.close()

        # Sólo con el UPDATE commiteado el clip viejo deja de estar referenciado.
        self._delete_replaced_clip(old_clip, clip_path)
        return event_id

    # ------------------------------------------------------------------
    # Trigger processing (from AudioMonitor.event_queue)
    # ------------------------------------------------------------------
    def _process_trigger(self, trigger: dict):
        scene: dict = {}
        try:
            self._process_trigger_impl(trigger, scene)
        finally:
            self._close_in_scene(scene.get("entry"), scene.get("event_id"))

    def _close_in_scene(self, entry: dict | None, event_id: int | None) -> None:
        """Cierra el registro en escena de un disparo, con o sin fila `both`.

        Con fila, queda abierto durante el TTL para el pasaje que cierre
        después; si un pasaje ya se adjuntó, se escribe acá. Sin fila (el
        disparo se descartó o falló), se quita y un pasaje ya adjunto sigue
        el camino normal: nunca se pierde.
        """
        if entry is None:
            return
        with self._passages_lock:
            if event_id is not None:
                entry["event_id"] = event_id
            elif entry in self._in_scene:
                self._in_scene.remove(entry)
            passage = entry["passage"]
        if passage is None:
            return
        if event_id is not None and self._write_in_scene_passage(event_id, passage):
            logger.info("Pasaje %s adjunto al evento en escena #%d", passage.get("track_id"), event_id)
            return
        self._record_passage(passage)

    def _process_trigger_impl(self, trigger: dict, scene: dict):
        trigger_type = trigger["type"]
        perceptual = trigger_type == "psychoacoustic"
        db_max = trigger["db"]
        db_avg = trigger.get("db_avg", db_max)
        timestamp = trigger["timestamp"]

        if self.cooldown_seconds > 0 and timestamp - self._last_event_ts < self.cooldown_seconds:
            if perceptual:
                self.discarded_cooldown += 1
            return
        if perceptual and time.time() >= trigger.get("expires_at", float("inf")):
            self.discarded_stale += 1
            logger.warning("Disparo psicoacústico vencido antes de guardar el clip")
            return

        # YAMNet classification (may be absent if classifier not loaded)
        sound_class = trigger.get("sound_class")
        sound_category = trigger.get("sound_category", "other")
        sound_confidence = trigger.get("sound_confidence")
        sound_labels = []
        mute_audio = bool(trigger.get("audio_muted")) or sound_class in SPEECH_CLASSES

        # Ventana real del clip [t-pre, t+post]; se espera a que el post exista.
        sample_rate = self.config["audio"]["sample_rate"]
        t_start = timestamp - self.pre_seconds
        t_end = timestamp + self.post_seconds
        wait = t_end - time.time()
        audio_list = []
        video_list = []
        # El historial se snapshotea junto al pre-roll: el deque sigue
        # deslizándose durante el sleep y podía expulsar esos niveles.
        level_history = self.audio_monitor.level_snapshot()
        if wait > 0:
            # Hold the pre-roll before waiting so that the ring cannot evict it.
            audio_list = [(ts, block) for ts, block in self.audio_monitor.audio_buffer.copy()
                          if t_start <= ts <= t_end]
            video_list = [(ts, frame) for ts, frame in self.video_monitor.video_buffer.copy()
                          if t_start <= ts <= t_end]
        if wait > 0:
            time.sleep(min(wait, self.post_seconds))
        audio_boundary = audio_list[-1][0] if audio_list else float("-inf")
        video_boundary = video_list[-1][0] if video_list else float("-inf")
        level_boundary = level_history[-1][0] if level_history else float("-inf")
        level_history.extend(
            level for level in self.audio_monitor.level_snapshot() if level[0] > level_boundary
        )
        audio_list.extend(
            (ts, block) for ts, block in self.audio_monitor.audio_buffer.copy()
            if t_start <= ts <= t_end and ts > audio_boundary
        )
        video_list.extend(
            (ts, frame) for ts, frame in self.video_monitor.video_buffer.copy()
            if t_start <= ts <= t_end and ts > video_boundary
        )
        # Culpable sobre TODAS las inferencias de la ventana, no sobre el
        # último frame: un frame suelto puede no contener los vehículos que
        # sí pasaron durante el evento (el caso "dice person con 3 autos").
        window_dets = self.video_monitor.detections_between(t_start, t_end)
        # S6 sólo con detecciones dentro de la ventana, no con el fallback.
        dets_in_window = bool(window_dets)
        if not window_dets and not perceptual:
            window_dets = self.video_monitor.recent_detections
        vehicle_dets = [d for d in window_dets if d.get("class") in NOISE_PRIORITY]

        # Skip low-level non-vehicle events before doing any inference.
        if not perceptual and not vehicle_dets and db_max < self.peak_db and db_max < self.correlation_db:
            return

        # db_max/db_avg describen el clip, no el bloque que disparó: el pico
        # real del pasaje suele llegar después del primer cruce del umbral.
        level_stats = self.audio_monitor.level_stats(audio_list, level_history)
        if level_stats:
            # Nunca por debajo del nivel que disparó: con cobertura parcial
            # el bloque del trigger puede haber quedado fuera.
            db_max = max(level_stats["max"], db_max)
            db_avg = level_stats["avg"]
        else:
            logger.warning(
                "Sin estadísticas de nivel utilizables para el clip de %.3f; se conserva el nivel del disparo",
                timestamp,
            )

        if perceptual or sound_class is None:
            sound_class, sound_category, sound_confidence, speech, sound_labels = (
                # La voz se busca en todo el clip también en dBA: las etiquetas
                # siguen saliendo de las ventanas del disparo. Sin YAMNet, sólo
                # el psicoacústico va mudo (como antes).
                self.audio_monitor.classify_event_clip(
                    audio_list, timestamp, scan_all_speech=True, mute_unclassified=perceptual,
                )
            )
            mute_audio |= speech

        if perceptual:
            candidates = {d["class"] for d in vehicle_dets if d["class"] not in {"person", "bicycle"}}
            heard_vehicle = (sound_class or "").lower().removesuffix(" passing by")
            culprit = heard_vehicle if sound_category == "vehicle" and candidates == {heard_vehicle} else None
            confidence = "low"
            lane = get_lane(vehicle_dets) if culprit else "none"
        elif sound_category == "vehicle" or (sound_category == "other" and vehicle_dets):
            culprit = assign_culprit(window_dets)
            confidence = calculate_confidence(window_dets)
            lane = get_lane(vehicle_dets) if vehicle_dets else "none"
        else:
            culprit = None
            confidence = "low"
            lane = "none"

        # Correlación con un pasaje traffic reciente.
        passage = None
        if culprit and culprit != "unknown":
            passage = self._find_matching_passage(
                culprit, timestamp, window=(t_start, t_end) if perceptual else None,
            )
            if perceptual and passage is None:
                culprit, lane = None, "none"

        # S6: sin pasaje cerrado, pero con el vehículo detectado en la ventana
        # (p. ej. un bus que todavía cruza): el evento nace `both` con el clip
        # del ruido. Sólo dBA: el psicoacústico descarta el culpable sin pasaje.
        in_scene = None
        if (
            passage is None and not perceptual and dets_in_window
            and culprit in VEHICLE_CLASSES
        ):
            in_scene = self._open_in_scene(culprit, t_start, t_end)
            scene["entry"] = in_scene

        if passage is None and in_scene is None and self.trigger_policy.noise_only_with_traffic:
            self.discarded_without_traffic += 1
            logger.info("Ruido %.1f dB sin pasaje correlacionado: descartado", db_max)
            return

        ts_iso = _iso(timestamp)
        fields = {
            "timestamp": ts_iso, "peak_ts": timestamp, "db_max": db_max, "db_avg": db_avg,
            "trigger_type": trigger_type, "sound_class": sound_class,
            "sound_category": sound_category, "sound_confidence": sound_confidence,
            "sound_labels_json": (
                json.dumps(sound_labels, ensure_ascii=False, allow_nan=False) if sound_labels else None
            ),
            "confidence": confidence,
        }

        event_id = None
        kind = "noise"
        row = {
            "start_ts": t_start, "end_ts": t_end, "vehicle_class": culprit, "lane": lane,
            "audio_muted": 1 if mute_audio else 0,
        }
        noise_attribution = {"vehicle_class": culprit, "lane": lane}
        if passage is not None and passage.get("db_id") is None:
            # Pasaje sin fila (tráfico apagado, fuera de horario, clase no
            # registrada o traffic_only_with_noise): el evento nace `both`.
            try:
                clip_path, duration_ms, muted, start_ts, end_ts, union_ok = self._union_clip(
                    passage, audio_list, video_list, (t_start, t_end), timestamp,
                    union_muted=mute_audio,
                    unique_name=perceptual, label=f"del pasaje {passage.get('track_id')}",
                )
            except Exception:
                # Igual que el upgrade: nunca perder el evento acústico. Se
                # libera el pasaje y el ruido sigue como `noise` con su clip.
                logger.exception(
                    "Falló el both del pasaje %s; el ruido se guarda aparte", passage.get("track_id"),
                )
                passage["claimed"] = False
            else:
                kind = "both"
                if not union_ok:
                    # Se conserva el `both` (no se pierden los metadatos del
                    # pasaje); el clip es sólo del ruido, o no hay clip.
                    logger.warning(
                        "Both del pasaje %s sin clip unión: clip %s",
                        passage.get("track_id"), "del ruido" if clip_path else "ninguno",
                    )
                row.update({
                    "start_ts": start_ts, "end_ts": end_ts, "audio_muted": 1 if muted else 0,
                    "vehicle_class": passage["vehicle_class"], "lane": passage.get("lane"),
                    "direction": passage.get("direction"), "speed_kmh": passage.get("speed_kmh"),
                    "detection_score": passage.get("detection_score"), "track_id": passage.get("track_id"),
                })
        elif passage is not None:
            try:
                event_id = self._upgrade_to_both(
                    passage, fields, audio_list, video_list, (t_start, t_end),
                    mute_audio, unique_name=perceptual,
                )
            except Exception:
                # Nunca perder el evento acústico por un fallo del upgrade:
                # se libera el reclamo y se guarda como `noise` aparte.
                logger.exception(
                    "Falló el upgrade del pasaje #%s; el ruido se guarda aparte",
                    passage.get("db_id"),
                )
                passage["claimed"] = False
                event_id = None
            if event_id is not None:
                kind = "both"

        if event_id is None and kind == "noise":
            clip_path, duration_ms = self._save_clip(
                audio_list, video_list, sample_rate, timestamp, mute_audio=mute_audio, unique_name=perceptual,
            )
        if event_id is None and not clip_path:
            clip_path, duration_ms = None, None
        if in_scene is not None:
            # Conserva el clip del ruido; el pasaje, si cierra, se adjunta
            # después (track_id, sentido, velocidad) sin regenerar unión.
            kind = "both"

        conn = sqlite3.connect(self.db_path)
        try:
            if event_id is None:
                row.update({
                    "kind": kind, "timestamp": ts_iso, "peak_ts": timestamp, "db_max": db_max,
                    "db_avg": db_avg, "trigger_type": trigger_type, "sound_class": sound_class,
                    "sound_category": sound_category, "sound_confidence": sound_confidence,
                    "sound_labels_json": fields["sound_labels_json"], "confidence": confidence,
                    "duration_ms": duration_ms, "clip_path": clip_path,
                })
                try:
                    cur = conn.execute(
                        f"INSERT INTO events ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})",
                        tuple(row.values()),
                    )
                except sqlite3.Error:
                    if kind != "both":
                        raise
                    # El `both` nacido en memoria no entró: se libera el pasaje
                    # y se reintenta como `noise` con el clip ya guardado (la
                    # unión contiene la ventana del ruido).
                    logger.exception(
                        "No se pudo insertar el both %s; se reintenta como noise",
                        f"del pasaje {passage.get('track_id')}" if passage is not None else "en escena",
                    )
                    if passage is not None:
                        passage["claimed"] = False
                    kind = "noise"
                    for key in ("direction", "speed_kmh", "detection_score", "track_id"):
                        row.pop(key, None)
                    row.update(noise_attribution, kind=kind)
                    try:
                        cur = conn.execute(
                            f"INSERT INTO events ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})",
                            tuple(row.values()),
                        )
                    except sqlite3.Error:
                        # Sin fila no hay quien referencie el clip: se borra
                        # y el worker sigue con el próximo disparo.
                        logger.exception(
                            "Tampoco entró como noise: evento perdido, se borra el clip %s", clip_path,
                        )
                        self.events_lost += 1
                        self._delete_replaced_clip(clip_path, "")
                        return
                event_id = cur.lastrowid
                if in_scene is not None and kind == "both":
                    scene["event_id"] = event_id
            if perceptual and trigger.get("trigger_decision"):
                conn.execute(
                    "INSERT INTO acoustic_trigger_decisions (event_table, event_id, decision_json) VALUES (?, ?, ?)",
                    ("events", event_id, json.dumps(trigger["trigger_decision"], allow_nan=False)),
                )
            conn.commit()
        finally:
            conn.close()
        self._last_event_ts = timestamp

        logger.info(
            "Event #%d (%s): %.1f dB, %s/%s (%s)%s @ %s",
            event_id,
            kind,
            db_max,
            sound_category or "?",
            sound_class or "?",
            culprit or "none",
            " (audio mudo)" if mute_audio else "",
            ts_iso,
        )

    # ------------------------------------------------------------------
    # Run loops
    # ------------------------------------------------------------------
    def _passage_loop(self):
        """Consume passage_queue from the tracker."""
        passage_queue = self.video_monitor.passage_queue
        while self._running:
            try:
                passage = passage_queue.get(timeout=0.5)
                self._process_passage(passage)
            except Empty:
                pass
            except Exception as e:
                logger.exception("Error procesando passage: %s", e)

    def _trigger_loop(self):
        """Consume trigger_queue from AudioMonitor.

        Los eventos psicoacústicos se derivan a un worker propio: guardar el
        clip incluye esperar el post-roll (~5 s) y no debe bloquear la cola
        de audio (los siguientes eventos vencerían encolados).
        """
        while self._running:
            try:
                trigger = self.audio_monitor.event_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                if trigger.get("type") == "psychoacoustic":
                    try:
                        self._perceptual_queue.put_nowait(trigger)
                    except Full:
                        self.perceptual_queue_full += 1
                        logger.warning("Cola psicoacústica llena; evento descartado")
                    continue
                self._process_trigger(trigger)
            except Exception as e:
                logger.exception("Error procesando trigger: %s", e)

    def _perceptual_loop(self):
        while self._running:
            try:
                trigger = self._perceptual_queue.get(timeout=0.5)
            except Empty:
                continue
            try:
                self._process_trigger(trigger)
            except Exception as e:
                logger.exception("Error procesando trigger psicoacústico: %s", e)

    def _require_events_table(self):
        conn = sqlite3.connect(self.db_path)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='events'"
            ).fetchone()
        finally:
            conn.close()
        if not exists:
            raise RuntimeError(
                f"{self.db_path} no tiene la tabla events: correr scripts/migrate_unified.py"
            )

    def run(self):
        """Start both consumer loops in separate threads."""
        # Sin la tabla cada evento fallaría en su INSERT y sólo quedaría en el log.
        self._require_events_table()
        self._running = True
        self._t_passage = threading.Thread(
            target=self._passage_loop, daemon=True, name="passage-consumer"
        )
        self._t_passage.start()
        self._t_perceptual = threading.Thread(
            target=self._perceptual_loop, daemon=True, name="perceptual-consumer"
        )
        self._t_perceptual.start()
        self._trigger_loop()  # run trigger loop in current thread

    def status(self):
        return {
            "discarded_stale": self.discarded_stale,
            "discarded_cooldown": self.discarded_cooldown,
            "perceptual_queue_full": self.perceptual_queue_full,
            "perceptual_queue_size": self._perceptual_queue.qsize(),
            "passages_not_recorded": self.passages_not_recorded,
            "discarded_without_traffic": self.discarded_without_traffic,
            "union_clip_fallbacks": self.union_clip_fallbacks,
            "events_lost": self.events_lost,
        }

    def stop(self):
        self._running = False
        thread = getattr(self, "_t_perceptual", None)
        if thread is not None:
            thread.join(timeout=2)
