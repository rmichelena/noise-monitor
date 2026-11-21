"""
Event Manager v3 — modelo unificado.

Todo se guarda en la tabla `events` con `kind`:
  traffic  vehículo del tracker, sin trigger de audio
  noise    trigger de audio sin vehículo
  both     ambos coinciden: una fila y un clip con la ventana unión
           start = min(inicio_ruido, inicio_vehículo)
           end   = max(fin_ruido, fin_vehículo)

El clip se genera una sola vez por evento; si llega la segunda mitad del
match se reescribe con la unión y se borra el anterior.
"""

import logging
import sqlite3
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
from scipy.io import wavfile

from .utils import init_database, load_config, resolve_path

logger = logging.getLogger(__name__)

# Clases de YAMNet que contienen voz humana: el audio del clip se elimina
# antes de publicarlo (privacidad de terceros en la vía pública).
SPEECH_CLASSES = {
    "Speech", "Child speech", "Conversation", "Narration",
    "Babbling", "Speech synthesizer", "Whispering",
}

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
    """Confianza en la atribución según cuántas clases compiten."""
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


class EventManager:
    """Consume las colas de pasajes y triggers y escribe filas unificadas."""

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
        self.traffic_clip_seconds = float(buffer_cfg.get("traffic_clip_seconds", 5))
        self.peak_db = float(self.config["triggers"]["peak_db"])
        self.correlation_db = float(self.config["triggers"]["correlation_db"])
        self.cooldown_seconds = float(self.config["triggers"].get("cooldown_seconds", 0))
        self._last_event_ts = 0.0

        self.db_path = str(resolve_path(self.config, "database", self.base_dir))
        self.clips_dir = resolve_path(self.config, "clips", self.base_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        init_database(self.config, self.base_dir)
        self._running = False

        # Eventos recientes para casar ruido con vehículo (en cualquier orden).
        self._recent: list[dict] = []
        self._lock = threading.Lock()
        self._TTL = 30.0

    # ------------------------------------------------------------------
    # Buffers y clips
    # ------------------------------------------------------------------
    def _slice(self, t0: float, t1: float):
        audio = [
            (ts, block)
            for ts, block in list(self.audio_monitor.audio_buffer)
            if t0 <= ts <= t1
        ]
        video = [
            (ts, frame)
            for ts, frame in list(self.video_monitor.video_buffer)
            if t0 <= ts <= t1
        ]
        return audio, video

    def _buffer_bounds(self) -> tuple[float, float] | None:
        audio = list(self.audio_monitor.audio_buffer)
        video = list(self.video_monitor.video_buffer)
        if not audio or not video:
            return None
        start = max(audio[0][0], video[0][0])
        end = min(audio[-1][0], video[-1][0])
        return (start, end) if start < end else None

    def _speech_in(self, audio_blocks) -> bool:
        classifier = getattr(self.audio_monitor, "_classifier", None)
        if classifier is None or not audio_blocks:
            return False
        try:
            window = np.concatenate([b for _, b in audio_blocks], axis=0)
            sound_class, _, _ = classifier.classify(
                window, source_sr=self.config["audio"]["sample_rate"]
            )
            return sound_class in SPEECH_CLASSES
        except Exception:
            logger.debug("Sin clasificación para el clip", exc_info=True)
            return False

    def _encode(self, t0: float, t1: float, ts: float, mute: bool,
                prefix: str, suffix: str = "") -> tuple[str, int]:
        bounds = self._buffer_bounds()
        if bounds is None:
            return "", 0
        t0 = max(t0, bounds[0])
        t1 = min(t1, bounds[1])
        if t1 - t0 < 0.5:
            return "", 0
        audio, video = self._slice(t0, t1)
        if len(video) < 5 or not audio:
            return "", 0
        path, duration = self._save_clip(
            audio, video, self.config["audio"]["sample_rate"], ts,
            mute_audio=mute, prefix=prefix, suffix=suffix,
        )
        return path or "", duration or 0

    def _save_clip(
        self,
        audio_blocks: list[tuple[float, np.ndarray]],
        video_frames: list[tuple[float, np.ndarray]],
        sample_rate: int,
        event_timestamp: float,
        mute_audio: bool = False,
        prefix: str = "event",
        suffix: str = "",
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

        video_cfg = self.config.get("video", {})
        overlay = bool(video_cfg.get("overlay_db", True))
        every = max(1, int(video_cfg.get("overlay_every_frames", 5)))
        db_val = None
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
        out.release()

        duration_ms = int(len(audio_data) / sample_rate * 1000)
        audio_span = audio_blocks[-1][0] - audio_blocks[0][0]
        logger.info(
            "Clip: audio %.2fs (%d bloques), video %.2fs (%d frames, %.1f fps)",
            audio_span, len(audio_blocks), span, len(video_frames), fps_out,
        )
        ts_tag = datetime.fromtimestamp(event_timestamp).strftime("%Y%m%d_%H%M%S")
        clip_filename = f"{prefix}_{ts_tag}{suffix}.mp4"
        clip_path = self.clips_dir / clip_filename

        try:
            cmd = ["ffmpeg", "-y", "-i", video_path]
            if not mute_audio:
                cmd += ["-i", f_audio.name]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
                    "-pix_fmt", "yuv420p"]
            if mute_audio:
                cmd += ["-an"]
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

        try:
            stored = clip_path.relative_to(self.base_dir)
        except ValueError:
            stored = clip_path
        return str(stored), duration_ms

    def _delete_clip(self, stored: str | None):
        if not stored:
            return
        path = Path(stored)
        if not path.is_absolute():
            path = self.base_dir / path
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("No se pudo borrar el clip viejo %s", path)

    # ------------------------------------------------------------------
    # Persistencia y matching
    # ------------------------------------------------------------------
    def _remember(self, row: dict):
        with self._lock:
            now = time.time()
            self._recent = [r for r in self._recent if now - r["end_ts"] < self._TTL]
            self._recent.append(row)

    def _update_recent(self, event_id: int, **fields):
        with self._lock:
            for r in self._recent:
                if r["id"] == event_id:
                    r.update(fields)

    def _best_overlap(self, start: float, end: float, kinds: tuple[str, ...]):
        with self._lock:
            best = None
            best_overlap = 0.0
            for r in self._recent:
                if r["kind"] not in kinds:
                    continue
                overlap = min(end, r["end_ts"]) - max(start, r["start_ts"])
                if overlap > best_overlap:
                    best, best_overlap = r, overlap
            return dict(best) if best else None

    def _insert(self, **f) -> int:
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            """
            INSERT INTO events
                (kind, start_ts, end_ts, peak_ts, timestamp, db_max, db_avg,
                 trigger_type, sound_class, sound_category, sound_confidence,
                 vehicle_class, direction, lane, speed_kmh, detection_score,
                 track_id, clip_path, audio_muted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f["kind"], f["start_ts"], f["end_ts"], f.get("peak_ts"),
                datetime.fromtimestamp(f["start_ts"]).strftime("%Y-%m-%dT%H:%M:%S"),
                f.get("db_max"), f.get("db_avg"), f.get("trigger_type"),
                f.get("sound_class"), f.get("sound_category"),
                f.get("sound_confidence"), f.get("vehicle_class"),
                f.get("direction"), f.get("lane"), f.get("speed_kmh"),
                f.get("detection_score"), f.get("track_id"),
                f.get("clip_path", ""), 1 if f.get("audio_muted") else 0,
            ),
        )
        event_id = cur.lastrowid
        conn.commit()
        conn.close()
        return event_id

    def _update(self, event_id: int, **f):
        allowed = {
            "kind", "start_ts", "end_ts", "peak_ts", "timestamp",
            "db_max", "db_avg", "trigger_type", "sound_class",
            "sound_category", "sound_confidence", "vehicle_class",
            "direction", "lane", "speed_kmh", "detection_score",
            "track_id", "clip_path", "audio_muted",
        }
        fields = {k: v for k, v in f.items() if k in allowed}
        if not fields:
            return
        sets = ", ".join(f"{k} = ?" for k in fields)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            f"UPDATE events SET {sets} WHERE id = ?",
            (*fields.values(), event_id),
        )
        conn.commit()
        conn.close()

    def _merge(self, event: dict, *, start: float, end: float,
               passage: dict | None = None, trigger: dict | None = None,
               culprit: str | None = None, lane: str | None = None):
        """Fusiona las dos mitades en una fila `both` con clip unión."""
        t0 = min(start, event["start_ts"])
        t1 = max(end, event["end_ts"])
        peak = (trigger or {}).get("timestamp") or event.get("peak_ts") or t0
        mute = (trigger or {}).get("sound_class") in SPEECH_CLASSES
        audio, _ = self._slice(t0, t1)
        mute = mute or self._speech_in(audio)

        suffix = f"_{int(t0 * 10) % 10000:04d}"
        clip, _ = self._encode(t0, t1, peak, mute, "event", suffix=suffix)
        old_clip = event.get("clip_path")
        if clip and old_clip and clip != old_clip:
            self._delete_clip(old_clip)
        elif not clip:
            clip = old_clip or ""

        db_max = event.get("db_max")
        if trigger and trigger.get("db") is not None:
            db_max = max(db_max or 0, trigger["db"])
        fields = {
            "kind": "both",
            "start_ts": t0,
            "end_ts": t1,
            "peak_ts": peak,
            "timestamp": datetime.fromtimestamp(t0).strftime("%Y-%m-%dT%H:%M:%S"),
            "db_max": db_max,
            "db_avg": (trigger or {}).get("db_avg", event.get("db_avg")),
            "trigger_type": (trigger or {}).get("type") or event.get("trigger_type"),
            "sound_class": (trigger or {}).get("sound_class") or event.get("sound_class"),
            "sound_category": (trigger or {}).get("sound_category") or event.get("sound_category"),
            "sound_confidence": (trigger or {}).get("sound_confidence") or event.get("sound_confidence"),
            "vehicle_class": (
                (passage or {}).get("vehicle_class")
                or event.get("vehicle_class")
                or (culprit if culprit not in (None, "unknown") else None)
            ),
            "direction": (passage or {}).get("direction") or event.get("direction"),
            "lane": (
                (passage or {}).get("lane")
                or event.get("lane")
                or (lane if lane not in (None, "none") else None)
            ),
            "speed_kmh": (passage or {}).get("speed_kmh") or event.get("speed_kmh"),
            "detection_score": (passage or {}).get("detection_score") or event.get("detection_score"),
            "track_id": (passage or {}).get("track_id") or event.get("track_id"),
            "clip_path": clip,
            "audio_muted": 1 if mute else 0,
        }
        self._update(event["id"], **fields)
        self._update_recent(event["id"], **fields)

        logger.info(
            "Evento #%d fusionado (%s): %s .. %s clip=%s%s",
            event["id"], fields["kind"],
            datetime.fromtimestamp(t0).strftime("%H:%M:%S"),
            datetime.fromtimestamp(t1).strftime("%H:%M:%S"),
            clip or "none", " (audio mudo)" if mute else "",
        )

    # ------------------------------------------------------------------
    # Pasajes (tráfico)
    # ------------------------------------------------------------------
    def _process_passage(self, passage: dict):
        enter = passage["timestamp_enter"]
        exit_ts = passage["timestamp_exit"]
        t1 = min(time.time(), exit_ts + 2.0)
        t0 = max(enter - 2.0, t1 - self.traffic_clip_seconds)

        # ¿Hay un evento de ruido solapado al que pertenezca?
        noise = self._best_overlap(enter, exit_ts, kinds=("noise",))
        if noise:
            self._merge(noise, start=min(t0, noise["start_ts"]),
                        end=max(t1, noise["end_ts"]), passage=passage)
            return

        audio, _ = self._slice(t0, t1)
        mute = self._speech_in(audio)
        suffix = f"_{int(enter * 10) % 10000:04d}"
        clip, dur = self._encode(t0, t1, exit_ts, mute, "traffic", suffix=suffix)
        event_id = self._insert(
            kind="traffic", start_ts=t0, end_ts=t1, peak_ts=None,
            db_max=None, db_avg=None, vehicle_class=passage["vehicle_class"],
            direction=passage.get("direction"), lane=passage.get("lane"),
            speed_kmh=passage.get("speed_kmh"),
            detection_score=passage.get("detection_score"),
            track_id=passage.get("track_id"), clip_path=clip,
            audio_muted=mute,
        )
        self._remember({
            "id": event_id, "kind": "traffic", "start_ts": t0, "end_ts": t1,
            "peak_ts": None, "clip_path": clip, "vehicle_class": passage["vehicle_class"],
            "direction": passage.get("direction"), "lane": passage.get("lane"),
            "speed_kmh": passage.get("speed_kmh"),
            "detection_score": passage.get("detection_score"),
            "track_id": passage.get("track_id"),
            "db_max": None, "db_avg": None, "trigger_type": None,
            "sound_class": None, "sound_category": None, "sound_confidence": None,
            "audio_muted": mute,
        })
        logger.info(
            "Evento #%d tráfico: %s (%s) %.1f km/h @ %s",
            event_id, passage["vehicle_class"], passage.get("lane", "?"),
            passage.get("speed_kmh") or 0.0,
            datetime.fromtimestamp(enter).strftime("%H:%M:%S"),
        )

    # ------------------------------------------------------------------
    # Triggers (ruido)
    # ------------------------------------------------------------------
    def _process_trigger(self, trigger: dict):
        trigger_type = trigger["type"]
        db_max = trigger["db"]
        db_avg = trigger.get("db_avg", db_max)
        timestamp = trigger["timestamp"]

        if self.cooldown_seconds > 0 and timestamp - self._last_event_ts < self.cooldown_seconds:
            return

        sound_class = trigger.get("sound_class")
        sound_category = trigger.get("sound_category", "other")
        sound_confidence = trigger.get("sound_confidence")

        # Esperar a que el post exista en los buffers.
        wait = (timestamp + self.post_seconds) - time.time()
        if wait > 0:
            time.sleep(min(wait, self.post_seconds))
        t_start = timestamp - self.pre_seconds
        t_end = timestamp + self.post_seconds

        window_dets = self.video_monitor.detections_between(t_start, t_end)
        if not window_dets:
            window_dets = self.video_monitor.recent_detections
        vehicle_dets = [d for d in window_dets if d.get("class") in NOISE_PRIORITY]

        culprit = None
        lane = "none"
        if sound_category == "vehicle" or (sound_category == "other" and vehicle_dets):
            culprit = assign_culprit(window_dets)
            lane = get_lane(vehicle_dets) if vehicle_dets else "none"

        if not vehicle_dets and db_max < self.peak_db and db_max < self.correlation_db:
            return

        # ¿Coincide con un pasaje ya registrado?
        traffic = self._best_overlap(t_start, t_end, kinds=("traffic",))
        if traffic:
            self._merge(traffic, start=min(t_start, traffic["start_ts"]),
                        end=max(t_end, traffic["end_ts"]), trigger=trigger,
                        culprit=culprit, lane=lane)
            self._last_event_ts = timestamp
            return

        mute = sound_class in SPEECH_CLASSES
        audio, _ = self._slice(t_start, t_end)
        mute = mute or self._speech_in(audio)
        suffix = f"_{int(timestamp * 10) % 10000:04d}"
        clip, dur = self._encode(t_start, t_end, timestamp, mute, "noise", suffix=suffix)

        event_id = self._insert(
            kind="noise", start_ts=t_start, end_ts=t_end, peak_ts=timestamp,
            db_max=db_max, db_avg=db_avg, trigger_type=trigger_type,
            sound_class=sound_class, sound_category=sound_category,
            sound_confidence=sound_confidence, vehicle_class=culprit,
            direction=None, lane=lane, speed_kmh=None, detection_score=None,
            track_id=None, clip_path=clip, audio_muted=mute,
        )
        self._remember({
            "id": event_id, "kind": "noise", "start_ts": t_start, "end_ts": t_end,
            "peak_ts": timestamp, "clip_path": clip, "db_max": db_max,
            "db_avg": db_avg, "trigger_type": trigger_type,
            "sound_class": sound_class, "sound_category": sound_category,
            "sound_confidence": sound_confidence, "vehicle_class": culprit,
            "direction": None, "lane": lane, "speed_kmh": None,
            "detection_score": None, "track_id": None, "audio_muted": mute,
        })
        self._last_event_ts = timestamp

        logger.info(
            "Evento #%d ruido: %.1f dB, %s/%s (%s) clip=%s%s @ %s",
            event_id, db_max, sound_category or "?", sound_class or "?",
            culprit or "none", clip or "none",
            " (audio mudo)" if mute else "",
            datetime.fromtimestamp(timestamp).strftime("%H:%M:%S"),
        )

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------
    def _passage_loop(self):
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
        while self._running:
            try:
                trigger = self.audio_monitor.event_queue.get(timeout=0.5)
                self._process_trigger(trigger)
            except Empty:
                pass
            except Exception as e:
                logger.exception("Error procesando trigger: %s", e)

    def run(self):
        self._running = True
        self._t_passage = threading.Thread(
            target=self._passage_loop, daemon=True, name="passage-consumer"
        )
        self._t_passage.start()
        self._trigger_loop()

    def stop(self):
        self._running = False
