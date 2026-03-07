"""
Event Manager: correlación audio-visión, asignación de culpable, guardado de clips.
Consume triggers del AudioMonitor, consulta VideoMonitor, guarda eventos en SQLite.
"""

import logging
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from queue import Empty

import cv2
import numpy as np
from scipy.io import wavfile

from .utils import init_database, load_config, resolve_path

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


def assign_culprit(detections: list[dict]) -> str:
    """Clase más probable de ser la fuente del ruido."""
    vehicle_dets = [d for d in detections if d.get("class") in NOISE_PRIORITY]
    if not vehicle_dets:
        return "unknown"

    sorted_dets = sorted(
        vehicle_dets,
        key=lambda d: NOISE_PRIORITY.get(d["class"], 99),
    )
    return sorted_dets[0]["class"]


def calculate_confidence(detections: list[dict]) -> str:
    """Nivel de confianza en la atribución."""
    vehicle_dets = [d for d in detections if d.get("class") in NOISE_PRIORITY]
    if len(vehicle_dets) == 0:
        return "low"
    if len(vehicle_dets) == 1:
        return "high"

    priorities = [NOISE_PRIORITY.get(d["class"], 99) for d in vehicle_dets]
    if min(priorities) < max(priorities):
        return "medium"
    return "low"


def get_lane(detections: list[dict]) -> str:
    """Carril predominante de las detecciones."""
    lanes = [d.get("lane") for d in detections if d.get("lane")]
    if not lanes:
        return "none"
    left = sum(1 for l in lanes if l == "left")
    right = sum(1 for l in lanes if l == "right")
    if left and right:
        return "both"
    return "left" if left else "right"


class EventManager:
    """
    Consume triggers de la cola, correlaciona con video, guarda clips y registra en DB.
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
        self.peak_db = float(self.config["triggers"]["peak_db"])
        self.correlation_db = float(self.config["triggers"]["correlation_db"])

        self.db_path = str(resolve_path(self.config, "database", self.base_dir))
        self.clips_dir = resolve_path(self.config, "clips", self.base_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

        init_database(self.config, self.base_dir)
        self._running = False

    def _save_clip(
        self,
        audio_blocks: list[tuple[float, np.ndarray]],
        video_frames: list[tuple[float, np.ndarray]],
        sample_rate: int,
        fps: int,
        resolution: tuple[int, int],
        event_timestamp: float,
    ) -> tuple[str | None, int]:
        """
        Guarda clip combinando audio y video con ffmpeg.
        Retorna (ruta_relativa, duration_ms) o (None, 0) si falla.
        """
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

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, fps, resolution)

        for _, frame in video_frames:
            out.write(frame)
        out.release()

        duration_ms = int(len(audio_data) / sample_rate * 1000)
        ts = datetime.fromtimestamp(event_timestamp).strftime("%Y%m%d_%H%M%S")
        clip_filename = f"event_{ts}.mp4"
        clip_path = self.clips_dir / clip_filename

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-i",
                    f_audio.name,
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-shortest",
                    str(clip_path),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("ffmpeg falló: %s", e.stderr.decode() if e.stderr else e)
            return None, 0
        finally:
            Path(f_audio.name).unlink(missing_ok=True)
            Path(video_path).unlink(missing_ok=True)

        return str(Path("data") / "clips" / clip_filename), duration_ms

    def _process_trigger(self, trigger: dict):
        """Procesa un trigger: correlación, guardado, inserción en DB."""
        trigger_type = trigger["type"]
        db_max = trigger["db"]
        db_avg = trigger.get("db_avg", db_max)
        timestamp = trigger["timestamp"]

        detections = self.video_monitor.recent_detections
        vehicle_dets = [d for d in detections if d.get("class") in NOISE_PRIORITY]

        # Confirmar: hay vehículo O es pico fuerte
        if not vehicle_dets and db_max < self.peak_db:
            if db_max < self.correlation_db:
                return  # Sin correlación y dB bajo, descartar
            # correlation_db: aceptar si hay algo de nivel

        culprit = assign_culprit(detections)
        confidence = calculate_confidence(detections)
        lane = get_lane(vehicle_dets) if vehicle_dets else "none"

        # Extraer ventana de buffers
        sample_rate = self.config["audio"]["sample_rate"]
        block_size = self.config["audio"]["block_size"]
        fps = self.config["video"]["framerate"]
        resolution = tuple(self.config["video"]["resolution"])

        blocks_per_second = sample_rate / block_size
        n_audio_blocks = int((self.pre_seconds + self.post_seconds) * blocks_per_second)
        n_video_frames = (self.pre_seconds + self.post_seconds) * fps

        audio_list = list(self.audio_monitor.audio_buffer)[-n_audio_blocks:]
        video_list = list(self.video_monitor.video_buffer)[-int(n_video_frames) :]

        clip_path, duration_ms = self._save_clip(
            audio_list, video_list, sample_rate, fps, resolution, timestamp
        )

        if clip_path is None:
            logger.warning("No se pudo guardar clip para evento")
            clip_path = ""
            duration_ms = 0

        ts_iso = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%dT%H:%M:%S")

        import sqlite3

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO events (timestamp, db_max, db_avg, vehicle_class, lane, confidence, clip_path, duration_ms, trigger_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts_iso, db_max, db_avg, culprit, lane, confidence, clip_path, duration_ms, trigger_type),
        )
        conn.commit()
        conn.close()

        logger.info(
            "Evento guardado: %s dB, %s, %s @ %s",
            round(db_max, 1),
            culprit,
            confidence,
            ts_iso,
        )

    def run(self):
        """Loop que consume la cola de triggers."""
        self._running = True
        while self._running:
            try:
                trigger = self.audio_monitor.event_queue.get(timeout=0.5)
                self._process_trigger(trigger)
            except Empty:
                pass
            except Exception as e:
                logger.exception("Error procesando trigger: %s", e)

    def stop(self):
        """Detiene el loop."""
        self._running = False
