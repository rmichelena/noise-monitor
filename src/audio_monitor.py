"""
Monitor de audio: captura continua, A-weighting, detección de triggers.
Se comunica con EventManager vía queue cuando se dispara un trigger.
Integra AudioClassifier (YAMNet) para enriquecer triggers con clasificación de sonido.
Soporta modo device (micrófono real) y file (WAV para desarrollo simulado).
"""

import logging
import math
import os
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Full, Queue
from typing import TYPE_CHECKING

import numpy as np
from scipy.io import wavfile

from .perceptual_trigger import PerceptualTrigger, Settings
from .utils import MIN_DB, apply_a_weighting, load_config, resolve_path, rms_to_db

if TYPE_CHECKING:
    import sounddevice as sd

    from .audio_classifier import AudioClassifier

logger = logging.getLogger(__name__)


class AudioMonitor:
    """
    Captura audio continuo, aplica A-weighting, detecta triggers (pico/sostenido).
    Expone buffer circular y current_db para el EventManager.
    """

    def __init__(
        self,
        config: dict | None = None,
        event_queue: Queue | None = None,
        base_dir: Path | None = None,
        audio_classifier: "AudioClassifier | None" = None,
    ):
        self.config = config or load_config()
        settings = Settings.load(self.config)
        psycho = (self.config.get("triggers") or {}).get("psychoacoustic") or {}
        try:
            queue_size = int(psycho.get("queue_size") or 8)
        except (TypeError, ValueError):
            queue_size = 8
        queue_size = max(1, min(64, queue_size))
        self.event_queue = event_queue or Queue(maxsize=queue_size if settings else 0)
        self.base_dir = Path(base_dir or os.getcwd())
        self._classifier = audio_classifier

        audio_cfg = self.config["audio"]
        triggers_cfg = self.config["triggers"]
        buffer_cfg = self.config["buffer"]

        self.sample_rate = int(audio_cfg["sample_rate"])
        self.block_size = int(audio_cfg["block_size"])
        self.device_index = audio_cfg.get("device_index")
        self.calibration_offset = float(audio_cfg.get("calibration_offset", 0.0))

        # Modo: "device" = micrófono, "file" = WAV (modo simulado)
        debug_cfg = self.config.get("debug", {})
        audio_from_video = bool(debug_cfg.get("audio_from_video", False))
        video_source = self.config.get("video", {}).get("source")

        if audio_from_video and isinstance(video_source, str):
            # Derivar ruta de audio: video.mp4 -> video_audio.wav
            video_path = Path(video_source)
            if not video_path.is_absolute():
                video_path = self.base_dir / video_path
            audio_path = video_path.with_stem(video_path.stem + "_audio").with_suffix(".wav")
            file_path = str(audio_path.resolve())
        else:
            file_path = audio_cfg.get("file_path")

        if audio_from_video or audio_cfg.get("source") == "file":
            self.audio_source = "file"
        else:
            self.audio_source = str(audio_cfg.get("source", "device")).lower()

        if file_path:
            path = Path(file_path)
            if not path.is_absolute():
                path = self.base_dir / path
            self.audio_file_path = str(path.resolve())
        else:
            self.audio_file_path = None

        if settings is None:
            self.peak_db = float(triggers_cfg["peak_db"])
            self.sustained_db = float(triggers_cfg["sustained_db"])
            self.sustained_duration_ms = int(triggers_cfg["sustained_duration_ms"])
            self.correlation_db = float(triggers_cfg["correlation_db"])
        else:
            self.sustained_duration_ms = 0

        audio_seconds = int(buffer_cfg["audio_seconds"])
        blocks_per_second = self.sample_rate / self.block_size
        self.buffer_maxlen = int(audio_seconds * blocks_per_second)

        self.audio_buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=self.buffer_maxlen)
        self._current_db = 0.0
        self._lock = threading.Lock()

        # Sustained: bloques consecutivos sobre umbral
        self._sustained_blocks = 0
        self._sustained_blocks_needed = max(
            1, int(self.sustained_duration_ms / 1000.0 * blocks_per_second)
        )

        self._running = False
        self._event_queue_warned = False
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None

        # Nivel de fondo por minuto (tabla audio_levels): sirve para calibrar
        # umbrales con datos reales, incluidos los minutos sin trigger.
        self.db_path = str(resolve_path(self.config, "database", self.base_dir))
        self._level_minute: int | None = None
        self._level_vals: list[float] = []

        # Historial reciente de dB por bloque, para calcular el nivel de los
        # clips: (timestamp, dB). Mismo horizonte que audio_buffer: si un
        # bloque sobrevive para el clip, su nivel también. Un margen fijo
        # podía perder bloques cuando el worker acumula latencia en ráfagas.
        self._db_history: deque[tuple[float, float]] = deque(maxlen=max(600, self.buffer_maxlen))

        self._perceptual_trigger = (
            PerceptualTrigger(settings, self.sample_rate, self.block_size, self.event_queue.put_nowait)
            if settings else None
        )

    def db_at(self, timestamp: float, tolerance: float = 2.0) -> float | None:
        """dB del bloque de audio más cercano al instante dado."""
        with self._lock:
            history = list(self._db_history)
        if not history:
            return None
        best_ts, best_db = min(history, key=lambda item: abs(item[0] - timestamp))
        if abs(best_ts - timestamp) > tolerance:
            return None
        return best_db

    def level_snapshot(self) -> list[tuple[float, float]]:
        """Copia inmutable del historial de niveles para un evento."""
        with self._lock:
            return list(self._db_history)

    def level_stats(
        self,
        blocks: list[tuple[float, np.ndarray]],
        history: list[tuple[float, float]] | None = None,
    ) -> dict | None:
        """Máximo y promedio dB(A) de los bloques que forman el clip.

        Usa el historial ya calculado (mismo A-weighting y offset que el
        medidor y los triggers), restringido a los timestamps del clip, para
        no recalcular ni abrir una segunda ruta de medición.
        """
        if not blocks:
            return None
        if history is None:
            history = self.level_snapshot()
        # Multimapa: dos bloques pueden compartir timestamp (p. ej. modo
        # archivo); un dict simple perdía uno en silencio.
        by_timestamp: dict[float, list[float]] = {}
        for ts, db in history:
            by_timestamp.setdefault(ts, []).append(db)
        values = []
        for ts, _ in blocks:
            pending = by_timestamp.get(ts)
            if pending:
                values.append(pending.pop(0))
        if not values:
            return None
        coverage = len(values) / len(blocks)
        if coverage < 0.8:
            logger.warning(
                "Historial de niveles insuficiente para el clip: %d de %d bloques (%.0f%%)",
                len(values), len(blocks), coverage * 100,
            )
            return None
        if coverage < 1:
            logger.warning(
                "Historial de niveles incompleto para el clip: %d de %d bloques",
                len(values), len(blocks),
            )
        # Promedio energético (equivalente a LAeq sobre los bloques) y sin el
        # piso de silencio digital, que de otro modo hunde la media. El piso
        # incluye el offset de calibración, porque db ya lo lleva aplicado.
        floor = MIN_DB + self.calibration_offset
        audible = [value for value in values if value > floor]
        average_source = audible or values
        energy = sum(10 ** (value / 10) for value in average_source) / len(average_source)
        return {"max": max(values), "avg": 10 * math.log10(energy), "count": len(values)}

    @property
    def current_db(self) -> float:
        """Nivel dB actual (thread-safe)."""
        with self._lock:
            return self._current_db

    def trigger_status(self):
        status = (
            self._perceptual_trigger.status()
            if self._perceptual_trigger
            else {"criterion": "dba", "peak_db": self.peak_db, "sustained_db": self.sustained_db}
        )
        manager = getattr(self, "event_manager", None)
        if manager is not None:
            status.update(manager.status())
        return status

    def classifier_status(self):
        if self._classifier is None:
            return {"active_backend": "disabled"}
        return self._classifier.status()

    def _process_block(self, block: np.ndarray, timestamp: float):
        """Procesa un bloque de audio: A-weighting, buffer, triggers."""
        # Mono: flatten si viene (frames, 1)
        block = np.squeeze(block).astype(np.float32)
        if block.ndim > 1:
            block = block.mean(axis=1)

        # A-weighting
        filtered = apply_a_weighting(block, self.sample_rate)

        # RMS → dB
        rms = np.sqrt(np.mean(filtered**2))
        db = rms_to_db(float(rms)) + self.calibration_offset

        with self._lock:
            self._current_db = db
            # El append va bajo el mismo lock que los lectores: iterar el
            # deque mientras el callback lo modifica puede lanzar
            # RuntimeError y tirar el evento completo.
            self._db_history.append((timestamp, db))

        self._accumulate_level(db, timestamp)

        # Guardar en buffer (timestamp, bloque raw para clips)
        buffered = block.copy()
        self.audio_buffer.append((timestamp, buffered))

        if self._perceptual_trigger:
            self._perceptual_trigger.offer(timestamp, buffered, db)
            return

        # Evaluar triggers
        triggered = False
        trigger_type = None

        if db >= self.peak_db:
            triggered = True
            trigger_type = "peak"
            self._sustained_blocks = 0
        elif db >= self.sustained_db:
            self._sustained_blocks += 1
            if self._sustained_blocks >= self._sustained_blocks_needed:
                triggered = True
                trigger_type = "sustained"
                self._sustained_blocks = 0
        else:
            self._sustained_blocks = max(0, self._sustained_blocks - 1)

        if triggered:
            event = {"type": trigger_type, "db": db, "timestamp": timestamp, "db_avg": db}
            try:
                self.event_queue.put_nowait(event)
            except Full:
                if not self._event_queue_warned:
                    logger.warning("Cola de eventos llena; se descartó un disparo dBA")
                    self._event_queue_warned = True

    def classify_event_clip(self, blocks, timestamp: float, *, scan_all_speech: bool):
        """Clasifica el audio del evento a partir de la ventana ya capturada."""
        if self._classifier is None or not blocks:
            if scan_all_speech and self._classifier is None and not getattr(self, "_no_classifier_warned", False):
                logger.warning(
                    "Sin clasificador YAMNet: los clips psicoacústicos se guardan sin audio"
                )
                self._no_classifier_warned = True
            return None, "other", None, scan_all_speech, []

        # Los timestamps marcan el final de cada bloque. Traducir el disparo
        # a una muestra del clip mantiene el ancla aunque falte pre-roll.
        anchor = 0
        for end, block in blocks:
            if end >= timestamp:
                anchor += max(0, len(block) - round((end - timestamp) * self.sample_rate))
                break
            anchor += len(block)
        try:
            audio = np.concatenate([block for _, block in blocks])
            labels, speech = self._classifier.classify_event(
                audio, self.sample_rate, anchor, scan_all_speech=scan_all_speech,
            )
        except Exception:
            logger.exception("No se pudo clasificar el audio del evento")
            return None, "other", None, True, []
        if labels and labels[0]["passes_threshold"]:
            best = labels[0]
            return best["label"], best["category"], best["score"], speech, labels
        return "unknown", "other", labels[0]["score"] if labels else None, speech, labels

    def _accumulate_level(self, db: float, timestamp: float):
        """Agrega el nivel del bloque al minuto en curso; persiste al cambiar."""
        minute = int(timestamp // 60)
        if self._level_minute is None:
            self._level_minute = minute
        elif minute != self._level_minute:
            self._flush_level_minute()
            self._level_minute = minute
            self._level_vals = []
        self._level_vals.append(db)

    def _flush_level_minute(self):
        """Escribe el resumen del minuto cerrado en audio_levels."""
        vals = self._level_vals
        if not vals or self._level_minute is None:
            return
        label = datetime.fromtimestamp(self._level_minute * 60).strftime("%Y-%m-%dT%H:%M")
        try:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute(
                "INSERT OR REPLACE INTO audio_levels (minute, db_avg, db_max, db_p95, blocks)"
                " VALUES (?, ?, ?, ?, ?)",
                (
                    label,
                    sum(vals) / len(vals),
                    max(vals),
                    float(np.percentile(vals, 95)),
                    len(vals),
                ),
            )
            conn.commit()
            conn.close()
        except Exception:
            logger.exception("No se pudo guardar nivel de audio del minuto %s", label)
        finally:
            # El minuto ya está persistido: si no se limpia, el mismo parcial
            # se reescribe en cada flush posterior (stop, reinicio).
            self._level_vals = []
            self._level_minute = None


    def _audio_callback(self, indata, frames, time_info, status):
        """Callback de sounddevice (modo device)."""
        if status:
            logger.warning("Audio status: %s", status)
        # time_info.inputBufferAdcTime es relativo al arranque del stream en ALSA
        # (epoch 1970 y desalineado con video). Usar reloj de pared.
        timestamp = time.time()
        self._process_block(indata.copy(), timestamp)

    def _file_reader_loop(self):
        """Lee WAV en chunks simulando captura en tiempo real (modo file)."""
        if not self.audio_file_path or not Path(self.audio_file_path).exists():
            logger.error("Archivo de audio no encontrado: %s", self.audio_file_path)
            return

        sr, data = wavfile.read(self.audio_file_path)
        if sr != self.sample_rate:
            logger.error(
                "WAV tiene %d Hz, se requiere %d Hz. Re-extraer con: "
                "ffmpeg -i video.mp4 -vn -ar %d -ac 1 -f wav output.wav",
                sr,
                self.sample_rate,
                self.sample_rate,
            )
            return

        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max

        if data.ndim > 1:
            data = data.mean(axis=1)

        if len(data) < self.block_size:
            logger.error(
                "Archivo de audio demasiado corto: %d muestras, se requiere al menos %d (block_size). "
                "Duración mínima: %.2f s",
                len(data),
                self.block_size,
                self.block_size / self.sample_rate,
            )
            return

        block_duration = self.block_size / self.sample_rate
        pos = 0
        start_time = time.time()

        while self._running:
            chunk = data[pos : pos + self.block_size]
            if len(chunk) < self.block_size:
                # Fin del archivo: loop al inicio
                pos = 0
                chunk = data[pos : pos + self.block_size]
                if len(chunk) < self.block_size:
                    logger.error(
                        "Chunk incompleto tras loop (%d < %d muestras). Archivo corrupto?",
                        len(chunk),
                        self.block_size,
                    )
                    break

            pos += self.block_size
            if pos >= len(data):
                pos = 0

            elapsed = time.time() - start_time
            timestamp = start_time + elapsed
            self._process_block(chunk.astype(np.float32), timestamp)
            time.sleep(block_duration)

    def start(self):
        """Inicia captura (device o file según config)."""
        if self._running:
            return

        if self._perceptual_trigger:
            self._perceptual_trigger.start()
        self._running = True

        if self.audio_source == "file" and self.audio_file_path:
            self._thread = threading.Thread(target=self._file_reader_loop, daemon=True)
            self._thread.start()
            logger.info(
                "AudioMonitor iniciado (file): %s, %d Hz, block=%d",
                self.audio_file_path,
                self.sample_rate,
                self.block_size,
            )
        else:
            import sounddevice as sd

            self._stream = sd.InputStream(
                device=self.device_index,
                channels=1,
                dtype="float32",
                blocksize=self.block_size,
                samplerate=self.sample_rate,
                callback=self._audio_callback,
            )
            self._stream.start()
            logger.info(
                "AudioMonitor iniciado: device=%s, %d Hz, block=%d",
                self.device_index,
                self.sample_rate,
                self.block_size,
            )

    def stop(self):
        """Detiene captura."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._perceptual_trigger:
            self._perceptual_trigger.stop()
        # Sin join, `start()` podía lanzar un segundo lector mientras el
        # anterior seguía en su último `sleep`: dos hilos procesando el mismo
        # WAV duplicaba niveles, triggers y clips.
        if self._thread:
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning("El lector de audio no terminó en 5 s")
            self._thread = None
        self._flush_level_minute()
        logger.info("AudioMonitor detenido")
