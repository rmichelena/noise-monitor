"""
Monitor de audio: captura continua, A-weighting, detección de triggers.
Se comunica con EventManager vía queue cuando se dispara un trigger.
Integra AudioClassifier (YAMNet) para enriquecer triggers con clasificación de sonido.
Soporta modo device (micrófono real) y file (WAV para desarrollo simulado).
"""

import logging
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
import sounddevice as sd
from scipy.io import wavfile

from .utils import apply_a_weighting, load_config, resolve_path, rms_to_db

if TYPE_CHECKING:
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
        self.event_queue = event_queue or Queue()
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

        self.peak_db = float(triggers_cfg["peak_db"])
        self.sustained_db = float(triggers_cfg["sustained_db"])
        self.sustained_duration_ms = int(triggers_cfg["sustained_duration_ms"])
        self.correlation_db = float(triggers_cfg["correlation_db"])

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
        self._stream: sd.InputStream | None = None
        self._thread: threading.Thread | None = None

        # Nivel de fondo por minuto (tabla audio_levels): sirve para calibrar
        # umbrales con datos reales, incluidos los minutos sin trigger.
        self.db_path = str(resolve_path(self.config, "database", self.base_dir))
        self._level_minute: int | None = None
        self._level_vals: list[float] = []

        # Historial reciente de dB por bloque, para quemar el nivel en los
        # clips: (timestamp, dB). 30 s a 10.8 bloques/s.
        self._db_history: deque[tuple[float, float]] = deque(maxlen=600)

        # YAMNet corre en un thread aparte: clasificar dentro del callback de
        # PortAudio (~100 ms) provoca "input overflow" justo en el evento que
        # hay que documentar, perdiendo muestras.
        self._classify_queue: Queue = Queue(maxsize=8)
        self._classifier_thread: threading.Thread | None = None

    def db_at(self, timestamp: float, tolerance: float = 2.0) -> float | None:
        """dB del bloque de audio más cercano al instante dado."""
        if not self._db_history:
            return None
        best_ts, best_db = min(self._db_history, key=lambda item: abs(item[0] - timestamp))
        if abs(best_ts - timestamp) > tolerance:
            return None
        return best_db

    @property
    def current_db(self) -> float:
        """Nivel dB actual (thread-safe)."""
        with self._lock:
            return self._current_db

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
        self._db_history.append((timestamp, db))

        self._accumulate_level(db, timestamp)

        # Guardar en buffer (timestamp, bloque raw para clips)
        self.audio_buffer.append((timestamp, block.copy()))

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

            # YAMNet (~1 s de audio) se encola para el thread clasificador.
            if self._classifier:
                window = self._get_classification_window()
                if window is not None:
                    try:
                        self._classify_queue.put_nowait((event, window))
                    except Full:
                        logger.warning("Cola de clasificación llena; evento sin clasificar")
                        self.event_queue.put(event)
                else:
                    self.event_queue.put(event)
            else:
                self.event_queue.put(event)

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

    def _classify_loop(self):
        """Clasifica eventos fuera del callback de audio."""
        while True:
            item = self._classify_queue.get()
            if item is None:
                return
            event, window = item
            try:
                sc, cat, score = self._classifier.classify(
                    window, source_sr=self.sample_rate
                )
                event["sound_class"] = sc
                event["sound_category"] = cat
                event["sound_confidence"] = score
            except Exception:
                logger.exception("Error en clasificación YAMNet")
            self.event_queue.put(event)

    def _get_classification_window(self, duration_s: float = 1.0) -> np.ndarray | None:
        """Extract the last ~duration_s of audio from the buffer for YAMNet."""
        blocks_needed = max(1, int(duration_s * self.sample_rate / self.block_size))
        buf = list(self.audio_buffer)
        if len(buf) < blocks_needed:
            return None
        recent = buf[-blocks_needed:]
        return np.concatenate([b[1] for b in recent], axis=0)

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

        self._running = True

        if self._classifier and not self._classifier_thread:
            self._classifier_thread = threading.Thread(
                target=self._classify_loop, name="yamnet", daemon=True
            )
            self._classifier_thread.start()

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
        self._thread = None
        if self._classifier_thread:
            self._classify_queue.put(None)
            self._classifier_thread.join(timeout=5)
            self._classifier_thread = None
        self._flush_level_minute()
        logger.info("AudioMonitor detenido")
