"""
Monitor de audio: captura continua, A-weighting, detección de triggers.
Se comunica con EventManager vía queue cuando se dispara un trigger.
"""

import logging
import threading
from collections import deque
from queue import Queue

import numpy as np
import sounddevice as sd

from .utils import apply_a_weighting, load_config, rms_to_db

logger = logging.getLogger(__name__)


class AudioMonitor:
    """
    Captura audio continuo, aplica A-weighting, detecta triggers (pico/sostenido).
    Expone buffer circular y current_db para el EventManager.
    """

    def __init__(self, config: dict | None = None, event_queue: Queue | None = None):
        self.config = config or load_config()
        self.event_queue = event_queue or Queue()

        audio_cfg = self.config["audio"]
        triggers_cfg = self.config["triggers"]
        buffer_cfg = self.config["buffer"]

        self.sample_rate = int(audio_cfg["sample_rate"])
        self.block_size = int(audio_cfg["block_size"])
        self.device_index = audio_cfg.get("device_index")
        self.calibration_offset = float(audio_cfg.get("calibration_offset", 0.0))

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

    @property
    def current_db(self) -> float:
        """Nivel dB actual (thread-safe)."""
        with self._lock:
            return self._current_db

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            logger.warning("Audio status: %s", status)

        # Mono: flatten si viene (frames, 1)
        block = np.squeeze(indata.copy()).astype(np.float32)
        if block.ndim > 1:
            block = block.mean(axis=1)

        timestamp = time_info.inputBufferAdcTime

        # A-weighting
        filtered = apply_a_weighting(block, self.sample_rate)

        # RMS → dB
        rms = np.sqrt(np.mean(filtered**2))
        db = rms_to_db(float(rms)) + self.calibration_offset

        with self._lock:
            self._current_db = db

        # Guardar en buffer (timestamp, bloque raw para clips)
        self.audio_buffer.append((timestamp, block.copy()))

        # Evaluar triggers
        if db >= self.peak_db:
            self.event_queue.put(
                {"type": "peak", "db": db, "timestamp": timestamp, "db_avg": db}
            )
            self._sustained_blocks = 0
        elif db >= self.sustained_db:
            self._sustained_blocks += 1
            if self._sustained_blocks >= self._sustained_blocks_needed:
                self.event_queue.put(
                    {
                        "type": "sustained",
                        "db": db,
                        "timestamp": timestamp,
                        "db_avg": db,
                    }
                )
                self._sustained_blocks = 0
        else:
            self._sustained_blocks = max(0, self._sustained_blocks - 1)

    def start(self):
        """Inicia captura en un thread."""
        if self._running:
            return

        self._running = True
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
        logger.info("AudioMonitor detenido")
