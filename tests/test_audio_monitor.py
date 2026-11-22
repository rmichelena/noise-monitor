#!/usr/bin/env python3
"""
AudioMonitor sin hardware: silencio acotado, triggers y parada del lector.

- Un bloque de silencio (RMS 0) daba -inf y ese valor llegaba a SQLite y al
  JSON del dashboard: ahora se acota a MIN_DB.
- `stop()` no esperaba al hilo lector: un `start()` inmediato podía lanzar un
  segundo lector sobre el mismo WAV (niveles y clips duplicados).
"""

import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from queue import Empty, Queue

import numpy as np
from scipy.io import wavfile

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # PortAudio no está instalado en todos los hosts (ni en CI)
    import sounddevice  # noqa: F401
except Exception:  # pragma: no cover - depende del host
    _stub = types.ModuleType("sounddevice")
    _stub.InputStream = object  # type: ignore[attr-defined]
    sys.modules["sounddevice"] = _stub

from support import example_config  # noqa: E402
from src.audio_monitor import AudioMonitor  # noqa: E402
from src.utils import MIN_DB, init_database  # noqa: E402

SAMPLE_RATE = 44100
BLOCK = 4096


def tone(amplitude: float, freq: float = 1000.0, size: int = BLOCK) -> np.ndarray:
    t = np.arange(size, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class MonitorBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.cfg = example_config(self.base)
        self.cfg["audio"]["sample_rate"] = SAMPLE_RATE
        self.cfg["audio"]["block_size"] = BLOCK
        init_database(self.cfg, PROJECT_DIR)
        self.queue: Queue = Queue()
        self.monitor = AudioMonitor(
            config=self.cfg, event_queue=self.queue, base_dir=PROJECT_DIR
        )

    def tearDown(self):
        self.monitor.stop()
        self.tmp.cleanup()


class ProcessBlockTest(MonitorBase):
    def test_silence_is_floored_and_does_not_trigger(self):
        self.monitor._process_block(np.zeros(BLOCK, dtype=np.float32), timestamp=1000.0)
        self.assertEqual(self.monitor.current_db, MIN_DB)
        with self.assertRaises(Empty):
            self.queue.get(timeout=0.05)

    def test_current_db_is_always_finite(self):
        for block in (
            np.zeros(BLOCK, dtype=np.float32),
            tone(0.0),
            np.full(BLOCK, 1e-9, dtype=np.float32),
        ):
            self.monitor._process_block(block, timestamp=1000.0)
            self.assertTrue(np.isfinite(self.monitor.current_db))
            self.assertGreaterEqual(self.monitor.current_db, MIN_DB)

    def test_peak_trigger_reports_the_measured_level(self):
        self.monitor.peak_db = 80.0
        self.monitor.sustained_db = 95.0
        self.monitor._process_block(tone(0.7), timestamp=1000.0)
        event = self.queue.get(timeout=0.2)
        self.assertEqual(event["type"], "peak")
        # 1 kHz no se atenúa: rms 0.495 → ~87.9 dB SPL
        self.assertAlmostEqual(event["db"], 87.9, delta=0.6)
        self.assertEqual(event["db_avg"], event["db"])

    def test_sustained_trigger_needs_consecutive_blocks(self):
        self.monitor.peak_db = 95.0
        self.monitor.sustained_db = 65.0
        self.monitor.sustained_duration_ms = 250  # 3 bloques
        self.monitor._sustained_blocks_needed = 3

        for i in range(2):
            self.monitor._process_block(tone(0.1), timestamp=1000.0 + i * 0.093)
        with self.assertRaises(Empty):
            self.queue.get(timeout=0.05)

        self.monitor._process_block(tone(0.1), timestamp=1000.3)
        event = self.queue.get(timeout=0.2)
        self.assertEqual(event["type"], "sustained")

    def test_block_below_thresholds_resets_sustained_counter(self):
        self.monitor.peak_db = 95.0
        self.monitor.sustained_db = 65.0
        self.monitor._sustained_blocks_needed = 3
        self.monitor._process_block(tone(0.1), timestamp=1000.0)
        self.monitor._process_block(tone(0.1), timestamp=1000.1)
        self.monitor._process_block(np.zeros(BLOCK, dtype=np.float32), timestamp=1000.2)
        self.assertLess(self.monitor._sustained_blocks, 2)

    def test_stereo_block_is_mixed_down(self):
        stereo = np.stack([tone(0.5), tone(0.5)], axis=1)
        self.monitor._process_block(stereo, timestamp=1000.0)
        self.assertTrue(np.isfinite(self.monitor.current_db))

    def test_db_at_returns_nearest_block_within_tolerance(self):
        self.monitor._process_block(tone(0.2), timestamp=1000.0)
        self.assertIsNotNone(self.monitor.db_at(1000.4))
        self.assertIsNone(self.monitor.db_at(1010.0))

    def test_level_minute_is_written_once(self):
        import sqlite3

        self.monitor._process_block(tone(0.2), timestamp=60.0)
        self.monitor._process_block(tone(0.2), timestamp=61.0)
        self.monitor._flush_level_minute()
        self.assertIsNone(self.monitor._level_minute)
        self.assertEqual(self.monitor._level_vals, [])

        self.monitor._flush_level_minute()  # sin valores: no reescribe

        conn = sqlite3.connect(self.cfg["paths"]["database"])
        rows = conn.execute("SELECT minute, blocks FROM audio_levels").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], 2)

    def test_minute_change_flushes_the_previous_minute(self):
        import sqlite3

        self.monitor._process_block(tone(0.2), timestamp=60.0)
        self.monitor._process_block(tone(0.2), timestamp=121.0)  # otro minuto

        conn = sqlite3.connect(self.cfg["paths"]["database"])
        rows = conn.execute("SELECT minute, blocks FROM audio_levels ORDER BY minute").fetchall()
        conn.close()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0][1], 1)


class FileReaderLifecycleTest(MonitorBase):
    def setUp(self):
        super().setUp()
        self.wav = self.base / "clip.wav"
        data = tone(0.3, size=SAMPLE_RATE).astype(np.float32)  # 1 s
        wavfile.write(self.wav, SAMPLE_RATE, data)
        self.cfg["audio"]["source"] = "file"
        self.cfg["audio"]["file_path"] = str(self.wav)
        self.monitor = AudioMonitor(
            config=self.cfg, event_queue=self.queue, base_dir=PROJECT_DIR
        )
        self.calls: list[float] = []
        original = self.monitor._process_block

        def counting(block, timestamp):
            self.calls.append(time.time())
            return original(block, timestamp)

        self.monitor._process_block = counting  # type: ignore[method-assign]

    def test_stop_waits_for_the_reader_thread(self):
        self.monitor.start()
        time.sleep(0.3)
        self.monitor.stop()

        self.assertIsNone(self.monitor._thread)
        processed = len(self.calls)
        self.assertGreater(processed, 0)

        time.sleep(0.3)
        self.assertEqual(len(self.calls), processed, "el lector siguió tras stop()")

    def test_restart_resumes_processing(self):
        self.monitor.start()
        time.sleep(0.25)
        self.monitor.stop()
        first = len(self.calls)

        self.monitor.start()
        time.sleep(0.25)
        self.monitor.stop()
        self.assertGreater(len(self.calls), first)

    def test_only_one_reader_thread_alive(self):
        self.monitor.start()
        time.sleep(0.2)
        readers = [t for t in threading.enumerate()
                   if t is not threading.main_thread() and t.is_alive()
                   and t.name.startswith("Thread-")]
        self.assertLessEqual(len(readers), 1)
        self.monitor.stop()


if __name__ == "__main__":
    unittest.main()
