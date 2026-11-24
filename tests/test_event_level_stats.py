import math
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from src.audio_monitor import AudioMonitor
from src.event_manager import EventManager
from src.utils import MIN_DB, load_config

ROOT = Path(__file__).resolve().parents[1]


def energy_mean(values):
    return 10 * math.log10(sum(10 ** (value / 10) for value in values) / len(values))


def example_config(directory: str) -> dict:
    config = load_config(ROOT / "config.example.yaml")
    config["paths"].update(database=str(Path(directory) / "events.db"), clips=directory)
    config["audio"].update(sample_rate=16000, block_size=1600)
    config["buffer"].update(pre_event_seconds=1, post_event_seconds=1)
    config["triggers"].update(peak_db=40, correlation_db=65, cooldown_seconds=0)
    return config


class LevelStatsTests(unittest.TestCase):
    def test_only_counts_blocks_that_belong_to_the_clip(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = AudioMonitor(config=example_config(directory), base_dir=ROOT)
            audio._db_history.extend([
                (1.0, 40.0), (1.1, 50.0), (1.2, 30.0), (50.0, 90.0),
            ])
            stats = audio.level_stats([(1.0, None), (1.1, None), (1.2, None)])
            self.assertEqual(stats["max"], 50.0)
            self.assertAlmostEqual(stats["avg"], energy_mean([40.0, 50.0, 30.0]))
            self.assertEqual(stats["count"], 3)

    def test_silence_floor_does_not_sink_the_average(self):
        with tempfile.TemporaryDirectory() as directory:
            config = example_config(directory)
            # Con offset != 0 el piso calibrado deja de coincidir con MIN_DB.
            config["audio"]["calibration_offset"] = 4.0
            audio = AudioMonitor(config=config, base_dir=ROOT)
            self.assertEqual(audio.calibration_offset, 4.0)
            floor = MIN_DB + audio.calibration_offset
            audio._db_history.extend([(1.0, floor), (1.1, 40.0), (1.2, 50.0)])
            stats = audio.level_stats([(1.0, None), (1.1, None), (1.2, None)])
            self.assertEqual(stats["max"], 50.0)
            self.assertAlmostEqual(stats["avg"], energy_mean([40.0, 50.0]))

    def test_insufficient_history_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = AudioMonitor(config=example_config(directory), base_dir=ROOT)
            audio._db_history.append((1.0, 40.0))
            with self.assertLogs("src.audio_monitor", level="WARNING") as logs:
                stats = audio.level_stats([(1.0, None), (1.1, None)])
            self.assertIsNone(stats)
            self.assertTrue(any("insuficiente" in line for line in logs.output))

    def test_partial_history_is_used_with_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = AudioMonitor(config=example_config(directory), base_dir=ROOT)
            audio._db_history.extend((i / 10, 40.0) for i in range(9))
            blocks = [(i / 10, None) for i in range(10)]
            with self.assertLogs("src.audio_monitor", level="WARNING") as logs:
                stats = audio.level_stats(blocks)
            self.assertEqual(stats["count"], 9)
            self.assertTrue(any("incompleto" in line for line in logs.output))

    def test_duplicate_timestamps_keep_both_values(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = AudioMonitor(config=example_config(directory), base_dir=ROOT)
            audio._db_history.extend([(1.0, 40.0), (1.0, 50.0)])
            stats = audio.level_stats([(1.0, None), (1.0, None)])
            self.assertEqual(stats["max"], 50.0)
            self.assertAlmostEqual(stats["avg"], energy_mean([40.0, 50.0]))

    def test_history_lifetime_matches_audio_buffer(self):
        with tempfile.TemporaryDirectory() as directory:
            config = example_config(directory)
            # 120 s supera el piso de 600 bloques: el test distingue el
            # dimensionado por buffer del maxlen fijo anterior.
            config["buffer"]["audio_seconds"] = 120
            audio = AudioMonitor(config=config, base_dir=ROOT)
            needed = 120 * audio.sample_rate / audio.block_size
            self.assertGreaterEqual(audio._db_history.maxlen, needed)
            self.assertEqual(audio._db_history.maxlen, audio.buffer_maxlen)

    def test_missing_history_or_blocks_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = AudioMonitor(config=example_config(directory), base_dir=ROOT)
            self.assertIsNone(audio.level_stats([]))
            self.assertIsNone(audio.level_stats([(1.0, None)]))


class EventPersistenceTests(unittest.TestCase):
    def _event(self, directory: str, *, history: bool):
        config = example_config(directory)
        audio = AudioMonitor(config=config, base_dir=ROOT)
        classifier = Mock()
        classifier.classify_event.return_value = ([], False)
        audio._classifier = classifier
        stamp = time.time() - 10
        for i in range(-10, 11):
            ts = stamp + i / 10
            audio.audio_buffer.append((ts, np.zeros(1600)))
            if history:
                audio._db_history.append((ts, 56.0 if i == 3 else 30.0))
        if history:
            audio._db_history.append((stamp + 30, 90.0))
        video = SimpleNamespace(
            detections_between=lambda *args: [], recent_detections=[], video_buffer=[],
        )
        manager = EventManager(audio, video, config=config, base_dir=ROOT)
        manager._save_clip = Mock(return_value=("event.mp4", 1000))
        return manager, audio, stamp

    def test_persists_window_max_and_average_instead_of_trigger_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, _, stamp = self._event(directory, history=True)
            manager._process_trigger({"type": "peak", "db": 43.0, "timestamp": stamp})
            with sqlite3.connect(Path(directory) / "events.db") as conn:
                row, = conn.execute("SELECT db_max, db_avg FROM noise_events")
            self.assertAlmostEqual(row[0], 56.0)
            self.assertAlmostEqual(row[1], energy_mean([30.0] * 20 + [56.0]))

    def test_keeps_trigger_level_when_history_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            manager, _, stamp = self._event(directory, history=False)
            with self.assertLogs("src.event_manager", level="WARNING") as logs:
                manager._process_trigger({"type": "peak", "db": 43.0, "timestamp": stamp})
            with sqlite3.connect(Path(directory) / "events.db") as conn:
                row, = conn.execute("SELECT db_max, db_avg FROM noise_events")
            self.assertAlmostEqual(row[0], 43.0)
            self.assertAlmostEqual(row[1], 43.0)
            self.assertTrue(any("nivel del disparo" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
