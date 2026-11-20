#!/usr/bin/env python3
"""
Un pasaje y un trigger que se solapan producen UN evento `both` con un solo
clip cuya ventana es la unión de ambos.
"""

import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.event_manager import EventManager  # noqa: E402
from src.utils import load_config  # noqa: E402


class UnifiedMergeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = load_config(PROJECT_DIR / "config.yaml")
        cfg["paths"]["clips"] = self.tmp.name
        cfg["paths"]["database"] = str(Path(self.tmp.name) / "test.db")
        cfg["buffer"]["pre_event_seconds"] = 1
        cfg["buffer"]["post_event_seconds"] = 1
        cfg["buffer"]["traffic_clip_seconds"] = 5
        cfg["triggers"]["cooldown_seconds"] = 0

        now = time.time()
        audio = MagicMock()
        audio.audio_buffer = [
            (now - 6 + i * 0.1, np.zeros(4410, dtype=np.float32)) for i in range(95)
        ]
        audio.current_db = 45.0
        audio._classifier = None

        video = MagicMock()
        video.video_buffer = [
            (now - 6 + i * 0.04, np.zeros((180, 320, 3), dtype=np.uint8))
            for i in range(225)
        ]
        video.detections_between.return_value = []
        video.recent_detections = []

        self.cfg = cfg
        self.em = EventManager(audio, video, config=cfg, base_dir=PROJECT_DIR)
        self.db = cfg["paths"]["database"]
        self.passage = {
            "track_id": 7, "vehicle_class": "car", "lane": "left",
            "direction": "ltr", "speed_kmh": 32.0, "detection_score": 0.6,
            "timestamp_enter": now - 2.0, "timestamp_exit": now - 0.3,
        }
        self.trigger = {
            "type": "peak", "db": 46.0, "timestamp": now - 1.0,
            "sound_class": "Vehicle", "sound_category": "vehicle",
            "sound_confidence": 0.5,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_union(self):
        self.em._process_passage(self.passage)
        self.em._process_trigger(self.trigger)

        conn = sqlite3.connect(self.db)
        rows = conn.execute(
            "SELECT kind, start_ts, end_ts, vehicle_class, db_max, clip_path FROM events"
        ).fetchall()
        conn.close()

        self.assertEqual(len(rows), 1, f"esperaba 1 evento, hay {len(rows)}")
        kind, start, end, klass, db, clip = rows[0]
        self.assertEqual(kind, "both")
        self.assertEqual(klass, "car")
        self.assertAlmostEqual(db, 46.0, places=1)
        self.assertLessEqual(start, self.trigger["timestamp"] - 1.0)
        self.assertGreaterEqual(end, self.passage["timestamp_exit"])
        self.assertTrue(clip, "sin clip")
        files = list(Path(self.tmp.name).glob("*.mp4"))
        self.assertEqual(len(files), 1, f"clips sobrantes: {files}")


if __name__ == "__main__":
    unittest.main()
