#!/usr/bin/env python3
"""
Regresión: cada pasaje de vehículo guarda su propio clip (aunque no haya
ruido) y la fila de vehicle_passages queda con clip_path válido.
"""

import sqlite3
import subprocess
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


class PassageClipTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = load_config(PROJECT_DIR / "config.yaml")
        cfg["paths"]["clips"] = self.tmp.name
        cfg["paths"]["database"] = str(Path(self.tmp.name) / "test.db")
        cfg["buffer"]["pre_event_seconds"] = 2
        cfg["buffer"]["post_event_seconds"] = 2

        now = time.time()
        audio = MagicMock()
        audio.audio_buffer = [
            (now - 4 + i * 0.093, np.zeros(4096, dtype=np.float32)) for i in range(60)
        ]
        audio.current_db = 38.0
        audio._classifier = None

        video = MagicMock()
        video.video_buffer = [
            (now - 4 + i * 0.04, np.zeros((120, 160, 3), dtype=np.uint8))
            for i in range(120)
        ]

        self.cfg = cfg
        self.em = EventManager(audio, video, config=cfg, base_dir=PROJECT_DIR)
        self.passage = {
            "vehicle_class": "car",
            "lane": "left",
            "direction": "ltr",
            "speed_kmh": 32.0,
            "detection_score": 0.6,
            "timestamp_enter": now - 2.5,
            "timestamp_exit": now - 0.5,
        }
        self.db = cfg["paths"]["database"]

    def tearDown(self):
        self.tmp.cleanup()

    def test_pasaje_guarda_clip(self):
        self.em._process_passage(self.passage)
        conn = sqlite3.connect(self.db)
        row = conn.execute(
            "SELECT clip_path, kind FROM events"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row, "no se insertó el pasaje")
        self.assertEqual(row[1], "traffic")
        clip = row[0]
        self.assertTrue(clip, "el pasaje quedó sin clip")
        path = PROJECT_DIR / clip
        self.assertTrue(path.is_file(), f"no existe el archivo {path}")
        streams = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        self.assertIn("video", streams)
        self.assertIn("audio", streams)


if __name__ == "__main__":
    unittest.main()
