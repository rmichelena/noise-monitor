#!/usr/bin/env python3
"""
Regresión: cada pasaje de vehículo guarda su propio clip (aunque no haya
ruido) y su fila `traffic` en events queda con clip_path válido.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.event_manager import EventManager  # noqa: E402
from support import create_events_table, example_config, resolve_stored  # noqa: E402


class PassageClipTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = example_config(self.tmp.name)
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
        create_events_table(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def test_pasaje_guarda_clip(self):
        self.em._process_passage(self.passage)
        conn = sqlite3.connect(self.db)
        row = conn.execute(
            "SELECT clip_path FROM events WHERE kind = 'traffic'"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row, "no se insertó el pasaje")
        clip = row[0]
        self.assertTrue(clip, "el pasaje quedó sin clip")
        path = resolve_stored(clip)
        self.assertTrue(path.is_file(), f"no existe el archivo {path}")
        streams = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        self.assertIn("video", streams)
        self.assertIn("audio", streams)

    def test_error_de_clasificacion_guarda_pasaje_mudo(self):
        classifier = MagicMock()
        classifier.classify.side_effect = RuntimeError("Coral no disponible")
        self.em.audio_monitor._classifier = classifier
        clip, _, muted = self.em._save_passage_clip(self.passage)
        self.assertTrue(clip)
        self.assertTrue(muted)
        streams = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "csv=p=0", str(resolve_stored(clip))],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        self.assertEqual(streams, ["video"])


if __name__ == "__main__":
    unittest.main()
