#!/usr/bin/env python3
"""
Regresión: el culpable se decide sobre toda la ventana del clip, no sobre el
último frame. Si en la ventana pasaron dos autos, una moto y una persona, el
evento debe atribuirse a la moto (mayor prioridad de ruido), no a la persona.

Ejecutar: venv/bin/python -m unittest discover -s tests
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.event_manager import EventManager  # noqa: E402
from support import create_events_table, example_config, resolve_stored  # noqa: E402


class CulpritWindowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = example_config(self.tmp.name)
        cfg["buffer"]["pre_event_seconds"] = 1
        cfg["buffer"]["post_event_seconds"] = 1
        cfg["triggers"]["cooldown_seconds"] = 0

        now = time.time()
        detections = [
            {"class": "car", "score": 0.6, "cx": 400, "cy": 700, "lane": "left",
             "bbox": [0.6, 0.2, 0.7, 0.3]},
            {"class": "motorcycle", "score": 0.7, "cx": 900, "cy": 700, "lane": "right",
             "bbox": [0.6, 0.4, 0.7, 0.5]},
            {"class": "person", "score": 0.8, "cx": 1200, "cy": 650, "lane": "right",
             "bbox": [0.6, 0.6, 0.7, 0.65]},
        ]
        video = MagicMock()
        video.detections_between.return_value = detections
        video.recent_detections = detections
        video.video_buffer = [
            (now + i * 0.04, np.zeros((120, 160, 3), dtype=np.uint8)) for i in range(50)
        ]
        video.passage_queue = MagicMock()
        audio = MagicMock()
        audio.audio_buffer = [
            (now + i * 0.093, np.zeros(4096, dtype=np.float32)) for i in range(30)
        ]
        audio.current_db = 42.0
        # Sin historial de niveles: el evento conserva el nivel del disparo.
        audio.level_stats.return_value = None

        self.em = EventManager(audio, video, config=cfg, base_dir=PROJECT_DIR)
        self.db = cfg["paths"]["database"]
        create_events_table(self.db)
        self.trigger = {
            "type": "peak", "db": 42.0, "timestamp": now,
            "sound_class": "Vehicle", "sound_category": "vehicle",
            "sound_confidence": 0.6,
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_culpable_por_ventana(self):
        self.em._process_trigger(self.trigger)
        conn = sqlite3.connect(self.db)
        row = conn.execute(
            # Sin pasaje cerrado y con vehículos detectados en la ventana, el
            # ruido nace `both` (S6); el culpable sale igual de la ventana.
            "SELECT vehicle_class, confidence, clip_path FROM events WHERE kind = 'both'"
        ).fetchone()
        conn.close()
        self.assertIsNotNone(row, "no se insertó el evento")
        self.assertEqual(row[0], "motorcycle")
        self.assertEqual(row[1], "medium")
        self.assertTrue(row[2], "el clip no se generó")


if __name__ == "__main__":
    unittest.main()
