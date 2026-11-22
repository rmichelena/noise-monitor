#!/usr/bin/env python3
"""
Nombres de clip únicos y limpieza de temporales.

El nombre se derivaba al segundo: dos eventos en el mismo segundo escribían
el mismo .mp4, así que el segundo borraba el clip del primero mientras su fila
seguía apuntando a esa ruta (la evidencia de un evento era el video de otro).
"""

import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import example_config  # noqa: E402
from src.event_manager import EventManager  # noqa: E402

FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")


def frames(count: int = 25, h: int = 120, w: int = 160):
    return [(1000.0 + i * 0.04, np.zeros((h, w, 3), dtype=np.uint8)) for i in range(count)]


def audio(count: int = 10, size: int = 4096):
    return [(1000.0 + i * 0.093, np.zeros(size, dtype=np.float32)) for i in range(count)]


class ClipNamingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.cfg = example_config(self.base)
        self.em = EventManager(None, None, config=self.cfg, base_dir=PROJECT_DIR)

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_second_gets_a_suffix(self):
        ts = 1_700_000_000.5
        first = self.em._unique_clip_path("event", ts)
        first.write_bytes(b"clip")
        second = self.em._unique_clip_path("event", ts)
        self.assertNotEqual(first, second)
        self.assertTrue(second.name.endswith("_2.mp4"), second.name)
        self.assertEqual(first.parent, second.parent)

    def test_third_clip_in_the_same_second(self):
        ts = 1_700_000_000.0
        paths = []
        for _ in range(3):
            path = self.em._unique_clip_path("passage", ts)
            path.write_bytes(b"x")
            paths.append(path)
        self.assertEqual(len({p.name for p in paths}), 3)
        self.assertTrue(paths[-1].name.endswith("_3.mp4"))

    def test_prefixes_do_not_collide(self):
        ts = 1_700_000_000.0
        event = self.em._unique_clip_path("event", ts)
        passage = self.em._unique_clip_path("passage", ts)
        self.assertNotEqual(event, passage)


@unittest.skipUnless(FFMPEG, "requiere ffmpeg y ffprobe")
class ClipSavingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.cfg = example_config(self.base)
        self.em = EventManager(None, None, config=self.cfg, base_dir=PROJECT_DIR)

    def tearDown(self):
        self.tmp.cleanup()

    def test_two_clips_in_the_same_second_are_two_files(self):
        timestamp = time.time()
        first, d1 = self.em._save_clip(audio(), frames(), 44100, timestamp)
        second, d2 = self.em._save_clip(audio(), frames(), 44100, timestamp)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(first, second)
        self.assertGreater(d1, 0)
        self.assertGreater(d2, 0)

        first_path = Path(first) if Path(first).is_absolute() else PROJECT_DIR / first
        second_path = Path(second) if Path(second).is_absolute() else PROJECT_DIR / second
        self.assertTrue(first_path.is_file())
        self.assertTrue(second_path.is_file())
        self.assertNotEqual(first_path.read_bytes(), b"")

    def test_no_temporary_files_left_behind(self):
        before = set(Path(tempfile.gettempdir()).glob("tmp*"))
        self.em._save_clip(audio(), frames(), 44100, time.time())
        leftover = {
            p for p in set(Path(tempfile.gettempdir()).glob("tmp*")) - before
            if p.suffix in {".wav", ".mp4"}
        }
        self.assertEqual(leftover, set())


if __name__ == "__main__":
    unittest.main()
