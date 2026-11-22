#!/usr/bin/env python3
"""
Regresión: un clip normal debe traer video+audio y un clip con voz solo video.

Cubre el fallo de 2026-11-18: opciones de ffmpeg entre dos `-i` se aplican a
la entrada siguiente ("Unknown decoder 'libx264'") y todos los clips salían
sin archivo. Ejecutar con: venv/bin/python -m unittest discover -s tests
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.event_manager import EventManager  # noqa: E402
from support import example_config, resolve_stored  # noqa: E402


def _streams(path: str) -> set[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    ).stdout
    return {line.strip() for line in out.splitlines() if line.strip()}


class ClipAudioTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        cfg = example_config(self.tmp.name)
        self.em = EventManager(None, None, config=cfg, base_dir=PROJECT_DIR)
        self.frames = [
            (1000.0 + i * 0.04, np.zeros((120, 160, 3), dtype=np.uint8))
            for i in range(25)
        ]
        self.audio = [
            (1000.0 + i * 0.093, np.zeros(4096, dtype=np.float32))
            for i in range(10)
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def _clip(self, mute: bool) -> str:
        path, duration = self.em._save_clip(
            self.audio, self.frames, 44100, 1000.5, mute_audio=mute
        )
        self.assertIsNotNone(path, "el clip no se generó")
        self.assertGreater(duration, 0)
        return str(resolve_stored(path))

    def test_clip_con_audio(self):
        self.assertEqual(_streams(self._clip(False)), {"video", "audio"})

    def test_clip_sin_audio_por_voz(self):
        self.assertEqual(_streams(self._clip(True)), {"video"})


if __name__ == "__main__":
    unittest.main()
