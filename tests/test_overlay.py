#!/usr/bin/env python3
"""El overlay de fecha/hora y dB se dibuja sin mutar el frame original."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.event_manager import draw_frame_overlay  # noqa: E402


class OverlayTest(unittest.TestCase):
    def test_dibuja_sin_mutar(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        original = frame.copy()
        out = draw_frame_overlay(frame, 1700000000.0, "46.1 dB(A)")
        self.assertGreater(int(out.sum()), 0, "no dibujó nada")
        self.assertEqual(int(frame.sum()), int(original.sum()), "mutó el original")
        # esquina inferior izquierda (dB) y superior izquierda (fecha)
        self.assertGreater(int(out[-60:, :300].sum()), 0)
        self.assertGreater(int(out[:60, :300].sum()), 0)


if __name__ == "__main__":
    unittest.main()
