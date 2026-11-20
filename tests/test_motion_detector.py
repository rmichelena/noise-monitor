#!/usr/bin/env python3
"""Detección por movimiento: blobs de vehículo sí, personas altas no."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.motion_detector import MotionDetector  # noqa: E402


def make_config(**over):
    cfg = {
        "motion": {
            "enabled": True,
            "roi": [0, 0, 320, 240],
            "min_area": 400,
            "max_area": 100000,
            "min_aspect": 0.6,
            "max_aspect": 6.0,
            "min_width": 10,
            "var_threshold": 16,
            "history": 30,
            "warmup_frames": 5,
        }
    }
    cfg["motion"].update(over)
    return cfg


class MotionDetectorTest(unittest.TestCase):
    def _feed(self, det, frames=10, rect=None):
        out = []
        for _ in range(frames):
            frame = np.full((240, 320, 3), 255, dtype=np.uint8)
            if rect:
                x, y, w, h = rect
                frame[y:y + h, x:x + w] = 0
            out = det.detect(frame)
        return out

    def test_detecta_vehiculo(self):
        det = MotionDetector(make_config())
        self._feed(det, 6)  # warmup
        blobs = self._feed(det, 1, rect=(20, 100, 120, 50))
        self.assertTrue(blobs, "no detectó el blob del auto")
        blob = blobs[0]
        self.assertGreater(blob["cx"], 0)
        self.assertTrue(0 <= blob["bbox"][1] <= 1)

    def test_descarta_persona_alta(self):
        det = MotionDetector(make_config())
        self._feed(det, 6)
        blobs = self._feed(det, 1, rect=(50, 60, 20, 120))  # w/h = 0.17
        self.assertEqual(blobs, [], f"coló una persona como vehículo: {blobs}")

    def test_clasifica_con_modelo(self):
        blob = {"bbox": [0.4, 0.1, 0.6, 0.4], "class": "car", "score": 0.5}
        model = [{"class": "truck", "score": 0.8, "bbox": [0.42, 0.12, 0.58, 0.38]}]
        out = MotionDetector.classify([blob], model)
        self.assertEqual(out[0]["class"], "truck")
        self.assertAlmostEqual(out[0]["score"], 0.8)


if __name__ == "__main__":
    unittest.main()
