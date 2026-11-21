#!/usr/bin/env python3
"""El overlay de fecha/hora y dB se dibuja sin mutar el frame original."""

import sys
import threading
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.event_manager import draw_frame_overlay  # noqa: E402
from src.video_monitor import VideoMonitor  # noqa: E402


class OverlayTest(unittest.TestCase):
    def test_dibuja_sin_mutar(self):
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        original = frame.copy()
        out = draw_frame_overlay(frame, 1700000000.0, 46.1)
        self.assertGreater(int(out.sum()), 0, "no dibujó nada")
        self.assertEqual(int(frame.sum()), int(original.sum()), "mutó el original")
        # esquina inferior izquierda (dB) y superior izquierda (fecha)
        self.assertGreater(int(out[-60:, :300].sum()), 0)
        # barra: debe haber al menos 3 colores distintos en la franja inferior
        strip = out[-40:, :300].reshape(-1, 3)
        unique = {tuple(px) for px in strip if px.any()}
        self.assertGreaterEqual(len(unique), 3, "la barra no se dibujó")
        self.assertGreater(int(out[:60, :300].sum()), 0)


class LiveOverlayFilterTest(unittest.TestCase):
    def _monitor(self, classes):
        monitor = VideoMonitor.__new__(VideoMonitor)
        monitor.overlay_classes = set(classes)
        monitor._overlay_lock = threading.Lock()
        monitor._overlay_jpeg = None
        return monitor

    def _det(self, cls, x):
        return {
            "class": cls, "score": 0.13, "cx": x + 20, "cy": 280,
            "bbox": [0.3, x / 640.0, 0.45, (x + 40) / 640.0],
        }

    def test_solo_dibuja_las_clases_permitidas(self):
        monitor = self._monitor({"car"})
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        labels = []
        with mock.patch("src.video_monitor.cv2.putText", side_effect=lambda *a, **k: labels.append(a[1])):
            monitor._update_overlay(frame, [self._det("car", 100), self._det("potted plant", 300)])
        self.assertEqual(len(labels), 1)
        self.assertTrue(labels[0].startswith("car"))
        self.assertIsNotNone(monitor.latest_overlay())

    def test_sin_lista_dibuja_vehiculos(self):
        monitor = self._monitor({"car", "truck", "bus", "motorcycle", "bicycle"})
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        labels = []
        with mock.patch("src.video_monitor.cv2.putText", side_effect=lambda *a, **k: labels.append(a[1])):
            monitor._update_overlay(
                frame,
                [self._det("truck", 100), self._det("person", 300), self._det("bench", 500)],
            )
        self.assertEqual(len(labels), 1)
        self.assertTrue(labels[0].startswith("truck"))


if __name__ == "__main__":
    unittest.main()
