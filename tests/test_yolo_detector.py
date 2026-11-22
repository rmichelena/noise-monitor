#!/usr/bin/env python3
"""Decodificador YOLO rectangular: cuantización, mapeo y NMS."""

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.yolo_detector import YOLODetector, nms  # noqa: E402


class YoloDecodeTest(unittest.TestCase):
    def setUp(self):
        self.det = YOLODetector.__new__(YOLODetector)
        self.det.labels = {2: "car"}
        self.det.threshold = 0.12
        self.det.nms_iou = 0.45
        self.det.input_shape = (160, 640)
        self.det._output_scale = 0.01
        self.det._output_zero = -128.0
        self.det._output_channels = 84
        self.det._output_anchors = 1
        self.det.max_candidates = 300
        self.det.max_det = 100

    def _raw(self, boxes_scores):
        """boxes_scores: [(cx, cy, w, h, cls, score)] en valores reales."""
        raw = np.full((84, 1), -128, dtype=np.int8)
        for cx, cy, w, h, cls, score in boxes_scores:
            for row, value in enumerate((cx, cy, w, h)):
                raw[row, 0] = round(value / self.det._output_scale - 128)
            raw[4 + cls, 0] = round(score / self.det._output_scale - 128)
        return raw

    def test_caja_normalizada_se_mapea_al_frame(self):
        raw = self._raw([(0.5, 0.5, 0.1, 0.1, 2, 0.9)])
        dets = self.det._decode(raw, 640, 160, (0, 0, 640, 160), 1.0, 0, 0)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["class"], "car")
        self.assertAlmostEqual(dets[0]["score"], 0.9, places=2)
        self.assertAlmostEqual(dets[0]["bbox"][1], 0.45, places=2)
        self.assertAlmostEqual(dets[0]["bbox"][0], 0.45, places=2)
        self.assertAlmostEqual(dets[0]["bbox"][3], 0.55, places=2)

    def test_letterbox_desplaza_la_caja(self):
        # crop 1280x310 -> 640x155, dy=2 px; caja en el centro del canvas.
        raw = self._raw([(0.5, 0.5, 0.2, 0.2, 2, 0.8)])
        dets = self.det._decode(raw, 1280, 720, (0, 150, 1280, 460), 0.5, 0, 2)
        self.assertEqual(len(dets), 1)
        # centro canvas 80 -> crop (80-2)/0.5 = 156 px; +150 de ROI.
        self.assertAlmostEqual(dets[0]["cy"], 150 + 156, delta=1.0)
        self.assertAlmostEqual(dets[0]["cx"], 640, delta=1.0)

    def test_score_bajo_se_descarta(self):
        raw = self._raw([(0.5, 0.5, 0.1, 0.1, 2, 0.05)])
        self.assertEqual(
            self.det._decode(raw, 640, 160, (0, 0, 640, 160), 1.0, 0, 0), []
        )

    def test_preprocess_int8_normalizado(self):
        det = YOLODetector.__new__(YOLODetector)
        det.input_shape = (160, 640)
        det._input_scale = 1.0 / 255.0
        det._input_zero = -128.0
        det._input_dtype = np.int8
        det._input_use_pixels = False
        det._qmin, det._qmax = -128, 127
        crop = np.zeros((160, 640, 3), dtype=np.uint8)
        inp, scale, dx, dy = det._preprocess(crop)
        self.assertEqual(inp.shape, (1, 160, 640, 3))
        self.assertEqual(inp.dtype, np.int8)
        self.assertEqual(int(inp[0, 0, 0, 0]), -128)
        self.assertAlmostEqual(scale, 1.0)

    def test_preprocess_uint8_pixeles(self):
        det = YOLODetector.__new__(YOLODetector)
        det.input_shape = (160, 640)
        det._input_scale = 1.0
        det._input_zero = 0.0
        det._input_dtype = np.uint8
        det._input_use_pixels = True
        det._qmin, det._qmax = 0, 255
        crop = np.full((160, 640, 3), 114, dtype=np.uint8)
        inp, _scale, _dx, _dy = det._preprocess(crop)
        self.assertEqual(inp.dtype, np.uint8)
        self.assertEqual(int(inp[0, 0, 0, 0]), 114)

    def test_detect_con_interprete_stub(self):
        class FakeInterpreter:
            def __init__(self, output):
                self.output = output
            def set_tensor(self, index, value):
                self.input = value
            def invoke(self):
                pass
            def get_tensor(self, index):
                return self.output[np.newaxis, ...]

        det = YOLODetector.__new__(YOLODetector)
        det.labels = {i: name for i, name in enumerate(
            ["person", "bicycle", "car", "motorcycle", "airplane", "bus",
             "train", "truck"] + [f"c{i}" for i in range(8, 80)]
        )}
        det.threshold = 0.5
        det.nms_iou = 0.45
        det.max_candidates = 300
        det.max_det = 100
        det.input_shape = (160, 640)
        det._input_scale = 1.0 / 255.0
        det._input_zero = -128.0
        det._input_dtype = np.int8
        det._input_use_pixels = False
        det._qmin, det._qmax = -128, 127
        det._output_scale = 0.01
        det._output_zero = -128.0
        det._output_channels = 84
        det._output_anchors = 2100
        det._output_transposed = False
        det._slow_logged = True
        det._in_index = 0
        det._out_index = 0

        raw = np.full((84, 2100), -128, dtype=np.int8)
        for row, value in enumerate((0.5, 0.5, 0.1, 0.1)):
            raw[row, 0] = round(value / det._output_scale - 128)
        raw[4 + 2, 0] = round(0.9 / det._output_scale - 128)
        det.interpreter = FakeInterpreter(raw)
        dets = det.detect(np.zeros((720, 1280, 3), dtype=np.uint8), (0, 220, 1280, 400))
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0]["class"], "car")

    def test_nms_conserva_la_mejor(self):
        dets = [
            {"class": "car", "score": 0.9, "bbox": [0.1, 0.1, 0.3, 0.3]},
            {"class": "car", "score": 0.5, "bbox": [0.1, 0.1, 0.3, 0.3]},
            {"class": "truck", "score": 0.7, "bbox": [0.1, 0.1, 0.3, 0.3]},
        ]
        out = nms(dets, 0.45)
        self.assertEqual(len(out), 2)
        self.assertEqual({d["class"] for d in out}, {"car", "truck"})
        car = next(d for d in out if d["class"] == "car")
        self.assertEqual(car["score"], 0.9)


if __name__ == "__main__":
    unittest.main()
