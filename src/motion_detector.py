"""
Detección de movimiento para cámara fija.

Con el fondo estático, una resta de fondo (MOG2) sobre la banda de calzada
encuentra los vehículos que el detector de objetos puede perder (autos
oscuros, poca luz, objetos pequeños). Los blobs se entregan con el mismo
formato que las detecciones del modelo para que el tracker los consuma.

El modelo sigue usándose para clasificar: si una caja de vehículo del modelo
cae dentro del blob, se hereda su clase; si no, el blob cuenta como "car".
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetector:
    def __init__(self, config: dict):
        m = config.get("motion", {})
        self.enabled = bool(m.get("enabled", False))
        self.roi = tuple(int(v) for v in m.get("roi", [0, 0, 0, 0]))
        self.min_area = float(m.get("min_area", 1200))
        self.max_area = float(m.get("max_area", 200000))
        self.min_aspect = float(m.get("min_aspect", 0.6))   # ancho/alto
        self.max_aspect = float(m.get("max_aspect", 6.0))
        self.min_width = int(m.get("min_width", 20))
        self.var_threshold = float(m.get("var_threshold", 25))
        self.history = int(m.get("history", 300))
        self.warmup_frames = int(m.get("warmup_frames", 25))
        self.learning_rate = float(m.get("learning_rate", 0.005))

        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=False,
        )
        self._frames = 0

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Devuelve blobs en movimiento dentro del ROI, en coordenadas del frame."""
        if not self.enabled:
            return []

        x1, y1, x2, y2 = self.roi
        frame_h, frame_w = frame.shape[:2]
        x2 = min(x2, frame_w)
        y2 = min(y2, frame_h)
        if x2 <= x1 or y2 <= y1:
            return []

        crop = frame[y1:y2, x1:x2]
        self._frames += 1
        mask = self._bg.apply(crop, learningRate=self.learning_rate)
        if self._frames <= self.warmup_frames:
            return []

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        crop_w = x2 - x1
        crop_h = y2 - y1
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            if bh == 0 or bw < self.min_width:
                continue
            aspect = bw / bh
            if aspect < self.min_aspect or aspect > self.max_aspect:
                continue

            xmin = (x1 + bx) / frame_w
            xmax = (x1 + bx + bw) / frame_w
            ymin = (y1 + by) / frame_h
            ymax = (y1 + by + bh) / frame_h
            cx = (xmin + xmax) / 2.0 * frame_w
            cy = (ymin + ymax) / 2.0 * frame_h
            detections.append({
                "class": "car",
                "score": 0.5,
                "bbox": [ymin, xmin, ymax, xmax],
                "cx": cx,
                "cy": cy,
                "lane": "left" if cx < frame_w / 2 else "right",
                "area": int(area),
                "moving": True,
            })
        return detections

    @staticmethod
    def classify(detections: list[dict], model_detections: list[dict]) -> list[dict]:
        """Hereda la clase del modelo cuando su caja cae dentro del blob."""
        vehicles = [
            d for d in model_detections
            if d.get("class") in {"car", "truck", "bus", "motorcycle", "bicycle"}
        ]
        for blob in detections:
            ymin, xmin, ymax, xmax = blob["bbox"]
            for det in vehicles:
                dy = (det["bbox"][0] + det["bbox"][2]) / 2.0
                dx = (det["bbox"][1] + det["bbox"][3]) / 2.0
                if xmin <= dx <= xmax and ymin <= dy <= ymax:
                    blob["class"] = det["class"]
                    blob["score"] = det["score"]
                    break
        return detections
