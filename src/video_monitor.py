"""
Monitor de video: captura continua, inferencia con Coral Edge TPU.
Mantiene buffer circular de frames y detecciones recientes para correlación.
"""

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

from .utils import load_config

logger = logging.getLogger(__name__)

# Clases de vehículos relevantes para correlación ruido
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


def load_labels(path: str | Path) -> dict[int, str]:
    """Carga labels COCO: id -> nombre."""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}


class VideoMonitor:
    """
    Captura video continuo, ejecuta inferencia cada N frames.
    Expone buffer circular y recent_detections para el EventManager.
    """

    def __init__(self, config: dict | None = None, base_dir: Path | None = None):
        self.config = config or load_config()
        self.base_dir = Path(base_dir or os.getcwd())

        video_cfg = self.config["video"]
        buffer_cfg = self.config["buffer"]

        self.source = video_cfg.get("source", 0)
        self.resolution = tuple(video_cfg["resolution"])
        self.framerate = int(video_cfg["framerate"])
        self.inference_interval = int(video_cfg["inference_interval"])
        self.detection_threshold = float(video_cfg["detection_threshold"])

        video_seconds = int(buffer_cfg["video_seconds"])
        self.buffer_maxlen = video_seconds * self.framerate

        self.video_buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=self.buffer_maxlen)
        self._recent_detections: list[dict] = []
        self._lock = threading.Lock()

        # Modelo
        model_dir = self.base_dir / "models"
        self.model_path = model_dir / "ssd_mobilenet_v2_coco_edgetpu.tflite"
        self.labels_path = model_dir / "coco_labels.txt"
        self.labels = load_labels(self.labels_path)

        self._interpreter: Interpreter | None = None
        self._input_index = 0
        self._input_shape = (300, 300, 3)
        self._output_indices: list[int] = []
        self._running = False
        self._cap: cv2.VideoCapture | None = None
        self._frame_count = 0

    @property
    def recent_detections(self) -> list[dict]:
        """Detecciones recientes (thread-safe). Formato: {class, score, bbox, lane}."""
        with self._lock:
            return list(self._recent_detections)

    def _init_interpreter(self):
        """Inicializa el intérprete TFLite con delegate Edge TPU."""
        delegate = load_delegate("libedgetpu.so.1", options={"device": "usb"})
        self._interpreter = Interpreter(
            model_path=str(self.model_path),
            experimental_delegates=[delegate],
        )
        self._interpreter.allocate_tensors()

        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        self._input_index = input_details[0]["index"]
        shape = input_details[0]["shape"]
        self._input_shape = (int(shape[1]), int(shape[2]))  # H, W

        self._output_indices = [o["index"] for o in output_details]
        logger.info(
            "VideoMonitor: modelo cargado %s, input %s",
            self.model_path.name,
            self._input_shape,
        )

    def _run_inference(self, frame: np.ndarray) -> list[dict]:
        """Ejecuta inferencia y retorna lista de detecciones sobre umbral."""
        if self._interpreter is None:
            return []

        h, w = self._input_shape
        resized = cv2.resize(frame, (w, h))
        input_data = np.expand_dims(resized, axis=0).astype(np.uint8)

        self._interpreter.set_tensor(self._input_index, input_data)
        self._interpreter.invoke()

        boxes = self._interpreter.get_tensor(self._output_indices[0])[0]
        classes = self._interpreter.get_tensor(self._output_indices[1])[0]
        scores = self._interpreter.get_tensor(self._output_indices[2])[0]
        count = int(self._interpreter.get_tensor(self._output_indices[3])[0])

        detections = []
        frame_h, frame_w = frame.shape[:2]

        for i in range(min(count, len(scores))):
            if scores[i] < self.detection_threshold:
                continue

            class_id = int(classes[i])
            class_name = self.labels.get(class_id, f"id:{class_id}")

            # bbox: [ymin, xmin, ymax, xmax] normalizado 0-1
            ymin, xmin, ymax, xmax = boxes[i]
            center_x = (xmin + xmax) / 2.0 * frame_w
            lane = "left" if center_x < frame_w / 2 else "right"

            detections.append(
                {
                    "class": class_name,
                    "score": float(scores[i]),
                    "bbox": boxes[i].tolist(),
                    "lane": lane,
                }
            )

        return detections

    def start(self):
        """Inicia captura e inferencia."""
        if self._running:
            return

        self._cap = cv2.VideoCapture(self.source)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        self._cap.set(cv2.CAP_PROP_FPS, self.framerate)

        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir cámara {self.source}")

        self._init_interpreter()
        self._running = True
        self._frame_count = 0
        logger.info(
            "VideoMonitor iniciado: source=%s, %dx%d @ %d fps",
            self.source,
            self.resolution[0],
            self.resolution[1],
            self.framerate,
        )

    def process_frame(self) -> bool:
        """
        Captura un frame, lo guarda en buffer, ejecuta inferencia cada N frames.
        Retorna True si hay frame nuevo, False si error o fin.
        """
        if not self._running or self._cap is None:
            return False

        ret, frame = self._cap.read()
        if not ret:
            return False

        timestamp = time.time()
        self.video_buffer.append((timestamp, frame.copy()))

        self._frame_count += 1
        if self._frame_count % self.inference_interval == 0:
            detections = self._run_inference(frame)
            with self._lock:
                self._recent_detections = detections

        return True

    def run_loop(self):
        """Loop bloqueante de captura (para ejecutar en thread)."""
        while self._running and self._cap:
            self.process_frame()

    def stop(self):
        """Detiene captura."""
        self._running = False
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("VideoMonitor detenido")
