"""
Monitor de video: captura continua, inferencia con Coral Edge TPU.
Mantiene buffer circular de frames y detecciones recientes para correlación.
Integra CentroidTracker para registrar pasajes de vehículos.
"""

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from queue import Queue

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter, load_delegate

from .utils import load_config
from .vehicle_tracker import CentroidTracker

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

        raw_source = video_cfg.get("source", 0)
        # source: int = dispositivo USB, str = ruta a archivo de video
        if isinstance(raw_source, str):
            path = Path(raw_source)
            if not path.is_absolute():
                path = self.base_dir / path
            self.source = str(path.resolve())
        else:
            self.source = raw_source

        self.resolution = tuple(video_cfg["resolution"])
        self.framerate = int(video_cfg["framerate"])
        self.inference_interval = int(video_cfg["inference_interval"])
        self.detection_threshold = float(video_cfg["detection_threshold"])

        self.video_seconds = int(buffer_cfg["video_seconds"])
        # La cámara entrega más fps de los configurados (medido: ~60 con
        # framerate=30). El buffer se recorta por tiempo, no por número de
        # frames, y la captura se limita a `framerate` fps reales: si no, un
        # clip de 10 s de audio se escribía con 5 s de video a 30 fps.
        self.buffer_maxlen = int(self.video_seconds * self.framerate * 1.5)
        self._min_frame_interval = 1.0 / self.framerate
        self._last_stored_ts = 0.0
        self._frame_stride = 1   # 1 de cada N frames leídos (cámara > target)
        self._read_count = 0

        self.video_buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=self.buffer_maxlen)
        self._recent_detections: list[dict] = []
        self._lock = threading.Lock()

        # Modelo
        model_dir = self.base_dir / "models"
        self.model_path = model_dir / "ssd_mobilenet_v2_coco_edgetpu.tflite"
        self.labels_path = model_dir / "coco_labels.txt"
        self.labels = load_labels(self.labels_path)

        self._interpreter: Interpreter | None = None
        self._tpu_retry_at = 0.0  # epoch del próximo intento si el TPU falta
        self._tpu_retry_every = 60.0
        self._input_index = 0
        self._input_shape = (300, 300, 3)
        self._output_indices: list[int] = []
        self._running = False
        self._cap: cv2.VideoCapture | None = None
        self._frame_count = 0

        # Modo simulado: loop cuando el video termina
        debug_cfg = self.config.get("debug", {})
        self._loop_video = bool(debug_cfg.get("loop_video", True))
        self._is_file_source = isinstance(raw_source, str)

        # Centroid tracker
        tracker_cfg = self.config.get("tracker", {})
        self.passage_queue: Queue = Queue()
        self._tracker = CentroidTracker(
            max_distance=float(tracker_cfg.get("max_distance", 80)),
            max_disappeared=int(tracker_cfg.get("max_disappeared", 20)),
            min_track_seconds=float(tracker_cfg.get("min_track_seconds", 2.0)),
            min_displacement=float(tracker_cfg.get("min_displacement", 60)),
            passage_queue=self.passage_queue,
        )

    @property
    def recent_detections(self) -> list[dict]:
        """Detecciones recientes (thread-safe). Formato: {class, score, bbox, lane}."""
        with self._lock:
            return list(self._recent_detections)

    def _init_interpreter(self) -> bool:
        """Inicializa el intérprete TFLite con delegate Edge TPU.

        Si el acelerador no está disponible (ausente, cable, puerto USB),
        el pipeline sigue vivo sin detección visual en vez de entrar en
        bucle de reinicio: el audio y la clasificación YAMNet son la
        evidencia principal y no dependen del Coral. `_ensure_interpreter`
        reintenta cada minuto, así que un replug se recupera en caliente.
        """
        try:
            delegate = load_delegate("libedgetpu.so.1", options={"device": "usb"})
        except Exception:
            logger.error(
                "VideoMonitor: Edge TPU no disponible (libedgetpu no carga). "
                "Detección visual DESACTIVADA; reintento cada %ds. "
                "La captura de audio sigue activa.",
                int(self._tpu_retry_every),
            )
            self._interpreter = None
            return False

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
        return True

    def _ensure_interpreter(self):
        """Reintenta cargar el TPU como máximo una vez por minuto."""
        if self._interpreter is not None:
            return
        now = time.time()
        if now < self._tpu_retry_at:
            return
        self._tpu_retry_at = now + self._tpu_retry_every
        if self._init_interpreter():
            self._tpu_retry_at = 0.0
            logger.info("VideoMonitor: Edge TPU recuperado; detección visual activa")

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
            cx = (xmin + xmax) / 2.0 * frame_w
            cy = (ymin + ymax) / 2.0 * frame_h
            lane = "left" if cx < frame_w / 2 else "right"

            detections.append(
                {
                    "class": class_name,
                    "score": float(scores[i]),
                    "bbox": boxes[i].tolist(),
                    "cx": float(cx),
                    "cy": float(cy),
                    "lane": lane,
                }
            )

        return detections

    def _measure_camera_fps(self, frames: int = 30) -> float:
        """Estima los fps reales de la cámara leyendo frames consecutivos."""
        t0 = time.time()
        n = 0
        while n < frames:
            if not self._cap.read()[0]:
                break
            n += 1
        dt = time.time() - t0
        return (n / dt) if dt > 0 and n else float(self.framerate)

    def start(self):
        """Inicia captura e inferencia."""
        if self._running:
            return

        self._cap = cv2.VideoCapture(self.source)
        if not self._is_file_source:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self._cap.set(cv2.CAP_PROP_FPS, self.framerate)

        if not self._cap.isOpened():
            raise RuntimeError(f"No se pudo abrir cámara {self.source}")

        if not self._is_file_source:
            measured = self._measure_camera_fps()
            self._frame_stride = max(1, round(measured / self.framerate))
            logger.info(
                "Cámara entrega %.1f fps; se guarda 1 de cada %d frame(s)",
                measured,
                self._frame_stride,
            )

        if not self._init_interpreter():
            self._tpu_retry_at = time.time() + self._tpu_retry_every
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
            # Fin del video: loop si es archivo y loop_video está activo
            if self._is_file_source and self._loop_video:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return False
            else:
                return False

        self._read_count += 1
        if self._frame_stride > 1 and self._read_count % self._frame_stride:
            return True

        timestamp = time.time()
        # La cámara puede entregar 60 fps aunque se pida 30: descartar los
        # frames sobrantes mantiene 1 frame por intervalo real y evita que
        # el buffer (y el clip) se estiren en el tiempo.
        if (
            self._last_stored_ts
            and timestamp - self._last_stored_ts < self._min_frame_interval
        ):
            return True
        self._last_stored_ts = timestamp

        self.video_buffer.append((timestamp, frame.copy()))
        cutoff = timestamp - self.video_seconds
        while self.video_buffer and self.video_buffer[0][0] < cutoff:
            self.video_buffer.popleft()

        self._frame_count += 1
        if self._frame_count % self.inference_interval == 0:
            self._ensure_interpreter()
            detections = self._run_inference(frame)
            with self._lock:
                self._recent_detections = detections
            self._tracker.update(detections, timestamp)

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
