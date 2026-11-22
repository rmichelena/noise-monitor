"""
Detector YOLO11 rectangular sobre Edge TPU.

El modelo se exporta con `imgsz=(160, 640)` (4:1) para que el crop de la
calzada no se deforme al entrar a la red. Este módulo solo usa
`ai-edge-litert`: no depende de Ultralytics ni de TensorFlow en producción.

Salida del `.tflite` (INT8): `[1, 84, 2100]`, 4 coordenadas `cx, cy, w, h`
normalizadas al canvas de entrada + 80 scores de clase ya sigmoidales.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

try:  # El runtime no hace falta para importar el módulo en hosts sin Coral.
    from ai_edge_litert.interpreter import Interpreter, load_delegate
except ImportError:  # pragma: no cover - depende del host
    Interpreter = None
    load_delegate = None

from .coco80_labels import COCO80_NAMES

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "models/yolo11n_160x640_edgetpu.tflite"


def _iou(a: list[float], b: list[float]) -> float:
    """IoU de dos cajas [x1, y1, x2, y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(detections: list[dict], iou_threshold: float = 0.45) -> list[dict]:
    """NMS por clase; conserva la caja de mayor score."""
    keep: list[dict] = []
    for cls in sorted({d["class"] for d in detections}):
        candidates = sorted(
            (d for d in detections if d["class"] == cls),
            key=lambda d: d["score"],
            reverse=True,
        )
        selected: list[dict] = []
        for cand in candidates:
            box = [cand["bbox"][1], cand["bbox"][0], cand["bbox"][3], cand["bbox"][2]]
            if all(
                _iou(box, [s["bbox"][1], s["bbox"][0], s["bbox"][3], s["bbox"][2]])
                <= iou_threshold
                for s in selected
            ):
                selected.append(cand)
        keep.extend(selected)
    keep.sort(key=lambda d: d["score"], reverse=True)
    return keep


class YOLODetector:
    """Envoltorio fino sobre el `.tflite` y su decodificador."""

    def __init__(self, config: dict, base_dir: Path):
        video = config.get("video") or {}
        self.base_dir = Path(base_dir)
        raw_path = video.get("yolo_model_path") or DEFAULT_MODEL
        path = Path(raw_path)
        self.model_path = path if path.is_absolute() else self.base_dir / path
        # El umbral de YOLO es propio: el 0.12 del SSD admitía demasiados
        # candidatos con la distribución de scores de YOLO.
        self.threshold = float(video.get("yolo_threshold", 0.25))
        self.nms_iou = float(video.get("yolo_nms_iou", 0.45))
        self.max_det = int(video.get("yolo_max_det", 100))
        self.max_candidates = int(video.get("yolo_max_candidates", 300))
        self.labels = {i: name for i, name in enumerate(COCO80_NAMES)}
        self.labels_ok = (
            len(COCO80_NAMES) == 80
            and self.labels.get(2) == "car"
            and self.labels.get(5) == "bus"
            and self.labels.get(7) == "truck"
        )
        self.interpreter = None
        self.input_shape: tuple[int, int] = (160, 640)
        self._input_scale = 1.0 / 255.0
        self._input_zero = -128.0
        self._input_dtype = np.int8
        self._input_use_pixels = False
        self._output_scale = 0.006591304205358028
        self._output_zero = -128.0
        self._output_channels = 84
        self._output_anchors = 2100
        self._output_transposed = False
        self._slow_logged = False

    def init_interpreter(self) -> bool:
        if not self.labels_ok:
            logger.error(
                "YOLODetector: labelmap COCO-80 inválido (car/bus/truck no "
                "coinciden); detector desactivado"
            )
            return False
        if not self.model_path.exists():
            logger.error("YOLODetector: no existe el modelo %s", self.model_path)
            return False
        if Interpreter is None or load_delegate is None:
            logger.error(
                "YOLODetector: falta ai-edge-litert; detector yolo desactivado"
            )
            self.interpreter = None
            return False
        try:
            delegate = load_delegate("libedgetpu.so.1", options={"device": "usb"})
        except Exception:
            logger.error(
                "YOLODetector: Edge TPU no disponible; detector yolo desactivado"
            )
            self.interpreter = None
            return False
        try:
            self.interpreter = Interpreter(
                model_path=str(self.model_path),
                experimental_delegates=[delegate],
            )
            self.interpreter.allocate_tensors()
            inp = self.interpreter.get_input_details()[0]
            out = self.interpreter.get_output_details()[0]
            self._in_index = inp["index"]
            self._out_index = out["index"]
            self.input_shape = (int(inp["shape"][1]), int(inp["shape"][2]))
            self._input_dtype = np.dtype(inp["dtype"]).type
            if self._input_dtype not in (np.int8, np.uint8):
                raise ValueError(f"dtype de entrada no soportado: {inp['dtype']}")
            self._input_scale, self._input_zero = (float(v) for v in inp["quantization"])
            if self._input_scale <= 0:
                raise ValueError("input quantization scale <= 0")
            qmin, qmax = (
                (-128, 127) if self._input_dtype is np.int8 else (0, 255)
            )
            self._qmin, self._qmax = qmin, qmax
            # Si el rango real llega a ~255, el grafo espera píxeles 0..255
            # (normalización embebida); si llega a ~1, espera 0..1.
            self._input_use_pixels = (qmax - self._input_zero) * self._input_scale > 2.0

            shape = out["shape"]
            if len(shape) != 3 or int(shape[0]) != 1:
                raise ValueError(f"salida con forma inesperada: {shape}")
            channels = int(shape[1])
            anchors = int(shape[2])
            expected = 4 + len(self.labels)
            self._output_transposed = False
            if channels != expected:
                if anchors == expected:
                    self._output_transposed = True
                    channels, anchors = anchors, channels
                else:
                    raise ValueError(
                        f"salida {shape} no coincide con 4+{len(self.labels)} clases"
                    )
            self._output_channels = channels
            self._output_anchors = anchors
            self._output_scale, self._output_zero = (float(v) for v in out["quantization"])
            if self._output_scale <= 0:
                raise ValueError("output quantization scale <= 0")
        except Exception:
            logger.exception("YOLODetector: no se pudo cargar %s", self.model_path)
            self.interpreter = None
            return False
        logger.info(
            "YOLODetector: %s cargado, input %dx%d",
            self.model_path.name,
            self.input_shape[1],
            self.input_shape[0],
        )
        return True

    def _preprocess(self, crop: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """Letterbox del crop al tamaño del modelo; retorna canvas, escala, dx, dy."""
        h, w = crop.shape[:2]
        in_h, in_w = self.input_shape
        scale = min(in_w / w, in_h / h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        dx, dy = (in_w - new_w) // 2, (in_h - new_h) // 2
        canvas = np.full((in_h, in_w, 3), 114, dtype=np.uint8)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas[dy : dy + new_h, dx : dx + new_w] = resized
        # El modelo se entrenó con RGB; OpenCV entrega BGR.
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        real = canvas.astype(np.float32) if self._input_use_pixels else canvas.astype(np.float32) / 255.0
        quant = np.round(real / self._input_scale + self._input_zero).clip(
            self._qmin, self._qmax
        ).astype(self._input_dtype)
        return quant[np.newaxis, ...], scale, dx, dy

    def _decode(
        self,
        output: np.ndarray,
        frame_w: int,
        frame_h: int,
        crop: tuple[int, int, int, int],
        scale: float,
        dx: int,
        dy: int,
    ) -> list[dict]:
        """Dequantiza y mapea cajas del canvas al frame completo."""
        in_h, in_w = self.input_shape
        if output.shape != (self._output_channels, self._output_anchors):
            logger.error(
                "YOLODetector: salida %s distinta de (%d, %d); frame descartado",
                output.shape, self._output_channels, self._output_anchors,
            )
            return []
        raw = (output.astype(np.float32) - self._output_zero) * self._output_scale
        boxes = raw[:4].T
        scores = raw[4:].T
        best = scores.max(axis=1)
        classes = scores.argmax(axis=1)
        idx = np.flatnonzero(best >= self.threshold)
        if idx.size > self.max_candidates:
            top = np.argpartition(best[idx], -self.max_candidates)[-self.max_candidates:]
            idx = idx[top]

        crop_x1, crop_y1, crop_x2, crop_y2 = crop
        crop_w = max(1, crop_x2 - crop_x1)
        crop_h = max(1, crop_y2 - crop_y1)
        out: list[dict] = []
        for box, score, cls in zip(boxes[idx], best[idx], classes[idx]):
            cx, cy, bw, bh = box
            x1 = ((cx - bw / 2) * in_w - dx) / scale
            y1 = ((cy - bh / 2) * in_h - dy) / scale
            x2 = ((cx + bw / 2) * in_w - dx) / scale
            y2 = ((cy + bh / 2) * in_h - dy) / scale
            x1 = min(max(x1, 0.0), crop_w)
            x2 = min(max(x2, 0.0), crop_w)
            y1 = min(max(y1, 0.0), crop_h)
            y2 = min(max(y2, 0.0), crop_h)
            if x2 <= x1 or y2 <= y1:
                continue
            fx1 = (crop_x1 + x1) / frame_w
            fx2 = (crop_x1 + x2) / frame_w
            fy1 = (crop_y1 + y1) / frame_h
            fy2 = (crop_y1 + y2) / frame_h
            cx_full = (fx1 + fx2) / 2.0 * frame_w
            cy_full = (fy1 + fy2) / 2.0 * frame_h
            out.append(
                {
                    "class": self.labels.get(int(cls), f"id:{int(cls)}"),
                    "score": float(score),
                    "bbox": [float(fy1), float(fx1), float(fy2), float(fx2)],
                    "cx": float(cx_full),
                    "cy": float(cy_full),
                    "lane": "left" if cx_full < frame_w / 2 else "right",
                }
            )
        return nms(out, self.nms_iou)[: self.max_det]

    def detect(self, frame: np.ndarray, crop_box: tuple[int, int, int, int] | None) -> list[dict]:
        if self.interpreter is None:
            return []
        h, w = frame.shape[:2]
        if crop_box is None:
            crop_box = (0, 0, w, h)
        x1 = max(0, min(w, int(crop_box[0])))
        y1 = max(0, min(h, int(crop_box[1])))
        x2 = max(0, min(w, int(crop_box[2])))
        y2 = max(0, min(h, int(crop_box[3])))
        crop_box = (x1, y1, x2, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        inp, scale, dx, dy = self._preprocess(crop)
        t0 = time.perf_counter()
        self.interpreter.set_tensor(self._in_index, inp)
        self.interpreter.invoke()
        dt = time.perf_counter() - t0
        if dt > 0.020 and not self._slow_logged:
            logger.warning(
                "YOLODetector: inferencia lenta (%.1f ms); ¿modelo sin compilar "
                "para Edge TPU?", dt * 1000,
            )
            self._slow_logged = True
        output = self.interpreter.get_tensor(self._out_index)[0]
        if self._output_transposed:
            output = output.T
        return self._decode(output, w, h, crop_box, scale, dx, dy)
