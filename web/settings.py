"""
Ajustes editables del pipeline, con persistencia en config.yaml.

La UI manda un diccionario `{ruta: valor}`; aquí se valida contra un esquema
cerrado (nada de escribir claves arbitrarias), se aplica sobre una copia
round-trip del YAML para conservar comentarios y se reemplaza el archivo de
forma atómica. Los cambios requieren reiniciar el servicio: los monitores leen
la configuración al arrancar.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from src.perceptual_trigger import Settings
from src.utils import ConfigError, validate_config
from src.vehicle_tracker import VEHICLE_CLASSES

RECORD_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle")
DETECTORS = ("ssd", "yolo")
CRITERIA = ("dba", "psychoacoustic")
METRICS = ("loudness_mean", "annoyance_pa")

# Campos que la UI puede mostrar y guardar. La ruta usa puntos; el backend
# rechaza cualquier ruta que no esté aquí.
FIELD_GROUPS: list[dict[str, Any]] = [
    {
        "title": "Disparo de ruido",
        "fields": [
            {
                "path": "triggers.criterion",
                "label": "Criterio de detección",
                "type": "select",
                "default": "dba",
                "options": [["dba", "dB(A) ponderado"], ["psychoacoustic", "Psicoacústico"]],
            },
            {"path": "triggers.peak_db", "label": "Umbral de pico", "type": "number", "step": 0.1, "min": -60, "max": 194, "unit": "dB"},
            {"path": "triggers.sustained_db", "label": "Umbral sostenido", "type": "number", "step": 0.1, "min": -60, "max": 194, "unit": "dB"},
            {"path": "triggers.sustained_duration_ms", "label": "Duración sostenida", "type": "integer", "min": 0, "unit": "ms"},
            {"path": "triggers.correlation_db", "label": "Umbral con vehículo", "type": "number", "step": 0.1, "min": -60, "max": 194, "unit": "dB"},
            {"path": "triggers.cooldown_seconds", "label": "Enfriamiento entre eventos", "type": "number", "step": 0.1, "min": 0, "unit": "s"},
        ],
    },
    {
        "title": "Psicoacústico y calibración",
        "fields": [
            {
                "path": "triggers.psychoacoustic.metric",
                "label": "Métrica",
                "type": "select",
                "default": "loudness_mean",
                "options": [["loudness_mean", "Loudness media"], ["annoyance_pa", "Annoyance Pa"]],
            },
            {"path": "triggers.psychoacoustic.threshold", "label": "Umbral psicoacústico", "type": "number", "step": 0.01, "min": 0, "default": 20.0},
            {"path": "triggers.psychoacoustic.window_seconds", "label": "Ventana de análisis", "type": "number", "step": 0.1, "min": 2, "unit": "s", "default": 3.0},
            {"path": "triggers.psychoacoustic.hop_seconds", "label": "Salto entre ventanas", "type": "number", "step": 0.1, "min": 0.1, "unit": "s", "default": 1.5},
            {"path": "triggers.psychoacoustic.expected_latency_seconds", "label": "Latencia esperada", "type": "number", "step": 0.1, "min": 0.1, "unit": "s", "default": 1.6},
            {"path": "audio.pressure_calibration.id", "label": "ID de calibración", "type": "text", "default": ""},
            {"path": "audio.pressure_calibration.pa_per_unit", "label": "Pa por unidad", "type": "number", "step": 0.000001, "min": 0, "default": 1.0},
            {"path": "audio.calibration_offset", "label": "Offset de calibración", "type": "number", "step": 0.1, "unit": "dB"},
        ],
    },
    {
        "title": "Video y detector",
        "fields": [
            {"path": "video.detector", "label": "Modelo", "type": "select", "default": "ssd", "options": [["ssd", "SSD MobileNet"], ["yolo", "YOLO11"]]},
            {"path": "video.yolo_model_path", "label": "Ruta del modelo YOLO", "type": "text", "default": "models/yolo11n_160x640_edgetpu.tflite"},
            {"path": "video.yolo_threshold", "label": "Confianza YOLO", "type": "number", "step": 0.01, "min": 0, "max": 1, "default": 0.3},
            {"path": "video.resolution", "label": "Resolución", "type": "size", "unit": "px"},
            {"path": "video.framerate", "label": "Framerate", "type": "integer", "min": 1, "max": 240, "unit": "fps"},
            {"path": "video.inference_interval", "label": "Cada N frames", "type": "integer", "min": 1},
            {"path": "video.detection_threshold", "label": "Confianza mínima", "type": "number", "step": 0.01, "min": 0, "max": 1},
            {"path": "video.detection_min_width", "label": "Ancho mínimo de caja", "type": "integer", "min": 1, "unit": "px"},
            {"path": "video.detection_max_width", "label": "Ancho máximo de caja", "type": "integer", "min": 1, "unit": "px"},
            {"path": "video.detection_min_height", "label": "Alto mínimo de caja", "type": "integer", "min": 1, "unit": "px"},
            {"path": "video.detection_max_height", "label": "Alto máximo de caja", "type": "integer", "min": 1, "unit": "px"},
            {"path": "video.overlay_db", "label": "Quemar fecha y dB en clips", "type": "boolean"},
        ],
    },
    {
        "title": "Geometría y velocidad",
        "fields": [
            {"path": "video.detection_roi", "label": "ROI de inferencia", "type": "rect"},
            {"path": "video.detection_filter", "label": "Filtro de calzada", "type": "rect"},
            {"path": "speed.enabled", "label": "Calcular velocidad", "type": "boolean"},
            {"path": "speed.reference_left_x", "label": "Punto izquierdo", "type": "integer", "min": 0, "unit": "px"},
            {"path": "speed.reference_right_x", "label": "Punto derecho", "type": "integer", "min": 0, "unit": "px"},
            {"path": "speed.reference_distance_m", "label": "Distancia entre puntos", "type": "number", "step": 0.1, "min": 0.1, "unit": "m"},
        ],
    },
    {
        "title": "Tráfico y retención",
        "fields": [
            {"path": "tracker.record_classes", "label": "Vehículos a grabar", "type": "classes", "default": list(RECORD_CLASSES)},
            {"path": "tracker.max_distance", "label": "Distancia máxima del tracker", "type": "number", "step": 1, "min": 1, "unit": "px"},
            {"path": "tracker.min_displacement", "label": "Desplazamiento mínimo", "type": "number", "step": 1, "min": 0, "unit": "px"},
            {"path": "retention.clip_days", "label": "Retención de clips", "type": "integer", "min": 0, "unit": "días"},
            {"path": "retention.levels_days", "label": "Retención de niveles", "type": "integer", "min": 0, "unit": "días"},
        ],
    },
]

FIELD_MAP: dict[str, dict[str, Any]] = {
    f["path"]: f for group in FIELD_GROUPS for f in group["fields"]
}

_write_lock = threading.Lock()


def _get_path(config: dict, path: str) -> Any:
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _set_path(config: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = config
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _coerce(field: dict, raw: Any, path: str) -> Any:
    kind = field["type"]
    if kind == "boolean":
        if not isinstance(raw, bool):
            raise ValueError(f"{path} debe ser booleano")
        return raw
    if kind == "text":
        if not isinstance(raw, str):
            raise ValueError(f"{path} debe ser texto")
        return raw.strip()
    if kind == "select":
        allowed = {v for v, _ in field["options"]}
        if raw not in allowed:
            raise ValueError(f"{path} debe ser uno de {sorted(allowed)}")
        return raw
    if kind == "classes":
        if not isinstance(raw, list) or any(not isinstance(v, str) for v in raw):
            raise ValueError(f"{path} debe ser una lista de clases")
        unknown = sorted(set(raw) - set(RECORD_CLASSES))
        if unknown:
            raise ValueError(f"{path}: clases desconocidas {unknown}")
        return list(dict.fromkeys(raw))
    if kind in ("integer", "number"):
        if isinstance(raw, bool):
            raise ValueError(f"{path} debe ser numérico")
        try:
            value = int(raw) if kind == "integer" else float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{path} debe ser numérico") from None
        if kind == "number":
            if not (value == value and abs(value) != float("inf")):
                raise ValueError(f"{path} debe ser finito")
        lo, hi = field.get("min"), field.get("max")
        if lo is not None and value < lo:
            raise ValueError(f"{path} debe ser >= {lo}")
        if hi is not None and value > hi:
            raise ValueError(f"{path} debe ser <= {hi}")
        return value
    if kind == "rect":
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError(f"{path} debe ser [x1, y1, x2, y2]")
        vals = _coerce_list_int(raw, path)
        x1, y1, x2, y2 = vals
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"{path} requiere x1 < x2 e y1 < y2")
        return vals
    if kind == "size":
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError(f"{path} debe ser [ancho, alto]")
        vals = _coerce_list_int(raw, path)
        if any(v <= 0 for v in vals):
            raise ValueError(f"{path} requiere valores > 0")
        return vals
    raise ValueError(f"tipo de campo desconocido: {kind}")


def _coerce_list_int(raw: list, path: str) -> list[int]:
    out = []
    for v in raw:
        if isinstance(v, bool):
            raise ValueError(f"{path} debe contener enteros")
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            raise ValueError(f"{path} debe contener enteros") from None
    return out


def _validate_geometry(config: dict) -> None:
    width, height = config["video"]["resolution"]
    for path in ("video.detection_roi", "video.detection_filter"):
        rect = _get_path(config, path)
        if rect is None:
            continue
        x1, y1, x2, y2 = rect
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(f"{path} debe quedar dentro de 0..{width} x 0..{height}")
    speed = config.get("speed") or {}
    if speed.get("enabled"):
        left, right = speed.get("reference_left_x"), speed.get("reference_right_x")
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)) or left == right:
            raise ValueError("speed requiere puntos izquierdo y derecho distintos")
        if min(left, right) < 0 or max(left, right) > width:
            raise ValueError(f"los puntos de velocidad deben quedar dentro de 0..{width}")


def read_config(path: Path) -> Any:
    """Carga round-trip para conservar comentarios al guardar."""
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml_rt.load(fh)
    if not isinstance(data, dict):
        raise ConfigError("config.yaml no contiene un diccionario")
    return data


def public_payload(config: dict) -> dict:
    values = {}
    for path, field in FIELD_MAP.items():
        value = _get_path(config, path)
        values[path] = field.get("default") if value is None else value
    return {
        "groups": FIELD_GROUPS,
        "values": values,
        "vehicle_classes": list(RECORD_CLASSES),
    }


def apply_updates(config: Any, updates: dict[str, Any], base_dir: Path) -> tuple[Any, list[str]]:
    if not isinstance(updates, dict):
        raise ValueError("el cuerpo debe ser un objeto {ruta: valor}")
    unknown = sorted(set(updates) - set(FIELD_MAP))
    if unknown:
        raise ValueError(f"campos no editables: {unknown}")
    changed = []
    for path, raw in updates.items():
        field = FIELD_MAP[path]
        value = _coerce(field, raw, path)
        if _get_path(config, path) != value:
            _set_path(config, path, value)
            changed.append(path)
    criterion = str(config.get("triggers", {}).get("criterion", "dba"))
    if criterion == "psychoacoustic":
        for path, field in FIELD_MAP.items():
            if path.startswith("triggers.psychoacoustic.") and _get_path(config, path) is None:
                _set_path(config, path, field.get("default"))
    detector = str(config.get("video", {}).get("detector", "ssd")).lower()
    if detector == "yolo":
        if _get_path(config, "video.yolo_model_path") is None:
            _set_path(config, "video.yolo_model_path", FIELD_MAP["video.yolo_model_path"]["default"])
        if _get_path(config, "video.yolo_threshold") is None:
            _set_path(config, "video.yolo_threshold", FIELD_MAP["video.yolo_threshold"]["default"])
    if detector not in DETECTORS:
        raise ValueError(f"video.detector debe ser ssd|yolo (recibido: {detector!r})")
    if detector == "yolo" and not str(config["video"].get("yolo_model_path") or "").strip():
        raise ValueError("video.yolo_model_path es obligatorio con detector=yolo")
    validate_config(config, base_dir=base_dir)
    _validate_geometry(config)
    Settings.load(config)
    return config, changed


def save_config(path: Path, config: Any) -> None:
    """Escribe config.yaml de forma atómica, dejando un .bak de la versión previa."""
    path = Path(path)
    with _write_lock:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
        backup = path.with_name(path.name + ".bak")
        if path.exists():
            shutil.copy2(path, backup)
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            yaml_rt = YAML()
            yaml_rt.preserve_quotes = True
            with open(tmp, "w", encoding="utf-8") as fh:
                yaml_rt.dump(config, fh)
            os.chmod(tmp, mode)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
