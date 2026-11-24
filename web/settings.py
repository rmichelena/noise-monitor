"""
Ajustes editables del pipeline, con persistencia en config.yaml.

La UI manda un diccionario `{ruta: valor}`; aquí se valida contra un esquema
cerrado (nada de escribir claves arbitrarias), se aplica sobre una copia
round-trip del YAML para conservar comentarios y se reemplaza el archivo de
forma atómica, dejando un `.bak`. Antes de escribir, la config completa pasa
por `validate_config` (que incluye `TriggerPolicy.from_config`): un guardado
nunca deja un config.yaml con el que el servicio no arranque.

Los cambios requieren reiniciar el servicio: los monitores leen la
configuración al arrancar.
"""

from __future__ import annotations

import math
import os
import shutil
import stat
import tempfile
import threading
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import CommentMark
from ruamel.yaml.tokens import CommentToken
from ruamel.yaml.scalarstring import DoubleQuotedScalarString
from ruamel.yaml.util import load_yaml_guess_indent

from src.perceptual_trigger import Settings
from src.schedule import DEFAULT_END, DEFAULT_START, _parse_hhmm
from src.utils import ConfigError, load_config, validate_config
from src.vehicle_tracker import VEHICLE_CLASSES

DETECTORS = ("ssd", "yolo")

# Grupos de clases que la UI muestra juntos. Motos y bicis van en uno solo:
# el detector las confunde con facilidad.
CLASS_GROUPS: list[dict[str, Any]] = [
    {"key": "cars", "label": "Autos", "classes": ["car"]},
    {"key": "trucks", "label": "Camiones", "classes": ["truck"]},
    {"key": "buses", "label": "Buses", "classes": ["bus"]},
    {"key": "two_wheels", "label": "Motos y bicis", "classes": ["motorcycle", "bicycle"]},
]
assert {c for g in CLASS_GROUPS for c in g["classes"]} == set(VEHICLE_CLASSES)

# Defaults de tiempo por métrica: los mismos que aplica
# `perceptual_trigger.Settings.load` cuando la clave falta. PA tarda ~5,6 s por
# ventana; con los de loudness el trigger acumularía ventanas y descartaría.
PSYCHO_TIMING: dict[str, dict[str, float]] = {
    "loudness_mean": {"window_seconds": 3.0, "hop_seconds": 1.5, "expected_latency_seconds": 1.6},
    "annoyance_pa": {"window_seconds": 2.0, "hop_seconds": 2.0, "expected_latency_seconds": 5.6},
}
# Con criterio psicoacústico, la calibración es obligatoria: se escribe aunque
# coincida con el default (Settings.load no acepta que falte).
PSYCHO_REQUIRED = ("audio.pressure_calibration.id", "audio.pressure_calibration.pa_per_unit")

DAYS = [[1, "Lun"], [2, "Mar"], [3, "Mié"], [4, "Jue"], [5, "Vie"], [6, "Sáb"], [7, "Dom"]]


def _schedule_fields(section: str) -> list[dict[str, Any]]:
    base = f"triggers.{section}_schedule"
    return [
        {"path": f"{base}.enabled", "label": "Usar horario", "type": "boolean", "default": False},
        {"path": f"{base}.start", "label": "Desde", "type": "time", "default": DEFAULT_START},
        {"path": f"{base}.end", "label": "Hasta", "type": "time", "default": DEFAULT_END,
         "hint": "Si «hasta» es menor que «desde», la ventana cruza la medianoche."},
        {"path": f"{base}.days", "label": "Días", "type": "days", "default": [],
         "hint": "Ninguno marcado = todos. En una ventana nocturna cuenta el día en que empieza."},
    ]


# Campos que la UI puede mostrar y guardar. La ruta usa puntos; el backend
# rechaza cualquier ruta que no esté aquí.
FIELD_GROUPS: list[dict[str, Any]] = [
    {
        "key": "noise",
        "title": "Ruido",
        "toggle": "triggers.noise_enabled",
        "fields": [
            {"path": "triggers.noise_enabled", "label": "Detectar eventos de ruido", "type": "boolean", "default": True},
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
            {"path": "triggers.cooldown_seconds", "label": "Enfriamiento entre eventos", "type": "number", "step": 0.1, "min": 0, "unit": "s", "default": 0},
            {"path": "triggers.noise_only_with_traffic", "label": "Sólo con un pasaje de tráfico asociado", "type": "boolean", "default": False},
            *_schedule_fields("noise"),
        ],
    },
    {
        "key": "psycho",
        "title": "Ruido · psicoacústico y calibración",
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
            {"path": "audio.calibration_offset", "label": "Offset de calibración", "type": "number", "step": 0.1, "min": -200, "max": 200, "unit": "dB"},
        ],
    },
    {
        "key": "traffic",
        "title": "Tráfico",
        "toggle": "triggers.traffic_enabled",
        "fields": [
            {"path": "triggers.traffic_enabled", "label": "Registrar pasajes de tráfico", "type": "boolean", "default": True},
            {"path": "tracker.record_classes", "label": "Vehículos a registrar", "type": "class_groups",
             "default": [c for g in CLASS_GROUPS for c in g["classes"]]},
            {"path": "triggers.traffic_only_with_noise", "label": "Sólo con un evento de ruido asociado", "type": "boolean", "default": False},
            *_schedule_fields("traffic"),
            {"path": "tracker.max_distance", "label": "Distancia máxima del tracker", "type": "number", "step": 1, "min": 1, "unit": "px"},
            {"path": "tracker.max_disappeared", "label": "Frames sin ver antes de salir", "type": "integer", "min": 0},
            {"path": "tracker.min_track_seconds", "label": "Duración mínima del pasaje", "type": "number", "step": 0.1, "min": 0, "unit": "s"},
            {"path": "tracker.min_displacement", "label": "Desplazamiento mínimo", "type": "number", "step": 1, "min": 0, "unit": "px"},
            {"path": "tracker.max_track_seconds", "label": "Duración máxima del pasaje", "type": "number", "step": 0.1, "min": 0, "unit": "s"},
            {"path": "retention.clip_days", "label": "Retención de clips", "type": "integer", "min": 1, "unit": "días", "default": 30},
            {"path": "retention.levels_days", "label": "Retención de niveles", "type": "integer", "min": 1, "unit": "días", "default": 400},
        ],
    },
    {
        "key": "video",
        "title": "Video y detector",
        "fields": [
            {"path": "video.detector", "label": "Modelo", "type": "select", "default": "ssd", "options": [["ssd", "SSD MobileNet"], ["yolo", "YOLO11"]]},
            {"path": "video.yolo_model_path", "label": "Ruta del modelo YOLO", "type": "text", "default": "models/yolo11n_160x640_edgetpu.tflite"},
            {"path": "video.yolo_threshold", "label": "Confianza YOLO", "type": "number", "step": 0.01, "min": 0, "max": 1, "default": 0.25},
            {"path": "video.resolution", "label": "Resolución", "type": "size", "unit": "px"},
            {"path": "video.framerate", "label": "Framerate", "type": "integer", "min": 1, "max": 240, "unit": "fps"},
            {"path": "video.inference_interval", "label": "Cada N frames", "type": "integer", "min": 1},
            {"path": "video.detection_threshold", "label": "Confianza mínima", "type": "number", "step": 0.01, "min": 0, "max": 1},
            {"path": "video.detection_min_width", "label": "Ancho mínimo de caja", "type": "integer", "min": 1, "unit": "px"},
            {"path": "video.detection_max_width", "label": "Ancho máximo de caja", "type": "integer", "min": 1, "unit": "px"},
            {"path": "video.detection_min_height", "label": "Alto mínimo de caja", "type": "integer", "min": 1, "unit": "px"},
            {"path": "video.detection_max_height", "label": "Alto máximo de caja", "type": "integer", "min": 1, "unit": "px"},
            {"path": "video.overlay_db", "label": "Quemar fecha y dB en clips", "type": "boolean", "default": True},
        ],
    },
    {
        "key": "geometry",
        "title": "Geometría y velocidad",
        "fields": [
            {"path": "video.detection_roi", "label": "ROI de inferencia", "type": "rect"},
            {"path": "video.detection_filter", "label": "Filtro de calzada", "type": "rect"},
            {"path": "speed.enabled", "label": "Calcular velocidad", "type": "boolean", "default": False},
            {"path": "speed.reference_left_x", "label": "Punto izquierdo", "type": "integer", "min": 0, "unit": "px"},
            {"path": "speed.reference_right_x", "label": "Punto derecho", "type": "integer", "min": 0, "unit": "px"},
            {"path": "speed.reference_distance_m", "label": "Distancia entre puntos", "type": "number", "step": 0.1, "min": 0.1, "unit": "m"},
        ],
    },
]

FIELD_MAP: dict[str, dict[str, Any]] = {
    f["path"]: f for group in FIELD_GROUPS for f in group["fields"]
}

# Reentrante: update_config lo toma para leer+validar+escribir y save_config
# lo vuelve a tomar. Dos POST simultáneos no se pisan los cambios.
_write_lock = threading.RLock()


def _get_path(config: dict, path: str) -> Any:
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _trailing_slot(mapping: Any, key: Any) -> tuple[list, int] | None:
    """Dónde guarda ruamel el comentario que sigue al último valor de `key`.

    Baja por el último elemento de mapas y listas anidados: ahí cuelgan los
    comentarios y líneas en blanco que separan este bloque del siguiente.
    Devuelve (lista de comentarios del ítem, índice del token) o None.
    """
    container, item = mapping, key
    while True:
        value = container[item]
        if isinstance(value, CommentedMap) and len(value):
            container, item = value, list(value)[-1]
            continue
        if isinstance(value, CommentedSeq) and len(value):
            slot = value.ca.items.setdefault(len(value) - 1, [None, None])
            return slot, 0
        if not isinstance(container, CommentedMap):
            return None
        return container.ca.items.setdefault(item, [None, None, None, None]), 2


def _add_key(mapping: Any, key: str, value: Any) -> None:
    """Agrega una clave al final sin dejarla debajo del comentario del bloque siguiente.

    ruamel cuelga del último valor del mapa el comentario de fin de línea y
    todo lo que sigue hasta la próxima clave (líneas en blanco, el
    encabezado de la sección siguiente). Una clave nueva agregada al final
    quedaría después de ese encabezado; se mueve la cola a la clave nueva.
    """
    previous = list(mapping)[-1] if isinstance(mapping, CommentedMap) and len(mapping) else None
    slot = _trailing_slot(mapping, previous) if previous is not None else None
    tail = ""
    if slot is not None and slot[0][slot[1]] is not None:
        token = slot[0][slot[1]]
        head, _, tail = token.value.partition("\n")
        if tail:
            token.value = head + "\n"
    if isinstance(value, list) and not isinstance(value, CommentedSeq):
        value = CommentedSeq(value)
    mapping[key] = value
    if tail:
        new_slot = _trailing_slot(mapping, key)
        if new_slot is not None:
            new_slot[0][new_slot[1]] = CommentToken("\n" + tail, CommentMark(0), None)


def _set_path(config: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = config
    for i, part in enumerate(parts[:-1]):
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            # Se arma la rama completa y se agrega de una vez: así la cola de
            # comentarios se mueve a su última hoja y no queda colgando de
            # una sección todavía vacía.
            branch: Any = value
            for name in reversed(parts[i + 1:]):
                node = CommentedMap()
                node[name] = CommentedSeq(branch) if isinstance(branch, list) else branch
                branch = node
            _add_key(cur, part, branch)
            return
        cur = nxt
    if parts[-1] in cur:
        # Reemplazar una lista o un mapa descarta los comentarios que
        # colgaban de su último elemento (el encabezado de la sección
        # siguiente): se trasladan al valor nuevo.
        old = _trailing_slot(cur, parts[-1])
        token = old[0][old[1]] if old is not None else None
        cur[parts[-1]] = CommentedSeq(value) if isinstance(value, list) else value
        new = _trailing_slot(cur, parts[-1])
        if token is not None and new is not None and new[0][new[1]] is None:
            new[0][new[1]] = token
    else:
        _add_key(cur, parts[-1], value)


def _coerce_list_int(raw: list, path: str) -> list[int]:
    out = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v):
            raise ValueError(f"{path} debe contener enteros (recibido: {raw!r})")
        out.append(int(v))
    return out


def _coerce(field: dict, raw: Any, path: str) -> Any:
    kind = field["type"]
    if kind == "boolean":
        if not isinstance(raw, bool):
            raise ValueError(f"{path} debe ser booleano (recibido: {raw!r})")
        return raw
    if kind == "text":
        if not isinstance(raw, str):
            raise ValueError(f"{path} debe ser texto")
        return raw.strip()
    if kind == "select":
        allowed = {v for v, _ in field["options"]}
        if raw not in allowed:
            raise ValueError(f"{path} debe ser uno de {sorted(allowed)} (recibido: {raw!r})")
        return raw
    if kind == "time":
        # Entre comillas: sin ellas, PyYAML (YAML 1.1, el loader del servicio)
        # lee `19:00` como sexagesimal, el entero 1140.
        return DoubleQuotedScalarString(_parse_hhmm(raw, path).strftime("%H:%M"))
    if kind == "days":
        if not isinstance(raw, list) or any(
            isinstance(d, bool) or not isinstance(d, int) or not 1 <= d <= 7 for d in raw
        ):
            raise ValueError(f"{path} debe ser una lista de días 1–7 (recibido: {raw!r})")
        return sorted(set(raw))
    if kind == "class_groups":
        if not isinstance(raw, list) or any(not isinstance(v, str) for v in raw):
            raise ValueError(f"{path} debe ser una lista de clases")
        unknown = sorted(set(raw) - set(VEHICLE_CLASSES))
        if unknown:
            raise ValueError(f"{path}: clases desconocidas {unknown}")
        chosen = set(raw)
        for group in CLASS_GROUPS:
            members = set(group["classes"])
            if chosen & members and not members <= chosen:
                raise ValueError(
                    f"{path}: «{group['label']}» va completo ({', '.join(group['classes'])})"
                )
        # Orden estable, el de los grupos.
        return [c for g in CLASS_GROUPS for c in g["classes"] if c in chosen]
    if kind in ("integer", "number"):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{path} debe ser numérico (recibido: {raw!r})")
        if not math.isfinite(raw):
            raise ValueError(f"{path} debe ser finito")
        if kind == "integer":
            if raw != int(raw):
                raise ValueError(f"{path} debe ser entero (recibido: {raw!r})")
            value: int | float = int(raw)
        else:
            # Un entero se queda entero: `30` no se reescribe como `30.0`.
            value = raw if isinstance(raw, int) else float(raw)
        lo, hi = field.get("min"), field.get("max")
        if lo is not None and value < lo:
            raise ValueError(f"{path} debe ser >= {lo} (recibido: {raw!r})")
        if hi is not None and value > hi:
            raise ValueError(f"{path} debe ser <= {hi} (recibido: {raw!r})")
        return value
    if kind == "rect":
        if not isinstance(raw, list) or len(raw) != 4:
            raise ValueError(f"{path} debe ser [x1, y1, x2, y2]")
        x1, y1, x2, y2 = vals = _coerce_list_int(raw, path)
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


# Indentado de listas de config.example.yaml: `- x` dos espacios más adentro
# que su clave. Producción usa listas sin indentar (`sequence=2, offset=0`).
DEFAULT_SEQUENCE_INDENT = (4, 2)


def _sequence_indent(text: str) -> tuple[int, int]:
    """(sequence, offset) del archivo, para reescribirlo con su propio estilo.

    Ningún valor fijo sirve para los dos estilos en uso: cada uno reformatea
    todas las listas del otro (~60 líneas por guardado).
    """
    try:
        _, sequence, offset = load_yaml_guess_indent(text)
    except Exception:
        return DEFAULT_SEQUENCE_INDENT
    if sequence is None or offset is None:
        return DEFAULT_SEQUENCE_INDENT
    return sequence, offset


def _yaml(sequence_indent: tuple[int, int] = DEFAULT_SEQUENCE_INDENT) -> YAML:
    yaml_rt = YAML()
    yaml_rt.preserve_quotes = True
    # Mismo estilo que el archivo: indentado de listas detectado y `null`
    # explícito. Si no, cada guardado reformateaba líneas que nadie tocó.
    sequence, offset = sequence_indent
    yaml_rt.indent(mapping=2, sequence=sequence, offset=offset)
    yaml_rt.representer.add_representer(
        type(None), lambda rep, _: rep.represent_scalar("tag:yaml.org,2002:null", "null")
    )
    return yaml_rt


def read_config(path: Path) -> Any:
    """Carga round-trip para conservar comentarios al guardar."""
    with open(path, "r", encoding="utf-8") as fh:
        data = _yaml().load(fh)
    if not isinstance(data, dict):
        raise ConfigError("config.yaml no contiene un diccionario")
    return data


def _plain(value: Any) -> Any:
    """CommentedSeq/CommentedMap → list/dict, para JSON y comparaciones."""
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain(v) for v in value]
    return value


def _metric(config: dict) -> str:
    metric = _get_path(config, "triggers.psychoacoustic.metric")
    return metric if metric in PSYCHO_TIMING else "loudness_mean"


def _default(config: dict, path: str) -> Any:
    """Default de un campo; los tiempos psicoacústicos dependen de la métrica."""
    key = path.rsplit(".", 1)[-1]
    if path.startswith("triggers.psychoacoustic.") and key in PSYCHO_TIMING["loudness_mean"]:
        return PSYCHO_TIMING[_metric(config)][key]
    return FIELD_MAP[path].get("default")


def public_payload(config: dict) -> dict:
    values = {}
    for path in FIELD_MAP:
        value = _get_path(config, path)
        values[path] = _plain(_default(config, path) if value is None else value)
    return {
        "groups": FIELD_GROUPS,
        "values": values,
        "class_groups": CLASS_GROUPS,
        "days": DAYS,
        "psycho_timing": PSYCHO_TIMING,
    }


def _same(field: dict, a: Any, b: Any) -> bool:
    """Igualdad de valores; las listas de clases y días no dependen del orden."""
    if field["type"] in ("class_groups", "days") and isinstance(a, list) and isinstance(b, list):
        return set(a) == set(b)
    return a == b


def apply_updates(config: Any, updates: dict[str, Any], base_dir: Path) -> tuple[Any, list[str]]:
    """Aplica `{ruta: valor}` sobre la config y valida la config completa.

    Sólo escribe los campos cuyo valor cambia: un campo ausente que llega con
    su default queda ausente. Lanza ValueError/ConfigError sin tocar el disco.
    """
    if not isinstance(updates, dict):
        raise ValueError("el cuerpo debe ser un objeto {ruta: valor}")
    unknown = sorted(set(updates) - set(FIELD_MAP))
    if unknown:
        raise ValueError(f"campos no editables: {unknown}")
    changed = []
    # Lo que la página mostraba para un campo ausente: el default con la config
    # previa (la métrica de antes, para los tiempos psicoacústicos).
    shown = {path: _default(config, path) for path in updates}
    for path, raw in updates.items():
        field = FIELD_MAP[path]
        value = _coerce(field, raw, path)
        current = _get_path(config, path)
        if current is None and _same(field, value, shown[path]):
            continue  # ausente y con su default: sigue ausente
        if current is None or not _same(field, _plain(current), value):
            _set_path(config, path, value)
            changed.append(path)
    criterion = str((config.get("triggers") or {}).get("criterion", "dba"))
    if criterion == "psychoacoustic":
        # Settings.load exige estos valores: se completan con los defaults de
        # la métrica elegida (un valor salteado arriba era justo ese default).
        for path in FIELD_MAP:
            if not (path.startswith("triggers.psychoacoustic.") or path in PSYCHO_REQUIRED):
                continue
            if _get_path(config, path) is not None:
                continue
            _set_path(config, path, _default(config, path))
            changed.append(path)
    detector = str((config.get("video") or {}).get("detector", "ssd")).lower()
    if detector not in DETECTORS:
        raise ValueError(f"video.detector debe ser ssd|yolo (recibido: {detector!r})")
    if detector == "yolo":
        for path in ("video.yolo_model_path", "video.yolo_threshold"):
            if _get_path(config, path) is None:
                _set_path(config, path, FIELD_MAP[path]["default"])
        if not str(config["video"].get("yolo_model_path") or "").strip():
            raise ValueError("video.yolo_model_path es obligatorio con detector=yolo")
    # validate_config incluye TriggerPolicy.from_config: horarios, días, clases
    # y coherencia de flags se validan sobre la config completa resultante.
    validate_config(config, base_dir=base_dir)
    _validate_geometry(config)
    if Settings.load(config) is not None:
        # main.py sale con 78 (sin reinicio) si falta la dependencia: mejor
        # rechazar el guardado que dejar un servicio que no arranca.
        try:
            import phonometry  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "el criterio psicoacústico requiere phonometry (requirements-trigger.txt) "
                "en el entorno del servicio"
            ) from exc
    return config, changed


def save_config(path: Path, config: Any) -> Path:
    """Escribe config.yaml de forma atómica, dejando un .bak de la versión previa."""
    path = Path(path)
    backup = path.with_name(path.name + ".bak")
    with _write_lock:
        mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
        style = (
            _sequence_indent(path.read_text(encoding="utf-8"))
            if path.exists() else DEFAULT_SEQUENCE_INDENT
        )
        fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                _yaml(style).dump(config, fh)
                fh.flush()
                os.fsync(fh.fileno())
            # Validar los bytes escritos tal como los leerá el servicio
            # (PyYAML), no el objeto de ruamel: los dos loaders no coinciden
            # en todo (p. ej. `19:00`), y lo que importa es que arranque.
            try:
                validate_config(load_config(tmp), base_dir=path.parent)
            except ConfigError as exc:
                raise ConfigError(f"el archivo resultante no sería válido: {exc}") from exc
            os.chmod(tmp, mode)
            # El .bak se toma recién ahora: un guardado rechazado no lo pisa.
            if path.exists():
                shutil.copy2(path, backup)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()
    return backup


def update_config(path: Path, updates: dict[str, Any], base_dir: Path) -> tuple[list[str], Path | None]:
    """Lee, aplica, valida y guarda bajo un mismo lock. Devuelve (cambiados, backup)."""
    with _write_lock:
        config = read_config(path)
        config, changed = apply_updates(config, updates, base_dir)
        if not changed:
            return changed, None
        return changed, save_config(path, config)
