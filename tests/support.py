"""
Utilidades compartidas por los tests.

Antes, cada test cargaba `config.yaml` del checkout, que está en .gitignore:
en un clone limpio fallaban 4 de 6 con FileNotFoundError. Ahora se parte de
`config.example.yaml` (versionado) y se redirigen las rutas a un temporal.
"""

import sys
import tempfile
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def example_config(base_dir: str | Path, name: str = "config.example.yaml") -> dict:
    """Config de ejemplo con base de datos y clips dentro de `base_dir`."""
    cfg = yaml.safe_load((PROJECT_DIR / name).read_text(encoding="utf-8"))
    base = Path(base_dir)
    cfg.setdefault("paths", {})
    cfg["paths"]["database"] = str(base / "events.db")
    cfg["paths"]["clips"] = str(base / "clips")
    cfg["paths"]["reports"] = str(base / "reports")
    (base / "clips").mkdir(parents=True, exist_ok=True)
    return cfg


def resolve_stored(stored: str, base_dir: str | Path = PROJECT_DIR) -> Path:
    """Ruta absoluta de un `clip_path` guardado (relativo o absoluto)."""
    path = Path(stored)
    return path if path.is_absolute() else Path(base_dir) / path


def detection(cls: str = "car", cx: float = 100.0, cy: float = 100.0,
              score: float = 0.7, lane: str = "left") -> dict:
    return {
        "class": cls,
        "score": score,
        "cx": cx,
        "cy": cy,
        "lane": lane,
        "bbox": [0.4, 0.1, 0.5, 0.2],
    }


def tmpdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory()
