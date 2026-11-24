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


LEGACY_SCHEMA = PROJECT_DIR / "tests" / "fixtures" / "schema_v2.sql"


def create_legacy_schema(db_path: str | Path) -> None:
    """Crea el esquema anterior a `events` (schema.sql de d757f4c, congelado).

    Es la entrada de la migración. Se aplica como lo hacía init_database
    entonces: sentencia por sentencia, tolerando los ALTER ya aplicados.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        for statement in LEGACY_SCHEMA.read_text(encoding="utf-8").split(";"):
            if not any(l.strip() and not l.strip().startswith("--")
                       for l in statement.splitlines()):
                continue
            try:
                conn.execute(statement)
            except sqlite3.OperationalError:
                pass
        conn.commit()
    finally:
        conn.close()


def create_events_table(db_path: str | Path) -> None:
    """Crea `events` con el DDL de la migración (idempotente).

    init_database ya la crea desde schema.sql; los tests que no pasan por
    init_database usan este helper.
    """
    import sqlite3

    scripts = str(PROJECT_DIR / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from migrate_unified import EVENTS_DDL_STATEMENTS

    conn = sqlite3.connect(str(db_path))
    try:
        for statement in EVENTS_DDL_STATEMENTS:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def insert_event(db_path: str | Path, kind: str, when, **cols) -> int:
    """Inserta una fila en `events` con ventana [when, when + 2 s]; devuelve el id."""
    import sqlite3

    cols = {"kind": kind, "timestamp": when.strftime("%Y-%m-%dT%H:%M:%S"),
            "start_ts": when.timestamp(), "end_ts": when.timestamp() + 2, **cols}
    names = ", ".join(cols)
    marks = ", ".join("?" for _ in cols)
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute(f"INSERT INTO events ({names}) VALUES ({marks})", tuple(cols.values()))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()
