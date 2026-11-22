#!/usr/bin/env python3
"""
Purga de datos antiguos del Noise Monitor.

- Clips MP4 con más de `retention.clip_days` días: se borran del disco y la
  fila de `noise_events` se conserva (estadística) con `clip_path` vacío.
- Filas de `audio_levels` con más de `retention.levels_days` días: se borran.
- Filas con `clip_path` que apunta a un archivo inexistente: se limpian.

Uso:
  python scripts/purge.py --dry-run     # solo informa
  python scripts/purge.py               # ejecuta
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.utils import load_config, resolve_path  # noqa: E402

DEFAULT_CLIP_DAYS = 30
DEFAULT_LEVELS_DAYS = 400


def _positive_int(value: str) -> int:
    """argparse type: un `--clip-days 0` (o negativo) borraba todos los clips."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"se esperaba un entero: {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"debe ser >= 1 (recibido: {number})")
    return number


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _clear_missing_clips(
    conn: sqlite3.Connection, table: str, cutoff: float, dry_run: bool
) -> int:
    """Vacía `clip_path` de las filas no fijadas cuyo archivo falta o es viejo.

    La tabla se pasa como literal desde este módulo (nunca desde la CLI).
    """
    if not _table_exists(conn, table):
        return 0
    cleared = 0
    rows = conn.execute(
        f"SELECT id, clip_path FROM {table}"
        " WHERE clip_path IS NOT NULL AND clip_path != '' AND pinned = 0"
    ).fetchall()
    for row_id, stored in rows:
        path = Path(stored)
        if not path.is_absolute():
            path = PROJECT_DIR / path
        try:
            keep = path.is_file() and path.stat().st_mtime >= cutoff
        except OSError:
            keep = False
        if not keep:
            cleared += 1
            if not dry_run:
                conn.execute(f"UPDATE {table} SET clip_path = '' WHERE id = ?", (row_id,))
    return cleared


def main() -> int:
    ap = argparse.ArgumentParser(description="Purga clips y niveles antiguos")
    ap.add_argument("--dry-run", action="store_true", help="no borra nada")
    ap.add_argument("--clip-days", type=_positive_int, default=None)
    ap.add_argument("--levels-days", type=_positive_int, default=None)
    ap.add_argument("--config", default=None, help="ruta a config.yaml")
    args = ap.parse_args()


    config = load_config(args.config or (PROJECT_DIR / "config.yaml"))
    ret = config.get("retention", {})
    clip_days = args.clip_days if args.clip_days is not None else int(ret.get("clip_days", DEFAULT_CLIP_DAYS))
    levels_days = args.levels_days if args.levels_days is not None else int(ret.get("levels_days", DEFAULT_LEVELS_DAYS))
    if clip_days < 1 or levels_days < 1:
        print(
            "retención inválida: clip_days/levels_days deben ser >= 1 "
            f"(recibido: {clip_days}/{levels_days})",
            file=sys.stderr,
        )
        return 2

    db_path = resolve_path(config, "database", PROJECT_DIR)
    clips_dir = resolve_path(config, "clips", PROJECT_DIR)
    now = time.time()
    clip_cutoff = now - clip_days * 86400
    levels_cutoff = (time.strftime("%Y-%m-%dT%H:%M", time.localtime(now - levels_days * 86400)))

    deleted_files = 0
    freed = 0
    cleared = 0

    # Los clips de eventos fijados no se borran nunca: son la evidencia
    # que el dueño marcó explícitamente.
    pinned_paths: set[Path] = set()
    try:
        conn = sqlite3.connect(str(db_path))
        for (stored,) in conn.execute(
            "SELECT clip_path FROM noise_events WHERE pinned = 1 AND clip_path != ''"
        ):
            p = Path(stored)
            pinned_paths.add((p if p.is_absolute() else PROJECT_DIR / p).resolve())
        for (stored,) in conn.execute(
            "SELECT clip_path FROM vehicle_passages WHERE pinned = 1 AND clip_path IS NOT NULL AND clip_path != ''"
        ):
            p = Path(stored)
            pinned_paths.add((p if p.is_absolute() else PROJECT_DIR / p).resolve())
        conn.close()
    except sqlite3.Error:
        pass

    if clips_dir.is_dir():
        for f in clips_dir.glob("*.mp4"):
            if f.resolve() in pinned_paths:
                continue
            try:
                mtime = f.stat().st_mtime
                size = f.stat().st_size
            except OSError:
                continue
            if mtime >= clip_cutoff:
                continue
            deleted_files += 1
            freed += size
            if not args.dry_run:
                f.unlink(missing_ok=True)

    conn = sqlite3.connect(str(db_path))
    try:
        cleared = _clear_missing_clips(conn, "noise_events", clip_cutoff, args.dry_run)
        # Los clips de pasaje viven en el mismo directorio y se borran por
        # mtime, pero su fila no se tocaba: el dashboard mostraba "▶ ver" para
        # un pasaje cuyo archivo ya no existía y el modal daba 404.
        cleared_passages = _clear_missing_clips(
            conn, "vehicle_passages", clip_cutoff, args.dry_run
        )

        if _table_exists(conn, "audio_levels"):
            old_levels = conn.execute(
                "SELECT COUNT(*) FROM audio_levels WHERE minute < ?", (levels_cutoff,)
            ).fetchone()[0]
            if not args.dry_run:
                conn.execute("DELETE FROM audio_levels WHERE minute < ?", (levels_cutoff,))
        else:
            old_levels = 0
        conn.commit()
        if not args.dry_run and (deleted_files or cleared or cleared_passages or old_levels):
            try:
                conn.execute("VACUUM")
            except sqlite3.OperationalError as exc:
                print(f"aviso: VACUUM no se pudo ejecutar: {exc}", file=sys.stderr)
    finally:
        conn.close()

    mode = "DRY-RUN" if args.dry_run else "hecho"
    print(
        f"[{mode}] clips>={clip_days}d borrados={deleted_files} "
        f"({freed / 1024 / 1024:.1f} MB), filas sin clip={cleared} eventos "
        f"+ {cleared_passages} pasajes, "
        f"niveles>={levels_days}d borrados={old_levels}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
