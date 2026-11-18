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


def main() -> int:
    ap = argparse.ArgumentParser(description="Purga clips y niveles antiguos")
    ap.add_argument("--dry-run", action="store_true", help="no borra nada")
    ap.add_argument("--clip-days", type=int, default=None)
    ap.add_argument("--levels-days", type=int, default=None)
    args = ap.parse_args()

    config = load_config(PROJECT_DIR / "config.yaml")
    ret = config.get("retention", {})
    clip_days = args.clip_days if args.clip_days is not None else int(ret.get("clip_days", DEFAULT_CLIP_DAYS))
    levels_days = args.levels_days if args.levels_days is not None else int(ret.get("levels_days", DEFAULT_LEVELS_DAYS))

    db_path = resolve_path(config, "database", PROJECT_DIR)
    clips_dir = resolve_path(config, "clips", PROJECT_DIR)
    now = time.time()
    clip_cutoff = now - clip_days * 86400
    levels_cutoff = (time.strftime("%Y-%m-%dT%H:%M", time.localtime(now - levels_days * 86400)))

    deleted_files = 0
    freed = 0
    cleared = 0

    if clips_dir.is_dir():
        for f in clips_dir.glob("*.mp4"):
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
        rows = conn.execute(
            "SELECT id, clip_path FROM noise_events WHERE clip_path IS NOT NULL AND clip_path != ''"
        ).fetchall()
        for event_id, stored in rows:
            path = Path(stored)
            if not path.is_absolute():
                path = PROJECT_DIR / path
            keep = False
            try:
                keep = path.is_file() and path.stat().st_mtime >= clip_cutoff
            except OSError:
                keep = False
            if not keep:
                cleared += 1
                if not args.dry_run:
                    conn.execute("UPDATE noise_events SET clip_path = '' WHERE id = ?", (event_id,))

        old_levels = conn.execute(
            "SELECT COUNT(*) FROM audio_levels WHERE minute < ?", (levels_cutoff,)
        ).fetchone()[0]
        if not args.dry_run:
            conn.execute("DELETE FROM audio_levels WHERE minute < ?", (levels_cutoff,))
        conn.commit()
        if not args.dry_run and (deleted_files or cleared or old_levels):
            conn.execute("VACUUM")
    finally:
        conn.close()

    mode = "DRY-RUN" if args.dry_run else "hecho"
    print(
        f"[{mode}] clips>={clip_days}d borrados={deleted_files} "
        f"({freed / 1024 / 1024:.1f} MB), filas sin clip={cleared}, "
        f"niveles>={levels_days}d borrados={old_levels}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
