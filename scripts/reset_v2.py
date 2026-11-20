#!/usr/bin/env python3
"""
Reinicia la base al esquema v2 (un solo tipo de evento).

Renombra la base existente a events.db.v1.<timestamp> y, con --drop-clips,
borra los clips. No toca el servicio: páralo antes.

  sudo systemctl stop noise-monitor
  venv/bin/python scripts/reset_v2.py --yes --drop-clips
  sudo systemctl start noise-monitor
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.utils import init_database, load_config, resolve_path  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="confirmar el borrado")
    ap.add_argument("--drop-clips", action="store_true", help="borrar los clips viejos")
    args = ap.parse_args()
    if not args.yes:
        print("Se necesita --yes para confirmar")
        return 1

    config = load_config(PROJECT_DIR / "config.yaml")
    db = resolve_path(config, "database", PROJECT_DIR)
    clips = resolve_path(config, "clips", PROJECT_DIR)

    if db.exists():
        backup = db.with_suffix(f".db.v1.{int(time.time())}")
        shutil.move(str(db), str(backup))
        print(f"base anterior -> {backup}")
    if args.drop_clips and clips.is_dir():
        files = list(clips.glob("*.mp4"))
        for f in files:
            f.unlink(missing_ok=True)
        print(f"clips borrados: {len(files)}")

    init_database(config, PROJECT_DIR)
    print(f"base v2 creada en {db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
