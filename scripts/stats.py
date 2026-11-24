#!/usr/bin/env python3
"""
Consulta estadísticas del pipeline de monitoreo de ruido sobre la tabla
única `events`: pasajes = traffic + both, eventos de ruido = noise + both.

Uso: python scripts/stats.py [--limit N] [--days D]
"""

import argparse
import math
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def _positive_int(value: str) -> int:
    """argparse type: `--limit 0` imprimía cabeceras sin filas y `-1` (que
    SQLite lee como "sin límite") volcaba la tabla entera."""
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"se esperaba un entero: {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"debe ser >= 1 (recibido: {number})")
    return number


def _fmt(value, fmt: str = "{:.1f}", dash: str = "—") -> str:
    """Formatea opcionales de SQL sin reventar con NULL ni con -inf."""
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return dash
    if not math.isfinite(number):
        return dash
    return fmt.format(number)



def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


PASSAGES = "kind IN ('traffic', 'both')"
NOISE = "kind IN ('noise', 'both')"


def _and(where: str, clause: str) -> str:
    """Agrega `clause` al WHERE del periodo (que puede venir vacío)."""
    return f"{where} AND {clause}" if where else f"WHERE {clause}"


def print_passages(conn: sqlite3.Connection, where: str, params: list, limit: int):
    """Pasajes del tracker (traffic + both)."""
    where = _and(where, PASSAGES)

    total = conn.execute(
        f"SELECT COUNT(*) FROM events {where}", params
    ).fetchone()[0]
    print(f"\n{'=' * 60}")
    print(f"VEHÍCULOS DETECTADOS (tracker): {total} pasajes")
    print("=" * 60)

    if total == 0:
        return

    print("\n--- Por clase de vehículo ---")
    for row in conn.execute(
        f"SELECT vehicle_class, COUNT(*) as n FROM events {where} "
        "GROUP BY vehicle_class ORDER BY n DESC",
        params,
    ).fetchall():
        print(f"  {row['vehicle_class']}: {row['n']}")

    print("\n--- Por carril ---")
    for row in conn.execute(
        f"SELECT lane, COUNT(*) as n FROM events {where} "
        "GROUP BY lane ORDER BY n DESC",
        params,
    ).fetchall():
        print(f"  {row['lane'] or 'N/A'}: {row['n']}")

    # traffic: nivel al pasar; both: el del ruido que provocó.
    print("\n--- dB del pasaje ---")
    row = conn.execute(
        f"SELECT MIN(db_max) as lo, MAX(db_max) as hi, AVG(db_max) as avg "
        f"FROM events {_and(where, 'db_max > 0')}",
        params,
    ).fetchone()
    if row["lo"] is not None:
        print(
            f"  Mín: {_fmt(row['lo'])} dB  |  Máx: {_fmt(row['hi'])} dB"
            f"  |  Prom: {_fmt(row['avg'])} dB"
        )

    linked = conn.execute(
        f"SELECT COUNT(*) FROM events {_and(where, "kind = 'both'")}", params,
    ).fetchone()[0]
    print(f"\n  Pasajes con evento de ruido asociado: {linked} / {total}")

    print(f"\n--- Últimos {limit} pasajes ---")
    for row in conn.execute(
        f"SELECT id, kind, timestamp, vehicle_class, lane, db_max, detection_score "
        f"FROM events {where} ORDER BY timestamp DESC LIMIT ?",
        params + [limit],
    ).fetchall():
        noisy = "con ruido" if row["kind"] == "both" else "silencioso"
        print(
            f"  #{row['id']} {row['timestamp']} | {row['vehicle_class']} | "
            f"{row['lane'] or '-'} | {_fmt(row['db_max'])} dB | "
            f"score {_fmt(row['detection_score'], '{:.2f}')} | {noisy}"
        )


def print_noise_events(conn: sqlite3.Connection, where: str, params: list, limit: int):
    """Eventos de ruido (noise + both)."""
    where = _and(where, NOISE)

    total = conn.execute(
        f"SELECT COUNT(*) FROM events {where}", params
    ).fetchone()[0]
    print(f"\n{'=' * 60}")
    print(f"EVENTOS DE RUIDO: {total} total")
    print("=" * 60)

    if total == 0:
        return

    print("\n--- Por categoría de sonido ---")
    for row in conn.execute(
        f"SELECT sound_category, COUNT(*) as n FROM events {where} "
        "GROUP BY sound_category ORDER BY n DESC",
        params,
    ).fetchall():
        print(f"  {row['sound_category'] or 'sin clasificar'}: {row['n']}")

    print("\n--- Por trigger ---")
    for row in conn.execute(
        f"SELECT trigger_type, COUNT(*) as n FROM events {where} "
        "GROUP BY trigger_type ORDER BY n DESC",
        params,
    ).fetchall():
        print(f"  {row['trigger_type'] or 'N/A'}: {row['n']}")

    vehicular = conn.execute(
        f"SELECT vehicle_class, COUNT(*) as n FROM events "
        f"{_and(where, "sound_category = 'vehicle'")} "
        "GROUP BY vehicle_class ORDER BY n DESC",
        params,
    ).fetchall()
    if vehicular:
        print("\n--- Eventos vehiculares por clase ---")
        for row in vehicular:
            print(f"  {row['vehicle_class'] or 'unknown'}: {row['n']}")

    print("\n--- Nivel sonoro (dB) ---")
    row = conn.execute(
        f"SELECT MIN(db_max) as lo, MAX(db_max) as hi, AVG(db_max) as avg "
        f"FROM events {where}",
        params,
    ).fetchone()
    if row["lo"] is not None:
        print(
            f"  Mín: {_fmt(row['lo'])} dB  |  Máx: {_fmt(row['hi'])} dB"
            f"  |  Prom: {_fmt(row['avg'])} dB"
        )

    print(f"\n--- Top 5 más ruidosos ---")
    for row in conn.execute(
        f"SELECT id, timestamp, db_max, sound_category, sound_class, vehicle_class "
        f"FROM events {where} ORDER BY db_max DESC LIMIT 5",
        params,
    ).fetchall():
        print(
            f"  #{row['id']} {row['timestamp']} | {_fmt(row['db_max'])} dB | "
            f"{row['sound_category'] or '?'}/{row['sound_class'] or '?'} | veh: {row['vehicle_class'] or '-'}"
        )

    print(f"\n--- Últimos {limit} eventos ---")
    for row in conn.execute(
        f"SELECT id, timestamp, db_max, trigger_type, sound_category, sound_class, "
        f"vehicle_class, confidence, clip_path, kind "
        f"FROM events {where} ORDER BY timestamp DESC LIMIT ?",
        params + [limit],
    ).fetchall():
        clip = "clip" if row["clip_path"] else "—"
        linked = "+ pasaje" if row["kind"] == "both" else ""
        print(
            f"  #{row['id']} {row['timestamp']} | {_fmt(row['db_max'])} dB | "
            f"{row['trigger_type']} | {row['sound_category'] or '?'}/{row['sound_class'] or '?'} | "
            f"{row['vehicle_class'] or '-'} | {row['confidence']} | {clip} {linked}"
        )


def print_kinds(conn: sqlite3.Connection, where: str, params: list) -> None:
    """Conteo por kind de la tabla única."""
    rows = conn.execute(
        f"SELECT kind, COUNT(*) AS n FROM events {where} GROUP BY kind", params
    ).fetchall()
    counts = {r["kind"]: r["n"] for r in rows}
    print(f"\n{'=' * 60}")
    print(
        f"EVENTOS: {sum(counts.values())} | ruido {counts.get('noise', 0)}"
        f" · tránsito {counts.get('traffic', 0)} · ambos {counts.get('both', 0)}"
    )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Estadísticas del pipeline de monitoreo")
    parser.add_argument("--limit", type=_positive_int, default=15,
                        help="Últimos N items a listar (>= 1)")
    parser.add_argument("--days", type=_positive_int, default=None,
                        help="Solo últimos D días (>= 1)")
    parser.add_argument("--db", type=str, default=None, help="Ruta a events.db")
    args = parser.parse_args()

    db_path = args.db or str(PROJECT_DIR / "data" / "events.db")

    if not Path(db_path).exists():
        print(f"Base de datos no encontrada: {db_path}")
        print("Ejecuta el pipeline (python main.py) para generar datos.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    where = ""
    params: list = []
    if args.days:
        cutoff = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        where = "WHERE timestamp >= ?"
        params = [f"{cutoff}T00:00:00"]

    if not _table_exists(conn, "events"):
        print(f"{db_path} no tiene la tabla events: correr scripts/migrate_unified.py")
        sys.exit(1)

    print_kinds(conn, where, params)
    print_passages(conn, where, params, args.limit)
    print_noise_events(conn, where, params, args.limit)

    conn.close()
    print()


if __name__ == "__main__":
    main()
