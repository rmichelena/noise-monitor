#!/usr/bin/env python3
"""
Migra `noise_events` + `vehicle_passages` a una sola tabla `events`.

Una fila por evento y un solo clip por fila:

- `noise_events` conserva su `id`. Si su `passage_id` resuelve a un pasaje
  existente y es el ruido de mayor `db_max` de ese pasaje (desempate: `id`
  menor), la fila es `both`: datos de vehículo del pasaje, dB/etiquetas del
  ruido, `pinned` OR de ambos y clip de ruido o, si está vacío, el del pasaje.
  Los demás ruidos vinculados al mismo pasaje, y los de `passage_id` nulo o
  colgante, son `noise` con su propio clip.
- Un pasaje que ningún `noise_events.passage_id` referencia es `traffic`, con
  ids nuevos desde `max(id)+1` y `db_max = db_at_time`.
- Ventanas (epoch, hora local del host): `noise`/`both` usan
  `peak_ts ± buffer.pre/post_event_seconds` de la config; `traffic` va de
  `timestamp` a `timestamp_exit` (o a la entrada si falta).
- `acoustic_trigger_decisions` y `acoustic_analysis` pasan de
  `event_table='noise_events'` a `'events'`. `revision_sql` agrega columnas para
  `events`, así que las revisiones viejas no coinciden: el análisis se vuelve a
  correr bajo demanda. Las filas viejas se conservan.
- Al final `noise_events`/`vehicle_passages` se renombran a `*_legacy`. No se
  borra nada, ni filas ni clips. `vehicle_passages.noise_event_id` queda sólo
  en la tabla legacy.

Todo corre en una transacción con invariantes verificadas antes del COMMIT:
cualquier fallo hace ROLLBACK y sale != 0. Si `events` ya tiene filas aborta,
salvo `--force`, que las borra y migra de nuevo. Si la base ya está migrada
(sólo existen las `*_legacy`) no hace nada.

La tabla `events` NO está en schema.sql: `analysis_store.event_table()` la
prefiere en cuanto existe, así que crearla vacía antes del cutover dejaría el
análisis sin eventos. Con la captura escribiendo en las tablas viejas, el
servicio debe estar detenido durante la migración.

Uso:
  python scripts/migrate_unified.py --config config.yaml [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.utils import load_config, resolve_path  # noqa: E402

EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('traffic', 'noise', 'both')),
    timestamp TEXT NOT NULL,          -- ISO 8601 local, orden y filtros
    start_ts REAL NOT NULL,           -- epoch: ventana del evento
    end_ts REAL NOT NULL,
    peak_ts REAL,                     -- epoch del trigger de audio, si lo hubo
    db_max REAL,
    db_avg REAL,
    trigger_type TEXT,
    sound_class TEXT,
    sound_category TEXT,
    sound_confidence REAL,
    sound_labels_json TEXT,
    vehicle_class TEXT,
    direction TEXT,
    lane TEXT,
    speed_kmh REAL,
    detection_score REAL,
    track_id INTEGER,
    confidence TEXT,
    audio_muted INTEGER DEFAULT 0,
    duration_ms INTEGER,
    clip_path TEXT,
    pinned INTEGER DEFAULT 0,
    pinned_at TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind);
CREATE INDEX IF NOT EXISTS idx_events_vehicle ON events(vehicle_class);
CREATE INDEX IF NOT EXISTS idx_events_db ON events(db_max);
CREATE INDEX IF NOT EXISTS idx_events_pinned ON events(pinned);
CREATE INDEX IF NOT EXISTS idx_events_sound_category ON events(sound_category);
"""
EVENTS_DDL_STATEMENTS = [s.strip() for s in EVENTS_DDL.split(";") if s.strip()]

EVENTS_COLUMNS = (
    "id", "kind", "timestamp", "start_ts", "end_ts", "peak_ts", "db_max", "db_avg",
    "trigger_type", "sound_class", "sound_category", "sound_confidence",
    "sound_labels_json", "vehicle_class", "direction", "lane", "speed_kmh",
    "detection_score", "track_id", "confidence", "audio_muted", "duration_ms",
    "clip_path", "pinned", "pinned_at", "created_at",
)
SOURCES = ("noise_events", "vehicle_passages")
EVENT_TABLE_REFS = ("acoustic_trigger_decisions", "acoustic_analysis")


class MigrationError(RuntimeError):
    pass


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _epoch(value: str | None, what: str) -> float:
    """ISO local sin zona → epoch con la zona del host (no la de SQLite, que es UTC)."""
    if not value:
        raise MigrationError(f"{what}: timestamp vacío")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise MigrationError(f"{what}: timestamp ilegible {value!r}") from exc
    if parsed.tzinfo is not None:
        raise MigrationError(f"{what}: timestamp con zona {value!r}, se esperaba hora local")
    return parsed.timestamp()


def _pinned(row: sqlite3.Row) -> bool:
    return bool(row["pinned"])


def _both_pinned_at(noise: sqlite3.Row, passage: sqlite3.Row) -> str | None:
    stamps = [r["pinned_at"] for r in (noise, passage) if _pinned(r) and r["pinned_at"]]
    return max(stamps) if stamps else None


def _noise_row(noise: sqlite3.Row, kind: str, pre: float, post: float,
               passage: sqlite3.Row | None = None) -> dict:
    peak = _epoch(noise["timestamp"], f"noise_events #{noise['id']}")
    row = {
        "id": noise["id"], "kind": kind, "timestamp": noise["timestamp"],
        "start_ts": peak - pre, "end_ts": peak + post, "peak_ts": peak,
        "db_max": noise["db_max"], "db_avg": noise["db_avg"],
        "trigger_type": noise["trigger_type"], "sound_class": noise["sound_class"],
        "sound_category": noise["sound_category"],
        "sound_confidence": noise["sound_confidence"],
        "sound_labels_json": noise["sound_labels_json"],
        "vehicle_class": noise["vehicle_class"], "direction": None, "lane": noise["lane"],
        "speed_kmh": None, "detection_score": None, "track_id": None,
        "confidence": noise["confidence"], "audio_muted": noise["audio_muted"],
        "duration_ms": noise["duration_ms"], "clip_path": noise["clip_path"] or None,
        "pinned": 1 if _pinned(noise) else 0,
        "pinned_at": noise["pinned_at"] if _pinned(noise) else None,
        "created_at": noise["created_at"],
    }
    if passage is not None:
        row.update({
            "vehicle_class": passage["vehicle_class"], "direction": passage["direction"],
            "lane": passage["lane"], "speed_kmh": passage["speed_kmh"],
            "detection_score": passage["detection_score"],
            "pinned": 1 if _pinned(noise) or _pinned(passage) else 0,
            "pinned_at": _both_pinned_at(noise, passage),
        })
        if not noise["clip_path"]:
            # El clip del pasaje no tiene la duración del clip de ruido.
            row["clip_path"] = passage["clip_path"] or None
            row["duration_ms"] = None
    return row


def _traffic_row(passage: sqlite3.Row, new_id: int) -> dict:
    what = f"vehicle_passages #{passage['id']}"
    start = _epoch(passage["timestamp"], what)
    end = _epoch(passage["timestamp_exit"], what) if passage["timestamp_exit"] else start
    return {
        "id": new_id, "kind": "traffic", "timestamp": passage["timestamp"],
        "start_ts": start, "end_ts": end, "peak_ts": None,
        "db_max": passage["db_at_time"], "db_avg": None, "trigger_type": None,
        "sound_class": None, "sound_category": None, "sound_confidence": None,
        "sound_labels_json": None, "vehicle_class": passage["vehicle_class"],
        "direction": passage["direction"], "lane": passage["lane"],
        "speed_kmh": passage["speed_kmh"], "detection_score": passage["detection_score"],
        "track_id": None, "confidence": None, "audio_muted": 0, "duration_ms": None,
        "clip_path": passage["clip_path"] or None,
        "pinned": 1 if _pinned(passage) else 0,
        "pinned_at": passage["pinned_at"] if _pinned(passage) else None,
        "created_at": passage["created_at"],
    }


def build_rows(conn: sqlite3.Connection, pre: float, post: float) -> tuple[list[dict], dict]:
    noises = conn.execute("SELECT * FROM noise_events ORDER BY id").fetchall()
    passages = {r["id"]: r for r in conn.execute("SELECT * FROM vehicle_passages ORDER BY id")}

    # Ganador por pasaje: mayor db_max, desempate id menor.
    winners: dict[int, sqlite3.Row] = {}
    for noise in noises:
        pid = noise["passage_id"]
        if pid is None or pid not in passages:
            continue
        best = winners.get(pid)
        if best is None or (noise["db_max"], -noise["id"]) > (best["db_max"], -best["id"]):
            winners[pid] = noise
    winner_ids = {n["id"]: pid for pid, n in winners.items()}

    rows = []
    for noise in noises:
        pid = winner_ids.get(noise["id"])
        if pid is None:
            rows.append(_noise_row(noise, "noise", pre, post))
        else:
            rows.append(_noise_row(noise, "both", pre, post, passages[pid]))

    referenced = {n["passage_id"] for n in noises if n["passage_id"] is not None}
    orphans = [p for pid, p in passages.items() if pid not in referenced]
    next_id = max((n["id"] for n in noises), default=0) + 1
    for offset, passage in enumerate(orphans):
        rows.append(_traffic_row(passage, next_id + offset))

    expected = {
        "noise_events": len(noises),
        "linked_passages": len(winners),
        "orphan_passages": len(orphans),
    }
    return rows, expected


def verify(conn: sqlite3.Connection, expected: dict) -> dict:
    counts = dict(conn.execute("SELECT kind, COUNT(*) FROM events GROUP BY kind").fetchall())
    total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    distinct = conn.execute("SELECT COUNT(DISTINCT id) FROM events").fetchone()[0]
    both = counts.get("both", 0)
    noise = counts.get("noise", 0)
    traffic = counts.get("traffic", 0)
    checks = [
        (total == expected["noise_events"] + expected["orphan_passages"],
         f"total {total} != noise_events {expected['noise_events']}"
         f" + pasajes huérfanos {expected['orphan_passages']}"),
        (both == expected["linked_passages"],
         f"both {both} != pasajes vinculados {expected['linked_passages']}"),
        (traffic == expected["orphan_passages"],
         f"traffic {traffic} != pasajes huérfanos {expected['orphan_passages']}"),
        (noise == expected["noise_events"] - expected["linked_passages"],
         f"noise {noise} != noise_events {expected['noise_events']}"
         f" - pasajes vinculados {expected['linked_passages']}"),
        (distinct == total, f"ids duplicados: {total - distinct}"),
    ]
    unresolved = conn.execute(
        "SELECT COUNT(*) FROM events e JOIN noise_events n ON n.id = e.id"
        " LEFT JOIN vehicle_passages p ON p.id = n.passage_id"
        " WHERE e.kind = 'both' AND p.id IS NULL"
    ).fetchone()[0]
    checks.append((unresolved == 0, f"{unresolved} filas both sin pasaje resuelto"))
    duplicated_passage = conn.execute(
        "SELECT COUNT(*) FROM (SELECT n.passage_id FROM events e"
        " JOIN noise_events n ON n.id = e.id WHERE e.kind = 'both'"
        " GROUP BY n.passage_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    checks.append((duplicated_passage == 0, f"{duplicated_passage} pasajes con más de un both"))
    failures = [message for ok, message in checks if not ok]
    if failures:
        raise MigrationError("invariantes violadas: " + "; ".join(failures))
    return {"both": both, "noise": noise, "traffic": traffic, "total": total}


def _retag_event_table(conn: sqlite3.Connection) -> dict:
    moved = {}
    for table in EVENT_TABLE_REFS:
        if not _table_exists(conn, table):
            continue
        clash = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE event_table = 'events'"
        ).fetchone()[0]
        if clash:
            raise MigrationError(
                f"{table} ya tiene {clash} filas con event_table='events' de otra tabla"
            )
        moved[table] = conn.execute(
            f"UPDATE {table} SET event_table = 'events' WHERE event_table = 'noise_events'"
        ).rowcount
    return moved


def migrate(conn: sqlite3.Connection, pre: float, post: float,
            force: bool = False, dry_run: bool = False) -> dict:
    """Corre la migración en una transacción. Devuelve conteos o levanta MigrationError."""
    conn.row_factory = sqlite3.Row
    conn.isolation_level = None  # transacción manual: BEGIN ... COMMIT/ROLLBACK
    present = [t for t in SOURCES if _table_exists(conn, t)]
    legacy = [t for t in SOURCES if _table_exists(conn, f"{t}_legacy")]
    if not present and len(legacy) == len(SOURCES):
        if not _table_exists(conn, "events"):
            # Sin `events` no hay migración válida: sólo quedan las legacy.
            # Es el caso de una base a la que le borraron la tabla nueva a mano.
            raise MigrationError(
                "existen tablas *_legacy pero falta events; restaurar el backup previo"
            )
        return {"status": "already_migrated"}
    if len(present) != len(SOURCES):
        raise MigrationError(f"faltan tablas de origen: {sorted(set(SOURCES) - set(present))}")
    if legacy:
        raise MigrationError(f"ya existen tablas legacy junto a las de origen: {legacy}")
    if _table_exists(conn, "events"):
        missing = set(EVENTS_COLUMNS) - _columns(conn, "events")
        if missing:
            raise MigrationError(
                f"existe una tabla events con otra forma (faltan {sorted(missing)})"
            )

    conn.execute("BEGIN IMMEDIATE")
    try:
        for statement in EVENTS_DDL_STATEMENTS:
            conn.execute(statement)
        existing = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if existing and not force:
            raise MigrationError(f"events ya tiene {existing} filas (usar --force para rehacer)")
        if existing:
            conn.execute("DELETE FROM events")

        rows, expected = build_rows(conn, pre, post)
        placeholders = ", ".join("?" for _ in EVENTS_COLUMNS)
        conn.executemany(
            f"INSERT INTO events ({', '.join(EVENTS_COLUMNS)}) VALUES ({placeholders})",
            [tuple(row[c] for c in EVENTS_COLUMNS) for row in rows],
        )
        counts = verify(conn, expected)
        counts["retagged"] = _retag_event_table(conn)
        for table in SOURCES:
            conn.execute(f"ALTER TABLE {table} RENAME TO {table}_legacy")
        if dry_run:
            conn.execute("ROLLBACK")
            counts["status"] = "dry_run"
        else:
            conn.execute("COMMIT")
            counts["status"] = "migrated"
        return counts
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description="Migra noise_events + vehicle_passages a events")
    ap.add_argument("--config", default=None, help="ruta a config.yaml")
    ap.add_argument("--dry-run", action="store_true", help="migra, verifica y hace ROLLBACK")
    ap.add_argument("--force", action="store_true", help="rehace si events ya tiene filas")
    args = ap.parse_args()

    config = load_config(args.config or (PROJECT_DIR / "config.yaml"))
    buffer = config["buffer"]
    pre, post = float(buffer["pre_event_seconds"]), float(buffer["post_event_seconds"])
    db_path = resolve_path(config, "database", PROJECT_DIR)
    if not db_path.is_file():
        print(f"base no encontrada: {db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(str(db_path), timeout=10)
    try:
        counts = migrate(conn, pre, post, force=args.force, dry_run=args.dry_run)
    except (MigrationError, sqlite3.Error) as exc:
        print(f"migración abortada: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    if counts["status"] == "already_migrated":
        print("base ya migrada: sólo existen las tablas *_legacy")
        return 0
    print(
        f"{counts['status']}: both={counts['both']} noise={counts['noise']}"
        f" traffic={counts['traffic']} total={counts['total']}"
        f" event_table→events={counts['retagged']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
