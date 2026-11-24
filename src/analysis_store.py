"""Persistencia del análisis, independiente del esquema operativo de captura."""

import json
import sqlite3
from datetime import datetime
from time import time

ROAD_VEHICLES = ("car", "motorcycle", "truck", "bus", "van")
UNIFIED = "events"
MAX_ATTEMPTS = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS acoustic_analysis (
    id INTEGER PRIMARY KEY,
    event_table TEXT NOT NULL,
    event_id INTEGER NOT NULL,
    clip_path TEXT NOT NULL,
    revision TEXT NOT NULL,
    profile TEXT NOT NULL,
    profile_name TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending', 'completed', 'failed', 'unavailable')),
    attempts INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL,
    retry_after REAL,
    elapsed_ms REAL,
    result_json TEXT,
    UNIQUE(event_table, event_id, revision, profile)
);
CREATE INDEX IF NOT EXISTS idx_acoustic_analysis_event
ON acoustic_analysis(event_table, event_id, id DESC);
"""


def event_table(conn: sqlite3.Connection, preferred: str | None = None) -> str | None:
    """`events` si la base tiene la tabla única; si no, None.

    Las tablas `noise_events`/`vehicle_passages` ya no se analizan: tras la
    migración sólo quedan como `*_legacy`.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    if preferred in (None, UNIFIED) and {"kind", "start_ts", "end_ts", "track_id"} <= columns:
        return UNIFIED
    return None


def revision_sql(table: str) -> str:
    return ("json_array(e.clip_path, coalesce(e.audio_muted, 0),"
            " e.kind, e.start_ts, e.end_ts, e.track_id, e.vehicle_class)")


def pending_events(conn, table: str, profile: str, since: str, cutoff: str, limit: int) -> list:
    # Sólo hay audio que analizar en noise y both; `timestamp` no es la ventana.
    since_value = datetime.fromisoformat(since).timestamp() if since else 0
    cutoff_value = datetime.fromisoformat(cutoff).timestamp()
    return conn.execute(
        f"""
        SELECT e.*, {revision_sql(table)} AS revision
        FROM {table} e
        LEFT JOIN acoustic_analysis a
        ON a.event_table = ? AND a.event_id = e.id
          AND a.revision = {revision_sql(table)} AND a.profile = ?
        WHERE e.start_ts >= ? AND e.start_ts <= ?
        AND (coalesce(e.clip_path, '') <> '' OR e.audio_muted = 1)
        AND e.kind IN ('noise', 'both')
        AND (a.id IS NULL OR a.status = 'pending' OR
             (a.status = 'failed' AND a.attempts < ? AND a.retry_after <= ?))
        ORDER BY e.start_ts, e.id LIMIT ?
        """,
        (table, profile, since_value, cutoff_value, MAX_ATTEMPTS, time(), limit),
    ).fetchall()


def visual_context(conn, table: str, event) -> tuple[list[str], bool]:
    """Devuelve las clases de pasajes rastreados y si hay un único candidato vinculado."""
    road = ",".join("?" for _ in ROAD_VEHICLES)
    rows = conn.execute(
        f"""SELECT id, vehicle_class FROM events
            WHERE track_id IS NOT NULL AND vehicle_class IN ({road})
            AND start_ts <= ? AND end_ts >= ?""",
        (*ROAD_VEHICLES, event["end_ts"], event["start_ts"]),
    ).fetchall()
    linked = event["id"] if event["kind"] == "both" else None
    return sorted({row["vehicle_class"] for row in rows}), len(rows) == 1 and rows[0]["id"] == linked


def start_attempt(conn, table: str, event, profile: str, profile_name: str):
    with conn:
        conn.execute(
            """INSERT INTO acoustic_analysis
               (event_table, event_id, clip_path, revision, profile, profile_name, status, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
               ON CONFLICT(event_table, event_id, revision, profile)
               DO UPDATE SET status='pending', updated_at=excluded.updated_at,
                             retry_after=NULL, elapsed_ms=NULL, result_json=NULL""",
            (table, event["id"], event["clip_path"] or "", event["revision"], profile, profile_name, time()),
        )
    return conn.execute(
        """SELECT id, attempts FROM acoustic_analysis
           WHERE event_table=? AND event_id=? AND revision=? AND profile=?""",
        (table, event["id"], event["revision"], profile),
    ).fetchone()


def finish_attempt(conn, job, result: dict) -> None:
    attempts = job["attempts"] + 1
    now = time()
    retry = now + 60 * 2 ** (attempts - 1) if result["status"] == "failed" else None
    with conn:
        conn.execute(
            """UPDATE acoustic_analysis SET status=?, attempts=?, updated_at=?, retry_after=?,
               elapsed_ms=?, result_json=? WHERE id=?""",
            (result["status"], attempts, now, retry, result.get("elapsed_ms"),
             json.dumps(result, allow_nan=False), job["id"]),
        )


def latest_analysis(conn, event_id: int, preferred: str | None = None) -> dict | None:
    table = event_table(conn, preferred)
    if table is None:
        return None
    event = conn.execute(
        f"""SELECT e.id, e.clip_path, e.audio_muted, {revision_sql(table)} AS revision
            FROM {table} e WHERE e.id=? AND e.kind IN ('noise', 'both')""",
        (event_id,),
    ).fetchone()
    if event is None:
        return None
    trigger_decision = None
    if conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='acoustic_trigger_decisions'"
    ).fetchone():
        decision = conn.execute(
            "SELECT decision_json FROM acoustic_trigger_decisions WHERE event_table=? AND event_id=?",
            (table, event_id),
        ).fetchone()
        if decision:
            trigger_decision = json.loads(decision[0])
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='acoustic_analysis'"
    ).fetchone()
    if exists:
        row = conn.execute(
            """SELECT profile_name, status, attempts, elapsed_ms, result_json
               FROM acoustic_analysis WHERE event_table=? AND event_id=? AND revision=?
               ORDER BY id DESC LIMIT 1""",
            (table, event_id, event["revision"]),
        ).fetchone()
        if row is not None:
            data = dict(row)
            result = json.loads(data.pop("result_json") or "{}")
            result.pop("clip_path", None)
            result.pop("calibration", None)
            result["issue_codes"] = list(result.pop("issues", {}))
            return {"event_table": table, "event_id": event_id, "trigger_decision": trigger_decision, **result, **data}
    status = "unavailable" if event["audio_muted"] or not event["clip_path"] else "not_run"
    return {"event_table": table, "event_id": event_id, "status": status, "trigger_decision": trigger_decision}
