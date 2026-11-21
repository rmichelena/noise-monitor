"""Persistencia del análisis, independiente del esquema operativo de captura."""

import json
import sqlite3
from datetime import datetime, timedelta
from time import time

ROAD_VEHICLES = ("car", "motorcycle", "truck", "bus", "van")
LEGACY = "noise_events"
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
    columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
    if preferred in (None, UNIFIED) and {"kind", "start_ts", "end_ts", "track_id"} <= columns:
        return UNIFIED
    if preferred in (None, LEGACY) and conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (LEGACY,),
    ).fetchone():
        return LEGACY
    return None


def revision_sql(table: str) -> str:
    revision = "json_array(e.clip_path, coalesce(e.audio_muted, 0)"
    if table == UNIFIED:
        revision += ", e.kind, e.start_ts, e.end_ts, e.track_id, e.vehicle_class"
    return revision + ")"


def pending_events(conn, table: str, profile: str, since: str, cutoff: str, limit: int) -> list:
    kind = "AND e.kind IN ('noise', 'both')" if table == UNIFIED else ""
    return conn.execute(
        f"""
        SELECT e.*, {revision_sql(table)} AS revision
        FROM {table} e
        LEFT JOIN acoustic_analysis a
        ON a.event_table = ? AND a.event_id = e.id
          AND a.revision = {revision_sql(table)} AND a.profile = ?
        WHERE e.timestamp >= ? AND e.timestamp <= ?
        AND (coalesce(e.clip_path, '') <> '' OR e.audio_muted = 1)
        {kind}
        AND (a.id IS NULL OR a.status = 'pending' OR
             (a.status = 'failed' AND a.attempts < ? AND a.retry_after <= ?))
        ORDER BY e.timestamp, e.id LIMIT ?
        """,
        (table, profile, since, cutoff, MAX_ATTEMPTS, time(), limit),
    ).fetchall()


def visual_context(conn, table: str, event, pre: float, post: float) -> tuple[list[str], bool]:
    """Devuelve las clases de pasajes rastreados y si hay un único candidato vinculado."""
    road = ",".join("?" for _ in ROAD_VEHICLES)
    if table == UNIFIED:
        rows = conn.execute(
            f"""SELECT id, vehicle_class FROM events
                WHERE track_id IS NOT NULL AND vehicle_class IN ({road})
                AND start_ts <= ? AND end_ts >= ?""",
            (*ROAD_VEHICLES, event["end_ts"], event["start_ts"]),
        ).fetchall()
        linked = event["id"] if event["kind"] == "both" else None
    else:
        timestamp = datetime.fromisoformat(event["timestamp"])
        begin = (timestamp - timedelta(seconds=pre)).isoformat(timespec="seconds")
        end = (timestamp + timedelta(seconds=post + 1)).isoformat(timespec="seconds")
        rows = conn.execute(
            f"""SELECT id, vehicle_class FROM vehicle_passages
                WHERE vehicle_class IN ({road}) AND timestamp <= ?
                AND coalesce(timestamp_exit, timestamp) >= ?""",
            (*ROAD_VEHICLES, end, begin),
        ).fetchall()
        linked = event["passage_id"]
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
    kind = "AND e.kind IN ('noise', 'both')" if table == UNIFIED else ""
    event = conn.execute(
        f"""SELECT e.id, e.clip_path, e.audio_muted, {revision_sql(table)} AS revision
            FROM {table} e WHERE e.id=? {kind}""",
        (event_id,),
    ).fetchone()
    if event is None:
        return None
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='acoustic_analysis'"
    ).fetchone()
    if exists:
        row = conn.execute(
            """SELECT profile_name, status, attempts, elapsed_ms, result_json
               FROM acoustic_analysis WHERE event_table=? AND event_id=?
                 AND (revision=? OR (? AND status='completed')) ORDER BY id DESC LIMIT 1""",
            (table, event_id, event["revision"], table == LEGACY and not event["clip_path"] and not event["audio_muted"]),
        ).fetchone()
        if row is not None:
            data = dict(row)
            result = json.loads(data.pop("result_json") or "{}")
            result.pop("clip_path", None)
            result.pop("calibration", None)
            result["issue_codes"] = list(result.pop("issues", {}))
            return {"event_table": table, "event_id": event_id, **result, **data}
    status = "unavailable" if event["audio_muted"] or not event["clip_path"] else "not_run"
    return {"event_table": table, "event_id": event_id, "status": status}
