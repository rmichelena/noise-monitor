"""
Dashboard web (modelo unificado de eventos).

Lee SQLite en modo solo-lectura salvo el endpoint de fijar (pinned), que
abre una conexión de escritura. El nivel dB actual sale del AudioMonitor.

Endpoints:
  GET /                      página principal
  GET /partials/{live,events,spark,vehicle-chart,filters}
  GET /api/live              dB actual + contadores del día
  GET /api/events            eventos con filtros
  GET /api/clips/{id}        clip MP4 del evento
  POST /api/events/{id}/pin  fijar / desfijar
  GET /api/stats             agregados
  GET /api/detections        último frame de inferencia
  GET /api/health            estado
  GET /stream                MJPEG con overlay
"""

from __future__ import annotations

import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.utils import resolve_path

PAGE_SIZE = 20
DEFAULT_DB = 75.0

# Misma escala que el overlay de los clips.
DB_BANDS = [
    (40.0, "#50c850", "verde"),
    (50.0, "#ffd23f", "amarillo"),
    (60.0, "#ffb020", "naranja"),
    (75.0, "#ff5c5c", "rojo"),
]


def db_color(value: float) -> str:
    """Color de banda para un nivel (misma escala que el overlay)."""
    for limit, color, _ in DB_BANDS:
        if value <= limit:
            return color
    return DB_BANDS[-1][1]


KIND_LABELS = {"traffic": "tráfico", "noise": "ruido", "both": "ambos"}


def create_app(config: dict, base_dir: Path, audio_monitor=None, video_monitor=None) -> FastAPI:
    base_dir = Path(base_dir)
    db_path = resolve_path(config, "database", base_dir)
    clips_dir = resolve_path(config, "clips", base_dir)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    app = FastAPI(title="Noise Monitor", version="2.0", docs_url="/api/docs")

    @app.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/partials") or path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------
    def _connect() -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3)
        conn.row_factory = sqlite3.Row
        return conn

    def _query(sql: str, params: tuple = ()) -> list[dict]:
        try:
            conn = _connect()
        except sqlite3.Error:
            return []
        try:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.Error as exc:
            logger.warning("Consulta falló (%s): %s", sql.split()[1:3], exc)
            return []
        finally:
            conn.close()

    def _write(sql: str, params: tuple = ()) -> None:
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _current_db() -> float | None:
        if audio_monitor is None:
            return None
        try:
            return round(float(audio_monitor.current_db), 1)
        except Exception:
            return None

    def _coral_ok() -> bool | None:
        if video_monitor is None:
            return None
        return getattr(video_monitor, "_interpreter", None) is not None

    # ------------------------------------------------------------------
    # Consultas del día
    # ------------------------------------------------------------------
    def _today() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _counters() -> dict:
        day = _today()
        rows = _query(
            "SELECT kind, COUNT(*) n FROM events WHERE timestamp >= ? GROUP BY kind",
            (day,),
        )
        by_kind = {r["kind"]: r["n"] for r in rows}
        total = sum(by_kind.values())
        vehicles = _query(
            "SELECT COUNT(*) n FROM events WHERE timestamp >= ? AND vehicle_class IS NOT NULL",
            (day,),
        )[0]["n"]
        maxdb = _query(
            "SELECT MAX(db_max) m FROM events WHERE timestamp >= ?", (day,)
        )[0]["m"]
        hour_ago = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
        missing = _query(
            "SELECT COUNT(*) n FROM events WHERE timestamp >= ? AND (clip_path IS NULL OR clip_path = '')",
            (hour_ago,),
        )[0]["n"]
        return {
            "events": total,
            "vehicles": vehicles,
            "by_kind": by_kind,
            "max_db": round(maxdb, 1) if maxdb else None,
            "without_clip": missing,
        }

    def _events(
        limit: int,
        kind: str | None = None,
        vehicle: str | None = None,
        direction: str | None = None,
        min_db: float | None = None,
        min_speed: float | None = None,
        pinned_only: bool = False,
        start_ts_from: float | None = None,
    ) -> list[dict]:
        sql = (
            "SELECT e.id, e.kind, e.start_ts, e.end_ts, e.peak_ts, e.timestamp,"
            " e.db_max, e.db_avg, e.trigger_type, e.sound_class, e.sound_category,"
            " e.sound_confidence, e.vehicle_class, e.direction, e.lane,"
            " e.speed_kmh, e.pinned, e.clip_path"
            " FROM events e WHERE 1=1"
        )
        params: list = []
        if kind:
            sql += " AND e.kind = ?"
            params.append(kind)
        if vehicle:
            sql += " AND e.vehicle_class = ?"
            params.append(vehicle)
        if direction:
            sql += " AND e.direction = ?"
            params.append(direction)
        if min_db is not None:
            sql += " AND e.db_max >= ?"
            params.append(min_db)
        if min_speed is not None:
            sql += " AND e.speed_kmh >= ?"
            params.append(min_speed)
        if pinned_only:
            sql += " AND e.pinned = 1"
        if start_ts_from is not None:
            sql += " AND e.end_ts >= ?"
            params.append(start_ts_from)
        sql += " ORDER BY e.start_ts DESC LIMIT ?"
        params.append(limit)
        rows = _query(sql, tuple(params))
        for r in rows:
            r["has_clip"] = bool(r.pop("clip_path", None))
            r["kind_label"] = KIND_LABELS.get(r["kind"], r["kind"])
        return rows

    def _summary() -> dict:
        day = _today()
        by_kind = {
            r["kind"]: r["n"]
            for r in _query(
                "SELECT kind, COUNT(*) n FROM events WHERE timestamp >= ? GROUP BY kind",
                (day,),
            )
        }
        by_class = {
            r["c"]: r["n"]
            for r in _query(
                "SELECT vehicle_class c, COUNT(*) n FROM events"
                " WHERE timestamp >= ? AND vehicle_class IS NOT NULL GROUP BY c",
                (day,),
            )
        }
        by_dir = {
            r["d"]: r["n"]
            for r in _query(
                "SELECT direction d, COUNT(*) n FROM events"
                " WHERE timestamp >= ? AND direction IS NOT NULL GROUP BY d",
                (day,),
            )
        }
        speed = _query(
            "SELECT AVG(speed_kmh) v, MAX(speed_kmh) m, COUNT(speed_kmh) n"
            " FROM events WHERE timestamp >= ? AND speed_kmh IS NOT NULL",
            (day,),
        )[0]
        return {
            "by_kind": by_kind,
            "by_class": by_class,
            "by_dir": by_dir,
            "speed": {
                "avg": round(speed["v"], 1) if speed["n"] else None,
                "max": round(speed["m"], 1) if speed["n"] else None,
                "n": speed["n"],
            },
        }

    # ------------------------------------------------------------------
    # Gráficos
    # ------------------------------------------------------------------
    def _minute_labels(minutes: int):
        end = datetime.now().replace(second=0, microsecond=0)
        start = end - timedelta(minutes=minutes - 1)
        labels = [
            (start + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M")
            for i in range(minutes)
        ]
        ticks = [
            {"pct": round((i + 0.5) / minutes * 100, 3), "label": lbl[11:16]}
            for i, lbl in enumerate(labels)
            if lbl[14:16] in ("00", "10", "20", "30", "40", "50")
        ]
        return labels, ticks

    def _audio_bars(minutes: int = 120, height: int = 60) -> dict:
        labels, ticks = _minute_labels(minutes)
        rows = _query(
            "SELECT minute, db_max FROM audio_levels WHERE minute >= ?",
            (labels[0],),
        )
        by_minute = {r["minute"]: r["db_max"] for r in rows}
        width = 100.0 / minutes
        bars = []
        for idx, lbl in enumerate(labels):
            value = by_minute.get(lbl)
            if value is None:
                continue
            level = max(0.0, min(float(value), DEFAULT_DB))
            bars.append({
                "x": round(idx * width, 2),
                "w": round(width * 0.8, 2),
                "h": round(max(1.0, level / DEFAULT_DB * (height - 5)), 2),
                "color": db_color(level),
                "minute": lbl[11:16],
                "value": round(float(value), 1),
            })
        return {"bars": bars, "times": ticks}

    VEHICLE_GROUPS = {"car": "car", "truck": "heavy", "bus": "heavy",
                      "motorcycle": "moto"}

    def _vehicle_bars(minutes: int = 120, height: int = 60) -> dict:
        labels, ticks = _minute_labels(minutes)
        start = datetime.strptime(labels[0], "%Y-%m-%dT%H:%M")
        rows = _query(
            "SELECT substr(timestamp, 1, 16) m, vehicle_class c, COUNT(*) n"
            " FROM events WHERE timestamp >= ? AND vehicle_class IS NOT NULL"
            " AND kind IN ('traffic','both') GROUP BY m, c",
            (labels[0],),
        )
        by_minute: dict[str, dict[str, int]] = {}
        for r in rows:
            group = VEHICLE_GROUPS.get(r["c"])
            if not group:
                continue
            slot = by_minute.setdefault(r["m"], {"car": 0, "heavy": 0, "moto": 0})
            slot[group] += r["n"]

        counts = [by_minute.get(lbl, {"car": 0, "heavy": 0, "moto": 0}) for lbl in labels]
        max_total = max((c["car"] + c["heavy"] + c["moto"] for c in counts), default=0)
        scale = (height - 5) / max_total if max_total else 0
        width = 100.0 / minutes
        bars = []
        for idx, c in enumerate(counts):
            total = c["car"] + c["heavy"] + c["moto"]
            if not total:
                continue
            segments = []
            base = height
            for group, color in (("car", "#4cc2ff"), ("heavy", "#ffb020"), ("moto", "#ff5c5c")):
                if not c[group]:
                    continue
                h = c[group] * scale
                base -= h
                segments.append({"y": round(base, 2), "h": round(h, 2),
                                 "color": color, "group": group, "n": c[group]})
            bars.append({
                "x": round(idx * width, 2),
                "w": round(width * 0.8, 2),
                "total": total,
                "minute": labels[idx][11:16],
                "segments": segments,
            })
        return {"bars": bars, "max_total": max_total, "times": ticks}

    # ------------------------------------------------------------------
    # Contexto y páginas
    # ------------------------------------------------------------------
    def _context(**overrides) -> dict:
        counters = _counters()
        ctx = {
            "current_db": _current_db(),
            "counters": counters,
            "summary": _summary(),
            "coral_ok": _coral_ok(),
            "events": _events(PAGE_SIZE),
            "audio_chart": _audio_bars(),
            "vehicle_chart": _vehicle_bars(),
            "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "filters": {},
        }
        ctx.update(overrides)
        return ctx

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return templates.TemplateResponse(request, "dashboard.html", _context())

    @app.get("/partials/live", response_class=HTMLResponse)
    def partial_live(request: Request):
        return templates.TemplateResponse(request, "_live.html", _context())

    @app.get("/partials/events", response_class=HTMLResponse)
    def partial_events(
        request: Request,
        kind: str | None = None,
        vehicle: str | None = None,
        direction: str | None = None,
        min_db: float | None = None,
        min_speed: float | None = None,
        pinned: int = 0,
    ):
        events = _events(PAGE_SIZE, kind, vehicle, direction, min_db, min_speed, bool(pinned))
        label = kind or vehicle or direction
        return templates.TemplateResponse(
            request, "_events.html", _context(events=events, filters={"label": label, "kind": kind})
        )

    @app.get("/partials/spark", response_class=HTMLResponse)
    def partial_spark(request: Request):
        return templates.TemplateResponse(request, "_spark.html", _context())

    @app.get("/partials/vehicle-chart", response_class=HTMLResponse)
    def partial_vehicle_chart(request: Request):
        return templates.TemplateResponse(request, "_vehicle_chart.html", _context())

    @app.get("/partials/filters", response_class=HTMLResponse)
    def partial_filters(request: Request):
        return templates.TemplateResponse(request, "_filters.html", _context())

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    @app.get("/api/live")
    def api_live():
        c = _counters()
        return {
            "db": _current_db(),
            "events_today": c["events"],
            "vehicles_today": c["vehicles"],
            "by_kind": c["by_kind"],
            "max_db_today": c["max_db"],
            "events_without_clip_1h": c["without_clip"],
            "coral_ok": _coral_ok(),
            "now": datetime.now().strftime("%H:%M:%S"),
        }

    @app.get("/api/events")
    def api_events(
        limit: int = Query(20, ge=1, le=200),
        kind: str | None = None,
        vehicle: str | None = None,
        direction: str | None = None,
        min_db: float | None = None,
        min_speed: float | None = None,
        pinned: int = 0,
    ):
        return {
            "events": _events(limit, kind, vehicle, direction, min_db, min_speed, bool(pinned))
        }

    @app.get("/api/clips/{event_id}")
    def api_clip(event_id: int):
        rows = _query("SELECT clip_path FROM events WHERE id = ?", (event_id,))
        if not rows or not rows[0]["clip_path"]:
            raise HTTPException(status_code=404, detail="Clip no encontrado")
        stored = Path(rows[0]["clip_path"])
        path = (stored if stored.is_absolute() else base_dir / stored).resolve()
        if clips_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Content-Disposition": f'inline; filename="{path.name}"'},
        )

    @app.post("/api/events/{event_id}/pin")
    def pin_event(event_id: int, pinned: int = 1):
        rows = _query("SELECT id FROM events WHERE id = ?", (event_id,))
        if not rows:
            raise HTTPException(status_code=404, detail="Evento no encontrado")
        if pinned:
            _write(
                "UPDATE events SET pinned = 1, pinned_at = ? WHERE id = ?",
                (datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), event_id),
            )
        else:
            _write("UPDATE events SET pinned = 0, pinned_at = NULL WHERE id = ?", (event_id,))
        return {"ok": True, "id": event_id, "pinned": bool(pinned)}

    @app.get("/api/stats")
    def api_stats():
        day = _today()
        days = _query(
            "SELECT substr(timestamp, 1, 10) d, COUNT(*) n, MAX(db_max) m"
            " FROM events WHERE timestamp >= date(?, '-6 days') GROUP BY d ORDER BY d",
            (day,),
        )
        kinds = _query(
            "SELECT kind k, COUNT(*) n FROM events WHERE timestamp >= date(?, '-6 days')"
            " GROUP BY k ORDER BY n DESC",
            (day,),
        )
        return {"counters": _counters(), "summary": _summary(),
                "days": days, "kinds": kinds}

    @app.get("/api/detections")
    def api_detections():
        if video_monitor is None:
            return {"detections": [], "coral_ok": None}
        return {
            "detections": [
                {"class": d["class"], "score": round(d["score"], 2)}
                for d in video_monitor.recent_detections
            ],
            "coral_ok": _coral_ok(),
        }

    @app.get("/api/health")
    def api_health():
        return {"ok": True, "db": str(db_path), "db_exists": db_path.exists()}

    @app.get("/stream")
    def stream():
        if video_monitor is None:
            raise HTTPException(status_code=503, detail="Stream no disponible")
        if video_monitor.overlay_clients >= 3:
            raise HTTPException(status_code=503, detail="Demasiados espectadores")

        def gen():
            video_monitor.overlay_clients += 1
            try:
                while True:
                    jpeg = video_monitor.latest_overlay()
                    if jpeg:
                        yield (
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpeg)).encode()
                            + b"\r\n\r\n"
                            + jpeg
                            + b"\r\n"
                        )
                    time.sleep(0.08)
            finally:
                video_monitor.overlay_clients -= 1

        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.get("/api/tracker")
    def api_tracker():
        if video_monitor is None:
            return {"tracks": []}
        tracker = getattr(video_monitor, "_tracker", None)
        if tracker is None or not hasattr(tracker, "snapshot"):
            return {"tracks": []}
        return {
            "tracks": tracker.snapshot(),
            "inference_count": getattr(video_monitor, "inference_count", None),
            "frame_count": getattr(video_monitor, "_frame_count", None),
            "vehicles": list(getattr(video_monitor, "vehicle_log", [])),
        }

    @app.get("/api/health")
    def api_health():
        return {"ok": True, "db": str(db_path), "db_exists": db_path.exists()}

    return app
