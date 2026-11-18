"""
Dashboard web del Noise Monitor (Fase 2).

Lee la base SQLite en modo solo-lectura y expone el nivel dB actual desde el
AudioMonitor en memoria. No escribe nada en la base.

Endpoints:
  GET /                    página principal
  GET /api/live            dB actual + contadores del día
  GET /api/events          últimos eventos (filtros: limit, category, min_db)
  GET /api/clips/{id}      clip MP4 de un evento
  GET /api/stats           agregados de hoy / 7 días
  GET /api/health          estado del proceso web
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from src.utils import resolve_path

PAGE_SIZE = 20
logger = logging.getLogger(__name__)


def create_app(config: dict, base_dir: Path, audio_monitor=None, video_monitor=None) -> FastAPI:
    base_dir = Path(base_dir)
    db_path = resolve_path(config, "database", base_dir)
    clips_dir = resolve_path(config, "clips", base_dir)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    app = FastAPI(title="Noise Monitor", version="1.0", docs_url="/api/docs")

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
            # Sin este log una columna inexistente devolvía una lista vacía
            # y el dashboard parecía "sin datos" en vez de roto.
            logger.warning("Consulta falló (%s): %s", sql.split()[2] if sql.split() else "?", exc)
            return []
        finally:
            conn.close()

    def _today_prefix() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _counters() -> dict:
        day = _today_prefix()
        ev = _query("SELECT COUNT(*) n FROM noise_events WHERE timestamp >= ?", (day,))
        ps = _query("SELECT COUNT(*) n FROM vehicle_passages WHERE timestamp >= ?", (day,))
        maxdb = _query(
            "SELECT MAX(db_max) m FROM noise_events WHERE timestamp >= ?", (day,)
        )
        return {
            "events": ev[0]["n"] if ev else 0,
            "passages": ps[0]["n"] if ps else 0,
            "max_db": round(maxdb[0]["m"], 1) if maxdb and maxdb[0]["m"] else None,
        }

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

    def _levels(limit: int = 120) -> list[dict]:
        rows = _query(
            "SELECT minute, db_avg, db_max FROM audio_levels ORDER BY minute DESC LIMIT ?",
            (limit,),
        )
        return list(reversed(rows))

    def _sparkline(rows: list[dict], height: int = 60) -> list[dict]:
        """Convierte niveles en barras para el SVG (escala fija 0-70 dB)."""
        bars = []
        if not rows:
            return bars
        width = 100.0 / len(rows)
        for i, r in enumerate(rows):
            value = float(r["db_max"] or 0)
            h = max(1.0, min(value, 70.0) / 70.0 * height)
            bars.append({"x": round(i * width, 2), "w": round(width * 0.8, 2), "h": round(h, 1)})
        return bars

    def _events(limit: int, category: str | None, min_db: float | None) -> list[dict]:
        sql = (
            "SELECT e.id, e.timestamp, e.db_max, e.db_avg, e.trigger_type,"
            " e.sound_class, e.sound_category, e.sound_confidence, e.vehicle_class,"
            " e.clip_path, p.direction AS direction"
            " FROM noise_events e"
            " LEFT JOIN vehicle_passages p ON p.id = e.passage_id"
            " WHERE 1=1"
        )
        params: list = []
        if category:
            sql += " AND e.sound_category = ?"
            params.append(category)
        if min_db is not None:
            sql += " AND e.db_max >= ?"
            params.append(min_db)
        sql += " ORDER BY e.id DESC LIMIT ?"
        params.append(limit)
        rows = _query(sql, tuple(params))
        for r in rows:
            r["has_clip"] = bool(r.pop("clip_path", None))
        return rows

    def _passages_today() -> dict:
        day = _today_prefix()
        rows = _query(
            "SELECT vehicle_class, direction, COUNT(*) n FROM vehicle_passages"
            " WHERE timestamp >= ? GROUP BY vehicle_class, direction",
            (day,),
        )
        by_class: dict[str, int] = {}
        by_dir: dict[str, int] = {}
        for r in rows:
            by_class[r["vehicle_class"]] = by_class.get(r["vehicle_class"], 0) + r["n"]
            if r["direction"]:
                by_dir[r["direction"]] = by_dir.get(r["direction"], 0) + r["n"]
        return {"by_class": by_class, "by_dir": by_dir}

    # ------------------------------------------------------------------
    # Páginas
    # ------------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        counters = _counters()
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "current_db": _current_db(),
                "counters": counters,
                "coral_ok": _coral_ok(),
                "events": _events(PAGE_SIZE, None, None),
                "passages": _passages_today(),
                "bars": _sparkline(_levels()),
                "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------
    @app.get("/api/live")
    def api_live():
        counters = _counters()
        return {
            "db": _current_db(),
            "events_today": counters["events"],
            "passages_today": counters["passages"],
            "max_db_today": counters["max_db"],
            "coral_ok": _coral_ok(),
            "now": datetime.now().strftime("%H:%M:%S"),
        }

    @app.get("/api/events")
    def api_events(
        limit: int = Query(20, ge=1, le=200),
        category: str | None = None,
        min_db: float | None = None,
    ):
        return {"events": _events(limit, category, min_db)}

    @app.get("/api/clips/{event_id}")
    def api_clip(event_id: int):
        rows = _query("SELECT clip_path FROM noise_events WHERE id = ?", (event_id,))
        if not rows or not rows[0]["clip_path"]:
            raise HTTPException(status_code=404, detail="Clip no encontrado")
        stored = Path(rows[0]["clip_path"])
        path = stored if stored.is_absolute() else base_dir / stored
        path = path.resolve()
        # Solo se sirven archivos dentro del directorio de clips configurado.
        if clips_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        return FileResponse(path, media_type="video/mp4", filename=path.name)

    @app.get("/api/stats")
    def api_stats():
        day = _today_prefix()
        week = _query(
            "SELECT substr(timestamp, 1, 10) d, COUNT(*) n, MAX(db_max) m"
            " FROM noise_events WHERE timestamp >= date(?, '-6 days') GROUP BY d ORDER BY d",
            (day,),
        )
        cats = _query(
            "SELECT sound_category c, COUNT(*) n FROM noise_events"
            " WHERE timestamp >= date(?, '-6 days') GROUP BY c ORDER BY n DESC",
            (day,),
        )
        return {"counters": _counters(), "passages": _passages_today(),
                "days": week, "categories": cats}

    @app.get("/api/health")
    def api_health():
        return {"ok": True, "db": str(db_path), "db_exists": db_path.exists()}

    return app
