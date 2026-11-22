"""
Dashboard web del Noise Monitor (Fase 2).

Lee la base SQLite en modo solo-lectura y expone el nivel dB actual desde el
AudioMonitor en memoria. No escribe nada en la base.

Endpoints:
  GET /                    página principal
  GET /api/live            dB actual + contadores del día
  GET /api/events          últimos eventos (filtros: limit, category, min_db)
  GET /api/clips/{id}      clip MP4 de un evento
  GET /api/export/events.csv     eventos del periodo en CSV (evidencia)
  GET /api/export/passages.csv   pasajes del periodo en CSV
  GET /api/stats           agregados de hoy / 7 días
  GET /api/health          estado del proceso web
"""

from __future__ import annotations

import csv
import io
import logging
import math
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from src.utils import resolve_path

PAGE_SIZE = 20
# Misma escala que el overlay de los clips.
DB_BANDS = [
    (40.0, "#50c850", "verde"),
    (50.0, "#ffd23f", "amarillo"),
    (60.0, "#ffb020", "naranja"),
    (75.0, "#ff5c5c", "rojo"),
]
DB_MAX = 75.0


def db_color(value: float) -> str:
    """Color de banda para un nivel (misma escala que el overlay)."""
    for limit, color, _ in DB_BANDS:
        if value <= limit:
            return color
    return DB_BANDS[-1][1]
logger = logging.getLogger(__name__)

EVENTS_CSV_COLUMNS = [
    "id", "timestamp", "db_max", "db_avg", "trigger_type", "sound_class",
    "sound_category", "sound_confidence", "vehicle_class", "lane", "confidence",
    "passage_id", "duration_ms", "audio_muted", "pinned", "clip_path",
]
PASSAGES_CSV_COLUMNS = [
    "id", "timestamp", "timestamp_exit", "vehicle_class", "lane", "direction",
    "speed_kmh", "detection_score", "db_at_time", "noise_event_id", "pinned",
    "clip_path",
]



def create_app(config: dict, base_dir: Path, audio_monitor=None, video_monitor=None) -> FastAPI:
    base_dir = Path(base_dir)
    db_path = resolve_path(config, "database", base_dir)
    clips_dir = resolve_path(config, "clips", base_dir)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    app = FastAPI(title="Noise Monitor", version="1.0", docs_url="/api/docs")

    @app.middleware("http")
    async def no_cache(request, call_next):
        """El dashboard cambia cada segundo: que el navegador no lo cachee."""
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/partials") or path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def _connect() -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3)
        conn.row_factory = sqlite3.Row
        return conn

    def _write(sql: str, params: tuple = ()) -> None:
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()

    def _query(sql: str, params: tuple = ()) -> list[dict]:
        try:
            conn = _connect()
        except sqlite3.Error:
            return []
        try:
            rows = [dict(r) for r in conn.execute(sql, params).fetchall()]
            # Un -inf guardado antes del piso de silencio (o una fila heredada)
            # se serializa como `-Infinity`, que no es JSON válido: la respuesta
            # entera deja de parsearse en el navegador. En la API se sirve null.
            for row in rows:
                for key, value in row.items():
                    if isinstance(value, float) and not math.isfinite(value):
                        row[key] = None
            return rows
        except sqlite3.Error as exc:
            # Sin este log una columna inexistente devolvía una lista vacía
            # y el dashboard parecía "sin datos" en vez de roto.
            logger.warning("Consulta falló (%s): %s", sql.split()[2] if sql.split() else "?", exc)
            return []
        finally:
            conn.close()

    def _today_prefix() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _since(days: int) -> str:
        """Inicio (00:00 local) de una ventana de `days` días, hoy incluido."""
        return (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%dT00:00:00")

    def _stamp() -> str:
        return datetime.now().strftime("%Y%m%d")

    def _csv_response(rows: list[dict], columns: list[str], filename: str) -> Response:
        """CSV completo en memoria: una semana de eventos son unos miles de filas."""
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=columns, extrasaction="ignore", lineterminator="\r\n"
        )
        writer.writeheader()
        writer.writerows(rows)
        return Response(
            content=buf.getvalue(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def _counters() -> dict:
        day = _today_prefix()
        ev = _query("SELECT COUNT(*) n FROM noise_events WHERE timestamp >= ?", (day,))
        ps = _query("SELECT COUNT(*) n FROM vehicle_passages WHERE timestamp >= ?", (day,))
        maxdb = _query(
            "SELECT MAX(db_max) m FROM noise_events WHERE timestamp >= ?", (day,)
        )
        hour_ago = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
        missing = _query(
            "SELECT COUNT(*) n FROM noise_events WHERE timestamp >= ? AND COALESCE(clip_path, '') = ''",
            (hour_ago,),
        )
        peak = maxdb[0]["m"] if maxdb else None
        return {
            "events": ev[0]["n"] if ev else 0,
            "passages": ps[0]["n"] if ps else 0,
            # Un -inf guardado en la base (bloque de silencio, filas antiguas)
            # sale por la API como `-Infinity`, que no es JSON válido: ese
            # valor deja el panel en vivo congelado en el navegador.
            "max_db": round(peak, 1) if peak is not None and math.isfinite(peak) else None,
            "without_clip": missing[0]["n"] if missing else 0,
        }

    def _current_db() -> float | None:
        if audio_monitor is None:
            return None
        try:
            value = float(audio_monitor.current_db)
        except Exception:
            return None
        return round(value, 1) if math.isfinite(value) else None

    def _coral_ok() -> bool | None:
        if video_monitor is None:
            return None
        return getattr(video_monitor, "_interpreter", None) is not None

    VEHICLE_GROUPS = {"car": "car", "truck": "heavy", "bus": "heavy",
                      "motorcycle": "moto"}

    # Valores aceptados por los filtros. Un filtro con una errata
    # (`?vehicle=cards`) devolvía 200 con lista vacía y el dashboard decía
    # "sin eventos": indistinguible de un día tranquilo.
    NOISE_CATEGORIES = set(
        (config.get("audio_classifier") or {}).get("category_map") or {}
    ) | {"other"}
    VEHICLE_FILTERS = {"car", "truck", "bus", "motorcycle", "bicycle", "van",
                       "person", "unknown"} | set(config.get("noise_priority") or {})
    DIRECTIONS = {"ltr", "rtl"}

    def _validate_filter(name: str, value: str | None, allowed: set[str]) -> None:
        if value is not None and value not in allowed:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{name} desconocido: {value!r}. "
                    f"Valores válidos: {', '.join(sorted(allowed))}"
                ),
            )

    def _minute_labels(minutes: int) -> tuple[list[str], list[dict]]:
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
        """Pico de dB por minuto, coloreado por banda, con eje de tiempo."""
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
            level = max(0.0, min(float(value), DB_MAX))
            bars.append({
                "x": round(idx * width, 2),
                "w": round(width * 0.8, 2),
                "h": round(max(1.0, level / DB_MAX * (height - 5)), 2),
                "color": db_color(level),
                "minute": lbl[11:16],
                "value": round(float(value), 1),
            })
        return {"bars": bars, "times": ticks}

    def _vehicle_bars(minutes: int = 120, height: int = 60) -> dict:
        """Pasos por minuto, apilados por grupo: auto, pesado, moto.

        Bicicletas y personas no se cuentan.
        """
        end = datetime.now().replace(second=0, microsecond=0)
        start = end - timedelta(minutes=minutes - 1)
        rows = _query(
            "SELECT substr(timestamp, 1, 16) m, vehicle_class c, COUNT(*) n"
            " FROM vehicle_passages WHERE timestamp >= ? GROUP BY m, c",
            (start.strftime("%Y-%m-%dT%H:%M"),),
        )
        by_minute: dict[str, dict[str, int]] = {}
        for r in rows:
            group = VEHICLE_GROUPS.get(r["c"])
            if not group:
                continue
            slot = by_minute.setdefault(r["m"], {"car": 0, "heavy": 0, "moto": 0})
            slot[group] += r["n"]

        labels = [
            (start + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M")
            for i in range(minutes)
        ]
        counts = [by_minute.get(lbl, {"car": 0, "heavy": 0, "moto": 0}) for lbl in labels]
        max_total = max((c["car"] + c["heavy"] + c["moto"] for c in counts), default=0)
        scale = (height - 5) / max_total if max_total else 0
        width = 100.0 / minutes
        bars = []
        for idx, (lbl, c) in enumerate(zip(labels, counts)):
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
                segments.append({"y": round(base, 2), "h": round(h, 2), "color": color,
                                 "group": group, "n": c[group]})
            bars.append({
                "x": round(idx * width, 2),
                "w": round(width * 0.8, 2),
                "total": total,
                "minute": lbl[11:16],
                "segments": segments,
            })
        return {"bars": bars, "max_total": max_total, "times": _minute_labels(minutes)[1]}

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

    def _events(
        limit: int,
        category: str | None,
        min_db: float | None,
        vehicle: str | None = None,
        direction: str | None = None,
        pinned_only: bool = False,
    ) -> list[dict]:
        _validate_filter("category", category, NOISE_CATEGORIES)
        _validate_filter("vehicle", vehicle, VEHICLE_FILTERS)
        _validate_filter("direction", direction, DIRECTIONS)
        sql = (
            "SELECT e.id, e.timestamp, e.db_max, e.db_avg, e.trigger_type,"
            " e.sound_class, e.sound_category, e.sound_confidence, e.vehicle_class,"
            " e.clip_path, e.audio_muted, e.pinned, p.direction AS direction"
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
        if vehicle:
            sql += " AND e.vehicle_class = ?"
            params.append(vehicle)
        if direction:
            sql += " AND p.direction = ?"
            params.append(direction)
        if pinned_only:
            sql += " AND e.pinned = 1"
        sql += " ORDER BY e.id DESC LIMIT ?"
        params.append(limit)
        rows = _query(sql, tuple(params))
        for r in rows:
            r["has_clip"] = bool(r.pop("clip_path", None))
        return rows

    def _passages(
        limit: int,
        vehicle: str | None = None,
        direction: str | None = None,
        min_db: float | None = None,
        min_speed: float | None = None,
        pinned_only: bool = False,
    ) -> list[dict]:
        _validate_filter("vehicle", vehicle, VEHICLE_FILTERS)
        _validate_filter("direction", direction, DIRECTIONS)
        sql = (
            "SELECT id, timestamp, timestamp_exit, vehicle_class, lane, direction,"
            " speed_kmh, detection_score, db_at_time, clip_path, pinned"
            " FROM vehicle_passages WHERE 1=1"
        )
        params: list = []
        if vehicle:
            sql += " AND vehicle_class = ?"
            params.append(vehicle)
        if direction:
            sql += " AND direction = ?"
            params.append(direction)
        if min_db is not None:
            sql += " AND db_at_time >= ?"
            params.append(min_db)
        if min_speed is not None:
            sql += " AND speed_kmh >= ?"
            params.append(min_speed)
        if pinned_only:
            sql += " AND pinned = 1"
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return _query(sql, tuple(params))

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
        ev_rows = _query(
            "SELECT e.vehicle_class c, COUNT(*) n FROM noise_events e"
            " WHERE e.timestamp >= ? AND e.vehicle_class IS NOT NULL GROUP BY c",
            (day,),
        )
        events_by_class = {r["c"]: r["n"] for r in ev_rows}
        ev_dir = _query(
            "SELECT p.direction d, COUNT(*) n FROM noise_events e"
            " JOIN vehicle_passages p ON p.id = e.passage_id"
            " WHERE e.timestamp >= ? AND p.direction IS NOT NULL GROUP BY d",
            (day,),
        )
        events_by_dir = {r["d"]: r["n"] for r in ev_dir}

        sp = _query(
            "SELECT AVG(speed_kmh) v, COUNT(speed_kmh) n, MAX(speed_kmh) mx"
            " FROM vehicle_passages WHERE timestamp >= ? AND speed_kmh IS NOT NULL",
            (day,),
        )
        speed = None
        if sp and sp[0]["n"]:
            speed = {
                "avg": round(sp[0]["v"], 1),
                "max": round(sp[0]["mx"], 1),
                "n": sp[0]["n"],
            }
        return {
            "by_class": by_class,
            "by_dir": by_dir,
            "speed": speed,
            "events_by_class": events_by_class,
            "events_by_dir": events_by_dir,
            "events_total": (
                _query("SELECT COUNT(*) n FROM noise_events WHERE timestamp >= ?", (day,))[0]["n"]
            ),
        }

    # ------------------------------------------------------------------
    # Páginas
    # ------------------------------------------------------------------
    def _context() -> dict:
        return {
            "current_db": _current_db(),
            "counters": _counters(),
            "coral_ok": _coral_ok(),
            "events": _events(PAGE_SIZE, None, None),
            "passages": _passages_today(),
            "passages_list": _passages(PAGE_SIZE),
            "audio_chart": _audio_bars(),
            "vehicle_chart": _vehicle_bars(),
            "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard(request: Request):
        return templates.TemplateResponse(request, "dashboard.html", _context())

    # Fragmentos para auto-refresh sin recargar ni tocar el modal abierto.
    @app.get("/partials/live", response_class=HTMLResponse)
    def partial_live(request: Request):
        return templates.TemplateResponse(request, "_live.html", _context())

    @app.get("/partials/events", response_class=HTMLResponse)
    def partial_events(
        request: Request,
        vehicle: str | None = None,
        direction: str | None = None,
        min_db: float | None = None,
        pinned: int = 0,
    ):
        ctx = _context()
        ctx["events"] = _events(
            PAGE_SIZE, None, min_db, vehicle, direction, pinned_only=bool(pinned)
        )
        ctx["filter_label"] = vehicle or direction
        return templates.TemplateResponse(request, "_events.html", ctx)

    @app.get("/partials/spark", response_class=HTMLResponse)
    def partial_spark(request: Request):
        return templates.TemplateResponse(request, "_spark.html", _context())

    @app.get("/partials/filters", response_class=HTMLResponse)
    def partial_filters(request: Request):
        return templates.TemplateResponse(request, "_filters.html", _context())

    @app.get("/partials/traffic", response_class=HTMLResponse)
    def partial_traffic(request: Request):
        return templates.TemplateResponse(request, "_traffic.html", _context())

    @app.get("/partials/vehicle-chart", response_class=HTMLResponse)
    def partial_vehicle_chart(request: Request):
        return templates.TemplateResponse(request, "_vehicle_chart.html", _context())

    @app.get("/partials/passages", response_class=HTMLResponse)
    def partial_passages(
        request: Request,
        vehicle: str | None = None,
        direction: str | None = None,
        min_db: float | None = None,
        min_speed: float | None = None,
        pinned: int = 0,
    ):
        ctx = _context()
        ctx["passages_list"] = _passages(
            PAGE_SIZE, vehicle, direction, min_db, min_speed, bool(pinned)
        )
        ctx["filter_label"] = vehicle or direction
        return templates.TemplateResponse(request, "_passages.html", ctx)

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
            "events_without_clip_1h": counters["without_clip"],
            "coral_ok": _coral_ok(),
            "now": datetime.now().strftime("%H:%M:%S"),
        }

    @app.get("/api/events")
    def api_events(
        limit: int = Query(20, ge=1, le=200),
        category: str | None = None,
        min_db: float | None = None,
        vehicle: str | None = None,
        direction: str | None = None,
    ):
        return {"events": _events(limit, category, min_db, vehicle, direction)}

    @app.get("/api/export/events.csv")
    def export_events(
        days: int = Query(7, ge=1, le=366),
        min_db: float | None = None,
        vehicle: str | None = None,
        category: str | None = None,
    ):
        """Eventos de los últimos `days` días en CSV (evidencia para un reclamo)."""
        since = _since(days)
        sql = (
            "SELECT id, timestamp, db_max, db_avg, trigger_type, sound_class,"
            " sound_category, sound_confidence, vehicle_class, lane, confidence,"
            " passage_id, duration_ms, audio_muted, pinned, clip_path"
            " FROM noise_events WHERE timestamp >= ?"
        )
        params: list = [since]
        if min_db is not None:
            sql += " AND db_max >= ?"
            params.append(min_db)
        if vehicle:
            _validate_filter("vehicle", vehicle, VEHICLE_FILTERS)
            sql += " AND vehicle_class = ?"
            params.append(vehicle)
        if category:
            _validate_filter("category", category, NOISE_CATEGORIES)
            sql += " AND sound_category = ?"
            params.append(category)
        sql += " ORDER BY timestamp"
        rows = _query(sql, tuple(params))
        return _csv_response(
            rows, EVENTS_CSV_COLUMNS, f"eventos_{_stamp()}.csv"
        )

    @app.get("/api/export/passages.csv")
    def export_passages(
        days: int = Query(7, ge=1, le=366),
        min_db: float | None = None,
        min_speed: float | None = None,
        vehicle: str | None = None,
        direction: str | None = None,
    ):
        """Pasajes de vehículos de los últimos `days` días en CSV."""
        since = _since(days)
        sql = (
            "SELECT id, timestamp, timestamp_exit, vehicle_class, lane, direction,"
            " speed_kmh, detection_score, db_at_time, noise_event_id, pinned, clip_path"
            " FROM vehicle_passages WHERE timestamp >= ?"
        )
        params: list = [since]
        if min_db is not None:
            sql += " AND db_at_time >= ?"
            params.append(min_db)
        if min_speed is not None:
            sql += " AND speed_kmh >= ?"
            params.append(min_speed)
        if vehicle:
            _validate_filter("vehicle", vehicle, VEHICLE_FILTERS)
            sql += " AND vehicle_class = ?"
            params.append(vehicle)
        if direction:
            _validate_filter("direction", direction, DIRECTIONS)
            sql += " AND direction = ?"
            params.append(direction)
        sql += " ORDER BY timestamp"
        rows = _query(sql, tuple(params))
        return _csv_response(
            rows, PASSAGES_CSV_COLUMNS, f"pasajes_{_stamp()}.csv"
        )

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
        # inline: el navegador lo reproduce en el modal en vez de descargarlo
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Content-Disposition": f'inline; filename="{path.name}"'},
        )

    @app.get("/api/detections")
    def api_detections():
        """Detecciones del último frame de inferencia (diagnóstico)."""
        if video_monitor is None:
            return {"detections": [], "coral_ok": None}
        dets = []
        for d in video_monitor.recent_detections:
            dets.append({"class": d["class"], "score": round(d["score"], 2)})
        return {"detections": dets, "coral_ok": _coral_ok()}

    @app.post("/api/events/{event_id}/pin")
    def pin_event(event_id: int, pinned: int = 1):
        rows = _query("SELECT id FROM noise_events WHERE id = ?", (event_id,))
        if not rows:
            raise HTTPException(status_code=404, detail="Evento no encontrado")
        if pinned:
            _write(
                "UPDATE noise_events SET pinned = 1, pinned_at = ? WHERE id = ?",
                (datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), event_id),
            )
        else:
            _write(
                "UPDATE noise_events SET pinned = 0, pinned_at = NULL WHERE id = ?",
                (event_id,),
            )
        return {"ok": True, "id": event_id, "pinned": bool(pinned)}

    @app.post("/api/passages/{passage_id}/pin")
    def pin_passage(passage_id: int, pinned: int = 1):
        rows = _query("SELECT id FROM vehicle_passages WHERE id = ?", (passage_id,))
        if not rows:
            raise HTTPException(status_code=404, detail="Pasaje no encontrado")
        if pinned:
            _write(
                "UPDATE vehicle_passages SET pinned = 1, pinned_at = ? WHERE id = ?",
                (datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), passage_id),
            )
        else:
            _write(
                "UPDATE vehicle_passages SET pinned = 0, pinned_at = NULL WHERE id = ?",
                (passage_id,),
            )
        return {"ok": True, "id": passage_id, "pinned": bool(pinned)}

    @app.get("/api/passage-clips/{passage_id}")
    def api_passage_clip(passage_id: int):
        rows = _query(
            "SELECT clip_path FROM vehicle_passages WHERE id = ?", (passage_id,)
        )
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

    @app.get("/stream")
    def stream():
        """Vista en vivo MJPEG con cajas dibujadas sobre el frame actual."""
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
