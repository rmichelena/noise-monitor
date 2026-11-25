"""
Dashboard web del Noise Monitor (Fase 2).

Lee la tabla única `events` (kind = noise | traffic | both) en modo
solo-lectura y expone el nivel dB actual desde el AudioMonitor en memoria.
Sólo escribe la marca de fijado.

- "Ruido" son las filas `noise` y `both` (hubo disparo acústico).
- "Pasaje" son las filas `traffic` y `both` (el tracker vio un vehículo).

Endpoints:
  GET /                    página principal
  GET /api/live            dB actual + contadores del día
  GET /api/events          últimos eventos de todos los kinds (filtros)
  GET /api/passages        alias: eventos con pasaje (traffic + both)
  GET /api/clips/{id}      clip MP4 de un evento (cualquier kind)
  GET /api/passage-clips/{id}    alias de /api/clips para pasajes
  POST /api/events/{id}/pin      fija/desfija un evento
  POST /api/passages/{id}/pin    alias para pasajes
  GET /api/export/events.csv     eventos del periodo en CSV (evidencia)
  GET /api/export/passages.csv   pasajes del periodo en CSV
  GET /api/stats           agregados de hoy / 7 días
  GET /api/health          estado del proceso web
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import os
import secrets
import shutil
import signal
import sqlite3
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.templating import Jinja2Templates

from src.trigger_policy import TriggerPolicy
from src.utils import ConfigError, load_config, resolve_path

from .analysis import analysis_router
from .settings import public_payload, read_config, update_config, validate_startable

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

KINDS = ("noise", "traffic", "both")
NOISE_KINDS = "('noise', 'both')"
PASSAGE_KINDS = "('traffic', 'both')"

EVENTS_CSV_COLUMNS = [
    "id", "kind", "timestamp", "timestamp_end", "db_max", "db_avg", "trigger_type",
    "sound_class", "sound_category", "sound_confidence", "vehicle_class", "lane",
    "direction", "speed_kmh", "confidence", "duration_ms", "audio_muted", "pinned",
    "clip_path",
]
PASSAGES_CSV_COLUMNS = [
    "id", "kind", "timestamp", "timestamp_end", "vehicle_class", "lane", "direction",
    "speed_kmh", "detection_score", "db_max", "pinned", "clip_path",
]
# Fin de la ventana del evento en el mismo formato ISO local que `timestamp`.
TIMESTAMP_END = "strftime('%Y-%m-%dT%H:%M:%S', end_ts, 'unixepoch', 'localtime') AS timestamp_end"



# Reinicio desde /settings. El proceso web corre como el usuario del servicio,
# con sudo sin contraseña; --no-block encola el reinicio y vuelve enseguida.
RESTART_UNIT = "noise-monitor.service"
RESTART_CMD = ["sudo", "-n", "systemctl", "--no-block", "restart", RESTART_UNIT]
# `sudo -l <cmd>` comprueba sin ejecutar que sudo permite ese comando sin pedir clave.
RESTART_PREFLIGHT = ["sudo", "-n", "-l", "systemctl", "--no-block", "restart", RESTART_UNIT]
RESTART_DELAY_SECONDS = 1.0


def create_app(config: dict, base_dir: Path, audio_monitor=None, video_monitor=None) -> FastAPI:
    base_dir = Path(base_dir)
    db_path = resolve_path(config, "database", base_dir)
    clips_dir = resolve_path(config, "clips", base_dir)
    templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

    app = FastAPI(title="Noise Monitor", version="1.0", docs_url="/api/docs")

    app.include_router(analysis_router(db_path, templates))

    settings_security = HTTPBasic(auto_error=False)

    def require_settings(
        credentials: HTTPBasicCredentials | None = Depends(settings_security),
    ) -> None:
        """Basic auth sólo para los ajustes; falla cerrado si no hay password."""
        password = os.environ.get("NOISE_SETTINGS_PASSWORD", "")
        if not password:
            raise HTTPException(
                status_code=503,
                detail="Ajustes deshabilitados: definir NOISE_SETTINGS_PASSWORD",
            )
        user = os.environ.get("NOISE_SETTINGS_USER", "admin")
        valid = credentials is not None and secrets.compare_digest(
            credentials.username.encode(), user.encode()
        ) and secrets.compare_digest(credentials.password.encode(), password.encode())
        if not valid:
            raise HTTPException(
                status_code=401,
                detail="Autenticación requerida",
                headers={"WWW-Authenticate": 'Basic realm="Noise Monitor settings"'},
            )

    @app.middleware("http")
    async def no_cache(request, call_next):
        """El dashboard cambia cada segundo: que el navegador no lo cachee."""
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/partials") or path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def _connect() -> sqlite3.Connection:
        # check_same_thread=False: el generador del CSV se itera en el
        # threadpool de Starlette, distinto del thread que abrió la conexión.
        conn = sqlite3.connect(
            f"file:{db_path}?mode=ro", uri=True, timeout=3, check_same_thread=False
        )
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

    MAX_EXPORT_ROWS = 100_000
    EXPORT_BATCH = 500

    def _csv_stream(sql: str, params: tuple, columns: list[str]):
        """Genera el CSV por lotes; nunca materializa el periodo completo."""
        conn = _connect()
        try:
            cur = conn.execute(sql, params)
            buf = io.StringIO()
            writer = csv.DictWriter(
                buf, fieldnames=columns, extrasaction="ignore", lineterminator="\r\n"
            )
            writer.writeheader()
            yield buf.getvalue()
            sent = 0
            while sent < MAX_EXPORT_ROWS:
                rows = cur.fetchmany(min(EXPORT_BATCH, MAX_EXPORT_ROWS - sent))
                if not rows:
                    break
                buf = io.StringIO()
                writer = csv.DictWriter(
                    buf, fieldnames=columns, extrasaction="ignore", lineterminator="\r\n"
                )
                writer.writerows(dict(r) for r in rows)
                yield buf.getvalue()
                sent += len(rows)
            if sent >= MAX_EXPORT_ROWS:
                logger.warning(
                    "Export CSV truncado a %d filas; acota days/filtros",
                    MAX_EXPORT_ROWS,
                )
        finally:
            conn.close()

    def _csv_response(
        sql: str, params: tuple, columns: list[str], filename: str
    ) -> StreamingResponse:
        """Valida la consulta antes de enviar cabeceras y luego hace streaming."""
        stream = _csv_stream(sql, params, columns)
        try:
            first = next(stream)
        except sqlite3.Error as exc:
            logger.exception("Export CSV falló: %s", exc)
            raise HTTPException(status_code=500, detail="No se pudo exportar")
        except StopIteration:
            first = ""

        def body():
            yield first
            yield from stream

        return StreamingResponse(
            body(),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    def _counters() -> dict:
        day = _today_prefix()
        ev = _query(
            f"SELECT COUNT(*) n FROM events WHERE kind IN {NOISE_KINDS} AND timestamp >= ?", (day,)
        )
        ps = _query(
            f"SELECT COUNT(*) n FROM events WHERE kind IN {PASSAGE_KINDS} AND timestamp >= ?", (day,)
        )
        maxdb = _query(
            f"SELECT MAX(db_max) m FROM events WHERE kind IN {NOISE_KINDS} AND timestamp >= ?", (day,)
        )
        hour_ago = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
        missing = _query(
            f"SELECT COUNT(*) n FROM events WHERE kind IN {NOISE_KINDS} AND timestamp >= ?"
            " AND COALESCE(clip_path, '') = ''",
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
    # "sin eventos": indistinguible de un día tranquilo. Se incluyen los
    # valores históricos de la DB para que un cambio de config no congele
    # el panel con un 400.
    _BASE_CATEGORIES = set(
        (config.get("audio_classifier") or {}).get("category_map") or {}
    ) | {"other"}
    _BASE_VEHICLES = {"car", "truck", "bus", "motorcycle", "bicycle", "van",
                      "person", "unknown"} | set(config.get("noise_priority") or {})
    NOISE_CATEGORIES = _BASE_CATEGORIES | {
        r["c"] for r in _query(
            "SELECT DISTINCT sound_category c FROM events"
            " WHERE sound_category IS NOT NULL AND sound_category != ''"
        )
    }
    VEHICLE_FILTERS = _BASE_VEHICLES | {
        r["c"] for r in _query(
            "SELECT DISTINCT vehicle_class c FROM events"
            " WHERE vehicle_class IS NOT NULL AND vehicle_class != ''"
        )
    }
    DIRECTIONS = {"ltr", "rtl"}
    # Conjuntos además de los kinds: `sound` = noise + both (lo que cuentan los
    # contadores de eventos) y `passages` = traffic + both (los de tráfico).
    KIND_FILTERS = set(KINDS) | {"sound", "passages"}

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
            f" FROM events WHERE kind IN {PASSAGE_KINDS} AND timestamp >= ? GROUP BY m, c",
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

    def _filter_sql(
        kind: str | None,
        category: str | None,
        min_db: float | None,
        vehicle: str | None,
        direction: str | None,
        min_speed: float | None,
        pinned_only: bool,
    ) -> tuple[str, list]:
        """WHERE común a la lista, la API y los CSV; valida antes de consultar."""
        _validate_filter("kind", kind, KIND_FILTERS)
        _validate_filter("category", category, NOISE_CATEGORIES)
        _validate_filter("vehicle", vehicle, VEHICLE_FILTERS)
        _validate_filter("direction", direction, DIRECTIONS)
        sql = " WHERE 1=1"
        params: list = []
        if kind == "passages":
            sql += f" AND kind IN {PASSAGE_KINDS}"
        elif kind == "sound":
            sql += f" AND kind IN {NOISE_KINDS}"
        elif kind:
            sql += " AND kind = ?"
            params.append(kind)
        if category:
            sql += " AND sound_category = ?"
            params.append(category)
        if min_db is not None:
            sql += " AND db_max >= ?"
            params.append(min_db)
        if vehicle:
            sql += " AND vehicle_class = ?"
            params.append(vehicle)
        if direction:
            sql += " AND direction = ?"
            params.append(direction)
        if min_speed is not None:
            sql += " AND speed_kmh >= ?"
            params.append(min_speed)
        if pinned_only:
            sql += " AND pinned = 1"
        return sql, params

    def _events(
        limit: int,
        category: str | None = None,
        min_db: float | None = None,
        vehicle: str | None = None,
        direction: str | None = None,
        pinned_only: bool = False,
        kind: str | None = None,
        min_speed: float | None = None,
    ) -> list[dict]:
        where, params = _filter_sql(
            kind, category, min_db, vehicle, direction, min_speed, pinned_only
        )
        sql = (
            "SELECT id, kind, timestamp, db_max, db_avg, trigger_type,"
            " sound_class, sound_category, sound_confidence, sound_labels_json,"
            " vehicle_class, lane, direction, speed_kmh, detection_score,"
            " clip_path, audio_muted, pinned FROM events"
            + where
            # Los ids de traffic migrados no siguen el orden temporal.
            + " ORDER BY timestamp DESC, id DESC LIMIT ?"
        )
        params.append(limit)
        rows = _query(sql, tuple(params))
        for r in rows:
            r["has_clip"] = bool(r.pop("clip_path", None))
            encoded = r.pop("sound_labels_json", None)
            try:
                labels = json.loads(encoded) if encoded else []
            except json.JSONDecodeError:
                labels = []
            r["sound_labels"] = labels if isinstance(labels, list) else []
        return rows

    def _passages_today() -> dict:
        day = _today_prefix()
        rows = _query(
            "SELECT vehicle_class, direction, COUNT(*) n FROM events"
            f" WHERE kind IN {PASSAGE_KINDS} AND timestamp >= ?"
            " GROUP BY vehicle_class, direction",
            (day,),
        )
        by_class: dict[str, int] = {}
        by_dir: dict[str, int] = {}
        for r in rows:
            by_class[r["vehicle_class"]] = by_class.get(r["vehicle_class"], 0) + r["n"]
            if r["direction"]:
                by_dir[r["direction"]] = by_dir.get(r["direction"], 0) + r["n"]
        ev_rows = _query(
            f"SELECT vehicle_class c, COUNT(*) n FROM events WHERE kind IN {NOISE_KINDS}"
            " AND timestamp >= ? AND vehicle_class IS NOT NULL GROUP BY c",
            (day,),
        )
        events_by_class = {r["c"]: r["n"] for r in ev_rows}
        ev_dir = _query(
            f"SELECT direction d, COUNT(*) n FROM events WHERE kind IN {NOISE_KINDS}"
            " AND timestamp >= ? AND direction IS NOT NULL GROUP BY d",
            (day,),
        )
        events_by_dir = {r["d"]: r["n"] for r in ev_dir}
        by_kind = {
            r["kind"]: r["n"] for r in _query(
                "SELECT kind, COUNT(*) n FROM events WHERE timestamp >= ? GROUP BY kind",
                (day,),
            )
        }

        sp = _query(
            "SELECT AVG(speed_kmh) v, COUNT(speed_kmh) n, MAX(speed_kmh) mx"
            f" FROM events WHERE kind IN {PASSAGE_KINDS} AND timestamp >= ?"
            " AND speed_kmh IS NOT NULL",
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
            "by_kind": {k: by_kind.get(k, 0) for k in KINDS},
            "events_total": by_kind.get("noise", 0) + by_kind.get("both", 0),
        }

    # ------------------------------------------------------------------
    # Páginas
    # ------------------------------------------------------------------
    def _context() -> dict:
        return {
            "current_db": _current_db(),
            "counters": _counters(),
            "coral_ok": _coral_ok(),
            "events": _events(PAGE_SIZE),
            "passages": _passages_today(),
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
        kind: str | None = None,
        category: str | None = None,
        vehicle: str | None = None,
        direction: str | None = None,
        min_db: float | None = None,
        min_speed: float | None = None,
        pinned: int = 0,
    ):
        ctx = _context()
        ctx["events"] = _events(
            PAGE_SIZE, category, min_db, vehicle, direction,
            pinned_only=bool(pinned), kind=kind, min_speed=min_speed,
        )
        ctx["filter_label"] = vehicle or direction or category
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
        kind: str | None = None,
        category: str | None = None,
        min_db: float | None = None,
        vehicle: str | None = None,
        direction: str | None = None,
        min_speed: float | None = None,
        pinned: int = 0,
    ):
        return {"events": _events(
            limit, category, min_db, vehicle, direction,
            pinned_only=bool(pinned), kind=kind, min_speed=min_speed,
        )}

    @app.get("/api/passages")
    def api_passages(
        limit: int = Query(20, ge=1, le=200),
        min_db: float | None = None,
        vehicle: str | None = None,
        direction: str | None = None,
        min_speed: float | None = None,
        pinned: int = 0,
    ):
        """Alias de /api/events con pasaje (traffic + both)."""
        return {"passages": _events(
            limit, None, min_db, vehicle, direction,
            pinned_only=bool(pinned), kind="passages", min_speed=min_speed,
        )}

    @app.get("/api/export/events.csv")
    def export_events(
        days: int = Query(7, ge=1, le=366),
        kind: str | None = None,
        min_db: float | None = None,
        vehicle: str | None = None,
        category: str | None = None,
        direction: str | None = None,
        min_speed: float | None = None,
    ):
        """Eventos de los últimos `days` días en CSV (evidencia para un reclamo)."""
        where, params = _filter_sql(
            kind, category, min_db, vehicle, direction, min_speed, False
        )
        sql = (
            "SELECT id, kind, timestamp, " + TIMESTAMP_END + ", db_max, db_avg,"
            " trigger_type, sound_class, sound_category, sound_confidence,"
            " vehicle_class, lane, direction, speed_kmh, confidence, duration_ms,"
            " audio_muted, pinned, clip_path FROM events"
            + where + " AND timestamp >= ? ORDER BY timestamp, id"
        )
        params.append(_since(days))
        return _csv_response(
            sql, tuple(params), EVENTS_CSV_COLUMNS, f"eventos_{_stamp()}.csv"
        )

    @app.get("/api/export/passages.csv")
    def export_passages(
        days: int = Query(7, ge=1, le=366),
        min_db: float | None = None,
        min_speed: float | None = None,
        vehicle: str | None = None,
        direction: str | None = None,
    ):
        """Pasajes (traffic + both) de los últimos `days` días en CSV."""
        where, params = _filter_sql(
            "passages", None, min_db, vehicle, direction, min_speed, False
        )
        sql = (
            "SELECT id, kind, timestamp, " + TIMESTAMP_END + ", vehicle_class, lane,"
            " direction, speed_kmh, detection_score, db_max, pinned, clip_path"
            " FROM events" + where + " AND timestamp >= ? ORDER BY timestamp, id"
        )
        params.append(_since(days))
        return _csv_response(
            sql, tuple(params), PASSAGES_CSV_COLUMNS, f"pasajes_{_stamp()}.csv"
        )

    def _serve_clip(event_id: int, kinds: str | None = None) -> FileResponse:
        sql = "SELECT clip_path FROM events WHERE id = ?"
        if kinds:
            sql += f" AND kind IN {kinds}"
        rows = _query(sql, (event_id,))
        if not rows or not rows[0]["clip_path"]:
            raise HTTPException(status_code=404, detail="Clip no encontrado")
        stored = Path(rows[0]["clip_path"])
        path = (stored if stored.is_absolute() else base_dir / stored).resolve()
        # Solo se sirven archivos dentro del directorio de clips configurado.
        if clips_dir.resolve() not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        # inline: el navegador lo reproduce en el modal en vez de descargarlo
        return FileResponse(
            path,
            media_type="video/mp4",
            headers={"Content-Disposition": f'inline; filename="{path.name}"'},
        )

    @app.get("/api/clips/{event_id}")
    def api_clip(event_id: int):
        return _serve_clip(event_id)

    @app.get("/api/passage-clips/{event_id}")
    def api_passage_clip(event_id: int):
        """Alias de /api/clips restringido a eventos con pasaje."""
        return _serve_clip(event_id, PASSAGE_KINDS)

    @app.get("/api/detections")
    def api_detections():
        """Detecciones del último frame de inferencia (diagnóstico)."""
        if video_monitor is None:
            return {"detections": [], "coral_ok": None}
        dets = []
        for d in video_monitor.recent_detections:
            dets.append({"class": d["class"], "score": round(d["score"], 2)})
        return {"detections": dets, "coral_ok": _coral_ok()}

    def _pin(event_id: int, pinned: int, kinds: str | None, missing: str) -> dict:
        sql = "SELECT id FROM events WHERE id = ?"
        if kinds:
            sql += f" AND kind IN {kinds}"
        if not _query(sql, (event_id,)):
            raise HTTPException(status_code=404, detail=missing)
        if pinned:
            _write(
                "UPDATE events SET pinned = 1, pinned_at = ? WHERE id = ?",
                (datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), event_id),
            )
        else:
            _write(
                "UPDATE events SET pinned = 0, pinned_at = NULL WHERE id = ?",
                (event_id,),
            )
        return {"ok": True, "id": event_id, "pinned": bool(pinned)}

    @app.post("/api/events/{event_id}/pin")
    def pin_event(event_id: int, pinned: int = 1):
        return _pin(event_id, pinned, None, "Evento no encontrado")

    @app.post("/api/passages/{event_id}/pin")
    def pin_passage(event_id: int, pinned: int = 1):
        """Alias de /api/events/{id}/pin restringido a eventos con pasaje."""
        return _pin(event_id, pinned, PASSAGE_KINDS, "Pasaje no encontrado")

    @app.get("/api/stats")
    def api_stats():
        day = _today_prefix()
        week = _query(
            "SELECT substr(timestamp, 1, 10) d, COUNT(*) n, MAX(db_max) m"
            f" FROM events WHERE kind IN {NOISE_KINDS} AND timestamp >= date(?, '-6 days')"
            " GROUP BY d ORDER BY d",
            (day,),
        )
        cats = _query(
            "SELECT sound_category c, COUNT(*) n FROM events"
            f" WHERE kind IN {NOISE_KINDS} AND timestamp >= date(?, '-6 days')"
            " GROUP BY c ORDER BY n DESC",
            (day,),
        )
        kinds = _query(
            "SELECT kind, COUNT(*) n FROM events WHERE timestamp >= date(?, '-6 days')"
            " GROUP BY kind ORDER BY kind",
            (day,),
        )
        return {"counters": _counters(), "passages": _passages_today(),
                "days": week, "categories": cats, "kinds": kinds}

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

    settings_path = base_dir / "config.yaml"

    def _settings_payload() -> dict:
        try:
            return public_payload(read_config(settings_path))
        except Exception as exc:
            logger.exception("No se pudo leer %s", settings_path)
            raise HTTPException(status_code=500, detail=f"config.yaml: {exc}") from exc

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(request: Request, _: None = Depends(require_settings)):
        return templates.TemplateResponse(request, "settings.html", {"payload": _settings_payload()})

    @app.get("/api/settings")
    def settings_read(_: None = Depends(require_settings)):
        return _settings_payload()

    def _reject_cross_site(request: Request) -> None:
        """CSRF: el navegador reenvía Basic auth a cualquier sitio que postee acá.

        Un `<form enctype="text/plain">` de otro origen es una request simple
        (sin preflight) cuyo cuerpo puede ser JSON válido. Se exige
        `application/json` —que cross-origin obliga a un preflight que este
        server no autoriza— y, además, que el navegador no la marque como de
        otro sitio: `Sec-Fetch-Site` si lo manda; si no, `Origin` contra el host.
        """
        ctype = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            raise HTTPException(status_code=415, detail="Content-Type debe ser application/json")
        site = request.headers.get("sec-fetch-site")
        if site is not None:
            if site not in ("same-origin", "none"):
                raise HTTPException(status_code=403, detail="Origen no permitido")
            return
        origin = request.headers.get("origin")
        if origin is not None:
            hosts = {h for h in (request.headers.get("host"), request.headers.get("x-forwarded-host")) if h}
            if origin == "null" or urlsplit(origin).netloc not in hosts:
                raise HTTPException(status_code=403, detail="Origen no permitido")

    @app.post("/api/settings")
    async def settings_write(request: Request, _: None = Depends(require_settings)):
        _reject_cross_site(request)
        try:
            updates = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="JSON inválido") from exc
        try:
            changed, backup = update_config(settings_path, updates, base_dir)
        except (ValueError, ConfigError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("No se pudo guardar %s", settings_path)
            raise HTTPException(status_code=500, detail=f"config.yaml: {exc}") from exc
        if changed:
            logger.info("Ajustes guardados en %s: %s", settings_path, ", ".join(changed))
        return {
            "saved": bool(changed),
            "changed": changed,
            "restart_required": bool(changed),
            "backup": str(backup) if backup else None,
        }

    # Distingue el proceso nuevo del viejo: la página espera a que cambie.
    started_at = time.time()
    # La política con la que corre el proceso (la de AudioMonitor; si no hay,
    # la de la config cargada al arrancar). Se expone en /api/health para que
    # una config degradada —p. ej. tráfico sin clases— no pase inadvertida.
    trigger_policy = getattr(audio_monitor, "trigger_policy", None)
    if not isinstance(trigger_policy, TriggerPolicy):
        try:
            trigger_policy = TriggerPolicy.from_config(config)
        except ValueError:
            trigger_policy = None
    restart_state = {"pending": False}
    restart_lock = threading.Lock()

    def _run_restart() -> None:
        try:
            result = subprocess.run(RESTART_CMD, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("No se pudo reiniciar %s: %s", RESTART_UNIT, exc)
        else:
            if result.returncode == 0:
                logger.info("Reinicio de %s encolado desde /settings", RESTART_UNIT)
                return
            if result.returncode == -signal.SIGTERM:
                # Lo esperable es que el reinicio encolado baje el servicio y
                # termine a systemctl, que corre en su cgroup; pero un SIGTERM
                # no prueba que el reinicio se haya encolado. Aviso, no error,
                # y se libera el pedido para poder reintentar.
                logger.warning("Reinicio de %s: systemctl terminado por SIGTERM (rc=-15); "
                               "si el servicio no se reinició, reintentar", RESTART_UNIT)
            else:
                logger.error("Reinicio de %s falló (rc=%s): %s", RESTART_UNIT,
                             result.returncode, result.stderr.strip())
        with restart_lock:
            restart_state["pending"] = False

    @app.post("/api/settings/restart", status_code=202)
    def settings_restart(request: Request, _: None = Depends(require_settings)):
        _reject_cross_site(request)
        # La config que leerá el servicio al arrancar, con su mismo loader: si
        # no arrancaría, no se reinicia (el servicio actual sigue andando).
        try:
            validate_startable(load_config(settings_path), base_dir)
        except (FileNotFoundError, ValueError, ConfigError) as exc:
            raise HTTPException(status_code=400, detail=f"config.yaml no arrancaría: {exc}") from exc
        with restart_lock:
            if restart_state["pending"]:
                return {"restarting": True, "delay_seconds": RESTART_DELAY_SECONDS}
            if shutil.which("sudo") is None or shutil.which("systemctl") is None:
                logger.error("Reinicio desde /settings: falta sudo o systemctl en PATH")
                raise HTTPException(status_code=500, detail="No hay sudo/systemctl para reiniciar el servicio")
            try:
                check = subprocess.run(RESTART_PREFLIGHT, capture_output=True, text=True, timeout=10)
            except (OSError, subprocess.SubprocessError) as exc:
                logger.error("Reinicio desde /settings: sudo no respondió: %s", exc)
                raise HTTPException(status_code=500, detail="No se pudo comprobar el permiso de reinicio") from exc
            if check.returncode != 0:
                logger.error("Reinicio desde /settings: sudo no lo permite sin clave (rc=%s): %s",
                             check.returncode, check.stderr.strip())
                raise HTTPException(status_code=500, detail="sudo no permite reiniciar el servicio sin contraseña")
            restart_state["pending"] = True
            # Se responde primero: el reinicio mata a este mismo proceso.
            timer = threading.Timer(RESTART_DELAY_SECONDS, _run_restart)
            timer.daemon = True
            timer.start()
        logger.info("Reinicio de %s pedido desde /settings", RESTART_UNIT)
        return {"restarting": True, "delay_seconds": RESTART_DELAY_SECONDS}

    @app.get("/api/settings/snapshot")
    def settings_snapshot(_: None = Depends(require_settings)):
        getter = getattr(video_monitor, "latest_frame_jpeg", None)
        jpeg = getter() if getter else None
        if not jpeg:
            raise HTTPException(status_code=404, detail="Aún no hay frames de video")
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/health")
    def api_health():
        health = {"ok": True, "db": str(db_path), "db_exists": db_path.exists(),
                  "started_at": started_at}
        status = getattr(audio_monitor, "trigger_status", None)
        if status is not None:
            health["trigger"] = status()
        classifier_status = getattr(audio_monitor, "classifier_status", None)
        if classifier_status is not None:
            health["audio_classifier"] = classifier_status()
        if trigger_policy is not None:
            now = time.time()
            health["policy"] = {
                "noise_enabled": trigger_policy.noise_enabled,
                "noise_schedule_enabled": trigger_policy.noise_schedule.enabled,
                "noise_active_now": trigger_policy.noise_active(now),
                "noise_only_with_traffic": trigger_policy.noise_only_with_traffic,
                "traffic_enabled": trigger_policy.traffic_enabled,
                "traffic_schedule_enabled": trigger_policy.traffic_schedule.enabled,
                "traffic_active_now": trigger_policy.traffic_active(now),
                "traffic_only_with_noise": trigger_policy.traffic_only_with_noise,
                "record_classes": sorted(trigger_policy.record_classes),
            }
        return health

    return app
