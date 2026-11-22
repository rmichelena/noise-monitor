"""Detalle experimental de un evento, sin cargar los modelos en FastAPI."""

import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.analysis_store import latest_analysis

logger = logging.getLogger(__name__)


def analysis_router(database: Path, templates: Jinja2Templates) -> APIRouter:
    router = APIRouter()

    @router.get("/api/events/{event_id}/analysis")
    def api_analysis(event_id: int, event_table: Literal["noise_events", "events"] | None = None):
        try:
            with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=3)) as conn:
                conn.row_factory = sqlite3.Row
                conn.execute("BEGIN")
                result = latest_analysis(conn, event_id, event_table)
        except sqlite3.Error as exc:
            logger.warning("Consulta de análisis acústico: %s", exc)
            raise HTTPException(status_code=503, detail="Análisis no disponible") from exc
        if result is None:
            raise HTTPException(status_code=404, detail="Evento no encontrado")
        return result

    @router.get("/events/{event_id}/analysis", response_class=HTMLResponse)
    def analysis_page(
        request: Request, event_id: int, event_table: Literal["noise_events", "events"] | None = None,
    ):
        return templates.TemplateResponse(
            request, "analysis.html", {"result": api_analysis(event_id, event_table)},
            headers={"Cache-Control": "no-store"},
        )

    return router
