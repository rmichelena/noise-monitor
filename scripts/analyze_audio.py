#!/usr/bin/env python3
"""Analiza un manifiesto de clips o procesa eventos guardados por Noise Monitor."""

import argparse
import csv
import fcntl
import hashlib
import json
import logging
import sqlite3
import subprocess
import sys
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep, time

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.acoustic_analysis import ClipAnalyzer, validate_calibration  # noqa: E402
from src.analysis_evaluation import summarize  # noqa: E402
from src.analysis_store import (  # noqa: E402
    SCHEMA, UNIFIED, event_table, finish_attempt, pending_events, start_attempt, visual_context,
)
from src.utils import load_config, resolve_path  # noqa: E402

logger = logging.getLogger(__name__)


def write_reports(results: list[dict], csv_path: Path, summary: Path | None) -> None:
    keys = [
        "event_id", "clip_path", "clip_sha256", "duration_s", "status", "elapsed_ms",
        "laeq_dba", "lafmax_dba", "calibration_id", "sample_rate_hz", "codec",
        "loudness_n5_sone", "sharpness_acum", "roughness_asper", "tonality", "fluctuation_strength_vacil",
        "model_annoyance_score", "algorithm_version", "source_attribution", "vehicle_source",
        "audio_labels", "silence_score", "raw_yamnet_top_label", "acoustic_behaviors",
        "visual_labels", "visual_vehicle_confirmed",
        "parameters", "calibration", "human_review", "issues",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in results:
            fields = {**row, **row["audio"], **row["psychoacoustics"], **row["classification"]}
            writer.writerow({
                key: json.dumps(fields[key], ensure_ascii=False, allow_nan=False)
                if isinstance(fields.get(key), (dict, list)) else fields.get(key)
                for key in keys
            })
    if summary is not None:
        summary.write_text(json.dumps(summarize(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_offline(analyzer: ClipAnalyzer, args) -> None:
    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    results = []
    for entry in entries:
        path = (args.manifest.parent / entry["clip_path"]).resolve()
        result = analyzer.analyze(
            path, calibration=entry.get("calibration"), visual_labels=entry.get("visual_labels"),
            human_review=entry.get("human_review"), audio_muted=entry.get("audio_muted", False),
            visual_vehicle_confirmed=entry.get("visual_vehicle_confirmed", False),
        )
        result["event_id"] = entry["event_id"]
        results.append(result)
        logger.info("Evento %s %s en %.1f ms (%s)", entry["event_id"], result["status"],
                    result["elapsed_ms"], ", ".join(result["issues"]) or "sin incidencias")
    write_reports(results, args.csv, args.summary)


def site_calibration(options: dict) -> dict | None:
    calibration = validate_calibration(options.get("calibration"))
    if calibration is not None:
        for key in ("valid_from", "valid_until"):
            if not isinstance(calibration.get(key), str):
                raise ValueError(f"la calibración SQLite requiere {key} como fecha ISO local entre comillas")
            parse_local_time(calibration[key])
        if parse_local_time(calibration["valid_from"]) >= parse_local_time(calibration["valid_until"]):
            raise ValueError("la ventana de calibración debe tener duración positiva")
        if not calibration.get("codec"):
            raise ValueError("la calibración SQLite requiere el códec decodificado que se validó")
    return calibration


def parse_local_time(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is not None:
        raise ValueError("los timestamps de la instalación usan hora local sin offset")
    return result


def calibration_for(event, table: str, calibration: dict | None, pre: float, post: float) -> dict | None:
    if calibration is None:
        return None
    if table == UNIFIED:
        left = datetime.fromtimestamp(event["start_ts"])
        right = datetime.fromtimestamp(event["end_ts"])
    else:
        timestamp = parse_local_time(event["timestamp"])
        left = timestamp - timedelta(seconds=pre)
        right = timestamp + timedelta(seconds=post + 1)
    if parse_local_time(calibration["valid_from"]) <= left and right <= parse_local_time(calibration["valid_until"]):
        return calibration
    return None


def run_worker(analyzer: ClipAnalyzer, args, config: dict) -> None:
    database = resolve_path(config, "database", PROJECT_DIR)
    clips = resolve_path(config, "clips", PROJECT_DIR)
    calibration = site_calibration(config.get("acoustic_analysis") or {})
    buffer = config["buffer"]
    pre, post = float(buffer["pre_event_seconds"]), float(buffer["post_event_seconds"])
    since = parse_local_time(args.since).isoformat(timespec="seconds") if args.since else ""
    profile_config = {
        "name": args.profile, "algorithm": analyzer.parameters, "calibration": calibration,
        "legacy_pre_s": pre, "legacy_post_s": post, "settle_s": args.settle_seconds,
    }
    profile = hashlib.sha256(json.dumps(profile_config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    with database.with_suffix(database.suffix + ".analysis.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with closing(sqlite3.connect(f"{database.as_uri()}?mode=rw", uri=True, timeout=5)) as conn:
            conn.row_factory = sqlite3.Row
            table = event_table(conn, args.event_table)
            if table is None:
                raise ValueError("la base seleccionada no tiene un esquema de eventos reconocido")
            conn.executescript(SCHEMA)
            if args.retry_failed:
                with conn:
                    conn.execute(
                        """UPDATE acoustic_analysis SET attempts=0, retry_after=0
                           WHERE event_table=? AND profile=? AND status='failed'""",
                        (table, profile),
                    )
            logger.info("Procesando %s con perfil %s (%s)", table, args.profile, profile[:12])
            while True:
                cutoff = (datetime.now() - timedelta(seconds=args.settle_seconds)).isoformat(timespec="seconds")
                events = pending_events(conn, table, profile, since, cutoff, args.limit)
                for event in events:
                    job = start_attempt(conn, table, event, profile, args.profile)
                    try:
                        if table == UNIFIED:
                            origin = (
                                event["peak_ts"]
                                if "peak_ts" in event.keys() and event["peak_ts"] is not None
                                else event["end_ts"]
                            )
                        else:
                            origin = parse_local_time(event["timestamp"]).timestamp()
                        event_to_analysis_ms = max(0, round((time() - origin) * 1000))
                        stored = event["clip_path"] or ""
                        path = (PROJECT_DIR / stored).resolve()
                        muted = bool(event["audio_muted"])
                        if not muted and clips not in path.parents:
                            raise ValueError("clip fuera del directorio configurado")
                        visual, confirmed = visual_context(conn, table, event, pre, post)
                        result = analyzer.analyze(
                            path, calibration=calibration_for(event, table, calibration, pre, post),
                            visual_labels=visual, visual_vehicle_confirmed=confirmed, audio_muted=muted,
                        )
                        result["clip_path"] = stored
                        result["event_to_analysis_ms"] = event_to_analysis_ms
                    except (ValueError, OSError) as exc:
                        result = {"status": "failed", "issues": {"worker": str(exc)},
                                  "parameters": analyzer.parameters}
                    finish_attempt(conn, job, result)
                    logger.info("Evento %s %s (%s ms)", event["id"], result["status"], result.get("elapsed_ms"))
                if not args.watch:
                    return
                sleep(args.poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_DIR / "config.yaml")
    parser.add_argument("--model", type=Path, help="modelo TFLite de YAMNet")
    parser.add_argument("--labels", type=Path, help="lista ordenada de etiquetas AudioSet")
    parser.add_argument("--field", choices=("free", "diffuse"), default="free")
    parser.add_argument("--min-score", type=float, default=.3)
    parser.add_argument("--annoyance", action="store_true", help="calcular PA experimental en campo libre")
    commands = parser.add_subparsers(dest="command", required=True)
    offline = commands.add_parser("offline", help="analizar clips sin abrir SQLite")
    offline.add_argument("manifest", type=Path)
    offline.add_argument("--csv", type=Path, required=True)
    offline.add_argument("--summary", type=Path)
    worker = commands.add_parser("worker", help="procesar un lote desde SQLite")
    worker.add_argument("--event-table", choices=("noise_events", "events"), help="esquema de eventos explícito")
    worker.add_argument("--watch", action="store_true", help="seguir buscando eventos")
    worker.add_argument("--since", help="fecha ISO local mínima, para limitar reprocesamiento")
    worker.add_argument("--limit", type=int, default=4, help="máximo de eventos por lote")
    worker.add_argument("--poll-seconds", type=float, default=15)
    worker.add_argument("--settle-seconds", type=int, default=45, help="espera para consolidar pasajes y clips")
    worker.add_argument("--profile", default="baseline", help="nuevo nombre para forzar reprocesamiento")
    worker.add_argument("--retry-failed", action="store_true", help="reiniciar intentos fallidos del perfil")
    args = parser.parse_args()
    if args.command == "worker":
        if not (1 <= args.limit <= 100 and 0 < args.poll_seconds < float("inf")):
            parser.error("limit debe ser 1–100 y poll-seconds debe ser positivo y finito")
        if not 0 <= args.settle_seconds <= 86400:
            parser.error("settle-seconds debe ser 0–86400")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_config(args.config) if args.command == "worker" or args.config.exists() else {}
        model = config.get("audio_classifier") or {}
        analyzer = ClipAnalyzer(
            args.model or PROJECT_DIR / model.get("model_path", "models/yamnet.tflite"),
            args.labels or PROJECT_DIR / model.get("labels_path", "models/yamnet_labels.txt"),
            min_score=args.min_score, field=args.field, annoyance=args.annoyance,
        )
        if args.command == "offline":
            run_offline(analyzer, args)
        else:
            run_worker(analyzer, args, config)
    except BlockingIOError:
        logger.error("Ya hay un worker de análisis usando esta base")
        return 1
    except (KeyError, TypeError, ValueError, OSError, RuntimeError,
            sqlite3.Error, subprocess.SubprocessError) as exc:
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
