#!/usr/bin/env python3
"""
Reporte de turno (por defecto la noche) del pipeline de ruido.

Lee events.db y produce un resumen en Markdown. Con --post lo publica en el
canal de Buzz usando la clave del agente (BUZZ_PRIVATE_KEY / systemd credential).

Uso:
  python scripts/nightly_report.py                 # imprime el reporte
  python scripts/nightly_report.py --post          # además lo publica
  python scripts/nightly_report.py --hours 10 --end-hour 7
"""

import argparse
import os
import sqlite3
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_DIR / "data" / "events.db"
DEFAULT_CHANNEL = "d113eafd-6fc3-4477-80ed-f08b3fc77770"  # #noise-monitor


def _fmt(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}"


def build_report(db: Path, start: datetime, end: datetime) -> str:
    if not db.exists():
        return f"**Reporte {start:%Y-%m-%d}**: base de datos no encontrada ({db})."

    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    lo, hi = start.isoformat(), end.isoformat()

    events = conn.execute(
        "SELECT * FROM noise_events WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
        (lo, hi),
    ).fetchall()
    passages = conn.execute(
        "SELECT * FROM vehicle_passages WHERE timestamp >= ? AND timestamp < ? ORDER BY timestamp",
        (lo, hi),
    ).fetchall()

    lines = [f"### Reporte de turno — {start:%d/%m %H:%M} a {end:%d/%m %H:%M} (Lima)", ""]

    if not events:
        lines.append(
            f"**Ruido:** 0 eventos registrados ({len(passages)} pasajes de vehículos)."
        )
    else:
        peak = max(events, key=lambda e: e["db_max"])
        avg = sum(e["db_max"] for e in events) / len(events)
        lines.append(
            f"**Ruido:** {len(events)} eventos | pico {_fmt(peak['db_max'])} dB(A) "
            f"a las {peak['timestamp'][11:16]} | media {avg:.1f} dB(A)"
        )
        cats = Counter(e["sound_category"] or "sin clasificar" for e in events)
        lines.append(
            "**Categorías:** "
            + " · ".join(f"{k} {v}" for k, v in cats.most_common())
        )
        veh = Counter(e["vehicle_class"] for e in events if e["vehicle_class"])
        if veh:
            lines.append(
                "**Eventos vehiculares:** "
                + " · ".join(f"{k} {v}" for k, v in veh.most_common())
            )
        top = sorted(events, key=lambda e: -(e["db_max"] or 0))[:5]
        lines.append("")
        lines.append("**Los más fuertes:**")
        for e in top:
            kind = e["vehicle_class"] or e["sound_class"] or "?"
            statue = e["trigger_type"] or ""
            clip = Path(e["clip_path"]).name if e["clip_path"] else "sin clip"
            lines.append(
                f"- {e['timestamp'][11:16]} · {_fmt(e['db_max'])} dB(A) · {kind} ({statue}) · `{clip}`"
            )

    if passages:
        by_class = Counter(p["vehicle_class"] for p in passages)
        dbs = [p["db_at_time"] for p in passages if p["db_at_time"] is not None]
        extra = f", {min(dbs):.0f}–{max(dbs):.0f} dB(A)" if dbs else ""
        lines.append("")
        lines.append(
            f"**Vehículos:** {len(passages)} pasajes — "
            + " · ".join(f"{k} {v}" for k, v in by_class.most_common())
            + extra
        )
        speeds = [p["speed_kmh"] for p in passages if p["speed_kmh"]]
        if speeds:
            lines.append(
                f"**Velocidad:** media {sum(speeds)/len(speeds):.0f} km/h · "
                f"máx {max(speeds):.0f} km/h ({len(speeds)} medidos)"
            )
        try:
            dirs = Counter(p["direction"] for p in passages if p["direction"])
        except (IndexError, KeyError):
            dirs = Counter()  # base anterior a la columna direction
        if dirs:
            label = {"ltr": "izq→der", "rtl": "der→izq"}
            lines.append(
                "**Sentido:** "
                + " · ".join(f"{label.get(k, k)} {v}" for k, v in dirs.most_common())
            )

    # Nivel de fondo (una fila por minuto, exista o no trigger)
    try:
        levels = conn.execute(
            "SELECT db_avg, db_max, db_p95 FROM audio_levels WHERE minute >= ? AND minute < ?",
            (lo[:16], hi[:16]),
        ).fetchall()
    except sqlite3.OperationalError:
        levels = []
    if levels:
        avgs = sorted(r["db_avg"] for r in levels)
        peaks = [r["db_max"] for r in levels]
        p50 = avgs[len(avgs) // 2]
        lines.append("")
        lines.append(
            f"**Fondo sonoro:** {len(levels)} min medidos · mediana {p50:.1f} dB(A) · "
            f"P95 de minuto {avgs[int(len(avgs) * 0.95) - 1]:.1f} · "
            f"pico absoluto {max(peaks):.1f} dB(A)"
        )

    # Distribución horaria
    hours = Counter(e["timestamp"][11:13] for e in events)
    if hours:
        lines.append("")
        lines.append("**Por hora:** " + " · ".join(f"{h}h:{n}" for h, n in sorted(hours.items())))

    # Salud del servicio dentro de la ventana
    try:
        out = subprocess.run(
            ["systemctl", "is-active", "noise-monitor"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        lines.append("")
        lines.append(f"Servicio `noise-monitor`: {out or 'desconocido'}")
    except Exception:
        pass

    conn.close()
    return "\n".join(lines)


def post(text: str, channel: str) -> None:
    cmd = [
        "/home/roberto/buzz/bin/buzz", "messages", "send",
        "--channel", channel, "--content", text,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"ERROR publicando en Buzz: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(result.returncode)
    print(result.stdout.strip() or "publicado")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reporte de turno del monitor de ruido")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--hours", type=float, default=10.0, help="ventana en horas (fin: ahora)")
    ap.add_argument("--end-hour", type=int, default=None,
                    help="hora local de fin (0-23); por defecto, ahora")
    ap.add_argument("--post", action="store_true", help="publicar en el canal de Buzz")
    ap.add_argument("--channel", default=DEFAULT_CHANNEL)
    args = ap.parse_args()

    now = datetime.now()
    if args.end_hour is not None:
        end = now.replace(hour=args.end_hour, minute=0, second=0, microsecond=0)
        if end > now:
            end -= timedelta(days=1)
    else:
        end = now
    start = end - timedelta(hours=args.hours)

    text = build_report(Path(args.db), start, end)
    print(text)
    if args.post:
        post(text, args.channel)


if __name__ == "__main__":
    main()
