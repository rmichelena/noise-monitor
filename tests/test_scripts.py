#!/usr/bin/env python3
"""
Scripts operativos: argumentos validados, purga de pasajes y P95 correcto.

- `--clip-days 0` (o negativo) dejaba el corte en "ahora" y borraba TODOS los
  clips; `--limit 0` imprimía la cabecera sin filas y `--limit -1` (que SQLite
  lee como "sin límite") volcaba la tabla entera.
- La purga borraba el archivo de un clip de pasaje pero dejaba su `clip_path`:
  el dashboard ofrecía "▶ ver" para un archivo ya borrado.
- El P95 de minuto con dos minutos medidos devolvía el mínimo.
"""

import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

import nightly_report  # noqa: E402
from support import create_events_table, insert_event  # noqa: E402
from src.utils import init_database  # noqa: E402


def run(script: str, *args: str, cwd: Path = PROJECT_DIR) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PROJECT_DIR / "scripts" / script), *args],
        cwd=str(cwd), capture_output=True, text=True, timeout=120,
    )


class ScriptFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.clips = self.base / "clips"
        self.clips.mkdir()
        self.db = self.base / "events.db"
        self.cfg_path = self.base / "config.yaml"
        cfg = {
            "paths": {"database": str(self.db), "clips": str(self.clips)},
            "retention": {"clip_days": 30, "levels_days": 400},
        }
        self.cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        self.cfg = cfg
        init_database(cfg, PROJECT_DIR)
        create_events_table(self.db)

    def tearDown(self):
        self.tmp.cleanup()

    def conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db))
        conn.row_factory = sqlite3.Row
        return conn


class PurgeTest(ScriptFixture):
    def _seed(self):
        # Clip de evento inexistente, evento fijado (conserva su clip) y un
        # pasaje cuyo clip ya no está en disco.
        missing = self.base / "clips" / "event_missing.mp4"
        pinned = self.clips / "event_pinned.mp4"
        pinned.write_bytes(b"x" * 512)
        old = time.time() - 40 * 86400
        pinned.touch()
        import os
        os.utime(pinned, (old, old))

        passage_missing = self.clips / "passage_missing.mp4"
        passage_fresh = self.clips / "passage_fresh.mp4"
        passage_fresh.write_bytes(b"y" * 128)

        now = datetime.now()
        insert_event(self.db, "noise", now, db_max=70.0, trigger_type="peak",
                     clip_path=str(missing), pinned=0)
        insert_event(self.db, "noise", now, db_max=80.0, trigger_type="peak",
                     clip_path=str(pinned), pinned=1)
        insert_event(self.db, "traffic", now, vehicle_class="car",
                     clip_path=str(passage_missing), pinned=0)
        insert_event(self.db, "traffic", now, vehicle_class="truck",
                     clip_path=str(passage_fresh), pinned=0)
        return pinned, passage_fresh

    def test_purge_clears_event_and_passage_rows(self):
        pinned, passage_fresh = self._seed()
        result = run("purge.py", "--config", str(self.cfg_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("noise=1 traffic=1 both=0", result.stdout)

        conn = self.conn()
        events = {r["pinned"]: r["clip_path"] for r in conn.execute(
            "SELECT pinned, clip_path FROM events WHERE kind = 'noise'"
        )}
        passages = {r["vehicle_class"]: r["clip_path"] for r in conn.execute(
            "SELECT vehicle_class, clip_path FROM events WHERE kind = 'traffic'"
        )}
        conn.close()

        self.assertIsNone(events[0])              # faltaba el archivo
        self.assertEqual(events[1], str(pinned))  # fijado: se conserva
        self.assertIsNone(passages["car"])        # faltaba el archivo
        self.assertEqual(passages["truck"], str(passage_fresh))

    def test_dry_run_changes_nothing(self):
        pinned, passage_fresh = self._seed()
        result = run("purge.py", "--config", str(self.cfg_path), "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY-RUN", result.stdout)

        conn = self.conn()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM events WHERE clip_path IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(remaining, 4)
        self.assertIn("noise=1 traffic=1 both=0", result.stdout)
        self.assertTrue(pinned.exists())

    def test_zero_days_is_refused(self):
        self._seed()
        result = run("purge.py", "--config", str(self.cfg_path), "--clip-days", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn(">= 1", result.stderr)
        conn = self.conn()
        kept = conn.execute(
            "SELECT COUNT(*) FROM events WHERE clip_path IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        self.assertEqual(kept, 4)

    def test_negative_days_is_refused(self):
        result = run("purge.py", "--config", str(self.cfg_path), "--levels-days", "-5")
        self.assertEqual(result.returncode, 2)
        self.assertIn("levels-days", result.stderr)


class StatsScriptTest(ScriptFixture):
    def _seed(self):
        now = datetime.now()
        insert_event(self.db, "noise", now, db_max=float("-inf"), trigger_type="peak")
        insert_event(self.db, "traffic", now, vehicle_class="car")
        insert_event(self.db, "both", now, db_max=72.0, trigger_type="peak",
                     vehicle_class="truck", sound_category="vehicle")

    def test_null_columns_do_not_crash(self):
        self._seed()
        result = run("stats.py", "--db", str(self.db), "--limit", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("VEHÍCULOS DETECTADOS", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_counts_by_kind(self):
        self._seed()
        result = run("stats.py", "--db", str(self.db), "--limit", "5")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("EVENTOS: 3 | ruido 1 · tránsito 1 · ambos 1", result.stdout)
        # Un `both` es pasaje y es ruido a la vez.
        self.assertIn("VEHÍCULOS DETECTADOS (tracker): 2 pasajes", result.stdout)
        self.assertIn("EVENTOS DE RUIDO: 2 total", result.stdout)
        self.assertIn("Pasajes con evento de ruido asociado: 1 / 2", result.stdout)

    def test_missing_events_table_fails(self):
        with self.conn() as conn:
            conn.execute("DROP TABLE events")
        result = run("stats.py", "--db", str(self.db))
        self.assertEqual(result.returncode, 1)
        self.assertIn("migrate_unified", result.stdout)

    def test_limit_zero_is_refused(self):
        self._seed()
        result = run("stats.py", "--db", str(self.db), "--limit", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn(">= 1", result.stderr)

    def test_negative_days_is_refused(self):
        self._seed()
        result = run("stats.py", "--db", str(self.db), "--days", "-3")
        self.assertEqual(result.returncode, 2)


class NightlyReportTest(ScriptFixture):
    def test_percentile_95_with_two_minutes_is_the_maximum(self):
        conn = self.conn()
        now = datetime.now().replace(second=0, microsecond=0)
        for minutes_ago, avg, peak in ((10, 40.0, 45.0), (20, 68.0, 70.0)):
            minute = (now - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M")
            conn.execute(
                "INSERT INTO audio_levels (minute, db_avg, db_max, db_p95, blocks)"
                " VALUES (?, ?, ?, ?, 600)",
                (minute, avg, peak, avg),
            )
        conn.commit()
        conn.close()

        text = nightly_report.build_report(
            self.db, now - timedelta(hours=2), now
        )
        self.assertIn("P95 de minuto 68.0", text)
        self.assertIn("mediana", text)

    def test_report_counts_both_as_noise_and_passage(self):
        now = datetime.now().replace(second=0, microsecond=0)
        when = now - timedelta(minutes=30)
        insert_event(self.db, "noise", when, db_max=65.0, trigger_type="peak")
        insert_event(self.db, "traffic", when, db_max=50.0, vehicle_class="car")
        insert_event(self.db, "both", when, db_max=78.0, trigger_type="peak",
                     vehicle_class="truck", direction="ltr")
        text = nightly_report.build_report(self.db, now - timedelta(hours=2), now)
        self.assertIn("ruido 1 · tránsito 1 · ambos 1", text)
        self.assertIn("**Ruido:** 2 eventos", text)
        self.assertIn("**Vehículos:** 2 pasajes", text)
        self.assertIn("· both ·", text)

    def test_hours_must_be_positive(self):
        result = run("nightly_report.py", "--db", str(self.db), "--hours", "0")
        self.assertEqual(result.returncode, 2)
        self.assertIn("> 0", result.stderr)

    def test_end_hour_range(self):
        result = run("nightly_report.py", "--db", str(self.db), "--end-hour", "24")
        self.assertEqual(result.returncode, 2)
        self.assertIn("0 y 23", result.stderr)


if __name__ == "__main__":
    unittest.main()
