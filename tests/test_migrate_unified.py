#!/usr/bin/env python3
"""
Migración a la tabla única `events` (scripts/migrate_unified.py).

- Un pasaje con varios ruidos vinculados produce UN solo `both` (el de mayor
  dB); el resto queda `noise` con su propio clip: un clip por fila.
- `clip_path = ''` es "sin clip": el capturador y la purga lo escriben así, y
  un COALESCE sin NULLIF elegía el vacío por sobre el clip real del pasaje.
- Las ventanas se calculan con la hora local del host. `strftime('%s')` de
  SQLite lee el ISO local como UTC y las corría horas.
- Cualquier invariante rota hace ROLLBACK: la base queda como estaba.
"""

import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import migrate_unified  # noqa: E402
from migrate_unified import MigrationError, migrate  # noqa: E402
from support import example_config  # noqa: E402
from src.utils import init_database, resolve_path  # noqa: E402

PRE, POST = 5.0, 10.0


def epoch(ts: str) -> float:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S").timestamp()


class MigrationFixture(unittest.TestCase):
    def setUp(self):
        # Zona fija distinta de UTC: en un host UTC, leer el ISO local como UTC
        # daría el mismo epoch y el test no detectaría el corrimiento.
        self._tz = os.environ.get("TZ")
        os.environ["TZ"] = "America/Lima"
        time.tzset()
        self.addCleanup(self._restore_tz)
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.cfg = example_config(self.base)
        self.cfg["buffer"]["pre_event_seconds"] = PRE
        self.cfg["buffer"]["post_event_seconds"] = POST
        self.db = resolve_path(self.cfg, "database", PROJECT_DIR)
        init_database(self.cfg, PROJECT_DIR)
        self.conn = sqlite3.connect(str(self.db))
        self.conn.row_factory = sqlite3.Row

    def _restore_tz(self):
        if self._tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = self._tz
        time.tzset()

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def passage(self, pid, ts="2026-11-20T10:00:00", exit_ts="2026-11-20T10:00:04",
                clip="clips/p.mp4", pinned=0, pinned_at=None, db=55.5, vehicle="car"):
        self.conn.execute(
            "INSERT INTO vehicle_passages (id, timestamp, timestamp_exit, vehicle_class, lane,"
            " direction, speed_kmh, detection_score, db_at_time, clip_path, pinned, pinned_at)"
            " VALUES (?, ?, ?, ?, 'left', 'ltr', 32.0, 0.8, ?, ?, ?, ?)",
            (pid, ts, exit_ts, vehicle, db, clip, pinned, pinned_at),
        )

    def noise(self, nid, ts="2026-11-20T10:00:01", db=70.0, passage_id=None,
              clip="clips/n.mp4", pinned=0, pinned_at=None, muted=0):
        self.conn.execute(
            "INSERT INTO noise_events (id, timestamp, db_max, db_avg, trigger_type, sound_class,"
            " sound_category, sound_confidence, sound_labels_json, vehicle_class, lane,"
            " confidence, passage_id, clip_path, audio_muted, pinned, pinned_at, duration_ms)"
            " VALUES (?, ?, ?, 60.0, 'peak', 'Vehicle', 'vehicle', 0.9, '[\"Vehicle\"]',"
            " 'truck', 'right', 'high', ?, ?, ?, ?, ?, 15000)",
            (nid, ts, db, passage_id, clip, muted, pinned, pinned_at),
        )

    def run_migration(self, **kwargs):
        self.conn.commit()
        return migrate(self.conn, PRE, POST, **kwargs)

    def event(self, eid):
        return self.conn.execute("SELECT * FROM events WHERE id = ?", (eid,)).fetchone()

    def table_exists(self, name):
        return self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None


class KindsTest(MigrationFixture):
    def test_noise_traffic_both(self):
        self.passage(1)                                   # vinculado → both
        self.passage(2, ts="2026-11-20T11:00:00", exit_ts=None)  # huérfano → traffic
        self.noise(10, passage_id=1)
        self.noise(11, ts="2026-11-20T12:00:00")          # sin pasaje → noise
        counts = self.run_migration()
        self.assertEqual(
            {k: counts[k] for k in ("both", "noise", "traffic", "total")},
            {"both": 1, "noise": 1, "traffic": 1, "total": 3},
        )

        both = self.event(10)
        self.assertEqual(both["kind"], "both")
        # Vehículo del pasaje, acústica del ruido.
        self.assertEqual(
            (both["vehicle_class"], both["lane"], both["direction"], both["speed_kmh"]),
            ("car", "left", "ltr", 32.0),
        )
        self.assertEqual((both["db_max"], both["sound_category"]), (70.0, "vehicle"))
        self.assertEqual(both["clip_path"], "clips/n.mp4")
        self.assertIsNone(both["track_id"])
        peak = epoch("2026-11-20T10:00:01")
        self.assertEqual(peak, 1795186801.0)  # 10:00:01 en Lima (UTC-5)
        self.assertEqual((both["peak_ts"], both["start_ts"], both["end_ts"]),
                         (peak, peak - PRE, peak + POST))

        noise = self.event(11)
        self.assertEqual(noise["kind"], "noise")
        self.assertIsNone(noise["direction"])
        self.assertEqual(noise["vehicle_class"], "truck")

        traffic = self.conn.execute("SELECT * FROM events WHERE kind='traffic'").fetchone()
        self.assertEqual(traffic["id"], 12)  # max(noise id) + 1
        self.assertEqual(traffic["db_max"], 55.5)  # db_at_time
        start = epoch("2026-11-20T11:00:00")
        self.assertEqual((traffic["start_ts"], traffic["end_ts"]), (start, start))
        self.assertIsNone(traffic["peak_ts"])

    def test_ventana_de_traffic_usa_la_salida(self):
        self.passage(1)
        self.run_migration()
        row = self.conn.execute("SELECT * FROM events WHERE kind='traffic'").fetchone()
        self.assertEqual((row["start_ts"], row["end_ts"]),
                         (epoch("2026-11-20T10:00:00"), epoch("2026-11-20T10:00:04")))

    def test_varios_ruidos_un_pasaje_dan_un_solo_both(self):
        self.passage(1)
        self.noise(10, db=65.0, passage_id=1, clip="clips/a.mp4")
        self.noise(11, db=72.0, passage_id=1, clip="clips/b.mp4")  # mayor dB → both
        self.noise(12, db=72.0, passage_id=1, clip="clips/c.mp4")  # empate, id mayor
        counts = self.run_migration()
        self.assertEqual((counts["both"], counts["noise"], counts["traffic"]), (1, 2, 0))
        self.assertEqual(self.event(11)["kind"], "both")
        for eid, clip in ((10, "clips/a.mp4"), (12, "clips/c.mp4")):
            row = self.event(eid)
            self.assertEqual((row["kind"], row["clip_path"], row["direction"]),
                             ("noise", clip, None))

    def test_passage_id_colgante_queda_noise(self):
        self.noise(10, passage_id=999)
        counts = self.run_migration()
        self.assertEqual((counts["noise"], counts["both"]), (1, 0))
        self.assertEqual(self.event(10)["kind"], "noise")


class ClipAndPinTest(MigrationFixture):
    def test_clip_vacio_cae_al_del_pasaje(self):
        self.passage(1, clip="clips/pasaje.mp4")
        self.noise(10, passage_id=1, clip="")
        self.run_migration()
        row = self.event(10)
        self.assertEqual(row["clip_path"], "clips/pasaje.mp4")
        self.assertIsNone(row["duration_ms"])  # no es la duración del clip de ruido

    def test_sin_clip_en_ninguno_queda_nulo(self):
        self.passage(1, clip="")
        self.noise(10, passage_id=1, clip="")
        self.noise(11, clip="")
        self.run_migration()
        self.assertIsNone(self.event(10)["clip_path"])
        self.assertIsNone(self.event(11)["clip_path"])

    def test_pinned_en_cualquiera_de_los_dos(self):
        self.passage(1, pinned=1, pinned_at="2026-11-21T09:00:00")
        self.noise(10, passage_id=1)
        self.passage(2)
        self.noise(11, passage_id=2, pinned=1, pinned_at="2026-11-21T08:00:00")
        self.passage(3, pinned=1, pinned_at="2026-11-21T07:00:00")
        self.noise(12, passage_id=3, pinned=1, pinned_at="2026-11-21T09:30:00")
        self.passage(4)
        self.noise(13, passage_id=4)
        self.run_migration()
        got = {eid: (self.event(eid)["pinned"], self.event(eid)["pinned_at"])
               for eid in (10, 11, 12, 13)}
        self.assertEqual(got, {
            10: (1, "2026-11-21T09:00:00"),
            11: (1, "2026-11-21T08:00:00"),
            12: (1, "2026-11-21T09:30:00"),
            13: (0, None),
        })

    def test_pinned_de_traffic(self):
        self.passage(1, pinned=1, pinned_at="2026-11-21T09:00:00")
        self.run_migration()
        row = self.conn.execute("SELECT * FROM events WHERE kind='traffic'").fetchone()
        self.assertEqual((row["pinned"], row["pinned_at"]), (1, "2026-11-21T09:00:00"))


class AnalysisRefsTest(MigrationFixture):
    def test_decision_acustica_pasa_a_events(self):
        self.noise(10)
        self.conn.execute(
            "INSERT INTO acoustic_trigger_decisions VALUES ('noise_events', 10, '{\"x\": 1}')"
        )
        counts = self.run_migration()
        self.assertEqual(counts["retagged"]["acoustic_trigger_decisions"], 1)
        self.assertEqual(
            self.conn.execute(
                "SELECT event_table, event_id FROM acoustic_trigger_decisions"
            ).fetchone()[:],
            ("events", 10),
        )

    def test_analisis_acustico_pasa_a_events(self):
        from src.analysis_store import SCHEMA
        self.conn.executescript(SCHEMA)
        self.noise(10)
        self.conn.execute(
            "INSERT INTO acoustic_analysis (event_table, event_id, clip_path, revision, profile,"
            " profile_name, status, updated_at) VALUES ('noise_events', 10, 'c', 'r', 'p', 'n',"
            " 'completed', 0)"
        )
        counts = self.run_migration()
        self.assertEqual(counts["retagged"]["acoustic_analysis"], 1)
        self.assertEqual(
            self.conn.execute("SELECT event_table FROM acoustic_analysis").fetchone()[0],
            "events",
        )

    def test_choque_con_event_table_events_aborta(self):
        self.noise(10)
        self.conn.execute(
            "INSERT INTO acoustic_trigger_decisions VALUES ('events', 10, '{}')"
        )
        with self.assertRaises(MigrationError):
            self.run_migration()
        self.assertTrue(self.table_exists("noise_events"))
        self.assertFalse(self.table_exists("events"))


class TransactionTest(MigrationFixture):
    def test_renombra_a_legacy_sin_borrar(self):
        self.passage(1)
        self.noise(10, passage_id=1)
        self.run_migration()
        self.assertFalse(self.table_exists("noise_events"))
        self.assertFalse(self.table_exists("vehicle_passages"))
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM noise_events_legacy").fetchone()[0], 1)
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM vehicle_passages_legacy").fetchone()[0], 1)
        # El esquema nuevo no lleva back-link: queda sólo en la tabla legacy.
        self.assertIn("noise_event_id", {
            r[1] for r in self.conn.execute("PRAGMA table_info(vehicle_passages_legacy)")})
        self.assertNotIn("noise_event_id", {
            r[1] for r in self.conn.execute("PRAGMA table_info(events)")})

    def test_segunda_corrida_no_hace_nada(self):
        self.noise(10)
        self.run_migration()
        self.assertEqual(self.run_migration(), {"status": "already_migrated"})
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)

    def test_events_con_filas_aborta_salvo_force(self):
        self.noise(10)
        self.conn.executescript(migrate_unified.EVENTS_DDL)
        self.conn.execute(
            "INSERT INTO events (id, kind, timestamp, start_ts, end_ts)"
            " VALUES (99, 'noise', '2026-11-20T10:00:00', 0, 0)"
        )
        with self.assertRaises(MigrationError):
            self.run_migration()
        self.assertTrue(self.table_exists("noise_events"))
        self.assertEqual(self.conn.execute("SELECT id FROM events").fetchall()[0][0], 99)

        counts = self.run_migration(force=True)
        self.assertEqual(counts["total"], 1)
        self.assertEqual([r[0] for r in self.conn.execute("SELECT id FROM events")], [10])

    def test_events_con_otra_forma_aborta(self):
        self.noise(10)
        self.conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, timestamp TEXT)")
        with self.assertRaises(MigrationError):
            self.run_migration()
        self.assertTrue(self.table_exists("noise_events"))

    def test_timestamp_ilegible_hace_rollback(self):
        self.noise(10)
        self.noise(11, ts="ayer")
        with self.assertRaises(MigrationError):
            self.run_migration()
        self.assertTrue(self.table_exists("noise_events"))
        self.assertFalse(self.table_exists("noise_events_legacy"))
        self.assertFalse(self.table_exists("events"))  # el DDL también vuelve atrás

    def test_invariante_rota_hace_rollback(self):
        self.passage(1)
        self.noise(10, passage_id=1)
        original = migrate_unified.build_rows

        def drop_one(conn, pre, post):
            rows, expected = original(conn, pre, post)
            return rows[:-1], expected

        migrate_unified.build_rows = drop_one
        try:
            with self.assertRaisesRegex(MigrationError, "invariantes"):
                self.run_migration()
        finally:
            migrate_unified.build_rows = original
        self.assertTrue(self.table_exists("noise_events"))
        self.assertFalse(self.table_exists("events"))  # el DDL también vuelve atrás

    def test_dry_run_no_persiste(self):
        self.passage(1)
        self.noise(10, passage_id=1)
        counts = self.run_migration(dry_run=True)
        self.assertEqual((counts["status"], counts["total"]), ("dry_run", 1))
        self.assertTrue(self.table_exists("noise_events"))
        self.assertFalse(self.table_exists("events"))


class CliTest(MigrationFixture):
    def cli(self, *args):
        cfg_path = self.base / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(self.cfg), encoding="utf-8")
        self.conn.commit()
        return subprocess.run(
            [sys.executable, str(PROJECT_DIR / "scripts" / "migrate_unified.py"),
             "--config", str(cfg_path), *args],
            cwd=str(PROJECT_DIR), capture_output=True, text=True, timeout=60,
        )

    def test_cli_imprime_conteos_y_falla_con_rc(self):
        self.passage(1)
        self.passage(2)
        self.noise(10, passage_id=1)
        self.noise(11)
        ok = self.cli()
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("migrated: both=1 noise=1 traffic=1 total=3", ok.stdout)

        again = self.cli()
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("ya migrada", again.stdout)

    def test_cli_aborta_con_rc_1(self):
        self.noise(10, ts="ayer")
        bad = self.cli()
        self.assertEqual(bad.returncode, 1)
        self.assertIn("ROLLBACK", bad.stderr)


if __name__ == "__main__":
    unittest.main()
