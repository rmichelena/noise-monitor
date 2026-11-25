"""
Cutover a la tabla única `events` (Slice D).

- schema.sql y el DDL de la migración no divergen (columnas, tipos, defaults,
  CHECK e índices).
- Una instalación limpia crea sólo `events` y las tablas auxiliares.
- Sobre una base migrada, init_database no recrea las tablas viejas.
- init_database se niega a arrancar sobre una base sin migrar, y main() sale
  con 78 (RestartPreventExitStatus) en vez de reiniciar en bucle.
"""

import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from migrate_unified import EVENTS_DDL_STATEMENTS, migrate  # noqa: E402
from support import create_legacy_schema, example_config  # noqa: E402
from src.utils import UnmigratedDatabaseError, init_database, resolve_path  # noqa: E402

AUX_TABLES = {"audio_levels", "acoustic_trigger_decisions", "traffic_counts"}


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _events_shape(conn: sqlite3.Connection) -> tuple:
    columns = [tuple(row[1:]) for row in conn.execute("PRAGMA table_info(events)")]
    indexes = sorted(
        (row[0], tuple(c[2] for c in conn.execute(f"PRAGMA index_info({row[0]})")))
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='events' "
            "AND name NOT LIKE 'sqlite_%'"
        )
    )
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='events'").fetchone()[0]
    check = sql[sql.index("CHECK"):sql.index(")", sql.index("CHECK")) + 2]
    return columns, indexes, check


class SchemaCutoverTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = example_config(self.tmp.name)
        self.db = resolve_path(self.cfg, "database", PROJECT_DIR)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db))
        self.addCleanup(conn.close)
        return conn

    def test_schema_sql_matches_migration_ddl(self):
        init_database(self.cfg, PROJECT_DIR)
        reference = sqlite3.connect(":memory:")
        self.addCleanup(reference.close)
        for statement in EVENTS_DDL_STATEMENTS:
            reference.execute(statement)
        self.assertEqual(_events_shape(self.connect()), _events_shape(reference))

    def test_clean_install_creates_only_events(self):
        init_database(self.cfg, PROJECT_DIR)
        tables = _tables(self.connect())
        self.assertEqual(tables, {"events"} | AUX_TABLES)

    def test_migrated_database_does_not_recreate_old_tables(self):
        create_legacy_schema(self.db)
        conn = self.connect()
        conn.execute(
            "INSERT INTO noise_events (timestamp, db_max) VALUES ('2026-11-20T12:00:00', 80)"
        )
        conn.commit()
        self.assertEqual(migrate(conn, 5, 5)["status"], "migrated")

        init_database(self.cfg, PROJECT_DIR)
        init_database(self.cfg, PROJECT_DIR)
        tables = _tables(conn)
        self.assertNotIn("noise_events", tables)
        self.assertNotIn("vehicle_passages", tables)
        self.assertIn("noise_events_legacy", tables)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)
        # Migración e init_database repetidos no cambian nada.
        self.assertEqual(migrate(conn, 5, 5)["status"], "already_migrated")
        init_database(self.cfg, PROJECT_DIR)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0], 1)

    def test_unmigrated_database_is_refused_before_creating_events(self):
        create_legacy_schema(self.db)
        with self.assertRaisesRegex(UnmigratedDatabaseError, "migrate_unified"):
            init_database(self.cfg, PROJECT_DIR)
        self.assertNotIn("events", _tables(self.connect()))

    def test_old_tables_next_to_events_are_refused(self):
        # El arranque en el orden equivocado ya creó `events`: igual hay que migrar.
        create_legacy_schema(self.db)
        conn = self.connect()
        for statement in EVENTS_DDL_STATEMENTS:
            conn.execute(statement)
        conn.commit()
        with self.assertRaisesRegex(UnmigratedDatabaseError, "sin migrar"):
            init_database(self.cfg, PROJECT_DIR)

    def test_legacy_without_events_is_refused(self):
        conn = self.connect()
        conn.execute("CREATE TABLE noise_events_legacy (id INTEGER PRIMARY KEY)")
        conn.commit()
        with self.assertRaisesRegex(UnmigratedDatabaseError, "falta events"):
            init_database(self.cfg, PROJECT_DIR)
        self.assertNotIn("events", _tables(conn))

    def test_events_with_other_shape_is_refused(self):
        # La `events` v1 (anterior a noise_events) no tiene kind/start_ts.
        conn = self.connect()
        conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY, timestamp TEXT, db_max REAL)")
        conn.commit()
        with self.assertRaisesRegex(UnmigratedDatabaseError, "otra forma"):
            init_database(self.cfg, PROJECT_DIR)


class MainExitCodeTests(unittest.TestCase):
    def test_unmigrated_database_exits_78_before_opening_devices(self):
        import main

        with tempfile.TemporaryDirectory() as tmp:
            cfg = example_config(tmp)
            create_legacy_schema(resolve_path(cfg, "database", PROJECT_DIR))
            with mock.patch.object(main, "load_config", return_value=cfg), \
                 mock.patch.object(main, "validate_config", side_effect=lambda c, **_: c), \
                 mock.patch.object(main.Settings, "load", return_value=None), \
                 mock.patch.object(main, "_load_audio_classifier", return_value=None), \
                 mock.patch.object(main, "AudioMonitor") as audio:
                self.assertEqual(main.main(), 78)
            audio.assert_not_called()


if __name__ == "__main__":
    unittest.main()
