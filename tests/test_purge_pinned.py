#!/usr/bin/env python3
"""
Regresión: la purga borra clips viejos salvo los de eventos fijados (pinned),
que son evidencia marcada por el dueño.

Tras la migración a `events`, un `both` apunta a un solo clip; el otro archivo
sólo lo nombra la tabla legacy y su fijado también se respeta.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import create_events_table, insert_event  # noqa: E402
from src.utils import init_database  # noqa: E402


class PurgePinnedTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.clips = base / "clips"
        self.clips.mkdir()
        self.db = base / "events.db"

        self.pinned_clip = self.clips / "event_pinned.mp4"
        self.old_clip = self.clips / "event_old.mp4"
        self.pinned_passage = self.clips / "passage_pinned.mp4"
        self.legacy_pinned = self.clips / "passage_legacy_pinned.mp4"
        for f in (self.pinned_clip, self.old_clip, self.pinned_passage, self.legacy_pinned):
            f.write_bytes(b"x" * 1024)
            old = time.time() - 40 * 86400
            os.utime(f, (old, old))

        cfg = {
            "paths": {"database": str(self.db), "clips": str(self.clips),
                      "reports": str(base)},
            "retention": {"clip_days": 30, "levels_days": 400},
        }
        self.cfg_path = base / "config.yaml"
        self.cfg_path.write_text(yaml.safe_dump(cfg))
        init_database(cfg, PROJECT_DIR)

        create_events_table(self.db)
        when = datetime(2020, 1, 1)
        self.pinned_id = insert_event(
            self.db, "noise", when, db_max=55, clip_path=str(self.pinned_clip), pinned=1
        )
        self.old_id = insert_event(
            self.db, "noise", when, db_max=55, clip_path=str(self.old_clip), pinned=0
        )
        self.passage_id = insert_event(
            self.db, "traffic", when, vehicle_class="car",
            clip_path=str(self.pinned_passage), pinned=1,
        )
        # El clip del pasaje que la migración no eligió para el `both`.
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE vehicle_passages_legacy (id INTEGER, clip_path TEXT, pinned INTEGER)")
        conn.execute(
            "INSERT INTO vehicle_passages_legacy VALUES (1, ?, 1)", (str(self.legacy_pinned),)
        )
        conn.commit()
        conn.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_purga_respeta_fijados(self):
        subprocess.run(
            [sys.executable, str(PROJECT_DIR / "scripts/purge.py"),
             "--config", str(self.cfg_path)],
            check=True, capture_output=True,
        )
        self.assertTrue(self.pinned_clip.is_file(), "borró el clip fijado")
        self.assertFalse(self.old_clip.is_file(), "no borró el clip viejo")
        self.assertTrue(self.pinned_passage.is_file(), "borró el clip de pasaje fijado")
        self.assertTrue(self.legacy_pinned.is_file(), "borró el clip fijado en legacy")
        conn = sqlite3.connect(self.db)
        rows = dict(conn.execute("SELECT id, clip_path FROM events").fetchall())
        conn.close()
        self.assertEqual(rows[self.pinned_id], str(self.pinned_clip))
        self.assertEqual(rows[self.passage_id], str(self.pinned_passage))
        self.assertIsNone(rows[self.old_id])

    def test_fila_no_fijada_conserva_clip_fijado_en_legacy(self):
        conn = sqlite3.connect(self.db)
        conn.execute(
            "UPDATE events SET clip_path=? WHERE id=?", (str(self.legacy_pinned), self.old_id)
        )
        conn.commit()
        conn.close()
        subprocess.run(
            [sys.executable, str(PROJECT_DIR / "scripts/purge.py"),
             "--config", str(self.cfg_path)],
            check=True, capture_output=True,
        )
        self.assertTrue(self.legacy_pinned.is_file())
        conn = sqlite3.connect(self.db)
        stored = conn.execute("SELECT clip_path FROM events WHERE id=?", (self.old_id,)).fetchone()[0]
        conn.close()
        self.assertEqual(stored, str(self.legacy_pinned))

    def test_sin_tabla_events_no_borra_nada(self):
        conn = sqlite3.connect(self.db)
        conn.execute("DROP TABLE events")
        conn.commit()
        conn.close()
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "scripts/purge.py"),
             "--config", str(self.cfg_path)],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("migrate_unified", result.stderr)
        self.assertTrue(self.old_clip.is_file())


if __name__ == "__main__":
    unittest.main()
