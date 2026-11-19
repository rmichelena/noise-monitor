#!/usr/bin/env python3
"""
Regresión: la purga borra clips viejos salvo los de eventos fijados (pinned),
que son evidencia marcada por el dueño.
"""

import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

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
        for f in (self.pinned_clip, self.old_clip, self.pinned_passage):
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

        conn = sqlite3.connect(self.db)
        conn.execute(
            "INSERT INTO noise_events (timestamp, db_max, clip_path, pinned)"
            " VALUES ('2020-01-01T00:00:00', 55, ?, 1)", (str(self.pinned_clip),)
        )
        conn.execute(
            "INSERT INTO noise_events (timestamp, db_max, clip_path, pinned)"
            " VALUES ('2020-01-01T00:00:01', 55, ?, 0)", (str(self.old_clip),)
        )
        conn.execute(
            "INSERT INTO vehicle_passages (timestamp, vehicle_class, clip_path, pinned)"
            " VALUES ('2020-01-01T00:00:02', 'car', ?, 1)", (str(self.pinned_passage),)
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
        conn = sqlite3.connect(self.db)
        rows = dict(conn.execute("SELECT pinned, clip_path FROM noise_events").fetchall())
        conn.close()
        self.assertEqual(rows[1], str(self.pinned_clip))
        self.assertEqual(rows[0], "")


if __name__ == "__main__":
    unittest.main()
