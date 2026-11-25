#!/usr/bin/env python3
"""
S5: la medición de tráfico (`traffic_counts`) no depende de los flags que
deciden qué se graba en `events`.

- Cada pasaje cerrado se cuenta por minuto, clase y sentido, con velocidad
  (n, suma, máximo), se guarde o no como evento.
- Al crearse la tabla se rellena una sola vez desde `events`; volver a
  inicializar la base no duplica.
- Dashboard, stats, reporte nocturno y purga leen/borran la tabla.
"""

import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

import nightly_report  # noqa: E402
from src.event_manager import EventManager  # noqa: E402
from src.utils import init_database  # noqa: E402
from support import create_events_table, example_config, insert_event  # noqa: E402
from web.app import create_app  # noqa: E402


def counts(db) -> list[tuple]:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            "SELECT minute, vehicle_class, direction, count, speed_n, speed_sum, speed_max"
            " FROM traffic_counts ORDER BY minute, vehicle_class, direction"
        ).fetchall()
    finally:
        conn.close()


class CountingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cfg = example_config(self.tmp.name)
        video = MagicMock()
        video.detections_between.return_value = []
        video.recent_detections = []
        video.video_buffer = []
        audio = MagicMock()
        audio.audio_buffer = []
        audio.current_db = 40.0
        self.em = EventManager(audio, video, config=cfg, base_dir=PROJECT_DIR)
        self.db = cfg["paths"]["database"]
        create_events_table(self.db)
        self.em._save_passage_clip = lambda passage: ("", 0, False)
        self.base = datetime(2026, 11, 25, 6, 54, 10).timestamp()

    def passage(self, offset=0.0, cls="bus", direction="ltr", speed=None):
        self.em._process_passage({
            "vehicle_class": cls, "lane": "left", "direction": direction,
            "speed_kmh": speed, "detection_score": 0.5, "track_id": 1,
            "timestamp_enter": self.base + offset, "timestamp_exit": self.base + offset + 3,
        })

    def events(self) -> int:
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        finally:
            conn.close()

    def test_se_cuenta_aunque_no_se_grabe(self):
        policy = self.em.trigger_policy
        for name, gated in (
            ("traffic_only_with_noise", replace(policy, traffic_only_with_noise=True)),
            ("traffic_enabled", replace(policy, traffic_enabled=False)),
            ("record_classes", replace(policy, record_classes=frozenset({"car"}))),
        ):
            with self.subTest(flag=name):
                self.em.trigger_policy = gated
                before = sum(row[3] for row in counts(self.db))
                self.passage()
                self.assertEqual(self.events(), 0, "no se graba")
                self.assertEqual(sum(row[3] for row in counts(self.db)), before + 1, "sí se cuenta")

    def test_agrega_por_minuto_clase_y_sentido_con_velocidad(self):
        self.passage(0, speed=30.0)
        self.passage(5, speed=50.0)
        self.passage(10)                      # sin velocidad
        self.passage(15, direction=None, speed=20.0)
        self.passage(20, cls="car", speed=40.0)
        self.passage(60, speed=10.0)          # minuto siguiente
        self.assertEqual(counts(self.db), [
            ("2026-11-25T06:54", "bus", "", 1, 1, 20.0, 20.0),
            ("2026-11-25T06:54", "bus", "ltr", 3, 2, 80.0, 50.0),
            ("2026-11-25T06:54", "car", "ltr", 1, 1, 40.0, 40.0),
            ("2026-11-25T06:55", "bus", "ltr", 1, 1, 10.0, 10.0),
        ])
        self.assertEqual(self.em.passages_counted, 6)

    def test_un_fallo_del_conteo_no_cuesta_el_evento(self):
        conn = sqlite3.connect(self.db)
        conn.execute("DROP TABLE traffic_counts")
        conn.commit()
        conn.close()
        with self.assertLogs("src.event_manager", level="ERROR"):
            self.passage()
        self.assertEqual(self.events(), 1)
        self.assertEqual(self.em.count_errors, 1)


class BackfillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = example_config(self.tmp.name)
        self.db = Path(self.cfg["paths"]["database"])
        # Base de la versión anterior: events con pasajes, sin traffic_counts.
        create_events_table(self.db)
        when = datetime(2026, 11, 24, 8, 0, 30)
        insert_event(self.db, "traffic", when, vehicle_class="car", direction="ltr", speed_kmh=30.0)
        insert_event(self.db, "both", when, vehicle_class="car", direction="ltr", speed_kmh=50.0)
        insert_event(self.db, "traffic", when, vehicle_class="truck")
        insert_event(self.db, "noise", when, vehicle_class="car", direction="ltr")
        insert_event(self.db, "traffic", when, vehicle_class=None)

    def test_rellena_una_sola_vez(self):
        expected = [
            ("2026-11-24T08:00", "car", "ltr", 2, 2, 80.0, 50.0),
            ("2026-11-24T08:00", "truck", "", 1, 0, 0.0, None),
        ]
        init_database(self.cfg, PROJECT_DIR)
        self.assertEqual(counts(self.db), expected, "noise y sin clase no cuentan")
        init_database(self.cfg, PROJECT_DIR)
        init_database(self.cfg, PROJECT_DIR)
        self.assertEqual(counts(self.db), expected, "reinicializar no duplica")

    def test_events_con_otra_forma_no_deja_la_tabla(self):
        conn = sqlite3.connect(self.db)
        conn.execute("ALTER TABLE events DROP COLUMN speed_kmh")
        conn.commit()
        conn.close()
        with self.assertRaises(Exception):
            init_database(self.cfg, PROJECT_DIR)
        conn = sqlite3.connect(self.db)
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'traffic_counts'"
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNone(exists, "el rollback deshace el CREATE: se rellena al arreglarla")


class ReadersTest(unittest.TestCase):
    """Sin ninguna fila de tráfico en events, todo sale de traffic_counts."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.cfg = example_config(self.base)
        self.db = Path(self.cfg["paths"]["database"])
        init_database(self.cfg, PROJECT_DIR)
        self.now = datetime.now().replace(second=0, microsecond=0)
        rows = [
            (self.now - timedelta(minutes=1), "car", "ltr", 3, 2, 70.0, 40.0),
            (self.now - timedelta(minutes=1), "bus", "rtl", 1, 1, 25.0, 25.0),
            (self.now - timedelta(minutes=3), "motorcycle", "", 2, 1, 10.0, 10.0),
            (self.now - timedelta(days=3), "car", "ltr", 9, 9, 900.0, 120.0),
        ]
        conn = sqlite3.connect(self.db)
        conn.executemany(
            "INSERT INTO traffic_counts VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(t.strftime("%Y-%m-%dT%H:%M"), *rest) for t, *rest in rows],
        )
        conn.commit()
        conn.close()

    def test_dashboard_y_stats(self):
        if self.now.hour == 0 and self.now.minute < 4:
            self.skipTest("las filas de hoy caerían ayer")
        client = TestClient(create_app(self.cfg, PROJECT_DIR,
                                       audio_monitor=SimpleNamespace(current_db=40.0)))
        self.assertEqual(client.get("/api/live").json()["passages_today"], 6)
        passages = client.get("/api/stats").json()["passages"]
        self.assertEqual(passages["by_class"], {"car": 3, "bus": 1, "motorcycle": 2})
        self.assertEqual(passages["by_dir"], {"ltr": 3, "rtl": 1})
        # Media ponderada por pasajes medidos (105 / 4), no por filas (3).
        self.assertEqual(passages["speed"], {"avg": 26.2, "max": 40.0, "n": 4})
        page = client.get("/partials/traffic").text
        self.assertIn(">6<", page.replace(" ", "").replace("\n", ""))

    def test_grafico_de_vehiculos(self):
        app = create_app(self.cfg, PROJECT_DIR, audio_monitor=SimpleNamespace(current_db=40.0))
        html = TestClient(app).get("/partials/vehicle-chart").text
        one = (self.now - timedelta(minutes=1)).strftime("%H:%M")
        three = (self.now - timedelta(minutes=3)).strftime("%H:%M")
        self.assertIn(f"<title>{one} · 4 vehículos</title>", html)
        self.assertIn(f"<title>{three} · 2 vehículos</title>", html)
        self.assertEqual(html.count("<title>"), 2, "la fila de hace 3 días queda fuera de las 2 h")

    def test_reporte_nocturno(self):
        text = nightly_report.build_report(self.db, self.now - timedelta(hours=1), self.now)
        self.assertIn("**Vehículos:** 6 pasajes", text)
        self.assertIn("**Velocidad:** media 26 km/h · máx 40 km/h (4 medidos)", text)
        self.assertIn("**Ruido:** 0 eventos registrados (6 pasajes de vehículos).", text)

    def test_purga_con_levels_days(self):
        cfg_path = self.base / "purge.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "paths": {"database": str(self.db), "clips": str(self.base / "clips")},
            "retention": {"clip_days": 30, "levels_days": 2},
        }), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(PROJECT_DIR / "scripts" / "purge.py"), "--config", str(cfg_path)],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("conteos de tráfico borrados=1", result.stdout)
        self.assertEqual(sum(row[3] for row in counts(self.db)), 6)


if __name__ == "__main__":
    unittest.main()
