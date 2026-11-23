#!/usr/bin/env python3
"""
Dashboard/API: JSON válido con silencio, filtros que no mienten y CSV.

- Un `-inf` guardado por un bloque de silencio salía como `-Infinity`, que no
  es JSON válido: el panel en vivo se quedaba congelado en el navegador.
- `?vehicle=cards` (errata) devolvía 200 con lista vacía, indistinguible de un
  día sin tráfico.
- La exportación CSV es la evidencia que se presenta con un reclamo.
"""

import csv
import io
import json
import math
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient  # noqa: E402

from support import example_config  # noqa: E402
from src.utils import init_database, resolve_path  # noqa: E402
from web.app import create_app  # noqa: E402


def iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


class WebTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.cfg = example_config(self.base)
        self.db_path = resolve_path(self.cfg, "database", PROJECT_DIR)
        self.clips = resolve_path(self.cfg, "clips", PROJECT_DIR)
        init_database(self.cfg, PROJECT_DIR)

        self.now = datetime.now().replace(microsecond=0)
        self._insert_events()
        self._insert_passages()

        self.audio = SimpleNamespace(current_db=float("-inf"))
        app = create_app(self.cfg, PROJECT_DIR, audio_monitor=self.audio, video_monitor=None)
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp.cleanup()

    def _insert_events(self):
        conn = sqlite3.connect(str(self.db_path))
        rows = [
            # (timestamp, db_max, category, class, vehicle, clip_path, muted)
            (iso(self.now), 71.2, "vehicle", "Truck", "truck", "", 0),
            (iso(self.now - timedelta(minutes=5)), float("-inf"), "other", None, None, "", 0),
            (iso(self.now - timedelta(days=10)), 68.0, "animal", "Dog", None, "", 0),
        ]
        for ts, db, category, sound_class, vehicle, clip, muted in rows:
            conn.execute(
                "INSERT INTO noise_events (timestamp, db_max, db_avg, trigger_type,"
                " sound_class, sound_category, vehicle_class, confidence, clip_path,"
                " audio_muted, duration_ms) VALUES (?, ?, ?, 'peak', ?, ?, ?, 'high', ?, ?, 900)",
                (ts, db, db, sound_class, category, vehicle, clip, muted),
            )
        conn.commit()
        conn.close()

    def _insert_passages(self):
        self.inside = self.clips / "passage_inside.mp4"
        self.inside.write_bytes(b"\x00" * 64)
        outside = self.base / "fuera.mp4"
        outside.write_bytes(b"\x00" * 32)

        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            "INSERT INTO vehicle_passages (timestamp, vehicle_class, lane, direction,"
            " speed_kmh, detection_score, db_at_time, clip_path, pinned)"
            " VALUES (?, 'car', 'left', 'ltr', 48.0, 0.71, 55.4, ?, 0)",
            (iso(self.now), str(self.inside)),
        )
        conn.execute(
            "INSERT INTO vehicle_passages (timestamp, vehicle_class, lane, direction,"
            " speed_kmh, detection_score, db_at_time, clip_path, pinned)"
            " VALUES (?, 'truck', 'right', 'rtl', NULL, NULL, NULL, ?, 0)",
            (iso(self.now - timedelta(days=10)), str(outside)),
        )
        conn.commit()
        conn.close()


class LiveApiTest(WebTestCase):
    def test_health_distinguishes_requested_backend_from_fallback(self):
        status = {"requested_backend": "coral", "active_backend": "cpu", "fallbacks": 1}
        self.audio.classifier_status = lambda: status
        health = self.client.get("/api/health").json()
        self.assertEqual(health["audio_classifier"], status)

    def test_live_returns_parseable_json_with_silence(self):
        response = self.client.get("/api/live")
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.text)  # -Infinity rompería aquí
        self.assertIsNone(payload["db"])
        self.assertEqual(payload["max_db_today"], 71.2)

    def test_live_max_db_is_null_when_only_silence_was_recorded(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM noise_events WHERE db_max > 0")
        conn.commit()
        conn.close()
        payload = json.loads(self.client.get("/api/live").text)
        self.assertIsNone(payload["max_db_today"])
        self.assertEqual(payload["events_today"], 1)

    def test_live_reports_finite_db(self):
        self.audio.current_db = 61.24
        payload = self.client.get("/api/live").json()
        self.assertEqual(payload["db"], 61.2)

    def test_counters_count_only_today(self):
        payload = self.client.get("/api/live").json()
        self.assertEqual(payload["events_today"], 2)
        self.assertEqual(payload["passages_today"], 1)

    def test_events_json_keeps_finite_values(self):
        payload = self.client.get("/api/events?limit=10").json()
        dbs = [e["db_max"] for e in payload["events"]]
        self.assertIn(71.2, dbs)
        self.assertTrue(all(math.isfinite(v) for v in dbs if v is not None))


class FilterValidationTest(WebTestCase):
    def test_unknown_vehicle_is_rejected(self):
        response = self.client.get("/api/events?vehicle=cards")
        self.assertEqual(response.status_code, 400)
        self.assertIn("cards", response.json()["detail"])
        self.assertIn("truck", response.json()["detail"])

    def test_unknown_category_is_rejected(self):
        response = self.client.get("/api/events?category=ruido")
        self.assertEqual(response.status_code, 400)

    def test_unknown_direction_is_rejected(self):
        self.assertEqual(self.client.get("/api/events?direction=up").status_code, 400)
        self.assertEqual(
            self.client.get("/partials/passages?direction=up").status_code, 400
        )

    def test_valid_filters_still_work(self):
        payload = self.client.get("/api/events?vehicle=truck").json()
        self.assertEqual(len(payload["events"]), 1)
        self.assertEqual(payload["events"][0]["vehicle_class"], "truck")

    def test_bad_limit_is_a_validation_error(self):
        self.assertEqual(self.client.get("/api/events?limit=0").status_code, 422)


class ExportCsvTest(WebTestCase):
    def test_events_csv_has_header_and_rows(self):
        response = self.client.get("/api/export/events.csv?days=1")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers["content-type"])
        self.assertIn("attachment", response.headers["content-disposition"])

        rows = list(csv.DictReader(io.StringIO(response.text)))
        self.assertEqual(len(rows), 2)  # la fila de hace 10 días queda fuera
        self.assertEqual(rows[0]["trigger_type"], "peak")
        self.assertTrue(rows[0]["timestamp"].startswith(self.now.strftime("%Y-%m-%d")))

    def test_events_csv_respects_min_db(self):
        response = self.client.get("/api/export/events.csv?days=1&min_db=70")
        rows = list(csv.DictReader(io.StringIO(response.text)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["db_max"], "71.2")

    def test_events_csv_rejects_unknown_vehicle(self):
        self.assertEqual(
            self.client.get("/api/export/events.csv?vehicle=ovni").status_code, 400
        )

    def test_events_csv_bad_days(self):
        self.assertEqual(self.client.get("/api/export/events.csv?days=0").status_code, 422)

    def test_passages_csv_rows(self):
        response = self.client.get("/api/export/passages.csv?days=1")
        rows = list(csv.DictReader(io.StringIO(response.text)))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["vehicle_class"], "car")
        self.assertEqual(rows[0]["speed_kmh"], "48.0")
        self.assertEqual(rows[0]["direction"], "ltr")

    def test_events_csv_direction_filter_and_column(self):
        conn = sqlite3.connect(str(self.db_path))
        car = conn.execute("SELECT id FROM vehicle_passages WHERE vehicle_class='car'").fetchone()[0]
        conn.execute("UPDATE noise_events SET passage_id=? WHERE vehicle_class='truck'", (car,))
        conn.commit()
        conn.close()
        rows = list(csv.DictReader(io.StringIO(
            self.client.get("/api/export/events.csv?days=1&direction=ltr").text
        )))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["direction"], "ltr")
        self.assertIn("direction", rows[0])

    def test_events_csv_truck_without_passage_has_empty_direction(self):
        rows = list(csv.DictReader(io.StringIO(
            self.client.get("/api/export/events.csv?days=1&vehicle=truck").text
        )))
        self.assertEqual(rows[0]["direction"], "")

    def test_export_returns_500_if_db_is_gone(self):
        self.db_path.unlink()
        response = self.client.get("/api/export/events.csv?days=1")
        self.assertEqual(response.status_code, 500)
        self.assertIn("exportar", response.json()["detail"])

    def test_passages_csv_filters(self):
        self.assertEqual(
            len(list(csv.DictReader(io.StringIO(
                self.client.get("/api/export/passages.csv?days=1&min_speed=60").text
            )))),
            0,
        )


class DashboardHtmlTest(WebTestCase):
    def test_event_label_candidates_are_available_in_api_and_dashboard(self):
        ranked = [
            {"label": "Truck", "category": "vehicle", "score": .81, "passes_threshold": True},
            {"label": "Air horn, truck horn", "category": "vehicle", "score": .44, "passes_threshold": True},
            {"label": "Engine", "category": "vehicle", "score": .23, "passes_threshold": False},
        ]
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE noise_events SET sound_labels_json=? WHERE sound_class='Truck'", (json.dumps(ranked),))
        events = self.client.get("/api/events").json()["events"]
        self.assertEqual(next(row["sound_labels"] for row in events if row["sound_class"] == "Truck"), ranked)
        self.assertEqual(next(row["sound_labels"] for row in events if row["sound_class"] == "Dog"), [])
        page = self.client.get("/partials/events").text
        self.assertIn("Air horn, truck horn", page)
        self.assertIn("por debajo del umbral", page)

    def test_dashboard_renders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Noise Monitor", response.text)

    def test_partials_render(self):
        for path in ("/partials/live", "/partials/events", "/partials/spark",
                     "/partials/filters", "/partials/traffic",
                     "/partials/vehicle-chart", "/partials/passages"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_silence_row_renders_as_dash(self):
        text = self.client.get("/partials/events").text
        self.assertIn("71.2", text)
        self.assertNotIn("-inf", text.lower())
        self.assertIn("—", text)

    def test_spark_ignores_non_finite_levels(self):
        conn = sqlite3.connect(str(self.db_path))
        minute = self.now.strftime("%Y-%m-%dT%H:%M")
        conn.execute(
            "INSERT INTO audio_levels (minute, db_avg, db_max, db_p95, blocks)"
            " VALUES (?, ?, ?, ?, 600)",
            (minute, float("-inf"), float("-inf"), float("-inf")),
        )
        conn.commit()
        conn.close()
        response = self.client.get("/partials/spark")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("nan", response.text.lower())


class ClipServingTest(WebTestCase):
    def test_clip_inside_the_folder_is_served(self):
        event_id = self._passage_id(self.inside)
        response = self.client.get(f"/api/passage-clips/{event_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "video/mp4")

    def test_clip_outside_the_folder_is_refused(self):
        outside = self.base / "fuera.mp4"
        event_id = self._passage_id(outside)
        self.assertEqual(self.client.get(f"/api/passage-clips/{event_id}").status_code, 404)

    def test_missing_event_clip(self):
        self.assertEqual(self.client.get("/api/clips/99999").status_code, 404)

    def _passage_id(self, path: Path) -> int:
        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute(
                "SELECT id FROM vehicle_passages WHERE clip_path = ?", (str(path),)
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row, f"sin fila para {path}")
        return int(row[0])


if __name__ == "__main__":
    unittest.main()
