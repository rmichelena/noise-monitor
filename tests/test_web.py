#!/usr/bin/env python3
"""
Dashboard/API: JSON válido con silencio, filtros que no mienten y CSV.

- Un `-inf` guardado por un bloque de silencio salía como `-Infinity`, que no
  es JSON válido: el panel en vivo se quedaba congelado en el navegador.
- `?vehicle=cards` (errata) devolvía 200 con lista vacía, indistinguible de un
  día sin tráfico.
- La exportación CSV es la evidencia que se presenta con un reclamo.
- Todo se lee de la tabla única `events` (noise | traffic | both).
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

from support import create_events_table, example_config, insert_event  # noqa: E402
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
        create_events_table(self.db_path)

        self.now = datetime.now().replace(microsecond=0)
        self._insert_events()
        self._insert_passages()

        self.audio = SimpleNamespace(current_db=float("-inf"))
        app = create_app(self.cfg, PROJECT_DIR, audio_monitor=self.audio, video_monitor=None)
        self.client = TestClient(app)

    def tearDown(self):
        self.tmp.cleanup()

    def _insert_events(self):
        rows = [
            # (timestamp, db_max, category, class, vehicle)
            (self.now, 71.2, "vehicle", "Truck", "truck"),
            (self.now - timedelta(minutes=5), float("-inf"), "other", None, None),
            (self.now - timedelta(days=10), 68.0, "animal", "Dog", None),
        ]
        self.noise_ids = [
            insert_event(
                self.db_path, "noise", when, db_max=db, db_avg=db, trigger_type="peak",
                sound_class=sound_class, sound_category=category, vehicle_class=vehicle,
                confidence="high", clip_path=None, audio_muted=0, duration_ms=900,
            )
            for when, db, category, sound_class, vehicle in rows
        ]

    def _insert_passages(self):
        self.inside = self.clips / "passage_inside.mp4"
        self.inside.write_bytes(b"\x00" * 64)
        outside = self.base / "fuera.mp4"
        outside.write_bytes(b"\x00" * 32)

        self.car_id = insert_event(
            self.db_path, "traffic", self.now, vehicle_class="car", lane="left",
            direction="ltr", speed_kmh=48.0, detection_score=0.71, db_max=55.4,
            track_id=7, clip_path=str(self.inside),
        )
        insert_event(
            self.db_path, "traffic", self.now - timedelta(days=10), vehicle_class="truck",
            lane="right", direction="rtl", track_id=8, clip_path=str(outside),
        )


class LiveApiTest(WebTestCase):
    def test_health_distinguishes_requested_backend_from_fallback(self):
        status = {"requested_backend": "coral", "active_backend": "cpu", "fallbacks": 1}
        self.audio.classifier_status = lambda: status
        health = self.client.get("/api/health").json()
        self.assertEqual(health["audio_classifier"], status)

    def test_health_exposes_the_effective_trigger_policy(self):
        # r3, M3: una config degradada (tráfico sin clases) arranca; el
        # health la deja a la vista.
        self.cfg["tracker"]["record_classes"] = []
        self.cfg["triggers"]["noise_schedule"] = {"enabled": True, "start": "00:00", "end": "23:59"}
        self.cfg["triggers"]["noise_only_with_traffic"] = True
        client = TestClient(create_app(self.cfg, PROJECT_DIR, audio_monitor=self.audio))
        policy = client.get("/api/health").json()["policy"]
        self.assertEqual(policy["record_classes"], [])
        self.assertTrue(policy["traffic_enabled"])
        self.assertTrue(policy["traffic_active_now"])
        self.assertFalse(policy["traffic_schedule_enabled"])
        self.assertTrue(policy["noise_schedule_enabled"])
        self.assertTrue(policy["noise_only_with_traffic"])
        self.assertFalse(policy["traffic_only_with_noise"])

    def test_health_prefers_the_running_policy(self):
        from src.trigger_policy import TriggerPolicy
        self.audio.trigger_policy = TriggerPolicy(record_classes=frozenset({"bus"}), noise_enabled=False)
        client = TestClient(create_app(self.cfg, PROJECT_DIR, audio_monitor=self.audio))
        policy = client.get("/api/health").json()["policy"]
        self.assertEqual(policy["record_classes"], ["bus"])
        self.assertFalse(policy["noise_enabled"])
        self.assertFalse(policy["noise_active_now"])

    def test_live_returns_parseable_json_with_silence(self):
        response = self.client.get("/api/live")
        self.assertEqual(response.status_code, 200)
        payload = json.loads(response.text)  # -Infinity rompería aquí
        self.assertIsNone(payload["db"])
        self.assertEqual(payload["max_db_today"], 71.2)

    def test_live_max_db_is_null_when_only_silence_was_recorded(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("DELETE FROM events WHERE kind = 'noise' AND db_max > 0")
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
            self.client.get("/partials/events?direction=up").status_code, 400
        )

    def test_unknown_kind_is_rejected(self):
        self.assertEqual(self.client.get("/api/events?kind=vehicle").status_code, 400)

    def test_valid_filters_still_work(self):
        payload = self.client.get("/api/events?vehicle=truck").json()
        # La lista es única: el ruido de hoy y el pasaje de hace 10 días.
        self.assertEqual(
            sorted((e["kind"], e["vehicle_class"]) for e in payload["events"]),
            [("noise", "truck"), ("traffic", "truck")],
        )

    def test_bad_limit_is_a_validation_error(self):
        self.assertEqual(self.client.get("/api/events?limit=0").status_code, 422)


class ExportCsvTest(WebTestCase):
    def test_events_csv_has_header_and_rows(self):
        response = self.client.get("/api/export/events.csv?days=1")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers["content-type"])
        self.assertIn("attachment", response.headers["content-disposition"])

        rows = list(csv.DictReader(io.StringIO(response.text)))
        # Las filas de hace 10 días quedan fuera; hoy: dos ruidos y un pasaje.
        self.assertEqual(sorted(r["kind"] for r in rows), ["noise", "noise", "traffic"])
        self.assertEqual(rows[0]["trigger_type"], "peak")
        self.assertTrue(rows[0]["timestamp"].startswith(self.now.strftime("%Y-%m-%d")))
        self.assertTrue(all(r["timestamp_end"] >= r["timestamp"] for r in rows))

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
        self.assertEqual(rows[0]["kind"], "traffic")

    def test_passages_csv_includes_both_and_excludes_noise(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("UPDATE events SET kind='both' WHERE id=?", (self.noise_ids[0],))
        rows = list(csv.DictReader(io.StringIO(
            self.client.get("/api/export/passages.csv?days=1").text
        )))
        self.assertEqual(sorted(r["kind"] for r in rows), ["both", "traffic"])

    def test_events_csv_direction_filter_and_column(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE events SET kind='both', direction='ltr' WHERE id=?", (self.noise_ids[0],)
            )
        rows = list(csv.DictReader(io.StringIO(
            self.client.get("/api/export/events.csv?days=1&direction=ltr").text
        )))
        self.assertEqual(sorted(r["kind"] for r in rows), ["both", "traffic"])
        self.assertTrue(all(r["direction"] == "ltr" for r in rows))

    def test_events_csv_kind_filter(self):
        rows = list(csv.DictReader(io.StringIO(
            self.client.get("/api/export/events.csv?days=1&kind=traffic").text
        )))
        self.assertEqual([r["id"] for r in rows], [str(self.car_id)])

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
            conn.execute("UPDATE events SET sound_labels_json=? WHERE sound_class='Truck'", (json.dumps(ranked),))
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
                     "/partials/vehicle-chart"):
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
                "SELECT id FROM events WHERE clip_path = ?", (str(path),)
            ).fetchone()
        finally:
            conn.close()
        self.assertIsNotNone(row, f"sin fila para {path}")
        return int(row[0])


class UnifiedEventsTest(WebTestCase):
    """Lista única: badge de origen, alias de pasajes, pin y clip por id."""

    def _make_both(self) -> int:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE events SET kind='both', direction='rtl', clip_path=? WHERE id=?",
                (str(self.inside), self.noise_ids[0]),
            )
        return self.noise_ids[0]

    def test_list_renders_one_badge_per_kind(self):
        self._make_both()
        page = self.client.get("/partials/events").text
        for label in ("ruido", "tránsito", "ambos"):
            with self.subTest(label=label):
                self.assertIn(f">{label}</span>", page)

    def test_analysis_link_only_for_sound_kinds(self):
        page = self.client.get("/partials/events?kind=traffic").text
        self.assertNotIn("Ver métricas", page)
        page = self.client.get("/partials/events?kind=noise").text
        self.assertIn("Ver métricas", page)

    def test_kind_sets(self):
        both = self._make_both()
        ids = lambda kind: sorted(  # noqa: E731
            e["id"] for e in self.client.get(f"/api/events?kind={kind}").json()["events"]
        )
        self.assertEqual(ids("both"), [both])
        self.assertEqual(ids("sound"), sorted(self.noise_ids))
        self.assertIn(both, ids("passages"))
        self.assertNotIn(self.noise_ids[1], ids("passages"))

    def test_passages_alias_includes_both_and_excludes_noise(self):
        both = self._make_both()
        passages = self.client.get("/api/passages").json()["passages"]
        self.assertEqual({p["kind"] for p in passages}, {"traffic", "both"})
        self.assertIn(both, [p["id"] for p in passages])

    def test_counters_include_both_on_each_side(self):
        self._make_both()
        payload = self.client.get("/api/live").json()
        self.assertEqual(payload["events_today"], 2)    # noise + both
        self.assertEqual(payload["passages_today"], 2)  # traffic + both

    def test_pin_works_for_every_kind(self):
        both = self._make_both()
        for event_id in (self.noise_ids[1], self.car_id, both):
            with self.subTest(event_id=event_id):
                response = self.client.post(f"/api/events/{event_id}/pin?pinned=1")
                self.assertEqual(response.status_code, 200)
        pinned = {e["id"] for e in self.client.get("/api/events?pinned=1").json()["events"]}
        self.assertEqual(pinned, {self.noise_ids[1], self.car_id, both})
        self.client.post(f"/api/events/{self.car_id}/pin?pinned=0")
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT pinned, pinned_at FROM events WHERE id=?", (self.car_id,)
            ).fetchone()
        self.assertEqual(row, (0, None))

    def test_passage_pin_alias_refuses_noise(self):
        self.assertEqual(
            self.client.post(f"/api/passages/{self.car_id}/pin").status_code, 200
        )
        self.assertEqual(
            self.client.post(f"/api/passages/{self.noise_ids[0]}/pin").status_code, 404
        )

    def test_clip_served_by_unified_id(self):
        both = self._make_both()
        for path in (f"/api/clips/{self.car_id}", f"/api/clips/{both}",
                     f"/api/passage-clips/{both}"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 200)

    def test_passage_clip_alias_refuses_noise(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                "UPDATE events SET clip_path=? WHERE id=?", (str(self.inside), self.noise_ids[0])
            )
        self.assertEqual(self.client.get(f"/api/clips/{self.noise_ids[0]}").status_code, 200)
        self.assertEqual(
            self.client.get(f"/api/passage-clips/{self.noise_ids[0]}").status_code, 404
        )

    def test_analysis_is_404_for_traffic(self):
        both = self._make_both()
        self.assertEqual(
            self.client.get(f"/api/events/{self.car_id}/analysis").status_code, 404
        )
        for event_id in (self.noise_ids[0], both):
            with self.subTest(event_id=event_id):
                response = self.client.get(f"/api/events/{event_id}/analysis")
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["event_table"], "events")

    def test_stats_by_kind(self):
        self._make_both()
        kinds = {r["kind"]: r["n"] for r in self.client.get("/api/stats").json()["kinds"]}
        self.assertEqual(kinds, {"noise": 1, "traffic": 1, "both": 1})


if __name__ == "__main__":
    unittest.main()
