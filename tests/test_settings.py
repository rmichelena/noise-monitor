#!/usr/bin/env python3
"""
Ajustes web (settings v2, S2): auth cerrada, esquema cerrado, validación de
la config completa antes de escribir y persistencia que conserva comentarios.

Cada test parte de `config.example.yaml` copiado a un temporal: el config.yaml
real está en .gitignore y no existe en un clone limpio.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from src.utils import load_config, validate_config  # noqa: E402
from web.app import create_app  # noqa: E402
from web.settings import FIELD_MAP, apply_updates, read_config  # noqa: E402

AUTH = ("admin", "test-pass")
COMMENT = "# Detector: \"ssd\" (histórico 300x300) o \"yolo\" (160x640, sin deformación)."


class SettingsBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "config.yaml"
        shutil.copy(PROJECT_DIR / "config.example.yaml", self.path)
        env = mock.patch.dict(os.environ, {"NOISE_SETTINGS_PASSWORD": AUTH[1]})
        env.start()
        self.addCleanup(env.stop)
        config = load_config(self.path)
        self.client = TestClient(create_app(config, self.dir, None, None))

    def post(self, updates, auth=AUTH):
        return self.client.post("/api/settings", auth=auth, json=updates)

    def on_disk(self) -> dict:
        return yaml.safe_load(self.path.read_text(encoding="utf-8"))


class AuthTest(SettingsBase):
    def test_fail_closed_without_password(self):
        with mock.patch.dict(os.environ, {"NOISE_SETTINGS_PASSWORD": ""}):
            for method, url in (("get", "/settings"), ("get", "/api/settings"),
                                ("post", "/api/settings"), ("get", "/api/settings/snapshot")):
                with self.subTest(url=url, method=method):
                    self.assertEqual(getattr(self.client, method)(url, auth=AUTH).status_code, 503)

    def test_requires_basic_auth(self):
        self.assertEqual(self.client.get("/settings").status_code, 401)
        self.assertEqual(self.client.get("/api/settings", auth=("admin", "wrong")).status_code, 401)
        self.assertEqual(self.client.get("/api/settings", auth=("otro", AUTH[1])).status_code, 401)
        before = self.path.read_bytes()
        self.assertEqual(self.post({"triggers.peak_db": 50}, auth=("admin", "wrong")).status_code, 401)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(self.client.get("/settings", auth=AUTH).status_code, 200)

    def test_custom_user(self):
        with mock.patch.dict(os.environ, {"NOISE_SETTINGS_USER": "roberto"}):
            self.assertEqual(self.client.get("/api/settings", auth=AUTH).status_code, 401)
            self.assertEqual(self.client.get("/api/settings", auth=("roberto", AUTH[1])).status_code, 200)

    def test_dashboard_stays_public(self):
        self.assertEqual(self.client.get("/api/health").status_code, 200)


class ReadTest(SettingsBase):
    def test_old_config_reads_defaults(self):
        # config.example.yaml todavía no tiene las claves nuevas (S3).
        values = self.client.get("/api/settings", auth=AUTH).json()["values"]
        self.assertIs(values["triggers.noise_enabled"], True)
        self.assertIs(values["triggers.traffic_enabled"], True)
        self.assertIs(values["triggers.noise_only_with_traffic"], False)
        self.assertIs(values["triggers.noise_schedule.enabled"], False)
        self.assertEqual(values["triggers.noise_schedule.start"], "22:00")
        self.assertEqual(values["triggers.traffic_schedule.days"], [])
        self.assertEqual(sorted(values["tracker.record_classes"]),
                         ["bicycle", "bus", "car", "motorcycle", "truck"])
        self.assertEqual(values["triggers.peak_db"], 85)

    def test_page_renders_groups_and_new_types(self):
        page = self.client.get("/settings", auth=AUTH)
        self.assertEqual(page.status_code, 200)
        text = page.text
        for needle in ("Ruido", "Tráfico", "Motos y bicis", 'type="time"',
                       'data-path="triggers.noise_schedule.days"', 'data-classes="motorcycle,bicycle"',
                       'data-toggle="triggers.traffic_enabled"', "Sólo con un evento de ruido asociado"):
            self.assertIn(needle, text)
        # 7 días por cada horario.
        self.assertEqual(text.count('data-path="triggers.traffic_schedule.days"'), 7)

    def test_partial_class_group_is_shown_indeterminate(self):
        cfg = self.on_disk()
        cfg["tracker"]["record_classes"] = ["car", "motorcycle"]
        self.path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        text = self.client.get("/settings", auth=AUTH).text
        self.assertIn('data-partial="1"', text)


class WriteTest(SettingsBase):
    def test_partial_save_keeps_comments_and_leaves_backup(self):
        original = self.path.read_text(encoding="utf-8")
        response = self.post({"triggers.peak_db": 82.5})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["changed"], ["triggers.peak_db"])
        self.assertTrue(body["restart_required"])
        text = self.path.read_text(encoding="utf-8")
        self.assertIn(COMMENT, text)
        self.assertIn("peak_db: 82.5", text)
        # El resto del archivo queda idéntico: sólo cambia esa línea.
        diff = [(a, b) for a, b in zip(original.splitlines(), text.splitlines()) if a != b]
        self.assertEqual(len(diff), 1, diff)
        self.assertEqual(len(original.splitlines()), len(text.splitlines()))
        backup = Path(body["backup"])
        self.assertEqual(backup.read_text(encoding="utf-8"), original)
        # La config guardada arranca: pasa la validación del servicio.
        validate_config(load_config(self.path))

    def test_each_list_style_survives_a_single_field_save(self):
        # config.example.yaml indenta las listas bajo su clave; producción no
        # (indentless). Guardar un campo no debe reformatear las listas.
        for fixture in (PROJECT_DIR / "config.example.yaml",
                        PROJECT_DIR / "tests" / "fixtures" / "config_prod_style.yaml"):
            with self.subTest(fixture=fixture.name):
                shutil.copy(fixture, self.path)
                original = self.path.read_text(encoding="utf-8").splitlines()
                response = self.post({"triggers.peak_db": 82.5})
                self.assertEqual(response.status_code, 200, response.text)
                text = self.path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(text), len(original))
                diff = [(a, b) for a, b in zip(original, text) if a != b]
                self.assertEqual(diff, [(next(l for l in original if "peak_db:" in l),
                                         next(l for l in text if "peak_db:" in l))])
                self.assertIn("peak_db: 82.5", diff[0][1])

    def test_new_keys_go_before_the_next_section_comment(self):
        for fixture in (PROJECT_DIR / "config.example.yaml",
                        PROJECT_DIR / "tests" / "fixtures" / "config_prod_style.yaml"):
            with self.subTest(fixture=fixture.name):
                shutil.copy(fixture, self.path)
                original = self.path.read_text(encoding="utf-8").splitlines()
                response = self.post({"triggers.noise_enabled": False,
                                      "triggers.noise_schedule.enabled": True,
                                      "triggers.noise_schedule.days": [5]})
                self.assertEqual(response.status_code, 200, response.text)
                text = self.path.read_text(encoding="utf-8").splitlines()
                header = next(i for i, l in enumerate(text) if l.startswith("# Parámetros de video"))
                for key in ("  noise_enabled:", "  noise_schedule:"):
                    self.assertLess(next(i for i, l in enumerate(text) if l.startswith(key)), header, key)
                # El contenido de noise_schedule queda entero arriba del encabezado,
                # y el encabezado sigue pegado a su sección.
                start = next(i for i, l in enumerate(text) if l.startswith("  noise_schedule:"))
                body = [l for l in text[start + 1:header] if l.strip()]
                self.assertTrue(body and all(l.startswith("    ") for l in body), body)
                self.assertEqual(len(body), 3, body)  # enabled, days, - 5
                self.assertEqual(text[header + 1], "video:")
                # Lo único nuevo son las líneas agregadas; nada viejo cambia ni se pierde.
                added = [l for l in text if l not in original]
                self.assertEqual(len(text) - len(original), len(added), added)
                self.assertEqual(self.on_disk()["triggers"]["noise_schedule"], {"enabled": True, "days": [5]})

    def test_replacing_a_list_keeps_the_following_section_header(self):
        for fixture in (PROJECT_DIR / "config.example.yaml",
                        PROJECT_DIR / "tests" / "fixtures" / "config_prod_style.yaml"):
            with self.subTest(fixture=fixture.name):
                shutil.copy(fixture, self.path)
                self.post({"tracker.record_classes": ["car", "bus"]})       # se agrega al final de tracker
                before = self.path.read_text(encoding="utf-8").splitlines()
                self.post({"tracker.record_classes": ["car", "truck"]})     # se reemplaza la lista
                after = self.path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(after), len(before))
                diff = [(a, b) for a, b in zip(before, after) if a != b]
                self.assertEqual(len(diff), 1, diff)
                self.assertIn("truck", diff[0][1])
                self.assertIn("# Clasificador de audio (YAMNet)", after)

    def test_backup_round_trip_restores_previous(self):
        original = self.path.read_bytes()
        self.post({"triggers.cooldown_seconds": 12})
        self.post({"triggers.cooldown_seconds": 13})
        self.assertIn("cooldown_seconds: 12", (self.dir / "config.yaml.bak").read_text())
        shutil.copy(self.dir / "config.yaml.bak", self.path)
        self.post({"triggers.cooldown_seconds": 30})  # el valor original
        self.assertEqual(self.path.read_bytes(), original)

    def test_default_of_absent_key_is_not_written(self):
        original = self.path.read_bytes()
        response = self.post({
            "triggers.noise_enabled": True, "triggers.noise_schedule.days": [],
            "tracker.record_classes": ["bicycle", "motorcycle", "bus", "truck", "car"],
        })
        self.assertEqual(response.json()["changed"], [])
        self.assertIsNone(response.json()["backup"])
        self.assertEqual(self.path.read_bytes(), original)

    def test_same_set_in_other_order_is_not_a_change(self):
        cfg = self.on_disk()
        cfg["tracker"]["record_classes"] = ["bicycle", "motorcycle", "car"]
        cfg["triggers"]["noise_schedule"] = {"enabled": True, "days": [7, 6]}
        self.path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        original = self.path.read_bytes()
        response = self.post({"tracker.record_classes": ["car", "motorcycle", "bicycle"],
                              "triggers.noise_schedule.days": [6, 7]})
        self.assertEqual(response.json()["changed"], [])
        self.assertEqual(self.path.read_bytes(), original)

    def test_new_types_are_saved_normalized(self):
        response = self.post({
            "triggers.traffic_enabled": True,
            "triggers.traffic_schedule.enabled": True,
            "triggers.traffic_schedule.start": "07:30",
            "triggers.traffic_schedule.end": "19:00",
            "triggers.traffic_schedule.days": [5, 1, 1, 3],
            "tracker.record_classes": ["bicycle", "car", "motorcycle"],
        })
        self.assertEqual(response.status_code, 200, response.text)
        cfg = self.on_disk()
        self.assertEqual(cfg["triggers"]["traffic_schedule"],
                         {"enabled": True, "start": "07:30", "end": "19:00", "days": [1, 3, 5]})
        self.assertEqual(cfg["tracker"]["record_classes"], ["car", "motorcycle", "bicycle"])
        self.assertNotIn("traffic_enabled", cfg["triggers"], "default sin cambio")

    def test_flags_and_schedule_reach_the_policy(self):
        from src.trigger_policy import TriggerPolicy

        self.post({"triggers.noise_only_with_traffic": True,
                   "triggers.noise_schedule.enabled": True,
                   "triggers.noise_schedule.days": [6, 7]})
        policy = TriggerPolicy.from_config(load_config(self.path))
        self.assertTrue(policy.noise_only_with_traffic)
        self.assertEqual(policy.noise_schedule.days, frozenset({6, 7}))


class RejectTest(SettingsBase):
    def assert_rejected(self, updates, match):
        before = self.path.read_bytes()
        response = self.post(updates)
        self.assertEqual(response.status_code, 400, response.text)
        self.assertRegex(response.json()["detail"], match)
        self.assertEqual(self.path.read_bytes(), before, "un rechazo no toca el archivo")
        self.assertFalse((self.dir / "config.yaml.bak").exists())

    def test_unknown_path(self):
        self.assert_rejected({"paths.database": "/tmp/x.db"}, "no editables")
        self.assert_rejected({"triggers.peak_db": 80, "web.port": 1}, "web.port")

    def test_invalid_values(self):
        cases = [
            ({"triggers.peak_db": 500}, "<= 194"),
            ({"triggers.peak_db": "80"}, "numérico"),
            ({"triggers.peak_db": True}, "numérico"),
            ({"triggers.sustained_duration_ms": 1.5}, "entero"),
            ({"triggers.noise_enabled": "true"}, "booleano"),
            ({"triggers.criterion": "loud"}, "uno de"),
            ({"triggers.noise_schedule.start": "7:00"}, "HH:MM"),
            ({"triggers.noise_schedule.end": "24:00"}, "fuera de rango"),
            ({"triggers.noise_schedule.days": [0, 1]}, "1–7"),
            ({"triggers.noise_schedule.days": [True]}, "1–7"),
            ({"tracker.record_classes": ["car", "person"]}, "desconocidas"),
            ({"tracker.record_classes": ["car", "motorcycle"]}, "Motos y bicis"),
            ({"retention.clip_days": 0}, ">= 1"),
            ({"video.detection_roi": [10, 10, 5, 20]}, "x1 < x2"),
            ({"video.detection_roi": [0, 0, 5000, 20]}, "dentro de"),
        ]
        for updates, match in cases:
            with self.subTest(updates=updates):
                self.assert_rejected(updates, match)

    def test_incoherent_flags(self):
        self.assert_rejected(
            {"triggers.noise_only_with_traffic": True, "triggers.traffic_enabled": False},
            "noise_only_with_traffic requiere",
        )
        self.assert_rejected(
            {"triggers.traffic_only_with_noise": True, "triggers.noise_enabled": False},
            "traffic_only_with_noise requiere",
        )

    def test_empty_window_is_rejected(self):
        self.assert_rejected(
            {"triggers.traffic_schedule.start": "07:00", "triggers.traffic_schedule.end": "07:00"},
            "ventana vacía",
        )

    def test_file_is_validated_as_the_service_reads_it(self):
        # Si un valor se escribiera de forma que PyYAML lo lea distinto que
        # ruamel (p. ej. una hora sin comillas → sexagesimal), el guardado se
        # rechaza en vez de dejar un config.yaml con el que no arranca.
        import web.settings as settings

        with mock.patch.object(settings, "DoubleQuotedScalarString", str):
            self.assert_rejected(
                {"triggers.traffic_schedule.end": "19:00"}, "no sería válido.*traffic_schedule.end",
            )

    def test_bad_json(self):
        response = self.client.post("/api/settings", auth=AUTH, content=b"{no",
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.post(["triggers.peak_db"]).status_code, 400)


class SchemaTest(unittest.TestCase):
    def test_every_field_path_is_read_by_the_program(self):
        # Una ruta del esquema que nadie lee sería un ajuste que no hace nada.
        import subprocess

        sources = subprocess.run(
            ["git", "grep", "-h", "-e", "", "--", "src", "main.py", "scripts"],
            cwd=PROJECT_DIR, capture_output=True, text=True, check=True,
        ).stdout
        for path in FIELD_MAP:
            key = path.rsplit(".", 1)[-1]
            with self.subTest(path=path):
                self.assertIn(key, sources)

    def test_apply_validates_the_whole_config(self):
        # Primera barrera, antes de tocar el disco; save_config re-valida los bytes.
        from src.utils import ConfigError

        cfg = read_config(PROJECT_DIR / "config.example.yaml")
        with self.assertRaisesRegex(ConfigError, "noise_only_with_traffic requiere"):
            apply_updates(cfg, {"triggers.noise_only_with_traffic": True,
                                "triggers.traffic_enabled": False}, PROJECT_DIR)

    def test_apply_without_disk(self):
        cfg = read_config(PROJECT_DIR / "config.example.yaml")
        cfg, changed = apply_updates(cfg, {"triggers.noise_enabled": False}, PROJECT_DIR)
        self.assertEqual(changed, ["triggers.noise_enabled"])
        self.assertIs(cfg["triggers"]["noise_enabled"], False)


if __name__ == "__main__":
    unittest.main()
