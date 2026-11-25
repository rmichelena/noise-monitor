#!/usr/bin/env python3
"""
Reinicio del servicio desde /settings (S4): mismo Basic auth y guard CSRF que
el guardado, config validada antes de reiniciar y comando exacto sin shell.
`subprocess.run` y `threading.Timer` se reemplazan: nada se ejecuta de verdad.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import web.app as webapp  # noqa: E402
from src.utils import load_config  # noqa: E402

AUTH = ("admin", "test-pass")
CMD = ["sudo", "-n", "systemctl", "--no-block", "restart", "noise-monitor.service"]
PREFLIGHT = ["sudo", "-n", "-l", "systemctl", "--no-block", "restart", "noise-monitor.service"]


class FakeTimer:
    """Registra el timer; `fire()` corre la función como lo haría el thread."""

    created: list["FakeTimer"] = []

    def __init__(self, interval, function):
        self.interval, self.function, self.started, self.daemon = interval, function, False, False
        FakeTimer.created.append(self)

    def start(self):
        self.started = True

    def fire(self):
        self.function()


class RestartTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.path = self.dir / "config.yaml"
        shutil.copy(PROJECT_DIR / "config.example.yaml", self.path)
        FakeTimer.created = []
        self.calls = []
        self.preflight_rc = 0
        self.restart_rc = 0

        def fake_run(args, **kwargs):
            self.calls.append((list(args), kwargs))
            rc = self.preflight_rc if list(args) == PREFLIGHT else self.restart_rc
            return subprocess.CompletedProcess(args, rc, stdout="", stderr="denied" if rc else "")

        for target in (
            mock.patch.dict(os.environ, {"NOISE_SETTINGS_PASSWORD": AUTH[1]}),
            mock.patch.object(webapp.subprocess, "run", side_effect=fake_run),
            mock.patch.object(webapp.threading, "Timer", FakeTimer),
            mock.patch.object(webapp.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
        ):
            target.start()
            self.addCleanup(target.stop)
        self.client = TestClient(webapp.create_app(load_config(self.path), self.dir, None, None))

    def restart(self, auth=AUTH, **kwargs):
        kwargs.setdefault("json", {})
        return self.client.post("/api/settings/restart", auth=auth, **kwargs)

    def assert_nothing_ran(self):
        self.assertEqual(self.calls, [])
        self.assertEqual(FakeTimer.created, [])

    def test_requires_auth(self):
        self.assertEqual(self.restart(auth=None).status_code, 401)
        self.assertEqual(self.restart(auth=("admin", "mala")).status_code, 401)
        with mock.patch.dict(os.environ, {"NOISE_SETTINGS_PASSWORD": ""}):
            self.assertEqual(self.restart().status_code, 503)
        self.assert_nothing_ran()

    def test_csrf_guard(self):
        response = self.client.post("/api/settings/restart", auth=AUTH, content=b"{}",
                                    headers={"Content-Type": "text/plain"})
        self.assertEqual(response.status_code, 415)
        for headers in ({"Sec-Fetch-Site": "cross-site"}, {"Origin": "https://evil.example"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.restart(headers=headers).status_code, 403)
        self.assert_nothing_ran()

    def test_guard_does_not_apply_to_get(self):
        headers = {"Sec-Fetch-Site": "cross-site", "Content-Type": "text/plain"}
        self.assertEqual(self.client.get("/api/settings", auth=AUTH, headers=headers).status_code, 200)
        self.assertEqual(self.client.get("/settings", auth=AUTH, headers=headers).status_code, 200)
        self.assertEqual(self.client.get("/api/settings/restart", auth=AUTH).status_code, 405)
        self.assert_nothing_ran()

    def test_invalid_config_is_not_restarted(self):
        text = self.path.read_text(encoding="utf-8").replace(
            "cooldown_seconds: 30", "cooldown_seconds: 30\n  noise_enable: false")  # clave mal escrita
        self.path.write_text(text, encoding="utf-8")
        response = self.restart()
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("noise_enable", response.json()["detail"])
        self.assert_nothing_ran()

    def test_psychoacoustic_without_phonometry_is_not_restarted(self):
        text = self.path.read_text(encoding="utf-8")
        text = text.replace("criterion: dba", "criterion: psychoacoustic")
        text = text.replace("audio_seconds: 10", "audio_seconds: 16").replace("video_seconds: 10", "video_seconds: 16")
        text += "\n"  # la calibración va bajo audio
        text = text.replace("audio:\n", "audio:\n  pressure_calibration: {id: cal-1, pa_per_unit: 1.0}\n", 1)
        text = text.replace("triggers:\n", "triggers:\n  psychoacoustic: {metric: loudness_mean, threshold: 20}\n", 1)
        self.path.write_text(text, encoding="utf-8")
        with mock.patch.dict(sys.modules, {"phonometry": None}):
            response = self.restart()
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("phonometry", response.json()["detail"])
        self.assert_nothing_ran()
        # La misma config, con la dependencia presente, sí se reinicia.
        with mock.patch.dict(sys.modules, {"phonometry": types.ModuleType("phonometry")}):
            self.assertEqual(self.restart().status_code, 202)

    def test_happy_path_answers_202_then_runs_the_exact_command(self):
        response = self.restart()
        self.assertEqual(response.status_code, 202, response.text)
        self.assertTrue(response.json()["restarting"])
        # Antes de responder sólo se comprobó el permiso; el reinicio va diferido.
        self.assertEqual([c[0] for c in self.calls], [PREFLIGHT])
        self.assertEqual(len(FakeTimer.created), 1)
        timer = FakeTimer.created[0]
        self.assertTrue(timer.started and timer.daemon)
        self.assertEqual(timer.interval, 1.0)
        timer.fire()
        self.assertEqual([c[0] for c in self.calls], [PREFLIGHT, CMD])
        for args, kwargs in self.calls:
            self.assertIsInstance(args, list)
            self.assertFalse(kwargs.get("shell", False))

    def test_a_second_request_does_not_schedule_another_restart(self):
        self.assertEqual(self.restart().status_code, 202)
        self.assertEqual(self.restart().status_code, 202)
        self.assertEqual(len(FakeTimer.created), 1)

    def test_failed_restart_allows_a_new_attempt(self):
        self.restart_rc = 1
        self.restart()
        with self.assertLogs("web.app", level="ERROR"):
            FakeTimer.created[0].fire()
        self.assertEqual(self.restart().status_code, 202)
        self.assertEqual(len(FakeTimer.created), 2)

    def test_sudo_without_permission_is_500_before_answering(self):
        self.preflight_rc = 1
        with self.assertLogs("web.app", level="ERROR"):
            response = self.restart()
        self.assertEqual(response.status_code, 500, response.text)
        self.assertEqual(FakeTimer.created, [])
        self.assertNotIn(CMD, [c[0] for c in self.calls])

    def test_missing_sudo_is_500(self):
        with mock.patch.object(webapp.shutil, "which", return_value=None), \
                self.assertLogs("web.app", level="ERROR"):
            response = self.restart()
        self.assertEqual(response.status_code, 500)
        self.assert_nothing_ran()

    def test_preflight_oserror_is_500(self):
        with mock.patch.object(webapp.subprocess, "run", side_effect=FileNotFoundError("sudo")), \
                self.assertLogs("web.app", level="ERROR"):
            response = self.restart()
        self.assertEqual(response.status_code, 500)
        self.assertEqual(FakeTimer.created, [])

    def test_health_exposes_the_process_start(self):
        started = self.client.get("/api/health").json()["started_at"]
        self.assertIsInstance(started, float)

    def test_settings_page_links_back_and_offers_restart(self):
        text = self.client.get("/settings", auth=AUTH).text
        self.assertIn('href="/"', text)
        self.assertIn('id="settings-restart"', text)


if __name__ == "__main__":
    unittest.main()
