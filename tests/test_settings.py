"""Ajustes web: auth cerrada, validación y persistencia con comentarios."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.utils import load_config
from web.app import create_app

BASE_DIR = Path(__file__).resolve().parent.parent
AUTH = ("admin", "test-pass")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    shutil.copy(BASE_DIR / "config.yaml", tmp_path / "config.yaml")
    config = load_config(tmp_path / "config.yaml")
    monkeypatch.setenv("NOISE_SETTINGS_PASSWORD", AUTH[1])
    return TestClient(create_app(config, tmp_path, None, None)), tmp_path


def test_settings_fail_closed_without_password(client, monkeypatch):
    c, _ = client
    monkeypatch.delenv("NOISE_SETTINGS_PASSWORD")
    response = c.get("/api/settings", auth=AUTH)
    assert response.status_code == 503


def test_settings_requires_basic_auth(client):
    c, _ = client
    assert c.get("/settings").status_code == 401
    assert c.get("/api/settings", auth=("admin", "wrong")).status_code == 401
    assert c.get("/settings", auth=AUTH).status_code == 200


def test_settings_page_and_get(client):
    c, _ = client
    page = c.get("/settings", auth=AUTH)
    assert page.status_code == 200
    assert "Ajustes" in page.text
    payload = c.get("/api/settings", auth=AUTH).json()
    assert payload["values"]["triggers.criterion"] == "dba"
    assert "groups" in payload


def test_settings_save_preserves_comments_and_backup(client):
    c, tmp_path = client
    response = c.post(
        "/api/settings", auth=AUTH, json={"triggers.peak_db": 41.5}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["changed"] == ["triggers.peak_db"]
    assert body["restart_required"] is True
    text = (tmp_path / "config.yaml").read_text()
    assert "# Umbrales de disparo" in text
    assert (tmp_path / "config.yaml.bak").exists()
    assert load_config(tmp_path / "config.yaml")["triggers"]["peak_db"] == 41.5


@pytest.mark.parametrize(
    "updates, detail",
    [
        ({"triggers.peak_db": 500}, "peak_db"),
        ({"triggers.criterion": "invented"}, "criterion"),
        ({"video.detector": "yolo", "video.yolo_model_path": ""}, "yolo_model_path"),
        ({"video.detection_roi": [0, 0, 2000, 100]}, "dentro"),
        ({"tracker.record_classes": ["car", "train"]}, "desconocidas"),
        ({"no.such": 1}, "no editables"),
    ],
)
def test_settings_reject_invalid_values(client, updates, detail):
    c, tmp_path = client
    before = (tmp_path / "config.yaml").read_text()
    response = c.post("/api/settings", auth=AUTH, json=updates)
    assert response.status_code == 400
    assert detail in response.json()["detail"]
    assert (tmp_path / "config.yaml").read_text() == before


def test_settings_snapshot_requires_camera(client):
    c, _ = client
    assert c.get("/api/settings/snapshot").status_code == 401
    assert c.get("/api/settings/snapshot", auth=AUTH).status_code == 404


def test_settings_snapshot_returns_latest_jpeg(tmp_path, monkeypatch):
    shutil.copy(BASE_DIR / "config.yaml", tmp_path / "config.yaml")
    config = load_config(tmp_path / "config.yaml")

    class FakeVideo:
        def latest_frame_jpeg(self):
            return b"\xff\xd8\xff\xd9"

    monkeypatch.setenv("NOISE_SETTINGS_PASSWORD", AUTH[1])
    c = TestClient(create_app(config, tmp_path, None, FakeVideo()))
    response = c.get("/api/settings/snapshot", auth=AUTH)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"\xff\xd8\xff\xd9"
