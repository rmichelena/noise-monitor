#!/usr/bin/env python3
"""
Habilitación, horarios y filtros de los disparadores (settings v2, S1).

- `Schedule`: ventanas diurnas y nocturnas, fin exclusivo, días ISO (en una
  ventana nocturna cuenta el día de inicio).
- `TriggerPolicy`: defaults = comportamiento anterior; validación de tipos,
  clases y coherencia entre flags.
- AudioMonitor: ruido apagado o fuera de horario no emite disparos, pero
  sigue midiendo.
- EventManager: tráfico apagado/fuera de horario/clase no registrada/
  `traffic_only_with_noise` no guarda el pasaje, pero un ruido que lo reclama
  nace `both` con un solo clip; `noise_only_with_traffic` descarta el ruido
  sin pasaje.
"""

import copy
import sqlite3
import sys
import tempfile
import threading
import time
import types
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from queue import Empty, Queue
from unittest.mock import MagicMock

import numpy as np
import yaml

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:  # PortAudio no está instalado en todos los hosts (ni en CI)
    import sounddevice  # noqa: F401
except Exception:  # pragma: no cover - depende del host
    _stub = types.ModuleType("sounddevice")
    _stub.InputStream = object  # type: ignore[attr-defined]
    sys.modules["sounddevice"] = _stub

from src.audio_classifier import AudioClassifier  # noqa: E402
from src.audio_monitor import AudioMonitor  # noqa: E402
from src.event_manager import EventManager  # noqa: E402
from src.schedule import Schedule  # noqa: E402
from src.trigger_policy import TriggerPolicy  # noqa: E402
from src.utils import ConfigError, init_database, validate_config  # noqa: E402
from support import create_events_table, example_config  # noqa: E402

PRE = 2.0
POST = 2.0


def at(day: int, hhmm: str) -> datetime:
    """2026-11-23 es lunes (ISO 1): `at(1, "22:00")` = lunes 22:00."""
    hour, minute = map(int, hhmm.split(":"))
    return datetime(2026, 11, 22 + day, hour, minute)


def window_around(epoch: float, inside: bool) -> dict:
    """Horario de una hora que contiene (o no) al instante dado."""
    t = datetime.fromtimestamp(epoch)
    if inside:
        start, end = t - timedelta(minutes=30), t + timedelta(minutes=30)
    else:
        start, end = t + timedelta(minutes=60), t + timedelta(minutes=120)
    return {"enabled": True, "start": start.strftime("%H:%M"), "end": end.strftime("%H:%M")}


class ScheduleTest(unittest.TestCase):
    def test_disabled_is_always_active(self):
        schedule = Schedule.parse({"enabled": False, "start": "08:00", "end": "09:00"})
        self.assertTrue(schedule.is_active(at(1, "03:00")))
        self.assertTrue(Schedule.parse(None).is_active(at(1, "03:00")))

    def test_day_window_end_is_exclusive(self):
        schedule = Schedule.parse({"enabled": True, "start": "08:00", "end": "18:00"})
        self.assertFalse(schedule.is_active(at(1, "07:59")))
        self.assertTrue(schedule.is_active(at(1, "08:00")))
        self.assertTrue(schedule.is_active(at(1, "17:59")))
        self.assertFalse(schedule.is_active(at(1, "18:00")))

    def test_night_window_crosses_midnight(self):
        schedule = Schedule.parse({"enabled": True, "start": "22:00", "end": "07:00"})
        self.assertFalse(schedule.is_active(at(1, "21:59")))
        self.assertTrue(schedule.is_active(at(1, "22:00")))
        self.assertTrue(schedule.is_active(at(2, "03:00")))
        self.assertTrue(schedule.is_active(at(2, "06:59")))
        self.assertFalse(schedule.is_active(at(2, "07:00")))
        self.assertFalse(schedule.is_active(at(2, "12:00")))

    def test_days_filter_day_window(self):
        schedule = Schedule.parse({"enabled": True, "start": "08:00", "end": "18:00", "days": [6, 7]})
        self.assertFalse(schedule.is_active(at(5, "10:00")))  # viernes
        self.assertTrue(schedule.is_active(at(6, "10:00")))   # sábado
        self.assertTrue(schedule.is_active(at(7, "10:00")))   # domingo

    def test_night_window_belongs_to_start_day(self):
        # Sólo la noche del viernes: viernes 22:00 → sábado 07:00.
        schedule = Schedule.parse({"enabled": True, "start": "22:00", "end": "07:00", "days": [5]})
        self.assertFalse(schedule.is_active(at(5, "03:00")))  # madrugada del viernes = noche del jueves
        self.assertTrue(schedule.is_active(at(5, "23:00")))
        self.assertTrue(schedule.is_active(at(6, "03:00")))
        self.assertFalse(schedule.is_active(at(6, "23:00")))

    def test_invalid_values_are_rejected(self):
        for raw in (
            {"enabled": True, "start": "8:00", "end": "09:00"},
            {"enabled": True, "start": "24:00", "end": "09:00"},
            {"enabled": True, "start": "08:60", "end": "09:00"},
            {"enabled": True, "start": "08:00", "end": "08:00"},
            {"enabled": True, "start": 800, "end": "09:00"},
            {"enabled": "yes"},
            {"enabled": True, "days": [0]},
            {"enabled": True, "days": [8]},
            {"enabled": True, "days": [True]},
            {"enabled": True, "days": "1,2"},
            {"enabled": True, "dias": [1]},
            ["22:00", "07:00"],
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                Schedule.parse(raw)


class PolicyConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = example_config(self.tmp.name)

    def test_defaults_keep_previous_behaviour(self):
        policy = TriggerPolicy.from_config(self.cfg)
        now = time.time()
        self.assertTrue(policy.noise_active(now))
        self.assertTrue(policy.traffic_active(now))
        for cls in ("car", "truck", "bus", "motorcycle", "bicycle"):
            self.assertTrue(policy.records_passage(cls, now), cls)
        validate_config(self.cfg)

    def assert_config_error(self, match: str):
        with self.assertRaisesRegex(ConfigError, match):
            validate_config(self.cfg)

    def test_flags_must_be_booleans(self):
        for key in ("noise_enabled", "noise_only_with_traffic", "traffic_enabled", "traffic_only_with_noise"):
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                cfg["triggers"][key] = "true"
                with self.assertRaisesRegex(ConfigError, f"triggers.{key}"):
                    validate_config(cfg)

    def test_schedule_errors_name_the_key(self):
        self.cfg["triggers"]["traffic_schedule"] = {"enabled": True, "start": "25:00", "end": "07:00"}
        self.assert_config_error("triggers.traffic_schedule.start")

    def test_record_classes_must_be_known(self):
        self.cfg["tracker"]["record_classes"] = ["car", "person"]
        self.assert_config_error("person")
        self.cfg["tracker"]["record_classes"] = "car"
        self.assert_config_error("tracker.record_classes")

    def test_empty_record_classes_with_traffic_starts_but_logs_error(self):
        # Issue #21 r2, M-A: tráfico encendido sin clases no grababa ningún
        # pasaje y no lo avisaba. r3, M3: al arrancar no aborta (una config
        # vieja no puede dejar caído el servicio y su página de ajustes),
        # pero lo registra como error. El guardado sí lo rechaza (test_settings).
        self.cfg["tracker"]["record_classes"] = []
        with self.assertLogs(level="ERROR") as logs:
            validate_config(self.cfg)
        self.assertTrue(any("record_classes vacío" in line for line in logs.output))
        policy = TriggerPolicy.from_config(self.cfg)
        self.assertTrue(policy.traffic_enabled)
        self.assertFalse(policy.records_passage("car", time.time()))
        self.cfg["triggers"]["traffic_enabled"] = False
        with self.assertNoLogs(level="ERROR"):
            validate_config(self.cfg)

    def test_noise_only_with_traffic_needs_traffic(self):
        self.cfg["triggers"].update(noise_only_with_traffic=True, traffic_enabled=False)
        self.assert_config_error("noise_only_with_traffic")

    def test_traffic_only_with_noise_needs_noise(self):
        self.cfg["triggers"].update(traffic_only_with_noise=True, noise_enabled=False)
        self.assert_config_error("traffic_only_with_noise")

    def test_both_only_with_flags_are_valid_together(self):
        # Sólo eventos `both`: combinación legítima.
        self.cfg["triggers"].update(noise_only_with_traffic=True, traffic_only_with_noise=True)
        validate_config(self.cfg)

    def test_falsy_days_are_rejected_not_widened(self):
        # Sólo null/ausente es «todos los días»; un falsy no se amplía en silencio.
        for days in (False, 0, "", {}):
            with self.subTest(days=days):
                cfg = copy.deepcopy(self.cfg)
                cfg["triggers"]["noise_schedule"] = {"enabled": True, "start": "22:00", "end": "07:00", "days": days}
                with self.assertRaisesRegex(ConfigError, "triggers.noise_schedule.days"):
                    validate_config(cfg)
        self.cfg["triggers"]["noise_schedule"] = {"enabled": True, "start": "22:00", "end": "07:00", "days": None}
        self.assertEqual(TriggerPolicy.from_config(self.cfg).noise_schedule.days, frozenset())

    def test_unknown_keys_are_rejected(self):
        # Un typo no puede dejar encendida la sección que se quería apagar.
        for section, key in (("triggers", "noise_enable"), ("triggers", "trafic_enabled"),
                             ("tracker", "record_class")):
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                cfg[section][key] = False
                with self.assertRaisesRegex(ConfigError, f"{section}: claves desconocidas.*{key}"):
                    validate_config(cfg)

    def test_every_known_key_is_accepted(self):
        # Riesgo espejo de la lista cerrada: ninguna clave legítima se rechaza.
        self.cfg["triggers"].update(
            criterion="dba", psychoacoustic={}, noise_enabled=True, noise_only_with_traffic=False,
            traffic_enabled=True, traffic_only_with_noise=False,
            noise_schedule={"enabled": False}, traffic_schedule={"enabled": False},
        )
        self.cfg["tracker"]["record_classes"] = ["car"]
        validate_config(self.cfg)
        for name in ("config.example.yaml", "config.simulate.yaml.example"):
            with self.subTest(config=name):
                raw = yaml.safe_load((PROJECT_DIR / name).read_text(encoding="utf-8"))
                TriggerPolicy.from_config(raw)

    def test_flags_of_a_disabled_section_do_not_block(self):
        # El flag de una sección apagada no exige nada a la otra.
        self.cfg["triggers"].update(noise_enabled=False, noise_only_with_traffic=True, traffic_enabled=False)
        validate_config(self.cfg)


SAMPLE_RATE = 44100
BLOCK = 4096


def tone(amplitude: float) -> np.ndarray:
    t = np.arange(BLOCK, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)


class AudioGatingTest(unittest.TestCase):
    def monitor(self, **triggers) -> tuple[AudioMonitor, Queue]:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = example_config(tmp.name)
        cfg["audio"]["sample_rate"] = SAMPLE_RATE
        cfg["audio"]["block_size"] = BLOCK
        cfg["triggers"].update(triggers)
        init_database(cfg, PROJECT_DIR)
        queue: Queue = Queue()
        monitor = AudioMonitor(config=cfg, event_queue=queue, base_dir=PROJECT_DIR)
        self.addCleanup(monitor.stop)
        monitor.peak_db = 80.0
        monitor.sustained_db = 95.0
        return monitor, queue

    def test_noise_disabled_measures_but_does_not_trigger(self):
        monitor, queue = self.monitor(noise_enabled=False)
        monitor._process_block(tone(0.7), timestamp=time.time())
        with self.assertRaises(Empty):
            queue.get(timeout=0.05)
        self.assertGreater(monitor.current_db, 80.0)
        self.assertEqual(len(monitor.audio_buffer), 1)
        self.assertEqual(monitor.trigger_status()["noise_gated"], 1)

    def test_noise_schedule_gates_by_trigger_timestamp(self):
        now = time.time()
        monitor, queue = self.monitor(noise_schedule=window_around(now, inside=False))
        monitor._process_block(tone(0.7), timestamp=now)
        with self.assertRaises(Empty):
            queue.get(timeout=0.05)

        monitor, queue = self.monitor(noise_schedule=window_around(now, inside=True))
        monitor._process_block(tone(0.7), timestamp=now)
        self.assertEqual(queue.get(timeout=0.2)["type"], "peak")
        self.assertEqual(monitor.noise_gated, 0)

    def test_psychoacoustic_emit_goes_through_the_gate(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        cfg = example_config(tmp.name)
        cfg["audio"].update(sample_rate=48000, block_size=4800)
        cfg["audio"]["pressure_calibration"] = {"id": "test-tone", "pa_per_unit": 2.0}
        cfg["triggers"].update(criterion="psychoacoustic", psychoacoustic={"threshold": 4},
                               noise_enabled=False)
        queue: Queue = Queue()
        monitor = AudioMonitor(config=cfg, event_queue=queue, base_dir=PROJECT_DIR)
        self.addCleanup(monitor.stop)
        # Camino real del detector: su _emit arma el evento y llama al callback.
        now = time.time()
        monitor._perceptual_trigger._emit(now, 5.0, [(now, None, 90.0)], np.zeros(16))
        with self.assertRaises(Empty):
            queue.get(timeout=0.05)
        self.assertEqual(monitor.noise_gated, 1)


class PassageGatingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = example_config(self.tmp.name)
        self.cfg["buffer"]["pre_event_seconds"] = PRE
        self.cfg["buffer"]["post_event_seconds"] = POST
        self.cfg["triggers"]["cooldown_seconds"] = 0
        self.base = time.time() - 20.0

    def manager(self, triggers: dict | None = None, record_classes=None) -> EventManager:
        cfg = copy.deepcopy(self.cfg)
        cfg["triggers"].update(triggers or {})
        if record_classes is not None:
            cfg["tracker"]["record_classes"] = record_classes
        car = {"class": "car", "score": 0.7, "cx": 400, "cy": 700, "lane": "left",
               "bbox": [0.6, 0.2, 0.7, 0.3]}
        video = MagicMock()
        video.detections_between.return_value = [car]
        video.recent_detections = [car]
        video.video_buffer = [
            (self.base + i * 0.1, np.zeros((8, 8, 3), dtype=np.uint8)) for i in range(181)
        ]
        audio = MagicMock()
        audio.audio_buffer = [
            (self.base + i * 0.1, np.zeros(16, dtype=np.float32)) for i in range(181)
        ]
        audio.current_db = 40.0
        audio.level_stats.return_value = None
        audio.level_snapshot.return_value = []
        audio._classifier = None

        em = EventManager(audio, video, config=cfg, base_dir=PROJECT_DIR)
        self.db = cfg["paths"]["database"]
        create_events_table(self.db)
        self.clip_calls = []
        self.passage_clip_calls = 0
        self.fail_clip = set()

        def fake_save_clip(audio_blocks, video_frames, sample_rate, event_timestamp,
                           mute_audio=False, prefix="event", unique_name=False):
            n = len(self.clip_calls)
            self.clip_calls.append({
                "audio": [ts for ts, _ in audio_blocks], "mute": mute_audio,
            })
            if n in self.fail_clip:
                raise RuntimeError("ffmpeg falló")
            path = em.clips_dir / f"{prefix}_{n}.mp4"
            path.write_bytes(b"mp4")
            return str(path), 1000 + n

        def fake_passage_clip(passage):
            self.passage_clip_calls += 1
            path = em.clips_dir / f"passage_{passage['track_id']}.mp4"
            path.write_bytes(b"mp4")
            return str(path), 5000, False

        em._save_clip = fake_save_clip
        em._save_passage_clip = fake_passage_clip
        return em

    def passage(self, em, enter=3.0, exit_=9.0, track_id=5, cls="car"):
        passage = {
            "vehicle_class": cls, "lane": "left", "direction": "ltr",
            "speed_kmh": 30.0, "detection_score": 0.6, "track_id": track_id,
            "timestamp_enter": self.base + enter, "timestamp_exit": self.base + exit_,
        }
        em._process_passage(passage)
        return passage

    def trigger(self, em, at_=10.0, sound_class="Car", category="vehicle"):
        em._process_trigger({
            "type": "peak", "db": 70.0, "timestamp": self.base + at_,
            "sound_class": sound_class, "sound_category": category, "sound_confidence": 0.8,
        })

    def rows(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM events ORDER BY id")]
        finally:
            conn.close()

    def assert_both_from_memory(self, row, track_id=5):
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["track_id"], track_id)
        self.assertEqual(row["vehicle_class"], "car")
        self.assertEqual(row["direction"], "ltr")
        self.assertEqual(row["speed_kmh"], 30.0)
        self.assertEqual(row["sound_class"], "Car")
        self.assertEqual(row["db_max"], 70.0)
        self.assertAlmostEqual(row["peak_ts"], self.base + 10.0)
        # Un solo clip, con la ventana unión [3, 12].
        self.assertEqual(len(self.clip_calls), 1)
        self.assertAlmostEqual(min(self.clip_calls[0]["audio"]), self.base + 3.0, places=3)
        self.assertAlmostEqual(max(self.clip_calls[0]["audio"]), self.base + 12.0, places=3)
        self.assertAlmostEqual(row["start_ts"], self.base + 3.0, places=3)
        self.assertAlmostEqual(row["end_ts"], self.base + 12.0, places=3)
        self.assertTrue(Path(row["clip_path"]).exists())

    def test_defaults_record_traffic_as_before(self):
        em = self.manager()
        self.passage(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "traffic")
        self.assertEqual(self.passage_clip_calls, 1)

    def test_traffic_disabled_keeps_passage_for_correlation(self):
        em = self.manager({"traffic_enabled": False})
        self.passage(em)
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.passage_clip_calls, 0, "sin fila no se graba clip de pasaje")
        self.assertEqual(em.status()["passages_not_recorded"], 1)

        self.trigger(em)
        (row,) = self.rows()
        self.assert_both_from_memory(row)

    def test_traffic_schedule(self):
        em = self.manager({"traffic_schedule": window_around(self.base + 3.0, inside=False)})
        self.passage(em)
        self.assertEqual(self.rows(), [])

        em = self.manager({"traffic_schedule": window_around(self.base + 3.0, inside=True)})
        self.passage(em, track_id=6)
        self.assertEqual([r["kind"] for r in self.rows()], ["traffic"])

    def test_record_classes_filter_traffic_rows(self):
        em = self.manager(record_classes=["car", "bus"])
        self.passage(em, track_id=1, cls="truck")
        self.passage(em, track_id=2, cls="car")
        self.assertEqual([(r["kind"], r["vehicle_class"]) for r in self.rows()], [("traffic", "car")])
        self.assertEqual(em.passages_not_recorded, 1)

    def test_traffic_only_with_noise(self):
        em = self.manager({"traffic_only_with_noise": True})
        self.passage(em)
        self.assertEqual(self.rows(), [], "un pasaje sin ruido no deja fila")
        self.trigger(em)
        (row,) = self.rows()
        self.assert_both_from_memory(row)

    def test_unclaimed_passage_leaves_nothing(self):
        em = self.manager({"traffic_only_with_noise": True})
        self.passage(em)
        self.trigger(em, sound_class="Dog", category="animal")
        (row,) = self.rows()
        self.assertEqual(row["kind"], "noise")
        self.assertIsNone(row["track_id"])

    def test_noise_only_with_traffic_discards_lonely_noise(self):
        em = self.manager({"noise_only_with_traffic": True})
        self.trigger(em, sound_class="Dog", category="animal")
        self.assertEqual(self.rows(), [])
        self.assertEqual(self.clip_calls, [], "no se graba clip de un ruido descartado")
        self.assertEqual(em.status()["discarded_without_traffic"], 1)

    def test_noise_only_with_traffic_keeps_correlated_noise(self):
        em = self.manager({"noise_only_with_traffic": True})
        self.passage(em)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(em.discarded_without_traffic, 0)

    def test_only_both_events(self):
        em = self.manager({"noise_only_with_traffic": True, "traffic_only_with_noise": True})
        self.passage(em, track_id=1, enter=0.0, exit_=1.0, cls="bus")
        self.trigger(em, sound_class="Dog", category="animal")
        self.assertEqual(self.rows(), [])
        self.passage(em)
        self.trigger(em)
        (row,) = self.rows()
        self.assert_both_from_memory(row)

    def test_memory_passage_union_clip_fallback_keeps_the_event(self):
        em = self.manager({"traffic_enabled": False})
        self.fail_clip = {0}
        self.passage(em)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["track_id"], 5)
        self.assertEqual(len(self.clip_calls), 2, "unión fallida → clip del ruido")
        self.assertAlmostEqual(row["start_ts"], self.base + 10.0 - PRE)
        self.assertAlmostEqual(row["end_ts"], self.base + 10.0 + POST)
        self.assertTrue(Path(row["clip_path"]).exists())
        self.assertEqual(em.status()["union_clip_fallbacks"], 1)

    def record_union(self, em):
        results = []
        original = em._union_clip

        def recording(*args, **kwargs):
            results.append(original(*args, **kwargs))
            return results[-1]

        em._union_clip = recording
        return results

    def test_memory_passage_both_clips_fail_keeps_both_without_clip(self):
        em = self.manager({"traffic_enabled": False})
        self.fail_clip = {0, 1}
        results = self.record_union(em)
        passage = self.passage(em)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["track_id"], 5)
        self.assertIsNone(row["clip_path"])
        (result,) = results
        self.assertFalse(result[-1], "union_ok")
        self.assertEqual(em.status()["union_clip_fallbacks"], 1)
        self.assertTrue(passage["claimed"])

    def test_memory_passage_union_ok_when_union_clip_saved(self):
        em = self.manager({"traffic_enabled": False})
        results = self.record_union(em)
        self.passage(em)
        self.trigger(em)
        self.assertTrue(results[0][-1])
        self.assertEqual(em.status()["union_clip_fallbacks"], 0)

    # -- Voz en el audio que la unión agrega (H1) ------------------------
    VOICE = 1.0

    def with_real_classifier(self, em, voice_between=None):
        """`AudioClassifier` real con sólo el intérprete falso.

        Los bloques son de 0,1 s a 16 kHz; los que caen en `voice_between`
        llevan la marca VOICE, y el intérprete marca Speech si la ve en los
        15600 muestras que recibe. Así el recorte de `predict_scores` queda
        dentro del test: clasificar sólo la cola de la unión no la ve.
        """
        sr = em.config["audio"]["sample_rate"]
        block = sr // 10
        em.audio_monitor.audio_buffer = [
            (self.base + i * 0.1,
             np.full(block, self.VOICE if voice_between and voice_between[0] < i * 0.1 <= voice_between[1]
                     else 0.0, dtype=np.float32))
            for i in range(181)
        ]
        clf = AudioClassifier.__new__(AudioClassifier)
        clf.labels = ["Speech", "Car", "Silence"]
        clf.min_score = 0.3
        clf.category_map = {}
        clf._build_reverse_map()
        clf._lock = threading.Lock()
        clf.inference_count = 0
        clf.inference_ms_total = 0.0
        clf.active_backend = "cpu"
        clf.fallback_to_cpu = False
        clf._invoke = lambda audio: np.array(
            [0.9 if (audio == self.VOICE).any() else 0.0, 0.5, 0.0], dtype=np.float32)
        em.audio_monitor._classifier = clf
        return clf

    def manager_16k(self, triggers):
        self.cfg["audio"]["sample_rate"] = 16000
        return self.manager(triggers)

    def test_memory_passage_voice_at_start_mutes_union(self):
        # Voz al inicio del pasaje [3, 5]; el ruido dispara a los 10 s (ventana
        # [8, 12]). La cola de la unión es post-roll del ruido, sin voz.
        em = self.manager_16k({"traffic_enabled": False})
        self.with_real_classifier(em, voice_between=(3.0, 4.0))
        self.passage(em, enter=3.0, exit_=5.0)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["audio_muted"], 1)
        self.assertTrue(self.clip_calls[0]["mute"])

    def test_passage_inside_noise_window_voice_at_start_mutes_union(self):
        # Caso normal: pasaje [8.5, 10.5] dentro de la ventana del ruido
        # [8, 12]. La unión no agrega nada fuera de esa ventana, y el disparo
        # dBA no clasificó su audio.
        em = self.manager_16k({"traffic_enabled": False})
        self.with_real_classifier(em, voice_between=(8.5, 9.0))
        self.passage(em, enter=8.5, exit_=10.5)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["audio_muted"], 1)
        self.assertTrue(self.clip_calls[0]["mute"])

    def test_union_fallback_noise_clip_is_muted_too(self):
        # Si la unión falla, el clip del ruido también contiene el pasaje.
        em = self.manager_16k({"traffic_enabled": False})
        self.with_real_classifier(em, voice_between=(8.5, 9.0))
        self.fail_clip = {0}
        self.passage(em, enter=8.5, exit_=10.5)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(len(self.clip_calls), 2)
        self.assertTrue(self.clip_calls[1]["mute"])
        self.assertEqual(row["audio_muted"], 1)

    def test_memory_passage_voice_in_the_gap_mutes_union(self):
        # Entre el fin del pasaje (5) y el inicio de la ventana del ruido (8)
        # tampoco pasó ninguna clasificación, y entra en el clip.
        em = self.manager_16k({"traffic_enabled": False})
        self.with_real_classifier(em, voice_between=(6.0, 7.0))
        self.passage(em, enter=3.0, exit_=5.0)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["audio_muted"], 1)

    def test_memory_passage_without_voice_keeps_audio(self):
        em = self.manager_16k({"traffic_enabled": False})
        clf = self.with_real_classifier(em)
        self.passage(em, enter=3.0, exit_=5.0)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["audio_muted"], 0)
        self.assertGreater(clf.inference_count, 0)

    def test_upgrade_of_clipless_passage_checks_voice_at_start(self):
        # Pasaje `traffic` cuyo clip falló: el upgrade a `both` es la primera
        # vez que su audio pasa por la detección de voz.
        em = self.manager_16k({})
        em._save_passage_clip = lambda passage: ("", 0, False)
        self.with_real_classifier(em, voice_between=(3.0, 4.0))
        self.passage(em, enter=3.0, exit_=5.0)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["audio_muted"], 1)

    def test_upgrade_with_passage_clip_scans_noise_preroll(self):
        # H2: pasaje [10.5, 11.5] con clip propio. Ese clip cubre [8.5, 13.5]
        # y el disparo dBA mira ±0,97 s de los 10 s; la voz en [8.0, 8.4]
        # (pre-roll del ruido) sólo la ve el escaneo de la unión.
        em = self.manager_16k({})
        self.with_real_classifier(em, voice_between=(8.0, 8.4))
        self.passage(em, enter=10.5, exit_=11.5)
        (traffic,) = self.rows()
        self.assertTrue(traffic["clip_path"])
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["audio_muted"], 1)
        self.assertTrue(self.clip_calls[0]["mute"])

    def test_upgrade_union_failure_keeps_passage_clip_and_window(self):
        # M3: con la unión fallida, el clip del pasaje (la evidencia con el
        # vehículo) no se reemplaza por el sólo-ruido ni se borra.
        em = self.manager({})
        self.passage(em)
        (traffic,) = self.rows()
        self.fail_clip = {0}
        with self.assertLogs("src.event_manager", "WARNING") as logs:
            self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "both")
        self.assertEqual(row["sound_class"], "Car")
        for key in ("clip_path", "start_ts", "end_ts", "duration_ms", "audio_muted"):
            self.assertEqual(row[key], traffic[key], key)
        self.assertTrue(Path(traffic["clip_path"]).exists())
        self.assertEqual(len(self.clip_calls), 2, "unión fallida → fallback")
        self.assertEqual(list(em.clips_dir.glob("event_*.mp4")), [], "fallback huérfano")
        self.assertTrue(any(f"#{row['id']} sin clip unión" in m for m in logs.output), logs.output)

    def dba_noise_without_passage(self, em):
        """Disparo dBA real: sin `sound_class` (audio_monitor no lo trae), con
        `classify_event_clip` real sobre el clasificador del test."""
        em.audio_monitor.sample_rate = em.config["audio"]["sample_rate"]
        em.audio_monitor.classify_event_clip = types.MethodType(
            AudioMonitor.classify_event_clip, em.audio_monitor)
        em._process_trigger({"type": "peak", "db": 90.0, "timestamp": self.base + 10.0})
        (row,) = self.rows()
        # Hay un auto detectado en la ventana y ningún pasaje cerrado: nace
        # `both` en escena (S6), sin pasaje adjunto.
        self.assertEqual(row["kind"], "both")
        self.assertIsNone(row["track_id"])
        return row

    def test_dba_noise_scans_the_whole_clip_for_voice(self):
        # Voz en [8.0, 8.5]: fuera de ±0,97 s del pico (10 s) y de las tres
        # ventanas de etiquetas; sólo la ve el escaneo de todo el clip.
        em = self.manager_16k({})
        clf = self.with_real_classifier(em, voice_between=(8.0, 8.5))
        row = self.dba_noise_without_passage(em)
        self.assertEqual(row["audio_muted"], 1)
        self.assertTrue(self.clip_calls[0]["mute"])
        self.assertEqual(row["sound_class"], "Car", "las etiquetas salen de las ventanas del disparo")
        self.assertGreater(clf.inference_count, 3)

    def test_dba_noise_without_voice_keeps_audio(self):
        em = self.manager_16k({})
        self.with_real_classifier(em)
        row = self.dba_noise_without_passage(em)
        self.assertEqual(row["audio_muted"], 0)

    def test_dba_noise_without_classifier_is_not_muted(self):
        # Sin YAMNet, sólo el psicoacústico va mudo (comportamiento previo).
        em = self.manager_16k({})
        em.audio_monitor._classifier = None
        row = self.dba_noise_without_passage(em)
        self.assertEqual(row["audio_muted"], 0)

    # -- Fallos de la rama `both` nacida en memoria (M3) ------------------
    def test_memory_passage_union_failure_keeps_noise_and_releases_passage(self):
        em = self.manager({"traffic_enabled": False})

        def broken_union(*args, **kwargs):
            raise RuntimeError("buffer roto")

        em._union_clip = broken_union
        passage = self.passage(em)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "noise")
        self.assertIsNone(row["track_id"])
        self.assertFalse(passage["claimed"])
        self.assertEqual(len(self.clip_calls), 1, "clip normal del ruido")
        self.assertTrue(Path(row["clip_path"]).exists())

    def test_memory_passage_insert_failure_keeps_noise_and_releases_passage(self):
        em = self.manager({"traffic_enabled": False})
        conn = sqlite3.connect(self.db)
        conn.execute(
            "CREATE TRIGGER no_both BEFORE INSERT ON events WHEN NEW.kind = 'both'"
            " BEGIN SELECT RAISE(ABORT, 'both rechazado'); END"
        )
        conn.commit()
        conn.close()
        passage = self.passage(em)
        self.trigger(em)
        (row,) = self.rows()
        self.assertEqual(row["kind"], "noise")
        self.assertEqual(row["vehicle_class"], "car")
        for key in ("track_id", "direction", "speed_kmh", "detection_score"):
            self.assertIsNone(row[key], key)
        self.assertFalse(passage["claimed"])
        self.assertTrue(Path(row["clip_path"]).exists())

    def test_memory_passage_retry_insert_failure_does_not_raise(self):
        em = self.manager({"traffic_enabled": False})
        conn = sqlite3.connect(self.db)
        conn.execute(
            "CREATE TRIGGER no_events BEFORE INSERT ON events"
            " BEGIN SELECT RAISE(ABORT, 'base bloqueada'); END"
        )
        conn.commit()
        conn.close()
        passage = self.passage(em)
        self.trigger(em)  # no debe lanzar: mataría el worker
        self.assertEqual(self.rows(), [])
        self.assertFalse(passage["claimed"])
        self.assertEqual(len(self.clip_calls), 1)
        self.assertEqual(list(em.clips_dir.glob("event_*.mp4")), [], "clip huérfano")
        self.assertEqual(em.status()["events_lost"], 1)


if __name__ == "__main__":
    unittest.main()
