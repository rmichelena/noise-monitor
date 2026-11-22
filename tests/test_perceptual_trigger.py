import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.analysis_store import latest_analysis
from src.audio_classifier import AudioClassifier
from src.audio_monitor import AudioMonitor
from src.event_manager import EventManager
from src.perceptual_trigger import PerceptualTrigger, Settings, score_window
from src.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


class PerceptualTriggerTests(unittest.TestCase):
    def config(self, criterion="psychoacoustic"):
        config = load_config(ROOT / "config.example.yaml")
        config["audio"].update(sample_rate=48000, block_size=4800)
        config["audio"]["pressure_calibration"] = {"id": "test-tone", "pa_per_unit": 2.0}
        config["triggers"].update(criterion=criterion, psychoacoustic={"threshold": 4})
        return config

    def test_modes_and_calibration(self):
        config = self.config()
        config["audio"].pop("pressure_calibration")
        with self.assertRaisesRegex(ValueError, "pressure_calibration"):
            Settings.load(config)
        config["triggers"].pop("criterion")
        self.assertIsNone(Settings.load(config))

    def test_rejects_configuration_with_no_time_left_to_score(self):
        config = self.config()
        config["triggers"]["psychoacoustic"]["window_seconds"] = 2
        config["buffer"].update(audio_seconds=2, video_seconds=2, pre_event_seconds=1, post_event_seconds=1)
        with self.assertRaisesRegex(ValueError, "tiempo de cálculo"):
            Settings.load(config)

    def test_live_score_calibrates_input_and_rejects_digital_saturation(self):
        settings = Settings.load(self.config())
        metrics = SimpleNamespace(loudness_zwicker=Mock(return_value=SimpleNamespace(loudness_vs_time=[2, 6])))
        self.assertEqual(score_window(np.array([.25, -.25]), 48000, settings, metrics), 4)
        waveform = metrics.loudness_zwicker.call_args.args[0]
        np.testing.assert_array_equal(waveform, [.5, -.5])
        self.assertIsNone(score_window(np.ones(8), 48000, settings, metrics))

    def test_dba_does_not_override_psychoacoustic_mode(self):
        cases = [("dba", "peak"), ("psychoacoustic", None)]
        for criterion, expected in cases:
            with self.subTest(criterion=criterion):
                monitor = AudioMonitor(config=self.config(criterion), base_dir=ROOT)
                if criterion == "psychoacoustic":
                    monitor._perceptual_trigger = Mock()
                block = .99 * np.sin(2 * np.pi * 1000 * np.arange(4800) / 48000)
                monitor._process_block(block, time.time())
                if expected:
                    self.assertEqual(monitor.event_queue.get_nowait()["type"], expected)
                else:
                    monitor._perceptual_trigger.offer.assert_called_once()
                    with self.assertRaises(Empty):
                        monitor.event_queue.get_nowait()

    def test_pending_window_replaced_without_waiting(self):
        settings = Settings.load(self.config())
        trigger = PerceptualTrigger(settings, 48000, 4800, Mock())
        block = np.zeros(4800)
        for index in range(60):
            trigger.offer(index / 10, block, 40)
        self.assertGreater(trigger.dropped, 0)
        self.assertEqual(trigger._pending.qsize(), 1)
        self.assertGreater(trigger._pending.get_nowait()[-1][0], 4)

    def test_speech_can_mute_clip_even_when_a_vehicle_is_the_top_label(self):
        classifier = AudioClassifier.__new__(AudioClassifier)
        classifier.labels = ["Car", "Speech"]
        classifier.min_score = .3
        classifier._class_to_category = {"car": "vehicle"}
        classifier.predict_scores = Mock(return_value=np.array([.8, .45]))
        kind, category, _, speech = classifier.classify_with_speech(np.zeros(8), 48000)
        self.assertEqual((kind, category, speech), ("Car", "vehicle", True))
        classifier.predict_scores.return_value = np.array([np.nan, .1])
        with self.assertRaisesRegex(ValueError, "YAMNet inválidos"):
            classifier.classify_with_speech(np.zeros(8), 48000)
        classifier.predict_scores.return_value = np.array([.9, .1, .05])
        with self.assertRaisesRegex(ValueError, "YAMNet inválidos"):
            classifier.classify_with_speech(np.zeros(8), 48000)

    def test_preserves_pre_roll_and_screens_post_roll(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config()
            config["paths"].update(database=str(Path(directory) / "test.db"), clips=directory)
            stamp = time.time()
            audio = AudioMonitor(config=config, base_dir=ROOT)
            classifier = Mock()
            classifier.classify_with_speech.side_effect = [("Car", "vehicle", .8, False),
                                                           ("Car", "vehicle", .9, True)]
            audio._classifier = classifier
            audio.audio_buffer.extend([(stamp - 4, np.zeros(4800)), (stamp, np.zeros(42000))])
            video = SimpleNamespace(detections_between=lambda *args: [], video_buffer=[(stamp - 4, None)])
            manager = EventManager(audio, video, config=config, base_dir=ROOT)
            manager._save_clip = Mock(return_value=("event.mp4", 1000))

            def capture_post_roll(_):
                audio.audio_buffer.clear()
                audio.audio_buffer.extend([(stamp, np.zeros(42000)), (stamp + 2, np.ones(44000))])
                video.video_buffer[:] = [(stamp + 2, None)]

            with patch("src.event_manager.time.sleep", side_effect=capture_post_roll):
                manager._process_trigger({"type": "psychoacoustic", "timestamp": stamp, "db": 55.})
            audio_blocks, video_frames, *_ = manager._save_clip.call_args.args
            self.assertEqual([t for t, _ in audio_blocks], [stamp - 4, stamp, stamp + 2])
            self.assertEqual([t for t, _ in video_frames], [stamp - 4, stamp + 2])
            self.assertEqual(classifier.classify_with_speech.call_count, 2)
            self.assertTrue(manager._save_clip.call_args.kwargs["mute_audio"])

    def test_published_low_db_trigger_saves_without_visual_correlation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.config()
            config["buffer"].update(audio_seconds=12, video_seconds=12)
            config["paths"]["database"] = str(Path(directory) / "events.db")
            config["paths"]["clips"] = directory
            audio = AudioMonitor(config=config, base_dir=ROOT)
            video = SimpleNamespace(detections_between=lambda *args: [], video_buffer=[])
            manager = EventManager(audio, video, config=config, base_dir=ROOT)
            manager._save_clip = Mock(return_value=("event.mp4", 1000))
            stamp = time.time() - 6
            audio._perceptual_trigger._emit(
                stamp, 4.5, [(stamp, np.zeros(4), 55.)], np.zeros(4),
            )
            manager._process_trigger(audio.event_queue.get_nowait())
            manager._save_clip.assert_called_once()
            self.assertTrue(manager._save_clip.call_args.kwargs["mute_audio"])
            with sqlite3.connect(config["paths"]["database"]) as conn:
                conn.row_factory = sqlite3.Row
                event = conn.execute("SELECT * FROM noise_events").fetchone()
                result = latest_analysis(conn, event["id"])
            self.assertEqual(event["trigger_type"], "psychoacoustic")
            self.assertEqual(event["db_max"], 55.)
            self.assertIsNone(event["vehicle_class"])
            self.assertEqual(event["audio_muted"], 1)
            self.assertEqual(result["trigger_decision"]["score"], 4.5)
            self.assertEqual(result["trigger_decision"]["threshold"], 4)
