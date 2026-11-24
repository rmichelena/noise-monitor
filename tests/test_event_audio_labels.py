import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.audio_classifier import AudioClassifier
from src.audio_monitor import AudioMonitor
from src.event_manager import EventManager
from src.utils import init_database, load_config

sys.path.insert(0, str(Path(__file__).resolve().parent))
from support import create_events_table  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def classifier_with_scores(labels, transform):
    classifier = AudioClassifier.__new__(AudioClassifier)
    classifier.labels = labels
    classifier.min_score = .3
    classifier._class_to_category = {"car": "vehicle", "motorcycle": "vehicle"}
    classifier.predict_scores = Mock(side_effect=transform)
    return classifier


class WindowRankingTests(unittest.TestCase):
    def test_three_windows_straddle_trigger_and_preserve_subthreshold_scores(self):
        labels = ["Silence", "Car", "Motorcycle", "Dog", "Chainsaw"]
        predictions = iter([
            [.99, .1, .12, .23, .1],
            [.93, .55, .08, .17, .01],
            [.55, .12, .61, .15, .05],
        ])
        classifier = classifier_with_scores(labels, lambda *_: np.array(next(predictions)))
        pcm = np.arange(48000, dtype=np.float32)
        ranked, speech = classifier.classify_event(pcm, 16000, 16000, scan_all_speech=False)
        windows = [call.args[0] for call in classifier.predict_scores.call_args_list]
        self.assertEqual([len(w) for w in windows], [15600] * 3)
        self.assertEqual([w[0] for w in windows], [520, 8200, 15880])
        self.assertEqual([label["label"] for label in ranked], ["Motorcycle", "Car", "Dog"])
        self.assertEqual([label["passes_threshold"] for label in ranked], [True, True, False])
        self.assertFalse(speech)

    def test_full_speech_check_includes_audio_outside_source_window(self):
        def predict(window, _rate):
            if window[0] == 1:
                return np.array([.8, .9, .92])
            return np.array([.65, .0, .2])

        classifier = classifier_with_scores(["Car", "Speech", "Silence"], predict)
        pcm = np.zeros(128000, dtype=np.float32)
        pcm[-16000:] = 1
        ranked, speech = classifier.classify_event(pcm, 16000, 48000, scan_all_speech=True)
        self.assertEqual(ranked[0]["label"], "Car")
        self.assertTrue(speech)

    def test_short_audio_is_scored_once(self):
        classifier = classifier_with_scores(["Car", "Silence"], lambda *_: np.array([.2, .9]))
        ranked, _ = classifier.classify_event(np.zeros(8000), 16000, 0, scan_all_speech=False)
        classifier.predict_scores.assert_called_once()
        self.assertEqual(ranked[0]["label"], "Car")
        self.assertFalse(ranked[0]["passes_threshold"])


class CaptureRankingTests(unittest.TestCase):
    def test_dba_event_consumes_post_trigger_audio_and_persists_top_three(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "events.db"
            config = load_config(ROOT / "config.example.yaml")
            config["paths"].update(database=str(db_path), clips=tmp)
            config["audio"].update(sample_rate=16000, block_size=1600)
            config["triggers"]["cooldown_seconds"] = 30
            # The only confident Car score is attributable to audio after the trigger.
            classifier = classifier_with_scores(
                ["Silence", "Car", "Dog", "Motorcycle"],
                lambda x, _: np.array([.99, .86 if np.any(x > .1) else .04, .32, .07]),
            )
            audio = AudioMonitor(config=config, base_dir=ROOT, audio_classifier=classifier)
            stamp = time.time()
            audio.audio_buffer.extend((stamp - i / 10, np.zeros(1600)) for i in range(9, -1, -1))
            video = SimpleNamespace(detections_between=lambda *args: [], recent_detections=[], video_buffer=[])
            manager = EventManager(audio, video, config=config, base_dir=ROOT)
            manager._save_clip = Mock(return_value=("event.mp4", 1000))
            create_events_table(db_path)

            def capture_postroll(_):
                audio.audio_buffer.clear()
                audio.audio_buffer.extend((stamp + i / 10, np.ones(1600)) for i in range(1, 11))

            with patch("src.event_manager.time.sleep", side_effect=capture_postroll):
                manager._process_trigger({"type": "peak", "db": 90, "timestamp": stamp})
                manager._process_trigger({"type": "peak", "db": 90, "timestamp": stamp + .1})

            self.assertEqual(classifier.predict_scores.call_count, 3)
            with sqlite3.connect(db_path) as conn:
                row, = conn.execute("SELECT sound_class,sound_category,sound_labels_json FROM events")
            self.assertEqual(row[:2], ("Car", "vehicle"))
            self.assertEqual([tag["label"] for tag in json.loads(row[2])], ["Car", "Dog", "Motorcycle"])

    def test_old_database_gets_nullable_label_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "events.db"
            config = load_config(ROOT / "config.example.yaml")
            config["paths"]["database"] = str(db_path)
            with sqlite3.connect(db_path) as conn:
                conn.execute("CREATE TABLE noise_events(id INTEGER PRIMARY KEY, sound_class TEXT)")
                conn.execute("INSERT INTO noise_events VALUES(1, 'Dog')")
            init_database(config, ROOT)
            with sqlite3.connect(db_path) as conn:
                self.assertEqual(conn.execute("SELECT sound_class,sound_labels_json FROM noise_events").fetchone(),
                                 ("Dog", None))
