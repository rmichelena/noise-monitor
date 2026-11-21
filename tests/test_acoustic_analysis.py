import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import wavfile

from src.acoustic_analysis import ClipAnalyzer, infer_source, read_clip
from src.analysis_evaluation import classification_report
from src.analysis_store import (
    LEGACY, SCHEMA, UNIFIED, event_table, finish_attempt, latest_analysis,
    pending_events, start_attempt, visual_context,
)
from src.event_manager import SPEECH_CLASSES


def labels(*names):
    return [{"label": name, "score": .8} for name in names]


class AcousticMetricsTests(unittest.TestCase):
    def test_physical_levels_at_reference_frequency(self):
        analyzer = ClipAnalyzer.__new__(ClipAnalyzer)
        rate = 48000
        # A quarter second bypasses MoSQITo and still exercises the Fast response.
        time = np.arange(rate // 4) / rate
        pressure = .02 * np.sqrt(2) * np.sin(2 * np.pi * 1000 * time)
        result = {"audio": {}, "psychoacoustics": {}, "issues": {}}
        analyzer._physical_metrics(pressure, rate, result)
        self.assertAlmostEqual(result["audio"]["laeq_dba"], 60, places=1)
        self.assertAlmostEqual(result["audio"]["lafmax_dba"], 60 + 10 * np.log10(1 - np.exp(-2)), places=1)

    def test_pcm16_full_scale_is_identified_after_decoding(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "full-scale.wav"
            wavfile.write(path, 44100, np.array([-32767, 32767], dtype=np.int16))
            samples, _, _, digital_limit = read_clip(path)
        self.assertLess(np.abs(samples).max(), 1)
        self.assertEqual(np.abs(samples).max(), digital_limit)

    def test_source_needs_specific_agreement_without_competition(self):
        self.assertEqual(infer_source(labels("Car"), ["car"], False)["source_attribution"], "uncertain")
        self.assertEqual(infer_source(labels("Vehicle"), ["car"], True)["source_attribution"], "uncertain")
        self.assertEqual(infer_source(labels("Car", "Motorcycle"), ["car"], True)["source_attribution"], "uncertain")
        self.assertEqual(infer_source(labels("Car"), ["car", "van"], True)["source_attribution"], "uncertain")
        self.assertEqual(infer_source(labels("Car", "Chainsaw"), ["car"], True)["source_attribution"], "uncertain")
        self.assertEqual(infer_source(labels("Car passing by"), ["car"], True)["vehicle_source"], "car")

    def test_indoor_and_horn_conflict(self):
        dishes = "Dishes, pots, and pans"
        self.assertEqual(infer_source(labels(dishes), [])["source_attribution"], "indoor")
        mixed = infer_source(labels(dishes, "Vehicle horn, car horn, honking"), [])
        self.assertEqual(mixed["source_attribution"], "uncertain")
        self.assertIsNone(mixed["vehicle_source"])
        self.assertIn("vehicular_horn", mixed["acoustic_behaviors"])

    def test_full_and_old_yamnet_speech_labels_mute(self):
        self.assertTrue({"Child speech", "Child speech, kid speaking",
                         "Narration", "Narration, monologue"} <= SPEECH_CLASSES)

    def test_precision_and_recall_use_expected_denominators(self):
        report = classification_report([("indoor", "indoor"), ("indoor", "uncertain"), ("outdoor", "indoor")])
        self.assertEqual(report["metrics"]["indoor"], {
            "precision": .5, "recall": .5, "false_positives": 1, "support": 2,
        })
        self.assertIsNone(report["metrics"]["outdoor"]["precision"])


class AnalysisStoreTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(":memory:")
        self.addCleanup(self.conn.close)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.cutoff = "2026-11-22T00:00:00"

    def load_unified(self):
        self.conn.executescript("""
            CREATE TABLE noise_events(id INTEGER PRIMARY KEY);
            CREATE TABLE events(
                id INTEGER PRIMARY KEY, kind TEXT, start_ts REAL, end_ts REAL, track_id INTEGER,
                vehicle_class TEXT, timestamp TEXT, clip_path TEXT, audio_muted INTEGER
            );
            INSERT INTO events VALUES (1, 'noise', 1, 3, NULL, NULL, '2026-11-20T12:00:00', 'old.mp4', 0);
        """)

    def test_unified_replacement_never_displays_old_result(self):
        self.load_unified()
        self.assertEqual(event_table(self.conn), UNIFIED)
        self.assertEqual(event_table(self.conn, LEGACY), LEGACY)
        event, = pending_events(self.conn, UNIFIED, "profile", "", self.cutoff, 1)
        job = start_attempt(self.conn, UNIFIED, event, "profile", "base")
        self.assertEqual(latest_analysis(self.conn, 1)["status"], "pending")
        finish_attempt(self.conn, job, {"status": "completed", "elapsed_ms": 3.0, "clip_sha256": "old"})
        self.assertEqual(latest_analysis(self.conn, 1)["clip_sha256"], "old")

        self.conn.execute("UPDATE events SET kind='both', end_ts=4, clip_path='new.mp4'")
        self.assertEqual(latest_analysis(self.conn, 1)["status"], "not_run")
        event, = pending_events(self.conn, UNIFIED, "profile", "", self.cutoff, 1)
        self.assertEqual(event["clip_path"], "new.mp4")

        self.conn.execute("UPDATE events SET clip_path=''")
        self.assertEqual(latest_analysis(self.conn, 1)["status"], "unavailable")

    def test_retry_backoff_and_pending_restart(self):
        self.load_unified()
        event, = pending_events(self.conn, UNIFIED, "p", "", self.cutoff, 1)
        job = start_attempt(self.conn, UNIFIED, event, "p", "base")
        self.assertEqual(len(pending_events(self.conn, UNIFIED, "p", "", self.cutoff, 1)), 1)
        finish_attempt(self.conn, job, {"status": "failed"})
        self.assertEqual(pending_events(self.conn, UNIFIED, "p", "", self.cutoff, 1), [])
        self.conn.execute("UPDATE acoustic_analysis SET retry_after=0")
        self.assertEqual(len(pending_events(self.conn, UNIFIED, "p", "", self.cutoff, 1)), 1)
        self.conn.execute("UPDATE acoustic_analysis SET attempts=3")
        self.assertEqual(pending_events(self.conn, UNIFIED, "p", "", self.cutoff, 1), [])
        self.assertEqual(len(pending_events(self.conn, UNIFIED, "new", "", self.cutoff, 1)), 1)

    def test_legacy_visually_linked_passage_with_competing_vehicle(self):
        self.conn.executescript("""
            CREATE TABLE events(id INTEGER PRIMARY KEY, timestamp TEXT);
            CREATE TABLE noise_events(id INTEGER PRIMARY KEY, clip_path TEXT, audio_muted INTEGER,
                                      timestamp TEXT, passage_id INTEGER);
            CREATE TABLE vehicle_passages(id INTEGER PRIMARY KEY, vehicle_class TEXT,
                                          timestamp TEXT, timestamp_exit TEXT);
            INSERT INTO noise_events VALUES (1, 'event.mp4', 0, '2026-11-20T12:00:00', 5);
            INSERT INTO vehicle_passages VALUES (5, 'car', '2026-11-20T11:59:56', '2026-11-20T12:00:00');
        """)
        self.assertEqual(event_table(self.conn), LEGACY)
        event, = pending_events(self.conn, LEGACY, "p", "", self.cutoff, 1)
        self.assertEqual(visual_context(self.conn, LEGACY, event, 5, 5), (["car"], True))
        self.conn.execute(
            "INSERT INTO vehicle_passages VALUES (6, 'car', '2026-11-20T12:00:00', '2026-11-20T12:00:03')"
        )
        self.assertEqual(visual_context(self.conn, LEGACY, event, 5, 5), (["car"], False))


if __name__ == "__main__":
    unittest.main()
