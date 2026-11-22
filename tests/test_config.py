#!/usr/bin/env python3
"""
Validación de config: los errores deben nombrar la clave, no reventar como
KeyError dentro de un thread. Además, los dos ejemplos versionados tienen que
pasar la validación (el de modo simulado antes ni siquiera traía `audio`).
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import example_config  # noqa: E402
from src.utils import ConfigError, init_database, validate_config  # noqa: E402


class ValidateConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = example_config(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_example_config_is_valid(self):
        self.assertIs(validate_config(self.cfg), self.cfg)

    def test_simulate_example_is_valid(self):
        cfg = example_config(self.tmp.name, "config.simulate.yaml.example")
        validate_config(cfg)
        self.assertEqual(cfg["audio"]["source"], "file")

    def test_simulate_example_keeps_the_sections_the_pipeline_reads(self):
        cfg = example_config(self.tmp.name, "config.simulate.yaml.example")
        for section in ("audio", "triggers", "buffer", "paths", "video"):
            self.assertIn(section, cfg)

    def test_missing_section(self):
        del self.cfg["triggers"]
        with self.assertRaises(ConfigError) as ctx:
            validate_config(self.cfg)
        self.assertIn("triggers", str(ctx.exception))

    def test_missing_key_names_the_path(self):
        del self.cfg["buffer"]["post_event_seconds"]
        with self.assertRaises(ConfigError) as ctx:
            validate_config(self.cfg)
        self.assertIn("buffer.post_event_seconds", str(ctx.exception))

    def test_missing_video_section_is_rejected(self):
        del self.cfg["video"]
        with self.assertRaises(ConfigError) as ctx:
            validate_config(self.cfg)
        self.assertIn("video", str(ctx.exception))

    def test_bad_video_resolution_is_rejected(self):
        self.cfg["video"]["resolution"] = [1280]
        with self.assertRaises(ConfigError) as ctx:
            validate_config(self.cfg)
        self.assertIn("video.resolution", str(ctx.exception))

    def test_impossible_detection_threshold_is_rejected(self):
        self.cfg["video"]["detection_threshold"] = 1.5
        with self.assertRaises(ConfigError) as ctx:
            validate_config(self.cfg)
        self.assertIn("video.detection_threshold", str(ctx.exception))

    def test_nan_and_inf_are_rejected(self):
        for value in (float("nan"), float("inf")):
            self.cfg["triggers"]["peak_db"] = value
            with self.assertRaises(ConfigError):
                validate_config(self.cfg)

    def test_audio_file_requires_existing_path(self):
        self.cfg["audio"]["source"] = "file"
        self.cfg["audio"]["file_path"] = "no-existe.wav"
        with self.assertRaises(ConfigError) as ctx:
            validate_config(self.cfg, base_dir=Path(self.tmp.name))
        self.assertIn("audio.file_path", str(ctx.exception))

    def test_invalid_tracker_value_is_rejected(self):
        self.cfg.setdefault("tracker", {})["max_distance"] = -1
        with self.assertRaises(ConfigError) as ctx:
            validate_config(self.cfg)
        self.assertIn("tracker.max_distance", str(ctx.exception))

    def test_init_database_enables_wal(self):
        import sqlite3
        db = init_database(self.cfg, PROJECT_DIR)
        conn = sqlite3.connect(db)
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()
        self.assertEqual(mode.lower(), "wal")

    def test_wrong_type_is_rejected(self):
        self.cfg["audio"]["sample_rate"] = "44100"
        with self.assertRaises(ConfigError) as ctx:
            validate_config(self.cfg)
        self.assertIn("audio.sample_rate", str(ctx.exception))

    def test_bool_is_not_a_valid_number(self):
        self.cfg["buffer"]["pre_event_seconds"] = True
        with self.assertRaises(ConfigError):
            validate_config(self.cfg)

    def test_impossible_sample_rate(self):
        self.cfg["audio"]["sample_rate"] = 4000
        with self.assertRaises(ConfigError) as ctx:
            validate_config(self.cfg)
        self.assertIn(">= 8000", str(ctx.exception))

    def test_zero_block_size(self):
        self.cfg["audio"]["block_size"] = 0
        with self.assertRaises(ConfigError):
            validate_config(self.cfg)

    def test_empty_path_is_rejected(self):
        self.cfg["paths"]["clips"] = "   "
        with self.assertRaises(ConfigError) as ctx:
            validate_config(self.cfg)
        self.assertIn("paths.clips", str(ctx.exception))

    def test_not_a_dict(self):
        with self.assertRaises(ConfigError):
            validate_config(["audio"])

    def test_warns_on_inverted_triggers(self):
        self.cfg["triggers"]["sustained_db"] = 90
        self.cfg["triggers"]["peak_db"] = 85
        with self.assertLogs("src.utils", level="WARNING") as logs:
            validate_config(self.cfg)
        self.assertTrue(any("sustained_db" in line for line in logs.output))

    def test_warns_when_speed_has_no_reference(self):
        self.cfg["speed"] = {"enabled": True, "reference_distance_m": 10.0}
        with self.assertLogs("src.utils", level="WARNING") as logs:
            validate_config(self.cfg)
        self.assertTrue(any("reference_left_x" in line for line in logs.output))

    def test_no_warning_when_speed_disabled(self):
        self.cfg["speed"] = {"enabled": False}
        with self.assertNoLogs("src.utils", level="WARNING"):
            validate_config(self.cfg)

    def test_warns_when_clip_window_exceeds_buffer(self):
        self.cfg["buffer"]["audio_seconds"] = 5
        with self.assertLogs("src.utils", level="WARNING") as logs:
            validate_config(self.cfg)
        self.assertTrue(any("pre+post" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
