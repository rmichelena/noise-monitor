#!/usr/bin/env python3
"""
VideoMonitor sin cámara ni Coral.

- El módulo importaba `ai_edge_litert` en la primera línea: cualquier script o
  test en un host sin el runtime del TPU moría al importar.
- `run_loop` llamaba a `process_frame` sin pausa: si la cámara dejaba de
  entregar frames, el thread quedaba girando al 100 % de CPU sin un solo aviso.
"""

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import example_config  # noqa: E402
from src import video_monitor  # noqa: E402
from src.video_monitor import VideoMonitor  # noqa: E402


class ImportWithoutAcceleratorTest(unittest.TestCase):
    def test_module_imports_without_the_tpu_runtime(self):
        # En este host no está instalado ai-edge-litert: el import no debe fallar.
        self.assertTrue(hasattr(video_monitor, "VideoMonitor"))

    def test_interpreter_start_is_refused_cleanly(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        monitor = VideoMonitor(config=example_config(tmp.name), base_dir=PROJECT_DIR)
        self.assertIsNone(monitor._interpreter)
        with self.assertLogs("src.video_monitor", level="ERROR"):
            self.assertFalse(monitor._init_interpreter())


class SpeedScaleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = example_config(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_scale_from_reference_coordinates(self):
        self.cfg["speed"] = {
            "enabled": True,
            "reference_left_x": 300,
            "reference_right_x": 900,
            "reference_distance_m": 10.0,
        }
        monitor = VideoMonitor(config=self.cfg, base_dir=PROJECT_DIR)
        self.assertAlmostEqual(monitor._tracker.px_per_m, 60.0)

    def test_missing_reference_warns_instead_of_silence(self):
        self.cfg["speed"] = {"enabled": True, "reference_distance_m": 10.0}
        with self.assertLogs("src.video_monitor", level="WARNING") as logs:
            monitor = VideoMonitor(config=self.cfg, base_dir=PROJECT_DIR)
        self.assertIsNone(monitor._tracker.px_per_m)
        self.assertTrue(any("reference_left_x" in line for line in logs.output))

    def test_disabled_speed_does_not_warn(self):
        self.cfg["speed"] = {"enabled": False}
        with self.assertNoLogs("src.video_monitor", level="WARNING"):
            monitor = VideoMonitor(config=self.cfg, base_dir=PROJECT_DIR)
        self.assertIsNone(monitor._tracker.px_per_m)


class RunLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = example_config(self.tmp.name)
        self.monitor = VideoMonitor(config=self.cfg, base_dir=PROJECT_DIR)

    def tearDown(self):
        self.tmp.cleanup()

    def test_loop_reopens_instead_of_dying_when_camera_stays_down(self):
        calls = []
        opens = []
        self.monitor._running = True
        self.monitor._cap = object()  # type: ignore[assignment]

        def failing() -> bool:
            calls.append(1)
            return False

        def open_capture() -> bool:
            opens.append(1)
            self.monitor._running = False  # el test corta el loop
            return False

        self.monitor.process_frame = failing  # type: ignore[method-assign]
        self.monitor._open_capture = open_capture  # type: ignore[method-assign]

        with self.assertLogs("src.video_monitor", level="ERROR") as logs:
            self.monitor.run_loop(max_failures=3, backoff=0.01)

        self.assertEqual(len(calls), 3)
        self.assertEqual(len(opens), 1)
        self.assertTrue(any("reabriendo" in line for line in logs.output))

    def test_loop_resumes_after_reopen(self):
        calls = []
        opens = []
        state = {"fail": True}
        self.monitor._running = True
        self.monitor._cap = object()  # type: ignore[assignment]

        def process() -> bool:
            calls.append(1)
            if state["fail"]:
                return False
            if len(calls) >= 6:
                self.monitor._running = False
            return True

        def open_capture() -> bool:
            opens.append(1)
            state["fail"] = False
            return True

        self.monitor.process_frame = process  # type: ignore[method-assign]
        self.monitor._open_capture = open_capture  # type: ignore[method-assign]
        self.monitor.run_loop(max_failures=2, backoff=0.01)

        self.assertEqual(len(opens), 1)
        self.assertGreaterEqual(len(calls), 6)

    def test_loop_keeps_going_while_frames_arrive(self):
        calls = []

        def working() -> bool:
            calls.append(1)
            if len(calls) >= 5:
                self.monitor._running = False
            return True

        self.monitor._running = True
        self.monitor._cap = object()  # type: ignore[assignment]
        self.monitor.process_frame = working  # type: ignore[method-assign]
        self.monitor.run_loop(max_failures=3, backoff=0.01)
        self.assertEqual(len(calls), 5)

    def test_file_source_without_loop_stops_at_the_end(self):
        self.monitor._is_file_source = True
        self.monitor._loop_video = False
        self.monitor._running = True
        self.monitor._cap = object()  # type: ignore[assignment]
        self.monitor.process_frame = lambda: False  # type: ignore[method-assign]
        with self.assertLogs("src.video_monitor", level="INFO"):
            self.monitor.run_loop(max_failures=3, backoff=0.01)
        self.assertFalse(self.monitor._running)


if __name__ == "__main__":
    unittest.main()
