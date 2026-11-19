#!/usr/bin/env python3
"""
CentroidTracker: emparejamiento, filtros de pasaje y estimación de velocidad.

El emparejamiento fijaba para cada objeto su detección más cercana antes de
saber si seguía libre: con dos objetos sobre la misma detección, el segundo se
daba por perdido y esa detección se registraba como vehículo nuevo (un pasaje
partido en dos). Además `timestamp or time.time()` convertía un 0.0 legítimo
en "ahora".
"""

import sys
import unittest
from pathlib import Path
from queue import Empty, Queue

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import detection  # noqa: E402
from src.vehicle_tracker import CentroidTracker  # noqa: E402


def make_tracker(**kwargs) -> tuple[CentroidTracker, Queue]:
    queue: Queue = Queue()
    defaults = dict(
        max_distance=80.0,
        max_disappeared=2,
        min_track_seconds=0.0,
        min_displacement=0.0,
        max_track_seconds=60.0,
        passage_queue=queue,
    )
    defaults.update(kwargs)
    return CentroidTracker(**defaults), queue


class TrackerMatchingTest(unittest.TestCase):
    def test_timestamp_zero_is_respected(self):
        tracker, _ = make_tracker()
        tracker.update([detection(cx=100, cy=100)], timestamp=0.0)
        obj = next(iter(tracker._objects.values()))
        self.assertEqual(obj["first_seen"], 0.0)

    def test_none_timestamp_falls_back_to_clock(self):
        tracker, _ = make_tracker()
        tracker.update([detection(cx=100, cy=100)], timestamp=None)
        obj = next(iter(tracker._objects.values()))
        self.assertGreater(obj["first_seen"], 0.0)

    def test_second_object_takes_the_remaining_detection(self):
        # Objeto A en (100,100) y B en (110,100). La detección nueva en (104,100)
        # es la más cercana para AMBOS; la otra está en (300,100).
        tracker, _ = make_tracker(max_distance=250.0)
        tracker.update(
            [detection(cx=100, cy=100), detection(cx=110, cy=100)], timestamp=10.0
        )
        self.assertEqual(tracker.active_count, 2)

        tracker.update(
            [detection(cx=104, cy=100), detection(cx=300, cy=100)], timestamp=11.0
        )

        # Nadie se da por perdido y no se registra un vehículo extra.
        self.assertEqual(tracker.active_count, 2)
        self.assertEqual(tracker._next_id, 2)
        self.assertEqual(sorted(tracker._disappeared.values()), [0, 0])
        centroids = sorted(round(obj["centroid"][0]) for obj in tracker._objects.values())
        self.assertEqual(centroids, [104, 300])

    def test_detection_beyond_max_distance_is_not_matched(self):
        tracker, _ = make_tracker(max_distance=50.0)
        tracker.update([detection(cx=100, cy=100)], timestamp=10.0)
        tracker.update([detection(cx=400, cy=100)], timestamp=11.0)
        self.assertEqual(tracker.active_count, 2)  # el viejo sigue, el nuevo entra
        self.assertEqual(tracker._disappeared[0], 1)

    def test_non_vehicle_classes_are_ignored(self):
        tracker, _ = make_tracker()
        tracker.update(
            [detection(cls="person", cx=10, cy=10), detection(cls="dog", cx=20, cy=20)],
            timestamp=1.0,
        )
        self.assertEqual(tracker.active_count, 0)

    def test_object_keeps_its_identity_while_moving(self):
        tracker, _ = make_tracker(max_distance=100.0)
        for i in range(5):
            tracker.update([detection(cx=100 + i * 20, cy=100)], timestamp=float(i))
        self.assertEqual(tracker.active_count, 1)
        self.assertEqual(tracker._next_id, 1)


class TrackerPassageTest(unittest.TestCase):
    def _run_passage(self, tracker: CentroidTracker, queue: Queue, **kwargs):
        # Entrada en x=100 y salida en x=500 en 3 s (4 frames).
        tracker.update([detection(cx=100, cy=100)], timestamp=0.0)
        tracker.update([detection(cx=240, cy=100)], timestamp=1.0)
        tracker.update([detection(cx=380, cy=100)], timestamp=2.0)
        tracker.update([detection(cx=500, cy=100)], timestamp=3.0)
        # Sin detecciones durante max_disappeared+1 frames → pasaje cerrado.
        for _ in range(4):
            tracker.update([], timestamp=3.5)
        return queue.get(timeout=1)

    def test_passage_is_emitted_with_direction(self):
        tracker, queue = make_tracker(max_distance=200.0, min_displacement=60.0)
        passage = self._run_passage(tracker, queue)
        self.assertEqual(passage["vehicle_class"], "car")
        self.assertEqual(passage["direction"], "ltr")
        self.assertAlmostEqual(passage["timestamp_enter"], 0.0)
        self.assertAlmostEqual(passage["timestamp_exit"], 3.0)
        self.assertIsNone(passage["speed_kmh"])  # sin px_per_m no se inventa

    def test_speed_uses_horizontal_scale(self):
        tracker, queue = make_tracker(
            max_distance=200.0, min_displacement=60.0, px_per_m=10.0
        )
        passage = self._run_passage(tracker, queue)
        # 400 px / 10 px/m = 40 m en 3 s → 13.3 m/s → 48 km/h
        self.assertAlmostEqual(passage["speed_kmh"], 48.0, delta=0.5)

    def test_stationary_object_is_not_a_passage(self):
        tracker, queue = make_tracker(max_distance=50.0, min_displacement=60.0)
        for i in range(4):
            tracker.update([detection(cx=100, cy=100)], timestamp=float(i))
        for _ in range(4):
            tracker.update([], timestamp=4.0)
        with self.assertRaises(Empty):
            queue.get(timeout=0.05)

    def test_short_track_is_discarded(self):
        tracker, queue = make_tracker(max_distance=200.0, min_track_seconds=2.0)
        tracker.update([detection(cx=100, cy=100)], timestamp=0.0)
        tracker.update([detection(cx=400, cy=100)], timestamp=0.5)
        for _ in range(4):
            tracker.update([], timestamp=1.0)
        with self.assertRaises(Empty):
            queue.get(timeout=0.05)

    def test_endless_track_is_discarded(self):
        tracker, queue = make_tracker(
            max_distance=2000.0, max_track_seconds=5.0, min_displacement=0.0
        )
        for i in range(8):  # 0..7 s: más largo que max_track_seconds
            tracker.update([detection(cx=100 + i * 50, cy=100)], timestamp=float(i))
        for _ in range(4):
            tracker.update([], timestamp=8.0)
        with self.assertRaises(Empty):
            queue.get(timeout=0.05)

    def test_absurd_speed_is_not_recorded(self):
        # px_per_m enorme → velocidad fuera de rango: se guarda None, no 900 km/h.
        tracker, queue = make_tracker(
            max_distance=200.0, min_displacement=60.0, px_per_m=10000.0
        )
        passage = self._run_passage(tracker, queue)
        self.assertIsNone(passage["speed_kmh"])

    def test_reverse_direction(self):
        tracker, queue = make_tracker(max_distance=200.0, min_displacement=60.0)
        for ts, cx in ((0.0, 500), (1.0, 380), (2.0, 240), (3.0, 100)):
            tracker.update([detection(cx=cx, cy=100)], timestamp=ts)
        for _ in range(4):
            tracker.update([], timestamp=3.5)
        self.assertEqual(queue.get(timeout=1)["direction"], "rtl")


if __name__ == "__main__":
    unittest.main()
