"""
Centroid-based vehicle tracker.
Matches detections across consecutive inference frames by pixel-space centroid distance.
When a tracked object disappears for N frames, emits a passage_complete event.
"""

import logging
import time
from collections import OrderedDict
from queue import Queue

import numpy as np

logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


class CentroidTracker:
    """
    Tracks vehicles across frames using pixel-space centroid distance.

    Detections must include 'cx' and 'cy' keys (pixel coordinates).
    Only VEHICLE_CLASSES are tracked; other detections are ignored.

    When an object is not seen for max_disappeared consecutive inference
    frames, it is deregistered and a passage dict is put on the queue.
    """

    def __init__(
        self,
        max_distance: float = 80.0,
        max_disappeared: int = 15,
        min_track_seconds: float = 2.0,
        min_displacement: float = 60.0,
        passage_queue: Queue | None = None,
    ):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.min_track_seconds = min_track_seconds
        self.min_displacement = min_displacement
        self.passage_queue = passage_queue or Queue()

        self._next_id = 0
        self._objects: OrderedDict[int, dict] = OrderedDict()
        self._disappeared: dict[int, int] = {}

    def _register(self, detection: dict, timestamp: float):
        cx, cy = detection["cx"], detection["cy"]
        obj = {
            "class": detection["class"],
            "centroid": (cx, cy),
            "start_centroid": (cx, cy),
            "score": detection["score"],
            "lane": detection["lane"],
            "first_seen": timestamp,
            "last_seen": timestamp,
            "scores": [detection["score"]],
        }
        self._objects[self._next_id] = obj
        self._disappeared[self._next_id] = 0
        self._next_id += 1

    def _deregister(self, object_id: int):
        obj = self._objects.pop(object_id)
        del self._disappeared[object_id]

        duration = obj["last_seen"] - obj["first_seen"]
        if duration < self.min_track_seconds:
            return

        sx, sy = obj["start_centroid"]
        ex, ey = obj["centroid"]
        displacement = np.hypot(ex - sx, ey - sy)
        if displacement < self.min_displacement:
            return  # stationary — parked vehicle or persistent false positive

        # Sentido real de circulación: la etiqueta 'lane' solo indica en qué
        # mitad del cuadro estaba el centroide, no por dónde iba el vehículo.
        direction = "ltr" if ex > sx else "rtl"

        avg_score = float(np.mean(obj["scores"])) if obj["scores"] else 0.0
        self.passage_queue.put({
            "vehicle_class": obj["class"],
            "lane": obj["lane"],
            "direction": direction,
            "detection_score": round(avg_score, 3),
            "timestamp_enter": obj["first_seen"],
            "timestamp_exit": obj["last_seen"],
        })

    def update(self, detections: list[dict], timestamp: float | None = None):
        """
        Update tracker with new detections from a single inference frame.
        Detections must include: class, score, cx, cy, lane.
        Only vehicles are tracked.
        """
        ts = timestamp or time.time()

        vehicles = [d for d in detections if d.get("class") in VEHICLE_CLASSES]

        if not vehicles:
            for oid in list(self._disappeared.keys()):
                self._disappeared[oid] += 1
                if self._disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)
            return

        if not self._objects:
            for det in vehicles:
                self._register(det, ts)
            return

        object_ids = list(self._objects.keys())
        object_centroids = np.array(
            [self._objects[oid]["centroid"] for oid in object_ids]
        )
        input_centroids = np.array([(d["cx"], d["cy"]) for d in vehicles])

        # Pairwise euclidean distances in pixel space
        dists = np.linalg.norm(
            object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        # Greedy matching: closest pairs first
        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if dists[row, col] > self.max_distance:
                continue

            oid = object_ids[row]
            det = vehicles[col]
            self._objects[oid]["centroid"] = (det["cx"], det["cy"])
            self._objects[oid]["score"] = det["score"]
            self._objects[oid]["lane"] = det["lane"]
            self._objects[oid]["last_seen"] = ts
            self._objects[oid]["scores"].append(det["score"])
            self._disappeared[oid] = 0

            used_rows.add(row)
            used_cols.add(col)

        for row in range(len(object_ids)):
            if row not in used_rows:
                oid = object_ids[row]
                self._disappeared[oid] += 1
                if self._disappeared[oid] > self.max_disappeared:
                    self._deregister(oid)

        for col in range(len(vehicles)):
            if col not in used_cols:
                self._register(vehicles[col], ts)

    @property
    def active_count(self) -> int:
        return len(self._objects)
