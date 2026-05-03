"""
Face Tracker – Centroid-based Multi-Person Tracking
====================================================
Assigns persistent IDs to detected faces across video frames by matching
face-region centroids from frame to frame using Euclidean distance.

Dependencies: numpy, scipy
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    """
    Lightweight centroid tracker.

    1. Receive a list of bounding boxes each frame.
    2. Compute centroids.
    3. Match centroids to existing tracked objects via minimum
       Euclidean distance (Hungarian-style greedy assignment).
    4. Register / deregister faces as they appear / disappear.
    """

    def __init__(self, max_disappeared: int = 15) -> None:
        """
        Parameters
        ----------
        max_disappeared : int
            Number of consecutive frames a face can be absent before
            its ID is dropped.
        """
        self._next_id: int = 0
        # id -> centroid (cx, cy)
        self.objects: OrderedDict[int, tuple[int, int]] = OrderedDict()
        # id -> number of consecutive frames the object was missing
        self._disappeared: OrderedDict[int, int] = OrderedDict()
        self.max_disappeared = max_disappeared

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register(self, centroid: tuple[int, int]) -> int:
        """Register a new face, return the assigned ID."""
        obj_id = self._next_id
        self.objects[obj_id] = centroid
        self._disappeared[obj_id] = 0
        self._next_id += 1
        return obj_id

    def _deregister(self, obj_id: int) -> None:
        """Remove a tracked face."""
        del self.objects[obj_id]
        del self._disappeared[obj_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        rects: list[tuple[int, int, int, int]],
    ) -> dict[int, tuple[int, int]]:
        """
        Accept a list of detected face rectangles ``(x, y, w, h)`` for the
        current frame.  Returns a mapping ``{person_id: (cx, cy)}`` of all
        currently active tracked faces **after** the update.

        Parameters
        ----------
        rects : list of (x, y, w, h)
            Bounding boxes from the face detector.

        Returns
        -------
        dict[int, tuple[int, int]]
            Active person IDs mapped to their centroid coordinates.
        """

        # --- No detections this frame ---
        if len(rects) == 0:
            for obj_id in list(self._disappeared.keys()):
                self._disappeared[obj_id] += 1
                if self._disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return dict(self.objects)

        # --- Compute input centroids ---
        input_centroids = np.array([(x + w // 2, y + h // 2) for x, y, w, h in rects])

        # --- No existing objects – register everything ---
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self._register(tuple(centroid))
            return dict(self.objects)

        # --- Match existing objects to new detections ---
        object_ids = list(self.objects.keys())
        object_centroids = np.array(list(self.objects.values()))

        # Pairwise distance matrix: rows=existing, cols=new detections
        D = dist.cdist(object_centroids, input_centroids)

        # Greedy assignment: sort by distance, assign closest first
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            obj_id = object_ids[row]
            self.objects[obj_id] = tuple(input_centroids[col])
            self._disappeared[obj_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (disappeared)
        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            obj_id = object_ids[row]
            self._disappeared[obj_id] += 1
            if self._disappeared[obj_id] > self.max_disappeared:
                self._deregister(obj_id)

        # Handle unmatched new detections (new faces)
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self._register(tuple(input_centroids[col]))

        return dict(self.objects)

    def assign_ids(self) -> dict[int, tuple[int, int]]:
        """Return the current mapping of active person IDs to centroids."""
        return dict(self.objects)

    def reset(self) -> None:
        """Clear all tracked objects and reset the ID counter."""
        self.objects.clear()
        self._disappeared.clear()
        self._next_id = 0


# ---------------------------------------------------------------------------
# Per-person emotion history helper
# ---------------------------------------------------------------------------


def match_rects_to_ids(
    rects: list[tuple[int, int, int, int]],
    tracked_ids: dict[int, tuple[int, int]],
) -> dict[int, tuple[int, int, int, int]]:
    """
    Given the original detection rects and the tracker's ID→centroid map,
    return a mapping of person_id → rect by matching centroids.

    This lets callers know which bounding box belongs to which person ID.
    """
    if not rects or not tracked_ids:
        return {}

    input_centroids = np.array([(x + w // 2, y + h // 2) for x, y, w, h in rects])
    id_list = list(tracked_ids.keys())
    id_centroids = np.array(list(tracked_ids.values()))

    mapping: dict[int, tuple[int, int, int, int]] = {}

    D = dist.cdist(id_centroids, input_centroids)

    rows = D.min(axis=1).argsort()
    cols = D.argmin(axis=1)[rows]

    used_cols: set[int] = set()
    for row, col in zip(rows, cols):
        if col in used_cols:
            continue
        mapping[id_list[row]] = rects[col]
        used_cols.add(col)

    return mapping


def record_emotion(
    history: dict[int, list[tuple[float, str, dict[str, float]]]],
    person_id: int,
    timestamp: float,
    emotion: str,
    scores: dict[str, float],
) -> None:
    """Append an emotion record for a given person to the history dict."""
    if person_id not in history:
        history[person_id] = []
    history[person_id].append((timestamp, emotion, scores))
