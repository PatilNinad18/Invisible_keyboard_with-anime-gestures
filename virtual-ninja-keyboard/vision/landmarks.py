"""Hand landmark indices, the ``HandData`` container and small geometry helpers.

Everything here is pure NumPy so it can be tested without a webcam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# MediaPipe hand landmark indices (21 points).
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

#: (MCP, PIP, DIP, TIP) landmark indices for each non-thumb finger.
FINGER_JOINTS: Dict[str, Tuple[int, int, int, int]] = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

#: Pairs of landmarks joined by a bone when drawing the skeleton.
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

FINGERTIP_IDS: Tuple[int, ...] = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)


@dataclass
class HandData:
    """One detected hand.

    Attributes:
        label: ``"Left"`` or ``"Right"`` as seen by the user.
        score: MediaPipe handedness confidence (0-1).
        points: ``(21, 3)`` float array. Columns are x and y in camera pixels
            and z in the same pixel scale (negative = closer to the camera).
            Values are already smoothed.
        fully_in_frame: ``True`` if every landmark lies inside the image.
    """

    label: str
    score: float
    points: np.ndarray
    fully_in_frame: bool = True

    @property
    def xy(self) -> np.ndarray:
        """The ``(21, 2)`` pixel coordinates."""
        return self.points[:, :2]

    @property
    def index_tip(self) -> Tuple[float, float]:
        """Pixel position of the index fingertip."""
        p = self.points[INDEX_TIP]
        return float(p[0]), float(p[1])

    @property
    def thumb_tip(self) -> Tuple[float, float]:
        """Pixel position of the thumb tip."""
        p = self.points[THUMB_TIP]
        return float(p[0]), float(p[1])


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points (2-D or 3-D)."""
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at point ``b`` between the segments ``b->a`` and ``b->c``.

    A perfectly straight joint gives 180, a fully folded joint gives 0.
    """
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def hand_size(points: np.ndarray) -> float:
    """A scale reference for the hand, independent of camera distance.

    Average of wrist->middle knuckle and index knuckle->pinky knuckle lengths.
    Averaging two measurements makes it less sensitive to hand rotation.
    """
    length = distance(points[WRIST], points[MIDDLE_MCP])
    width = distance(points[INDEX_MCP], points[PINKY_MCP])
    return 0.5 * (length + width)
