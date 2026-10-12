"""Synthetic hand landmarks for tests (no webcam or MediaPipe needed).

Coordinates are in "hand units" (hand size ~ 1) using image axes: x to the
right, y DOWN, so fingers pointing up have negative y.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from vision.landmarks import HandData, INDEX_TIP

_MCP = {
    "index": (-0.30, -1.00),
    "middle": (0.00, -1.05),
    "ring": (0.28, -1.00),
    "pinky": (0.52, -0.90),
}
_BASE_IDS = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}

#: Tip of the extended index finger in hand units.
INDEX_TIP_UNIT = (-0.30, -2.00)


def make_points(index=False, middle=False, ring=False, pinky=False, thumb=False,
                pinch=False, scale=100.0, tip_at: Tuple[float, float] = (400.0, 300.0)) -> np.ndarray:
    """Build a (21, 3) landmark array. ``tip_at`` is where the INDEX fingertip lands
    when the index finger is extended."""
    pts = np.zeros((21, 3), np.float32)
    pts[0] = (0.0, 0.0, 0.0)
    pts[1] = (-0.35, -0.25, 0.0)
    if thumb:
        pts[2], pts[3], pts[4] = (-0.60, -0.50, 0), (-0.80, -0.75, 0), (-0.95, -0.95, 0)
    else:
        pts[2], pts[3], pts[4] = (-0.60, -0.50, 0), (-0.50, -0.70, 0), (-0.25, -0.75, 0)

    for name, extended in (("index", index), ("middle", middle), ("ring", ring), ("pinky", pinky)):
        mx, my = _MCP[name]
        base = _BASE_IDS[name]
        pts[base] = (mx, my, 0)
        if extended:
            pts[base + 1] = (mx, my - 0.45, 0)
            pts[base + 2] = (mx, my - 0.75, 0)
            pts[base + 3] = (mx, my - 1.00, 0)
        else:
            pts[base + 1] = (mx, my - 0.35, 0)
            pts[base + 2] = (mx + 0.20, my - 0.35, 0)
            pts[base + 3] = (mx + 0.20, my - 0.10, 0)

    if pinch:
        pts[4] = pts[8] + np.array([0.02, 0.02, 0.0], np.float32)
        pts[3] = (pts[4] + pts[2]) / 2.0

    pts *= scale
    # Move so the index fingertip (extended) sits at tip_at.
    origin = np.array(tip_at, np.float32) - scale * np.array(INDEX_TIP_UNIT, np.float32)
    pts[:, 0] += origin[0]
    pts[:, 1] += origin[1]
    return pts


def make_hand(label="Right", **kwargs) -> HandData:
    """Same as :func:`make_points` but wrapped in :class:`HandData`."""
    pts = make_points(**kwargs)
    return HandData(label=label, score=0.99, points=pts, fully_in_frame=True)


# Convenient named poses -----------------------------------------------------
POSE_VOLUME_UP = dict(index=True, middle=True)
POSE_VOLUME_DOWN = dict(index=True, pinky=True)
POSE_MUTE = dict(index=True, middle=True, ring=True, pinky=True, thumb=True)
POSE_FIST = dict()
POSE_POINT = dict(index=True)
assert INDEX_TIP == 8
