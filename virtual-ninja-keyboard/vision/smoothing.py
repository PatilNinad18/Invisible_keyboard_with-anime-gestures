"""Adaptive landmark smoothing to reduce webcam jitter."""

from __future__ import annotations

from typing import Optional

import numpy as np


class LandmarkSmoother:
    """Exponential moving average whose strength depends on movement speed.

    When a landmark barely moves, ``alpha_min`` smooths it heavily (kills jitter).
    When it moves quickly, alpha rises towards ``alpha_max`` so the cursor does
    not lag behind the finger.
    """

    def __init__(self, alpha_min: float, alpha_max: float) -> None:
        if not 0.0 < alpha_min <= alpha_max <= 1.0:
            raise ValueError("Need 0 < alpha_min <= alpha_max <= 1")
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._state: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Forget the previous position (next update starts fresh)."""
        self._state = None

    def update(self, points: np.ndarray, speed_ref_px: float) -> np.ndarray:
        """Blend new ``(N, 3)`` points into the smoothed state and return a copy."""
        points = np.asarray(points, dtype=np.float32)
        if self._state is None or self._state.shape != points.shape:
            self._state = points.copy()
            return self._state.copy()

        delta = np.linalg.norm(points[:, :2] - self._state[:, :2], axis=1)
        speed = np.clip(delta / max(speed_ref_px, 1e-6), 0.0, 1.0)
        alpha = (self.alpha_min + (self.alpha_max - self.alpha_min) * speed)[:, None]
        self._state = self._state + alpha.astype(np.float32) * (points - self._state)
        return self._state.copy()
