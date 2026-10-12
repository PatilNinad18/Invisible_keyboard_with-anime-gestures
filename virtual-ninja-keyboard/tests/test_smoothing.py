"""Landmark smoothing."""

import numpy as np
import pytest

from vision.smoothing import LandmarkSmoother


def pts(x, y):
    p = np.zeros((21, 3), np.float32)
    p[:, 0], p[:, 1] = x, y
    return p


def test_first_sample_passes_through():
    s = LandmarkSmoother(0.3, 0.9)
    out = s.update(pts(100, 200), 50)
    assert out[0, 0] == pytest.approx(100)


def test_jitter_is_reduced():
    rng = np.random.default_rng(0)
    s = LandmarkSmoother(0.25, 0.9)
    raw, smooth = [], []
    for _ in range(200):
        jitter = rng.normal(0, 2.0, 2)
        raw.append(100 + jitter[0])
        smooth.append(s.update(pts(100 + jitter[0], 100 + jitter[1]), 50)[8, 0])
    assert np.std(smooth[20:]) < 0.6 * np.std(raw[20:])


def test_fast_motion_follows_closely():
    s = LandmarkSmoother(0.2, 0.95)
    s.update(pts(0, 0), 20)
    out = s.update(pts(100, 0), 20)             # jump far above the speed reference
    assert out[0, 0] > 90


def test_invalid_alphas():
    with pytest.raises(ValueError):
        LandmarkSmoother(0.9, 0.2)


def test_reset():
    s = LandmarkSmoother(0.3, 0.9)
    s.update(pts(0, 0), 10)
    s.reset()
    assert s.update(pts(500, 500), 10)[0, 0] == pytest.approx(500)
