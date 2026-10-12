"""Pinch detection logic."""

import pytest

from tests.helpers import make_points
from vision.gesture_classifier import PinchDetector, pinch_ratio


def test_ratio_small_when_pinched_and_large_when_open():
    assert pinch_ratio(make_points(index=True, pinch=True)) < 0.15
    assert pinch_ratio(make_points(index=True, thumb=True)) > 0.8


def test_ratio_is_scale_invariant():
    near = pinch_ratio(make_points(index=True, thumb=True, scale=300))
    far = pinch_ratio(make_points(index=True, thumb=True, scale=60))
    assert near == pytest.approx(far, rel=1e-3)


def test_fires_once_per_pinch():
    d = PinchDetector(0.3, 0.5, 0.0)
    assert d.update(0.9, 0.0) is False       # open: arms
    assert d.update(0.2, 0.1) is True        # close: fires
    for i in range(20):                      # holding does not repeat
        assert d.update(0.1, 0.2 + i * 0.1) is False


def test_release_to_rearm():
    d = PinchDetector(0.3, 0.5, 0.0)
    d.update(0.9, 0.0)
    assert d.update(0.2, 0.1)
    d.update(0.4, 0.2)                       # between thresholds: still pinched
    assert d.is_pinched
    assert d.update(0.2, 0.3) is False       # no release, no new event
    d.update(0.8, 0.4)                       # release
    assert not d.is_pinched
    assert d.update(0.2, 0.5) is True


def test_hysteresis_band_does_not_flicker():
    d = PinchDetector(0.3, 0.5, 0.0)
    d.update(0.9, 0.0)
    d.update(0.25, 0.1)
    for ratio in (0.35, 0.45, 0.31, 0.49):
        d.update(ratio, 0.2)
        assert d.is_pinched


def test_cooldown_blocks_fast_second_event():
    d = PinchDetector(0.3, 0.5, 0.5)
    d.update(0.9, 0.0)
    assert d.update(0.2, 1.0)
    d.update(0.9, 1.1)
    assert d.update(0.2, 1.2) is False       # only 0.2 s later
    d.update(0.9, 1.3)
    assert d.update(0.2, 1.6) is True


def test_pinch_already_closed_when_hand_appears_never_fires():
    d = PinchDetector(0.3, 0.5, 0.0)
    assert d.update(0.1, 0.0) is False
    assert d.update(0.1, 0.1) is False
    d.update(0.9, 0.2)
    assert d.update(0.1, 0.3) is True


def test_tracking_loss_resets():
    d = PinchDetector(0.3, 0.5, 0.0)
    d.update(0.9, 0.0)
    d.update(None, 0.1)
    assert d.update(0.1, 0.2) is False       # must open again first


def test_progress_values():
    d = PinchDetector(0.3, 0.5, 0.0)
    assert d.progress(0.9) == 0.0
    assert d.progress(0.2) == 1.0
    assert d.progress(0.4) == pytest.approx(0.5)
    assert d.progress(None) == 0.0


def test_invalid_thresholds():
    with pytest.raises(ValueError):
        PinchDetector(0.6, 0.5)
