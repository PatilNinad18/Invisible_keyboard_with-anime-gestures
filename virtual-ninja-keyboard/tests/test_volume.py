"""Volume clamping and gesture -> command mapping (no real system volume touched)."""

import pytest

from controls.volume_controller import InMemoryVolumeBackend, VolumeController, clamp_percent
from vision.gesture_classifier import Gesture


@pytest.mark.parametrize("value,expected", [(-10, 0), (0, 0), (49.6, 50), (100, 100), (140, 100)])
def test_clamp_percent(value, expected):
    assert clamp_percent(value) == expected


def make(scalar=0.5, muted=False):
    backend = InMemoryVolumeBackend(scalar, muted)
    return VolumeController(backend, step_percent=5), backend


def test_up_down_steps_of_five():
    ctrl, backend = make(0.50)
    assert ctrl.handle_gesture(Gesture.VOLUME_UP).percent == 55
    assert backend.scalar == pytest.approx(0.55)
    assert ctrl.handle_gesture(Gesture.VOLUME_DOWN).percent == 50
    assert ctrl.handle_gesture(Gesture.VOLUME_DOWN).message == "VOLUME DOWN"


def test_clamped_at_limits():
    ctrl, backend = make(0.98)
    r = ctrl.handle_gesture(Gesture.VOLUME_UP)
    assert (r.percent, r.message) == (100, "VOLUME MAX")
    assert backend.scalar == pytest.approx(1.0)
    ctrl, backend = make(0.03)
    r = ctrl.handle_gesture(Gesture.VOLUME_DOWN)
    assert (r.percent, r.message) == (0, "VOLUME MIN")
    assert backend.scalar == pytest.approx(0.0)


def test_mute_toggles():
    ctrl, backend = make()
    assert ctrl.handle_gesture(Gesture.MUTE_TOGGLE).message == "MUTED"
    assert backend.muted is True
    assert ctrl.handle_gesture(Gesture.MUTE_TOGGLE).message == "UNMUTED"
    assert backend.muted is False


def test_volume_change_unmutes_when_configured():
    backend = InMemoryVolumeBackend(0.5, True)
    VolumeController(backend, 5, unmute_on_change=True).handle_gesture(Gesture.VOLUME_UP)
    assert backend.muted is False
    backend = InMemoryVolumeBackend(0.5, True)
    VolumeController(backend, 5, unmute_on_change=False).handle_gesture(Gesture.VOLUME_UP)
    assert backend.muted is True


@pytest.mark.parametrize("gesture", [Gesture.NONE, Gesture.FIST, Gesture.POINTING, Gesture.UNKNOWN])
def test_non_volume_gestures_do_nothing(gesture):
    ctrl, backend = make()
    assert ctrl.handle_gesture(gesture) is None
    assert backend.scalar == 0.5 and backend.muted is False
