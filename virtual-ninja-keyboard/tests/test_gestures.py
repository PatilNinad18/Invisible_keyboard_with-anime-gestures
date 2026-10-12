"""Gesture classification and stabilisation."""

import pytest

import config
from tests import helpers as H
from tests.helpers import make_points
from vision.gesture_classifier import (
    Gesture,
    GestureStabilizer,
    classify_gesture,
    finger_states,
)


@pytest.mark.parametrize("pose,expected", [
    (H.POSE_VOLUME_UP, Gesture.VOLUME_UP),
    (H.POSE_VOLUME_DOWN, Gesture.VOLUME_DOWN),
    (H.POSE_MUTE, Gesture.MUTE_TOGGLE),
    (H.POSE_FIST, Gesture.FIST),
    (H.POSE_POINT, Gesture.POINTING),
])
def test_poses_are_classified(pose, expected):
    assert classify_gesture(make_points(**pose)).gesture == expected


@pytest.mark.parametrize("thumb", [True, False])
def test_thumb_does_not_change_command_gestures(thumb):
    assert classify_gesture(make_points(index=True, middle=True, thumb=thumb)).gesture == Gesture.VOLUME_UP
    assert classify_gesture(make_points(index=True, pinky=True, thumb=thumb)).gesture == Gesture.VOLUME_DOWN


def test_undefined_pose_is_unknown():
    assert classify_gesture(make_points(middle=True)).gesture == Gesture.UNKNOWN
    assert classify_gesture(make_points(index=True, middle=True, ring=True)).gesture == Gesture.UNKNOWN


def test_classification_is_scale_and_position_invariant():
    for scale, tip in ((40, (50, 60)), (100, (400, 300)), (300, (900, 500))):
        pts = make_points(index=True, middle=True, scale=scale, tip_at=tip)
        assert classify_gesture(pts).gesture == Gesture.VOLUME_UP


def test_finger_states_and_angles():
    s = finger_states(make_points(index=True, ring=True, thumb=True))
    assert (s.thumb, s.index, s.middle, s.ring, s.pinky) == (True, True, False, True, False)
    assert s.angles["index"] > 170          # straight
    assert s.angles["middle"] < 90          # folded


def test_custom_patterns_are_respected():
    patterns = {"VOLUME_UP": (False, False, False, True)}
    assert classify_gesture(make_points(pinky=True), patterns).gesture == Gesture.VOLUME_UP
    assert classify_gesture(make_points(index=True, middle=True), patterns).gesture == Gesture.UNKNOWN


def test_config_patterns_are_distinct():
    patterns = list(config.GESTURE_PATTERNS.values())
    assert len(patterns) == len(set(patterns))


# ---- stabiliser -----------------------------------------------------------
def test_needs_consecutive_frames():
    s = GestureStabilizer(required_frames=4, cooldown_s=0.0)
    results = [s.update(Gesture.VOLUME_UP, i * 0.03) for i in range(4)]
    assert results == [None, None, None, Gesture.VOLUME_UP]


def test_interrupted_gesture_restarts_count():
    s = GestureStabilizer(required_frames=3, cooldown_s=0.0)
    s.update(Gesture.VOLUME_UP, 0.0)
    s.update(Gesture.VOLUME_UP, 0.1)
    s.update(Gesture.UNKNOWN, 0.2)
    assert s.update(Gesture.VOLUME_UP, 0.3) is None
    assert s.update(Gesture.VOLUME_UP, 0.4) is None
    assert s.update(Gesture.VOLUME_UP, 0.5) is Gesture.VOLUME_UP


def test_holding_fires_only_once_until_released():
    s = GestureStabilizer(required_frames=2, cooldown_s=0.0)
    fired = [s.update(Gesture.VOLUME_DOWN, i * 0.1) for i in range(30)]
    assert fired.count(Gesture.VOLUME_DOWN) == 1
    s.update(Gesture.NONE, 3.1)
    fired = [s.update(Gesture.VOLUME_DOWN, 3.2 + i * 0.1) for i in range(5)]
    assert fired.count(Gesture.VOLUME_DOWN) == 1


def test_changing_gesture_allows_new_command():
    s = GestureStabilizer(required_frames=2, cooldown_s=0.0)
    for i in range(3):
        s.update(Gesture.VOLUME_UP, i * 0.1)
    out = [s.update(Gesture.MUTE_TOGGLE, 0.5 + i * 0.1) for i in range(3)]
    assert Gesture.MUTE_TOGGLE in out


def test_cooldown_delays_acceptance():
    s = GestureStabilizer(required_frames=1, cooldown_s=1.0)
    assert s.update(Gesture.VOLUME_UP, 0.0) is Gesture.VOLUME_UP
    s.update(Gesture.NONE, 0.1)
    assert s.update(Gesture.VOLUME_UP, 0.2) is None      # inside cooldown
    s.update(Gesture.NONE, 0.3)
    assert s.update(Gesture.VOLUME_UP, 1.5) is Gesture.VOLUME_UP


def test_non_command_gestures_never_fire():
    s = GestureStabilizer(required_frames=1, cooldown_s=0.0)
    for g in (Gesture.POINTING, Gesture.FIST, Gesture.UNKNOWN, Gesture.NONE):
        assert s.update(g, 0.0) is None


def test_reset_clears_state():
    s = GestureStabilizer(required_frames=2, cooldown_s=0.0)
    s.update(Gesture.VOLUME_UP, 0.0)
    s.reset()
    assert s.update(Gesture.VOLUME_UP, 0.1) is None
