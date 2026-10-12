"""Pure gesture logic: finger states, gesture classification, pinch and stabilising.

Nothing in this module touches the webcam, OpenCV or Windows, so it is fully
unit-testable.  All inputs are ``(21, 3)`` NumPy arrays of hand landmarks.

How the ninja gestures are defined
----------------------------------
The pose of each non-thumb finger is reduced to "extended" or "folded" using
joint angles and fingertip distances (see :func:`finger_states`).  A gesture is
simply a required pattern of (index, middle, ring, pinky) states, defined in
``config.GESTURE_PATTERNS``.  The thumb is measured for the calibration view but
ignored by the patterns because it is the least reliable finger on a webcam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

import config
from vision.landmarks import (
    FINGER_JOINTS,
    INDEX_MCP,
    INDEX_TIP,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    distance,
    hand_size,
    joint_angle_deg,
)


class Gesture(Enum):
    """Hand poses the classifier can report."""

    NONE = "NONE"                  # no hand / hand unusable
    UNKNOWN = "UNKNOWN"            # hand present, no defined pose matches
    POINTING = "POINTING"
    FIST = "FIST"
    VOLUME_UP = "VOLUME_UP"
    VOLUME_DOWN = "VOLUME_DOWN"
    MUTE_TOGGLE = "MUTE_TOGGLE"


#: Gestures that trigger an action in Ninja Control mode.
COMMAND_GESTURES: Tuple[Gesture, ...] = (
    Gesture.VOLUME_UP,
    Gesture.VOLUME_DOWN,
    Gesture.MUTE_TOGGLE,
)

FINGER_NAMES: Tuple[str, ...] = ("index", "middle", "ring", "pinky")


@dataclass(frozen=True)
class FingerStates:
    """Extended/folded state for each finger, plus the measurements behind it."""

    thumb: bool
    index: bool
    middle: bool
    ring: bool
    pinky: bool
    angles: Mapping[str, float] = field(default_factory=dict)

    @property
    def pattern(self) -> Tuple[bool, bool, bool, bool]:
        """(index, middle, ring, pinky) states, the part used to match gestures."""
        return (self.index, self.middle, self.ring, self.pinky)


@dataclass(frozen=True)
class GestureResult:
    """Everything the classifier knows about one hand in one frame."""

    gesture: Gesture
    states: FingerStates
    pinch_ratio: float


# --------------------------------------------------------------------------
# Finger states
# --------------------------------------------------------------------------
def finger_states(
    points: np.ndarray,
    finger_angle_deg: float = config.FINGER_EXTENDED_ANGLE_DEG,
    thumb_angle_deg: float = config.THUMB_EXTENDED_ANGLE_DEG,
    thumb_out_ratio: float = config.THUMB_OUT_MIN_RATIO,
) -> FingerStates:
    """Decide which fingers are extended.

    Finger: PIP angle >= threshold AND tip is farther from the wrist than the PIP.
    Thumb: IP angle >= threshold AND tip is far enough from the index knuckle
    (measured in hand sizes).
    """
    wrist = points[WRIST]
    angles: Dict[str, float] = {}
    extended: Dict[str, bool] = {}

    for name, (mcp, pip, _dip, tip) in FINGER_JOINTS.items():
        angle = joint_angle_deg(points[mcp], points[pip], points[tip])
        tip_farther = distance(points[tip], wrist) > distance(points[pip], wrist)
        angles[name] = angle
        extended[name] = angle >= finger_angle_deg and tip_farther

    size = max(hand_size(points), 1e-6)
    thumb_angle = joint_angle_deg(points[THUMB_MCP], points[THUMB_IP], points[THUMB_TIP])
    thumb_out = distance(points[THUMB_TIP], points[INDEX_MCP]) / size
    angles["thumb"] = thumb_angle

    return FingerStates(
        thumb=thumb_angle >= thumb_angle_deg and thumb_out >= thumb_out_ratio,
        index=extended["index"],
        middle=extended["middle"],
        ring=extended["ring"],
        pinky=extended["pinky"],
        angles=angles,
    )


def pinch_ratio(points: np.ndarray) -> float:
    """Thumb-tip to index-tip distance divided by hand size.

    Small values mean the fingertips are touching.  Dividing by hand size makes
    the number independent of how close the hand is to the camera.
    """
    size = hand_size(points)
    if size < 1e-6:
        return float("inf")
    return distance(points[THUMB_TIP], points[INDEX_TIP]) / size


def classify_gesture(
    points: np.ndarray,
    patterns: Optional[Mapping[str, Tuple[bool, bool, bool, bool]]] = None,
) -> GestureResult:
    """Classify one hand into a :class:`Gesture` using finger-state patterns."""
    patterns = config.GESTURE_PATTERNS if patterns is None else patterns
    states = finger_states(points)
    ratio = pinch_ratio(points)

    lookup: Dict[Tuple[bool, bool, bool, bool], Gesture] = {}
    for name, pattern in patterns.items():
        gesture = Gesture[name]
        lookup.setdefault(tuple(pattern), gesture)  # type: ignore[arg-type]

    return GestureResult(
        gesture=lookup.get(states.pattern, Gesture.UNKNOWN),
        states=states,
        pinch_ratio=ratio,
    )


# --------------------------------------------------------------------------
# Pinch detection with hysteresis
# --------------------------------------------------------------------------
class PinchDetector:
    """Turns a stream of pinch ratios into single "pinch happened" events.

    * Hysteresis: the pinch closes at ``on_ratio`` and only reopens above the
      larger ``off_ratio`` so noise near the threshold cannot flicker.
    * Release-to-rearm: after an event the hand must open fully (ratio above
      ``off_ratio``) before another event can fire, so holding a pinch types once.
    * Cooldown: events are at least ``cooldown_s`` apart.
    * A hand that appears already pinched never fires; it must open first.
    """

    def __init__(
        self,
        on_ratio: float = config.PINCH_ON_RATIO,
        off_ratio: float = config.PINCH_OFF_RATIO,
        cooldown_s: float = config.PINCH_COOLDOWN_S,
    ) -> None:
        if on_ratio >= off_ratio:
            raise ValueError("on_ratio must be smaller than off_ratio")
        self.on_ratio = on_ratio
        self.off_ratio = off_ratio
        self.cooldown_s = cooldown_s
        self.is_pinched = False
        self._armed = False
        self._last_event = -1e9

    def reset(self) -> None:
        """Call when the hand disappears or input is suspended."""
        self.is_pinched = False
        self._armed = False

    def progress(self, ratio: Optional[float]) -> float:
        """0.0 (open) .. 1.0 (closed), for drawing a pinch indicator."""
        if ratio is None:
            return 0.0
        span = self.off_ratio - self.on_ratio
        return float(np.clip((self.off_ratio - ratio) / span, 0.0, 1.0))

    def update(self, ratio: Optional[float], now: float) -> bool:
        """Feed one frame. Returns ``True`` exactly when a pinch event fires."""
        if ratio is None:
            self.reset()
            return False

        if self.is_pinched:
            if ratio >= self.off_ratio:
                self.is_pinched = False
                self._armed = True
            return False

        if ratio >= self.off_ratio:
            self._armed = True
            return False

        if ratio <= self.on_ratio:
            self.is_pinched = True
            fire = self._armed and (now - self._last_event) >= self.cooldown_s
            self._armed = False
            if fire:
                self._last_event = now
            return fire
        return False


# --------------------------------------------------------------------------
# Gesture stabiliser (debounce for ninja commands)
# --------------------------------------------------------------------------
class GestureStabilizer:
    """Accepts a command gesture only after it is stable, once per hold.

    * The same gesture must be seen for ``required_frames`` consecutive frames.
    * After a command fires, the gesture must change (or disappear) before the
      same command can fire again.
    * ``cooldown_s`` separates any two accepted commands.
    """

    def __init__(
        self,
        required_frames: int = config.GESTURE_STABLE_FRAMES,
        cooldown_s: float = config.GESTURE_COOLDOWN_S,
        commands: Tuple[Gesture, ...] = COMMAND_GESTURES,
    ) -> None:
        if required_frames < 1:
            raise ValueError("required_frames must be >= 1")
        self.required_frames = required_frames
        self.cooldown_s = cooldown_s
        self.commands = commands
        self._candidate: Optional[Gesture] = None
        self._count = 0
        self._locked: Optional[Gesture] = None
        self._last_fire = -1e9

    def reset(self) -> None:
        """Forget everything (used when switching modes)."""
        self._candidate = None
        self._count = 0
        self._locked = None

    @property
    def progress(self) -> float:
        """0..1 progress towards accepting the current candidate."""
        if self._candidate is None or self._candidate == self._locked:
            return 0.0
        return min(1.0, self._count / self.required_frames)

    @property
    def candidate(self) -> Optional[Gesture]:
        """Gesture currently being confirmed, if any."""
        return self._candidate

    def update(self, gesture: Gesture, now: float) -> Optional[Gesture]:
        """Feed one frame. Returns the accepted command gesture or ``None``."""
        if gesture not in self.commands:
            self._candidate = None
            self._count = 0
            self._locked = None  # released: next command may fire
            return None

        if gesture != self._candidate:
            self._candidate = gesture
            self._count = 1
            if gesture != self._locked:
                self._locked = None
        else:
            self._count += 1

        if (
            self._count >= self.required_frames
            and gesture != self._locked
            and (now - self._last_fire) >= self.cooldown_s
        ):
            self._locked = gesture
            self._last_fire = now
            return gesture
        return None
