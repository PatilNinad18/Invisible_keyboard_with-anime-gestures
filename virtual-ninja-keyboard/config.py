"""Central configuration for Virtual Ninja Keyboard.

Every threshold, colour, size and cooldown lives here so you can tune the
application without touching the logic.  This file is plain data: it does not
import OpenCV, MediaPipe or anything Windows-specific.

Colours are BGR tuples (OpenCV order), not RGB.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, Tuple

Color = Tuple[int, int, int]


class AppMode(Enum):
    """The two top-level modes of the application."""

    TYPING = "TYPING"
    NINJA = "NINJA CONTROL"


# --------------------------------------------------------------------------
# Camera and window
# --------------------------------------------------------------------------
CAMERA_INDEX: int = 0
CAMERA_WIDTH: int = 1280          # lower to 640 / 480 if your laptop is slow
CAMERA_HEIGHT: int = 720
CAMERA_FPS: int = 30
MIRROR_VIEW: bool = True          # flip the image so movement feels natural
WINDOW_NAME: str = "Virtual Ninja Keyboard"
WINDOW_TOPMOST: bool = True       # keep the window above other apps
MAX_CONSECUTIVE_READ_FAILURES: int = 60

# --------------------------------------------------------------------------
# Hand tracking (MediaPipe)
# --------------------------------------------------------------------------
MAX_NUM_HANDS: int = 2
MODEL_COMPLEXITY: int = 0         # 0 = fastest, 1 = more accurate
MIN_DETECTION_CONFIDENCE: float = 0.7
MIN_TRACKING_CONFIDENCE: float = 0.6
PROCESS_WIDTH: int = 640          # frames are downscaled to this width for inference
TRACK_LOSS_GRACE_FRAMES: int = 8  # keep smoothing state this long after losing a hand

# Adaptive smoothing: slow movement is smoothed strongly (low alpha) and fast
# movement follows closely (high alpha).  Speed reference is a fraction of the
# frame width: movement of that many pixels per frame counts as "fast".
SMOOTH_ALPHA_MIN: float = 0.30
SMOOTH_ALPHA_MAX: float = 0.90
SMOOTH_SPEED_REF_FRACTION: float = 0.04

# Gesture recognition needs every landmark; a half-visible hand is unreliable.
GESTURE_REQUIRES_FULL_HAND: bool = True

# --------------------------------------------------------------------------
# Finger-state and gesture definitions
# --------------------------------------------------------------------------
# A finger counts as "extended" when the angle at its PIP joint (between the
# bones towards the knuckle and towards the tip) is at least this many degrees
# AND the fingertip is farther from the wrist than the PIP joint is.
FINGER_EXTENDED_ANGLE_DEG: float = 150.0
# Thumb: angle at the IP joint plus how far the tip is from the index knuckle,
# measured in "palm sizes" so it works at any distance from the camera.
THUMB_EXTENDED_ANGLE_DEG: float = 150.0
THUMB_OUT_MIN_RATIO: float = 0.45

# Gesture -> required state of (index, middle, ring, pinky).  True = extended.
# The thumb is deliberately ignored: it is the least reliable finger to detect
# from a webcam and ninja seals use it inconsistently.
GESTURE_PATTERNS: Dict[str, Tuple[bool, bool, bool, bool]] = {
    "VOLUME_UP": (True, True, False, False),     # two-finger seal (index + middle)
    "VOLUME_DOWN": (True, False, False, True),   # "horns" seal (index + pinky)
    "MUTE_TOGGLE": (True, True, True, True),     # open palm, four fingers up
    "POINTING": (True, False, False, False),     # informational only
    "FIST": (False, False, False, False),        # informational only
}

GESTURE_DESCRIPTIONS: Dict[str, str] = {
    "VOLUME_UP": "Index + middle up",
    "VOLUME_DOWN": "Index + pinky up",
    "MUTE_TOGGLE": "Open palm",
}

# --------------------------------------------------------------------------
# Typing interaction
# --------------------------------------------------------------------------
# Pinch ratio = (thumb tip <-> index tip distance) / hand size.  Using hand size
# makes it independent of how far you sit from the camera.
PINCH_ON_RATIO: float = 0.30      # at or below this the pinch "closes"
PINCH_OFF_RATIO: float = 0.50     # must rise above this before the next pinch
PINCH_COOLDOWN_S: float = 0.35    # minimum time between two accepted pinches
HOVER_MIN_DWELL_S: float = 0.08   # a key must be hovered this long to be "stable"
HOVER_LOOKBACK_S: float = 0.25    # the stable key stays latched this long after
                                  # the fingertip drifts (pinching moves the tip)
KEY_FLASH_S: float = 0.30         # duration of the "selected" flash
MAX_TEXT_LENGTH: int = 1000
SEND_KEYS_TO_OS_DEFAULT: bool = False   # also type into the focused app (F3)

# --------------------------------------------------------------------------
# Ninja gestures / volume
# --------------------------------------------------------------------------
GESTURE_STABLE_FRAMES: int = 6        # consecutive frames needed to accept
GESTURE_COOLDOWN_S: float = 1.0       # minimum time between accepted gestures
VOLUME_STEP_PERCENT: int = 5
UNMUTE_ON_VOLUME_CHANGE: bool = True  # volume up/down while muted unmutes
VOLUME_REFRESH_INTERVAL_S: float = 0.5
NOTIFICATION_DURATION_S: float = 1.3
MODE_SWITCH_GRACE_S: float = 0.8      # input ignored right after switching modes

# --------------------------------------------------------------------------
# Layout (fractions of the frame size)
# --------------------------------------------------------------------------
STATUS_HEIGHT_FRACTION: float = 0.09
KEYBOARD_REGION: Tuple[float, float, float, float] = (0.03, 0.15, 0.97, 0.75)  # x0, y0, x1, y1
TEXT_PANEL_REGION: Tuple[float, float, float, float] = (0.03, 0.78, 0.97, 0.97)
KEY_GAP_FRACTION: float = 0.08        # gap between keys, as a fraction of one key unit
KEY_MAX_ASPECT: float = 1.3           # key height <= width * this (keeps keys tidy)
KEY_CORNER_FRACTION: float = 0.22     # corner radius as a fraction of key height
KEY_UNITS_PER_ROW: float = 15.0

# --------------------------------------------------------------------------
# Colours (BGR) and drawing
# --------------------------------------------------------------------------
FONT_FACE: int = 0                    # cv2.FONT_HERSHEY_SIMPLEX
COLOR_KEYBOARD_BG: Color = (140, 20, 110)
COLOR_KEY_FILL: Color = (190, 40, 160)
COLOR_KEY_FILL_SPECIAL: Color = (150, 30, 120)
COLOR_KEY_FILL_TOGGLED: Color = (60, 200, 90)
COLOR_KEY_OUTLINE: Color = (70, 255, 130)       # lime
COLOR_KEY_LABEL: Color = (255, 255, 255)
COLOR_HOVER_FILL: Color = (110, 255, 170)
COLOR_PRESS_FLASH: Color = (255, 255, 255)
KEYBOARD_BG_ALPHA: float = 0.28
KEY_FILL_ALPHA: float = 0.42
HOVER_ALPHA: float = 0.55

COLOR_PANEL: Color = (40, 25, 40)
COLOR_PANEL_ALPHA: float = 0.65
COLOR_TEXT: Color = (255, 255, 255)
COLOR_ACCENT: Color = (70, 255, 130)
COLOR_WARNING: Color = (60, 160, 255)
COLOR_DANGER: Color = (80, 80, 255)
COLOR_BUTTON: Color = (90, 50, 90)
COLOR_BUTTON_ACTIVE: Color = (60, 200, 90)
COLOR_HAND_BONE: Color = (70, 255, 130)
COLOR_HAND_JOINT: Color = (255, 255, 255)
COLOR_POINTER: Color = (255, 255, 0)
