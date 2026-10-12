"""End-to-end test of the app pipeline using a stub tracker (no webcam, no Windows)."""

import numpy as np
import pytest

pytest.importorskip("cv2")

import config
from config import AppMode
from controls.volume_controller import InMemoryVolumeBackend, VolumeController
from keyboard.key_controller import KeyInjector
from keyboard.layout import KeyboardLayout
from main import VirtualNinjaApp
from tests import helpers as H

W, H_ = 1280, 720
DT = 1 / 30


class StubTracker:
    def __init__(self):
        self.hands = []

    def process(self, frame):
        return list(self.hands)


class Driver:
    def __init__(self, mirror=False):
        self.tracker = StubTracker()
        self.backend = InMemoryVolumeBackend(0.5)
        self.app = VirtualNinjaApp(self.tracker, VolumeController(self.backend), KeyInjector(False), mirror=mirror)
        self.t = 10.0
        self.layout = KeyboardLayout(W, H_)

    def frame(self, **hand):
        self.tracker.hands = [H.make_hand(**hand)] if hand is not None else []
        self.t += DT
        return self.app.process_frame(np.zeros((H_, W, 3), np.uint8), self.t)

    def none(self, n=1):
        for _ in range(n):
            self.tracker.hands = []
            self.t += DT
            self.app.process_frame(np.zeros((H_, W, 3), np.uint8), self.t)

    def wait(self, seconds):
        self.t += seconds


def key_center(drv, label):
    return next(k for k in drv.layout.keys if k.spec.label == label).center


def tap_key(drv, label, hover_frames=8):
    pos = key_center(drv, label)
    for _ in range(hover_frames):
        drv.frame(**H.POSE_POINT, thumb=True, tip_at=pos)
    drv.frame(**H.POSE_POINT, pinch=True, tip_at=pos)   # closes the pinch
    for _ in range(4):                                  # keep holding
        drv.frame(**H.POSE_POINT, pinch=True, tip_at=pos)
    for _ in range(3):                                  # release
        drv.frame(**H.POSE_POINT, thumb=True, tip_at=pos)
    drv.wait(0.4)


def test_typing_hello_world_with_shift():
    d = Driver()
    d.app.set_mode(AppMode.TYPING, d.t)
    d.wait(1.0)
    tap_key(d, "SHIFT")
    tap_key(d, "h")
    tap_key(d, "i")
    tap_key(d, "SPACE")
    tap_key(d, "1")
    assert d.app.engine.text == "Hi 1"


def test_holding_pinch_types_once():
    d = Driver()
    pos = key_center(d, "q")
    for _ in range(8):
        d.frame(**H.POSE_POINT, thumb=True, tip_at=pos)
    for _ in range(40):
        d.frame(**H.POSE_POINT, pinch=True, tip_at=pos)
    assert d.app.engine.text == "q"


def test_hover_alone_never_types():
    d = Driver()
    pos = key_center(d, "w")
    for _ in range(60):
        d.frame(**H.POSE_POINT, thumb=True, tip_at=pos)
    assert d.app.engine.text == ""


def test_pinch_over_empty_area_types_nothing():
    d = Driver()
    empty = (640, 98)                                 # between the volume bar and the keyboard
    assert d.layout.key_at(*empty) is None
    for _ in range(8):
        d.frame(**H.POSE_POINT, thumb=True, tip_at=empty)
    for _ in range(6):
        d.frame(**H.POSE_POINT, pinch=True, tip_at=empty)
    assert d.app.engine.text == ""


def test_swooping_across_keys_then_pinching_types_nothing():
    d = Driver()
    xs = np.linspace(200, 1000, 12)
    y = key_center(d, "a")[1]
    for x in xs:
        d.frame(**H.POSE_POINT, thumb=True, tip_at=(x, y))
    d.frame(**H.POSE_POINT, pinch=True, tip_at=(xs[-1], y))
    assert d.app.engine.text == ""


def test_hand_lost_mid_pinch_does_not_type():
    d = Driver()
    pos = key_center(d, "e")
    for _ in range(8):
        d.frame(**H.POSE_POINT, thumb=True, tip_at=pos)
    d.none(3)
    d.frame(**H.POSE_POINT, pinch=True, tip_at=pos)  # reappears already pinched
    assert d.app.engine.text == ""


def test_ninja_volume_up_down_mute_and_hold_once():
    d = Driver()
    d.app.set_mode(AppMode.NINJA, d.t)
    d.wait(1.0)
    for _ in range(30):                               # long hold: exactly one step
        d.frame(**H.POSE_VOLUME_UP)
    assert d.backend.scalar == pytest.approx(0.55)
    d.none(3)
    d.wait(1.1)
    for _ in range(10):
        d.frame(**H.POSE_VOLUME_DOWN)
    assert d.backend.scalar == pytest.approx(0.50)
    d.none(3)
    d.wait(1.1)
    for _ in range(10):
        d.frame(**H.POSE_MUTE)
    assert d.backend.muted is True
    d.none(3)
    d.wait(1.1)
    for _ in range(10):
        d.frame(**H.POSE_MUTE)
    assert d.backend.muted is False
    assert d.app.engine.text == ""                    # no typing happened


def test_volume_gestures_ignored_in_typing_mode():
    d = Driver()
    assert d.app.mode == AppMode.TYPING
    for _ in range(40):
        d.frame(**H.POSE_VOLUME_UP)
    assert d.backend.scalar == pytest.approx(0.5)


def test_typing_pinch_ignored_in_ninja_mode():
    d = Driver()
    d.app.set_mode(AppMode.NINJA, d.t)
    d.wait(1.0)
    pos = key_center(d, "a")
    for _ in range(8):
        d.frame(**H.POSE_POINT, thumb=True, tip_at=pos)
    for _ in range(8):
        d.frame(**H.POSE_POINT, pinch=True, tip_at=pos)
    assert d.app.engine.text == ""
    assert d.backend.scalar == pytest.approx(0.5)


def test_mode_switch_grace_period_blocks_gestures():
    d = Driver()
    d.app.set_mode(AppMode.NINJA, d.t)
    for _ in range(10):                               # within MODE_SWITCH_GRACE_S
        d.frame(**H.POSE_VOLUME_UP)
    assert d.backend.scalar == pytest.approx(0.5)


def test_on_screen_buttons_switch_modes_by_pinch():
    d = Driver()
    from ui.overlay import UILayout
    ui = UILayout(W, H_)
    ninja_btn = next(b for b in ui.buttons if b.button_id == "mode_ninja")
    bx, by, bw, bh = ninja_btn.rect
    pos = (bx + bw / 2, by + bh / 2)
    for _ in range(8):
        d.frame(**H.POSE_POINT, thumb=True, tip_at=pos)
    for _ in range(4):
        d.frame(**H.POSE_POINT, pinch=True, tip_at=pos)
    assert d.app.mode == AppMode.NINJA
    assert d.app.engine.text == ""


def test_keyboard_shortcuts():
    d = Driver()
    assert d.app.handle_key(7365632, d.t) and d.app.mode == AppMode.NINJA   # F2
    assert d.app.handle_key(7340032, d.t) and d.app.mode == AppMode.TYPING  # F1
    assert d.app.handle_key(7391232, d.t) and d.app.injector.enabled        # F3
    assert d.app.handle_key(ord("2"), d.t) and d.app.mode == AppMode.TYPING  # plain keys off while sending
    assert d.app.handle_key(27, d.t) is False                                # ESC


def test_renders_without_hands_and_on_other_resolutions():
    for size in ((640, 480), (1920, 1080), (800, 450)):
        app = VirtualNinjaApp(StubTracker(), None, KeyInjector(False), mirror=True, calibration=True)
        out = app.process_frame(np.full((size[1], size[0], 3), 90, np.uint8), 1.0)
        assert out.shape == (size[1], size[0], 3)
        app.set_mode(AppMode.NINJA, 2.0)
        app.process_frame(np.full((size[1], size[0], 3), 90, np.uint8), 5.0)


def test_volume_unavailable_shows_message_not_crash():
    tracker = StubTracker()
    app = VirtualNinjaApp(tracker, None, KeyInjector(False), mirror=False)
    app.set_mode(AppMode.NINJA, 0.0)
    t = 5.0
    for _ in range(10):
        tracker.hands = [H.make_hand(**H.POSE_VOLUME_UP)]
        t += DT
        app.process_frame(np.zeros((H_, W, 3), np.uint8), t)
    assert app.overlay.notification.text == "VOLUME UNAVAILABLE"
