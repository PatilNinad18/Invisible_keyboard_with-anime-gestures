"""Virtual Ninja Keyboard - entry point.

Run with:  python main.py
Keys:      F1 typing mode | F2 ninja control mode | F3 send-to-app | F4 calibration | ESC quit
Fallback:  1 / 2 / 3 / c  (only while SEND is OFF, because they would otherwise be typed text)
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Protocol, Tuple

try:
    import cv2
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on the user's environment
    sys.exit(f"Missing dependency: {exc.name}. Run:  pip install -r requirements.txt")

import config
from config import AppMode
from controls.volume_controller import (
    InMemoryVolumeBackend,
    VolumeBackendError,
    VolumeController,
    create_real_controller,
)
from keyboard.key_controller import HoverLatch, KeyController, KeyInjector, TypingEngine
from keyboard.layout import Key, KeyboardLayout
from keyboard.renderer import KeyboardRenderer
from ui.overlay import BUTTON_NINJA, BUTTON_SEND, BUTTON_TYPING, Overlay, UILayout
from vision.gesture_classifier import (
    GestureResult,
    Gesture,
    GestureStabilizer,
    PinchDetector,
    classify_gesture,
    pinch_ratio,
)
from vision.hand_tracker import HandTracker, HandTrackerError, draw_hand
from vision.landmarks import HandData

ESC = 27
# cv2.waitKeyEx codes for F1-F4 (Windows HighGUI, then X11/Qt keysyms).
KEY_F1 = (7340032, 65470)
KEY_F2 = (7365632, 65471)
KEY_F3 = (7391232, 65472)
KEY_F4 = (7416832, 65473)


class Tracker(Protocol):
    """What the app needs from a hand tracker (lets tests inject a stub)."""

    def process(self, frame_bgr: np.ndarray) -> List[HandData]: ...


class VirtualNinjaApp:
    """Owns all components and turns one camera frame into one annotated frame."""

    def __init__(
        self,
        tracker: Tracker,
        volume: Optional[VolumeController],
        injector: Optional[KeyInjector] = None,
        mirror: bool = config.MIRROR_VIEW,
        calibration: bool = False,
    ) -> None:
        self.tracker = tracker
        self.volume = volume
        self.injector = injector or KeyInjector()
        self.mirror = mirror
        self.show_calibration = calibration

        self.mode = AppMode.TYPING
        self.engine = TypingEngine()
        self.keys = KeyController(self.engine, self.injector)
        self.button_latch = HoverLatch()
        self.pinch = PinchDetector()
        self.stabilizer = GestureStabilizer()
        self.renderer = KeyboardRenderer()
        self.overlay = Overlay()

        self._layout: Optional[KeyboardLayout] = None
        self._ui: Optional[UILayout] = None
        self._input_block_until = 0.0
        self._pressed: Optional[Key] = None
        self._pressed_at = -1e9
        self._volume_cache: Tuple[Optional[int], bool] = (None, False)
        self._volume_checked = -1e9
        self._fps = 0.0
        self._last_time: Optional[float] = None

    # ------------------------------------------------------------------
    # Mode handling
    # ------------------------------------------------------------------
    def set_mode(self, mode: AppMode, now: float) -> None:
        """Switch mode and suspend input briefly so no gesture leaks across modes."""
        if mode == self.mode:
            return
        self.mode = mode
        self.pinch.reset()
        self.stabilizer.reset()
        self.keys.reset()
        self.button_latch.reset()
        self._input_block_until = now + config.MODE_SWITCH_GRACE_S
        self.overlay.notification.show(f"{mode.value} MODE", now, config.COLOR_ACCENT, 1.0)

    def toggle_send(self, now: float) -> None:
        """Turn "also type into the focused application" on or off."""
        enabled = self.injector.toggle()
        self.overlay.notification.show("SEND TO APP: " + ("ON" if enabled else "OFF"), now,
                                       config.COLOR_ACCENT if enabled else config.COLOR_WARNING, 1.0)

    def handle_key(self, code: int, now: float) -> bool:
        """Handle a keyboard shortcut. Returns ``False`` when the app should quit."""
        if code == ESC:
            return False
        if code in KEY_F1:
            self.set_mode(AppMode.TYPING, now)
        elif code in KEY_F2:
            self.set_mode(AppMode.NINJA, now)
        elif code in KEY_F3:
            self.toggle_send(now)
        elif code in KEY_F4:
            self.show_calibration = not self.show_calibration
        elif not self.injector.enabled:  # plain keys would collide with injected text
            if code == ord("1"):
                self.set_mode(AppMode.TYPING, now)
            elif code == ord("2"):
                self.set_mode(AppMode.NINJA, now)
            elif code == ord("3"):
                self.toggle_send(now)
            elif code in (ord("c"), ord("C")):
                self.show_calibration = not self.show_calibration
        return True

    # ------------------------------------------------------------------
    # Volume helpers
    # ------------------------------------------------------------------
    def _current_volume(self, now: float, force: bool = False) -> Tuple[Optional[int], bool]:
        if self.volume is None:
            return None, False
        if force or now - self._volume_checked >= config.VOLUME_REFRESH_INTERVAL_S:
            try:
                self._volume_cache = (self.volume.get_percent(), self.volume.is_muted())
            except Exception:  # device unplugged etc.
                self._volume_cache = (None, False)
            self._volume_checked = now
        return self._volume_cache

    def _run_volume_gesture(self, gesture: Gesture, now: float) -> None:
        if self.volume is None:
            self.overlay.notification.show("VOLUME UNAVAILABLE", now, config.COLOR_WARNING)
            return
        try:
            result = self.volume.handle_gesture(gesture)
        except Exception as exc:
            self.overlay.notification.show("VOLUME ERROR", now, config.COLOR_DANGER)
            print(f"Volume command failed: {exc}", file=sys.stderr)
            return
        if result is None:
            return
        self._volume_cache = (result.percent, result.muted)
        self._volume_checked = now
        if result.message in ("MUTED",):
            color = config.COLOR_DANGER
        elif result.message in ("VOLUME DOWN", "VOLUME MIN"):
            color = config.COLOR_WARNING
        else:
            color = config.COLOR_ACCENT
        self.overlay.notification.show(result.message, now, color)

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, now: Optional[float] = None) -> np.ndarray:
        """Run tracking, logic and drawing on one BGR frame; returns the frame to show."""
        now = time.monotonic() if now is None else now
        if self.mirror:
            frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]

        if self._last_time is not None and now > self._last_time:
            inst = 1.0 / (now - self._last_time)
            self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        self._last_time = now

        if self._layout is None or (self._layout.frame_width, self._layout.frame_height) != (width, height):
            self._layout = KeyboardLayout(width, height)
            self._ui = UILayout(width, height)
        layout, ui = self._layout, self._ui
        assert ui is not None

        hands = self.tracker.process(frame)
        primary: Optional[HandData] = hands[0] if hands else None
        blocked = now < self._input_block_until

        pointer: Optional[Tuple[float, float]] = None
        ratio: Optional[float] = None
        result: Optional[GestureResult] = None
        if primary is not None:
            ix, iy = primary.index_tip
            tx, ty = primary.thumb_tip
            if 0 <= ix < width and 0 <= iy < height and 0 <= tx < width and 0 <= ty < height:
                pointer = (ix, iy)
                ratio = pinch_ratio(primary.points)
            if primary.fully_in_frame or not config.GESTURE_REQUIRES_FULL_HAND:
                result = classify_gesture(primary.points)

        # Input is ignored while a mode switch settles or when no usable hand exists.
        if blocked or pointer is None:
            self.pinch.reset()
            if blocked:
                pointer, ratio = None, None
        pinch_event = self.pinch.update(ratio if not blocked else None, now)

        # --- on-screen buttons (work in both modes) -----------------------
        hovered_button = ui.button_at(*pointer) if pointer is not None else None
        self.button_latch.update(now, hovered_button.button_id if hovered_button else None)
        button_consumed = False
        if pinch_event:
            button_id = self.button_latch.resolve(now)
            if button_id == BUTTON_TYPING:
                self.set_mode(AppMode.TYPING, now)
                button_consumed = True
            elif button_id == BUTTON_NINJA:
                self.set_mode(AppMode.NINJA, now)
                button_consumed = True
            elif button_id == BUTTON_SEND:
                self.toggle_send(now)
                button_consumed = True

        # --- mode logic ---------------------------------------------------
        hovered_key: Optional[Key] = None
        stable_text = ""
        active_name: Optional[str] = None
        if self.mode == AppMode.TYPING:
            key_pointer = None if hovered_button is not None else pointer
            hovered_key, pressed = self.keys.update(
                layout, key_pointer, pinch_event and not button_consumed, now)
            if pressed is not None:
                self._pressed, self._pressed_at = pressed, now
        else:
            gesture = result.gesture if (result is not None and not blocked) else Gesture.NONE
            accepted = self.stabilizer.update(gesture, now)
            if accepted is not None:
                self._run_volume_gesture(accepted, now)
            cand = self.stabilizer.candidate
            active_name = cand.value if cand is not None else None
            stable_text = (f"stable: {active_name or '-'} "
                           f"{int(self.stabilizer.progress * config.GESTURE_STABLE_FRAMES)}/{config.GESTURE_STABLE_FRAMES}")

        # --- drawing ------------------------------------------------------
        if self.mode == AppMode.TYPING:
            progress = min(1.0, (now - self._pressed_at) / config.KEY_FLASH_S)
            self.renderer.render(frame, layout, self.engine.shift, self.engine.caps, hovered_key,
                                 self._pressed, progress)
        else:
            self.overlay.draw_ninja_guide(frame, ui, self.stabilizer.progress, active_name,
                                          shift_left=self.show_calibration)

        for hand in hands:
            draw_hand(frame, hand, highlight_index=(hand is primary))
        if hovered_button is not None:
            self.overlay.draw_button_hover(frame, hovered_button, self.pinch.progress(ratio))
        if pointer is not None:
            self.overlay.draw_pointer(frame, pointer, self.pinch.progress(ratio), self.pinch.is_pinched)

        if not hands:
            tracking = "LOST"
        else:
            tracking = f"{len(hands)} HAND{'S' if len(hands) > 1 else ''}"
            if primary is not None and not primary.fully_in_frame:
                tracking += " (PARTLY OUT OF FRAME)"
        gesture_text = result.gesture.value if result is not None else "-"
        self.overlay.draw_status(frame, ui, self.mode, tracking, gesture_text,
                                 self.injector.enabled, blocked)
        percent, muted = self._current_volume(now)
        self.overlay.draw_volume(frame, ui, percent, muted, self._fps)
        if self.mode == AppMode.TYPING:
            self.overlay.draw_text_panel(frame, ui, self.engine.text, self.engine.shift,
                                         self.engine.caps, self.injector.enabled, now)
        self.overlay.draw_notification(frame, ui, now, percent if self.mode == AppMode.NINJA else None)
        if self.show_calibration:
            self.overlay.draw_calibration(frame, ui, result, stable_text)
        if self.injector.last_error:
            self.overlay.notification.show("SEND DISABLED: see console", now, config.COLOR_DANGER)
            print(f"Key injection disabled: {self.injector.last_error}", file=sys.stderr)
            self.injector.last_error = None
        return frame


# ----------------------------------------------------------------------
# Camera + run loop
# ----------------------------------------------------------------------
def open_camera(index: int, width: int, height: int) -> "cv2.VideoCapture":
    """Open the webcam or raise ``RuntimeError`` with a helpful message."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
    cap = None
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            break
        cap.release()
        cap = None
    if cap is None:
        raise RuntimeError(
            f"Could not open camera {index}. Close other apps using the webcam, check "
            "Settings > Privacy > Camera, or try --camera 1."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Virtual Ninja Keyboard: type and control volume with hand gestures.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="webcam index (default 0)")
    parser.add_argument("--width", type=int, default=config.CAMERA_WIDTH)
    parser.add_argument("--height", type=int, default=config.CAMERA_HEIGHT)
    parser.add_argument("--no-mirror", action="store_true", help="do not flip the camera image")
    parser.add_argument("--no-topmost", action="store_true", help="do not keep the window on top")
    parser.add_argument("--calibration", action="store_true", help="start with the calibration panel visible")
    parser.add_argument("--send-keys", action="store_true", help="start with SEND-to-app enabled")
    parser.add_argument("--simulate-volume", action="store_true",
                        help="developer option: use an in-memory volume instead of the real Windows volume")
    return parser.parse_args(argv)


def build_volume(simulate: bool) -> Optional[VolumeController]:
    """Create the real Windows volume controller (or the explicit simulation)."""
    if simulate:
        print("NOTE: --simulate-volume is on. The real system volume is NOT changed.")
        return VolumeController(InMemoryVolumeBackend(0.5))
    try:
        return create_real_controller()
    except VolumeBackendError as exc:
        print(f"WARNING: volume control unavailable: {exc}\n"
              "         Typing still works. Use --simulate-volume to test gestures without Windows.",
              file=sys.stderr)
        return None


def run(args: argparse.Namespace) -> int:
    """Open the camera and run the main loop until ESC is pressed."""
    mirror = not args.no_mirror
    try:
        tracker = HandTracker(mirrored=mirror)
    except HandTrackerError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        cap = open_camera(args.camera, args.width, args.height)
    except RuntimeError as exc:
        tracker.close()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    app = VirtualNinjaApp(tracker, build_volume(args.simulate_volume),
                          KeyInjector(enabled=args.send_keys or config.SEND_KEYS_TO_OS_DEFAULT),
                          mirror=mirror, calibration=args.calibration)

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    if config.WINDOW_TOPMOST and not args.no_topmost:
        try:
            cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except cv2.error:
            pass  # older OpenCV builds: window just will not stay on top

    failures = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                failures += 1
                if failures >= config.MAX_CONSECUTIVE_READ_FAILURES:
                    print("ERROR: the camera stopped delivering frames.", file=sys.stderr)
                    return 4
                time.sleep(0.01)
                continue
            failures = 0

            output = app.process_frame(frame)
            cv2.imshow(config.WINDOW_NAME, output)
            if not app.handle_key(cv2.waitKeyEx(1), time.monotonic()):
                break
            try:
                if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break  # window closed with the X button
            except cv2.error:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
