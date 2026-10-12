"""All non-keyboard on-screen UI: status bar, mode buttons, text panel,
volume bar, big notifications, ninja gesture guide and the calibration view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from config import AppMode
from ui.draw_utils import (
    Rect,
    blend_rounded_rect,
    put_text,
    put_text_centered,
    rounded_rect,
    text_size,
)
from vision.gesture_classifier import FINGER_NAMES, GestureResult

BUTTON_TYPING = "mode_typing"
BUTTON_NINJA = "mode_ninja"
BUTTON_SEND = "toggle_send"


@dataclass(frozen=True)
class Button:
    """A pinch-clickable button."""

    button_id: str
    rect: Rect

    def contains(self, px: float, py: float, pad: float = 0.0) -> bool:
        x, y, w, h = self.rect
        return x - pad <= px < x + w + pad and y - pad <= py < y + h + pad


class UILayout:
    """Pixel rectangles for the panels and buttons at a given frame size."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.scale = height / 1300.0                       # base font scale
        self.bold = 2 if height >= 600 else 1              # stroke weight for emphasised text
        self.status_h = max(30, int(height * config.STATUS_HEIGHT_FRACTION))
        self.margin = max(6, int(width * 0.012))

        pad = max(3, int(self.status_h * 0.10))
        btn_h = self.status_h - 2 * pad
        btn_w = int(min(width * 0.15, btn_h * 3.4))
        total = 3 * btn_w + 2 * pad
        x = width - total - self.margin
        self.buttons: List[Button] = [
            Button(BUTTON_TYPING, (x, pad, btn_w, btn_h)),
            Button(BUTTON_NINJA, (x + btn_w + pad, pad, btn_w, btn_h)),
            Button(BUTTON_SEND, (x + 2 * (btn_w + pad), pad, btn_w, btn_h)),
        ]
        self.button_pad = pad

        tx0, ty0, tx1, ty1 = config.TEXT_PANEL_REGION
        self.text_panel: Rect = (int(tx0 * width), int(ty0 * height),
                                 int((tx1 - tx0) * width), int((ty1 - ty0) * height))

    def button_at(self, px: float, py: float) -> Optional[Button]:
        """Button under a pixel position, if any."""
        for button in self.buttons:
            if button.contains(px, py, self.button_pad / 2.0):
                return button
        return None


class Notification:
    """A large fading message such as "VOLUME UP"."""

    def __init__(self) -> None:
        self.text = ""
        self.color = config.COLOR_ACCENT
        self._start = -1e9
        self._duration = config.NOTIFICATION_DURATION_S

    def show(self, text: str, now: float, color=None, duration: Optional[float] = None) -> None:
        self.text = text
        self.color = config.COLOR_ACCENT if color is None else color
        self._start = now
        self._duration = config.NOTIFICATION_DURATION_S if duration is None else duration

    def alpha(self, now: float) -> float:
        """0 when finished; 1 while fully visible; fades during the last 35%."""
        age = now - self._start
        if age < 0 or age >= self._duration:
            return 0.0
        fade_start = self._duration * 0.65
        if age <= fade_start:
            return 1.0
        return max(0.0, 1.0 - (age - fade_start) / (self._duration - fade_start))


def wrap_text(text: str, max_width: int, scale: float, thickness: int = 1) -> List[str]:
    """Wrap text to a pixel width, honouring explicit newlines."""
    lines: List[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for ch in paragraph:
            if text_size(current + ch, scale, thickness)[0] > max_width and current:
                lines.append(current)
                current = ch
            else:
                current += ch
        lines.append(current)
    return lines


class Overlay:
    """Draws every UI element except the keyboard and the hand skeleton."""

    def __init__(self) -> None:
        self.notification = Notification()

    # ------------------------------------------------------------------
    def draw_status(self, frame: np.ndarray, ui: UILayout, mode: AppMode, tracking: str,
                    gesture_text: str, send_enabled: bool, input_blocked: bool) -> None:
        """Top bar: mode, tracking status, gesture, plus the three buttons."""
        blend_rounded_rect(frame, (0, 0, ui.width, ui.status_h), 0, config.COLOR_PANEL, config.COLOR_PANEL_ALPHA)
        s = ui.scale
        line_h = int(ui.status_h / 2)
        mode_color = config.COLOR_ACCENT if mode == AppMode.TYPING else config.COLOR_WARNING
        put_text(frame, f"MODE: {mode.value}", (ui.margin, int(line_h * 0.15)), s * 1.05, mode_color, ui.bold)
        suffix = "   (switching...)" if input_blocked else ""
        put_text(frame, f"TRACKING: {tracking}   GESTURE: {gesture_text}{suffix}",
                 (ui.margin, line_h + int(line_h * 0.05)), s * 0.9, config.COLOR_TEXT, 1)

        labels = {
            BUTTON_TYPING: ("TYPING", mode == AppMode.TYPING),
            BUTTON_NINJA: ("NINJA", mode == AppMode.NINJA),
            BUTTON_SEND: ("SEND:ON" if send_enabled else "SEND:OFF", send_enabled),
        }
        for button in ui.buttons:
            label, active = labels[button.button_id]
            color = config.COLOR_BUTTON_ACTIVE if active else config.COLOR_BUTTON
            rounded_rect(frame, button.rect, button.rect[3] // 3, color, -1)
            rounded_rect(frame, button.rect, button.rect[3] // 3, config.COLOR_KEY_OUTLINE, 2)
            bx, by, bw, bh = button.rect
            put_text_centered(frame, label, (bx + bw // 2, by + bh // 2), s * 0.95, config.COLOR_TEXT, ui.bold)

    def draw_button_hover(self, frame: np.ndarray, button: Button, progress: float) -> None:
        """Highlight the button under the fingertip."""
        blend_rounded_rect(frame, button.rect, button.rect[3] // 3, config.COLOR_HOVER_FILL,
                           0.35 + 0.4 * progress)

    # ------------------------------------------------------------------
    def draw_text_panel(self, frame: np.ndarray, ui: UILayout, text: str, shift: bool,
                        caps: bool, send_enabled: bool, now: float) -> None:
        """Bottom panel showing the typed text."""
        x, y, w, h = ui.text_panel
        blend_rounded_rect(frame, ui.text_panel, 12, config.COLOR_PANEL, config.COLOR_PANEL_ALPHA)
        rounded_rect(frame, ui.text_panel, 12, config.COLOR_KEY_OUTLINE, 2)
        s = ui.scale
        flags = [f for f, on in (("SHIFT", shift), ("CAPS", caps), ("SENDING TO APP", send_enabled)) if on]
        header = "OUTPUT" + ("   [" + " | ".join(flags) + "]" if flags else "")
        put_text(frame, header, (x + 14, y + 12), s * 0.8, config.COLOR_ACCENT, 1)

        body_scale = s * 1.5
        pad = 14
        cursor = "|" if int(now * 2) % 2 == 0 else " "
        lines = wrap_text(text + cursor, w - 2 * pad, body_scale, 2)
        _, th = text_size("Ag", body_scale, 2)
        line_h = int(th * 1.5)
        top = y + int(s * 0.8 * 30) + 12
        max_lines = max(1, (y + h - top - 6) // line_h)
        for i, line in enumerate(lines[-max_lines:]):
            put_text(frame, line, (x + pad, top + i * line_h), body_scale, config.COLOR_TEXT, 2)

    # ------------------------------------------------------------------
    def draw_volume(self, frame: np.ndarray, ui: UILayout, percent: Optional[int],
                    muted: bool, fps: float) -> None:
        """Small volume bar under the status bar (left) and the FPS counter."""
        s = ui.scale
        x, y = ui.margin, ui.status_h + 6
        bar_w, bar_h = int(ui.width * 0.18), max(10, int(ui.height * 0.022))
        if percent is None:
            put_text(frame, f"VOLUME: UNAVAILABLE   {fps:.0f} FPS", (x, y), s * 0.9, config.COLOR_WARNING, 1)
            return
        label = "MUTED" if muted else f"VOL {percent}%"
        color = config.COLOR_DANGER if muted else config.COLOR_ACCENT
        put_text(frame, label, (x, y), s * 0.9, color, ui.bold)
        lw, _ = text_size(label, s * 0.9, ui.bold)
        bx = x + lw + 10
        rounded_rect(frame, (bx, y, bar_w, bar_h), bar_h // 2, config.COLOR_TEXT, 1)
        fill_w = int((bar_w - 4) * percent / 100.0)
        if fill_w > 0:
            rounded_rect(frame, (bx + 2, y + 2, fill_w, bar_h - 4), (bar_h - 4) // 2, color, -1)
        put_text(frame, f"{fps:.0f} FPS", (bx + bar_w + 12, y), s * 0.8, config.COLOR_TEXT, 1)

    def draw_notification(self, frame: np.ndarray, ui: UILayout, now: float,
                          percent: Optional[int] = None) -> None:
        """Big centred message, fading out."""
        alpha = self.notification.alpha(now)
        if alpha <= 0.0 or not self.notification.text:
            return
        text = self.notification.text
        scale = ui.height / 260.0
        tw, th = text_size(text, scale, 5)
        max_w = ui.width * 0.9
        if tw > max_w:
            scale *= max_w / tw
            tw, th = text_size(text, scale, 5)
        cx, cy = ui.width // 2, int(ui.height * 0.45)
        pad = int(th * 0.7)
        box: Rect = (cx - tw // 2 - pad, cy - th // 2 - pad, tw + 2 * pad, th + 2 * pad
                     + (int(th * 0.55) if percent is not None else 0))
        blend_rounded_rect(frame, box, pad // 2, config.COLOR_PANEL, 0.75 * alpha)
        overlay = frame[max(0, box[1]):box[1] + box[3], max(0, box[0]):box[0] + box[2]]
        if overlay.size == 0:
            return
        layer = frame.copy() if alpha < 1.0 else frame
        put_text_centered(layer, text, (cx, cy), scale, self.notification.color, 5)
        if percent is not None:
            put_text_centered(layer, f"{percent}%", (cx, cy + int(th * 0.95)), scale * 0.45,
                              config.COLOR_TEXT, 2)
        if layer is not frame:
            cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0, dst=frame)

    # ------------------------------------------------------------------
    def draw_pointer(self, frame: np.ndarray, point: Tuple[float, float], pinch_progress: float,
                     pinched: bool) -> None:
        """Ring around the fingertip that fills as the pinch closes."""
        center = (int(point[0]), int(point[1]))
        radius = max(14, int(frame.shape[0] * 0.03))
        cv2.circle(frame, center, radius, config.COLOR_TEXT, 1, cv2.LINE_AA)
        if pinch_progress > 0.02:
            color = config.COLOR_ACCENT if pinched else config.COLOR_POINTER
            cv2.ellipse(frame, center, (radius, radius), -90, 0, 360 * pinch_progress,
                        color, 4, cv2.LINE_AA)

    # ------------------------------------------------------------------
    def draw_ninja_guide(self, frame: np.ndarray, ui: UILayout, stabilizer_progress: float,
                         active_name: Optional[str], shift_left: bool = False) -> None:
        """Gesture cheat-sheet with small finger-pattern icons (Ninja Control mode)."""
        s = ui.scale
        if shift_left:   # leave the right-hand side free for the calibration panel
            x0, w = int(ui.width * 0.02), int(ui.width * 0.64)
        else:
            x0, w = int(ui.width * 0.12), int(ui.width * 0.76)
        y0, h = int(ui.height * 0.20), int(ui.height * 0.50)
        blend_rounded_rect(frame, (x0, y0, w, h), 16, config.COLOR_KEYBOARD_BG, 0.35)
        rounded_rect(frame, (x0, y0, w, h), 16, config.COLOR_KEY_OUTLINE, 2)
        put_text_centered(frame, "NINJA SEALS", (x0 + w // 2, y0 + int(h * 0.09)), s * 1.5, config.COLOR_ACCENT, 2)

        names = ["VOLUME_UP", "VOLUME_DOWN", "MUTE_TOGGLE"]
        titles = {"VOLUME_UP": "VOLUME UP", "VOLUME_DOWN": "VOLUME DOWN", "MUTE_TOGGLE": "MUTE / UNMUTE"}
        card_w = w // 3
        for i, name in enumerate(names):
            cx = x0 + i * card_w + card_w // 2
            active = name == active_name
            card = (x0 + i * card_w + 10, y0 + int(h * 0.18), card_w - 20, int(h * 0.76))
            if active:
                blend_rounded_rect(frame, card, 12, config.COLOR_HOVER_FILL, 0.25 + 0.35 * stabilizer_progress)
            rounded_rect(frame, card, 12, config.COLOR_KEY_OUTLINE if active else config.COLOR_BUTTON, 2)
            self._draw_pattern_icon(frame, cx, card[1] + int(card[3] * 0.40), int(card[3] * 0.50),
                                    config.GESTURE_PATTERNS[name])
            put_text_centered(frame, titles[name], (cx, card[1] + int(card[3] * 0.76)), s * 1.05, config.COLOR_TEXT, 2)
            put_text_centered(frame, config.GESTURE_DESCRIPTIONS.get(name, ""),
                              (cx, card[1] + int(card[3] * 0.90)), s * 0.85, config.COLOR_ACCENT, 1)

    @staticmethod
    def _draw_pattern_icon(frame: np.ndarray, cx: int, cy: int, size: int,
                           pattern: Tuple[bool, bool, bool, bool]) -> None:
        """Four finger bars (tall = extended, short = folded) above a palm block."""
        bar_w = max(6, size // 7)
        gap = max(4, bar_w // 2)
        total = 4 * bar_w + 3 * gap
        left = cx - total // 2
        base = cy + size // 3
        for i, extended in enumerate(pattern):
            height = int(size * (0.62 if extended else 0.22))
            x = left + i * (bar_w + gap)
            color = config.COLOR_ACCENT if extended else config.COLOR_BUTTON
            rounded_rect(frame, (x, base - height, bar_w, height), bar_w // 2, color, -1)
        rounded_rect(frame, (left - gap, base, total + 2 * gap, int(size * 0.28)), bar_w // 2,
                     config.COLOR_BUTTON, -1)

    # ------------------------------------------------------------------
    def draw_calibration(self, frame: np.ndarray, ui: UILayout, result: Optional[GestureResult],
                         stable_text: str) -> None:
        """Debug/calibration panel showing finger states, angles and pinch ratio."""
        s = ui.scale
        w, h = int(ui.width * 0.30), int(ui.height * 0.36)
        x, y = ui.width - w - ui.margin, ui.status_h + 8
        blend_rounded_rect(frame, (x, y, w, h), 10, (0, 0, 0), 0.6)
        rounded_rect(frame, (x, y, w, h), 10, config.COLOR_WARNING, 2)
        put_text(frame, "CALIBRATION (F4 to hide)", (x + 10, y + 6), s * 0.85, config.COLOR_WARNING, 1)
        if result is None:
            put_text(frame, "No usable hand", (x + 10, y + int(h * 0.2)), s, config.COLOR_TEXT, 1)
            return
        line_h = int(h * 0.115)
        rows = [("thumb", result.states.thumb)] + [(n, getattr(result.states, n)) for n in FINGER_NAMES]
        for i, (name, extended) in enumerate(rows):
            angle = result.states.angles.get(name, 0.0)
            color = config.COLOR_ACCENT if extended else config.COLOR_DANGER
            text = f"{name:<6} {'EXTENDED' if extended else 'folded  '} {angle:5.0f} deg"
            put_text(frame, text, (x + 10, y + int(h * 0.17) + i * line_h), s * 0.8, color, 1)
        base = y + int(h * 0.17) + 5 * line_h + 4
        put_text(frame, f"pinch ratio: {result.pinch_ratio:.2f}  (on<{config.PINCH_ON_RATIO}, off>{config.PINCH_OFF_RATIO})",
                 (x + 10, base), s * 0.7, config.COLOR_TEXT, 1)
        put_text(frame, f"gesture: {result.gesture.value}", (x + 10, base + line_h), s * 0.85, config.COLOR_ACCENT, 1)
        put_text(frame, stable_text, (x + 10, base + 2 * line_h), s * 0.7, config.COLOR_TEXT, 1)
