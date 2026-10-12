"""Typing logic: text buffer, modifier state, hover latching and OS key injection.

``TypingEngine`` and ``HoverLatch`` are pure Python and fully unit-tested.
``KeyInjector`` is the only part that talks to the operating system (pynput).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import config
from keyboard.layout import Key, KeyboardLayout, KeyKind, KeySpec, resolve_char


@dataclass(frozen=True)
class KeyAction:
    """Result of pressing a key."""

    kind: KeyKind
    text: str = ""          # characters produced (CHAR / SPACE only)


class TypingEngine:
    """Holds the typed text and the Shift / Caps Lock state.

    Shift behaves like a sticky "one-shot" key: it applies to the next
    character and then releases itself.  Caps Lock stays on until pressed again.
    """

    def __init__(self, max_length: int = config.MAX_TEXT_LENGTH) -> None:
        self.text = ""
        self.shift = False
        self.caps = False
        self.max_length = max_length

    def press(self, spec: KeySpec) -> KeyAction:
        """Apply a key press and report what was produced."""
        kind = spec.kind

        if kind == KeyKind.SHIFT:
            self.shift = not self.shift
        elif kind == KeyKind.CAPS:
            self.caps = not self.caps
        elif kind == KeyKind.CLEAR:
            self.text = ""
        elif kind == KeyKind.BACKSPACE:
            self.text = self.text[:-1]
        elif kind == KeyKind.ENTER:
            self._append("\n")
        elif kind == KeyKind.SPACE:
            self._append(" ")
            return KeyAction(kind, " ")
        elif kind == KeyKind.CHAR:
            char = resolve_char(spec, self.shift, self.caps)
            self._append(char)
            self.shift = False  # one-shot
            return KeyAction(kind, char)
        return KeyAction(kind)

    def _append(self, chars: str) -> None:
        if len(self.text) + len(chars) <= self.max_length:
            self.text += chars


class HoverLatch:
    """Remembers which item the fingertip was *deliberately* resting on.

    Closing a pinch moves the fingertip a little, so the item under the tip at
    the instant of the pinch may be a neighbour of the one the user aimed at.
    The latch therefore tracks the last item hovered for at least
    ``min_dwell_s`` ("stable") and keeps returning it for ``lookback_s`` after
    the fingertip leaves.  An item that was never held steady is never returned,
    which prevents accidental presses while the hand is swooping across keys.
    """

    def __init__(
        self,
        min_dwell_s: float = config.HOVER_MIN_DWELL_S,
        lookback_s: float = config.HOVER_LOOKBACK_S,
    ) -> None:
        self.min_dwell_s = min_dwell_s
        self.lookback_s = lookback_s
        self._candidate: Optional[str] = None
        self._candidate_since = 0.0
        self._stable: Optional[str] = None
        self._stable_seen = -1e9

    def reset(self) -> None:
        """Forget all hover history."""
        self._candidate = None
        self._stable = None
        self._stable_seen = -1e9

    def update(self, now: float, item_id: Optional[str]) -> None:
        """Feed the item currently under the fingertip (or ``None``)."""
        if item_id != self._candidate:
            self._candidate = item_id
            self._candidate_since = now
        if item_id is not None and (now - self._candidate_since) >= self.min_dwell_s:
            self._stable = item_id
            self._stable_seen = now

    def resolve(self, now: float) -> Optional[str]:
        """The item a pinch right now should act on, or ``None``."""
        if self._stable is not None and (now - self._stable_seen) <= self.lookback_s:
            return self._stable
        return None


class KeyInjector:
    """Optionally types into whichever application currently has focus.

    Uses ``pynput`` (imported lazily).  If injection fails once it disables
    itself and remembers the error instead of crashing the video loop.
    """

    def __init__(self, enabled: bool = config.SEND_KEYS_TO_OS_DEFAULT) -> None:
        self.enabled = enabled
        self.last_error: Optional[str] = None
        self._controller = None
        self._key_enum = None

    def toggle(self) -> bool:
        """Flip the enabled flag and return the new value."""
        self.enabled = not self.enabled
        return self.enabled

    def send(self, action: KeyAction) -> None:
        """Send the key press to the OS if enabled. Never raises."""
        if not self.enabled:
            return
        try:
            self._ensure_controller()
            kb, Key = self._controller, self._key_enum
            if action.kind in (KeyKind.CHAR, KeyKind.SPACE) and action.text:
                kb.type(action.text)
            elif action.kind == KeyKind.BACKSPACE:
                kb.tap(Key.backspace)
            elif action.kind == KeyKind.ENTER:
                kb.tap(Key.enter)
            # Shift / Caps / Clear only change the on-screen state: the real
            # character (already shifted) is typed, so OS modifiers stay untouched.
        except Exception as exc:  # pynput backends raise many different errors
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.enabled = False

    def _ensure_controller(self) -> None:
        if self._controller is None:
            from pynput.keyboard import Controller, Key as PynputKey

            self._controller = Controller()
            self._key_enum = PynputKey


class KeyController:
    """Connects fingertip position + pinch events to key presses."""

    def __init__(self, engine: TypingEngine, injector: KeyInjector, latch: Optional[HoverLatch] = None) -> None:
        self.engine = engine
        self.injector = injector
        self.latch = latch or HoverLatch()

    def reset(self) -> None:
        """Forget hover history (mode switches, tracking loss)."""
        self.latch.reset()

    def update(
        self,
        layout: KeyboardLayout,
        pointer: Optional[Tuple[float, float]],
        pinch_triggered: bool,
        now: float,
    ) -> Tuple[Optional[Key], Optional[Key]]:
        """Process one frame.

        Returns:
            ``(hovered_key, pressed_key)``.  ``hovered_key`` is the key directly
            under the fingertip (for highlighting); ``pressed_key`` is set only
            on the frame a press was accepted.
        """
        hovered = layout.key_at(*pointer) if pointer is not None else None
        self.latch.update(now, hovered.key_id if hovered else None)

        pressed: Optional[Key] = None
        if pinch_triggered:
            key_id = self.latch.resolve(now)
            if key_id is not None:
                pressed = layout.key_by_id.get(key_id)
        if pressed is not None:
            action = self.engine.press(pressed.spec)
            self.injector.send(action)
        return hovered, pressed
