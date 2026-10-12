"""Virtual keyboard layout: which keys exist and where they sit on screen.

Pure Python (no OpenCV) so layout and hit-testing can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import config


class KeyKind(Enum):
    """What a key does when pressed."""

    CHAR = "char"
    SPACE = "space"
    BACKSPACE = "backspace"
    ENTER = "enter"
    SHIFT = "shift"
    CAPS = "caps"
    CLEAR = "clear"


@dataclass(frozen=True)
class KeySpec:
    """Static description of one key (independent of screen size)."""

    label: str                 # base character, or a name for special keys
    kind: KeyKind = KeyKind.CHAR
    width: float = 1.0         # in key units
    shifted: str = ""          # character produced with Shift (non-letters)


@dataclass(frozen=True)
class Key:
    """A key placed on screen. Coordinates are integer pixels."""

    key_id: str
    spec: KeySpec
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def contains(self, px: float, py: float, pad: float = 0.0) -> bool:
        """True if the point is inside the key rectangle grown by ``pad`` pixels."""
        return (self.x - pad <= px < self.x + self.w + pad
                and self.y - pad <= py < self.y + self.h + pad)


def _chars(base: str, shifted: str) -> List[KeySpec]:
    """Make character keys from two equally long strings."""
    assert len(base) == len(shifted)
    return [KeySpec(b, KeyKind.CHAR, 1.0, s) for b, s in zip(base, shifted)]


#: The keyboard rows. Row widths are centred on screen.
KEY_ROWS: Tuple[Tuple[KeySpec, ...], ...] = (
    tuple(_chars("`1234567890-=", "~!@#$%^&*()_+")) + (KeySpec("BKSP", KeyKind.BACKSPACE, 2.0),),
    tuple(_chars("qwertyuiop[]\\", "QWERTYUIOP{}|")),
    (KeySpec("CAPS", KeyKind.CAPS, 1.75),)
    + tuple(_chars("asdfghjkl;'", 'ASDFGHJKL:"'))
    + (KeySpec("ENTER", KeyKind.ENTER, 2.25),),
    (KeySpec("SHIFT", KeyKind.SHIFT, 2.25),)
    + tuple(_chars("zxcvbnm,./", "ZXCVBNM<>?"))
    + (KeySpec("SHIFT", KeyKind.SHIFT, 2.75),),
    (KeySpec("CLEAR", KeyKind.CLEAR, 2.5), KeySpec("SPACE", KeyKind.SPACE, 9.0)),
)


def resolve_char(spec: KeySpec, shift: bool, caps: bool) -> str:
    """Character a CHAR key produces for the given Shift / Caps Lock state.

    Letters respond to Shift XOR Caps Lock; all other keys respond to Shift only.
    """
    if spec.label.isalpha():
        return spec.label.upper() if (shift != caps) else spec.label.lower()
    return spec.shifted if (shift and spec.shifted) else spec.label


def display_label(spec: KeySpec, shift: bool, caps: bool) -> str:
    """Text drawn on the key for the current modifier state."""
    if spec.kind == KeyKind.CHAR:
        return resolve_char(spec, shift, caps)
    return spec.label


def keyboard_region(width: int, height: int) -> Tuple[int, int, int, int]:
    """Pixel rectangle ``(x, y, w, h)`` reserved for the keyboard."""
    x0, y0, x1, y1 = config.KEYBOARD_REGION
    return int(x0 * width), int(y0 * height), int((x1 - x0) * width), int((y1 - y0) * height)


class KeyboardLayout:
    """Computes key rectangles for a given frame size and answers hit-tests."""

    def __init__(self, frame_width: int, frame_height: int) -> None:
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Frame size must be positive")
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.keys: List[Key] = []
        self.key_by_id: Dict[str, Key] = {}
        self.gap = 0
        self.corner_radius = 0
        self.bounds: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._build()

    def _build(self) -> None:
        rx, ry, rw, rh = keyboard_region(self.frame_width, self.frame_height)
        rows = len(KEY_ROWS)
        unit_w = rw / config.KEY_UNITS_PER_ROW
        row_h = min(rh / rows, unit_w * config.KEY_MAX_ASPECT)
        total_h = row_h * rows
        top = ry + (rh - total_h) / 2.0
        self.gap = max(2, int(unit_w * config.KEY_GAP_FRACTION))
        self.corner_radius = max(2, int(row_h * config.KEY_CORNER_FRACTION))

        for r, row in enumerate(KEY_ROWS):
            row_units = sum(spec.width for spec in row)
            cursor = rx + (rw - row_units * unit_w) / 2.0
            for c, spec in enumerate(row):
                x0 = int(round(cursor))
                x1 = int(round(cursor + spec.width * unit_w))
                y0 = int(round(top + r * row_h))
                y1 = int(round(top + (r + 1) * row_h))
                key = Key(
                    key_id=f"r{r}c{c}",
                    spec=spec,
                    x=x0 + self.gap // 2,
                    y=y0 + self.gap // 2,
                    w=max(1, (x1 - x0) - self.gap),
                    h=max(1, (y1 - y0) - self.gap),
                )
                self.keys.append(key)
                self.key_by_id[key.key_id] = key
                cursor += spec.width * unit_w

        xs0 = min(k.x for k in self.keys)
        ys0 = min(k.y for k in self.keys)
        xs1 = max(k.x + k.w for k in self.keys)
        ys1 = max(k.y + k.h for k in self.keys)
        pad = self.gap * 2
        self.bounds = (xs0 - pad, ys0 - pad, (xs1 - xs0) + 2 * pad, (ys1 - ys0) + 2 * pad)

    def key_at(self, px: float, py: float) -> Optional[Key]:
        """The key under a pixel position, or ``None``.

        Keys are tested with half a gap of padding so there are no dead zones
        between neighbouring keys.
        """
        pad = self.gap / 2.0
        for key in self.keys:
            if key.contains(px, py, pad):
                return key
        return None
