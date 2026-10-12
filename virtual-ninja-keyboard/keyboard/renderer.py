"""Draws the translucent purple keyboard with lime outlines using OpenCV.

Performance: the keys, outlines and labels only change when the layout or the
Shift / Caps state changes, so they are rendered once into cached float arrays.
Each frame then needs a single multiply-add over the keyboard region:

    out = frame * A + B

where ``A`` and ``B`` already contain the alpha blending of the translucent key
fills and the opaque outlines/labels.  Only the hover highlight and the
"pressed" flash are drawn per frame.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

import config
from keyboard.layout import Key, KeyboardLayout, KeyKind, display_label
from ui.draw_utils import blend_rounded_rect, put_text_centered, rounded_rect


class _StaticLayer:
    """Pre-computed blend arrays for one (layout, shift, caps) combination."""

    def __init__(self, origin: Tuple[int, int], mul: np.ndarray, add: np.ndarray) -> None:
        self.origin = origin
        self.mul = mul      # (h, w, 1) float32
        self.add = add      # (h, w, 3) float32


class KeyboardRenderer:
    """Renders a :class:`KeyboardLayout` onto video frames."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, bool, bool], _StaticLayer] = {}

    # ------------------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        layout: KeyboardLayout,
        shift: bool,
        caps: bool,
        hovered: Optional[Key] = None,
        pressed: Optional[Key] = None,
        press_progress: float = 1.0,
    ) -> None:
        """Draw the keyboard onto ``frame`` in place.

        Args:
            hovered: Key under the fingertip (highlighted).
            pressed: Key that was just selected (flashes).
            press_progress: 0.0 right after the press, 1.0 when the flash is over.
        """
        layer = self._get_layer(layout, shift, caps)
        x, y = layer.origin
        h, w = layer.mul.shape[:2]
        fh, fw = frame.shape[:2]
        if x < 0 or y < 0 or x + w > fw or y + h > fh:
            return  # frame smaller than layout (resize in progress)

        roi = frame[y:y + h, x:x + w]
        out = roi * layer.mul + layer.add
        np.minimum(out, 255.0, out=out)
        roi[:] = out.astype(np.uint8)

        if hovered is not None:
            self._draw_highlight(frame, layout, hovered, shift, caps)
        if pressed is not None and press_progress < 1.0:
            flash = config.HOVER_ALPHA + (1.0 - config.HOVER_ALPHA) * (1.0 - press_progress)
            blend_rounded_rect(frame, (pressed.x, pressed.y, pressed.w, pressed.h),
                               layout.corner_radius, config.COLOR_PRESS_FLASH, min(1.0, flash))
            rounded_rect(frame, (pressed.x, pressed.y, pressed.w, pressed.h),
                         layout.corner_radius, config.COLOR_ACCENT, 4)
            self._draw_label(frame, pressed, display_label(pressed.spec, shift, caps),
                             layout, (40, 0, 40), 0, 0)

    # ------------------------------------------------------------------
    def _draw_highlight(self, frame: np.ndarray, layout: KeyboardLayout, key: Key,
                        shift: bool, caps: bool) -> None:
        rect = (key.x, key.y, key.w, key.h)
        blend_rounded_rect(frame, rect, layout.corner_radius, config.COLOR_HOVER_FILL, config.HOVER_ALPHA)
        rounded_rect(frame, rect, layout.corner_radius, config.COLOR_ACCENT, 3)
        self._draw_label(frame, key, display_label(key.spec, shift, caps), layout, (40, 0, 40), 0, 0)

    @staticmethod
    def _label_scale(key: Key, text: str) -> float:
        """Font scale so the label fits inside the key."""
        base = key.h / 62.0 * (0.9 if len(text) == 1 else 0.55)
        (tw, _), _ = cv2.getTextSize(text, config.FONT_FACE, base, 1)
        max_w = key.w * 0.86
        return base if tw <= max_w else base * max_w / tw

    def _draw_label(self, img, key: Key, text: str, layout: KeyboardLayout, color,
                    ox: int, oy: int, mask=None) -> None:
        scale = self._label_scale(key, text)
        thickness = 2 if key.h >= 50 else 1
        cx, cy = key.center
        put_text_centered(img, text, (cx - ox, cy - oy), scale, color, thickness)
        if mask is not None:
            put_text_centered(mask, text, (cx - ox, cy - oy), scale, 255, thickness)

    # ------------------------------------------------------------------
    def _get_layer(self, layout: KeyboardLayout, shift: bool, caps: bool) -> _StaticLayer:
        cache_key = (layout.frame_width, layout.frame_height, shift, caps)
        layer = self._cache.get(cache_key)
        if layer is None:
            if len(self._cache) > 16:
                self._cache.clear()
            layer = self._build_layer(layout, shift, caps)
            self._cache[cache_key] = layer
        return layer

    def _build_layer(self, layout: KeyboardLayout, shift: bool, caps: bool) -> _StaticLayer:
        bx, by, bw, bh = layout.bounds
        bx, by = max(0, bx), max(0, by)
        bw = min(bw, layout.frame_width - bx)
        bh = min(bh, layout.frame_height - by)

        fill = np.zeros((bh, bw, 3), np.uint8)       # colours of translucent fills
        bg_mask = np.zeros((bh, bw), np.uint8)       # keyboard panel shape
        key_mask = np.zeros((bh, bw), np.uint8)      # union of key shapes
        lines = np.zeros((bh, bw, 3), np.uint8)      # opaque outlines + labels (premultiplied)
        line_mask = np.zeros((bh, bw), np.uint8)     # their coverage
        radius = layout.corner_radius

        rounded_rect(fill, (0, 0, bw, bh), radius * 2, config.COLOR_KEYBOARD_BG, -1)
        rounded_rect(bg_mask, (0, 0, bw, bh), radius * 2, 255, -1)

        for key in layout.keys:
            rect = (key.x - bx, key.y - by, key.w, key.h)
            toggled = (key.spec.kind == KeyKind.SHIFT and shift) or (key.spec.kind == KeyKind.CAPS and caps)
            if toggled:
                color = config.COLOR_KEY_FILL_TOGGLED
            elif key.spec.kind == KeyKind.CHAR:
                color = config.COLOR_KEY_FILL
            else:
                color = config.COLOR_KEY_FILL_SPECIAL
            rounded_rect(fill, rect, radius, color, -1)
            rounded_rect(key_mask, rect, radius, 255, -1)
            rounded_rect(lines, rect, radius, config.COLOR_KEY_OUTLINE, 2)
            rounded_rect(line_mask, rect, radius, 255, 2)

            # Labels are drawn white; the mask needs matching coverage, so draw
            # the text into a scratch pair and merge it.
            text = display_label(key.spec, shift, caps)
            scale = self._label_scale(key, text)
            thickness = 2 if key.h >= 50 else 1
            cx, cy = key.center
            cx, cy = cx - bx, cy - by
            label_img = np.zeros_like(lines)
            label_cov = np.zeros_like(line_mask)
            put_text_centered(label_img, text, (cx, cy), scale, config.COLOR_KEY_LABEL, thickness)
            put_text_centered(label_cov, text, (cx, cy), scale, 255, thickness)
            cov = label_cov.astype(np.float32) / 255.0
            lines[:] = (lines.astype(np.float32) * (1.0 - cov[..., None]) + label_img).clip(0, 255).astype(np.uint8)
            line_mask[:] = np.maximum(line_mask, label_cov)

        bg_a = bg_mask.astype(np.float32) / 255.0 * config.KEYBOARD_BG_ALPHA
        key_a = key_mask.astype(np.float32) / 255.0
        alpha = bg_a * (1.0 - key_a) + config.KEY_FILL_ALPHA * key_a            # (h, w)
        cover = (line_mask.astype(np.float32) / 255.0)[..., None]               # (h, w, 1)
        alpha3 = alpha[..., None]

        mul = ((1.0 - alpha3) * (1.0 - cover)).astype(np.float32)
        add = (fill.astype(np.float32) * alpha3 * (1.0 - cover) + lines.astype(np.float32)).astype(np.float32)
        return _StaticLayer((bx, by), mul, add)
