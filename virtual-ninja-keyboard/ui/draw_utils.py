"""Small OpenCV drawing helpers shared by the keyboard renderer and the UI."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

import config

Rect = Tuple[int, int, int, int]  # x, y, w, h


def rounded_rect(img: np.ndarray, rect: Rect, radius: int, color, thickness: int = -1) -> None:
    """Draw a rounded rectangle (filled when ``thickness`` < 0), anti-aliased."""
    x, y, w, h = rect
    r = int(max(0, min(radius, w // 2, h // 2)))
    aa = cv2.LINE_AA
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, -1)
        cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, -1)
        if r > 0:
            for cx, cy in ((x + r, y + r), (x + w - r, y + r), (x + r, y + h - r), (x + w - r, y + h - r)):
                cv2.circle(img, (cx, cy), r, color, -1, aa)
        return
    cv2.line(img, (x + r, y), (x + w - r, y), color, thickness, aa)
    cv2.line(img, (x + r, y + h), (x + w - r, y + h), color, thickness, aa)
    cv2.line(img, (x, y + r), (x, y + h - r), color, thickness, aa)
    cv2.line(img, (x + w, y + r), (x + w, y + h - r), color, thickness, aa)
    if r > 0:
        cv2.ellipse(img, (x + r, y + r), (r, r), 180, 0, 90, color, thickness, aa)
        cv2.ellipse(img, (x + w - r, y + r), (r, r), 270, 0, 90, color, thickness, aa)
        cv2.ellipse(img, (x + w - r, y + h - r), (r, r), 0, 0, 90, color, thickness, aa)
        cv2.ellipse(img, (x + r, y + h - r), (r, r), 90, 0, 90, color, thickness, aa)


def blend_rounded_rect(frame: np.ndarray, rect: Rect, radius: int, color, alpha: float) -> None:
    """Alpha-blend a filled rounded rectangle onto ``frame`` (only touches its ROI)."""
    x, y, w, h = rect
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + w), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = roi.copy()
    rounded_rect(overlay, (x - x0, y - y0, w, h), radius, color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def text_size(text: str, scale: float, thickness: int = 1) -> Tuple[int, int]:
    """Width and height in pixels of ``text``."""
    (tw, th), _ = cv2.getTextSize(text, config.FONT_FACE, scale, thickness)
    return tw, th


def put_text_centered(img: np.ndarray, text: str, center: Tuple[int, int], scale: float,
                      color, thickness: int = 1) -> None:
    """Draw text centred on ``center``."""
    tw, th = text_size(text, scale, thickness)
    org = (int(center[0] - tw / 2), int(center[1] + th / 2))
    cv2.putText(img, text, org, config.FONT_FACE, scale, color, thickness, cv2.LINE_AA)


def put_text(img: np.ndarray, text: str, origin: Tuple[int, int], scale: float,
             color=None, thickness: int = 1) -> None:
    """Draw left-aligned text whose top-left corner is ``origin``."""
    color = config.COLOR_TEXT if color is None else color
    _, th = text_size(text, scale, thickness)
    cv2.putText(img, text, (int(origin[0]), int(origin[1] + th)), config.FONT_FACE,
                scale, color, thickness, cv2.LINE_AA)
