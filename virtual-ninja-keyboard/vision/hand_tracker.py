"""MediaPipe hand tracking wrapper plus hand-skeleton drawing.

``mediapipe`` is imported lazily inside :class:`HandTracker` so that the rest of
the code base (and the test-suite) can be imported on machines without it.
"""

from __future__ import annotations

from typing import Dict, List, Set

import cv2
import numpy as np

import config
from vision.landmarks import (
    FINGERTIP_IDS,
    HAND_CONNECTIONS,
    INDEX_TIP,
    NUM_LANDMARKS,
    HandData,
)
from vision.smoothing import LandmarkSmoother


class HandTrackerError(RuntimeError):
    """Raised when MediaPipe is missing or incompatible."""


class HandTracker:
    """Detects up to ``max_hands`` hands and returns smoothed pixel landmarks.

    Args:
        mirrored: ``True`` if the frames passed to :meth:`process` were already
            flipped horizontally (selfie view).  MediaPipe assumes mirrored input
            when it decides "Left" versus "Right", so unmirrored frames need the
            label swapped.
    """

    def __init__(
        self,
        mirrored: bool = config.MIRROR_VIEW,
        max_hands: int = config.MAX_NUM_HANDS,
        model_complexity: int = config.MODEL_COMPLEXITY,
        min_detection_confidence: float = config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.MIN_TRACKING_CONFIDENCE,
    ) -> None:
        try:
            import mediapipe as mp  # noqa: WPS433 - intentional lazy import
        except ImportError as exc:
            raise HandTrackerError(
                "MediaPipe is not installed. Run: pip install -r requirements.txt"
            ) from exc
        if not hasattr(mp, "solutions"):
            raise HandTrackerError(
                f"Installed MediaPipe {getattr(mp, '__version__', '?')} no longer provides the "
                "'solutions' API. Install the pinned version: pip install mediapipe==0.10.14 "
                "(requires Python 3.10 or 3.11)."
            )

        self._mirrored = mirrored
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._smoothers: Dict[str, LandmarkSmoother] = {}
        self._missed: Dict[str, int] = {}

    def process(self, frame_bgr: np.ndarray) -> List[HandData]:
        """Run detection on one BGR frame and return hands sorted by confidence."""
        height, width = frame_bgr.shape[:2]

        small = frame_bgr
        if width > config.PROCESS_WIDTH:
            new_h = max(1, int(height * config.PROCESS_WIDTH / width))
            small = cv2.resize(frame_bgr, (config.PROCESS_WIDTH, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        hands: List[HandData] = []
        seen: Set[str] = set()
        speed_ref = config.SMOOTH_SPEED_REF_FRACTION * width

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                classification = handedness.classification[0]
                label = classification.label
                if not self._mirrored:
                    label = "Left" if label == "Right" else "Right"

                slot = label
                suffix = 2
                while slot in seen:  # MediaPipe occasionally labels both hands alike
                    slot = f"{label}{suffix}"
                    suffix += 1
                seen.add(slot)

                raw = np.array(
                    [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks.landmark],
                    dtype=np.float32,
                )
                if raw.shape[0] != NUM_LANDMARKS:
                    continue

                smoother = self._smoothers.get(slot)
                if smoother is None:
                    smoother = LandmarkSmoother(config.SMOOTH_ALPHA_MIN, config.SMOOTH_ALPHA_MAX)
                    self._smoothers[slot] = smoother
                self._missed[slot] = 0
                points = smoother.update(raw, speed_ref)

                inside = bool(
                    np.all(
                        (points[:, 0] >= 0) & (points[:, 0] < width)
                        & (points[:, 1] >= 0) & (points[:, 1] < height)
                    )
                )
                hands.append(
                    HandData(label=label, score=float(classification.score),
                             points=points, fully_in_frame=inside)
                )

        # Hands that vanished: keep smoothing state briefly, then discard.
        for slot in list(self._smoothers):
            if slot in seen:
                continue
            self._missed[slot] = self._missed.get(slot, 0) + 1
            if self._missed[slot] > config.TRACK_LOSS_GRACE_FRAMES:
                del self._smoothers[slot]
                self._missed.pop(slot, None)

        hands.sort(key=lambda h: h.score, reverse=True)
        return hands

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()


def draw_hand(frame: np.ndarray, hand: HandData, highlight_index: bool = True) -> None:
    """Draw the hand skeleton and joints in place; emphasise the index fingertip."""
    pts = hand.xy.astype(np.int32)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), config.COLOR_HAND_BONE, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        radius = 5 if i in FINGERTIP_IDS else 3
        cv2.circle(frame, tuple(p), radius, config.COLOR_HAND_JOINT, -1, cv2.LINE_AA)
    if highlight_index:
        tip = tuple(pts[INDEX_TIP])
        cv2.circle(frame, tip, 11, config.COLOR_POINTER, 2, cv2.LINE_AA)
        cv2.circle(frame, tip, 4, config.COLOR_POINTER, -1, cv2.LINE_AA)
