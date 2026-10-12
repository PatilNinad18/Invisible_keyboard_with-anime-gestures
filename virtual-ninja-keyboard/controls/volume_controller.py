"""Windows master-volume control (pycaw) behind a small, testable interface.

* :class:`VolumeBackend` is the minimal interface the controller needs.
* :class:`PycawVolumeBackend` talks to the real Windows Core Audio API.
* :class:`InMemoryVolumeBackend` is for unit tests and the explicit
  ``--simulate-volume`` developer flag.  It never touches the system volume.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, Protocol

import config
from vision.gesture_classifier import Gesture


class VolumeBackendError(RuntimeError):
    """Raised when the real Windows volume interface cannot be opened."""


class VolumeBackend(Protocol):
    """What :class:`VolumeController` needs from a volume implementation."""

    def get_scalar(self) -> float: ...
    def set_scalar(self, value: float) -> None: ...
    def get_mute(self) -> bool: ...
    def set_mute(self, muted: bool) -> None: ...


class PycawVolumeBackend:
    """Controls the real Windows master volume via pycaw / Core Audio."""

    def __init__(self) -> None:
        if not sys.platform.startswith("win"):
            raise VolumeBackendError("Windows volume control is only available on Windows.")
        try:
            from ctypes import POINTER, cast

            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        except ImportError as exc:
            raise VolumeBackendError(
                "pycaw/comtypes are missing. Run: pip install -r requirements.txt"
            ) from exc

        try:
            speakers = AudioUtilities.GetSpeakers()
            # Newer pycaw releases return a wrapper that exposes EndpointVolume
            # directly; older ones return the raw IMMDevice that must be activated.
            endpoint = getattr(speakers, "EndpointVolume", None)
            if endpoint is None:
                interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                endpoint = cast(interface, POINTER(IAudioEndpointVolume))
            self._endpoint = endpoint
            self._endpoint.GetMasterVolumeLevelScalar()  # fail early if unusable
        except Exception as exc:  # COMError and friends
            raise VolumeBackendError(f"Could not open the default audio output device: {exc}") from exc

    def get_scalar(self) -> float:
        return float(self._endpoint.GetMasterVolumeLevelScalar())

    def set_scalar(self, value: float) -> None:
        self._endpoint.SetMasterVolumeLevelScalar(float(value), None)

    def get_mute(self) -> bool:
        return bool(self._endpoint.GetMute())

    def set_mute(self, muted: bool) -> None:
        self._endpoint.SetMute(1 if muted else 0, None)


class InMemoryVolumeBackend:
    """Fake backend that only stores values in memory (tests / simulation)."""

    def __init__(self, scalar: float = 0.5, muted: bool = False) -> None:
        self.scalar = scalar
        self.muted = muted

    def get_scalar(self) -> float:
        return self.scalar

    def set_scalar(self, value: float) -> None:
        self.scalar = value

    def get_mute(self) -> bool:
        return self.muted

    def set_mute(self, muted: bool) -> None:
        self.muted = muted


def clamp_percent(value: float) -> int:
    """Clamp to the 0-100 range and round to a whole percent."""
    return int(max(0, min(100, round(value))))


@dataclass(frozen=True)
class VolumeResult:
    """Outcome of a volume command, used for on-screen feedback."""

    message: str
    percent: int
    muted: bool


class VolumeController:
    """Maps ninja gestures to volume actions on a :class:`VolumeBackend`."""

    def __init__(
        self,
        backend: VolumeBackend,
        step_percent: int = config.VOLUME_STEP_PERCENT,
        unmute_on_change: bool = config.UNMUTE_ON_VOLUME_CHANGE,
    ) -> None:
        self._backend = backend
        self.step_percent = step_percent
        self.unmute_on_change = unmute_on_change

    # -- queries ---------------------------------------------------------
    def get_percent(self) -> int:
        """Current master volume, 0-100."""
        return clamp_percent(self._backend.get_scalar() * 100.0)

    def is_muted(self) -> bool:
        """Whether the master volume is muted."""
        return self._backend.get_mute()

    # -- commands --------------------------------------------------------
    def change(self, delta_percent: int) -> VolumeResult:
        """Change the volume by ``delta_percent`` points, clamped to 0-100."""
        current = self.get_percent()
        target = clamp_percent(current + delta_percent)
        self._backend.set_scalar(target / 100.0)
        if self.unmute_on_change and self._backend.get_mute():
            self._backend.set_mute(False)

        if delta_percent > 0:
            message = "VOLUME MAX" if target >= 100 else "VOLUME UP"
        else:
            message = "VOLUME MIN" if target <= 0 else "VOLUME DOWN"
        return VolumeResult(message, self.get_percent(), self.is_muted())

    def toggle_mute(self) -> VolumeResult:
        """Flip the mute state."""
        muted = not self._backend.get_mute()
        self._backend.set_mute(muted)
        return VolumeResult("MUTED" if muted else "UNMUTED", self.get_percent(), muted)

    def handle_gesture(self, gesture: Gesture) -> Optional[VolumeResult]:
        """Run the action mapped to ``gesture``; ``None`` if it has no volume action."""
        if gesture == Gesture.VOLUME_UP:
            return self.change(+self.step_percent)
        if gesture == Gesture.VOLUME_DOWN:
            return self.change(-self.step_percent)
        if gesture == Gesture.MUTE_TOGGLE:
            return self.toggle_mute()
        return None


def create_real_controller() -> VolumeController:
    """Build a controller for the real Windows master volume (raises on failure)."""
    return VolumeController(PycawVolumeBackend())
