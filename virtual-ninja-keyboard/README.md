# Virtual Ninja Keyboard

Type on a floating keyboard with your index finger and pinch, and control the
**Windows master volume** with ninja-style hand seals. Everything runs locally on
your PC: webcam -> OpenCV + MediaPipe -> your screen. No cloud, no LLM, no paid services.

## Features

* Translucent purple keyboard with lime key outlines, drawn over your live webcam image
  (QWERTY, digits 0-9, punctuation, Space, Backspace, Enter, Shift, Caps Lock, Clear).
* **Pinch-to-type**: point with your index finger to highlight a key, then touch thumb and
  index fingertip together to press it. Hovering alone never types.
* Text panel at the bottom; optional **SEND** mode also types into the app that has focus.
* **Ninja control mode**: three hand seals change the real Windows volume (+5 / -5 points)
  and toggle mute, with a large on-screen notification and a volume bar.
* Live **calibration view** with finger states, joint angles, pinch ratio and recognised gesture.
* Two-hand tracking, adaptive smoothing, mirror view, resolution-independent layout.

## Requirements

* Windows 10 or 11 (typing works anywhere; real volume control needs Windows)
* **64-bit Python 3.10 or 3.11** (MediaPipe 0.10.14, the version this project is built
  against, does not support Python 3.12+)
* A webcam, good lighting, and a plain-ish background

## Setup (Windows)

Open **PowerShell** or **Command Prompt** in the project folder:

```bat
cd virtual-ninja-keyboard
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activate script, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

(If you only have 3.10 installed, use `py -3.10 -m venv .venv`.)

## Run

```bat
python main.py
```

Useful options:

```bat
python main.py --camera 1             :: use a different webcam
python main.py --width 640 --height 480   :: faster on slow laptops
python main.py --calibration          :: start with the calibration panel
python main.py --send-keys            :: start with SEND-to-app enabled
python main.py --no-mirror            :: do not flip the image
python main.py --simulate-volume      :: developer test, does NOT change real volume
```

Windows may ask for camera permission the first time. If the image stays black, go to
**Settings > Privacy & security > Camera** and allow desktop apps to use the camera.

## Controls

| Key | Action |
|-----|--------|
| `F1` | Typing mode |
| `F2` | Ninja control mode |
| `F3` | Toggle SEND (also type into the focused app) |
| `F4` | Show/hide calibration panel |
| `ESC` | Quit |
| `1` `2` `3` `c` | Same as F1 F2 F3 F4, only while SEND is OFF (otherwise they'd be typed text) |

You can also **pinch the buttons** in the top-right corner (TYPING / NINJA / SEND), which is
how you switch modes without touching the keyboard. Every mode switch pauses input for
0.8 s so a gesture cannot leak into the other mode, and typing pinches are ignored
in Ninja mode while volume seals are ignored in Typing mode.

## How to type

1. Hold your hand up so the camera sees your whole hand. A lime skeleton appears and a
   ring follows your index fingertip.
2. Move the fingertip over a key; it lights up. Rest on it for a moment.
3. Pinch thumb and index fingertip together. The ring fills as you close the pinch, the key
   flashes white, and the character appears in the OUTPUT panel.
4. **Open your hand again before the next key.** Holding a pinch types once, by design.

Shift is one-shot (applies to the next character, then releases). Caps Lock stays on until
pressed again. Shift and Caps Lock highlight green while active. CLEAR empties the text panel.

**Typing into other programs:** press `F3` (or pinch SEND) so it reads `SEND:ON`. The window stays
on top of other windows; click your target text box first, then type with gestures. Because
the video window then no longer has keyboard focus, use the on-screen buttons to switch
modes. Note: Windows does not let normal programs type into apps running "as administrator".

## Ninja gestures (Ninja control mode)

| Seal | Pose (thumb is ignored) | Action |
|------|--------------------------|--------|
| Volume up | **index + middle** extended, ring + pinky folded | +5% |
| Volume down | **index + pinky** extended, middle + ring folded ("horns") | -5% |
| Mute / unmute | **open palm**, all four fingers extended | toggle mute |

Hold the pose steady for about 0.2 s (6 frames). The command then fires **once**; change or
drop the pose before the next one (and wait at least 1 s). Volume is clamped to 0-100%.
Changing the volume while muted unmutes (set `UNMUTE_ON_VOLUME_CHANGE = False` to disable).

> **About the reference images:** no reference images were included with the project brief,
> so the poses above were designed from scratch rather than traced from pictures. Each is
> defined purely by MediaPipe landmark geometry, picked because the three patterns are
> mutually distinct and reliably detectable by a webcam. You can change them in `config.py`
> (`GESTURE_PATTERNS`) without touching any code.

### How the poses are detected

For each finger the code measures the **angle at the middle joint (PIP)** and checks whether the
fingertip is **farther from the wrist than that joint**. Angle >= 150 degrees and tip farther away
means "extended". The pattern of (index, middle, ring, pinky) is matched against
`GESTURE_PATTERNS`. Pinch uses thumb-tip to index-tip distance divided by hand size, so it
works at any distance from the camera.

## Calibration

Press `F4` (or `c`) to open the calibration panel. It shows, live: each finger's state and
joint angle, the pinch ratio, the recognised gesture and the stability counter. Then tune
`config.py`:

| Problem | Setting |
|---------|---------|
| Pinch triggers too easily / not easily enough | `PINCH_ON_RATIO` (smaller = tighter pinch) and `PINCH_OFF_RATIO` |
| Double typing | raise `PINCH_COOLDOWN_S` |
| Presses the wrong neighbouring key | raise `HOVER_LOOKBACK_S` / `HOVER_MIN_DWELL_S` carefully, or slow down |
| A finger is mis-read as folded/extended | `FINGER_EXTENDED_ANGLE_DEG` (lower = more forgiving) |
| Gestures fire by accident | raise `GESTURE_STABLE_FRAMES` |
| Gestures feel sluggish | lower `GESTURE_STABLE_FRAMES` |
| Cursor jitters or lags | `SMOOTH_ALPHA_MIN` (lower = steadier), `SMOOTH_ALPHA_MAX` |
| Slow frame rate | lower `CAMERA_WIDTH/HEIGHT`, keep `MODEL_COMPLEXITY = 0` |

## Troubleshooting

| Message / symptom | Fix |
|-------------------|-----|
| `Could not open camera 0` | Close Zoom/Teams/browsers using the camera; allow camera access; try `--camera 1` |
| `MediaPipe ... no longer provides the 'solutions' API` | You have a newer MediaPipe: `pip install mediapipe==0.10.14` on Python 3.10/3.11 |
| `pip` cannot find `mediapipe==0.10.14` | You are on Python 3.12+ or 32-bit Python; recreate the venv with `py -3.11` |
| `volume control unavailable` | `pip install pycaw comtypes`; make sure an audio output device is enabled. Typing still works |
| Keys typed into the wrong place | Click the target text field first (SEND mode); the video window is not the target |
| SEND turns itself off | `pynput` failed (e.g. target app runs as admin); see the console message |
| Hand not detected | More light, move back so the whole hand is visible, avoid backlighting |
| ESC does nothing | Click the video window once so it has keyboard focus (or close it with the X) |
| Low FPS | `python main.py --width 640 --height 480` |

## Run the tests

```bat
pip install -r requirements-dev.txt
python -m pytest
```

The tests need no webcam and no Windows: they cover the keyboard layout and hit-testing,
pinch detection, gesture classification and stabilising, typing logic (Shift, Caps, special
characters), smoothing, volume clamping and mapping (using an in-memory fake backend, never
your real volume), and an end-to-end run of the whole app pipeline with synthetic hands.

## Project layout

```
main.py                    app loop, mode switching, camera handling
config.py                  every threshold, colour and cooldown
vision/hand_tracker.py     MediaPipe wrapper, smoothing, skeleton drawing
vision/gesture_classifier.py  finger states, gestures, PinchDetector, GestureStabilizer
vision/landmarks.py        landmark ids, HandData, geometry helpers
vision/smoothing.py        adaptive jitter filter
keyboard/layout.py         keys, rectangles, hit-testing (pure logic)
keyboard/renderer.py       translucent keyboard drawing (cached alpha blending)
keyboard/key_controller.py typing engine, hover latch, optional OS key injection
controls/volume_controller.py  pycaw backend + testable controller
ui/overlay.py, ui/draw_utils.py   panels, buttons, notifications, calibration view
tests/                     pytest suite
```

## Limitations

* A webcam gives only a 2-D image plus an estimated depth; finger angles get unreliable when
  the hand turns edge-on to the camera. Keep the palm roughly facing the camera.
* Pinching moves your fingertip slightly. The hover latch compensates, but very small keys
  (low resolution, far from the camera) are harder to hit; increase the camera resolution or
  move closer.
* Poor light, motion blur, gloves or busy backgrounds reduce tracking quality.
* "Left/Right" labelling by MediaPipe can occasionally flip; the app does not depend on it.
* Only one hand types at a time (the most confident one); the second hand is tracked and drawn.
* Windows volume control uses the default output device at the time the app starts.
* SEND mode cannot type into elevated (administrator) windows or secure desktops.
* Only ASCII characters are on the on-screen keyboard.
