"""Typing engine, hover latch and key controller."""

from keyboard.key_controller import HoverLatch, KeyController, KeyInjector, TypingEngine
from keyboard.layout import KEY_ROWS, KeyboardLayout, KeyKind


def spec(label, kind=None):
    for row in KEY_ROWS:
        for s in row:
            if s.label == label and (kind is None or s.kind == kind):
                return s
    raise KeyError(label)


def type_keys(engine, *labels):
    for label in labels:
        engine.press(spec(label))


def test_basic_typing_space_enter_backspace():
    e = TypingEngine()
    type_keys(e, "h", "i")
    e.press(spec("SPACE"))
    type_keys(e, "y", "o")
    e.press(spec("ENTER"))
    type_keys(e, "x")
    assert e.text == "hi yo\nx"
    e.press(spec("BKSP"))
    assert e.text == "hi yo\n"
    e.press(spec("BKSP"))
    assert e.text == "hi yo"


def test_backspace_on_empty_text_is_safe():
    e = TypingEngine()
    e.press(spec("BKSP"))
    assert e.text == ""


def test_shift_is_one_shot():
    e = TypingEngine()
    e.press(spec("SHIFT"))
    type_keys(e, "a", "a")
    assert e.text == "Aa"
    assert e.shift is False


def test_shift_toggles_off_when_pressed_twice():
    e = TypingEngine()
    e.press(spec("SHIFT"))
    e.press(spec("SHIFT"))
    type_keys(e, "a")
    assert e.text == "a"


def test_caps_lock_stays_on_and_shift_inverts():
    e = TypingEngine()
    e.press(spec("CAPS"))
    type_keys(e, "a", "b")
    e.press(spec("SHIFT"))
    type_keys(e, "c")
    type_keys(e, "d")
    assert e.text == "ABcD"
    e.press(spec("CAPS"))
    type_keys(e, "e")
    assert e.text == "ABcDe"


def test_special_characters():
    e = TypingEngine()
    e.press(spec("SHIFT")); type_keys(e, "1")
    e.press(spec("SHIFT")); type_keys(e, "[")
    e.press(spec("SHIFT")); type_keys(e, ";")
    e.press(spec("SHIFT")); type_keys(e, "/")
    type_keys(e, "-", "=", "\\")
    assert e.text == "!{:?-=\\"


def test_digits_unaffected_by_caps():
    e = TypingEngine()
    e.press(spec("CAPS"))
    type_keys(e, "7")
    assert e.text == "7"


def test_clear_text():
    e = TypingEngine()
    type_keys(e, "a", "b")
    e.press(spec("CLEAR"))
    assert e.text == ""


def test_max_length_respected():
    e = TypingEngine(max_length=3)
    type_keys(e, "a", "b", "c", "d")
    assert e.text == "abc"


# ---- hover latch ----------------------------------------------------------
def test_latch_requires_dwell():
    latch = HoverLatch(min_dwell_s=0.1, lookback_s=0.25)
    latch.update(0.00, "k1")
    assert latch.resolve(0.00) is None          # just arrived
    latch.update(0.05, "k1")
    assert latch.resolve(0.05) is None
    latch.update(0.12, "k1")
    assert latch.resolve(0.12) == "k1"


def test_latch_survives_small_drift_but_expires():
    latch = HoverLatch(min_dwell_s=0.1, lookback_s=0.25)
    for t in (0.0, 0.05, 0.1, 0.15):
        latch.update(t, "k1")
    latch.update(0.20, "k2")                    # tip drifted to a neighbour while pinching
    assert latch.resolve(0.20) == "k1"
    latch.update(0.50, None)
    assert latch.resolve(0.50) is None


def test_latch_ignores_swoosh_across_keys():
    latch = HoverLatch(min_dwell_s=0.1, lookback_s=0.25)
    for i, key in enumerate(["a", "b", "c", "d", "e"]):
        latch.update(i * 0.03, key)             # 30 ms per key: never stable
    assert latch.resolve(0.15) is None


# ---- key controller -------------------------------------------------------
def test_key_controller_presses_only_on_pinch_over_valid_key():
    layout = KeyboardLayout(1280, 720)
    engine = TypingEngine()
    ctrl = KeyController(engine, KeyInjector(enabled=False))
    key_h = next(k for k in layout.keys if k.spec.label == "h")
    pos = key_h.center
    t = 0.0
    for _ in range(8):                           # hover only: nothing typed
        hovered, pressed = ctrl.update(layout, pos, False, t)
        t += 0.03
        assert pressed is None
    assert hovered is key_h and engine.text == ""
    hovered, pressed = ctrl.update(layout, pos, True, t)
    assert pressed is key_h and engine.text == "h"


def test_pinch_over_empty_space_types_nothing():
    layout = KeyboardLayout(1280, 720)
    engine = TypingEngine()
    ctrl = KeyController(engine, KeyInjector(enabled=False))
    t = 0.0
    for _ in range(8):
        ctrl.update(layout, (5, 5), False, t)
        t += 0.03
    _, pressed = ctrl.update(layout, (5, 5), True, t)
    assert pressed is None and engine.text == ""


def test_pinch_without_hand_types_nothing():
    layout = KeyboardLayout(1280, 720)
    engine = TypingEngine()
    ctrl = KeyController(engine, KeyInjector(enabled=False))
    _, pressed = ctrl.update(layout, None, True, 1.0)
    assert pressed is None


def test_injector_disabled_does_nothing_and_never_raises():
    inj = KeyInjector(enabled=False)
    from keyboard.key_controller import KeyAction
    inj.send(KeyAction(KeyKind.CHAR, "a"))
    assert inj.last_error is None
