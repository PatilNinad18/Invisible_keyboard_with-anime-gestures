"""Keyboard layout and key-boundary detection."""

import itertools

import pytest

from keyboard.layout import KEY_ROWS, KeyboardLayout, KeyKind, display_label, resolve_char

SIZES = [(640, 480), (800, 600), (1280, 720), (1920, 1080), (1024, 768), (640, 360)]


@pytest.mark.parametrize("size", SIZES)
def test_all_keys_inside_frame(size):
    w, h = size
    layout = KeyboardLayout(w, h)
    for key in layout.keys:
        assert key.x >= 0 and key.y >= 0
        assert key.x + key.w <= w and key.y + key.h <= h
        assert key.w > 4 and key.h > 4


@pytest.mark.parametrize("size", SIZES)
def test_keys_do_not_overlap(size):
    layout = KeyboardLayout(*size)
    for a, b in itertools.combinations(layout.keys, 2):
        separated = (a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y)
        assert separated, f"{a.key_id} overlaps {b.key_id}"


@pytest.mark.parametrize("size", SIZES)
def test_every_key_reachable_at_its_center(size):
    layout = KeyboardLayout(*size)
    for key in layout.keys:
        assert layout.key_at(*key.center) is key


def test_points_outside_keyboard_hit_nothing():
    layout = KeyboardLayout(1280, 720)
    assert layout.key_at(-5, -5) is None
    assert layout.key_at(5, 5) is None
    assert layout.key_at(640, 715) is None


def test_gap_between_keys_still_resolves_to_a_neighbour():
    layout = KeyboardLayout(1280, 720)
    a = layout.keys[1]
    b = layout.keys[2]
    mid_x = (a.x + a.w + b.x) / 2.0
    assert layout.key_at(mid_x, a.center[1]) in (a, b)


def test_required_keys_exist():
    labels = {k.spec.label for k in layout_keys()}
    for ch in "abcdefghijklmnopqrstuvwxyz0123456789":
        assert ch in labels
    for ch in "`-=[]\\;',./":
        assert ch in labels
    kinds = {k.spec.kind for k in layout_keys()}
    assert kinds >= {KeyKind.SPACE, KeyKind.BACKSPACE, KeyKind.ENTER, KeyKind.SHIFT,
                     KeyKind.CAPS, KeyKind.CLEAR}


def layout_keys():
    return KeyboardLayout(1280, 720).keys


def test_unique_ids():
    ids = [k.key_id for k in layout_keys()]
    assert len(ids) == len(set(ids))


def test_row_count_and_key_counts():
    assert len(KEY_ROWS) == 5


def test_invalid_size_rejected():
    with pytest.raises(ValueError):
        KeyboardLayout(0, 480)


def test_resolve_char_rules():
    a = next(s for row in KEY_ROWS for s in row if s.label == "a")
    one = next(s for row in KEY_ROWS for s in row if s.label == "1")
    assert resolve_char(a, False, False) == "a"
    assert resolve_char(a, True, False) == "A"
    assert resolve_char(a, False, True) == "A"
    assert resolve_char(a, True, True) == "a"        # shift cancels caps for letters
    assert resolve_char(one, False, True) == "1"     # caps does not affect digits
    assert resolve_char(one, True, False) == "!"
    assert display_label(one, True, False) == "!"
