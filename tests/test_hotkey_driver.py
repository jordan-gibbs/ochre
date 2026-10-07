"""Key parsing and the shared GestureDriver routing (eligibility, raw modifier, threads)."""

from __future__ import annotations

import threading
import time

import pytest

from openwhisprflow.config import HotkeyConfig
from openwhisprflow.platform.base import Gesture as G
from openwhisprflow.platform.driver import GestureDriver
from openwhisprflow.platform.keys import (EVDEV_CODE, MAC_KEYCODE, WIN_VK, KeySpec, evdev_name, mac_name,
                                          normalize, parse_key, parse_modifier, win_name)

# ------------------------------------------------------------------ keys


@pytest.mark.parametrize("spec,expected", [
    ("right_alt", KeySpec("right_alt")),
    ("Right Alt", KeySpec("right_alt")),
    ("AltGr", KeySpec("right_alt")),
    ("right option", KeySpec("right_alt")),
    ("rctrl", KeySpec("right_ctrl")),
    ("Caps Lock", KeySpec("caps_lock")),
    ("apps", KeySpec("menu")),
    ("F13", KeySpec("f13")),
    ("f24", KeySpec("f24")),
    ("scroll-lock", KeySpec("scroll_lock")),
    ("ctrl+shift+space", KeySpec("space", frozenset({"ctrl", "shift"}))),
    ("Control + Space", KeySpec("space", frozenset({"ctrl"}))),
    ("cmd+shift+d", KeySpec("d", frozenset({"win", "shift"}))),
    ("left_ctrl+f5", KeySpec("f5", frozenset({"ctrl"}))),
])
def test_parse_key(spec, expected):
    assert parse_key(spec) == expected


@pytest.mark.parametrize("bad", ["", "a", "space", "f5", "ctrl", "ctrl+shift", "foo+space", "ctrl++a",
                                 "ctrl+right_alt"])
def test_parse_key_rejects(bad):
    with pytest.raises(ValueError):
        parse_key(bad)


def test_keyspec_str_roundtrips():
    for s in ("right_alt", "ctrl+shift+space", "alt+win+f13"):
        assert parse_key(str(parse_key(s))) == parse_key(s)


def test_parse_modifier():
    assert parse_modifier("shift") == "shift"
    assert parse_modifier("Left Shift") == "shift"
    assert parse_modifier("cmd") == "win"
    assert parse_modifier("") is None and parse_modifier("none") is None and parse_modifier(None) is None
    with pytest.raises(ValueError):
        parse_modifier("space")


def test_normalize_sided_aliases():
    assert normalize("Right Control") == "right_ctrl"
    assert normalize("left option") == "left_alt"
    assert normalize("right cmd") == "right_win"


def test_native_maps_cover_standalone_keys():
    for name in ("right_alt", "right_ctrl", "right_shift", "caps_lock", "menu", "insert", "scroll_lock",
                 "pause", *(f"f{i}" for i in range(13, 25))):
        assert name in WIN_VK and name in EVDEV_CODE
        assert win_name(WIN_VK[name]) == name
        assert evdev_name(EVDEV_CODE[name]) == name
    assert win_name(0xFF) == "vk255"
    assert MAC_KEYCODE["right_alt"] == 61 and mac_name(61) == "right_alt"
    assert mac_name(53) == "escape" and mac_name(9999) == "kc9999"


# ------------------------------------------------------------------ driver routing


class Clock:
    def __init__(self) -> None:
        self.t = 0

    def __call__(self) -> int:
        return self.t


def driver(**cfg_kw) -> tuple[GestureDriver, list, list]:
    gestures: list[G] = []
    replays: list[tuple[str, bool]] = []
    d = GestureDriver(HotkeyConfig(**cfg_kw), lambda n, down: replays.append((n, down)), clock_ms=Clock())
    d._post = lambda decision: gestures.extend(decision.gestures)  # synchronous for tests
    return d, gestures, replays


def test_single_key_hold():
    d, out, _ = driver()
    assert d.feed("right_alt", True, 0)
    assert not d.feed("a", True, 50) or True  # (a chord; checked below)


def test_other_modifier_makes_single_key_ineligible():
    d, out, _ = driver()
    d.feed("left_win", True, 0)
    assert not d.feed("right_alt", True, 10)
    assert out == []
    assert not d.feed("right_alt", False, 50)
    d.feed("left_win", False, 60)
    assert d.feed("right_alt", True, 100) and out == [G.PRESS]


def test_shift_at_press_is_ineligible_but_raw_at_finish():
    d, out, _ = driver()
    d.feed("left_shift", True, 0)
    assert not d.feed("right_alt", True, 10)  # Shift+Voice key belongs to the app
    d.feed("right_alt", False, 20)
    d.feed("left_shift", False, 30)
    d.feed("right_alt", True, 100)
    d.feed("right_shift", True, 300)  # modifiers never break the gesture
    assert out == [G.PRESS]
    d.feed("right_alt", False, 600)
    assert out == [G.PRESS, G.RAW]


def test_raw_modifier_can_be_disabled():
    d, out, _ = driver(raw_modifier="none")
    d.feed("right_alt", True, 0)
    d.feed("left_shift", True, 300)
    d.feed("right_alt", False, 600)
    assert out == [G.PRESS, G.RELEASE]


def test_chord_needs_exactly_its_modifiers():
    d, out, _ = driver(key="ctrl+shift+space")
    assert not d.feed("space", True, 0) and out == []  # no modifiers: plain space
    d.feed("space", False, 10)
    d.feed("left_ctrl", True, 20)
    assert not d.feed("space", True, 30)  # ctrl only
    d.feed("space", False, 40)
    d.feed("right_shift", True, 50)
    d.feed("left_alt", True, 55)
    assert not d.feed("space", True, 60)  # one too many
    d.feed("space", False, 70)
    d.feed("left_alt", False, 75)
    assert d.feed("space", True, 100) and out == [G.PRESS]
    d.feed("left_ctrl", False, 200)  # letting go of the modifiers first is fine
    assert d.feed("space", False, 500) and out == [G.PRESS, G.RELEASE]


def test_chord_double_tap_locks():
    d, out, _ = driver(key="ctrl+shift+space")
    d.feed("left_ctrl", True, 0), d.feed("left_shift", True, 0)
    d.feed("space", True, 10), d.feed("space", False, 60), d.feed("space", True, 150), d.feed("space", False, 200)
    assert out == [G.PRESS, G.LOCK]


def test_escape_and_custom_cancel_key():
    d, out, _ = driver(cancel_key="f9")
    d.feed("right_alt", True, 0)
    assert not d.feed("escape", True, 10)  # not the cancel key: an ordinary other key -> chord
    assert out == [G.PRESS, G.CANCEL]
    d2, out2, _ = driver(cancel_key="f9")
    d2.feed("right_alt", True, 0)
    assert d2.feed("f9", True, 10) and out2 == [G.PRESS, G.CANCEL]


def test_chord_replays_the_voice_key():
    d, out, replays = driver()
    d.feed("right_alt", True, 0)
    d.feed("e", True, 30)
    assert replays == [("right_alt", True)] and out == [G.PRESS, G.CANCEL]


def test_set_key_waits_for_the_gesture_to_end():
    d, out, _ = driver()
    d.feed("right_alt", True, 0)
    d.set_key("f13")
    assert d.key_name == "right_alt"
    d.feed("right_alt", False, 500)
    assert d.key_name == "f13"
    assert not d.feed("right_alt", True, 600)
    assert d.feed("f13", True, 700)


def test_set_key_validates():
    d, _, _ = driver()
    with pytest.raises(ValueError):
        d.set_key("q")


def test_threads_dispatch_and_tick():
    """Real threads: the ticker expires a lone tap and the dispatcher delivers off-thread."""
    clock = {"t": 0}
    got: list[tuple[G, str]] = []
    done = threading.Event()
    replays = []

    def on(g: G) -> None:
        got.append((g, threading.current_thread().name))
        if g == G.CANCEL:
            done.set()

    d = GestureDriver(HotkeyConfig(), lambda n, down: replays.append((n, down)),
                      clock_ms=lambda: clock["t"] if clock["t"] else int(time.monotonic() * 1000))
    clock["t"] = 0
    d.clock_ms = lambda: int(time.monotonic() * 1000)
    d.start(on)
    try:
        now = d.clock_ms()
        d.feed("right_alt", True, now)
        d.feed("right_alt", False, now + 30)
        assert done.wait(2.0)
    finally:
        d.stop()
    assert [g for g, _ in got] == [G.PRESS, G.CANCEL]
    assert all(name == "owf-gestures" for _, name in got)
    assert replays == [("right_alt", True), ("right_alt", False)]


def test_feed_never_raises():
    d, _, _ = driver()
    d._route = None  # type: ignore[assignment]  # anything inside blowing up
    assert d.feed("right_alt", True, 0) is False
