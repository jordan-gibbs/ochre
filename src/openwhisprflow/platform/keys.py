"""Voice key names: parsing the config string and mapping names to native key codes.

Every OS adapter translates native events into the *sided* names used here ("right_alt",
"left_ctrl", "f13", "a", "space", ...), so the gesture routing logic is written once
(:mod:`openwhisprflow.platform.driver`). Unknown native keys get an opaque name ("vk123",
"kc99", "ev300") and count as ordinary non-modifier keys.

A *single* Voice key must be safe to hold and to double-tap without side effects, which is why
only the keys below are accepted on their own. Anything else needs at least one modifier, e.g.
``"ctrl+shift+space"``; a chord uses hold semantics on its last key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MOD_CLASSES = ("ctrl", "shift", "alt", "win")

# Sided modifier name -> modifier class. "win" is Cmd on macOS and Super on Linux.
SIDED_MODIFIERS: dict[str, str] = {
    "left_ctrl": "ctrl", "right_ctrl": "ctrl",
    "left_shift": "shift", "right_shift": "shift",
    "left_alt": "alt", "right_alt": "alt",
    "left_win": "win", "right_win": "win",
}

FUNCTION_KEYS = tuple(f"f{i}" for i in range(1, 25))

# Keys accepted as a Voice key on their own.
STANDALONE = frozenset({"right_alt", "right_ctrl", "right_shift", "right_win", "caps_lock", "menu",
                        "insert", "scroll_lock", "pause", "fn", *(f"f{i}" for i in range(13, 25))})

_ALIASES: dict[str, str] = {
    # modifier classes
    "control": "ctrl", "ctl": "ctrl", "option": "alt", "opt": "alt", "cmd": "win", "command": "win",
    "super": "win", "meta": "win", "windows": "win", "logo": "win",
    # sided keys
    "ralt": "right_alt", "altgr": "right_alt", "alt_gr": "right_alt", "right_option": "right_alt",
    "roption": "right_alt", "ropt": "right_alt", "alt_r": "right_alt",
    "rctrl": "right_ctrl", "right_control": "right_ctrl", "ctrl_r": "right_ctrl",
    "rshift": "right_shift", "shift_r": "right_shift",
    "rcmd": "right_win", "right_cmd": "right_win", "right_command": "right_win", "right_super": "right_win",
    "lalt": "left_alt", "lctrl": "left_ctrl", "lshift": "left_shift", "lwin": "left_win",
    "left_option": "left_alt", "left_control": "left_ctrl", "left_cmd": "left_win",
    # others
    "capslock": "caps_lock", "caps": "caps_lock", "apps": "menu", "context_menu": "menu",
    "application": "menu", "ins": "insert", "scrolllock": "scroll_lock", "scroll": "scroll_lock",
    "break": "pause", "esc": "escape", "return": "enter", "spacebar": "space", "globe": "fn",
    "del": "delete", "pgup": "page_up", "pgdn": "page_down", "bksp": "backspace",
}


@dataclass(frozen=True)
class KeySpec:
    """A parsed Voice key: the main key plus required modifier classes (empty = single key)."""

    key: str
    modifiers: frozenset[str] = frozenset()

    @property
    def is_chord(self) -> bool:
        return bool(self.modifiers)

    def __str__(self) -> str:
        return "+".join([*(m for m in MOD_CLASSES if m in self.modifiers), self.key])


def normalize(name: str) -> str:
    """Canonical form of one key name: lowercase, ``_`` separated, aliases resolved."""
    n = re.sub(r"[\s\-]+", "_", (name or "").strip().lower())
    n = _ALIASES.get(n, n)
    if n.startswith(("left_", "right_")):
        side, _, rest = n.partition("_")
        rest = _ALIASES.get(rest, rest)
        if rest in MOD_CLASSES:
            n = f"{side}_{rest}"
    return n


def modifier_class(name: str) -> str | None:
    """"ctrl"/"shift"/"alt"/"win" for a sided modifier name, else None."""
    return SIDED_MODIFIERS.get(name)


def parse_key(spec: str) -> KeySpec:
    """Parse a config string ("right_alt", "f13", "ctrl+shift+space"). Raises ValueError."""
    parts = [normalize(p) for p in (spec or "").split("+")]
    if not parts or any(not p for p in parts):
        raise ValueError(f"empty or malformed hotkey {spec!r}")
    *mods, key = parts
    classes: set[str] = set()
    for m in mods:
        cls = m if m in MOD_CLASSES else modifier_class(m)
        if cls is None:
            raise ValueError(f"{m!r} in {spec!r} is not a modifier (ctrl, shift, alt, win)")
        classes.add(cls)
    if key in MOD_CLASSES or (mods and key in SIDED_MODIFIERS):
        raise ValueError(f"{spec!r}: the last key of a chord must not be a modifier")
    if not classes and key not in STANDALONE:
        raise ValueError(
            f"{key!r} is not safe as a Voice key on its own; pick one of "
            f"{', '.join(sorted(STANDALONE))}, or a chord such as ctrl+shift+space")
    return KeySpec(key, frozenset(classes))


def parse_modifier(name: str | None) -> str | None:
    """The raw-modifier setting as a modifier class; "", "none" or "off" disable it."""
    n = normalize(name or "")
    if n in ("", "none", "off", "disabled"):
        return None
    cls = n if n in MOD_CLASSES else modifier_class(n)
    if cls is None:
        raise ValueError(f"raw modifier must be one of {', '.join(MOD_CLASSES)}, got {name!r}")
    return cls


# --------------------------------------------------------------------------- Windows virtual keys

WIN_VK: dict[str, int] = {
    "left_shift": 0xA0, "right_shift": 0xA1, "left_ctrl": 0xA2, "right_ctrl": 0xA3,
    "left_alt": 0xA4, "right_alt": 0xA5, "left_win": 0x5B, "right_win": 0x5C,
    "caps_lock": 0x14, "menu": 0x5D, "insert": 0x2D, "delete": 0x2E, "scroll_lock": 0x91, "pause": 0x13,
    "escape": 0x1B, "space": 0x20, "enter": 0x0D, "tab": 0x09, "backspace": 0x08,
    "home": 0x24, "end": 0x23, "page_up": 0x21, "page_down": 0x22,
    "left": 0x25, "up": 0x26, "right": 0x27, "down": 0x28,
    **{f"f{i}": 0x6F + i for i in range(1, 25)},
    **{chr(c): c - 32 for c in range(ord("a"), ord("z") + 1)},
    **{str(d): 0x30 + d for d in range(10)},
}
# The generic VKs some apps and older drivers report instead of the sided ones.
WIN_GENERIC_VK: dict[int, str] = {0x10: "left_shift", 0x11: "left_ctrl", 0x12: "left_alt"}
WIN_EXTENDED = frozenset({0xA3, 0xA5, 0x5B, 0x5C, 0x5D, 0x2D, 0x2E, 0x24, 0x23, 0x21, 0x22,
                          0x25, 0x26, 0x27, 0x28, 0x90})
WIN_NAME: dict[int, str] = {**{v: k for k, v in WIN_VK.items()}, **WIN_GENERIC_VK}


def win_name(vk: int) -> str:
    return WIN_NAME.get(vk, f"vk{vk}")


# --------------------------------------------------------------------------- macOS virtual key codes

# kVK_* from Carbon HIToolbox/Events.h (ANSI layout positions). Right Alt is Right Option and
# Insert/Scroll Lock/Pause map to the keys PC keyboards produce on a Mac (Help, F14, F15).
MAC_KEYCODE: dict[str, int] = {
    "right_alt": 61, "left_alt": 58, "right_ctrl": 62, "left_ctrl": 59, "right_shift": 60,
    "left_shift": 56, "right_win": 54, "left_win": 55, "fn": 63, "caps_lock": 57,
    "insert": 114, "scroll_lock": 107, "pause": 113, "escape": 53, "space": 49, "enter": 36,
    "tab": 48, "backspace": 51, "delete": 117, "home": 115, "end": 119, "page_up": 116,
    "page_down": 121, "left": 123, "right": 124, "down": 125, "up": 126,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97, "f7": 98, "f8": 100, "f9": 101,
    "f10": 109, "f11": 103, "f12": 111, "f13": 105, "f14": 107, "f15": 113, "f16": 106,
    "f17": 64, "f18": 79, "f19": 80, "f20": 90,
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11,
    "q": 12, "w": 13, "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21,
    "6": 22, "5": 23, "9": 25, "7": 26, "8": 28, "0": 29, "o": 31, "u": 32, "i": 34, "p": 35,
    "l": 37, "j": 38, "k": 40, "n": 45, "m": 46,
}
# Several names share a code (insert/help, scroll_lock/f14, pause/f15): prefer the F-key name
# when reading events, the config name only matters for the lookup above.
MAC_NAME: dict[int, str] = {}
for _n, _c in MAC_KEYCODE.items():
    MAC_NAME.setdefault(_c, _n)
MAC_NAME.update({107: "f14", 113: "f15", 114: "insert"})
MAC_UNSUPPORTED = {
    "caps_lock": "macOS reports Caps Lock only as a toggle, never its release, so it cannot be held",
    "menu": "Mac keyboards have no Menu key",
    **{f"f{i}": "macOS has no key code for it" for i in range(21, 25)},
}


def mac_name(code: int) -> str:
    return MAC_NAME.get(code, f"kc{code}")


# --------------------------------------------------------------------------- Linux evdev codes

EVDEV_CODE: dict[str, int] = {
    "escape": 1, **{str(d): 1 + d for d in range(1, 10)}, "0": 11, "backspace": 14, "tab": 15,
    "q": 16, "w": 17, "e": 18, "r": 19, "t": 20, "y": 21, "u": 22, "i": 23, "o": 24, "p": 25,
    "enter": 28, "left_ctrl": 29, "a": 30, "s": 31, "d": 32, "f": 33, "g": 34, "h": 35, "j": 36,
    "k": 37, "l": 38, "left_shift": 42, "z": 44, "x": 45, "c": 46, "v": 47, "b": 48, "n": 49,
    "m": 50, "right_shift": 54, "left_alt": 56, "space": 57, "caps_lock": 58,
    **{f"f{i}": 58 + i for i in range(1, 11)}, "scroll_lock": 70, "f11": 87, "f12": 88,
    "right_ctrl": 97, "right_alt": 100, "home": 102, "up": 103, "page_up": 104, "left": 105,
    "right": 106, "end": 107, "down": 108, "page_down": 109, "insert": 110, "delete": 111,
    "pause": 119, "left_win": 125, "right_win": 126, "menu": 127,
    **{f"f{i}": 170 + i for i in range(13, 25)},
}
EVDEV_NAME: dict[int, str] = {v: k for k, v in EVDEV_CODE.items()}
EVDEV_NAME[139] = "menu"  # KEY_MENU, which some keyboards send instead of KEY_COMPOSE


def evdev_name(code: int) -> str:
    return EVDEV_NAME.get(code, f"ev{code}")
