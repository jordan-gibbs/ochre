//! Voice key names: parsing the config string and mapping native key codes to names.
//!
//! Every OS adapter translates native events into the *sided* names used here (`"right_alt"`,
//! `"left_ctrl"`, `"f13"`, `"a"`, `"space"`, ...), so the routing logic in [`crate::driver`] is
//! written once. Native keys without a name map to [`OTHER`] and count as ordinary
//! non-modifier keys. All names are `&'static str`, so the hook path never allocates.
//!
//! A *single* Voice key must be safe to hold and to double-tap without side effects, which is
//! why only [`STANDALONE`] keys are accepted on their own. Anything else needs at least one
//! modifier (`"ctrl+shift+space"`); a chord uses hold semantics on its last key.

/// Modifier classes as bits. `WIN` is Cmd on macOS and Super on Linux.
pub mod mods {
    pub const CTRL: u8 = 1;
    pub const SHIFT: u8 = 2;
    pub const ALT: u8 = 4;
    pub const WIN: u8 = 8;
}

/// Name for any key we have no name for.
pub const OTHER: &str = "other";

/// Sided modifier names, in a fixed order (index = slot in the driver's held-key table).
pub const SIDED_MODIFIERS: [(&str, u8); 8] = [
    ("left_ctrl", mods::CTRL),
    ("right_ctrl", mods::CTRL),
    ("left_shift", mods::SHIFT),
    ("right_shift", mods::SHIFT),
    ("left_alt", mods::ALT),
    ("right_alt", mods::ALT),
    ("left_win", mods::WIN),
    ("right_win", mods::WIN),
];

/// Keys accepted as a Voice key on their own.
pub const STANDALONE: &[&str] = &[
    "right_alt",
    "right_ctrl",
    "right_shift",
    "right_win",
    "caps_lock",
    "menu",
    "insert",
    "scroll_lock",
    "pause",
    "fn",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
    "f20",
    "f21",
    "f22",
    "f23",
    "f24",
];

/// Every key name we know (the canonical, interned spelling).
const NAMES: &[&str] = &[
    "left_ctrl",
    "right_ctrl",
    "left_shift",
    "right_shift",
    "left_alt",
    "right_alt",
    "left_win",
    "right_win",
    "caps_lock",
    "menu",
    "insert",
    "delete",
    "scroll_lock",
    "pause",
    "fn",
    "escape",
    "space",
    "enter",
    "tab",
    "backspace",
    "home",
    "end",
    "page_up",
    "page_down",
    "left",
    "up",
    "right",
    "down",
    "f1",
    "f2",
    "f3",
    "f4",
    "f5",
    "f6",
    "f7",
    "f8",
    "f9",
    "f10",
    "f11",
    "f12",
    "f13",
    "f14",
    "f15",
    "f16",
    "f17",
    "f18",
    "f19",
    "f20",
    "f21",
    "f22",
    "f23",
    "f24",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
];

const ALIASES: &[(&str, &str)] = &[
    ("ralt", "right_alt"),
    ("altgr", "right_alt"),
    ("alt_gr", "right_alt"),
    ("right_option", "right_alt"),
    ("roption", "right_alt"),
    ("ropt", "right_alt"),
    ("alt_r", "right_alt"),
    ("rctrl", "right_ctrl"),
    ("right_control", "right_ctrl"),
    ("ctrl_r", "right_ctrl"),
    ("rshift", "right_shift"),
    ("shift_r", "right_shift"),
    ("rwin", "right_win"),
    ("rcmd", "right_win"),
    ("right_cmd", "right_win"),
    ("right_command", "right_win"),
    ("right_super", "right_win"),
    ("lalt", "left_alt"),
    ("left_option", "left_alt"),
    ("lctrl", "left_ctrl"),
    ("left_control", "left_ctrl"),
    ("lshift", "left_shift"),
    ("lwin", "left_win"),
    ("left_cmd", "left_win"),
    ("left_command", "left_win"),
    ("capslock", "caps_lock"),
    ("caps", "caps_lock"),
    ("apps", "menu"),
    ("context_menu", "menu"),
    ("application", "menu"),
    ("ins", "insert"),
    ("scrolllock", "scroll_lock"),
    ("scroll", "scroll_lock"),
    ("break", "pause"),
    ("esc", "escape"),
    ("return", "enter"),
    ("spacebar", "space"),
    ("globe", "fn"),
    ("del", "delete"),
    ("pgup", "page_up"),
    ("pgdn", "page_down"),
    ("bksp", "backspace"),
];

const CLASS_ALIASES: &[(&str, u8)] = &[
    ("ctrl", mods::CTRL),
    ("control", mods::CTRL),
    ("ctl", mods::CTRL),
    ("shift", mods::SHIFT),
    ("alt", mods::ALT),
    ("option", mods::ALT),
    ("opt", mods::ALT),
    ("win", mods::WIN),
    ("cmd", mods::WIN),
    ("command", mods::WIN),
    ("super", mods::WIN),
    ("meta", mods::WIN),
    ("windows", mods::WIN),
    ("logo", mods::WIN),
];

/// A parsed Voice key: the main key plus the modifier classes that must be held with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeySpec {
    pub key: &'static str,
    /// Bits from [`mods`]; 0 = a single key.
    pub mods: u8,
}

impl KeySpec {
    pub fn is_chord(&self) -> bool {
        self.mods != 0
    }
}

impl std::fmt::Display for KeySpec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (bit, name) in [
            (mods::CTRL, "ctrl"),
            (mods::SHIFT, "shift"),
            (mods::ALT, "alt"),
            (mods::WIN, "win"),
        ] {
            if self.mods & bit != 0 {
                write!(f, "{name}+")?;
            }
        }
        f.write_str(self.key)
    }
}

/// The interned spelling of a known key name.
pub fn intern(name: &str) -> Option<&'static str> {
    NAMES.iter().copied().find(|n| *n == name)
}

fn squeeze(name: &str) -> String {
    let mut out = String::new();
    for ch in name.trim().chars() {
        if ch.is_whitespace() || ch == '-' {
            if !out.ends_with('_') {
                out.push('_');
            }
        } else {
            out.extend(ch.to_lowercase());
        }
    }
    out
}

/// Canonical form of one key name (lowercase, `_` separated, aliases resolved). Returns the
/// normalized string even when the key is unknown.
pub fn normalize(name: &str) -> String {
    let n = squeeze(name);
    if let Some((_, to)) = ALIASES.iter().find(|(from, _)| *from == n) {
        return (*to).to_string();
    }
    for side in ["left_", "right_"] {
        if let Some(rest) = n.strip_prefix(side)
            && let Some(class) = class_of(rest)
        {
            let base = match class {
                mods::CTRL => "ctrl",
                mods::SHIFT => "shift",
                mods::ALT => "alt",
                _ => "win",
            };
            return format!("{side}{base}");
        }
    }
    n
}

fn class_of(word: &str) -> Option<u8> {
    CLASS_ALIASES
        .iter()
        .find(|(w, _)| *w == word)
        .map(|(_, c)| *c)
}

/// Modifier class bit of a sided modifier name, or 0.
pub fn modifier_class(name: &str) -> u8 {
    SIDED_MODIFIERS
        .iter()
        .find(|(n, _)| *n == name)
        .map_or(0, |(_, c)| *c)
}

/// Slot of a sided modifier in [`SIDED_MODIFIERS`].
pub fn modifier_slot(name: &str) -> Option<usize> {
    SIDED_MODIFIERS.iter().position(|(n, _)| *n == name)
}

/// Parse a config string (`"right_alt"`, `"f13"`, `"ctrl+shift+space"`).
pub fn parse_key(spec: &str) -> Result<KeySpec, String> {
    let parts: Vec<String> = spec.split('+').map(normalize).collect();
    if parts.iter().any(|p| p.is_empty()) {
        return Err(format!("empty or malformed hotkey {spec:?}"));
    }
    let (last, mod_parts) = parts
        .split_last()
        .ok_or_else(|| format!("empty hotkey {spec:?}"))?;
    let mut mods = 0u8;
    for m in mod_parts {
        let class = class_of(m).unwrap_or_else(|| modifier_class(m));
        if class == 0 {
            return Err(format!(
                "{m:?} in {spec:?} is not a modifier (ctrl, shift, alt, win)"
            ));
        }
        mods |= class;
    }
    if class_of(last).is_some() || (mods != 0 && modifier_class(last) != 0) {
        return Err(format!(
            "{spec:?}: the last key of a chord must not be a modifier"
        ));
    }
    let key = intern(last).ok_or_else(|| format!("unknown key {last:?} in {spec:?}"))?;
    if mods == 0 && !STANDALONE.contains(&key) {
        return Err(format!(
            "{key:?} is not safe as a Voice key on its own; pick one of {}, or a chord such as ctrl+shift+space",
            STANDALONE.join(", ")
        ));
    }
    Ok(KeySpec { key, mods })
}

/// The raw-modifier setting as a class bit; `""`, `"none"`, `"off"` disable it (0).
pub fn parse_modifier(name: &str) -> Result<u8, String> {
    let n = normalize(name);
    if matches!(n.as_str(), "" | "none" | "off" | "disabled") {
        return Ok(0);
    }
    let class = class_of(&n).unwrap_or_else(|| modifier_class(&n));
    if class == 0 {
        return Err(format!(
            "raw modifier must be one of ctrl, shift, alt, win; got {name:?}"
        ));
    }
    Ok(class)
}

/// The paste-last shortcut (`hotkey.paste_last`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PasteKey {
    Off,
    /// Pressed while the Voice key is held ("down": Voice key + Down).
    WithVoice(&'static str),
    /// A standalone chord ("ctrl+alt+v").
    Combo(KeySpec),
}

/// Parse `hotkey.paste_last`: `""` / `"off"` disable it, a plain key name pairs it with the
/// Voice key, and a chord with modifiers is a shortcut of its own.
pub fn parse_paste_key(spec: &str) -> Result<PasteKey, String> {
    let n = normalize(spec);
    if matches!(n.as_str(), "" | "none" | "off" | "disabled") {
        return Ok(PasteKey::Off);
    }
    if spec.contains('+') {
        let k = parse_key(spec)?;
        if !k.is_chord() {
            return Err(format!("{spec:?}: a paste-last shortcut needs a modifier"));
        }
        return Ok(PasteKey::Combo(k));
    }
    let key = intern(&n).ok_or_else(|| format!("unknown key {spec:?} for paste-last"))?;
    if modifier_class(key) != 0 || key == "escape" {
        return Err(format!(
            "{key:?} can't be paired with the Voice key for paste-last; try down or v"
        ));
    }
    Ok(PasteKey::WithVoice(key))
}

// ------------------------------------------------------------------------------------------
// Windows virtual-key codes (the low-level hook always reports sided modifiers).

pub fn win_vk(name: &str) -> Option<u16> {
    Some(match name {
        "left_shift" => 0xA0,
        "right_shift" => 0xA1,
        "left_ctrl" => 0xA2,
        "right_ctrl" => 0xA3,
        "left_alt" => 0xA4,
        "right_alt" => 0xA5,
        "left_win" => 0x5B,
        "right_win" => 0x5C,
        "caps_lock" => 0x14,
        "menu" => 0x5D,
        "insert" => 0x2D,
        "delete" => 0x2E,
        "scroll_lock" => 0x91,
        "pause" => 0x13,
        "escape" => 0x1B,
        "space" => 0x20,
        "enter" => 0x0D,
        "tab" => 0x09,
        "backspace" => 0x08,
        "home" => 0x24,
        "end" => 0x23,
        "page_up" => 0x21,
        "page_down" => 0x22,
        "left" => 0x25,
        "up" => 0x26,
        "right" => 0x27,
        "down" => 0x28,
        _ => {
            let b = name.as_bytes();
            if let Some(n) = name.strip_prefix('f').and_then(|d| d.parse::<u16>().ok())
                && (1..=24).contains(&n)
            {
                return Some(0x6F + n);
            }
            if b.len() == 1 && b[0].is_ascii_lowercase() {
                return Some(u16::from(b[0] - 32));
            }
            if b.len() == 1 && b[0].is_ascii_digit() {
                return Some(u16::from(b[0]));
            }
            return None;
        }
    })
}

pub fn win_name(vk: u32) -> &'static str {
    match vk {
        0xA0 | 0x10 => "left_shift",
        0xA1 => "right_shift",
        0xA2 | 0x11 => "left_ctrl",
        0xA3 => "right_ctrl",
        0xA4 | 0x12 => "left_alt",
        0xA5 => "right_alt",
        0x5B => "left_win",
        0x5C => "right_win",
        0x14 => "caps_lock",
        0x5D => "menu",
        0x2D => "insert",
        0x2E => "delete",
        0x91 => "scroll_lock",
        0x13 => "pause",
        0x1B => "escape",
        0x20 => "space",
        0x0D => "enter",
        0x09 => "tab",
        0x08 => "backspace",
        0x24 => "home",
        0x23 => "end",
        0x21 => "page_up",
        0x22 => "page_down",
        0x25 => "left",
        0x26 => "up",
        0x27 => "right",
        0x28 => "down",
        0x70..=0x87 => {
            NAMES[NAMES.iter().position(|n| *n == "f1").unwrap_or(0) + (vk - 0x70) as usize]
        }
        0x41..=0x5A => {
            NAMES[NAMES.iter().position(|n| *n == "a").unwrap_or(0) + (vk - 0x41) as usize]
        }
        0x30..=0x39 => {
            NAMES[NAMES.iter().position(|n| *n == "0").unwrap_or(0) + (vk - 0x30) as usize]
        }
        _ => OTHER,
    }
}

/// Keys whose scan code carries the E0 prefix, so a replay reads like the real key.
pub fn win_extended(vk: u16) -> bool {
    matches!(
        vk,
        0xA3 | 0xA5 | 0x5B | 0x5C | 0x5D | 0x2D | 0x2E | 0x24 | 0x23 | 0x21 | 0x22 | 0x25
            ..=0x28 | 0x90
    )
}

// ------------------------------------------------------------------------------------------
// macOS virtual key codes (kVK_* from Carbon HIToolbox/Events.h, ANSI positions).

/// macOS cannot use these as the Voice key, with the reason.
pub fn mac_unsupported(name: &str) -> Option<&'static str> {
    match name {
        "caps_lock" => Some(
            "macOS reports Caps Lock only as a toggle, never its release, so it cannot be held",
        ),
        "menu" => Some("Mac keyboards have no Menu key"),
        "f21" | "f22" | "f23" | "f24" => Some("macOS has no key code for it"),
        _ => None,
    }
}

pub fn mac_keycode(name: &str) -> Option<u16> {
    Some(match name {
        "right_alt" => 61,
        "left_alt" => 58,
        "right_ctrl" => 62,
        "left_ctrl" => 59,
        "right_shift" => 60,
        "left_shift" => 56,
        "right_win" => 54,
        "left_win" => 55,
        "fn" => 63,
        "caps_lock" => 57,
        "insert" => 114, // Help, where PC keyboards put Insert
        "scroll_lock" => 107,
        "pause" => 113,
        "escape" => 53,
        "space" => 49,
        "enter" => 36,
        "tab" => 48,
        "backspace" => 51,
        "delete" => 117,
        "home" => 115,
        "end" => 119,
        "page_up" => 116,
        "page_down" => 121,
        "left" => 123,
        "right" => 124,
        "down" => 125,
        "up" => 126,
        "f1" => 122,
        "f2" => 120,
        "f3" => 99,
        "f4" => 118,
        "f5" => 96,
        "f6" => 97,
        "f7" => 98,
        "f8" => 100,
        "f9" => 101,
        "f10" => 109,
        "f11" => 103,
        "f12" => 111,
        "f13" => 105,
        "f14" => 107,
        "f15" => 113,
        "f16" => 106,
        "f17" => 64,
        "f18" => 79,
        "f19" => 80,
        "f20" => 90,
        "a" => 0,
        "s" => 1,
        "d" => 2,
        "f" => 3,
        "h" => 4,
        "g" => 5,
        "z" => 6,
        "x" => 7,
        "c" => 8,
        "v" => 9,
        "b" => 11,
        "q" => 12,
        "w" => 13,
        "e" => 14,
        "r" => 15,
        "y" => 16,
        "t" => 17,
        "1" => 18,
        "2" => 19,
        "3" => 20,
        "4" => 21,
        "6" => 22,
        "5" => 23,
        "9" => 25,
        "7" => 26,
        "8" => 28,
        "0" => 29,
        "o" => 31,
        "u" => 32,
        "i" => 34,
        "p" => 35,
        "l" => 37,
        "j" => 38,
        "k" => 40,
        "n" => 45,
        "m" => 46,
        _ => return None,
    })
}

/// Name of a macOS key code. Codes shared by two names resolve to the one the config uses:
/// 107 and 113 are both F14/F15 and Scroll Lock/Pause, so the configured Voice key wins via
/// [`mac_matches`]; this returns the F-key spelling.
pub fn mac_name(code: u16) -> &'static str {
    const ORDER: &[&str] = &[
        "right_alt",
        "left_alt",
        "right_ctrl",
        "left_ctrl",
        "right_shift",
        "left_shift",
        "right_win",
        "left_win",
        "fn",
        "caps_lock",
        "insert",
        "escape",
        "space",
        "enter",
        "tab",
        "backspace",
        "delete",
        "home",
        "end",
        "page_up",
        "page_down",
        "left",
        "right",
        "down",
        "up",
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
        "f6",
        "f7",
        "f8",
        "f9",
        "f10",
        "f11",
        "f12",
        "f13",
        "f14",
        "f15",
        "f16",
        "f17",
        "f18",
        "f19",
        "f20",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ];
    ORDER
        .iter()
        .copied()
        .find(|n| mac_keycode(n) == Some(code))
        .unwrap_or(OTHER)
}

/// Whether a key code is the configured key (handles the shared F14/Scroll Lock codes).
pub fn mac_matches(code: u16, name: &str) -> bool {
    mac_keycode(name) == Some(code)
}

// ------------------------------------------------------------------------------------------
// Linux evdev codes (linux/input-event-codes.h). X11 keycodes are these plus 8.

pub fn evdev_code(name: &str) -> Option<u16> {
    Some(match name {
        "escape" => 1,
        "1" => 2,
        "2" => 3,
        "3" => 4,
        "4" => 5,
        "5" => 6,
        "6" => 7,
        "7" => 8,
        "8" => 9,
        "9" => 10,
        "0" => 11,
        "backspace" => 14,
        "tab" => 15,
        "q" => 16,
        "w" => 17,
        "e" => 18,
        "r" => 19,
        "t" => 20,
        "y" => 21,
        "u" => 22,
        "i" => 23,
        "o" => 24,
        "p" => 25,
        "enter" => 28,
        "left_ctrl" => 29,
        "a" => 30,
        "s" => 31,
        "d" => 32,
        "f" => 33,
        "g" => 34,
        "h" => 35,
        "j" => 36,
        "k" => 37,
        "l" => 38,
        "left_shift" => 42,
        "z" => 44,
        "x" => 45,
        "c" => 46,
        "v" => 47,
        "b" => 48,
        "n" => 49,
        "m" => 50,
        "right_shift" => 54,
        "left_alt" => 56,
        "space" => 57,
        "caps_lock" => 58,
        "f1" => 59,
        "f2" => 60,
        "f3" => 61,
        "f4" => 62,
        "f5" => 63,
        "f6" => 64,
        "f7" => 65,
        "f8" => 66,
        "f9" => 67,
        "f10" => 68,
        "scroll_lock" => 70,
        "f11" => 87,
        "f12" => 88,
        "right_ctrl" => 97,
        "right_alt" => 100,
        "home" => 102,
        "up" => 103,
        "page_up" => 104,
        "left" => 105,
        "right" => 106,
        "end" => 107,
        "down" => 108,
        "page_down" => 109,
        "insert" => 110,
        "delete" => 111,
        "pause" => 119,
        "left_win" => 125,
        "right_win" => 126,
        "menu" => 127, // KEY_COMPOSE, what PC keyboards send for the Menu key
        "fn" => 0x1d0,
        "f13" => 183,
        "f14" => 184,
        "f15" => 185,
        "f16" => 186,
        "f17" => 187,
        "f18" => 188,
        "f19" => 189,
        "f20" => 190,
        "f21" => 191,
        "f22" => 192,
        "f23" => 193,
        "f24" => 194,
        _ => return None,
    })
}

pub fn evdev_name(code: u16) -> &'static str {
    if code == 139 {
        return "menu"; // KEY_MENU, which some keyboards send instead of KEY_COMPOSE
    }
    NAMES
        .iter()
        .copied()
        .find(|n| evdev_code(n) == Some(code))
        .unwrap_or(OTHER)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_single_keys_and_aliases() {
        assert_eq!(
            parse_key("right_alt").unwrap(),
            KeySpec {
                key: "right_alt",
                mods: 0
            }
        );
        assert_eq!(parse_key("Right Option").unwrap().key, "right_alt");
        assert_eq!(parse_key("AltGr").unwrap().key, "right_alt");
        assert_eq!(parse_key("RCtrl").unwrap().key, "right_ctrl");
        assert_eq!(parse_key("right-control").unwrap().key, "right_ctrl");
        assert_eq!(parse_key("CapsLock").unwrap().key, "caps_lock");
        assert_eq!(parse_key("F13").unwrap().key, "f13");
        assert_eq!(parse_key("f24").unwrap().key, "f24");
        assert_eq!(parse_key("globe").unwrap().key, "fn");
        assert_eq!(parse_key("apps").unwrap().key, "menu");
    }

    #[test]
    fn parses_chords() {
        let k = parse_key("ctrl+shift+space").unwrap();
        assert_eq!(k.key, "space");
        assert_eq!(k.mods, mods::CTRL | mods::SHIFT);
        assert!(k.is_chord());
        assert_eq!(k.to_string(), "ctrl+shift+space");
        let k = parse_key("cmd + option + d").unwrap();
        assert_eq!(k.mods, mods::WIN | mods::ALT);
        assert_eq!(parse_key("right_ctrl+f").unwrap().mods, mods::CTRL);
    }

    #[test]
    fn rejects_unsafe_and_malformed() {
        assert!(parse_key("a").is_err());
        assert!(parse_key("space").is_err());
        assert!(parse_key("escape").is_err());
        assert!(parse_key("").is_err());
        assert!(parse_key("ctrl+").is_err());
        assert!(parse_key("ctrl+shift").is_err());
        assert!(parse_key("ctrl+left_shift").is_err());
        assert!(parse_key("q+w").is_err());
        assert!(parse_key("ctrl+nosuchkey").is_err());
        assert!(parse_key("ctrl").is_err());
    }

    #[test]
    fn raw_modifier() {
        assert_eq!(parse_modifier("shift").unwrap(), mods::SHIFT);
        assert_eq!(parse_modifier("Left Shift").unwrap(), mods::SHIFT);
        assert_eq!(parse_modifier("cmd").unwrap(), mods::WIN);
        assert_eq!(parse_modifier("").unwrap(), 0);
        assert_eq!(parse_modifier("none").unwrap(), 0);
        assert!(parse_modifier("space").is_err());
    }

    #[test]
    fn paste_key() {
        assert_eq!(
            parse_paste_key("down").unwrap(),
            PasteKey::WithVoice("down")
        );
        assert_eq!(parse_paste_key("V").unwrap(), PasteKey::WithVoice("v"));
        assert_eq!(parse_paste_key("").unwrap(), PasteKey::Off);
        assert_eq!(parse_paste_key("off").unwrap(), PasteKey::Off);
        let PasteKey::Combo(k) = parse_paste_key("ctrl+alt+v").unwrap() else {
            panic!("combo")
        };
        assert_eq!((k.key, k.mods), ("v", mods::CTRL | mods::ALT));
        assert!(parse_paste_key("left_shift").is_err());
        assert!(parse_paste_key("escape").is_err());
        assert!(parse_paste_key("nosuchkey").is_err());
        assert!(parse_paste_key("ctrl+").is_err());
    }

    #[test]
    fn windows_codes_round_trip() {
        for name in NAMES {
            if *name == "fn" {
                assert_eq!(win_vk(name), None);
                continue;
            }
            let vk = win_vk(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(win_name(u32::from(vk)), *name, "vk {vk:#x}");
        }
        assert_eq!(win_name(0x10), "left_shift");
        assert_eq!(win_name(0xFF), OTHER);
        assert!(win_extended(0xA5));
        assert!(!win_extended(0xA4));
    }

    #[test]
    fn mac_and_evdev_codes_round_trip() {
        for name in NAMES {
            if let Some(code) = evdev_code(name) {
                assert_eq!(evdev_name(code), *name);
            }
            if let Some(code) = mac_keycode(name)
                && !matches!(*name, "scroll_lock" | "pause")
            {
                assert_eq!(mac_name(code), *name);
            }
        }
        assert!(mac_matches(107, "scroll_lock"));
        assert!(mac_matches(107, "f14"));
        assert_eq!(evdev_name(139), "menu");
        assert_eq!(evdev_name(9999), OTHER);
        assert!(mac_unsupported("caps_lock").is_some());
        assert!(mac_unsupported("right_alt").is_none());
    }
}
