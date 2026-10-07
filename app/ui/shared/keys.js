// Voice-key names: the config spelling (`hotkey.key`), human labels per OS, and the "press a key"
// capture used by Settings and onboarding. Pure (tested in app/test/keys.test.mjs).
//
// Config names: single keys are snake_case ("right_alt", "caps_lock", "f13"); chords join generic
// modifiers and one key with "+", modifiers first in a fixed order ("ctrl+meta", "ctrl+shift+space").
// "right_alt" means Right Option on macOS. `platform` is "windows" | "macos" | "linux".

export const MOD_ORDER = ['ctrl', 'alt', 'shift', 'meta'];

// KeyboardEvent.code of modifiers -> [generic name, side-specific name]
const MODIFIERS = {
  ControlLeft: ['ctrl', 'left_ctrl'], ControlRight: ['ctrl', 'right_ctrl'],
  AltLeft: ['alt', 'left_alt'], AltRight: ['alt', 'right_alt'],
  ShiftLeft: ['shift', 'left_shift'], ShiftRight: ['shift', 'right_shift'],
  MetaLeft: ['meta', 'left_meta'], MetaRight: ['meta', 'right_meta'],
  OSLeft: ['meta', 'left_meta'], OSRight: ['meta', 'right_meta'],
};

// Keys nobody types with: fine as a Voice key on their own.
export const STANDALONE = new Set(['caps_lock', 'menu', 'insert', 'scroll_lock', 'pause',
  ...Array.from({ length: 12 }, (_, i) => `f${13 + i}`)]);

const SPECIAL = {
  CapsLock: 'caps_lock', ContextMenu: 'menu', Insert: 'insert', ScrollLock: 'scroll_lock', Pause: 'pause',
  Space: 'space', Enter: 'enter', Tab: 'tab', Backspace: 'backspace', Escape: 'escape', Delete: 'delete',
  Home: 'home', End: 'end', PageUp: 'page_up', PageDown: 'page_down',
  ArrowUp: 'up', ArrowDown: 'down', ArrowLeft: 'left', ArrowRight: 'right',
  Backquote: '`', Minus: '-', Equal: '=', BracketLeft: '[', BracketRight: ']', Backslash: '\\',
  Semicolon: ';', Quote: "'", Comma: ',', Period: '.', Slash: '/',
};

export function codeToName(code) {
  if (!code) return null;
  if (MODIFIERS[code]) return MODIFIERS[code][1];
  if (SPECIAL[code]) return SPECIAL[code];
  let m = /^Key([A-Z])$/.exec(code);
  if (m) return m[1].toLowerCase();
  m = /^Digit(\d)$/.exec(code);
  if (m) return m[1];
  m = /^F(\d{1,2})$/.exec(code);
  if (m) return `f${m[1]}`;
  return null;
}

const isMac = (platform) => platform === 'macos' || platform === 'darwin';

function partLabel(part, platform) {
  const mac = isMac(platform);
  const win = platform === 'windows' || platform === 'win32';
  const metaWord = mac ? 'Command' : win ? 'Win' : 'Super';
  const map = {
    ctrl: mac ? 'Control' : 'Ctrl', alt: mac ? 'Option' : 'Alt', shift: 'Shift', meta: metaWord,
    right_alt: mac ? 'Right Option' : 'Right Alt', left_alt: mac ? 'Left Option' : 'Left Alt',
    right_ctrl: mac ? 'Right Control' : 'Right Ctrl', left_ctrl: mac ? 'Left Control' : 'Left Ctrl',
    right_shift: 'Right Shift', left_shift: 'Left Shift',
    right_meta: `Right ${metaWord}`, left_meta: `Left ${metaWord}`,
    fn: mac ? 'fn (Globe)' : 'Fn',
    caps_lock: 'Caps Lock', menu: 'Menu', insert: 'Insert', scroll_lock: 'Scroll Lock', pause: 'Pause',
    space: 'Space', enter: 'Enter', tab: 'Tab', backspace: 'Backspace', escape: 'Esc', delete: 'Delete',
    home: 'Home', end: 'End', page_up: 'Page Up', page_down: 'Page Down',
    up: '↑', down: '↓', left: '←', right: '→',
  };
  if (map[part]) return map[part];
  if (/^f\d{1,2}$/.test(part)) return part.toUpperCase();
  return part.length === 1 ? part.toUpperCase() : part;
}

// "ctrl+meta" -> "Ctrl + Win" (Windows), "right_alt" -> "Right Option" (macOS)
export function label(name, platform) {
  if (!name) return '';
  return String(name).split('+').map((p) => partLabel(p.trim(), platform)).join(' + ');
}

// `hotkey.paste_last` as keycap text: a plain key pairs with the Voice key ("Right Ctrl + ↓"),
// a chord stands alone ("Ctrl + Alt + V"), "" (off) gives ''.
export function pasteLabel(pasteLast, voiceKey, platform) {
  const p = String(pasteLast ?? '').trim();
  if (['', 'off', 'none', 'disabled'].includes(p)) return '';
  return p.includes('+') ? label(p, platform) : `${label(voiceKey, platform)} + ${label(p, platform)}`;
}

// Keys the OS adapter can't hold as a Voice key (mirrors ochre-platform keys::mac_unsupported):
// macOS reports Caps Lock only as a toggle, never its release.
const MAC_UNSUPPORTED = new Set(['caps_lock', 'menu', 'insert', 'scroll_lock', 'pause', 'f21', 'f22', 'f23', 'f24']);

export function supported(name, platform) {
  return !(isMac(platform) && MAC_UNSUPPORTED.has(name));
}

// Ready-made choices for the picker, per OS, most useful first (the picker shows the first six).
//  * macOS: Right Option / Right Command / fn (Globe, the usual Mac dictation key); Mac keyboards
//    have no Menu / Insert / Scroll Lock / Pause, Caps Lock can't be held, F-keys stop at F19.
//  * Windows / Linux: many laptops have no Right Win key, so it comes last.
//  * No modifier-only chords (Ctrl + Win): the hotkey engines need a non-modifier last key.
export function presets(platform) {
  if (isMac(platform)) {
    const out = ['right_alt', 'right_meta', 'fn', 'right_ctrl'];
    for (let i = 13; i <= 19; i++) out.push(`f${i}`);
    return out;
  }
  const out = ['right_alt', 'right_ctrl', 'caps_lock', 'insert', 'right_meta', 'menu', 'scroll_lock', 'pause'];
  for (let i = 13; i <= 20; i++) out.push(`f${i}`);
  return out;
}

// One line of advice for the Voice key setting.
export function hint(platform) {
  return isMac(platform)
    ? 'Pick a key you never type with. Right Option is the default; Right Command or fn (Globe) work well too.'
    : 'Pick a key you never type with. Right Alt is the default; Right Ctrl, Caps Lock or F13–F20 work well too.';
}

// "Press a key" capture. Feed it keydown / keyup codes; when every key is up again it returns
// {name} or {error}. Rules:
//  * a right-hand modifier alone is the classic Voice key (right_alt, right_ctrl, ...);
//  * Windows' AltGr arrives as ControlLeft + AltRight: that is Right Alt, not a chord;
//  * a key nobody types with (Caps Lock, F13-F24, Menu, ...) works alone;
//  * a typing key (letters, Space, ...) needs a modifier, otherwise every "a" would start dictation;
//  * several modifiers without a key are refused: the hotkey engines can't hold a modifier-only
//    chord ("ctrl+meta").
export function createCapture(platform) {
  const example = label('right_alt', platform);
  const down = new Set();
  let seen = [];

  function finish() {
    let codes = seen.slice();
    seen = [];
    if (codes.includes('AltRight') && codes.includes('ControlLeft')) codes = codes.filter((c) => c !== 'ControlLeft');
    const mods = codes.filter((c) => MODIFIERS[c]);
    const keys = codes.filter((c) => !MODIFIERS[c]);
    if (keys.length) {
      const key = codeToName(keys[keys.length - 1]);
      if (!key) return { error: "That key can't be used. Try another one." };
      if (key === 'escape') return { error: 'Escape cancels dictation, so it can’t be the Voice key.' };
      const generic = MOD_ORDER.filter((m) => mods.some((c) => MODIFIERS[c][0] === m));
      if (!generic.length) {
        if (STANDALONE.has(key)) {
          if (!supported(key, platform)) return { error: `macOS can't use ${label(key, platform)} as the Voice key. Try ${example}.` };
          return { name: key };
        }
        return { error: `You type with that key. Pick one like ${example}, or hold a modifier with it.` };
      }
      return { name: [...generic, key].join('+') };
    }
    if (mods.length === 1) {
      const side = MODIFIERS[mods[0]][1];
      if (side.startsWith('left_')) {
        return { error: 'Left-hand modifiers are busy with shortcuts. Try the right-hand one.' };
      }
      return { name: side };
    }
    return { error: `Use one modifier, like ${example}, or hold modifiers with another key.` };
  }

  return {
    keydown(code) {
      if (!code) return null;
      down.add(code);
      if (!seen.includes(code)) seen.push(code);
      return null;
    },
    keyup(code) {
      down.delete(code);
      if (down.size === 0 && seen.length) return finish();
      return null;
    },
    reset() { down.clear(); seen = []; },
    get pressing() { return [...seen]; },
  };
}
