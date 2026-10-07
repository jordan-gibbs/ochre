//! Windows text injection: one `SendInput` with `KEYEVENTF_UNICODE`, no clipboard (SPEC §6.2).
//!
//! Lessons carried over from earlier native injection code and the Python port:
//!
//! * The whole result goes out in **one** `SendInput` call (§3.1 rule 5): UTF-16 units as
//!   Unicode key events (surrogate pairs as two units, so emoji arrive whole), newline as Enter,
//!   tab as Tab. `SendInput` is atomic with respect to other input, so the user's own typing
//!   can never interleave with the dictation.
//! * **Held modifiers** (the raw-modifier Shift, the Ctrl+Shift of a chord hotkey) would turn
//!   text into shortcuts. Inside that same call they are released first (with a "menu mask" key
//!   before a lone Alt/Win release, so no menu opens) and given back afterwards **only** if our
//!   hook saw the user is still physically holding them: re-pressing a key the user has let go
//!   would strand it down across the desktop, which is far worse than having to press it again.
//!   Without a hook we wait briefly for the user to let go, then release
//!   without restoring. Ctrl+V borrows a Ctrl the user is holding instead of releasing it.
//! * **Elevated targets** (UIPI): a normal process cannot type into an admin window, and
//!   `SendInput` does not report that. We refuse up front with a clear message.
//! * **XAML apps** (Win11 Notepad, Windows Terminal) translate injected characters lazily and
//!   corrupt a burst into repeats of the last character (verified on Notepad 11.2607:
//!   5 ms per character fails, 8 ms works). Short text is paced for them; longer text is pasted.
//! * The paste fallback snapshots every memory-backed clipboard format, pastes, and restores
//!   the snapshot on a background thread after the target has had time to read it, unless
//!   something else wrote to the clipboard meanwhile.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ochre_core::platform::{FocusInfo, Injector};
use ochre_core::{Error, Result};
use windows::Win32::UI::Input::KeyboardAndMouse::INPUT;

use super::{PHYSICAL, TAG_TEXT, clipboard, focus, key_down, key_input, send, unicode_input};
use crate::text::{Segment, plan};

const VK_LSHIFT: u16 = 0xA0;
const VK_RSHIFT: u16 = 0xA1;
const VK_LCONTROL: u16 = 0xA2;
const VK_RCONTROL: u16 = 0xA3;
const VK_LMENU: u16 = 0xA4;
const VK_RMENU: u16 = 0xA5;
const VK_LWIN: u16 = 0x5B;
const VK_RWIN: u16 = 0x5C;
const VK_RETURN: u16 = 0x0D;
const VK_TAB: u16 = 0x09;
const VK_V: u16 = 0x56;
/// Unassigned VK used as a "menu mask": a key event between Alt/Win down and up stops the
/// menu bar / Start menu from opening on the release.
const VK_MASK: u16 = 0xE8;
const MODIFIERS: [u16; 8] = [
    VK_LSHIFT,
    VK_RSHIFT,
    VK_LCONTROL,
    VK_RCONTROL,
    VK_LMENU,
    VK_RMENU,
    VK_LWIN,
    VK_RWIN,
];
const MENU_KEYS: [u16; 4] = [VK_LMENU, VK_RMENU, VK_LWIN, VK_RWIN];
/// Upper bound per `SendInput` (UTF-16 units); results longer than this go out back to back.
const MAX_UNITS_PER_CALL: usize = 4096;

pub const ELEVATED: &str = "is running as administrator, and Windows does not let a normal app type into it. \
The text is in History. To dictate into admin windows, run Ochre as administrator too.";
pub const BLOCKED: &str = "Windows blocked the keystrokes (an admin window, the lock screen or a UAC prompt has \
focus). The text is in History.";

#[derive(Debug, Clone)]
pub struct WindowsInjector {
    /// Apps that need paced typing (lowercase exe names).
    pub paced_apps: Vec<String>,
    pub paced_gap: Duration,
    /// Paced apps get longer text pasted instead of typed.
    pub paced_max_units: usize,
    /// Without a hook, how long to wait for the user to let go of modifiers.
    pub modifier_wait: Duration,
    /// How long the target gets to read the clipboard before it is restored.
    pub restore_delay: Duration,
}

impl Default for WindowsInjector {
    fn default() -> Self {
        Self {
            paced_apps: vec!["notepad".into(), "windowsterminal".into()],
            paced_gap: Duration::from_millis(10),
            // Paced typing pays ~15.6 ms per key (Windows timer tick for windowless processes,
            // docs/latency.md rule 7): keep it to a few characters, paste the rest.
            paced_max_units: 8,
            modifier_wait: Duration::from_millis(400),
            restore_delay: Duration::from_millis(750),
        }
    }
}

/// Modifier release before, and give-back after, our keystrokes.
struct ModifierWrap {
    prefix: Vec<INPUT>,
    suffix: Vec<INPUT>,
}

fn held_modifiers(keep: &[u16]) -> Vec<u16> {
    MODIFIERS
        .iter()
        .copied()
        .filter(|vk| !keep.contains(vk) && key_down(*vk))
        .collect()
}

fn wrap_modifiers(keep: &[u16], wait: Duration) -> ModifierWrap {
    let hook = PHYSICAL.active.load(Ordering::Acquire);
    let mut held = held_modifiers(keep);
    if !held.is_empty() && !hook {
        // We cannot tell a key the user holds from one stuck down: wait for them to let go.
        let deadline = Instant::now() + wait;
        while !held.is_empty() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
            held = held_modifiers(keep);
        }
    }
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    if held.is_empty() {
        return ModifierWrap { prefix, suffix };
    }
    if held.iter().any(|vk| MENU_KEYS.contains(vk)) {
        prefix.extend([
            key_input(VK_MASK, false, TAG_TEXT),
            key_input(VK_MASK, true, TAG_TEXT),
        ]);
    }
    prefix.extend(held.iter().map(|vk| key_input(*vk, true, TAG_TEXT)));
    let back: Vec<u16> = held
        .into_iter()
        .filter(|vk| hook && PHYSICAL.is_down(*vk))
        .collect();
    suffix.extend(back.iter().map(|vk| key_input(*vk, false, TAG_TEXT)));
    if back.iter().any(|vk| MENU_KEYS.contains(vk)) {
        // So the user's own release of the given-back Alt/Win opens no menu.
        suffix.extend([
            key_input(VK_MASK, false, TAG_TEXT),
            key_input(VK_MASK, true, TAG_TEXT),
        ]);
    }
    ModifierWrap { prefix, suffix }
}

fn tap(vk: u16) -> [INPUT; 2] {
    [
        key_input(vk, false, TAG_TEXT),
        key_input(vk, true, TAG_TEXT),
    ]
}

/// The keystrokes for `text` (without modifier handling), plus its UTF-16 length.
pub(crate) fn text_inputs(text: &str) -> Vec<INPUT> {
    let mut out = Vec::with_capacity(text.len() * 2 + 4);
    for seg in plan(text) {
        match seg {
            Segment::Text(run) => {
                for unit in run.encode_utf16() {
                    out.push(unicode_input(unit, false));
                    out.push(unicode_input(unit, true));
                }
            }
            Segment::Enter => out.extend(tap(VK_RETURN)),
            Segment::Tab => out.extend(tap(VK_TAB)),
        }
    }
    out
}

fn send_all(inputs: &[INPUT]) -> Result<()> {
    // Never split a down from its up: chunk at even indices.
    for chunk in inputs.chunks(MAX_UNITS_PER_CALL * 2) {
        if send(chunk) != chunk.len() {
            return Err(Error::Inject(BLOCKED.into()));
        }
    }
    Ok(())
}

struct PendingRestore {
    generation: u64,
    seq: u32,
    snapshot: clipboard::Snapshot,
}

static PENDING: Mutex<Option<PendingRestore>> = Mutex::new(None);
static GENERATION: AtomicU64 = AtomicU64::new(0);

impl WindowsInjector {
    pub fn new() -> Self {
        Self::default()
    }

    fn check_target(&self) -> Result<FocusInfo> {
        let target = focus::get_focus();
        if target.elevated {
            let app = if target.app_name.is_empty() {
                "The focused app"
            } else {
                target.app_name.as_str()
            };
            return Err(Error::Inject(format!("{app} {ELEVATED}")));
        }
        Ok(target)
    }

    fn paced(&self, text: &str) -> Result<()> {
        let wrap = wrap_modifiers(&[], self.modifier_wait);
        send_all(&wrap.prefix)?;
        let inputs = text_inputs(text);
        let presses = PHYSICAL.presses.load(Ordering::Relaxed);
        let watched = PHYSICAL.active.load(Ordering::Acquire);
        let mut result = Ok(());
        for pair in inputs.chunks(2) {
            // Paced keystrokes can interleave with the user's own typing: stop if they type.
            if watched && PHYSICAL.presses.load(Ordering::Relaxed) != presses {
                result = Err(Error::Inject(
                    "You started typing while dictated text was going out, so the rest was not typed. \
The text is in History."
                        .into(),
                ));
                break;
            }
            if let Err(e) = send_all(pair) {
                result = Err(e);
                break;
            }
            std::thread::sleep(self.paced_gap);
        }
        send_all(&wrap.suffix)?;
        result
    }
}

impl Injector for WindowsInjector {
    fn focus(&self) -> FocusInfo {
        focus::get_focus()
    }

    fn type_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let target = self.check_target()?;
        if self.paced_apps.contains(&target.app_name) {
            if text.encode_utf16().count() > self.paced_max_units {
                return self.paste_text(text);
            }
            return self.paced(text);
        }
        let wrap = wrap_modifiers(&[], self.modifier_wait);
        let mut inputs = wrap.prefix;
        inputs.extend(text_inputs(text));
        inputs.extend(wrap.suffix);
        send_all(&inputs)
    }

    fn paste_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        self.check_target()?;
        let generation = GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
        // A paste whose restore is still pending put *our* text on the clipboard: the user's
        // original is the snapshot it is holding, not what is there now.
        let original = {
            let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
            match pending.take() {
                Some(p) if p.seq == clipboard::sequence() => Some(p.snapshot),
                _ => None,
            }
        };
        let snapshot = match original {
            Some(s) => Some(s),
            None => match clipboard::save() {
                Ok(s) => Some(s),
                Err(e) => {
                    tracing::warn!("could not save the clipboard ({e}); it will not be restored");
                    None
                }
            },
        };
        let seq = clipboard::set_text(&crlf(text)).map_err(|e| {
            Error::Inject(format!(
                "could not use the clipboard ({e}). The text is in History."
            ))
        })?;
        let wrap = wrap_modifiers(&[VK_LCONTROL, VK_RCONTROL], self.modifier_wait);
        // Sampled after the wait, right before injecting: a Ctrl the user still holds is
        // borrowed (neither pressed nor released), so their key state is handed back intact.
        let user_ctrl = key_down(VK_LCONTROL) || key_down(VK_RCONTROL);
        let mut inputs = wrap.prefix;
        if !user_ctrl {
            inputs.push(key_input(VK_LCONTROL, false, TAG_TEXT));
        }
        inputs.extend(tap(VK_V));
        if !user_ctrl {
            inputs.push(key_input(VK_LCONTROL, true, TAG_TEXT));
        }
        inputs.extend(wrap.suffix);
        let sent = send_all(&inputs);
        if let Some(snapshot) = snapshot {
            *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some(PendingRestore {
                generation,
                seq,
                snapshot,
            });
            let delay = self.restore_delay;
            std::thread::spawn(move || {
                std::thread::sleep(delay); // apps read the clipboard asynchronously after Ctrl+V
                restore_if_current(generation);
            });
        }
        sent
    }

    fn press_enter(&self) -> Result<()> {
        self.check_target()?;
        let wrap = wrap_modifiers(&[], self.modifier_wait);
        let mut inputs = wrap.prefix;
        inputs.extend(tap(VK_RETURN));
        inputs.extend(wrap.suffix);
        send_all(&inputs)
    }
}

/// Windows clipboard text uses CRLF line endings.
fn crlf(text: &str) -> String {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\n', "\r\n")
}

/// Restore the snapshot of paste `generation` unless a newer paste took over or something
/// else wrote to the clipboard since.
fn restore_if_current(generation: u64) {
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    if pending.as_ref().is_some_and(|p| p.generation == generation)
        && let Some(p) = pending.take()
        && clipboard::sequence() == p.seq
        && let Err(e) = clipboard::restore(&p.snapshot)
    {
        tracing::warn!("could not restore the clipboard: {e}");
    }
}

/// Wait for a pending clipboard restore (tests, shutdown).
pub fn flush_pending_restore() {
    let generation = PENDING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(|p| p.generation);
    if let Some(g) = generation {
        restore_if_current(g);
    }
}
