//! macOS text injection: Unicode key events, with an NSPasteboard + Cmd+V fallback (SPEC §6.2).
//!
//! * `CGEventKeyboardSetUnicodeString` carries at most 20 UTF-16 units per event (longer
//!   strings are silently truncated by many apps), so text goes out as back-to-back events of
//!   <= 20 units that never split a surrogate pair, with no sleeps.
//!   Newline and tab are real Return / Tab key presses. Every event has its flags cleared so a
//!   held modifier (the raw-modifier Shift, a held Fn) cannot turn text into shortcuts
//!   (Handy #2051). Posting requires Accessibility.
//! * **Secure input** (a password field, or a terminal with "Secure Keyboard Entry") makes
//!   macOS drop synthetic keystrokes: `IsSecureEventInputEnabled` is checked first and we return
//!   `Error::Inject` instead of typing into nothing.
//! * The paste fallback saves every item and type on the general pasteboard, pastes with Cmd+V
//!   (the key that types "v" with Command in the current layout, see `layout.rs`: 9 on QWERTY /
//!   AZERTY, 47 on Dvorak), and restores the pasteboard on a background thread unless something
//!   else wrote to it meanwhile.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_app_kit::{NSPasteboard, NSPasteboardItem, NSPasteboardTypeString, NSPasteboardWriting};
use objc2_core_graphics::{
    CGEvent, CGEventField, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
};
use objc2_foundation::{NSArray, NSData, NSString};
use ochre_core::platform::{FocusInfo, Injector};
use ochre_core::{Error, Result};

use super::{focus, secure_input_enabled};
use crate::text::{Segment, chunks, plan};

/// `eventSourceUserData` on our typed text ("OWFT").
pub const OWN_TEXT_TAG: i64 = 0x4F57_4654;
pub const MAX_UNITS: usize = 20;
const KC_RETURN: u16 = 36;
const KC_TAB: u16 = 48;

pub const SECURE: &str = "macOS secure input is on (a password field, or an app with Secure Keyboard Entry), so \
typing is blocked. The text is in History.";
const NO_EVENT: &str =
    "macOS refused to create a key event (is Accessibility granted?). The text is in History.";

#[derive(Debug, Clone)]
pub struct MacInjector {
    /// How long the target gets to read the pasteboard before it is restored.
    pub restore_delay: Duration,
}

impl Default for MacInjector {
    fn default() -> Self {
        Self {
            restore_delay: Duration::from_millis(750),
        }
    }
}

/// Saved pasteboard: per item, (type, bytes) pairs.
type Saved = Vec<Vec<(String, Vec<u8>)>>;

struct Pending {
    generation: u64,
    change_count: isize,
    saved: Saved,
}

static PENDING: Mutex<Option<Pending>> = Mutex::new(None);
static GENERATION: AtomicU64 = AtomicU64::new(0);

fn source() -> Option<objc2_core_foundation::CFRetained<CGEventSource>> {
    CGEventSource::new(CGEventSourceStateID::HIDSystemState)
}

fn post(event: &CGEvent) {
    CGEvent::set_integer_value_field(Some(event), CGEventField::EventSourceUserData, OWN_TEXT_TAG);
    CGEvent::post(CGEventTapLocation::HIDEventTap, Some(event));
}

fn key_tap(src: Option<&CGEventSource>, code: u16, flags: CGEventFlags) -> Result<()> {
    for down in [true, false] {
        let event = CGEvent::new_keyboard_event(src, code, down)
            .ok_or_else(|| Error::Inject(NO_EVENT.into()))?;
        CGEvent::set_flags(Some(&event), flags);
        post(&event);
    }
    Ok(())
}

fn type_units(src: Option<&CGEventSource>, units: &[u16]) -> Result<()> {
    for chunk in chunks(units, MAX_UNITS) {
        for down in [true, false] {
            let event = CGEvent::new_keyboard_event(src, 0, down)
                .ok_or_else(|| Error::Inject(NO_EVENT.into()))?;
            CGEvent::set_flags(Some(&event), CGEventFlags(0));
            // SAFETY: `chunk` is valid for `chunk.len()` units during the call.
            unsafe {
                CGEvent::keyboard_set_unicode_string(Some(&event), chunk.len() as _, chunk.as_ptr())
            };
            post(&event);
        }
    }
    Ok(())
}

fn save(board: &NSPasteboard) -> Saved {
    let mut out = Vec::new();
    let Some(items) = board.pasteboardItems() else {
        return out;
    };
    for item in items.iter() {
        let mut entry = Vec::new();
        for t in item.types().iter() {
            if let Some(data) = item.dataForType(&t) {
                entry.push((t.to_string(), data.to_vec()));
            }
        }
        out.push(entry);
    }
    out
}

fn restore(board: &NSPasteboard, saved: &Saved) {
    board.clearContents();
    if saved.is_empty() {
        return;
    }
    let mut items: Vec<Retained<ProtocolObject<dyn NSPasteboardWriting>>> = Vec::new();
    for entry in saved {
        let item = NSPasteboardItem::new();
        for (t, bytes) in entry {
            item.setData_forType(&NSData::with_bytes(bytes), &NSString::from_str(t));
        }
        items.push(ProtocolObject::from_retained(item));
    }
    board.writeObjects(&NSArray::from_retained_slice(&items));
}

fn restore_if_current(generation: u64) {
    let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
    if pending.as_ref().is_some_and(|p| p.generation == generation)
        && let Some(p) = pending.take()
    {
        let board = NSPasteboard::generalPasteboard();
        if board.changeCount() == p.change_count {
            restore(&board, &p.saved);
        }
    }
}

impl MacInjector {
    pub fn new() -> Self {
        Self::default()
    }

    fn check(&self) -> Result<()> {
        if secure_input_enabled() {
            return Err(Error::Inject(SECURE.into()));
        }
        if !super::accessibility_trusted() {
            return Err(Error::Permission(
                crate::permissions::mac_accessibility_fix(),
            ));
        }
        Ok(())
    }
}

impl Injector for MacInjector {
    fn focus(&self) -> FocusInfo {
        focus::get_focus()
    }

    fn type_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        self.check()?;
        let src = source();
        let src = src.as_deref();
        for seg in plan(text) {
            match seg {
                Segment::Text(run) => type_units(src, &run.encode_utf16().collect::<Vec<u16>>())?,
                Segment::Enter => key_tap(src, KC_RETURN, CGEventFlags(0))?,
                Segment::Tab => key_tap(src, KC_TAB, CGEventFlags(0))?,
            }
        }
        Ok(())
    }

    fn paste_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        self.check()?;
        let board = NSPasteboard::generalPasteboard();
        let generation = GENERATION.fetch_add(1, Ordering::SeqCst) + 1;
        let original = {
            let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
            match pending.take() {
                Some(p) if p.change_count == board.changeCount() => Some(p.saved),
                _ => None,
            }
        };
        let saved = original.unwrap_or_else(|| save(&board));
        board.clearContents();
        // SAFETY: reading an immutable AppKit constant.
        let string_type = unsafe { NSPasteboardTypeString };
        if !board.setString_forType(&NSString::from_str(text), string_type) {
            return Err(Error::Inject(
                "could not write to the pasteboard. The text is in History.".into(),
            ));
        }
        let change_count = board.changeCount();
        let src = source();
        key_tap(
            src.as_deref(),
            super::layout::v_key(),
            CGEventFlags::MaskCommand,
        )?;
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some(Pending {
            generation,
            change_count,
            saved,
        });
        let delay = self.restore_delay;
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            restore_if_current(generation);
        });
        Ok(())
    }

    fn press_enter(&self) -> Result<()> {
        self.check()?;
        key_tap(source().as_deref(), KC_RETURN, CGEventFlags(0))
    }
}
