//! Which key types "v" with Command held in the current keyboard layout, for the paste fallback's
//! Cmd+V. Key code 9 is V on QWERTY and AZERTY, but K on Dvorak; "Dvorak - QWERTY ⌘" switches to
//! QWERTY while Command is held, hence the translation with Command down.
//!
//! Text Input Sources must be read on the main thread (macOS 14+ asserts otherwise), so
//! [`track`] runs there at startup: it caches the key code and refreshes it whenever the input
//! source changes (a distributed notification delivered on the main run loop). The paste path only
//! reads the cache. Layouts without a "v" (Cyrillic, Greek…) keep 9: macOS matches shortcuts
//! through their Latin fallback layout.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU16, Ordering};

pub const ANSI_V: u16 = 9;
static V_KEY: AtomicU16 = AtomicU16::new(ANSI_V);

type CFTypeRef = *const c_void;

#[link(name = "Carbon", kind = "framework")]
unsafe extern "C" {
    static kTISPropertyUnicodeKeyLayoutData: CFTypeRef;
    static kTISPropertyInputSourceID: CFTypeRef;
    fn TISCopyCurrentKeyboardLayoutInputSource() -> CFTypeRef;
    fn TISCreateInputSourceList(properties: CFTypeRef, include_all: u8) -> CFTypeRef;
    fn TISGetInputSourceProperty(source: CFTypeRef, key: CFTypeRef) -> CFTypeRef;
    fn LMGetKbdType() -> u8;
    #[allow(clippy::too_many_arguments)]
    fn UCKeyTranslate(
        layout: *const c_void,
        key_code: u16,
        action: u16,
        modifiers: u32,
        kbd_type: u32,
        options: u32,
        dead_key_state: *mut u32,
        max_len: usize,
        actual_len: *mut usize,
        chars: *mut u16,
    ) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
    fn CFRelease(cf: CFTypeRef);
    fn CFDataGetBytePtr(data: CFTypeRef) -> *const u8;
    fn CFArrayGetCount(array: CFTypeRef) -> isize;
    fn CFArrayGetValueAtIndex(array: CFTypeRef, index: isize) -> CFTypeRef;
    fn CFDictionaryCreate(
        allocator: CFTypeRef,
        keys: *const CFTypeRef,
        values: *const CFTypeRef,
        count: isize,
        key_callbacks: *const c_void,
        value_callbacks: *const c_void,
    ) -> CFTypeRef;
    fn CFStringCreateWithBytes(
        allocator: CFTypeRef,
        bytes: *const u8,
        len: isize,
        encoding: u32,
        external: u8,
    ) -> CFTypeRef;
    fn CFNotificationCenterGetDistributedCenter() -> CFTypeRef;
    fn CFNotificationCenterAddObserver(
        center: CFTypeRef,
        observer: *const c_void,
        callback: extern "C" fn(CFTypeRef, *mut c_void, CFTypeRef, *const c_void, CFTypeRef),
        name: CFTypeRef,
        object: *const c_void,
        behavior: isize,
    );
}

const UTF8: u32 = 0x0800_0100;
const KEY_ACTION_DOWN: u16 = 0;
const NO_DEAD_KEYS: u32 = 1;
const CMD_KEY_STATE: u32 = (0x0100 >> 8) & 0xFF; // (cmdKey >> 8) & 0xFF

fn cfstr(s: &str) -> CFTypeRef {
    // SAFETY: valid UTF-8 bytes for the given length.
    unsafe { CFStringCreateWithBytes(std::ptr::null(), s.as_ptr(), s.len() as isize, UTF8, 0) }
}

/// The key code that types `ch` with Command held in `source` (a TISInputSourceRef).
fn key_for(source: CFTypeRef, ch: char) -> Option<u16> {
    // SAFETY: `source` is a live input source; the layout data is owned by it (Get rule).
    let data = unsafe { TISGetInputSourceProperty(source, kTISPropertyUnicodeKeyLayoutData) };
    if data.is_null() {
        return None;
    }
    // SAFETY: CFData from the property above.
    let layout = unsafe { CFDataGetBytePtr(data) } as *const c_void;
    // SAFETY: no arguments.
    let kbd = unsafe { LMGetKbdType() } as u32;
    let (lower, upper) = (
        ch.to_ascii_lowercase() as u16,
        ch.to_ascii_uppercase() as u16,
    );
    (0u16..128).find(|&code| {
        let (mut dead, mut len, mut out) = (0u32, 0usize, [0u16; 4]);
        // SAFETY: `layout` is a 'uchr' blob; the out buffers are as large as we say.
        let st = unsafe {
            UCKeyTranslate(
                layout,
                code,
                KEY_ACTION_DOWN,
                CMD_KEY_STATE,
                kbd,
                NO_DEAD_KEYS,
                &mut dead,
                out.len(),
                &mut len,
                out.as_mut_ptr(),
            )
        };
        st == 0 && len == 1 && (out[0] == lower || out[0] == upper)
    })
}

/// Cmd+V's key code in the installed keyboard layout with this input source id
/// (e.g. `com.apple.keylayout.Dvorak`). None if it isn't installed or has no "v".
pub fn v_key_for_layout(id: &str) -> Option<u16> {
    // SAFETY: CF objects are created and released here; the list owns its sources.
    unsafe {
        let key = kTISPropertyInputSourceID;
        let value = cfstr(id);
        let filter = CFDictionaryCreate(
            std::ptr::null(),
            &key,
            &value,
            1,
            &kCFTypeDictionaryKeyCallBacks as *const c_void,
            &kCFTypeDictionaryValueCallBacks as *const c_void,
        );
        let list = TISCreateInputSourceList(filter, 1);
        let found = if list.is_null() || CFArrayGetCount(list) == 0 {
            None
        } else {
            key_for(CFArrayGetValueAtIndex(list, 0), 'v')
        };
        if !list.is_null() {
            CFRelease(list);
        }
        CFRelease(filter);
        CFRelease(value);
        found
    }
}

fn refresh() {
    // SAFETY: Copy rule: released below.
    let source = unsafe { TISCopyCurrentKeyboardLayoutInputSource() };
    if source.is_null() {
        return;
    }
    let code = key_for(source, 'v').unwrap_or(ANSI_V);
    // SAFETY: we own `source`.
    unsafe { CFRelease(source) };
    if V_KEY.swap(code, Ordering::Relaxed) != code {
        tracing::info!("keyboard layout changed: Cmd+V is key code {code}");
    }
}

extern "C" fn changed(_: CFTypeRef, _: *mut c_void, _: CFTypeRef, _: *const c_void, _: CFTypeRef) {
    refresh();
}

/// Call once on the main thread: cache Cmd+V's key code and keep it current.
pub fn track() {
    refresh();
    // SAFETY: the callback is a plain function; the name string lives for the process.
    unsafe {
        CFNotificationCenterAddObserver(
            CFNotificationCenterGetDistributedCenter(),
            std::ptr::null(),
            changed,
            cfstr("com.apple.Carbon.TISNotifySelectedKeyboardInputSourceChanged"),
            std::ptr::null(),
            4, // CFNotificationSuspensionBehaviorDeliverImmediately
        );
    }
}

/// Cmd+V's key code for the current layout (9 until [`track`] has run).
pub fn v_key() -> u16 {
    V_KEY.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v_moves_on_dvorak_only() {
        assert_eq!(v_key_for_layout("com.apple.keylayout.US"), Some(ANSI_V));
        assert_eq!(v_key_for_layout("com.apple.keylayout.French"), Some(ANSI_V));
        assert_eq!(v_key_for_layout("com.apple.keylayout.German"), Some(ANSI_V));
        // Dvorak's V sits on QWERTY's period key
        assert_eq!(v_key_for_layout("com.apple.keylayout.Dvorak"), Some(47));
        // "Dvorak - QWERTY ⌘" is QWERTY while Command is held
        assert_eq!(
            v_key_for_layout("com.apple.keylayout.DVORAK-QWERTYCMD"),
            Some(ANSI_V)
        );
        assert_eq!(v_key_for_layout("com.apple.keylayout.NoSuchLayout"), None);
    }
}
