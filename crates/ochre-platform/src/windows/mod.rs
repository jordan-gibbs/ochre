//! Windows: `WH_KEYBOARD_LL` hotkeys, `SendInput` Unicode typing, clipboard paste, focus.

pub mod clipboard;
pub mod focus;
pub mod hook;
pub mod inject;

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, SendInput, VIRTUAL_KEY,
};

pub use hook::WindowsHotkeys;
pub use inject::WindowsInjector;

/// `dwExtraInfo` on our replayed Voice key / chord events ("OWFR").
pub const TAG_REPLAY: usize = 0x4F57_4652;
/// `dwExtraInfo` on dictated text and the keys we press while injecting ("OWFT").
pub const TAG_TEXT: usize = 0x4F57_4654;
/// `dwExtraInfo` the hook treats as physical input while [`hook::accept_test_input`] is on.
pub const TAG_TEST: usize = 0x4F57_4658;

/// Which keys the user is *physically* holding, as seen by our hook (non-injected events).
/// The injector uses it to give back only modifiers the user still holds.
pub(crate) struct Physical {
    down: [AtomicBool; 256],
    /// Physical key presses seen (any key).
    pub presses: AtomicU32,
    /// A hook is running and feeding this.
    pub active: AtomicBool,
}

impl Physical {
    const fn new() -> Self {
        Self {
            down: [const { AtomicBool::new(false) }; 256],
            presses: AtomicU32::new(0),
            active: AtomicBool::new(false),
        }
    }

    pub fn update(&self, vk: u32, down: bool) {
        if let Some(slot) = self.down.get(vk as usize) {
            let was = slot.swap(down, Ordering::Relaxed);
            if down && !was {
                self.presses.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    pub fn is_down(&self, vk: u16) -> bool {
        self.down
            .get(usize::from(vk))
            .is_some_and(|s| s.load(Ordering::Relaxed))
    }

    pub fn clear(&self) {
        for s in &self.down {
            s.store(false, Ordering::Relaxed);
        }
    }
}

pub(crate) static PHYSICAL: Physical = Physical::new();

/// The OS's (logical) view of a key: down right now.
pub(crate) fn key_down(vk: u16) -> bool {
    // SAFETY: plain Win32 call without pointers.
    unsafe { GetAsyncKeyState(i32::from(vk)) as u16 & 0x8000 != 0 }
}

pub(crate) fn key_input(vk: u16, up: bool, tag: usize) -> INPUT {
    let mut flags = if up {
        KEYEVENTF_KEYUP
    } else {
        KEYBD_EVENT_FLAGS(0)
    };
    if crate::keys::win_extended(vk) {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: tag,
            },
        },
    }
}

/// A key event exactly as the hook saw it (virtual key, scan code, extended flag).
pub(crate) fn raw_key_input(vk: u16, scan: u16, extended: bool, up: bool, tag: usize) -> INPUT {
    let mut flags = if up {
        KEYEVENTF_KEYUP
    } else {
        KEYBD_EVENT_FLAGS(0)
    };
    if extended {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: tag,
            },
        },
    }
}

pub(crate) fn unicode_input(unit: u16, up: bool) -> INPUT {
    let flags = if up {
        KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
    } else {
        KEYEVENTF_UNICODE
    };
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(0),
                wScan: unit,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: TAG_TEXT,
            },
        },
    }
}

/// One `SendInput` call; returns how many events Windows accepted.
pub(crate) fn send(inputs: &[INPUT]) -> usize {
    if inputs.is_empty() {
        return 0;
    }
    // SAFETY: the slice is valid for the call; cbSize is the struct size.
    unsafe { SendInput(inputs, std::mem::size_of::<INPUT>() as i32) as usize }
}

pub(crate) fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(test)]
mod live_tests;
