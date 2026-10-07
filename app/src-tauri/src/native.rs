//! The few window behaviours Tauri doesn't cover, per OS.
//!
//! - **Windows:** show without activating (`SW_SHOWNOACTIVATE` + `HWND_TOPMOST`), read back the
//!   display affinity (Tauri's `content_protected` sets `WDA_EXCLUDEFROMCAPTURE`), the foreground
//!   window's center (to pick the monitor you're dictating on), and a cheap cursor / left-button poll
//!   for the HUD's own hit-testing. `WS_EX_NOACTIVATE` comes from `focusable(false)`.
//! - **macOS:** status-bar window level, visible on every Space and over full-screen apps, left out of
//!   the window cycle, `NSWindowSharingNone`, and `orderFrontRegardless` so showing it never
//!   activates the app.
//! - **Linux:** nothing native. Transparency needs a compositor. X11 honours click-through (input
//!   shape) and keep-above; on Wayland the compositor decides placement and stacking (a layer-shell
//!   surface would be needed), and there is no capture exclusion.

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;

    use windows_sys::Win32::Foundation::{HWND, POINT, RECT};
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_LBUTTON};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetCursorPos, GetForegroundWindow, GetWindowDisplayAffinity, GetWindowRect, HWND_TOPMOST,
        SW_SHOWNOACTIVATE, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE, SWP_SHOWWINDOW, SetWindowPos,
        ShowWindow,
    };

    pub type Handle = *mut c_void;

    pub fn show_no_activate(h: Handle) {
        unsafe {
            ShowWindow(h as HWND, SW_SHOWNOACTIVATE);
            SetWindowPos(
                h as HWND,
                HWND_TOPMOST,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW,
            );
        }
    }

    /// 0x11 = WDA_EXCLUDEFROMCAPTURE, 0 = visible to capture.
    pub fn display_affinity(h: Handle) -> Option<u32> {
        let mut v = 0u32;
        (unsafe { GetWindowDisplayAffinity(h as HWND, &mut v) } != 0).then_some(v)
    }

    /// Center of the foreground window in physical px, unless it's one of ours.
    pub fn foreground_center(own: &[isize]) -> Option<(f64, f64)> {
        let fg = unsafe { GetForegroundWindow() };
        if fg.is_null() || own.contains(&(fg as isize)) {
            return None;
        }
        let r = window_rect(fg as Handle)?;
        Some(((r.0 + r.2 / 2) as f64, (r.1 + r.3 / 2) as f64))
    }

    pub fn cursor() -> Option<(f64, f64)> {
        let mut p = POINT { x: 0, y: 0 };
        (unsafe { GetCursorPos(&mut p) } != 0).then_some((p.x as f64, p.y as f64))
    }

    /// (x, y, w, h) in physical px.
    pub fn window_rect(h: Handle) -> Option<(i32, i32, i32, i32)> {
        let mut r = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        (unsafe { GetWindowRect(h as HWND, &mut r) } != 0).then_some((
            r.left,
            r.top,
            r.right - r.left,
            r.bottom - r.top,
        ))
    }

    pub fn left_button_down() -> Option<bool> {
        Some(unsafe { GetAsyncKeyState(VK_LBUTTON as i32) } as u16 & 0x8000 != 0)
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use std::ffi::c_void;

    use objc2_app_kit::{
        NSStatusWindowLevel, NSWindow, NSWindowCollectionBehavior, NSWindowSharingType,
    };

    pub type Handle = *mut c_void;

    fn window<'a>(h: Handle) -> Option<&'a NSWindow> {
        // SAFETY: `h` is Tauri's `ns_window()` pointer for a live window, used on the main thread.
        unsafe { (h as *const NSWindow).as_ref() }
    }

    /// Must run on the main thread.
    pub fn harden(h: Handle, protect: bool) {
        let Some(w) = window(h) else { return };
        w.setLevel(NSStatusWindowLevel);
        w.setCollectionBehavior(
            NSWindowCollectionBehavior::CanJoinAllSpaces
                | NSWindowCollectionBehavior::FullScreenAuxiliary
                | NSWindowCollectionBehavior::Stationary
                | NSWindowCollectionBehavior::IgnoresCycle,
        );
        w.setHidesOnDeactivate(false);
        if protect {
            w.setSharingType(NSWindowSharingType::None);
        }
    }

    /// Must run on the main thread.
    pub fn show_no_activate(h: Handle) {
        if let Some(w) = window(h) {
            w.orderFrontRegardless();
        }
    }
}

#[cfg(windows)]
pub use imp::*;

#[cfg(target_os = "macos")]
pub use imp::{harden as mac_harden, show_no_activate as mac_show_no_activate};

#[cfg(not(windows))]
pub fn foreground_center(_own: &[isize]) -> Option<(f64, f64)> {
    None
}

#[cfg(not(windows))]
pub fn cursor() -> Option<(f64, f64)> {
    None
}

#[cfg(not(windows))]
pub fn left_button_down() -> Option<bool> {
    None
}
