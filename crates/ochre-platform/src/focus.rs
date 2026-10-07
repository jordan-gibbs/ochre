//! Giving the keyboard back to the user's window before typing into it.
//!
//! Choosing "Paste last transcript" in the tray menu moves the foreground away from the field
//! the user was typing in: on Windows to the taskbar / notification area (or to Ochre's own
//! hidden menu window), so text typed right away would go nowhere. [`track_foreground`] keeps
//! the last foreground window that is neither Ochre's nor the shell's, and [`restore_previous`]
//! brings it back (and waits until Windows agrees) before the text is typed.
//!
//! * **Windows:** an `EVENT_SYSTEM_FOREGROUND` WinEvent hook on its own thread records every
//!   foreground change (no polling). Restoring calls `SetForegroundWindow`, falling back to the
//!   `AttachThreadInput` + masked Alt tap workaround when the foreground lock refuses, and waits
//!   until `GetForegroundWindow` matches (about 150 ms at most).
//! * **macOS:** a status-item menu does not activate Ochre, so the frontmost app keeps its
//!   key window; restoring only waits for the menu to close. If an Ochre window (Settings) was
//!   frontmost, the text goes there: there is no previous-app tracking yet.
//! * **Linux:** best effort, as on macOS: wait for the menu to close and type into whatever
//!   has focus (tray menus on X11 / StatusNotifier rarely take the focus).
//!
//! The decision logic ([`restore`]) is OS-free and unit-tested with a fake desktop.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// What [`restore`] needs from the OS. Windows are opaque non-zero ids; 0 means none.
pub trait Desktop {
    fn foreground(&self) -> u64;
    /// The window still exists.
    fn alive(&self, w: u64) -> bool;
    /// Ask for `w` to become the foreground window. `forceful`: the first polite request did
    /// not take; use the OS's workaround for the foreground lock.
    fn activate(&self, w: u64, forceful: bool);
    fn now_ms(&self) -> u64;
    fn sleep(&self, d: Duration);
}

/// The last foreground window worth going back to (0 = none seen yet).
static LAST: AtomicU64 = AtomicU64::new(0);

/// Record a foreground change. `ours_or_shell`: Ochre's own window, the taskbar, the tray
/// overflow, a menu: never a place to type into, so the previous window is kept.
pub fn note_foreground(w: u64, ours_or_shell: bool) {
    if w != 0 && !ours_or_shell {
        LAST.store(w, Ordering::Relaxed);
    }
}

/// The window [`restore_previous`] would bring back.
pub fn previous() -> u64 {
    LAST.load(Ordering::Relaxed)
}

/// Bring `target` to the foreground and wait until it is there, for at most `timeout`.
/// Returns whether the target is in the foreground. A missing or closed target returns false
/// without touching anything (the text then goes wherever the focus is).
pub fn restore(d: &dyn Desktop, target: u64, timeout: Duration) -> bool {
    if target == 0 || !d.alive(target) {
        return false;
    }
    if d.foreground() == target {
        return true;
    }
    let timeout = timeout.as_millis() as u64;
    let start = d.now_ms();
    d.activate(target, false);
    let mut forced = false;
    loop {
        if d.foreground() == target {
            return true;
        }
        let elapsed = d.now_ms().saturating_sub(start);
        if elapsed >= timeout {
            return false;
        }
        if !forced && elapsed >= timeout / 3 {
            forced = true;
            d.activate(target, true);
        }
        d.sleep(Duration::from_millis(10));
    }
}

/// Start following the foreground window (idempotent). Windows only; a no-op elsewhere.
pub fn track_foreground() {
    #[cfg(windows)]
    win::start();
}

/// Give the keyboard back to the window the user was in before Ochre's tray menu, and wait
/// until it has it (at most `timeout`). Returns whether a window was restored.
pub fn restore_previous(timeout: Duration) -> bool {
    #[cfg(windows)]
    {
        restore(&win::Win, previous(), timeout)
    }
    #[cfg(not(windows))]
    {
        // The menu has to be gone before keystrokes go out (macOS / Linux keep the user's app
        // frontmost under a status menu).
        std::thread::sleep(timeout.min(Duration::from_millis(150)));
        false
    }
}

#[cfg(windows)]
mod win {
    use std::sync::Once;
    use std::time::Duration;

    use windows::Win32::Foundation::HWND;
    use windows::Win32::System::Threading::{
        AttachThreadInput, GetCurrentProcessId, GetCurrentThreadId,
    };
    use windows::Win32::UI::Accessibility::{HWINEVENTHOOK, SetWinEventHook};
    use windows::Win32::UI::WindowsAndMessaging::{
        BringWindowToTop, EVENT_SYSTEM_FOREGROUND, GetClassNameW, GetForegroundWindow, GetMessageW,
        GetWindowThreadProcessId, IsIconic, IsWindow, MSG, SW_RESTORE, SetForegroundWindow,
        ShowWindow, WINEVENT_OUTOFCONTEXT,
    };

    use super::{Desktop, note_foreground};
    use crate::windows::{TAG_TEXT, key_input, send};

    const VK_MENU: u16 = 0x12;
    /// Unassigned VK between Alt down and up, so the Alt tap opens no menu.
    const VK_MASK: u16 = 0xE8;

    /// Shell surfaces that take the foreground when the tray is used, and menus.
    const SHELL_CLASSES: &[&str] = &[
        "Shell_TrayWnd",
        "Shell_SecondaryTrayWnd",
        "NotifyIconOverflowWindow",
        "TopLevelWindowForOverflowXamlIsland",
        "XamlExplorerHostIslandWindow",
        "Windows.UI.Core.CoreWindow",
        "ForegroundStaging",
        "MultitaskingViewFrame",
        "Progman",
        "WorkerW",
        "#32768",
    ];

    fn hwnd(w: u64) -> HWND {
        HWND(w as usize as *mut core::ffi::c_void)
    }

    fn class_of(h: HWND) -> String {
        let mut buf = [0u16; 128];
        // SAFETY: valid buffer.
        let n = unsafe { GetClassNameW(h, &mut buf) };
        String::from_utf16_lossy(&buf[..n.max(0) as usize])
    }

    fn ours_or_shell(h: HWND) -> bool {
        let mut pid = 0u32;
        // SAFETY: valid out-pointer.
        unsafe { GetWindowThreadProcessId(h, Some(&mut pid)) };
        // SAFETY: no arguments.
        if pid == unsafe { GetCurrentProcessId() } {
            return true;
        }
        let class = class_of(h);
        SHELL_CLASSES.contains(&class.as_str())
    }

    fn record(h: HWND) {
        if !h.is_invalid() {
            note_foreground(h.0 as usize as u64, ours_or_shell(h));
        }
    }

    unsafe extern "system" fn on_foreground(
        _hook: HWINEVENTHOOK,
        _event: u32,
        h: HWND,
        _object: i32,
        _child: i32,
        _thread: u32,
        _time: u32,
    ) {
        let _ = std::panic::catch_unwind(|| record(h));
    }

    pub fn start() {
        static STARTED: Once = Once::new();
        STARTED.call_once(|| {
            // SAFETY: no arguments.
            record(unsafe { GetForegroundWindow() });
            let spawned = std::thread::Builder::new()
                .name("ochre-foreground".into())
                .spawn(|| {
                    // SAFETY: an out-of-context hook delivered to this thread's message loop,
                    // which runs for the life of the process.
                    unsafe {
                        let hook = SetWinEventHook(
                            EVENT_SYSTEM_FOREGROUND,
                            EVENT_SYSTEM_FOREGROUND,
                            None,
                            Some(on_foreground),
                            0,
                            0,
                            WINEVENT_OUTOFCONTEXT,
                        );
                        if hook.is_invalid() {
                            tracing::warn!("foreground tracking unavailable");
                            return;
                        }
                        let mut msg = MSG::default();
                        while GetMessageW(&mut msg, None, 0, 0).0 > 0 {}
                    }
                });
            if let Err(e) = spawned {
                tracing::warn!("foreground tracking thread: {e}");
            }
        });
    }

    pub struct Win;

    impl Desktop for Win {
        fn foreground(&self) -> u64 {
            // SAFETY: no arguments.
            unsafe { GetForegroundWindow() }.0 as usize as u64
        }

        fn alive(&self, w: u64) -> bool {
            // SAFETY: IsWindow accepts any handle value.
            unsafe { IsWindow(Some(hwnd(w))) }.as_bool()
        }

        fn activate(&self, w: u64, forceful: bool) {
            let h = hwnd(w);
            // SAFETY: plain Win32 calls on a window handle checked with IsWindow.
            unsafe {
                if IsIconic(h).as_bool() {
                    let _ = ShowWindow(h, SW_RESTORE);
                }
                if !forceful {
                    let _ = SetForegroundWindow(h);
                    return;
                }
                // The foreground lock refused: join the current foreground thread's input
                // state, and tap a masked Alt (counts as our input) before asking again.
                let fg = GetForegroundWindow();
                let fg_thread = GetWindowThreadProcessId(fg, None);
                let me = GetCurrentThreadId();
                let attached = fg_thread != 0
                    && fg_thread != me
                    && AttachThreadInput(me, fg_thread, true).as_bool();
                send(&[
                    key_input(VK_MENU, false, TAG_TEXT),
                    key_input(VK_MASK, false, TAG_TEXT),
                    key_input(VK_MASK, true, TAG_TEXT),
                    key_input(VK_MENU, true, TAG_TEXT),
                ]);
                let _ = SetForegroundWindow(h);
                let _ = BringWindowToTop(h);
                if attached {
                    let _ = AttachThreadInput(me, fg_thread, false);
                }
            }
        }

        fn now_ms(&self) -> u64 {
            crate::driver::monotonic_ms()
        }

        fn sleep(&self, d: Duration) {
            std::thread::sleep(d);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};

    /// A desktop whose foreground follows an activation after `polite_ms` (polite request) or
    /// only after a forceful one (`polite_ms = None`).
    struct Fake {
        fg: Cell<u64>,
        windows: Vec<u64>,
        t: Cell<u64>,
        polite_ms: Option<u64>,
        forceful_works: bool,
        pending: Cell<Option<(u64, u64)>>,
        calls: RefCell<Vec<(u64, bool)>>,
    }

    impl Fake {
        fn new(fg: u64, polite_ms: Option<u64>, forceful_works: bool) -> Self {
            Self {
                fg: Cell::new(fg),
                windows: vec![1, 2, 3],
                t: Cell::new(1000),
                polite_ms,
                forceful_works,
                pending: Cell::new(None),
                calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl Desktop for Fake {
        fn foreground(&self) -> u64 {
            if let Some((w, at)) = self.pending.get()
                && self.t.get() >= at
            {
                self.fg.set(w);
            }
            self.fg.get()
        }
        fn alive(&self, w: u64) -> bool {
            self.windows.contains(&w)
        }
        fn activate(&self, w: u64, forceful: bool) {
            self.calls.borrow_mut().push((w, forceful));
            match (forceful, self.polite_ms) {
                (false, Some(ms)) => self.pending.set(Some((w, self.t.get() + ms))),
                (true, _) if self.forceful_works => self.pending.set(Some((w, self.t.get()))),
                _ => {}
            }
        }
        fn now_ms(&self) -> u64 {
            self.t.get()
        }
        fn sleep(&self, d: Duration) {
            self.t.set(self.t.get() + d.as_millis() as u64);
        }
    }

    const T: Duration = Duration::from_millis(150);

    #[test]
    fn already_in_front_does_nothing() {
        let d = Fake::new(2, Some(0), true);
        assert!(restore(&d, 2, T));
        assert!(d.calls.borrow().is_empty());
    }

    #[test]
    fn polite_activation_waits_until_the_os_agrees() {
        let d = Fake::new(1, Some(30), true);
        assert!(restore(&d, 2, T));
        assert_eq!(*d.calls.borrow(), vec![(2, false)]);
        assert!(d.t.get() - 1000 >= 30 && d.t.get() - 1000 < 150);
    }

    #[test]
    fn foreground_lock_falls_back_to_the_workaround() {
        let d = Fake::new(1, None, true);
        assert!(restore(&d, 2, T));
        assert_eq!(*d.calls.borrow(), vec![(2, false), (2, true)]);
    }

    #[test]
    fn gives_up_after_the_timeout() {
        let d = Fake::new(1, None, false);
        assert!(!restore(&d, 2, T));
        let waited = d.t.get() - 1000;
        assert!((150..=170).contains(&waited), "{waited}");
    }

    #[test]
    fn missing_or_closed_target_is_left_alone() {
        let d = Fake::new(1, Some(0), true);
        assert!(!restore(&d, 0, T));
        assert!(!restore(&d, 9, T));
        assert!(d.calls.borrow().is_empty());
    }

    #[test]
    fn tracker_skips_ochre_and_the_shell() {
        note_foreground(42, false);
        note_foreground(7, true); // the taskbar
        note_foreground(0, false);
        assert_eq!(previous(), 42);
    }
}
