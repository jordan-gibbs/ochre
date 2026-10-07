//! Live Windows checks against real hooks and a window we create ourselves (`#[ignore]`d:
//! they need an interactive desktop). Run with
//! `cargo test -p ochre-platform -- --ignored --test-threads=1 --nocapture live_`.
//!
//! Safety for a machine someone is using: every keystroke goes to our own window, which must
//! hold the foreground or the test skips; the hook is installed for about two seconds; the
//! previous foreground window gets focus back; and no modifier may be left down.

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ochre_core::config::HotkeyConfig;
use ochre_core::platform::{Gesture, HotkeyListener, Injector};
use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
use windows::Win32::UI::Input::KeyboardAndMouse::SetFocus;
use windows::Win32::UI::WindowsAndMessaging::{
    BringWindowToTop, CreateWindowExW, DefWindowProcW, DispatchMessageW, GetForegroundWindow,
    GetMessageW, GetWindowThreadProcessId, MSG, PostMessageW, PostQuitMessage, RegisterClassW,
    SW_SHOW, SendMessageW, SetForegroundWindow, ShowWindow, TranslateMessage, WINDOW_EX_STYLE,
    WM_CLOSE, WM_DESTROY, WM_GETTEXT, WM_GETTEXTLENGTH, WM_SETTEXT, WNDCLASSW, WS_CHILD,
    WS_OVERLAPPEDWINDOW, WS_VISIBLE, WS_VSCROLL,
};
use windows::core::{PCWSTR, w};

use super::{TAG_TEST, hook, key_down, raw_key_input, send};

const ES_MULTILINE: u32 = 0x4;
const ES_AUTOVSCROLL: u32 = 0x40;
const ES_WANTRETURN: u32 = 0x1000;

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if msg == WM_DESTROY {
        // SAFETY: plain Win32 call.
        unsafe { PostQuitMessage(0) };
        return LRESULT(0);
    }
    // SAFETY: forwarding our own arguments.
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

/// A top-level window with a multiline EDIT control, owned by its own UI thread.
struct TestWindow {
    hwnd: HWND,
    edit: HWND,
    previous: HWND,
    thread: Option<JoinHandle<()>>,
}

#[derive(Clone, Copy)]
struct Handles(isize, isize, isize);
unsafe impl Send for Handles {}

impl TestWindow {
    /// Opens the window and makes it the foreground window, or returns None (and closes it).
    fn open() -> Option<Self> {
        let (tx, rx) = mpsc::channel::<Option<Handles>>();
        let thread = std::thread::spawn(move || {
            // SAFETY: Win32 window creation and a message loop, all on this thread.
            unsafe {
                let instance = GetModuleHandleW(None).ok();
                let class = WNDCLASSW {
                    lpfnWndProc: Some(wndproc),
                    hInstance: instance.map(|m| m.into()).unwrap_or_default(),
                    lpszClassName: w!("OchrePlatformLiveTest"),
                    ..Default::default()
                };
                RegisterClassW(&class);
                let previous = GetForegroundWindow();
                let Ok(hwnd) = CreateWindowExW(
                    WINDOW_EX_STYLE(0),
                    w!("OchrePlatformLiveTest"),
                    w!("ochre-platform live test (closes itself)"),
                    WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                    120,
                    120,
                    700,
                    420,
                    None,
                    None,
                    class.hInstance.into(),
                    None,
                ) else {
                    let _ = tx.send(None);
                    return;
                };
                let edit = CreateWindowExW(
                    WINDOW_EX_STYLE(0),
                    w!("EDIT"),
                    PCWSTR::null(),
                    WS_CHILD
                        | WS_VISIBLE
                        | WS_VSCROLL
                        | windows::Win32::UI::WindowsAndMessaging::WINDOW_STYLE(
                            ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN,
                        ),
                    0,
                    0,
                    680,
                    380,
                    Some(hwnd),
                    None,
                    class.hInstance.into(),
                    None,
                )
                .unwrap_or_default();
                // Raise the 30,000-character default limit (EM_SETLIMITTEXT).
                SendMessageW(edit, 0x00C5, Some(WPARAM(0)), Some(LPARAM(0)));
                let _ = ShowWindow(hwnd, SW_SHOW);
                if !SetForegroundWindow(hwnd).as_bool() || GetForegroundWindow() != hwnd {
                    // Borrow the foreground thread's input state long enough to take focus.
                    let fg_thread = GetWindowThreadProcessId(GetForegroundWindow(), None);
                    let me = GetCurrentThreadId();
                    let _ = AttachThreadInput(me, fg_thread, true);
                    let _ = BringWindowToTop(hwnd);
                    let _ = SetForegroundWindow(hwnd);
                    let _ = AttachThreadInput(me, fg_thread, false);
                }
                let _ = SetFocus(Some(edit));
                let _ = tx.send(Some(Handles(
                    hwnd.0 as isize,
                    edit.0 as isize,
                    previous.0 as isize,
                )));
                let mut msg = MSG::default();
                while GetMessageW(&mut msg, None, 0, 0).0 > 0 {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
        });
        let handles = rx.recv_timeout(Duration::from_secs(5)).ok().flatten()?;
        let win = TestWindow {
            hwnd: HWND(handles.0 as *mut _),
            edit: HWND(handles.1 as *mut _),
            previous: HWND(handles.2 as *mut _),
            thread: Some(thread),
        };
        std::thread::sleep(Duration::from_millis(150));
        // SAFETY: plain Win32 query.
        if unsafe { GetForegroundWindow() } != win.hwnd {
            eprintln!("could not take the foreground; skipping (nothing was injected)");
            win.close();
            return None;
        }
        Some(win)
    }

    fn is_foreground(&self) -> bool {
        // SAFETY: plain Win32 query.
        unsafe { GetForegroundWindow() == self.hwnd }
    }

    fn text_len(&self) -> usize {
        // SAFETY: cross-thread SendMessage to our own control (its thread pumps messages).
        unsafe { SendMessageW(self.edit, WM_GETTEXTLENGTH, None, None).0 as usize }
    }

    fn text(&self) -> String {
        let n = self.text_len();
        let mut buf = vec![0u16; n + 1];
        // SAFETY: the buffer holds n + 1 units.
        let got = unsafe {
            SendMessageW(
                self.edit,
                WM_GETTEXT,
                Some(WPARAM(buf.len())),
                Some(LPARAM(buf.as_mut_ptr() as isize)),
            )
            .0
        };
        String::from_utf16_lossy(&buf[..got as usize])
    }

    fn clear(&self) {
        let empty = [0u16];
        // SAFETY: NUL-terminated buffer, alive for the call.
        unsafe {
            SendMessageW(
                self.edit,
                WM_SETTEXT,
                None,
                Some(LPARAM(empty.as_ptr() as isize)),
            );
        }
    }

    fn close(mut self) {
        // SAFETY: our own window; then hand the foreground back to whoever had it.
        unsafe {
            let _ = PostMessageW(Some(self.hwnd), WM_CLOSE, WPARAM(0), LPARAM(0));
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        // SAFETY: plain Win32 call on a window that may no longer exist (then it just fails).
        unsafe {
            if !self.previous.is_invalid() {
                let _ = SetForegroundWindow(self.previous);
            }
        }
    }

    /// Wait until the control holds `units` UTF-16 units (or time out).
    fn wait_len(&self, units: usize, timeout: Duration) -> Duration {
        let start = Instant::now();
        while self.text_len() < units && start.elapsed() < timeout {
            std::thread::sleep(Duration::from_micros(250));
        }
        start.elapsed()
    }
}

fn assert_no_modifier_stuck() {
    for vk in [0xA0u16, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0x5B, 0x5C, 0x87] {
        assert!(!key_down(vk), "virtual key {vk:#x} left down");
    }
}

/// A key event the hook treats as physical (tagged), aimed at our foreground window.
fn key(vk: u16, down: bool) {
    let extended = crate::keys::win_extended(vk);
    assert_eq!(send(&[raw_key_input(vk, 0, extended, !down, TAG_TEST)]), 1);
}

fn tap(vk: u16, hold_ms: u64) {
    key(vk, true);
    std::thread::sleep(Duration::from_millis(hold_ms));
    key(vk, false);
}

fn drain(rx: &mpsc::Receiver<Gesture>, wait: Duration) -> Vec<Gesture> {
    let mut out = Vec::new();
    let deadline = Instant::now() + wait;
    while let Ok(g) = rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        out.push(g);
    }
    out
}

const VK_RMENU: u16 = 0xA5;
const VK_ESCAPE: u16 = 0x1B;
const VK_F23: u16 = 0x86;

#[test]
#[ignore = "installs a real keyboard hook and focuses a test window"]
fn live_hook_gestures() {
    let Some(win) = TestWindow::open() else {
        return;
    };
    hook::accept_test_input(true);
    let mut listener = super::WindowsHotkeys::new(&HotkeyConfig::default()).unwrap();
    let (tx, rx) = mpsc::channel();
    let tx = Arc::new(Mutex::new(tx));
    let sink = Arc::clone(&tx);
    listener
        .start(Box::new(move |g| {
            let _ = sink.lock().unwrap().send(g);
        }))
        .unwrap();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let short = Duration::from_millis(120);
        // Hold: press now, release after the hold threshold.
        key(VK_RMENU, true);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            Gesture::Press
        );
        std::thread::sleep(Duration::from_millis(320));
        key(VK_RMENU, false);
        assert_eq!(drain(&rx, short), vec![Gesture::Release]);
        assert!(
            !key_down(VK_RMENU),
            "Right Alt must be swallowed, never seen down"
        );

        // Double tap locks; a single tap finishes.
        tap(VK_RMENU, 40);
        std::thread::sleep(Duration::from_millis(70));
        tap(VK_RMENU, 40);
        assert_eq!(drain(&rx, short), vec![Gesture::Press, Gesture::Lock]);
        listener.set_recording(true);
        std::thread::sleep(Duration::from_millis(500)); // locked: no abort
        assert!(drain(&rx, short).is_empty());
        tap(VK_RMENU, 40);
        assert_eq!(drain(&rx, short), vec![Gesture::Finish]);
        listener.set_recording(false);

        // A lone tap aborts once the double-tap window closes.
        tap(VK_RMENU, 40);
        assert_eq!(
            drain(&rx, Duration::from_millis(600)),
            vec![Gesture::Press, Gesture::Abort]
        );

        // Escape while recording cancels (and is swallowed); Escape when idle is not ours.
        key(VK_RMENU, true);
        listener.set_recording(true);
        tap(VK_ESCAPE, 20);
        std::thread::sleep(Duration::from_millis(50));
        key(VK_RMENU, false);
        assert_eq!(drain(&rx, short), vec![Gesture::Press, Gesture::Cancel]);
        listener.set_recording(false);
        tap(VK_ESCAPE, 20);
        assert!(drain(&rx, short).is_empty());

        // Right Alt as a modifier (chord): abort, and the chord reaches the window intact.
        key(VK_RMENU, true);
        std::thread::sleep(Duration::from_millis(30));
        key(VK_F23, true);
        key(VK_F23, false);
        key(VK_RMENU, false);
        assert_eq!(drain(&rx, short), vec![Gesture::Press, Gesture::Abort]);
        assert!(
            win.is_foreground(),
            "focus stayed on the test window throughout"
        );
    }));
    listener.stop();
    hook::accept_test_input(false);
    win.close();
    assert_no_modifier_stuck();
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

fn paragraph() -> String {
    let base = "The quick brown fox jumps over the lazy dog while naïve café patrons sip crème brûlée; \
                Zoë's façade looks jaunty. ";
    let mut s = String::new();
    while s.chars().count() < 600 {
        s.push_str(base);
    }
    s
}

fn sample() -> String {
    format!(
        "Héllo wörld 😀👍🏽 — “quotes”, ñandú, 中文, Ελληνικά.\nSecond line\twith a tab.\n\n{}",
        paragraph()
    )
}

#[test]
#[ignore = "types into a test window it creates and focuses"]
fn live_type_text_round_trip_and_timing() {
    let Some(win) = TestWindow::open() else {
        return;
    };
    let injector = super::WindowsInjector::new();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let f = injector.focus();
        assert!(!f.elevated);
        assert!(f.window_title.contains("ochre-platform live test"), "{f:?}");
        assert!(!f.window_id.is_empty());
        for (label, text) in [
            ("short", "Hello from ochre-platform! ".to_string()),
            ("sample", sample()),
        ] {
            win.clear();
            let expected = text.replace('\n', "\r\n");
            let units = expected.encode_utf16().count();
            let t0 = Instant::now();
            injector.type_text(&text).unwrap();
            let call = t0.elapsed();
            let visible = call + win.wait_len(units, Duration::from_secs(10));
            let got = win.text();
            eprintln!(
                "type_text[{label}]: {} chars / {units} UTF-16 units: SendInput returned in {:.2} ms, \
                 all text in the control after {:.2} ms",
                text.chars().count(),
                call.as_secs_f64() * 1000.0,
                visible.as_secs_f64() * 1000.0
            );
            assert_eq!(got, expected);
        }
        // Again with our own hook installed, as in the running app (it adds one more hop per event).
        let mut listener = super::WindowsHotkeys::new(&HotkeyConfig::default()).unwrap();
        listener.start(Box::new(|_| {})).unwrap();
        let text = sample();
        win.clear();
        let expected = text.replace('\n', "\r\n");
        let t0 = Instant::now();
        injector.type_text(&text).unwrap();
        let call = t0.elapsed();
        let visible = call + win.wait_len(expected.encode_utf16().count(), Duration::from_secs(10));
        listener.stop();
        eprintln!(
            "type_text[sample, our hook installed]: SendInput returned in {:.2} ms, all text in the control after {:.2} ms",
            call.as_secs_f64() * 1000.0,
            visible.as_secs_f64() * 1000.0
        );
        assert_eq!(win.text(), expected);

        win.clear();
        injector.type_text("enter test").unwrap();
        injector.press_enter().unwrap();
        win.wait_len("enter test\r\n".len(), Duration::from_secs(2));
        assert_eq!(win.text(), "enter test\r\n");
        assert!(win.is_foreground());
    }));
    win.close();
    assert_no_modifier_stuck();
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
}

#[test]
#[ignore = "pastes into a test window it creates; restores the clipboard afterwards"]
fn live_paste_round_trip_restores_clipboard() {
    use super::clipboard;
    let before_seq = clipboard::sequence();
    let before = clipboard::save().expect("save clipboard");
    let Some(win) = TestWindow::open() else {
        return;
    };
    let injector = super::WindowsInjector::new();
    let text = sample();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let expected = text.replace('\n', "\r\n");
        let t0 = Instant::now();
        injector.paste_text(&text).unwrap();
        let call = t0.elapsed();
        let visible = call + win.wait_len(expected.encode_utf16().count(), Duration::from_secs(5));
        eprintln!(
            "paste_text: {} chars: call {:.2} ms, text in the control after {:.2} ms",
            text.chars().count(),
            call.as_secs_f64() * 1000.0,
            visible.as_secs_f64() * 1000.0
        );
        assert_eq!(win.text(), expected);
    }));
    super::inject::flush_pending_restore();
    win.close();
    let after = clipboard::save().expect("save clipboard");
    assert_no_modifier_stuck();
    if let Err(e) = result {
        std::panic::resume_unwind(e);
    }
    assert_ne!(clipboard::sequence(), before_seq);
    for (format, data) in &before {
        assert!(
            after.iter().any(|(f, d)| f == format && d == data),
            "clipboard format {format} not restored"
        );
    }
}
