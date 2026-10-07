//! Windows Voice key listener: a `WH_KEYBOARD_LL` hook on its own thread (SPEC §6.1).
//!
//! Hardening ("Voice never jams the keyboard"):
//!
//! * The hook lives on a dedicated, time-critical thread that does nothing but pump messages.
//!   Windows holds every keystroke on the machine until a low-level hook answers and silently
//!   lets the key through when it answers late, so the callback only translates the event and
//!   asks the [`Driver`] (a mutex held for microseconds); gestures reach the app through the
//!   driver's dispatcher thread over a channel.
//! * A release is never hidden while Windows thinks the key is down ([`safe_swallow`]): if a
//!   press leaked through a late hook, hiding its release would leave the key stuck everywhere.
//! * Gesture timing uses the event's own timestamp, not when the hook ran.
//! * A watchdog reinstalls the hook when Windows drops it (it does so silently after repeated
//!   timeouts): input the system saw that our hooks did not means they are gone. A tiny mouse
//!   hook only stamps the time, so mouse-only use is not mistaken for a dropped hook.
//! * Stopping releases a Voice key the gesture had a hand in, so nothing stays held.
//! * AltGr layouts send a synthesized Left Ctrl right before Right Alt (same timestamp, scan
//!   code 0x21D). It passes through untouched and never counts as a held modifier or as
//!   "another key".
//! * Chords: when the swallowed Voice key turns out to be a modifier (Right Ctrl+C, AltGr+e),
//!   the key press and then the current key are re-sent in one `SendInput`, in that order, and
//!   the original event is swallowed. (Sending only the Voice key from inside the hook would
//!   land *after* the current key.)
//! * Injected input (ours and everyone else's) always passes through untouched.

use std::panic::AssertUnwindSafe;
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use ochre_core::config::HotkeyConfig;
use ochre_core::platform::{GestureFn, HotkeyListener};
use ochre_core::{Error, Result};
use windows::Win32::Foundation::{HINSTANCE, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::SystemInformation::{GetTickCount, GetTickCount64};
use windows::Win32::System::Threading::{
    GetCurrentThread, GetCurrentThreadId, SetThreadPriority, THREAD_PRIORITY_TIME_CRITICAL,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{GetLastInputInfo, LASTINPUTINFO};
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, GetMessageW, HC_ACTION, HHOOK, KBDLLHOOKSTRUCT, KillTimer, LLKHF_EXTENDED,
    LLKHF_INJECTED, MSG, PM_NOREMOVE, PeekMessageW, PostThreadMessageW, SetTimer,
    SetWindowsHookExW, UnhookWindowsHookEx, WH_KEYBOARD_LL, WH_MOUSE_LL, WM_KEYDOWN, WM_QUIT,
    WM_SYSKEYDOWN, WM_TIMER, WM_USER,
};

use super::{PHYSICAL, TAG_REPLAY, TAG_TEST, key_down, key_input, raw_key_input, send};
use crate::driver::{Driver, parse_key};
use crate::gesture::{event_time, safe_swallow};
use crate::keys::{self, KeySpec};

const VK_LCONTROL: u32 = 0xA2;
const VK_RMENU: u32 = 0xA5;
const WATCHDOG_MS: u32 = 1000;
const SILENT_MS: i32 = 2000;
const REINSTALL_GAP_MS: u64 = 10_000;

/// State the hook procedure reads. Lives behind [`CURRENT`] while a hook thread runs.
struct Shared {
    driver: Arc<Driver>,
    last_hook_tick: AtomicU32,
    lctrl_time: AtomicU32,
    lctrl_fake: AtomicBool,
    reinstalls: AtomicU32,
}

/// The running listener's state (one per process: a hook procedure has no user data).
static CURRENT: AtomicPtr<Shared> = AtomicPtr::new(ptr::null_mut());
/// Tests drive the hook with `SendInput` events tagged [`TAG_TEST`]; they count as physical.
static ACCEPT_TEST_INPUT: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
pub(crate) fn accept_test_input(on: bool) {
    ACCEPT_TEST_INPUT.store(on, Ordering::SeqCst);
}

fn validate(spec: &KeySpec) -> std::result::Result<(), String> {
    match keys::win_vk(spec.key) {
        Some(_) => Ok(()),
        None => Err(format!(
            "{:?} cannot be used as the Voice key on Windows",
            spec.key
        )),
    }
}

fn tick64() -> u64 {
    // SAFETY: no arguments.
    unsafe { GetTickCount64() }
}

fn tick32() -> u32 {
    // SAFETY: no arguments.
    unsafe { GetTickCount() }
}

/// [`HotkeyListener`] on a low-level keyboard hook.
pub struct WindowsHotkeys {
    driver: Arc<Driver>,
    thread: Mutex<Option<(JoinHandle<()>, u32)>>,
}

impl WindowsHotkeys {
    pub fn new(cfg: &HotkeyConfig) -> Result<Self> {
        Ok(Self {
            driver: Driver::new(cfg, tick64, validate)?,
            thread: Mutex::new(None),
        })
    }

    /// Times the watchdog had to put the hook back (diagnostics).
    pub fn reinstalls() -> u32 {
        let p = CURRENT.load(Ordering::Acquire);
        // SAFETY: CURRENT is only freed after the hook thread exits, in `stop`.
        if p.is_null() {
            0
        } else {
            unsafe { (*p).reinstalls.load(Ordering::Relaxed) }
        }
    }

    fn release_stuck_key(&self) {
        // Never leave the Voice key held for other apps once the hook is gone. If a finger
        // really is on it, its own release passes straight through and does no harm.
        let spec = self.driver.key();
        if let Some(vk) = keys::win_vk(spec.key)
            && self.driver.involved()
            && key_down(vk)
        {
            send(&[key_input(vk, true, TAG_REPLAY)]);
        }
    }
}

impl HotkeyListener for WindowsHotkeys {
    fn start(&mut self, on_gesture: GestureFn) -> Result<()> {
        let mut slot = self.thread.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_some() {
            return Ok(());
        }
        let shared = Arc::new(Shared {
            driver: Arc::clone(&self.driver),
            last_hook_tick: AtomicU32::new(tick32()),
            lctrl_time: AtomicU32::new(u32::MAX),
            lctrl_fake: AtomicBool::new(false),
            reinstalls: AtomicU32::new(0),
        });
        let raw = Arc::into_raw(shared) as *mut Shared;
        if CURRENT
            .compare_exchange(ptr::null_mut(), raw, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // SAFETY: `raw` came from Arc::into_raw above and was never published.
            drop(unsafe { Arc::from_raw(raw) });
            return Err(Error::Other(
                "another Voice key listener is already running in this process".into(),
            ));
        }
        if let Err(e) = self.driver.start(on_gesture) {
            free_current();
            return Err(e);
        }
        let (ready_tx, ready_rx) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("ochre-keyhook".into())
            .spawn(move || hook_thread(ready_tx));
        let handle = match handle {
            Ok(h) => h,
            Err(e) => {
                self.driver.stop();
                free_current();
                return Err(e.into());
            }
        };
        match ready_rx.recv_timeout(Duration::from_secs(3)) {
            Ok(Ok(tid)) => {
                *slot = Some((handle, tid));
                Ok(())
            }
            outcome => {
                let _ = handle.join();
                self.driver.stop();
                free_current();
                Err(match outcome {
                    Ok(Err(e)) => e,
                    _ => Error::Other("the keyboard hook thread did not start".into()),
                })
            }
        }
    }

    fn set_key(&mut self, key: &str) -> Result<()> {
        self.driver.set_key(key)
    }

    fn set_recording(&self, recording: bool) {
        self.driver.set_recording(recording);
    }

    fn stop(&mut self) {
        let taken = self.thread.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some((handle, tid)) = taken {
            // SAFETY: posting to a thread id we own.
            unsafe {
                let _ = PostThreadMessageW(tid, WM_QUIT, WPARAM(0), LPARAM(0));
            }
            let _ = handle.join(); // no hook procedure runs past this line
            free_current();
            self.release_stuck_key();
            self.driver.stop();
            self.driver.reset();
        }
    }
}

impl Drop for WindowsHotkeys {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Validate a key name for Windows without starting anything.
pub fn check_key(key: &str) -> Result<KeySpec> {
    parse_key(key, validate)
}

fn free_current() {
    let p = CURRENT.swap(ptr::null_mut(), Ordering::AcqRel);
    if !p.is_null() {
        // SAFETY: `p` came from Arc::into_raw in `start`; the hook thread has exited.
        drop(unsafe { Arc::from_raw(p) });
    }
}

struct Hooks {
    keyboard: Option<HHOOK>,
    mouse: Option<HHOOK>,
}

impl Hooks {
    fn install(&mut self) -> std::result::Result<(), String> {
        self.uninstall();
        // SAFETY: the procedures are 'static functions; the module handle is our own image.
        unsafe {
            let module = GetModuleHandleW(None).map(|m| HINSTANCE(m.0)).ok();
            self.keyboard = Some(
                SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_proc), module, 0)
                    .map_err(|e| format!("SetWindowsHookExW(WH_KEYBOARD_LL) failed: {e}"))?,
            );
            self.mouse = SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_proc), module, 0).ok();
        }
        Ok(())
    }

    fn uninstall(&mut self) {
        // SAFETY: handles we installed on this thread.
        unsafe {
            if let Some(h) = self.keyboard.take() {
                let _ = UnhookWindowsHookEx(h);
            }
            if let Some(h) = self.mouse.take() {
                let _ = UnhookWindowsHookEx(h);
            }
        }
    }
}

fn hook_thread(ready: mpsc::Sender<Result<u32>>) {
    let p = CURRENT.load(Ordering::Acquire);
    if p.is_null() {
        let _ = ready.send(Err(Error::Other("listener state missing".into())));
        return;
    }
    // SAFETY: CURRENT outlives this thread (freed only after join).
    let shared = unsafe { &*p };
    let mut msg = MSG::default();
    let mut hooks = Hooks {
        keyboard: None,
        mouse: None,
    };
    // SAFETY: Win32 calls on this thread with valid pointers.
    unsafe {
        let _ = PeekMessageW(&mut msg, None, WM_USER, WM_USER, PM_NOREMOVE); // create the queue first
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    }
    shared.last_hook_tick.store(tick32(), Ordering::Relaxed);
    if let Err(e) = hooks.install() {
        let _ = ready.send(Err(Error::Other(e)));
        return;
    }
    PHYSICAL.clear();
    PHYSICAL.active.store(true, Ordering::Release);
    // SAFETY: no pointers.
    let tid = unsafe { GetCurrentThreadId() };
    let _ = ready.send(Ok(tid));
    // SAFETY: thread timer, killed below.
    let timer = unsafe { SetTimer(None, 0, WATCHDOG_MS, None) };
    let mut last_reinstall = 0u64;
    loop {
        // SAFETY: valid MSG pointer.
        let got = unsafe { GetMessageW(&mut msg, None, 0, 0) };
        if got.0 <= 0 {
            break;
        }
        if msg.message != WM_TIMER {
            continue;
        }
        // Watchdog: Windows removes a hook that times out without telling anyone.
        let mut info = LASTINPUTINFO {
            cbSize: std::mem::size_of::<LASTINPUTINFO>() as u32,
            dwTime: 0,
        };
        // SAFETY: valid struct pointer.
        if !unsafe { GetLastInputInfo(&mut info) }.as_bool() {
            continue;
        }
        let silent =
            info.dwTime
                .wrapping_sub(shared.last_hook_tick.load(Ordering::Relaxed)) as i32;
        let now = tick64();
        if silent > SILENT_MS && now.saturating_sub(last_reinstall) >= REINSTALL_GAP_MS {
            last_reinstall = now;
            match hooks.install() {
                Ok(()) => {
                    shared.last_hook_tick.store(tick32(), Ordering::Relaxed);
                    shared.reinstalls.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!("keyboard hook was dropped by Windows; reinstalled");
                }
                Err(e) => tracing::error!("could not reinstall the keyboard hook: {e}"),
            }
        }
    }
    // SAFETY: our own timer and hooks.
    unsafe {
        if timer != 0 {
            let _ = KillTimer(None, timer);
        }
    }
    hooks.uninstall();
    PHYSICAL.active.store(false, Ordering::Release);
    PHYSICAL.clear();
}

unsafe extern "system" fn mouse_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let p = CURRENT.load(Ordering::Acquire);
    if !p.is_null() {
        // SAFETY: see `keyboard_proc`.
        unsafe { (*p).last_hook_tick.store(tick32(), Ordering::Relaxed) };
    }
    // SAFETY: forwarding the arguments we were given.
    unsafe { CallNextHookEx(None, code, wparam, lparam) }
}

unsafe extern "system" fn keyboard_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    if code == HC_ACTION as i32 {
        let p = CURRENT.load(Ordering::Acquire);
        if !p.is_null() {
            // SAFETY: CURRENT is freed only after this (the only) hook thread has exited, and
            // lParam points at a KBDLLHOOKSTRUCT for HC_ACTION.
            let (shared, kb) = unsafe { (&*p, &*(lparam.0 as *const KBDLLHOOKSTRUCT)) };
            // Never let a bug break the user's keyboard: a panic passes the key through.
            let swallow =
                std::panic::catch_unwind(AssertUnwindSafe(|| handle(shared, wparam.0 as u32, kb)));
            if swallow.unwrap_or(false) {
                return LRESULT(1);
            }
        }
    }
    // SAFETY: forwarding the arguments we were given.
    unsafe { CallNextHookEx(None, code, wparam, lparam) }
}

/// One keyboard event. Returns whether to swallow it.
fn handle(shared: &Shared, message: u32, kb: &KBDLLHOOKSTRUCT) -> bool {
    shared.last_hook_tick.store(tick32(), Ordering::Relaxed);
    let injected = kb.flags.contains(LLKHF_INJECTED);
    if injected && !(ACCEPT_TEST_INPUT.load(Ordering::Relaxed) && kb.dwExtraInfo == TAG_TEST) {
        return false;
    }
    let down = message == WM_KEYDOWN || message == WM_SYSKEYDOWN;
    let vk = kb.vkCode;
    if !injected {
        PHYSICAL.update(vk, down);
    }
    let now = event_time(tick64(), tick32(), kb.time);
    if vk == VK_LCONTROL {
        if down && !shared.lctrl_fake.load(Ordering::Relaxed) {
            shared.lctrl_time.store(kb.time, Ordering::Relaxed);
            shared
                .lctrl_fake
                .store(kb.scanCode & 0x200 != 0, Ordering::Relaxed); // AltGr's Ctrl (0x21D)
        }
        if shared.lctrl_fake.load(Ordering::Relaxed) {
            PHYSICAL.update(VK_LCONTROL, false); // not a Ctrl the user is holding
            if !down {
                shared.lctrl_fake.store(false, Ordering::Relaxed);
            }
            return false; // AltGr's Ctrl: not a modifier the user pressed, not another key
        }
    } else if vk == VK_RMENU && down && kb.time == shared.lctrl_time.load(Ordering::Relaxed) {
        // AltGr whose synthesized Ctrl had a plain scan code: the shared timestamp gives it away.
        shared.lctrl_fake.store(true, Ordering::Relaxed);
        shared.lctrl_time.store(u32::MAX, Ordering::Relaxed);
        PHYSICAL.update(VK_LCONTROL, false);
        shared.driver.feed("left_ctrl", false, now);
    }
    let feed = shared.driver.feed(keys::win_name(vk), down, now);
    if let Some(key) = feed.replay_down
        && let Some(voice_vk) = keys::win_vk(key)
    {
        // The Voice key was a modifier in a chord: its press first, then this key, in order.
        let extended = kb.flags.contains(LLKHF_EXTENDED);
        let inputs = [
            key_input(voice_vk, false, TAG_REPLAY),
            raw_key_input(vk as u16, kb.scanCode as u16, extended, !down, TAG_REPLAY),
        ];
        if send(&inputs) == inputs.len() {
            return true;
        }
    }
    safe_swallow(feed.swallow, down, key_down(vk as u16))
}
