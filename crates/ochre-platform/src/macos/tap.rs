//! macOS Voice key listener: a CGEventTap on its own CFRunLoop thread (SPEC §6.1).
//!
//! Ported from the Python implementation. Default key: Right Option
//! (kVK_RightOption, 61), told apart from Left Option by the NX device-dependent flag bits, the
//! only way macOS exposes sides.
//!
//! * The tap listens to **keyboard events only** (keyDown, keyUp, flagsChanged). Handy's tap
//!   also filtered mouse buttons and broke back/forward buttons system-wide (Handy #1758).
//! * Only the Voice key's own events (and Escape while a take runs) are ever swallowed, and
//!   only while a gesture owns them. A key pressed with Right Option held still reaches the app
//!   with the Option flag (flags come from the HID state, not from the events we swallow), so
//!   Option-character entry keeps working and the gesture bows out (the chord rule).
//! * When the system disables the tap for being slow, it is re-enabled at once.
//! * **Revoked Accessibility freezes the whole Mac** while a filtering tap stays installed
//!   (Handy #1281). A watchdog polls `AXIsProcessTrusted` every 2 s and tears the tap down the
//!   moment the grant disappears; gestures stop until the listener is started again.
//! * Every event carries the full modifier state, so a modifier Voice key whose release was
//!   lost (tap briefly disabled) is recovered from the next event's flags.
//!
//! Limits: the tap needs **Accessibility** (to swallow) and **Input Monitoring** (to listen).
//! **Fn / Globe** (`"fn"`) works as a hold key from its flagsChanged event, but macOS acts on
//! Globe below the tap: set System Settings > Keyboard > "Press 🌐 key to" > "Do Nothing", or
//! the emoji picker / input-source switch fires too. **Caps Lock** cannot be held (macOS only
//! reports its toggle).

use std::ffi::c_void;
use std::panic::AssertUnwindSafe;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use objc2_core_foundation::{CFMachPort, CFRetained, CFRunLoop, kCFRunLoopCommonModes};
use objc2_core_graphics::{
    CGEvent, CGEventField, CGEventFlags, CGEventSource, CGEventSourceStateID, CGEventTapLocation,
    CGEventTapOptions, CGEventTapPlacement, CGEventTapProxy, CGEventType,
};
use ochre_core::config::HotkeyConfig;
use ochre_core::platform::{GestureFn, HotkeyListener};
use ochre_core::{Error, Result};

use crate::driver::{Driver, monotonic_ms};
use crate::keys::{self, KeySpec};

/// `eventSourceUserData` on events we post ("OWFR"); the tap passes them through.
pub const OWN_EVENT_TAG: i64 = 0x4F57_4652;

/// NX device-dependent modifier bits (IOLLEvent.h) per sided name, plus Fn's flag.
pub fn device_bit(name: &str) -> Option<u64> {
    Some(match name {
        "left_ctrl" => 0x1,
        "left_shift" => 0x2,
        "right_shift" => 0x4,
        "left_win" => 0x8,
        "right_win" => 0x10,
        "left_alt" => 0x20,
        "right_alt" => 0x40,
        "right_ctrl" => 0x2000,
        "fn" => 0x80_0000,
        _ => return None,
    })
}

/// Press (Some(true)), release (Some(false)) or not routed (None) for one event.
pub fn event_down(event_type: u32, name: &str, flags: u64) -> Option<bool> {
    match event_type {
        10 => Some(true),                                   // keyDown
        11 => Some(false),                                  // keyUp
        12 => device_bit(name).map(|bit| flags & bit != 0), // flagsChanged
        _ => None,
    }
}

fn validate(spec: &KeySpec) -> std::result::Result<(), String> {
    if let Some(why) = keys::mac_unsupported(spec.key) {
        return Err(format!(
            "{:?} cannot be the Voice key on macOS: {why}",
            spec.key
        ));
    }
    match keys::mac_keycode(spec.key) {
        Some(_) => Ok(()),
        None => Err(format!(
            "{:?} cannot be used as the Voice key on macOS",
            spec.key
        )),
    }
}

struct TapCtx {
    driver: Arc<Driver>,
    tap: AtomicPtr<CFMachPort>,
}

struct SendRunLoop(CFRetained<CFRunLoop>);
// SAFETY: CFRunLoopStop is documented thread-safe; nothing else is called from other threads.
unsafe impl Send for SendRunLoop {}

impl SendRunLoop {
    /// Stop the loop (a method, so closures capture the Send wrapper, not the field).
    fn stop(&self) {
        self.0.stop();
    }
}

struct Running {
    thread: JoinHandle<()>,
    run_loop: SendRunLoop,
    watchdog: JoinHandle<()>,
    stop: Arc<AtomicBool>,
}

/// [`HotkeyListener`] on a CGEventTap.
pub struct MacHotkeys {
    driver: Arc<Driver>,
    running: Mutex<Option<Running>>,
}

impl MacHotkeys {
    pub fn new(cfg: &HotkeyConfig) -> Result<Self> {
        Ok(Self {
            driver: Driver::new(cfg, monotonic_ms, validate)?,
            running: Mutex::new(None),
        })
    }
}

impl HotkeyListener for MacHotkeys {
    fn start(&mut self, on_gesture: GestureFn) -> Result<()> {
        let mut slot = self.running.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_some() {
            return Ok(());
        }
        if !super::accessibility_trusted() {
            return Err(Error::Permission(
                crate::permissions::mac_accessibility_fix(),
            ));
        }
        self.driver.start(on_gesture)?;
        let ctx = Arc::new(TapCtx {
            driver: Arc::clone(&self.driver),
            tap: AtomicPtr::new(ptr::null_mut()),
        });
        let (ready_tx, ready_rx) = mpsc::channel::<Result<SendRunLoop>>();
        let thread_ctx = Arc::clone(&ctx);
        let thread = std::thread::Builder::new()
            .name("ochre-eventtap".into())
            .spawn(move || run(thread_ctx, ready_tx))?;
        let run_loop = match ready_rx.recv_timeout(Duration::from_secs(3)) {
            Ok(Ok(rl)) => rl,
            outcome => {
                let _ = thread.join();
                self.driver.stop();
                return Err(match outcome {
                    Ok(Err(e)) => e,
                    _ => Error::Other("the event tap thread did not start".into()),
                });
            }
        };
        // Watchdog: a filtering tap left in place after Accessibility is revoked freezes input.
        let stop = Arc::new(AtomicBool::new(false));
        let watch_stop = Arc::clone(&stop);
        let watch_loop = SendRunLoop(run_loop.0.clone());
        let watchdog = std::thread::Builder::new().name("ochre-eventtap-watchdog".into()).spawn(move || {
            while !watch_stop.load(Ordering::Acquire) {
                std::thread::park_timeout(Duration::from_secs(2));
                if !watch_stop.load(Ordering::Acquire) && !super::accessibility_trusted() {
                    tracing::error!("Accessibility was revoked: removing the event tap so input keeps flowing");
                    watch_loop.stop();
                    return;
                }
            }
        })?;
        *slot = Some(Running {
            thread,
            run_loop,
            watchdog,
            stop,
        });
        Ok(())
    }

    fn set_key(&mut self, key: &str) -> Result<()> {
        self.driver.set_key(key)
    }

    fn set_recording(&self, recording: bool) {
        self.driver.set_recording(recording);
    }

    fn stop(&mut self) {
        let running = self
            .running
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(r) = running {
            r.stop.store(true, Ordering::Release);
            r.watchdog.thread().unpark();
            r.run_loop.stop();
            let _ = r.thread.join();
            let _ = r.watchdog.join();
            // Swallowed flagsChanged events never changed the HID state, so there is nothing
            // to release for modifiers; a replayed non-modifier key gets its up here.
            let spec = self.driver.key();
            if self.driver.involved()
                && keys::modifier_class(spec.key) == 0
                && let Some(code) = keys::mac_keycode(spec.key)
                && CGEventSource::key_state(CGEventSourceStateID::CombinedSessionState, code)
            {
                post_key(code, false);
            }
            self.driver.stop();
            self.driver.reset();
        }
    }
}

impl Drop for MacHotkeys {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(ctx: Arc<TapCtx>, ready: mpsc::Sender<Result<SendRunLoop>>) {
    let mask: u64 = (1 << CGEventType::KeyDown.0)
        | (1 << CGEventType::KeyUp.0)
        | (1 << CGEventType::FlagsChanged.0);
    let user_info = Arc::as_ptr(&ctx) as *mut c_void;
    // SAFETY: `ctx` outlives the tap: the run loop (and with it every callback) ends before
    // this function returns and drops it.
    let tap = unsafe {
        CGEvent::tap_create(
            CGEventTapLocation::SessionEventTap,
            CGEventTapPlacement::HeadInsertEventTap,
            CGEventTapOptions::Default,
            mask,
            Some(callback),
            user_info,
        )
    };
    let Some(tap) = tap else {
        let missing: Vec<String> = crate::permissions::check()
            .into_iter()
            .map(|i| i.fix)
            .collect();
        let msg = if missing.is_empty() {
            "macOS refused the keyboard event tap (Accessibility / Input Monitoring).".to_string()
        } else {
            format!(
                "macOS refused the keyboard event tap. {}",
                missing.join(" ")
            )
        };
        let _ = ready.send(Err(Error::Permission(msg)));
        return;
    };
    ctx.tap
        .store(CFRetained::as_ptr(&tap).as_ptr(), Ordering::Release);
    let Some(source) = CFMachPort::new_run_loop_source(None, Some(&tap), 0) else {
        let _ = ready.send(Err(Error::Other(
            "CFMachPortCreateRunLoopSource failed".into(),
        )));
        return;
    };
    let Some(run_loop) = CFRunLoop::current() else {
        let _ = ready.send(Err(Error::Other("no CFRunLoop on the tap thread".into())));
        return;
    };
    // SAFETY: reading an immutable framework constant.
    let common = unsafe { kCFRunLoopCommonModes };
    run_loop.add_source(Some(&source), common);
    CGEvent::tap_enable(&tap, true);
    let _ = ready.send(Ok(SendRunLoop(run_loop.clone())));
    CFRunLoop::run();
    CGEvent::tap_enable(&tap, false);
    run_loop.remove_source(Some(&source), common);
    tap.invalidate();
    ctx.tap.store(ptr::null_mut(), Ordering::Release);
}

unsafe extern "C-unwind" fn callback(
    proxy: CGEventTapProxy,
    event_type: CGEventType,
    event: NonNull<CGEvent>,
    user_info: *mut c_void,
) -> *mut CGEvent {
    // SAFETY: user_info is the TapCtx kept alive by `run`.
    let ctx = unsafe { &*(user_info as *const TapCtx) };
    let pass = event.as_ptr();
    if event_type == CGEventType::TapDisabledByTimeout
        || event_type == CGEventType::TapDisabledByUserInput
    {
        let tap = ctx.tap.load(Ordering::Acquire);
        if !tap.is_null() {
            // SAFETY: the tap is alive while its run loop runs this callback.
            CGEvent::tap_enable(unsafe { &*tap }, true);
        }
        return pass;
    }
    // SAFETY: the event is valid for the duration of the callback.
    let ev = unsafe { event.as_ref() };
    // Never let a bug break the user's keyboard: a panic passes the event through.
    let swallow = std::panic::catch_unwind(AssertUnwindSafe(|| handle(ctx, proxy, event_type, ev)));
    if swallow.unwrap_or(false) {
        ptr::null_mut()
    } else {
        pass
    }
}

fn handle(ctx: &TapCtx, proxy: CGEventTapProxy, event_type: CGEventType, ev: &CGEvent) -> bool {
    if CGEvent::integer_value_field(Some(ev), CGEventField::EventSourceUserData) == OWN_EVENT_TAG {
        return false;
    }
    let code = CGEvent::integer_value_field(Some(ev), CGEventField::KeyboardEventKeycode) as u16;
    let flags = CGEvent::flags(Some(ev)).0;
    let voice = ctx.driver.key().key;
    let name = if keys::mac_matches(code, voice) {
        voice
    } else {
        keys::mac_name(code)
    };
    let now = monotonic_ms();
    // A modifier Voice key we believe is down but whose flag is clear lost its release.
    if let Some(bit) = device_bit(voice)
        && name != voice
        && ctx.driver.key_down()
        && flags & bit == 0
    {
        ctx.driver.feed(voice, false, now);
    }
    let Some(down) = event_down(event_type.0, name, flags) else {
        return false;
    };
    // A chord Voice key's first modifier: the mic can start opening before the chord completes.
    let spec = ctx.driver.key();
    if down
        && event_type == CGEventType::FlagsChanged
        && spec.is_chord()
        && keys::modifier_class(name) & (spec.mods | keys::modifier_class(spec.key)) != 0
    {
        crate::prime();
    }
    let feed = ctx.driver.feed(name, down, now);
    if let Some(key) = feed.replay_down
        && keys::modifier_class(key) == 0
        && let Some(code) = keys::mac_keycode(key)
        && let Some(replay) = CGEvent::new_keyboard_event(None, code, true)
    {
        // A non-modifier Voice key used in a chord: put its press back ahead of this key.
        // (Modifier flags already reach the app from the HID state, so modifiers need nothing.)
        CGEvent::set_integer_value_field(
            Some(&replay),
            CGEventField::EventSourceUserData,
            OWN_EVENT_TAG,
        );
        // SAFETY: posting from inside the tap callback with the proxy we were given.
        unsafe { CGEvent::tap_post_event(proxy, Some(&replay)) };
    }
    feed.swallow
}

/// Post one tagged key event at the HID level.
pub(crate) fn post_key(code: u16, down: bool) {
    if let Some(event) = CGEvent::new_keyboard_event(None, code, down) {
        CGEvent::set_flags(Some(&event), CGEventFlags(0));
        CGEvent::set_integer_value_field(
            Some(&event),
            CGEventField::EventSourceUserData,
            OWN_EVENT_TAG,
        );
        CGEvent::post(CGEventTapLocation::HIDEventTap, Some(&event));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_decide_modifier_direction() {
        assert_eq!(event_down(12, "right_alt", 0x80040), Some(true));
        assert_eq!(
            event_down(12, "right_alt", 0x80020),
            Some(false),
            "only Left Option still down"
        );
        assert_eq!(event_down(12, "fn", 0x80_0000), Some(true));
        assert_eq!(event_down(10, "f13", 0), Some(true));
        assert_eq!(event_down(11, "f13", 0), Some(false));
        assert_eq!(event_down(12, "caps_lock", 0), None);
        assert!(validate(&keys::parse_key("caps_lock").unwrap()).is_err());
        assert!(validate(&keys::parse_key("right_alt").unwrap()).is_ok());
    }
}
