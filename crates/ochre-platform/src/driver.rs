//! Shared plumbing behind every OS hotkey listener.
//!
//! The OS adapters only translate native key events into key names (see [`crate::keys`]) and
//! apply the swallow answer. Everything else lives here, once:
//!
//! * **Routing:** is this the Voice key, Escape, the paste-last key, a modifier or some other
//!   key; is the press
//!   eligible (exactly the chord's modifiers held, so it is dictation and not a shortcut); is
//!   the raw modifier held.
//! * **Locking:** hook threads call [`Driver::feed`] and must get an answer in microseconds,
//!   so the mutex is held only around the pure state machine, never while calling out.
//! * **Timer:** one thread sleeps on a condvar until the machine's next deadline (the end of
//!   the double-tap window). No polling.
//! * **Dispatch:** gestures go through a channel to a dispatcher thread that calls
//!   `on_gesture`. The hook never runs app code, because Windows and macOS silently drop a hook
//!   that answers too slowly, which can leave a key (e.g. Right Alt) stuck for every app.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use ochre_core::config::HotkeyConfig;
use ochre_core::platform::{Gesture, GestureFn};
use ochre_core::{Error, Result};

use crate::gesture::{Decision, Machine};
use crate::keys::{self, KeySpec, PasteKey};

/// Checks that the OS can use a parsed key (e.g. macOS has no Caps Lock hold).
pub type Validate = fn(&KeySpec) -> std::result::Result<(), String>;

/// What the adapter must do with one native event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Feed {
    pub swallow: bool,
    /// Send this Voice key's *down* to the OS before the current event (a chord).
    pub replay_down: Option<&'static str>,
}

struct Inner {
    machine: Machine,
    spec: KeySpec,
    pending: Option<KeySpec>,
    raw_mod: u8,
    paste: PasteKey,
    /// The paste-last key whose press we hid: its repeats and release are hidden too.
    paste_owned: Option<&'static str>,
    held: [bool; 8],
    running: bool,
}

impl Inner {
    fn held_classes(&self) -> u8 {
        keys::SIDED_MODIFIERS
            .iter()
            .zip(self.held)
            .filter(|((name, _), down)| *down && *name != self.spec.key)
            .fold(0, |acc, ((_, class), _)| acc | class)
    }

    fn raw_usable(&self) -> bool {
        let raw = self.raw_mod;
        raw != 0 && self.spec.mods & raw == 0 && keys::modifier_class(self.spec.key) != raw
    }

    /// A single key: no other modifier held. A chord: exactly its modifiers. The raw modifier
    /// never makes a press ineligible (Shift+Voice key starts a raw take).
    fn eligible(&self) -> bool {
        let mut held = self.held_classes();
        if self.raw_usable() {
            held &= !self.raw_mod;
        }
        held == self.spec.mods
    }

    fn raw(&self) -> bool {
        self.raw_usable() && self.held_classes() & self.raw_mod != 0
    }

    fn route(&mut self, name: &'static str, down: bool, now: u64) -> Decision {
        let slot = keys::modifier_slot(name);
        if let Some(i) = slot {
            self.held[i] = down;
        }
        if name == self.spec.key {
            let (eligible, raw) = (self.eligible(), self.raw());
            return self.machine.key(down, now, eligible, raw);
        }
        if slot.is_some() {
            return Decision::default(); // modifiers never break a gesture (raw modifier, AltGr's Ctrl)
        }
        if name == "escape" {
            return self.machine.escape(down);
        }
        if let Some(d) = self.paste_key(name, down) {
            return d;
        }
        self.machine.other(down, now)
    }

    /// Every held modifier class, the Voice key's included.
    fn all_held_classes(&self) -> u8 {
        keys::SIDED_MODIFIERS
            .iter()
            .zip(self.held)
            .filter(|(_, down)| *down)
            .fold(0, |acc, ((_, class), _)| acc | class)
    }

    /// The paste-last key. `None`: not ours, route it as an ordinary key.
    fn paste_key(&mut self, name: &'static str, down: bool) -> Option<Decision> {
        if self.paste_owned == Some(name) {
            if !down {
                self.paste_owned = None;
            }
            return Some(Decision {
                swallow: true,
                ..Decision::default()
            }); // auto-repeat (fires once) or the release of the press we hid
        }
        if !down {
            return None;
        }
        let d = match self.paste {
            PasteKey::WithVoice(k) if k == name && k != self.spec.key => {
                self.machine.paste_chord()?
            }
            PasteKey::Combo(c)
                if c.key == name && c != self.spec && self.all_held_classes() == c.mods =>
            {
                self.machine.paste_combo()
            }
            _ => return None,
        };
        self.paste_owned = Some(name);
        Some(d)
    }

    fn apply_pending(&mut self) {
        if self.pending.is_some() && self.machine.idle() {
            self.spec = self.pending.take().unwrap_or(self.spec);
        }
    }
}

/// Thread-safe owner of a gesture [`Machine`] for one OS adapter.
pub struct Driver {
    inner: Mutex<Inner>,
    wake: Condvar,
    tx: Mutex<Option<Sender<Option<Gesture>>>>,
    clock: fn() -> u64,
    validate: Validate,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner()) // a panicked thread must never jam the keyboard
}

/// Parse and validate a key, mapping failures to a config error.
pub fn parse_key(key: &str, validate: Validate) -> Result<KeySpec> {
    let spec = keys::parse_key(key).map_err(Error::Config)?;
    validate(&spec).map_err(Error::Config)?;
    Ok(spec)
}

impl Driver {
    pub fn new(cfg: &HotkeyConfig, clock: fn() -> u64, validate: Validate) -> Result<Arc<Self>> {
        let spec = parse_key(&cfg.key, validate)?;
        let raw_mod = keys::parse_modifier(&cfg.raw_modifier).map_err(Error::Config)?;
        // A bad paste-last setting must never cost the user the Voice key: warn and turn it off.
        let paste = keys::parse_paste_key(&cfg.paste_last).unwrap_or_else(|e| {
            tracing::warn!("hotkey.paste_last ignored: {e}");
            PasteKey::Off
        });
        Ok(Arc::new(Self {
            inner: Mutex::new(Inner {
                machine: Machine::new(cfg.double_tap_ms, cfg.hold_min_ms),
                spec,
                pending: None,
                raw_mod,
                paste,
                paste_owned: None,
                held: [false; 8],
                running: false,
            }),
            wake: Condvar::new(),
            tx: Mutex::new(None),
            clock,
            validate,
            threads: Mutex::new(Vec::new()),
        }))
    }

    pub fn now(&self) -> u64 {
        (self.clock)()
    }

    /// Start the timer and dispatcher threads.
    pub fn start(self: &Arc<Self>, on_gesture: GestureFn) -> Result<()> {
        let (tx, rx) = channel::<Option<Gesture>>();
        *lock(&self.tx) = Some(tx);
        lock(&self.inner).running = true;
        let dispatcher = std::thread::Builder::new()
            .name("ochre-gestures".into())
            .spawn(move || dispatch_loop(rx, on_gesture))?;
        let me = Arc::clone(self);
        let timer = std::thread::Builder::new()
            .name("ochre-gesture-timer".into())
            .spawn(move || me.tick_loop())?;
        lock(&self.threads).extend([dispatcher, timer]);
        Ok(())
    }

    /// Stop the threads (idempotent). Gestures already queued are still delivered.
    pub fn stop(&self) {
        lock(&self.inner).running = false;
        self.wake.notify_all();
        if let Some(tx) = lock(&self.tx).take() {
            let _ = tx.send(None);
        }
        let threads = std::mem::take(&mut *lock(&self.threads));
        let me = std::thread::current().id();
        for t in threads {
            if t.thread().id() != me {
                let _ = t.join();
            }
        }
    }

    /// One native key event, already translated to a key name. Called on the hook thread:
    /// never blocks beyond the mutex, never panics out.
    pub fn feed(&self, name: &'static str, down: bool, now: u64) -> Feed {
        let (decision, key) = {
            let mut inner = lock(&self.inner);
            let key = inner.spec.key;
            let d = inner.route(name, down, now);
            inner.apply_pending();
            (d, key)
        };
        self.post(&decision);
        self.wake.notify_all(); // the deadline may have moved (a release opens the double-tap window)
        Feed {
            swallow: decision.swallow,
            replay_down: decision.replay_down.then_some(key),
        }
    }

    pub fn set_key(&self, key: &str) -> Result<()> {
        let spec = parse_key(key, self.validate)?;
        let mut inner = lock(&self.inner);
        inner.pending = Some(spec);
        inner.apply_pending(); // takes effect between gestures only
        Ok(())
    }

    pub fn set_recording(&self, recording: bool) {
        lock(&self.inner).machine.set_recording(recording);
        self.wake.notify_all();
    }

    /// The current Voice key.
    pub fn key(&self) -> KeySpec {
        lock(&self.inner).spec
    }

    /// The machine believes the Voice key is down.
    pub fn key_down(&self) -> bool {
        lock(&self.inner).machine.key_down()
    }

    pub fn involved(&self) -> bool {
        lock(&self.inner).machine.involved()
    }

    pub fn reset(&self) {
        let mut inner = lock(&self.inner);
        inner.machine.reset();
        inner.held = [false; 8];
        inner.paste_owned = None;
    }

    fn post(&self, d: &Decision) {
        if d.gestures.is_empty() {
            return;
        }
        if let Some(tx) = lock(&self.tx).as_ref() {
            for g in d.gestures.iter() {
                let _ = tx.send(Some(g));
            }
        }
    }

    fn tick_loop(&self) {
        let mut inner = lock(&self.inner);
        loop {
            if !inner.running {
                return;
            }
            let now = self.now();
            match inner.machine.next_deadline() {
                None => inner = self.wake.wait(inner).unwrap_or_else(|e| e.into_inner()),
                Some(at) if at > now => {
                    let wait = Duration::from_millis(at - now);
                    inner = self
                        .wake
                        .wait_timeout(inner, wait)
                        .unwrap_or_else(|e| e.into_inner())
                        .0;
                }
                Some(_) => {
                    let d = inner.machine.tick(now);
                    inner.apply_pending();
                    drop(inner);
                    self.post(&d);
                    inner = lock(&self.inner);
                }
            }
        }
    }
}

fn dispatch_loop(rx: Receiver<Option<Gesture>>, on_gesture: GestureFn) {
    while let Ok(Some(g)) = rx.recv() {
        let call = std::panic::AssertUnwindSafe(|| on_gesture(g));
        if std::panic::catch_unwind(call).is_err() {
            tracing::error!(?g, "on_gesture panicked");
        }
    }
}

/// Monotonic milliseconds since the first call (for adapters without an OS event clock).
pub fn monotonic_ms() -> u64 {
    use std::sync::OnceLock;
    use std::time::Instant;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// Gestures for a `toggle` / `start` / `stop` / `cancel` command from a second `ochre`
/// invocation (the Wayland compositor-shortcut fallback). `recording` is the app's state.
pub fn toggle_gestures(action: &str, recording: bool) -> Vec<Gesture> {
    let action = if action == "toggle" {
        if recording { "stop" } else { "start" }
    } else {
        action
    };
    match (action, recording) {
        ("start", false) => vec![Gesture::Press, Gesture::Lock],
        ("stop", true) => vec![Gesture::Finish],
        ("cancel", true) => vec![Gesture::Cancel],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    static CLOCK: AtomicU64 = AtomicU64::new(0);
    fn test_clock() -> u64 {
        CLOCK.load(Ordering::SeqCst)
    }
    fn ok(_: &KeySpec) -> std::result::Result<(), String> {
        Ok(())
    }

    fn driver(key: &str) -> Arc<Driver> {
        let cfg = HotkeyConfig {
            key: key.into(),
            ..HotkeyConfig::default()
        };
        Driver::new(&cfg, test_clock, ok).unwrap()
    }

    #[test]
    fn routing_eligibility_and_raw() {
        let d = driver("right_alt");
        // Left Ctrl held: a shortcut, not dictation.
        d.feed("left_ctrl", true, 0);
        assert!(!d.feed("right_alt", true, 0).swallow);
        assert!(!d.feed("right_alt", false, 10).swallow);
        d.feed("left_ctrl", false, 20);
        // Shift (the raw modifier) held: eligible, and a raw take.
        d.feed("left_shift", true, 30);
        assert!(d.feed("right_alt", true, 40).swallow);
        d.feed("left_shift", false, 50);
        assert!(d.feed("right_alt", false, 900).swallow);
        // Modifiers never count as "another key" (no chord abort).
        assert!(d.feed("right_alt", true, 2000).swallow);
        assert_eq!(d.feed("left_ctrl", true, 2010), Feed::default());
        d.feed("left_ctrl", false, 2020);
        // A letter does: replay the Voice key.
        let f = d.feed("e", true, 2030);
        assert_eq!(f.replay_down, Some("right_alt"));
        assert!(!f.swallow);
    }

    #[test]
    fn chord_key_requires_exact_modifiers() {
        let d = driver("ctrl+shift+space");
        assert!(!d.feed("space", true, 0).swallow, "space alone is typing");
        d.feed("space", false, 10);
        d.feed("left_ctrl", true, 20);
        assert!(
            !d.feed("space", true, 30).swallow,
            "ctrl+space is someone else's"
        );
        d.feed("space", false, 40);
        d.feed("right_shift", true, 50);
        assert!(d.feed("space", true, 60).swallow);
        assert!(d.feed("space", false, 900).swallow);
    }

    #[test]
    fn modifier_voice_key_does_not_count_against_itself() {
        let d = driver("right_ctrl");
        assert!(d.feed("right_ctrl", true, 0).swallow);
        assert!(d.feed("right_ctrl", false, 400).swallow);
    }

    #[test]
    fn escape_routing() {
        let d = driver("right_alt");
        assert!(!d.feed("escape", true, 0).swallow);
        d.feed("escape", false, 5);
        d.set_recording(true);
        assert!(d.feed("escape", true, 10).swallow);
        assert!(d.feed("escape", false, 15).swallow);
    }

    #[test]
    fn set_key_waits_for_idle() {
        let d = driver("right_alt");
        d.feed("right_alt", true, 0);
        d.set_key("f13").unwrap();
        assert_eq!(d.key().key, "right_alt", "a take in progress keeps its key");
        d.feed("right_alt", false, 400);
        assert_eq!(d.key().key, "f13");
        assert!(d.set_key("a").is_err());
        assert_eq!(d.key().key, "f13");
    }

    #[test]
    fn invalid_config_is_a_config_error() {
        let cfg = HotkeyConfig {
            key: "q".into(),
            ..HotkeyConfig::default()
        };
        assert_eq!(
            Driver::new(&cfg, test_clock, ok).err().unwrap().code(),
            "config"
        );
        let cfg = HotkeyConfig {
            raw_modifier: "space".into(),
            ..HotkeyConfig::default()
        };
        assert_eq!(
            Driver::new(&cfg, test_clock, ok).err().unwrap().code(),
            "config"
        );
        fn deny(_: &KeySpec) -> std::result::Result<(), String> {
            Err("nope".into())
        }
        assert!(Driver::new(&HotkeyConfig::default(), test_clock, deny).is_err());
    }

    #[test]
    fn timer_and_dispatcher_deliver_abort() {
        let cfg = HotkeyConfig {
            double_tap_ms: 60,
            ..HotkeyConfig::default()
        };
        let d = Driver::new(&cfg, monotonic_ms, ok).unwrap();
        let (tx, rx) = channel();
        let tx = Mutex::new(tx);
        d.start(Box::new(move |g| {
            let _ = lock(&tx).send(g);
        }))
        .unwrap();
        let t0 = monotonic_ms();
        d.feed("right_alt", true, t0);
        d.feed("right_alt", false, t0 + 20);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            Gesture::Press
        );
        let start = Instant::now();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            Gesture::Abort
        );
        assert!(start.elapsed() < Duration::from_millis(500));
        d.stop();
        d.stop(); // idempotent
    }

    #[test]
    fn panicking_callback_does_not_kill_dispatch() {
        let d = driver("right_alt");
        let (tx, rx) = channel();
        let tx = Mutex::new(tx);
        d.start(Box::new(move |g| {
            if g == Gesture::Press {
                panic!("boom");
            }
            let _ = lock(&tx).send(g);
        }))
        .unwrap();
        d.feed("right_alt", true, 0);
        d.feed("right_alt", false, 1000);
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            Gesture::Release
        );
        d.stop();
    }

    /// A started driver whose gestures land in a channel.
    fn started(key: &str, paste: &str) -> (Arc<Driver>, Receiver<Gesture>) {
        let cfg = HotkeyConfig {
            key: key.into(),
            paste_last: paste.into(),
            ..HotkeyConfig::default()
        };
        let d = Driver::new(&cfg, test_clock, ok).unwrap();
        let (tx, rx) = channel();
        let tx = Mutex::new(tx);
        d.start(Box::new(move |g| {
            let _ = lock(&tx).send(g);
        }))
        .unwrap();
        (d, rx)
    }

    /// Everything dispatched so far (waits briefly for the dispatcher thread).
    fn drain(rx: &Receiver<Gesture>) -> Vec<Gesture> {
        let mut out = Vec::new();
        while let Ok(g) = rx.recv_timeout(Duration::from_millis(80)) {
            out.push(g);
        }
        out
    }

    #[test]
    fn voice_key_plus_down_pastes_once_and_cancels_the_take() {
        let (d, rx) = started("right_ctrl", "down");
        assert!(d.feed("right_ctrl", true, 0).swallow);
        // Down (and its auto-repeat) is hidden from the app and fires once.
        assert!(d.feed("down", true, 120).swallow);
        for t in (150..900).step_by(30) {
            assert!(d.feed("right_ctrl", true, t).swallow);
            assert!(d.feed("down", true, t).swallow, "repeat at {t}");
        }
        assert!(d.feed("down", false, 950).swallow);
        // The Voice key's release is inert, whenever it comes.
        assert!(d.feed("right_ctrl", false, 1000).swallow);
        assert_eq!(
            drain(&rx),
            vec![Gesture::Press, Gesture::Abort, Gesture::PasteLast]
        );
        // ...and it does not count as the first tap of a double tap.
        assert!(d.feed("right_ctrl", true, 1100).swallow);
        assert!(d.feed("right_ctrl", false, 1500).swallow);
        assert_eq!(drain(&rx), vec![Gesture::Press, Gesture::Release]);
        d.stop();
    }

    #[test]
    fn paste_chord_with_a_quick_release_and_down_released_last() {
        let (d, rx) = started("right_ctrl", "down");
        d.feed("right_ctrl", true, 0);
        d.feed("down", true, 30);
        assert!(d.feed("right_ctrl", false, 60).swallow); // short: would be a first tap
        assert!(
            d.feed("down", false, 90).swallow,
            "release of the hidden Down"
        );
        // A press right after is a new take, not a double tap.
        d.feed("right_ctrl", true, 150);
        assert_eq!(
            drain(&rx),
            vec![
                Gesture::Press,
                Gesture::Abort,
                Gesture::PasteLast,
                Gesture::Press
            ]
        );
        d.stop();
    }

    #[test]
    fn down_without_the_voice_key_is_untouched() {
        let (d, rx) = started("right_ctrl", "down");
        assert_eq!(d.feed("down", true, 0), Feed::default());
        assert_eq!(d.feed("down", false, 40), Feed::default());
        // Left Ctrl + Down is somebody else's shortcut.
        d.feed("left_ctrl", true, 100);
        assert_eq!(d.feed("down", true, 110), Feed::default());
        assert_eq!(d.feed("down", false, 120), Feed::default());
        d.feed("left_ctrl", false, 130);
        assert!(drain(&rx).is_empty());
        d.stop();
    }

    #[test]
    fn normal_gestures_unaffected_by_paste_last() {
        let (d, rx) = started("right_ctrl", "down");
        // Hold.
        d.feed("right_ctrl", true, 0);
        d.feed("right_ctrl", false, 800);
        // Double tap, then finish.
        d.feed("right_ctrl", true, 2000);
        d.feed("right_ctrl", false, 2050);
        d.feed("right_ctrl", true, 2150);
        d.feed("right_ctrl", false, 2200);
        d.feed("right_ctrl", true, 5000);
        d.feed("right_ctrl", false, 5050);
        // Another key while held is still a chord (Right Ctrl+C), not a paste.
        d.feed("right_ctrl", true, 7000);
        let f = d.feed("c", true, 7050);
        assert_eq!(f.replay_down, Some("right_ctrl"));
        d.feed("c", false, 7080);
        // Down after the chord belongs to the OS too.
        assert!(!d.feed("down", true, 7100).swallow);
        d.feed("down", false, 7120);
        assert!(!d.feed("right_ctrl", false, 7200).swallow);
        assert_eq!(
            drain(&rx),
            vec![
                Gesture::Press,
                Gesture::Release,
                Gesture::Press,
                Gesture::Lock,
                Gesture::Finish,
                Gesture::Press,
                Gesture::Abort
            ]
        );
        d.stop();
    }

    #[test]
    fn paste_last_off_and_standalone_combo() {
        let (d, rx) = started("right_ctrl", "");
        d.feed("right_ctrl", true, 0);
        let f = d.feed("down", true, 50);
        assert_eq!(
            f.replay_down,
            Some("right_ctrl"),
            "off: Down is just a chord"
        );
        d.feed("down", false, 60);
        d.feed("right_ctrl", false, 70);
        assert_eq!(drain(&rx), vec![Gesture::Press, Gesture::Abort]);
        d.stop();

        let (d, rx) = started("right_alt", "ctrl+alt+v");
        assert_eq!(d.feed("v", true, 0), Feed::default(), "plain v types");
        d.feed("v", false, 10);
        d.feed("left_ctrl", true, 100);
        d.feed("left_alt", true, 110);
        assert!(d.feed("v", true, 120).swallow);
        assert!(d.feed("v", true, 150).swallow, "repeat");
        assert!(d.feed("v", false, 180).swallow);
        d.feed("left_alt", false, 190);
        d.feed("left_ctrl", false, 200);
        // Ctrl+V alone stays the app's paste.
        d.feed("left_ctrl", true, 300);
        assert!(!d.feed("v", true, 310).swallow);
        d.feed("v", false, 320);
        d.feed("left_ctrl", false, 330);
        assert_eq!(drain(&rx), vec![Gesture::PasteLast]);
        d.stop();
    }

    #[test]
    fn bad_paste_last_never_breaks_the_voice_key() {
        let cfg = HotkeyConfig {
            paste_last: "left_shift".into(),
            ..HotkeyConfig::default()
        };
        assert!(Driver::new(&cfg, test_clock, ok).is_ok());
    }

    #[test]
    fn toggle_commands() {
        assert_eq!(
            toggle_gestures("toggle", false),
            vec![Gesture::Press, Gesture::Lock]
        );
        assert_eq!(toggle_gestures("toggle", true), vec![Gesture::Finish]);
        assert_eq!(toggle_gestures("cancel", true), vec![Gesture::Cancel]);
        assert!(toggle_gestures("cancel", false).is_empty());
        assert!(toggle_gestures("start", true).is_empty());
        assert!(toggle_gestures("bogus", true).is_empty());
    }
}
