//! The Voice key gesture as a pure, clock-injected state machine (SPEC §6.1).
//!
//! A port of the earlier native gesture code (and its Python port), reshaped for
//! [`ochre_core::platform::Gesture`]. It only decides what a key transition *means*;
//! the OS adapters own the hooks, the clock and the replays. Nothing here touches the OS, so
//! every edge case is unit-tested below.
//!
//! What the orchestrator sees:
//!
//! * **Hold:** `Press` on key down (recording starts at once, so no audio is lost), then
//!   `Release` when a press of at least `hold_min_ms` ends.
//! * **Double tap:** `Press` on the first down. A short first press leaves recording running
//!   and waits; a second down within `double_tap_ms` of the first release gives `Lock`. The
//!   next press of the key gives `Finish` *on key down* (no need to wait for the release).
//! * **Lone tap:** `Press`, then `Abort` once the double-tap window expires, or as soon as
//!   another key is typed. The tap is not replayed: Right Alt must never open menus.
//! * **Escape** while anything is running (or the app says it is recording): `Cancel`.
//! * **Raw:** `FinishRaw` replaces `Release` / `Finish` when the raw modifier was held when
//!   the take started or when it finishes.
//! * **Chord:** another (non-modifier) key goes down while the Voice key is held, so the Voice
//!   key was being used as a modifier (AltGr+e, Right Ctrl+C): `Abort`, and the adapter
//!   replays the swallowed key press before the other key so the chord still works.
//! * **Paste last:** the paste-last key (Down by default) goes down while the Voice key is held:
//!   `Abort` for the take that press just started (silently discarded), then `PasteLast`. The
//!   Voice key's release afterwards is inert (no finish, no double-tap); the driver swallows
//!   the paste key's press, repeats and release so the app underneath never sees them.
//!
//! Swallowing rules (the "never jam the keyboard" hardening):
//!
//! * A press is swallowed only when this machine takes ownership of it; its repeats and its
//!   release are then swallowed too. A press that is passed through (ineligible, or part of a
//!   chord) has its release passed through as well, so the OS always sees matched pairs.
//! * A release we never saw pressed (hook installed mid-press) passes through untouched.
//! * Escape is swallowed only while a take runs, and its release only if its press was.
//! * A press while the key is believed down after more than `repeat_gap_ms` of silence means a
//!   release was missed (hook dropped by the OS, session switch). It is treated as release +
//!   new press instead of an endless "repeat" that would wedge the gesture.
//!
//! Times are integer milliseconds on any monotonic clock the caller chooses; adapters pass the
//! *event's* timestamp where the OS has one (a late hook must not turn a double tap into two
//! single taps).

use ochre_core::platform::Gesture;

/// Gestures produced by one input (at most three: a recovered release, an expiry, a press).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Gestures {
    buf: [Option<Gesture>; 3],
}

impl Gestures {
    pub fn push(&mut self, g: Gesture) {
        if let Some(slot) = self.buf.iter_mut().find(|s| s.is_none()) {
            *slot = Some(g);
        }
    }
    pub fn iter(&self) -> impl Iterator<Item = Gesture> + '_ {
        self.buf.iter().map_while(|g| *g)
    }
    pub fn is_empty(&self) -> bool {
        self.buf[0].is_none()
    }
    pub fn to_vec(&self) -> Vec<Gesture> {
        self.iter().collect()
    }
}

/// What one input means.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Decision {
    /// Hide this event from the system.
    pub swallow: bool,
    pub gestures: Gestures,
    /// The Voice key press we swallowed turned out to be a modifier in a chord: the adapter
    /// must send a Voice key *down* to the OS before the current event.
    pub replay_down: bool,
}

impl Decision {
    fn swallow(swallow: bool) -> Self {
        Self {
            swallow,
            ..Self::default()
        }
    }
    fn with(mut self, g: Gesture) -> Self {
        self.gestures.push(g);
        self
    }
    fn merge(mut self, other: Decision) -> Self {
        self.swallow |= other.swallow;
        self.replay_down |= other.replay_down;
        for g in other.gestures.iter() {
            self.gestures.push(g);
        }
        self
    }
}

/// Whether an event the machine wants hidden may really be hidden ("safe swallow").
///
/// A press may always be hidden. A release may only be hidden when the system never saw the
/// press: if the press leaked through (Windows skips a hook that answers too slowly), hiding
/// the release would leave the key held down for every app until it is pressed again.
/// `system_down` is the OS's view of the key from inside the hook, i.e. before this event.
pub fn safe_swallow(swallow: bool, down: bool, system_down: bool) -> bool {
    swallow && (down || !system_down)
}

/// Map a 32-bit event tick (Windows `KBDLLHOOKSTRUCT.time`) onto the 64-bit tick clock.
/// Implausibly old stamps (over 60 s, or before the clock began) fall back to `now64`.
pub fn event_time(now64: u64, now32: u32, event32: u32) -> u64 {
    let age = u64::from(now32.wrapping_sub(event32));
    if age <= 60_000 && age <= now64 {
        now64 - age
    } else {
        now64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    /// Key down, gesture undecided (a hold if released late, a first tap if released early).
    Pressed,
    /// A short first tap ended; recording runs while we wait for a second press.
    Waiting,
    /// Double-tapped (or a take started elsewhere): the next press finishes.
    Locked,
}

#[derive(Debug, Clone)]
pub struct Machine {
    pub double_tap_ms: u64,
    pub hold_min_ms: u64,
    pub repeat_gap_ms: u64,
    phase: Phase,
    /// When the current phase started (press time for Pressed, release time for Waiting).
    at: u64,
    /// The Voice key's last seen state.
    key_down: bool,
    /// We swallowed the current press; its repeats and release are ours too.
    owned: bool,
    /// The current press went to the OS (ineligible, or a chord); so must its release.
    forwarding: bool,
    /// The release of the locking second tap is ignored.
    latch_release: bool,
    /// We swallowed Escape's press; swallow its repeats and release too.
    escape_owned: bool,
    /// The app says a take (or its processing) is running.
    recording: bool,
    /// The app's current take was started by this machine.
    from_key: bool,
    /// A take started elsewhere (UI, tray, wake word): one press of the key finishes it.
    external: bool,
    /// The raw modifier was held when the current take started.
    raw_at_start: bool,
    last_key_at: u64,
}

impl Machine {
    pub fn new(double_tap_ms: u64, hold_min_ms: u64) -> Self {
        Self {
            double_tap_ms,
            hold_min_ms,
            repeat_gap_ms: 1500,
            phase: Phase::Idle,
            at: 0,
            key_down: false,
            owned: false,
            forwarding: false,
            latch_release: false,
            escape_owned: false,
            recording: false,
            from_key: false,
            external: false,
            raw_at_start: false,
            last_key_at: 0,
        }
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn key_down(&self) -> bool {
        self.key_down
    }

    /// No gesture in progress and the key is not held (safe to change the key).
    pub fn idle(&self) -> bool {
        self.phase == Phase::Idle && !self.key_down
    }

    /// Whether this machine has a hand in the key's OS state right now (a swallowed press or a
    /// replayed one): stopping the listener must then make sure the key is not left down.
    pub fn involved(&self) -> bool {
        self.key_down && (self.owned || self.forwarding)
    }

    /// When [`Machine::tick`] next has something to do.
    pub fn next_deadline(&self) -> Option<u64> {
        (self.phase == Phase::Waiting).then(|| self.at + self.double_tap_ms + 1)
    }

    pub fn tick(&mut self, now: u64) -> Decision {
        if self.phase == Phase::Waiting && now.saturating_sub(self.at) > self.double_tap_ms {
            return self.expire();
        }
        Decision::default()
    }

    /// One Voice key transition. `eligible`: no conflicting modifier is held (this is
    /// dictation, not a shortcut). `raw`: the raw modifier is held right now.
    pub fn key(&mut self, down: bool, now: u64, eligible: bool, raw: bool) -> Decision {
        if down && self.key_down {
            if now.saturating_sub(self.last_key_at) <= self.repeat_gap_ms {
                self.last_key_at = now;
                return Decision::swallow(self.owned && !self.forwarding); // auto-repeat
            }
            // A press while "down" after a long silence: the release was missed. Recover.
            let release = self.key(false, now, eligible, raw);
            return release.merge(self.key(true, now, eligible, raw));
        }
        if !down && !self.key_down {
            return Decision::default(); // a release whose press we never saw: not ours
        }
        self.key_down = down;
        self.last_key_at = now;
        if self.forwarding {
            if !down {
                self.forwarding = false;
            }
            return Decision::default();
        }
        if down {
            self.press(now, eligible, raw)
        } else {
            self.release(now, raw)
        }
    }

    /// Any other non-modifier key except Escape. Modifiers must not be fed here: the raw
    /// modifier and AltGr's synthesized Ctrl would otherwise break gestures.
    pub fn other(&mut self, down: bool, _now: u64) -> Decision {
        if !down {
            return Decision::default();
        }
        if self.phase == Phase::Waiting {
            return self.expire(); // typing after a lone tap: it was just a tap
        }
        if !self.key_down || self.forwarding || !self.owned {
            return Decision::default();
        }
        // The Voice key is held and another key went down: it was a modifier in a chord.
        self.forwarding = true;
        self.owned = false;
        self.latch_release = false;
        let d = Decision {
            replay_down: true,
            ..Decision::default()
        };
        if self.phase == Phase::Pressed {
            self.phase = Phase::Idle;
            return d.with(Gesture::Abort);
        }
        d // Locked (the second tap became a chord: stay locked) or Idle after a finish
    }

    /// The paste-last key went down while the Voice key may be held (Voice key + Down).
    /// Returns `None` when the Voice key is not held by us (not ours: an ordinary key).
    pub fn paste_chord(&mut self) -> Option<Decision> {
        if !self.key_down || !self.owned || self.forwarding {
            return None;
        }
        let mut d = Decision::swallow(true);
        if matches!(self.phase, Phase::Pressed | Phase::Locked) {
            // The press started (or locked) a take a moment ago: throw it away, silently.
            d = d.with(Gesture::Abort);
        }
        self.phase = Phase::Idle;
        self.latch_release = false;
        self.external = false;
        self.raw_at_start = false;
        // `owned` stays: the Voice key's release is hidden with its press, and means nothing.
        Some(d.with(Gesture::PasteLast))
    }

    /// A standalone paste-last shortcut (`ctrl+alt+v`). Interrupts a take the Voice key is
    /// holding, ends a lone tap's wait, and pastes.
    pub fn paste_combo(&mut self) -> Decision {
        if let Some(d) = self.paste_chord() {
            return d;
        }
        let d = Decision::swallow(true);
        if self.phase == Phase::Waiting {
            return d.merge(self.expire()).with(Gesture::PasteLast);
        }
        d.with(Gesture::PasteLast)
    }

    /// Escape (the cancel key).
    pub fn escape(&mut self, down: bool) -> Decision {
        if !down {
            let owned = std::mem::take(&mut self.escape_owned);
            return Decision::swallow(owned);
        }
        if self.escape_owned {
            return Decision::swallow(true); // auto-repeat after we cancelled
        }
        if self.phase != Phase::Idle || self.recording {
            self.phase = Phase::Idle;
            self.latch_release = false;
            self.external = false;
            self.escape_owned = true;
            return Decision::swallow(true).with(Gesture::Cancel);
        }
        Decision::default() // nothing running: Escape belongs to the app
    }

    /// The app's view: true while a take runs or is being processed (so Escape cancels it).
    ///
    /// A take the app started itself (UI, tray, wake word) becomes *external*: one press of the
    /// key finishes it. This never changes the gesture phase: a late "not recording" for the
    /// previous take must not reset the take the user is holding now, and a first tap waiting
    /// for its second belongs to the key (resetting it made double taps miss).
    pub fn set_recording(&mut self, recording: bool) {
        self.recording = recording;
        if !recording {
            self.from_key = false;
            self.external = false;
        } else if self.phase == Phase::Idle && !self.from_key {
            self.external = true;
        }
    }

    /// Forget everything (listener stopped or restarted).
    pub fn reset(&mut self) {
        *self = Self {
            repeat_gap_ms: self.repeat_gap_ms,
            ..Self::new(self.double_tap_ms, self.hold_min_ms)
        };
    }

    fn finish(raw: bool, normal: Gesture) -> Gesture {
        if raw { Gesture::FinishRaw } else { normal }
    }

    fn press(&mut self, now: u64, eligible: bool, raw: bool) -> Decision {
        let mut prefix = Decision::default();
        if self.phase == Phase::Waiting {
            if now.saturating_sub(self.at) <= self.double_tap_ms && eligible {
                self.phase = Phase::Locked;
                self.latch_release = true;
                self.owned = true;
                self.raw_at_start |= raw;
                return Decision::swallow(true).with(Gesture::Lock);
            }
            prefix = self.expire(); // too late for a double tap: the first was a lone tap
        }
        if self.phase == Phase::Locked || (self.phase == Phase::Idle && self.external) {
            // The press that finishes a locked (or externally started) take: finish at once.
            let g = Self::finish(raw || self.raw_at_start, Gesture::Finish);
            self.phase = Phase::Idle;
            self.latch_release = false;
            self.external = false;
            self.raw_at_start = false;
            self.owned = true;
            return prefix.merge(Decision::swallow(true).with(g));
        }
        if !eligible {
            self.forwarding = true;
            self.owned = false;
            return prefix;
        }
        self.phase = Phase::Pressed;
        self.at = now;
        self.owned = true;
        self.from_key = true;
        self.external = false;
        self.raw_at_start = raw;
        prefix.merge(Decision::swallow(true).with(Gesture::Press))
    }

    fn release(&mut self, now: u64, raw: bool) -> Decision {
        let owned = std::mem::take(&mut self.owned);
        match self.phase {
            Phase::Pressed => {
                if now.saturating_sub(self.at) >= self.hold_min_ms {
                    self.phase = Phase::Idle;
                    let g = Self::finish(raw || self.raw_at_start, Gesture::Release);
                    self.raw_at_start = false;
                    return Decision::swallow(true).with(g);
                }
                self.phase = Phase::Waiting;
                self.at = now;
                Decision::swallow(true)
            }
            Phase::Locked if self.latch_release => {
                self.latch_release = false;
                Decision::swallow(true)
            }
            // Idle after a finish-on-press or a cancel: the release stays hidden with its press.
            _ => Decision::swallow(owned),
        }
    }

    fn expire(&mut self) -> Decision {
        self.phase = Phase::Idle;
        self.raw_at_start = false;
        Decision::default().with(Gesture::Abort)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Gesture::*;

    const DT: u64 = 350;
    const HOLD: u64 = 250;

    fn m() -> Machine {
        Machine::new(DT, HOLD)
    }
    fn down(m: &mut Machine, t: u64) -> Decision {
        m.key(true, t, true, false)
    }
    fn up(m: &mut Machine, t: u64) -> Decision {
        m.key(false, t, true, false)
    }
    fn g(d: Decision) -> Vec<Gesture> {
        d.gestures.to_vec()
    }

    #[test]
    fn hold_press_then_release() {
        let mut m = m();
        let d = down(&mut m, 1000);
        assert!(d.swallow);
        assert_eq!(g(d), vec![Press]);
        for t in (1030..1900).step_by(30) {
            let r = down(&mut m, t); // auto-repeat
            assert!(r.swallow);
            assert!(r.gestures.is_empty());
        }
        let d = up(&mut m, 1900);
        assert!(d.swallow);
        assert_eq!(g(d), vec![Release]);
        assert!(m.idle());
        assert_eq!(m.next_deadline(), None);
    }

    #[test]
    fn hold_exactly_at_threshold_is_a_hold() {
        let mut m = m();
        down(&mut m, 0);
        assert_eq!(g(up(&mut m, HOLD)), vec![Release]);
    }

    #[test]
    fn lone_tap_aborts_after_window() {
        let mut m = m();
        assert_eq!(g(down(&mut m, 1000)), vec![Press]);
        let d = up(&mut m, 1100);
        assert!(d.swallow);
        assert!(d.gestures.is_empty());
        assert_eq!(m.phase(), Phase::Waiting);
        assert_eq!(m.next_deadline(), Some(1100 + DT + 1));
        assert!(
            m.tick(1100 + DT).gestures.is_empty(),
            "window still open at its edge"
        );
        let d = m.tick(1100 + DT + 1);
        assert_eq!(g(d), vec![Abort]);
        assert!(!d.replay_down, "a lone tap is never replayed");
        assert!(m.idle());
    }

    #[test]
    fn double_tap_locks_and_single_tap_finishes_on_press() {
        let mut m = m();
        assert_eq!(g(down(&mut m, 0)), vec![Press]);
        up(&mut m, 80);
        let d = down(&mut m, 200);
        assert!(d.swallow);
        assert_eq!(g(d), vec![Lock]);
        let d = up(&mut m, 260); // release of the locking tap is ignored
        assert!(d.swallow);
        assert!(d.gestures.is_empty());
        assert_eq!(m.phase(), Phase::Locked);
        assert!(
            m.tick(100_000).gestures.is_empty(),
            "locked never times out here"
        );
        let d = down(&mut m, 5000);
        assert!(d.swallow);
        assert_eq!(g(d), vec![Finish]);
        let d = up(&mut m, 5080);
        assert!(d.swallow, "the finishing tap's release stays hidden");
        assert!(d.gestures.is_empty());
        assert!(m.idle());
    }

    #[test]
    fn locked_second_press_held_long_still_locks() {
        let mut m = m();
        down(&mut m, 0);
        up(&mut m, 50);
        assert_eq!(g(down(&mut m, 100)), vec![Lock]);
        for t in (130..3000).step_by(30) {
            assert!(down(&mut m, t).gestures.is_empty());
        }
        assert!(up(&mut m, 3000).gestures.is_empty());
        assert_eq!(g(down(&mut m, 4000)), vec![Finish]);
    }

    #[test]
    fn second_press_after_window_is_a_new_take() {
        let mut m = m();
        down(&mut m, 0);
        up(&mut m, 100);
        // The timer thread was late: the press itself sees that the window has closed.
        let d = down(&mut m, 100 + DT + 5);
        assert_eq!(g(d), vec![Abort, Press]);
        assert_eq!(m.phase(), Phase::Pressed);
    }

    #[test]
    fn long_first_press_is_a_hold_not_a_tap() {
        let mut m = m();
        down(&mut m, 0);
        assert_eq!(g(up(&mut m, 400)), vec![Release]);
        assert_eq!(
            g(down(&mut m, 500)),
            vec![Press],
            "quick press after a hold starts a new take"
        );
    }

    #[test]
    fn raw_modifier_at_finish_or_start() {
        let mut m = m();
        down(&mut m, 0);
        assert_eq!(g(m.key(false, 500, true, true)), vec![FinishRaw]);

        let mut m = Machine::new(DT, HOLD);
        m.key(true, 0, true, true);
        assert_eq!(
            g(up(&mut m, 500)),
            vec![FinishRaw],
            "raw latched at the start of the take"
        );

        let mut m = Machine::new(DT, HOLD);
        down(&mut m, 0);
        up(&mut m, 50);
        down(&mut m, 100);
        up(&mut m, 150);
        assert_eq!(g(m.key(true, 2000, true, true)), vec![FinishRaw]);

        let mut m = Machine::new(DT, HOLD);
        down(&mut m, 0);
        assert_eq!(
            g(up(&mut m, 500)),
            vec![Release],
            "no raw without the modifier"
        );
    }

    #[test]
    fn escape_cancels_hold_and_swallows_pair() {
        let mut m = m();
        down(&mut m, 0);
        let d = m.escape(true);
        assert!(d.swallow);
        assert_eq!(g(d), vec![Cancel]);
        assert!(m.escape(true).swallow, "escape auto-repeat");
        assert!(m.escape(false).swallow);
        let d = up(&mut m, 900);
        assert!(d.swallow, "voice key release after a cancel stays hidden");
        assert!(d.gestures.is_empty());
        assert!(m.idle());
    }

    #[test]
    fn escape_cancels_waiting_and_locked() {
        let mut m = m();
        down(&mut m, 0);
        up(&mut m, 50);
        assert_eq!(g(m.escape(true)), vec![Cancel]);
        m.escape(false);
        assert!(
            m.tick(10_000).gestures.is_empty(),
            "no abort after a cancel"
        );

        let mut m = Machine::new(DT, HOLD);
        down(&mut m, 0);
        up(&mut m, 50);
        down(&mut m, 100);
        up(&mut m, 150);
        assert_eq!(g(m.escape(true)), vec![Cancel]);
        m.escape(false);
        assert_eq!(
            g(down(&mut m, 3000)),
            vec![Press],
            "after cancel the key starts fresh"
        );
    }

    #[test]
    fn escape_passes_through_when_idle_and_cancels_when_recording() {
        let mut m = m();
        let d = m.escape(true);
        assert!(!d.swallow);
        assert!(d.gestures.is_empty());
        assert!(!m.escape(false).swallow);
        m.set_recording(true); // e.g. still processing a finished take
        assert_eq!(g(m.escape(true)), vec![Cancel]);
        assert!(m.escape(false).swallow);
    }

    #[test]
    fn other_key_after_lone_tap_aborts_immediately() {
        let mut m = m();
        down(&mut m, 0);
        up(&mut m, 50);
        let d = m.other(true, 120);
        assert!(!d.swallow);
        assert_eq!(g(d), vec![Abort]);
        assert!(m.idle());
        assert!(m.other(false, 130).gestures.is_empty());
    }

    #[test]
    fn chord_aborts_and_replays() {
        let mut m = m();
        down(&mut m, 0);
        let d = m.other(true, 40); // AltGr+e, Right Ctrl+C
        assert!(!d.swallow);
        assert!(d.replay_down);
        assert_eq!(g(d), vec![Abort]);
        assert!(m.involved());
        // Repeats and the release now belong to the OS.
        assert!(!down(&mut m, 70).swallow);
        let d = up(&mut m, 100);
        assert!(!d.swallow);
        assert!(d.gestures.is_empty());
        assert!(m.idle());
        // A second key in the same chord does not replay again.
        let mut m = Machine::new(DT, HOLD);
        down(&mut m, 0);
        m.other(true, 10);
        assert!(!m.other(true, 20).replay_down);
    }

    #[test]
    fn chord_during_long_hold_aborts() {
        let mut m = m();
        down(&mut m, 0);
        let d = m.other(true, 2000);
        assert_eq!(g(d), vec![Abort]);
        assert!(d.replay_down);
        assert!(up(&mut m, 2100).gestures.is_empty());
    }

    #[test]
    fn chord_while_holding_the_locking_tap_stays_locked() {
        let mut m = m();
        down(&mut m, 0);
        up(&mut m, 50);
        down(&mut m, 100); // Lock, still held
        let d = m.other(true, 200);
        assert!(d.replay_down);
        assert!(d.gestures.is_empty());
        assert!(!up(&mut m, 300).swallow);
        assert_eq!(g(down(&mut m, 2000)), vec![Finish]);
    }

    #[test]
    fn other_key_up_or_without_voice_key_is_ignored() {
        let mut m = m();
        assert_eq!(m.other(true, 0), Decision::default());
        assert_eq!(m.other(false, 0), Decision::default());
        down(&mut m, 10);
        up(&mut m, 500);
        assert_eq!(m.other(true, 600), Decision::default());
    }

    #[test]
    fn ineligible_press_passes_through_with_its_release() {
        let mut m = m();
        let d = m.key(true, 0, false, false);
        assert!(!d.swallow);
        assert!(d.gestures.is_empty());
        assert!(!m.key(true, 30, false, false).swallow, "repeats pass too");
        let d = up(&mut m, 100);
        assert!(!d.swallow);
        assert!(d.gestures.is_empty());
        assert!(m.idle());
    }

    #[test]
    fn ineligible_second_press_ends_the_wait() {
        let mut m = m();
        down(&mut m, 0);
        up(&mut m, 50);
        let d = m.key(true, 100, false, false);
        assert_eq!(g(d), vec![Abort]);
        assert!(!d.swallow);
    }

    #[test]
    fn release_without_press_is_not_ours() {
        let mut m = m();
        let d = up(&mut m, 0);
        assert_eq!(d, Decision::default());
    }

    #[test]
    fn missed_release_recovers_instead_of_wedging() {
        let mut m = m();
        down(&mut m, 0);
        // The release was lost (hook dropped). Two seconds later the key goes down again.
        let d = down(&mut m, 2000);
        assert_eq!(g(d), vec![Release, Press]);
        assert!(d.swallow);
        assert_eq!(g(up(&mut m, 2400)), vec![Release]);
    }

    #[test]
    fn missed_release_of_lone_tap_recovers() {
        let mut m = m();
        down(&mut m, 0);
        up(&mut m, 50);
        down(&mut m, 100); // Lock
        // Its release is lost; the user taps again much later to finish.
        assert_eq!(g(down(&mut m, 9000)), vec![Finish]);
    }

    #[test]
    fn external_take_finishes_on_one_press() {
        let mut m = m();
        m.set_recording(true); // tray / UI / wake word started a take
        let d = down(&mut m, 0);
        assert!(d.swallow);
        assert_eq!(g(d), vec![Finish]);
        assert!(up(&mut m, 60).swallow);
        m.set_recording(false);
        assert_eq!(g(down(&mut m, 1000)), vec![Press]);
    }

    #[test]
    fn late_recording_flag_does_not_turn_a_key_take_external() {
        let mut m = m();
        down(&mut m, 0);
        up(&mut m, 500); // Release
        m.set_recording(true); // the orchestrator's late "recording" for that take
        assert_eq!(
            g(down(&mut m, 700)),
            vec![Press],
            "a new press is a new take, not a finish"
        );
    }

    #[test]
    fn recording_false_never_resets_a_waiting_tap() {
        let mut m = m();
        down(&mut m, 0);
        up(&mut m, 50);
        m.set_recording(false);
        assert_eq!(g(down(&mut m, 200)), vec![Lock]);
    }

    #[test]
    fn paste_chord_aborts_a_fresh_take_and_leaves_release_inert() {
        let mut m = m();
        assert!(m.paste_chord().is_none(), "Voice key not held");
        down(&mut m, 0);
        let d = m.paste_chord().unwrap();
        assert!(d.swallow);
        assert_eq!(g(d), vec![Abort, PasteLast]);
        let d = up(&mut m, 60);
        assert!(d.swallow);
        assert!(d.gestures.is_empty());
        assert!(m.idle());
        assert_eq!(m.next_deadline(), None, "no double-tap window");
        // Holding the locking second tap: the locked take is dropped too.
        let mut m = Machine::new(DT, HOLD);
        down(&mut m, 0);
        up(&mut m, 50);
        down(&mut m, 100); // Lock
        assert_eq!(g(m.paste_chord().unwrap()), vec![Abort, PasteLast]);
        assert!(up(&mut m, 200).gestures.is_empty());
        assert!(m.idle());
        // A chord press (forwarded) is not ours.
        let mut m = Machine::new(DT, HOLD);
        down(&mut m, 0);
        m.other(true, 10);
        assert!(m.paste_chord().is_none());
    }

    #[test]
    fn paste_combo_ends_a_lone_tap_wait() {
        let mut m = m();
        assert_eq!(g(m.paste_combo()), vec![PasteLast]);
        down(&mut m, 0);
        up(&mut m, 50);
        assert_eq!(g(m.paste_combo()), vec![Abort, PasteLast]);
        assert!(m.idle());
    }

    #[test]
    fn involved_and_reset() {
        let mut m = m();
        assert!(!m.involved());
        down(&mut m, 0);
        assert!(m.involved());
        m.reset();
        assert!(m.idle());
        assert!(!m.involved());
        assert_eq!(m.double_tap_ms, DT);
    }

    #[test]
    fn helpers() {
        assert!(safe_swallow(true, true, true));
        assert!(safe_swallow(true, false, false));
        assert!(!safe_swallow(true, false, true));
        assert!(!safe_swallow(false, true, false));
        assert_eq!(event_time(100_000, 5000, 4990), 99_990);
        assert_eq!(event_time(100_000, 5, u32::MAX - 4), 99_990, "32-bit wrap");
        assert_eq!(event_time(100_000, 5000, 5010), 100_000, "future stamp");
        assert_eq!(event_time(100_000, 100_000, 0), 100_000, "older than 60 s");
        assert_eq!(event_time(5, 5000, 4990), 5, "before the clock began");
    }

    /// Every random sequence keeps the swallow bookkeeping consistent: an owned press is
    /// always matched by a hidden release and a forwarded press by a visible one, and the
    /// machine never stays non-idle once the key is up and time has passed.
    #[test]
    fn fuzz_never_wedges_and_pairs_swallows() {
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..2000 {
            let mut m = Machine::new(DT, HOLD);
            let mut t = 0u64;
            let mut physical_down = false;
            let mut press_swallowed = false;
            for _ in 0..40 {
                t += rnd() % 400;
                match rnd() % 6 {
                    0 | 1 => {
                        let eligible = rnd() % 5 != 0;
                        let d = m.key(!physical_down, t, eligible, rnd() % 4 == 0);
                        if physical_down {
                            assert_eq!(d.swallow, press_swallowed, "release must match its press");
                        } else {
                            press_swallowed = d.swallow;
                        }
                        physical_down = !physical_down;
                    }
                    2 if rnd() % 3 == 0 => {
                        if rnd() % 2 == 0 {
                            m.paste_chord();
                        } else {
                            m.paste_combo();
                        }
                    }
                    2 => {
                        let d = m.other(true, t);
                        if d.replay_down {
                            press_swallowed = false; // replayed: the OS now owns the key
                        }
                    }
                    3 => {
                        m.escape(true);
                        m.escape(false);
                    }
                    4 => m.set_recording(rnd() % 2 == 0),
                    _ => {
                        m.tick(t);
                    }
                }
            }
            if physical_down {
                t += 10;
                assert_eq!(m.key(false, t, true, false).swallow, press_swallowed);
            }
            m.set_recording(false);
            m.tick(t + 10_000);
            if m.phase() == Phase::Locked {
                down(&mut m, t + 20_000);
                up(&mut m, t + 20_050);
            }
            assert!(m.idle(), "machine wedged in {:?}", m.phase());
        }
    }
}
