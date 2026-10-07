//! Hands-free session controller: wake word -> dictation -> "transcribe stop / send / cancel".
//! Port of `src/openwhisprflow/wake/handsfree.py` (`docs/wakeword.md` §7, SPEC §6.3).
//!
//! The controller only decides; it never records, transcribes or types:
//!
//! * [`HandsFree::feed`] takes every mic block (16 kHz mono, [-1, 1] floats, any size) while
//!   hands-free is on. While listening, the wake detector runs on it (VAD-gated inside the detector)
//!   and a pre-roll ring keeps the last `preroll_s` seconds.
//! * On a wake hit it emits [`HfEvent::Wake`] (and calls `on_wake`) with a [`WakeStart`]: the
//!   orchestrator plays the earcon, seeds the dictation capture with `WakeStart::preroll` and from
//!   then on passes each finished phrase transcript to [`HandsFree::on_phrase`].
//! * `on_phrase` checks the first phrase for the wake word (a false wake ends the session
//!   silently), strips it, and looks for a trailing control phrase ([`crate::commands`]). It
//!   returns a [`PhraseAction`]. Every session end (command, `idle_timeout_s`, the session cap,
//!   [`HandsFree::finish`]/[`HandsFree::cancel`]) is also emitted once as [`HfEvent::End`].
//! * Wake hits during a session arm control detection for `arm_s` (default 5 s): the audio said
//!   the wake word, so rougher ASR ("transcript stop", or a bare "Stop.") still counts.
//! * While another app holds the mic (a call; [`crate::calls`]), listening pauses.
//!
//! Callbacks run synchronously inside the call that caused them, so they must not call back into
//! the controller (it is `&mut self`; wrapping it in a Mutex would deadlock). Queue work instead.
//! A panicking callback is caught and logged, so it never kills the audio thread.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::time::Instant;

use ochre_core::Result;
use ochre_core::config::HandsFreeConfig;

use crate::FRAME;
use crate::FRAME_S;
use crate::calls::MicMonitor;
use crate::commands::{
    Control, ParseOptions, ends_with_phrase, parse_control, starts_with_phrase,
    strip_leading_phrase,
};
use crate::detector::{SpeechFn, WakeDetector, WakeHit};

/// What the controller needs from a wake detector; [`WakeDetector`] implements it, tests use fakes.
pub trait WakeSource: Send {
    /// One 1280-sample frame of [-1, 1] floats.
    fn process_frame(&mut self, frame: &[f32]) -> Option<WakeHit>;
    /// VAD decision for the last frame (true without a VAD).
    fn last_voiced(&self) -> bool;
    fn has_vad(&self) -> bool;
    fn reset(&mut self);
    fn hold(&mut self);
}

impl WakeSource for WakeDetector {
    fn process_frame(&mut self, frame: &[f32]) -> Option<WakeHit> {
        WakeDetector::process_frame(self, frame)
    }
    fn last_voiced(&self) -> bool {
        self.last_voiced
    }
    fn has_vad(&self) -> bool {
        WakeDetector::has_vad(self)
    }
    fn reset(&mut self) {
        WakeDetector::reset(self)
    }
    fn hold(&mut self) {
        WakeDetector::hold(self)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfState {
    Off,
    /// Wake word armed.
    Listening,
    /// Another app is using the mic (a call).
    Paused,
    /// Dictating.
    Session,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// `"<phrase> stop"` / `"<phrase> done"`: insert.
    Stop,
    /// `"<phrase> send"`, or `finish(send=true)`: insert, then press Enter.
    Send,
    /// `"<phrase> cancel"`, `cancel()`, or hands-free disabled: discard.
    Cancel,
    /// `idle_timeout_s` without speech: insert what was said.
    IdleTimeout,
    /// The session cap: insert.
    MaxDuration,
    /// The first phrase did not start with the wake word: drop silently.
    FalseWake,
    /// `finish()` from the UI / hotkey: insert.
    External,
}

/// Everything the orchestrator needs to start capturing.
#[derive(Debug, Clone, PartialEq)]
pub struct WakeStart {
    pub session_id: u64,
    /// 16 kHz mono [-1, 1], from the start of the wake word's speech run (capped to the ring).
    pub preroll: Vec<f32>,
    pub score: f32,
    /// The pre-roll starts in silence (the VAD found the gap before the word).
    pub clean_start: bool,
    /// Controller clock at the hit, seconds.
    pub at: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SessionEnd {
    pub session_id: u64,
    pub reason: EndReason,
    /// Phrases joined with single spaces (control phrases and the wake word stripped).
    pub text: String,
    pub phrases: Vec<String>,
    pub duration_s: f64,
    pub wake_score: f32,
    /// The false-wake transcript, the command span, ...
    pub detail: String,
}

impl SessionEnd {
    /// Insert `text`? (not for Cancel / FalseWake, and only if there is text).
    pub fn insert(&self) -> bool {
        !matches!(self.reason, EndReason::Cancel | EndReason::FalseWake)
            && !self.text.trim().is_empty()
    }

    /// Press Enter after inserting?
    pub fn press_enter(&self) -> bool {
        self.reason == EndReason::Send && self.insert()
    }
}

/// What [`HandsFree::on_phrase`] decided for one transcript.
#[derive(Debug, Clone, PartialEq)]
pub enum PhraseAction {
    /// No session (a late phrase after the end, or a stale session id).
    Ignored,
    /// Ordinary dictation: `kept` was appended ("" if nothing, e.g. an empty transcript).
    Continue { kept: String },
    /// "scratch that": `dropped` is the scratched text (a stored phrase, or the words said just
    /// before the command in the same phrase), `None` if there was nothing to drop.
    Scratch { dropped: Option<String> },
    /// Session over, insert `end.text`.
    Finish(SessionEnd),
    /// Session over, insert `end.text` and press Enter.
    Send(SessionEnd),
    /// Session discarded.
    Cancel(SessionEnd),
    /// The first phrase did not start with the wake word; nothing is inserted.
    FalseWake(SessionEnd),
}

/// Everything the controller reports; also delivered to the matching callback.
#[derive(Debug, Clone, PartialEq)]
pub enum HfEvent {
    Wake(WakeStart),
    End(SessionEnd),
    /// Listening paused; the apps holding the mic.
    Paused(Vec<String>),
    Resumed,
    State(HfState),
}

type Cb<T> = Option<Box<dyn FnMut(&T) + Send>>;

struct Session {
    id: u64,
    started: f64,
    score: f32,
    clean_start: bool,
    phrases: Vec<String>,
    verified: bool,
    last_activity: f64,
    armed_until: f64,
    /// (phrase index, text without the trailing wake word)
    dangling: Option<(usize, String)>,
}

pub struct HandsFree {
    phrase: String,
    aliases: Vec<String>,
    idle_timeout_s: f64,
    pause_on_calls: bool,
    max_session_s: f64,
    verify: bool,
    arm_s: f64,
    post_session_cooldown_s: f64,
    call_poll_s: f64,
    detector: Box<dyn WakeSource>,
    mic_monitor: Option<Box<dyn MicMonitor>>,
    clock: Box<dyn Fn() -> f64 + Send>,
    pending: Vec<f32>,
    idx: i64,
    preroll_frames: i64,
    ring: VecDeque<(i64, Vec<f32>, bool)>,
    ring_cap: usize,
    session: Option<Session>,
    next_id: u64,
    quiet_until: f64,
    next_call_poll: f64,
    paused_apps: Vec<String>,
    state: HfState,
    out: Vec<HfEvent>,
    on_wake: Cb<WakeStart>,
    on_end: Cb<SessionEnd>,
    on_pause: Cb<(bool, Vec<String>)>,
    on_state: Cb<HfState>,
}

fn join(phrases: &[String]) -> String {
    phrases
        .iter()
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

impl HandsFree {
    /// From `Config.handsfree` (phrase, preroll_s, idle_timeout_s, pause_on_calls, enabled) and a
    /// detector. Starts `Listening` if `cfg.enabled`, else `Off` (see [`set_enabled`](Self::set_enabled)).
    /// Defaults: `max_session_s` 600, verify on, `arm_s` 5, post-session cooldown 1.5 s, call poll
    /// every 3 s, no mic monitor.
    pub fn new(cfg: &HandsFreeConfig, detector: Box<dyn WakeSource>) -> Self {
        let preroll_frames = ((cfg.preroll_s as f64 / FRAME_S as f64).ceil() as i64).max(1);
        // extra room: the pre-roll may reach ~1.2 s before the word's end when the gate fired late
        let ring_cap = preroll_frames as usize + 25;
        let t0 = Instant::now();
        HandsFree {
            phrase: cfg.phrase.clone(),
            aliases: Vec::new(),
            idle_timeout_s: cfg.idle_timeout_s as f64,
            pause_on_calls: cfg.pause_on_calls,
            max_session_s: 600.0,
            verify: true,
            arm_s: 5.0,
            post_session_cooldown_s: 1.5,
            call_poll_s: 3.0,
            detector,
            mic_monitor: None,
            clock: Box::new(move || t0.elapsed().as_secs_f64()),
            pending: Vec::new(),
            idx: 0,
            preroll_frames,
            ring: VecDeque::with_capacity(ring_cap),
            ring_cap,
            session: None,
            next_id: 1,
            quiet_until: 0.0,
            next_call_poll: 0.0,
            paused_apps: Vec::new(),
            state: if cfg.enabled {
                HfState::Listening
            } else {
                HfState::Off
            },
            out: Vec::new(),
            on_wake: None,
            on_end: None,
            on_pause: None,
            on_state: None,
        }
    }

    /// The real thing: resolve the model (`cfg.model`, else `<bundled_dir>/<phrase>.onnx`), load the
    /// detector at `cfg.threshold` with the VAD, and on Windows the call monitor if
    /// `cfg.pause_on_calls`. `max_session_s` is `audio.max_session_s`.
    pub fn from_config(
        cfg: &HandsFreeConfig,
        max_session_s: f64,
        bundled_dir: &Path,
        vad: Option<SpeechFn>,
    ) -> Result<Self> {
        let path = crate::detector::resolve_model(&cfg.phrase, &cfg.model, bundled_dir)?;
        let mut det = WakeDetector::new(&path, cfg.threshold)?;
        det.set_vad(vad);
        let mut hf = HandsFree::new(cfg, Box::new(det)).with_max_session_s(max_session_s);
        if cfg.pause_on_calls {
            hf = hf.with_mic_monitor(Box::new(crate::calls::MicUsageMonitor::default()));
        }
        Ok(hf)
    }

    pub fn with_max_session_s(mut self, s: f64) -> Self {
        self.max_session_s = s;
        self
    }
    /// Extra spellings that count as the phrase.
    pub fn with_aliases(mut self, aliases: Vec<String>) -> Self {
        self.aliases = aliases;
        self
    }
    /// Off: the first phrase need not start with the wake word (no false-wake check).
    pub fn with_verify(mut self, verify: bool) -> Self {
        self.verify = verify;
        self
    }
    pub fn with_arm_s(mut self, s: f64) -> Self {
        self.arm_s = s;
        self
    }
    pub fn with_post_session_cooldown_s(mut self, s: f64) -> Self {
        self.post_session_cooldown_s = s;
        self
    }
    pub fn with_call_poll_s(mut self, s: f64) -> Self {
        self.call_poll_s = s;
        self
    }
    pub fn with_mic_monitor(mut self, m: Box<dyn MicMonitor>) -> Self {
        self.mic_monitor = Some(m);
        self
    }
    /// Monotonic seconds (tests inject a fake clock).
    pub fn with_clock(mut self, clock: impl Fn() -> f64 + Send + 'static) -> Self {
        self.clock = Box::new(clock);
        self
    }
    pub fn on_wake(mut self, f: impl FnMut(&WakeStart) + Send + 'static) -> Self {
        self.on_wake = Some(Box::new(f));
        self
    }
    pub fn on_end(mut self, f: impl FnMut(&SessionEnd) + Send + 'static) -> Self {
        self.on_end = Some(Box::new(f));
        self
    }
    /// `(paused, apps)`.
    pub fn on_pause(mut self, f: impl FnMut(&(bool, Vec<String>)) + Send + 'static) -> Self {
        self.on_pause = Some(Box::new(f));
        self
    }
    pub fn on_state(mut self, f: impl FnMut(&HfState) + Send + 'static) -> Self {
        self.on_state = Some(Box::new(f));
        self
    }

    pub fn state(&self) -> HfState {
        self.state
    }
    pub fn phrase(&self) -> &str {
        &self.phrase
    }
    pub fn in_session(&self) -> bool {
        self.session.is_some()
    }
    pub fn session_id(&self) -> Option<u64> {
        self.session.as_ref().map(|s| s.id)
    }
    /// The text so far (stored phrases joined).
    pub fn session_text(&self) -> String {
        self.session
            .as_ref()
            .map(|s| join(&s.phrases))
            .unwrap_or_default()
    }
    /// Apps holding the mic while `Paused`.
    pub fn paused_apps(&self) -> &[String] {
        &self.paused_apps
    }
    /// Live config changes that don't need a new detector.
    pub fn update_config(&mut self, cfg: &HandsFreeConfig) {
        self.phrase = cfg.phrase.clone();
        self.idle_timeout_s = cfg.idle_timeout_s as f64;
        self.pause_on_calls = cfg.pause_on_calls;
        self.preroll_frames = ((cfg.preroll_s as f64 / FRAME_S as f64).ceil() as i64).max(1);
        self.ring_cap = self.preroll_frames as usize + 25;
        while self.ring.len() > self.ring_cap {
            self.ring.pop_front();
        }
    }

    /// Turn listening on/off. Turning it off during a session cancels the session.
    pub fn set_enabled(&mut self, enabled: bool) -> Vec<HfEvent> {
        if !enabled {
            if self.session.is_some() {
                self.end(EndReason::Cancel, "hands-free disabled".into());
            }
            self.set_state(HfState::Off);
        } else if self.state == HfState::Off {
            self.detector.reset();
            self.ring.clear();
            self.set_state(HfState::Listening);
        }
        self.flush()
    }

    /// Forget the audio heard so far and ignore wake hits for the post-session cooldown: call it
    /// when listening resumes after a pause in feeding (e.g. a hotkey dictation, whose words must
    /// not wake hands-free). No effect on a running session.
    pub fn hold(&mut self) {
        if self.session.is_some() {
            return;
        }
        self.detector.reset();
        self.detector.hold();
        self.ring.clear();
        self.pending.clear();
        self.quiet_until = (self.clock)() + self.post_session_cooldown_s;
    }

    /// Samples fed but not yet framed (< one 80 ms frame): the pre-roll of a wake ends this many
    /// samples before the end of the audio fed so far.
    pub fn pending_samples(&self) -> usize {
        self.pending.len()
    }

    /// One mic block (any length, [-1, 1]). Runs the detector, the pre-roll ring and the timers.
    pub fn feed(&mut self, block: &[f32]) -> Vec<HfEvent> {
        if self.state == HfState::Off {
            return Vec::new();
        }
        self.pending.extend_from_slice(block);
        let n = self.pending.len() / FRAME;
        for k in 0..n {
            let frame = self.pending[k * FRAME..(k + 1) * FRAME].to_vec();
            self.frame(frame);
        }
        self.pending.drain(..n * FRAME);
        self.timers();
        self.flush()
    }

    /// Check timers without audio (e.g. from a UI timer).
    pub fn tick(&mut self) -> Vec<HfEvent> {
        self.timers();
        self.flush()
    }

    /// A finished phrase transcript of the current session (`session_id`: guards against a late
    /// phrase of an earlier session; `None` = current).
    pub fn on_phrase(&mut self, text: &str) -> PhraseAction {
        self.on_phrase_for(text, None)
    }

    pub fn on_phrase_for(&mut self, text: &str, session_id: Option<u64>) -> PhraseAction {
        let r = self.phrase_inner(text, session_id);
        self.flush();
        r
    }

    /// End the session from outside (UI button, hotkey) and insert what was said.
    pub fn finish(&mut self, send: bool) -> Option<SessionEnd> {
        self.external(if send {
            EndReason::Send
        } else {
            EndReason::External
        })
    }

    /// Discard the session (Escape / UI x).
    pub fn cancel(&mut self) -> Option<SessionEnd> {
        self.external(EndReason::Cancel)
    }

    // ------------------------------------------------------------------ audio

    fn frame(&mut self, frame: Vec<f32>) {
        self.idx += 1;
        let now = (self.clock)();
        if self.state == HfState::Paused {
            return; // no inference, no pre-roll while a call holds the mic
        }
        let hit = self.detector.process_frame(&frame);
        let voiced = self.detector.last_voiced();
        if self.ring.len() == self.ring_cap {
            self.ring.pop_front();
        }
        self.ring.push_back((self.idx, frame, voiced));
        let has_vad = self.detector.has_vad();
        if let Some(s) = self.session.as_mut() {
            if voiced && has_vad {
                s.last_activity = now;
            }
            if let Some(h) = &hit {
                s.armed_until = now + self.arm_s;
                tracing::debug!(
                    "wake word during session (score {:.2}): control phrases armed",
                    h.score
                );
            }
            return;
        }
        let Some(hit) = hit else { return };
        if now < self.quiet_until {
            return;
        }
        let (start, clean) = self.preroll_start(hit.rise_frame as i64);
        let preroll: Vec<f32> = self
            .ring
            .iter()
            .filter(|(i, _, _)| *i >= start)
            .flat_map(|(_, f, _)| f.iter().copied())
            .collect();
        let id = self.next_id;
        self.next_id += 1;
        self.session = Some(Session {
            id,
            started: now,
            score: hit.score,
            clean_start: clean,
            phrases: Vec::new(),
            verified: false,
            last_activity: now,
            armed_until: 0.0,
            dangling: None,
        });
        tracing::info!(
            "wake word (score {:.2}): session {id}, pre-roll {:.2} s{}",
            hit.score,
            preroll.len() as f64 / crate::SAMPLE_RATE as f64,
            if clean { "" } else { " (starts mid-speech)" }
        );
        self.set_state(HfState::Session);
        self.out.push(HfEvent::Wake(WakeStart {
            session_id: id,
            preroll,
            score: hit.score,
            clean_start: clean,
            at: now,
        }));
    }

    /// First frame of the wake word's speech run, capped to the pre-roll (a backward scan).
    fn preroll_start(&self, anchor: i64) -> (i64, bool) {
        let idx = self.idx;
        let mut floor = idx - self.preroll_frames + 1;
        if !self.detector.has_vad() {
            return (floor, false);
        }
        let anchor = anchor.min(idx);
        let word_end = self
            .ring
            .iter()
            .rev()
            .find(|(i, _, v)| *i <= anchor && *v)
            .map(|x| x.0)
            .unwrap_or(anchor);
        floor = floor.min(word_end - (1.2 / FRAME_S as f64).ceil() as i64);
        let (mut quiet, mut heard) = (0i64, false);
        for (i, _, v) in self.ring.iter().rev() {
            if *i < floor {
                break;
            }
            if !*v {
                quiet += 1;
                if quiet >= 5 && heard {
                    return (floor.max(i + quiet - 2), true); // keep 2 quiet frames of lead-in
                }
            } else {
                quiet = 0;
                if *i <= anchor {
                    heard = true;
                }
            }
        }
        (
            floor.max(self.ring.front().map(|x| x.0).unwrap_or(floor)),
            false,
        )
    }

    // ------------------------------------------------------------------ timers / calls

    fn timers(&mut self) {
        let now = (self.clock)();
        if let Some(s) = &self.session {
            if now - s.started >= self.max_session_s {
                self.end(
                    EndReason::MaxDuration,
                    format!("{:.0} s cap", self.max_session_s),
                );
            } else if now - s.last_activity >= self.idle_timeout_s {
                self.end(
                    EndReason::IdleTimeout,
                    format!("{} s without speech", self.idle_timeout_s),
                );
            }
            return;
        }
        self.poll_calls(now);
    }

    fn poll_calls(&mut self, now: f64) {
        if !self.pause_on_calls || now < self.next_call_poll {
            return;
        }
        let Some(mon) = self.mic_monitor.as_mut() else {
            return;
        };
        self.next_call_poll = now + self.call_poll_s;
        let apps = mon.in_use();
        if !apps.is_empty() && self.state == HfState::Listening {
            tracing::info!(
                "pausing hands-free: microphone in use by {}",
                apps.join(", ")
            );
            self.paused_apps = apps.clone();
            self.detector.reset();
            self.ring.clear();
            self.set_state(HfState::Paused);
            self.out.push(HfEvent::Paused(apps));
        } else if apps.is_empty() && self.state == HfState::Paused {
            tracing::info!("resuming hands-free: microphone free");
            self.paused_apps.clear();
            self.detector.reset();
            self.set_state(HfState::Listening);
            self.out.push(HfEvent::Resumed);
        }
    }

    // ------------------------------------------------------------------ phrases

    fn phrase_inner(&mut self, text: &str, session_id: Option<u64>) -> PhraseAction {
        let now = (self.clock)();
        let Some(s) = self.session.as_mut() else {
            return PhraseAction::Ignored;
        };
        if session_id.is_some_and(|id| id != s.id) {
            return PhraseAction::Ignored;
        }
        s.last_activity = now;
        let raw = text.trim();
        if raw.is_empty() {
            return PhraseAction::Continue {
                kept: String::new(),
            };
        }
        let armed = now < s.armed_until;
        let frags = if s.clean_start { 0 } else { 1 };
        let first = !s.verified;
        if first {
            if self.verify && !starts_with_phrase(raw, &self.phrase, &self.aliases, frags) {
                tracing::info!("false wake (score {:.2}), transcript: {raw:?}", s.score);
                let end = self.end(EndReason::FalseWake, raw.to_string());
                return PhraseAction::FalseWake(end);
            }
            s.verified = true;
        }

        // "... transcribe." | "Stop.": the segmenter cut between the wake word and the command
        if let Some((d_idx, d_text)) = s.dangling.take() {
            let opts = ParseOptions {
                aliases: &self.aliases,
                armed: true,
                min_ratio: None,
            };
            let bare = parse_control(raw, &self.phrase, &opts);
            if let Some(c) = bare.control
                && bare.text.is_empty()
            {
                if d_idx < s.phrases.len() {
                    if d_text.is_empty() {
                        s.phrases.remove(d_idx);
                    } else {
                        s.phrases[d_idx] = d_text;
                    }
                }
                return self.apply(Some(c), String::new(), bare.matched);
            }
        }

        let opts = ParseOptions {
            aliases: &self.aliases,
            armed,
            min_ratio: None,
        };
        let ctl = parse_control(raw, &self.phrase, &opts);
        let mut body = if ctl.control.is_some() {
            ctl.text
        } else {
            raw.to_string()
        };
        if first {
            body = strip_leading_phrase(&body, &self.phrase, &self.aliases, frags);
        }
        self.apply(ctl.control, body, ctl.matched)
    }

    fn apply(&mut self, action: Option<Control>, body: String, matched: String) -> PhraseAction {
        let s = self.session.as_mut().expect("apply needs a session");
        match action {
            Some(Control::Scratch) => {
                let dropped = if body.is_empty() {
                    s.phrases.pop()
                } else {
                    Some(body)
                };
                if let Some(d) = &dropped {
                    tracing::info!("scratch that: dropped {d:?}");
                }
                PhraseAction::Scratch { dropped }
            }
            Some(Control::Cancel) => PhraseAction::Cancel(self.end(EndReason::Cancel, matched)),
            Some(c @ (Control::Finish | Control::Send)) => {
                if !body.is_empty() {
                    s.phrases.push(body);
                }
                if c == Control::Send {
                    PhraseAction::Send(self.end(EndReason::Send, matched))
                } else {
                    PhraseAction::Finish(self.end(EndReason::Stop, matched))
                }
            }
            None => {
                if !body.is_empty() {
                    s.phrases.push(body.clone());
                    if let Some(rest) = ends_with_phrase(&body, &self.phrase, &self.aliases) {
                        s.dangling = Some((s.phrases.len() - 1, rest));
                    }
                }
                PhraseAction::Continue { kept: body }
            }
        }
    }

    // ------------------------------------------------------------------ ending

    fn external(&mut self, reason: EndReason) -> Option<SessionEnd> {
        self.session.as_ref()?;
        let end = self.end(reason, "external".into());
        self.flush();
        Some(end)
    }

    fn end(&mut self, reason: EndReason, detail: String) -> SessionEnd {
        let s = self.session.take().expect("end needs a session");
        let now = (self.clock)();
        let end = SessionEnd {
            session_id: s.id,
            reason,
            text: join(&s.phrases),
            phrases: s.phrases,
            duration_s: now - s.started,
            wake_score: s.score,
            detail,
        };
        tracing::info!(
            "session {} ended: {reason:?} ({} phrases, {:.1} s)",
            end.session_id,
            end.phrases.len(),
            end.duration_s
        );
        // don't let the session's own last words ("transcribe stop") wake it again
        self.detector.reset();
        self.detector.hold();
        self.quiet_until = now + self.post_session_cooldown_s;
        self.set_state(if self.state == HfState::Off {
            HfState::Off
        } else {
            HfState::Listening
        });
        self.out.push(HfEvent::End(end.clone()));
        end
    }

    fn set_state(&mut self, state: HfState) {
        if state != self.state {
            self.state = state;
            self.out.push(HfEvent::State(state));
        }
    }

    /// Deliver queued events to the callbacks; return them.
    fn flush(&mut self) -> Vec<HfEvent> {
        let events = std::mem::take(&mut self.out);
        for e in &events {
            let r = catch_unwind(AssertUnwindSafe(|| match e {
                HfEvent::Wake(w) => self.on_wake.as_mut().map(|f| f(w)),
                HfEvent::End(x) => self.on_end.as_mut().map(|f| f(x)),
                HfEvent::Paused(apps) => self.on_pause.as_mut().map(|f| f(&(true, apps.clone()))),
                HfEvent::Resumed => self.on_pause.as_mut().map(|f| f(&(false, Vec::new()))),
                HfEvent::State(s) => self.on_state.as_mut().map(|f| f(s)),
            }));
            if r.is_err() {
                tracing::error!("hands-free callback panicked on {e:?}");
            }
        }
        events
    }
}

#[cfg(test)]
mod tests;
