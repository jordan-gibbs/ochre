//! Port of `tests/test_handsfree.py`: the state machine driven by fake audio, hits and transcripts.

use std::sync::{Arc, Mutex};

use super::*;
use crate::calls::MicMonitor;

const LOUD_V: f32 = 3000.0 / 32767.0;

#[derive(Default)]
struct FakeState {
    vad: bool,
    idx: u64,
    fire_at: Vec<u64>,
    last_voiced: bool,
    processed: u64,
    resets: u64,
    holds: u64,
}

/// Voiced = loud frame (when it has a VAD); fires on scripted frame numbers.
struct FakeDetector(Arc<Mutex<FakeState>>);

impl WakeSource for FakeDetector {
    fn process_frame(&mut self, frame: &[f32]) -> Option<WakeHit> {
        let mut s = self.0.lock().unwrap();
        s.idx += 1;
        s.processed += 1;
        let mean = frame.iter().map(|v| v.abs()).sum::<f32>() / frame.len() as f32;
        s.last_voiced = if s.vad { mean > 100.0 / 32767.0 } else { true };
        let idx = s.idx;
        s.fire_at.contains(&idx).then(|| WakeHit {
            frame: idx,
            rise_frame: idx - 3,
            score: 0.9,
            model: "fake".into(),
        })
    }
    fn last_voiced(&self) -> bool {
        self.0.lock().unwrap().last_voiced
    }
    fn has_vad(&self) -> bool {
        self.0.lock().unwrap().vad
    }
    fn reset(&mut self) {
        self.0.lock().unwrap().resets += 1;
    }
    fn hold(&mut self) {
        self.0.lock().unwrap().holds += 1;
    }
}

struct FakeMonitor(Arc<Mutex<Vec<String>>>);

impl MicMonitor for FakeMonitor {
    fn in_use(&mut self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
}

#[derive(Default)]
struct Seen {
    wakes: Vec<WakeStart>,
    ends: Vec<SessionEnd>,
    pauses: Vec<(bool, Vec<String>)>,
    states: Vec<HfState>,
}

struct Rig {
    t: Arc<Mutex<f64>>,
    det: Arc<Mutex<FakeState>>,
    seen: Arc<Mutex<Seen>>,
    hf: HandsFree,
}

struct Opts {
    vad: bool,
    monitor: Option<Arc<Mutex<Vec<String>>>>,
    tweak: fn(HandsFree) -> HandsFree,
    cfg: fn(&mut HandsFreeConfig),
}

impl Default for Opts {
    fn default() -> Self {
        Opts {
            vad: true,
            monitor: None,
            tweak: |h| h,
            cfg: |_| {},
        }
    }
}

impl Rig {
    fn new() -> Self {
        Self::with(Opts::default())
    }

    fn with(o: Opts) -> Self {
        let t = Arc::new(Mutex::new(1000.0));
        let det = Arc::new(Mutex::new(FakeState {
            vad: o.vad,
            last_voiced: true,
            ..Default::default()
        }));
        let seen = Arc::new(Mutex::new(Seen::default()));
        let mut cfg = HandsFreeConfig {
            enabled: true,
            preroll_s: 1.5,
            idle_timeout_s: 45.0,
            ..Default::default()
        };
        (o.cfg)(&mut cfg);
        let (s1, s2, s3, s4) = (seen.clone(), seen.clone(), seen.clone(), seen.clone());
        let tc = t.clone();
        let mut hf = HandsFree::new(&cfg, Box::new(FakeDetector(det.clone())))
            .with_clock(move || *tc.lock().unwrap())
            .on_wake(move |w| s1.lock().unwrap().wakes.push(w.clone()))
            .on_end(move |e| s2.lock().unwrap().ends.push(e.clone()))
            .on_pause(move |p| s3.lock().unwrap().pauses.push(p.clone()))
            .on_state(move |s| s4.lock().unwrap().states.push(*s));
        if let Some(m) = o.monitor {
            hf = hf.with_mic_monitor(Box::new(FakeMonitor(m)));
        }
        Rig {
            t,
            det,
            seen,
            hf: (o.tweak)(hf),
        }
    }

    fn frames(&mut self, n: usize, loud: bool) {
        let f = vec![if loud { LOUD_V } else { 0.0 }; FRAME];
        for _ in 0..n {
            self.hf.feed(&f);
            *self.t.lock().unwrap() += 0.08;
        }
    }

    fn fire_in(&self, n: u64) {
        let mut d = self.det.lock().unwrap();
        d.fire_at = vec![d.idx + n];
    }

    /// Silence, then speech that fires the detector after `fire_after` voiced frames.
    fn wake(&mut self, fire_after: usize, lead_quiet: usize) -> WakeStart {
        self.frames(lead_quiet, false);
        self.fire_in(fire_after as u64);
        self.frames(fire_after, true);
        self.seen
            .lock()
            .unwrap()
            .wakes
            .last()
            .cloned()
            .expect("wake did not fire")
    }

    fn ends(&self) -> Vec<SessionEnd> {
        self.seen.lock().unwrap().ends.clone()
    }
    fn n_wakes(&self) -> usize {
        self.seen.lock().unwrap().wakes.len()
    }
}

fn end_of(a: &PhraseAction) -> &SessionEnd {
    match a {
        PhraseAction::Finish(e)
        | PhraseAction::Send(e)
        | PhraseAction::Cancel(e)
        | PhraseAction::FalseWake(e) => e,
        other => panic!("not an end: {other:?}"),
    }
}

#[test]
fn wake_starts_session_with_clean_preroll() {
    let mut r = Rig::new();
    let w = r.wake(10, 30);
    assert_eq!(r.hf.state(), HfState::Session);
    assert!(r.hf.in_session());
    assert_eq!(w.session_id, 1);
    assert!((w.score - 0.9).abs() < 1e-6);
    assert!(w.clean_start);
    // 10 voiced frames + 2 frames of quiet lead-in, not the whole 1.5 s ring
    assert_eq!(w.preroll.len(), 12 * FRAME);
    assert_eq!(r.seen.lock().unwrap().states, vec![HfState::Session]);
}

#[test]
fn preroll_without_vad_is_the_whole_window() {
    let mut r = Rig::with(Opts {
        vad: false,
        ..Default::default()
    });
    let w = r.wake(10, 40);
    assert!(!w.clean_start);
    assert_eq!(w.preroll.len(), 19 * FRAME);
}

#[test]
fn preroll_mid_speech_is_not_clean() {
    let mut r = Rig::new();
    r.frames(60, true);
    r.fire_in(3);
    r.frames(3, true);
    let w = r.seen.lock().unwrap().wakes.last().cloned().unwrap();
    assert!(!w.clean_start);
    assert!(
        (19 * FRAME..=25 * FRAME).contains(&w.preroll.len()),
        "{}",
        w.preroll.len() / FRAME
    );
}

#[test]
fn feed_returns_events_too() {
    let mut r = Rig::new();
    r.frames(20, false);
    r.fire_in(2);
    let f = vec![LOUD_V; FRAME];
    let ev1 = r.hf.feed(&f);
    assert!(ev1.is_empty());
    let ev2 = r.hf.feed(&f);
    assert!(
        matches!(
            ev2.as_slice(),
            [HfEvent::State(HfState::Session), HfEvent::Wake(_)]
        ),
        "{ev2:?}"
    );
}

#[test]
fn full_session_send() {
    let mut r = Rig::new();
    r.wake(10, 20);
    let a = r.hf.on_phrase("Transcribe, hey Sarah, just checking in.");
    assert_eq!(
        a,
        PhraseAction::Continue {
            kept: "Hey Sarah, just checking in.".into()
        }
    );
    let b = r.hf.on_phrase("Does five work for you? Transcribe send.");
    assert!(matches!(b, PhraseAction::Send(_)));
    let ends = r.ends();
    assert_eq!(ends.len(), 1);
    assert_eq!(ends[0].reason, EndReason::Send);
    assert_eq!(
        ends[0].text,
        "Hey Sarah, just checking in. Does five work for you?"
    );
    assert!(ends[0].insert() && ends[0].press_enter());
    assert_eq!(end_of(&b), &ends[0]);
    assert_eq!(r.hf.state(), HfState::Listening);
    assert!(!r.hf.in_session());
    let d = r.det.lock().unwrap();
    assert!(d.holds == 1 && d.resets >= 1);
}

#[test]
fn stop_in_first_phrase() {
    let mut r = Rig::new();
    r.wake(10, 20);
    let res =
        r.hf.on_phrase("Transcribe, buy milk and eggs. Transcribe stop.");
    assert!(matches!(res, PhraseAction::Finish(_)));
    let e = &r.ends()[0];
    assert_eq!(e.text, "Buy milk and eggs.");
    assert!(e.insert() && !e.press_enter());
}

#[test]
fn wake_word_then_immediate_stop_inserts_nothing() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe stop.");
    let e = &r.ends()[0];
    assert_eq!(
        (e.reason, e.text.as_str(), e.insert()),
        (EndReason::Stop, "", false)
    );
}

#[test]
fn false_wake_is_dropped() {
    let mut r = Rig::new();
    r.wake(10, 20);
    let res = r.hf.on_phrase("I need to transcribe this video by Friday.");
    assert!(matches!(res, PhraseAction::FalseWake(_)));
    let e = &r.ends()[0];
    assert_eq!(e.reason, EndReason::FalseWake);
    assert!(!e.insert());
    assert_eq!(e.detail, "I need to transcribe this video by Friday.");
}

#[test]
fn verify_off_keeps_text() {
    let mut r = Rig::with(Opts {
        tweak: |h| h.with_verify(false),
        ..Default::default()
    });
    r.wake(10, 20);
    r.hf.on_phrase("Something without the wake word.");
    r.hf.on_phrase("transcribe done");
    assert_eq!(r.ends()[0].text, "Something without the wake word.");
}

#[test]
fn cancel_discards() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe. Dear landlord, I am furious.");
    assert!(matches!(
        r.hf.on_phrase("Transcribe cancel."),
        PhraseAction::Cancel(_)
    ));
    let e = &r.ends()[0];
    assert_eq!(e.reason, EndReason::Cancel);
    assert!(!e.insert());
}

#[test]
fn scratch_that() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe. First line.");
    r.hf.on_phrase("Second line.");
    let res = r.hf.on_phrase("Transcribe scratch that.");
    assert_eq!(
        res,
        PhraseAction::Scratch {
            dropped: Some("Second line.".into())
        }
    );
    assert_eq!(r.hf.session_text(), "First line.");
    let res = r.hf.on_phrase("Wrong words here, transcribe scratch that.");
    assert_eq!(
        res,
        PhraseAction::Scratch {
            dropped: Some("Wrong words here".into())
        }
    );
    assert_eq!(r.hf.session_text(), "First line.");
    r.hf.on_phrase("Third line. Transcribe stop.");
    assert_eq!(r.ends()[0].text, "First line. Third line.");
}

#[test]
fn scratch_with_nothing_to_drop() {
    let mut r = Rig::new();
    r.wake(10, 20);
    assert_eq!(
        r.hf.on_phrase("Transcribe scratch that."),
        PhraseAction::Scratch { dropped: None }
    );
    assert!(r.hf.in_session());
}

#[test]
fn segmenter_split_between_phrase_and_command() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe, notes for Monday.");
    r.hf.on_phrase("That's all. Transcribe.");
    let res = r.hf.on_phrase("Send.");
    assert!(matches!(res, PhraseAction::Send(_)));
    let e = &r.ends()[0];
    assert_eq!(e.text, "Notes for Monday. That's all.");
    assert!(e.press_enter());
}

#[test]
fn dangling_wake_word_kept_when_no_command_follows() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe. We need someone to transcribe");
    r.hf.on_phrase("the interviews by Friday.");
    r.hf.on_phrase("transcribe done");
    assert_eq!(
        r.ends()[0].text,
        "We need someone to transcribe the interviews by Friday."
    );
}

#[test]
fn wake_hit_during_session_arms_bare_command() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe, hello.");
    assert_eq!(
        r.hf.on_phrase("Stop."),
        PhraseAction::Continue {
            kept: "Stop.".into()
        }
    ); // not armed
    r.fire_in(2);
    r.frames(3, true);
    assert!(r.hf.in_session()); // a hit during a session never starts another one
    assert!(matches!(r.hf.on_phrase("Stop."), PhraseAction::Finish(_)));
    assert_eq!(r.ends()[0].text, "Hello. Stop.");
    assert_eq!(r.n_wakes(), 1);
}

#[test]
fn arming_expires() {
    let mut r = Rig::with(Opts {
        tweak: |h| h.with_arm_s(2.0),
        ..Default::default()
    });
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe, hello.");
    r.fire_in(1);
    r.frames(1, true);
    r.frames(40, true);
    assert!(matches!(
        r.hf.on_phrase("Stop."),
        PhraseAction::Continue { .. }
    ));
}

#[test]
fn idle_timeout_inserts() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe, remember the dentist.");
    r.frames((44.0 / 0.08) as usize, false);
    assert!(r.hf.in_session());
    r.frames((2.0 / 0.08) as usize, false);
    let e = &r.ends()[0];
    assert_eq!(e.reason, EndReason::IdleTimeout);
    assert!(e.insert());
    assert_eq!(e.text, "Remember the dentist.");
}

#[test]
fn speech_keeps_session_alive() {
    let mut r = Rig::new();
    r.wake(10, 20);
    for _ in 0..3 {
        r.frames((30.0 / 0.08) as usize, false);
        r.frames(5, true);
    }
    assert!(r.hf.in_session());
}

#[test]
fn max_duration() {
    let mut r = Rig::with(Opts {
        tweak: |h| h.with_max_session_s(10.0),
        ..Default::default()
    });
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe, long story.");
    r.frames((11.0 / 0.08) as usize, true);
    let e = &r.ends()[0];
    assert_eq!(
        (e.reason, e.text.as_str()),
        (EndReason::MaxDuration, "Long story.")
    );
}

#[test]
fn cooldown_after_session_blocks_rewake() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe stop.");
    r.fire_in(2);
    r.frames(4, true);
    assert_eq!(r.n_wakes(), 1);
    r.frames(30, false);
    r.fire_in(5);
    r.frames(6, true);
    assert_eq!(r.n_wakes(), 2);
    assert_eq!(r.seen.lock().unwrap().wakes[1].session_id, 2);
}

#[test]
fn late_and_stale_phrases_are_ignored() {
    let mut r = Rig::new();
    let w = r.wake(10, 20);
    r.hf.on_phrase("Transcribe stop.");
    assert_eq!(r.hf.on_phrase("late words"), PhraseAction::Ignored);
    r.frames(30, false);
    r.fire_in(5);
    r.frames(6, true);
    assert_eq!(
        r.hf.on_phrase_for("Transcribe, x", Some(w.session_id)),
        PhraseAction::Ignored
    );
    let id = r.hf.session_id();
    assert_ne!(
        r.hf.on_phrase_for("Transcribe, x", id),
        PhraseAction::Ignored
    );
}

#[test]
fn empty_phrase_refreshes_idle_timer() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.frames((40.0 / 0.08) as usize, false);
    assert_eq!(
        r.hf.on_phrase("   "),
        PhraseAction::Continue {
            kept: String::new()
        }
    );
    r.frames((10.0 / 0.08) as usize, false);
    assert!(r.hf.in_session());
}

#[test]
fn external_finish_and_cancel() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.hf.on_phrase("Transcribe, note to self.");
    let end = r.hf.finish(true).unwrap();
    assert!(end.reason == EndReason::Send && end.press_enter());
    assert!(r.hf.finish(false).is_none() && r.hf.cancel().is_none());
    r.frames(30, false);
    r.fire_in(5);
    r.frames(6, true);
    assert_eq!(r.hf.cancel().unwrap().reason, EndReason::Cancel);
    assert_eq!(r.ends().len(), 2);
}

#[test]
fn disable_cancels_session_and_stops_listening() {
    let mut r = Rig::new();
    r.wake(10, 20);
    r.hf.set_enabled(false);
    assert_eq!(r.ends()[0].reason, EndReason::Cancel);
    assert_eq!(r.hf.state(), HfState::Off);
    let n = r.det.lock().unwrap().processed;
    r.frames(10, true);
    assert_eq!(r.det.lock().unwrap().processed, n);
    r.hf.set_enabled(true);
    assert_eq!(r.hf.state(), HfState::Listening);
}

#[test]
fn disabled_config_starts_off() {
    let mut r = Rig::with(Opts {
        cfg: |c| c.enabled = false,
        ..Default::default()
    });
    assert_eq!(r.hf.state(), HfState::Off);
    r.frames(5, true);
    assert_eq!(r.det.lock().unwrap().processed, 0);
}

#[test]
fn pause_while_mic_in_use() {
    let apps = Arc::new(Mutex::new(Vec::new()));
    let mut r = Rig::with(Opts {
        monitor: Some(apps.clone()),
        tweak: |h| h.with_call_poll_s(1.0),
        ..Default::default()
    });
    r.frames(5, false);
    *apps.lock().unwrap() = vec!["Zoom".into()];
    r.frames(15, false);
    assert_eq!(r.hf.state(), HfState::Paused);
    assert_eq!(r.hf.paused_apps(), ["Zoom".to_string()]);
    assert_eq!(
        r.seen.lock().unwrap().pauses,
        vec![(true, vec!["Zoom".to_string()])]
    );
    let n = r.det.lock().unwrap().processed;
    r.fire_in(1);
    r.frames(10, true);
    assert_eq!(r.det.lock().unwrap().processed, n);
    assert_eq!(r.n_wakes(), 0);
    r.det.lock().unwrap().fire_at.clear();
    apps.lock().unwrap().clear();
    r.frames(15, false);
    assert_eq!(r.hf.state(), HfState::Listening);
    assert_eq!(
        r.seen.lock().unwrap().pauses.last().unwrap(),
        &(false, vec![])
    );
}

#[test]
fn pause_on_calls_off() {
    let apps = Arc::new(Mutex::new(vec!["Teams".to_string()]));
    let mut r = Rig::with(Opts {
        monitor: Some(apps),
        cfg: |c| c.pause_on_calls = false,
        ..Default::default()
    });
    r.frames(50, false);
    assert_eq!(r.hf.state(), HfState::Listening);
}

#[test]
fn panicking_callback_does_not_break_feed() {
    let mut r = Rig::with(Opts {
        tweak: |h| h.on_wake(|_| panic!("orchestrator bug")),
        ..Default::default()
    });
    r.frames(20, false);
    r.fire_in(5);
    r.frames(10, true);
    assert!(r.hf.in_session());
}

#[test]
fn blocks_of_any_size() {
    let mut r = Rig::new();
    r.frames(20, false);
    r.fire_in(4);
    let audio = vec![0.1f32; FRAME * 5];
    for chunk in audio.chunks(audio.len() / 7 + 1) {
        r.hf.feed(chunk);
    }
    assert_eq!(r.det.lock().unwrap().idx, 25);
    assert_eq!(r.n_wakes(), 1);
}

#[test]
fn custom_phrase() {
    let mut r = Rig::with(Opts {
        cfg: |c| c.phrase = "juniper".into(),
        ..Default::default()
    });
    r.wake(10, 20);
    assert_eq!(
        r.hf.on_phrase("Juniper, call mom."),
        PhraseAction::Continue {
            kept: "Call mom.".into()
        }
    );
    assert!(matches!(
        r.hf.on_phrase("Juniper send"),
        PhraseAction::Send(_)
    ));
    assert_eq!(r.ends()[0].text, "Call mom.");
}
