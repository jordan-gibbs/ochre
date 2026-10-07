//! Hands-free end to end with fakes: a mic with an always-on tap, a scripted wake scorer (loud
//! frames score high), a segmenter that cuts a phrase on a marker block, and a scripted STT.
//! Real `HandsFree` controller, real orchestrator, no models, no keyboard.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use ochre::app::Cue;
use ochre::handsfree::{Listener, WakeFactory};
use ochre::seams::{
    LevelFn, Mic, PartialFn, SegmentSession, SegmenterFactory, Sink, Tap, TranscribeFn,
};
use ochre::{App, Engines, Loader};
use ochre_core::Result;
use ochre_core::config::Config;
use ochre_core::events::{Bus, Command, EngineInfo, Event, State, Trigger};
use ochre_core::history::History;
use ochre_core::platform::{FocusInfo, Gesture, Injector};
use ochre_core::stt::{SttEngine, SttOptions, SttResult};
use ochre_wake::{DetectorOptions, FRAME, FrameScorer, HandsFree, WakeDetector};

const LOUD: f32 = 0.5; // the fake scorer's "wake word"
const MARK: f32 = 0.2; // a block of MARK_LEN of these ends a phrase
const MARK_LEN: usize = 100;

// ------------------------------------------------------------------------------- fakes

/// Pops one scripted transcript per phrase.
#[derive(Default)]
struct ScriptStt {
    texts: Mutex<VecDeque<&'static str>>,
}
impl SttEngine for ScriptStt {
    fn info(&self) -> EngineInfo {
        EngineInfo {
            id: "fake".into(),
            label: "fake".into(),
            kind: "local".into(),
            models: vec![],
            default_model: String::new(),
            needs_key: false,
            note: String::new(),
            languages: "en".into(),
        }
    }
    fn load(&mut self, _: ochre_core::stt::ProgressFn) -> Result<()> {
        Ok(())
    }
    fn transcribe(&self, pcm: &[f32], _: &SttOptions) -> Result<SttResult> {
        let text = self.texts.lock().pop_front().unwrap_or_default();
        Ok(SttResult {
            text: text.into(),
            duration_ms: pcm.len() as u64 / 16,
            processing_ms: 1,
            language: None,
        })
    }
}

#[derive(Default)]
struct FakeInjector {
    typed: Mutex<Vec<String>>,
    enters: Mutex<u32>,
}
impl Injector for FakeInjector {
    fn focus(&self) -> FocusInfo {
        FocusInfo {
            app_name: "notepad".into(),
            window_id: "w1".into(),
            ..Default::default()
        }
    }
    fn type_text(&self, t: &str) -> Result<()> {
        self.typed.lock().push(t.into());
        Ok(())
    }
    fn paste_text(&self, t: &str) -> Result<()> {
        self.type_text(t)
    }
    fn press_enter(&self) -> Result<()> {
        *self.enters.lock() += 1;
        Ok(())
    }
}

/// Like the capture pump: every block goes to the tap (with its absolute position) and to the
/// session sink; `begin_from` replays the history from a position first.
#[derive(Default)]
struct TapMic {
    history: Mutex<Vec<f32>>,
    tap: Mutex<Option<Tap>>,
    sink: Mutex<Option<Sink>>,
    /// `from` of each `begin_from`.
    froms: Mutex<Vec<u64>>,
}
impl TapMic {
    fn push(&self, block: &[f32]) {
        let at = {
            let mut h = self.history.lock();
            h.extend_from_slice(block);
            (h.len() - block.len()) as u64
        };
        if let Some(t) = self.tap.lock().as_mut() {
            t(at, block);
        }
        if let Some(s) = self.sink.lock().as_mut() {
            s(block);
        }
    }
    fn frames(&self, n: usize, v: f32) {
        for _ in 0..n {
            self.push(&[v; FRAME]);
        }
    }
    fn wake_word(&self) {
        self.frames(2, 0.0);
        self.frames(3, LOUD);
    }
    /// End a phrase: the segmenter transcribes what it has.
    fn phrase(&self) {
        self.frames(2, 0.01);
        self.push(&[MARK; MARK_LEN]);
    }
    fn tapped(&self) -> bool {
        self.tap.lock().is_some()
    }
}
impl Mic for TapMic {
    fn begin(&self, _: u32, sink: Sink, _: LevelFn) -> Result<()> {
        *self.sink.lock() = Some(sink);
        Ok(())
    }
    fn begin_from(&self, from: u64, mut sink: Sink, _: LevelFn) -> Result<()> {
        self.froms.lock().push(from);
        let pre = self.history.lock()[from as usize..].to_vec();
        sink(&pre);
        *self.sink.lock() = Some(sink);
        Ok(())
    }
    fn end(&self) -> Result<u64> {
        self.sink.lock().take();
        Ok(1000)
    }
    fn cancel(&self) {
        self.sink.lock().take();
    }
    fn set_tap(&self, tap: Option<Tap>) -> Result<()> {
        *self.tap.lock() = tap;
        Ok(())
    }
}

/// Cuts a phrase at each marker block and transcribes it right away (as the real segmenter does
/// on its decode thread while the user keeps talking).
struct MarkSegmenter;
struct MarkSession {
    audio: Vec<f32>,
    transcribe: TranscribeFn,
    results: Vec<SttResult>,
}
impl SegmenterFactory for MarkSegmenter {
    fn start(&self, transcribe: TranscribeFn, _: Option<PartialFn>) -> Box<dyn SegmentSession> {
        Box::new(MarkSession {
            audio: vec![],
            transcribe,
            results: vec![],
        })
    }
}
impl SegmentSession for MarkSession {
    fn feed(&mut self, pcm: &[f32]) {
        if pcm.len() == MARK_LEN && pcm.iter().all(|&v| v == MARK) {
            let audio = std::mem::take(&mut self.audio);
            if let Ok(r) = (self.transcribe)(&audio) {
                self.results.push(r);
            }
        } else {
            self.audio.extend_from_slice(pcm);
        }
    }
    fn finish(self: Box<Self>) -> Result<Vec<SttResult>> {
        Ok(self.results)
    }
    fn cancel(self: Box<Self>) {}
}

/// Loud frames (int16 scale) score 0.9, anything else 0.
struct LoudScorer;
impl FrameScorer for LoudScorer {
    fn push_and_score(&mut self, frame: &[f32]) -> Result<f32> {
        let mean = frame.iter().map(|v| v.abs()).sum::<f32>() / frame.len() as f32;
        Ok(if mean > 10_000.0 { 0.9 } else { 0.0 })
    }
    fn reset(&mut self) {}
}

fn fake_factory(built: Arc<Mutex<u32>>) -> WakeFactory {
    Arc::new(move |cfg: &Config| {
        *built.lock() += 1;
        let det = WakeDetector::with_scorer(
            Box::new(LoudScorer),
            DetectorOptions {
                threshold: cfg.handsfree.threshold,
                ..Default::default()
            },
        );
        let hf = HandsFree::new(&cfg.handsfree, Box::new(det)).with_post_session_cooldown_s(0.2);
        Ok(Listener {
            hf,
            desc: "fake".into(),
        })
    })
}

// ------------------------------------------------------------------------------- rig

struct Rig {
    app: App,
    mic: Arc<TapMic>,
    stt: Arc<ScriptStt>,
    inj: Arc<FakeInjector>,
    events: Arc<Mutex<Vec<Event>>>,
    cues: Arc<Mutex<Vec<Cue>>>,
    built: Arc<Mutex<u32>>,
}

fn cfg() -> Config {
    let mut c = Config::default();
    c.debug.log_timings = false;
    c.handsfree.enabled = true;
    c.handsfree.threshold = 0.5;
    c.handsfree.idle_timeout_s = 30.0;
    c.handsfree.pause_on_calls = false;
    c
}

fn rig(cfg: Config) -> Rig {
    let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    let mic = Arc::new(TapMic::default());
    let stt = Arc::new(ScriptStt::default());
    let inj = Arc::new(FakeInjector::default());
    let cues = Arc::new(Mutex::new(vec![]));
    let built = Arc::new(Mutex::new(0));
    let (m, s, i, c) = (mic.clone(), stt.clone(), inj.clone(), cues.clone());
    let loader: Loader = Arc::new(move |_, _| {
        let c = c.clone();
        Ok(Engines {
            stt: s.clone(),
            fallback_stt: None,
            refiner: None,
            injector: i.clone(),
            mic: m.clone(),
            segmenter: Arc::new(MarkSegmenter),
            guard: |_raw, refined, _mode| refined.to_string(),
            cue: Some(Arc::new(move |cue| c.lock().push(cue))),
        })
    });
    let bus = Bus::new();
    let events = Arc::new(Mutex::new(vec![]));
    let ev = events.clone();
    bus.subscribe(move |e| ev.lock().push(e.clone()));
    let app = App::new(cfg.clone(), bus, History::disabled(), loader);
    app.set_wake_factory(fake_factory(built.clone()));
    app.start();
    wait(|| app.state() == State::Idle);
    if cfg.handsfree.enabled {
        wait(|| app.handsfree_armed() && mic.tapped());
    }
    Rig {
        app,
        mic,
        stt,
        inj,
        events,
        cues,
        built,
    }
}

fn wait(cond: impl Fn() -> bool) {
    let t = Instant::now();
    while !cond() {
        assert!(t.elapsed() < Duration::from_secs(5), "timed out");
        std::thread::sleep(Duration::from_millis(2));
    }
}

impl Rig {
    fn script(&self, texts: &[&'static str]) {
        self.stt.texts.lock().extend(texts.iter().copied());
    }
    fn wake(&self) {
        self.mic.wake_word();
        wait(|| self.app.state() == State::Handsfree);
    }
    fn typed(&self) -> Vec<String> {
        self.inj.typed.lock().clone()
    }
    fn saw_handsfree(&self) -> bool {
        self.events.lock().iter().any(|e| {
            matches!(
                e,
                Event::State {
                    state: State::Handsfree,
                    ..
                }
            )
        })
    }
    fn result_count(&self) -> usize {
        self.events
            .lock()
            .iter()
            .filter(|e| matches!(e, Event::Result { .. }))
            .count()
    }
    /// Let the wake thread and the session worker settle, then require Idle.
    fn settle(&self) {
        std::thread::sleep(Duration::from_millis(80));
        wait(|| self.app.state() == State::Idle);
    }
}

// ------------------------------------------------------------------------------- tests

#[test]
fn wake_then_stop_inserts_the_text_without_control_words() {
    let r = rig(cfg());
    r.script(&["Transcribe hello world. Transcribe stop."]);
    r.wake();
    assert!(r.events.lock().iter().any(|e| matches!(
        e,
        Event::State {
            state: State::Handsfree,
            trigger: Some(Trigger::Wake),
            handsfree_armed: true,
            ..
        }
    )));
    // The session's audio starts before the wake word (pre-roll from the tap position).
    let from = r.mic.froms.lock()[0];
    let loud_start = (r.mic.history.lock().len() - 3 * FRAME) as u64;
    assert!(
        from <= loud_start,
        "from {from} after the wake word {loud_start}"
    );
    r.mic.phrase();
    wait(|| !r.typed().is_empty());
    r.settle();
    assert_eq!(r.typed(), vec!["Hello world. "]);
    assert_eq!(*r.inj.enters.lock(), 0);
    assert_eq!(*r.cues.lock(), vec![Cue::Start, Cue::Stop]);
    assert!(r.app.handsfree_armed());
}

#[test]
fn send_inserts_then_presses_enter() {
    let r = rig(cfg());
    r.script(&["Transcribe see you soon.", "Transcribe send."]);
    r.wake();
    r.mic.phrase();
    r.mic.phrase();
    wait(|| *r.inj.enters.lock() == 1);
    r.settle();
    assert_eq!(r.typed(), vec!["See you soon."]);
}

#[test]
fn cancel_inserts_nothing() {
    let r = rig(cfg());
    r.script(&["Transcribe never mind this. Transcribe cancel."]);
    r.wake();
    r.mic.phrase();
    wait(|| r.cues.lock().contains(&Cue::Cancel));
    r.settle();
    assert!(r.typed().is_empty());
    assert_eq!(r.result_count(), 0);
}

#[test]
fn false_wake_ends_silently() {
    let r = rig(cfg());
    r.script(&["I need to transcribe this video later."]);
    r.wake();
    r.mic.phrase();
    r.settle();
    assert!(r.typed().is_empty());
    assert_eq!(r.result_count(), 0);
    assert_eq!(*r.cues.lock(), vec![Cue::Start]); // no cancel sound for a false wake
}

#[test]
fn scratch_that_drops_the_last_phrase() {
    let r = rig(cfg());
    r.script(&[
        "Transcribe first line.",
        "Wrong line.",
        "Transcribe scratch that.",
        "Third line. Transcribe done.",
    ]);
    r.wake();
    for _ in 0..4 {
        r.mic.phrase();
    }
    wait(|| !r.typed().is_empty());
    r.settle();
    assert_eq!(r.typed(), vec!["First line. Third line. "]);
}

#[test]
fn escape_cancels_a_handsfree_session() {
    let r = rig(cfg());
    r.script(&["Transcribe keep this out."]);
    r.wake();
    r.mic.phrase();
    r.app.on_gesture(Gesture::Cancel);
    assert_eq!(r.app.state(), State::Idle);
    std::thread::sleep(Duration::from_millis(50));
    r.mic.frames(5, 0.0);
    r.settle();
    assert!(r.typed().is_empty());
    // listening again afterwards
    r.script(&["Transcribe second try. Transcribe stop."]);
    std::thread::sleep(Duration::from_millis(250)); // post-session cooldown
    r.mic.frames(25, 0.0);
    r.wake();
    r.mic.phrase();
    wait(|| !r.typed().is_empty());
    assert_eq!(r.typed(), vec!["Second try. "]);
}

#[test]
fn idle_timeout_finishes_and_inserts() {
    let mut c = cfg();
    c.handsfree.idle_timeout_s = 0.3;
    let r = rig(c);
    r.script(&["Transcribe note to self."]);
    r.wake();
    r.mic.phrase();
    wait(|| !r.typed().is_empty()); // the wake thread's timer tick ends it
    r.settle();
    assert_eq!(r.typed(), vec!["Note to self. "]);
}

#[test]
fn hotkey_session_holds_the_detector() {
    let r = rig(cfg());
    r.script(&["Dictated by hotkey."]);
    r.app.on_gesture(Gesture::Press);
    assert_eq!(r.app.state(), State::Recording);
    r.mic.wake_word(); // the wake word said inside a hotkey dictation
    r.mic.frames(3, LOUD);
    r.mic.phrase();
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(r.app.state(), State::Recording);
    r.app.on_gesture(Gesture::Release);
    wait(|| !r.typed().is_empty());
    r.settle();
    assert!(!r.saw_handsfree());
    assert_eq!(r.typed(), vec!["Dictated by hotkey. "]);
    // Right after the session the detector is held: the same word does not wake it.
    r.mic.wake_word();
    std::thread::sleep(Duration::from_millis(100));
    assert!(!r.saw_handsfree());
    // After the cooldown it does.
    std::thread::sleep(Duration::from_millis(250));
    r.mic.frames(25, 0.0);
    r.wake();
}

#[test]
fn disabling_handsfree_stops_the_thread_and_removes_the_tap() {
    let r = rig(cfg());
    assert!(r.app.wake_running());
    r.app.command(Command::SetHandsfree { enabled: false });
    wait(|| !r.app.wake_running() && !r.app.handsfree_armed() && !r.mic.tapped());
    // The state events now say hands-free is not armed.
    assert!(r.events.lock().iter().rev().any(|e| matches!(
        e,
        Event::State {
            handsfree_armed: false,
            ..
        }
    )));
    // Nothing wakes it.
    r.mic.wake_word();
    std::thread::sleep(Duration::from_millis(100));
    assert!(!r.saw_handsfree());
    // Turning it back on starts a fresh listener.
    r.app.command(Command::SetHandsfree { enabled: true });
    wait(|| r.app.handsfree_armed() && r.mic.tapped());
    assert_eq!(*r.built.lock(), 2);
}

#[test]
fn disabling_during_a_session_cancels_it() {
    let r = rig(cfg());
    r.script(&["Transcribe half a thought."]);
    r.wake();
    r.mic.phrase();
    r.app.command(Command::SetHandsfree { enabled: false });
    wait(|| r.app.state() == State::Idle && !r.app.wake_running());
    std::thread::sleep(Duration::from_millis(50));
    assert!(r.typed().is_empty());
}

#[test]
fn off_by_default_no_listener() {
    let mut c = cfg();
    c.handsfree.enabled = false;
    let r = rig(c);
    std::thread::sleep(Duration::from_millis(50));
    assert!(!r.app.wake_running() && !r.app.handsfree_armed() && !r.mic.tapped());
    assert_eq!(*r.built.lock(), 0);
}

#[test]
fn a_factory_error_is_a_notice_not_silence() {
    let bus = Bus::new();
    let events = Arc::new(Mutex::new(vec![]));
    let ev = events.clone();
    bus.subscribe(move |e| ev.lock().push(e.clone()));
    let mic = Arc::new(TapMic::default());
    let m = mic.clone();
    let loader: Loader = Arc::new(move |_, _| {
        Ok(Engines {
            stt: Arc::new(ScriptStt::default()),
            fallback_stt: None,
            refiner: None,
            injector: Arc::new(FakeInjector::default()),
            mic: m.clone(),
            segmenter: Arc::new(MarkSegmenter),
            guard: |_raw, refined, _mode| refined.to_string(),
            cue: None,
        })
    });
    let app = App::new(cfg(), bus, History::disabled(), loader);
    app.set_wake_factory(Arc::new(|_: &Config| {
        Err(ochre_core::Error::Model("no wake word model".into()))
    }));
    app.start();
    wait(|| {
        events.lock().iter().any(|e| {
            matches!(e, Event::Notice { message } if message.contains("Hands-free couldn't start"))
        })
    });
    assert!(!app.handsfree_armed());
    assert!(!mic.tapped());
}
