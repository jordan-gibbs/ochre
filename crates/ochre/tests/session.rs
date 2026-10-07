//! End-to-end session logic with fake engines: no mic, no models, no keyboard.

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use ochre::app::Cue;
use ochre::seams::{LevelFn, Mic, PartialFn, SegmentSession, SegmenterFactory, Sink, TranscribeFn};
use ochre::{App, Engines, Loader};
use ochre_core::config::Config;
use ochre_core::events::{Bus, EngineInfo, Event, State};
use ochre_core::history::History;
use ochre_core::platform::{FocusInfo, Gesture, Injector};
use ochre_core::refine::{RefineContext, Refiner};
use ochre_core::stt::{SttEngine, SttOptions, SttResult};
use ochre_core::{Error, Result};

fn info(id: &str, kind: &str) -> EngineInfo {
    EngineInfo {
        id: id.into(),
        label: id.into(),
        kind: kind.into(),
        models: vec![],
        default_model: String::new(),
        needs_key: false,
        note: String::new(),
        languages: "en".into(),
    }
}

/// Returns `text` for any non-empty audio, or fails with `fail` if set.
struct FakeStt {
    id: &'static str,
    kind: &'static str,
    text: &'static str,
    fail: Option<fn() -> Error>,
}
impl SttEngine for FakeStt {
    fn info(&self) -> EngineInfo {
        info(self.id, self.kind)
    }
    fn load(&mut self, _: ochre_core::stt::ProgressFn) -> Result<()> {
        Ok(())
    }
    fn transcribe(&self, pcm: &[f32], _: &SttOptions) -> Result<SttResult> {
        if let Some(f) = self.fail {
            return Err(f());
        }
        let text = if pcm.is_empty() { "" } else { self.text };
        Ok(SttResult {
            text: text.into(),
            duration_ms: pcm.len() as u64 / 16,
            processing_ms: 5,
            language: None,
        })
    }
}

struct Upper(bool);
impl Refiner for Upper {
    fn info(&self) -> EngineInfo {
        info("upper", "local")
    }
    fn load(&mut self, _: ochre_core::stt::ProgressFn) -> Result<()> {
        Ok(())
    }
    fn refine(&self, text: &str, _: &RefineContext, t: Duration) -> Result<String> {
        if self.0 {
            Err(Error::Timeout(t))
        } else {
            Ok(text.to_uppercase())
        }
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

/// The test pushes audio through `speak`; `end` is a no-op flush.
#[derive(Default)]
struct FakeMic {
    sink: Mutex<Option<Sink>>,
}
impl FakeMic {
    fn speak(&self, samples: usize) {
        if let Some(s) = self.sink.lock().as_mut() {
            s(&vec![0.1; samples]);
        }
    }
}
impl Mic for FakeMic {
    fn begin(&self, _: u32, sink: Sink, _: LevelFn) -> Result<()> {
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
}

/// Decodes everything as one phrase at finish.
struct OnePhrase;
struct OnePhraseSession {
    audio: Vec<f32>,
    transcribe: TranscribeFn,
}
impl SegmenterFactory for OnePhrase {
    fn start(&self, transcribe: TranscribeFn, _: Option<PartialFn>) -> Box<dyn SegmentSession> {
        Box::new(OnePhraseSession {
            audio: vec![],
            transcribe,
        })
    }
}
impl SegmentSession for OnePhraseSession {
    fn feed(&mut self, pcm: &[f32]) {
        self.audio.extend_from_slice(pcm);
    }
    fn finish(self: Box<Self>) -> Result<Vec<SttResult>> {
        Ok(vec![(self.transcribe)(&self.audio)?])
    }
    fn cancel(self: Box<Self>) {}
}

/// Like `OnePhrase`, but every `feed` reports the next scripted partial (as the real segmenter
/// does from its decode thread), with the previous one as the stable prefix.
struct Scripted(&'static [&'static str]);
struct ScriptedSession {
    inner: OnePhraseSession,
    script: &'static [&'static str],
    n: usize,
    on_partial: Option<PartialFn>,
}
impl SegmenterFactory for Scripted {
    fn start(
        &self,
        transcribe: TranscribeFn,
        on_partial: Option<PartialFn>,
    ) -> Box<dyn SegmentSession> {
        Box::new(ScriptedSession {
            inner: OnePhraseSession {
                audio: vec![],
                transcribe,
            },
            script: self.0,
            n: 0,
            on_partial,
        })
    }
}
impl SegmentSession for ScriptedSession {
    fn feed(&mut self, pcm: &[f32]) {
        self.inner.feed(pcm);
        if let (Some(f), Some(t)) = (&self.on_partial, self.script.get(self.n)) {
            let stable = self.n.checked_sub(1).map_or(0, |i| self.script[i].len());
            f(t, stable);
        }
        self.n += 1;
    }
    fn finish(self: Box<Self>) -> Result<Vec<SttResult>> {
        Box::new(self.inner).finish()
    }
    fn cancel(self: Box<Self>) {}
}

struct Rig {
    app: App,
    mic: Arc<FakeMic>,
    inj: Arc<FakeInjector>,
    events: Arc<Mutex<Vec<Event>>>,
    cues: Arc<Mutex<Vec<Cue>>>,
}

fn rig(cfg: Config, stt: FakeStt, fallback: Option<FakeStt>, refiner: Option<Upper>) -> Rig {
    rig_seg(cfg, stt, fallback, refiner, Arc::new(OnePhrase))
}

fn rig_seg(
    cfg: Config,
    stt: FakeStt,
    fallback: Option<FakeStt>,
    refiner: Option<Upper>,
    segmenter: Arc<dyn SegmenterFactory>,
) -> Rig {
    let refiner: Option<Arc<dyn Refiner>> = refiner.map(|r| Arc::new(r) as _);
    rig_full(cfg, stt, fallback, refiner, segmenter, History::disabled())
}

fn rig_full(
    cfg: Config,
    stt: FakeStt,
    fallback: Option<FakeStt>,
    refiner: Option<Arc<dyn Refiner>>,
    segmenter: Arc<dyn SegmenterFactory>,
    history: History,
) -> Rig {
    let mut cfg = cfg;
    cfg.debug.log_timings = false; // keep tests out of the user's data dir
    let mic = Arc::new(FakeMic::default());
    let inj = Arc::new(FakeInjector::default());
    let cues = Arc::new(Mutex::new(vec![]));
    let stt: Arc<dyn SttEngine> = Arc::new(stt);
    let fallback: Option<Arc<dyn SttEngine>> = fallback.map(|f| Arc::new(f) as _);
    let (m, i, c) = (mic.clone(), inj.clone(), cues.clone());
    let loader: Loader = Arc::new(move |_, _| {
        let c = c.clone();
        Ok(Engines {
            stt: stt.clone(),
            fallback_stt: fallback.clone(),
            refiner: refiner.clone(),
            injector: i.clone(),
            mic: m.clone(),
            segmenter: segmenter.clone(),
            guard: |_raw, refined, _mode| refined.to_string(),
            cue: Some(Arc::new(move |cue| c.lock().push(cue))),
        })
    });
    let bus = Bus::new();
    let events = Arc::new(Mutex::new(vec![]));
    let ev = events.clone();
    bus.subscribe(move |e| ev.lock().push(e.clone()));
    let app = App::new(cfg, bus, history, loader);
    app.start();
    wait(|| app.state() == State::Idle);
    Rig {
        app,
        mic,
        inj,
        events,
        cues,
    }
}

fn wait(cond: impl Fn() -> bool) {
    let t = Instant::now();
    while !cond() {
        assert!(t.elapsed() < Duration::from_secs(5), "timed out");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn stt(text: &'static str) -> FakeStt {
    FakeStt {
        id: "fake",
        kind: "local",
        text,
        fail: None,
    }
}

fn result_of(r: &Rig) -> Option<(String, bool)> {
    r.events.lock().iter().find_map(|e| match e {
        Event::Result { text, refined, .. } => Some((text.clone(), *refined)),
        _ => None,
    })
}

#[test]
fn hold_and_release_types_the_text() {
    let r = rig(Config::default(), stt("Hello world."), None, None);
    r.app.on_gesture(Gesture::Press);
    assert_eq!(r.app.state(), State::Recording);
    r.mic.speak(16_000);
    r.app.on_gesture(Gesture::Release);
    wait(|| r.app.state() == State::Idle);
    assert_eq!(*r.inj.typed.lock(), vec!["Hello world. "]);
    assert_eq!(result_of(&r), Some(("Hello world.".into(), false)));
    assert_eq!(*r.cues.lock(), vec![Cue::Start, Cue::Stop]);
}

#[test]
fn double_tap_locks_then_tap_finishes() {
    let r = rig(Config::default(), stt("Locked."), None, None);
    r.app.on_gesture(Gesture::Press);
    r.app.on_gesture(Gesture::Lock);
    assert_eq!(r.app.state(), State::Locked);
    r.mic.speak(8000);
    r.app.on_gesture(Gesture::Finish);
    wait(|| !r.inj.typed.lock().is_empty());
}

#[test]
fn abort_and_cancel_insert_nothing() {
    let r = rig(Config::default(), stt("Nope."), None, None);
    r.app.on_gesture(Gesture::Press);
    r.app.on_gesture(Gesture::Abort);
    assert_eq!(r.app.state(), State::Idle);
    r.app.on_gesture(Gesture::Press);
    r.mic.speak(8000);
    r.app.on_gesture(Gesture::Cancel);
    std::thread::sleep(Duration::from_millis(50));
    assert!(r.inj.typed.lock().is_empty());
    assert_eq!(*r.cues.lock(), vec![Cue::Start, Cue::Start, Cue::Cancel]);
}

#[test]
fn refinement_applies_and_raw_gesture_skips_it() {
    let mut cfg = Config::default();
    cfg.inject.trailing_space = false;
    let r = rig(cfg, stt("make it loud"), None, Some(Upper(false)));
    r.app.on_gesture(Gesture::Press);
    r.mic.speak(8000);
    r.app.on_gesture(Gesture::Release);
    wait(|| r.inj.typed.lock().len() == 1);
    assert_eq!(r.inj.typed.lock()[0], "MAKE IT LOUD");
    wait(|| r.app.state() == State::Idle);
    r.app.on_gesture(Gesture::Press);
    r.mic.speak(8000);
    r.app.on_gesture(Gesture::FinishRaw);
    wait(|| r.inj.typed.lock().len() == 2);
    // Joined to the previous insertion in the same window with one space.
    assert_eq!(r.inj.typed.lock()[1], " make it loud");
}

#[test]
fn refiner_failure_falls_back_to_raw() {
    let r = rig(Config::default(), stt("keep me"), None, Some(Upper(true)));
    r.app.on_gesture(Gesture::Press);
    r.mic.speak(8000);
    r.app.on_gesture(Gesture::Release);
    wait(|| !r.inj.typed.lock().is_empty());
    assert_eq!(r.inj.typed.lock()[0], "keep me ");
}

#[test]
fn cloud_failure_uses_local_fallback() {
    let cloud = FakeStt {
        id: "cloud",
        kind: "cloud",
        text: "",
        fail: Some(|| Error::Network {
            provider: "x".into(),
            message: "down".into(),
        }),
    };
    let r = rig(Config::default(), cloud, Some(stt("From local.")), None);
    r.app.on_gesture(Gesture::Press);
    r.mic.speak(8000);
    r.app.on_gesture(Gesture::Release);
    wait(|| !r.inj.typed.lock().is_empty());
    assert_eq!(r.inj.typed.lock()[0], "From local. ");
    assert!(
        r.events
            .lock()
            .iter()
            .any(|e| matches!(e, Event::Notice { .. }))
    );
}

#[test]
fn silence_inserts_nothing_and_says_so() {
    let r = rig(Config::default(), stt("unused"), None, None);
    r.app.on_gesture(Gesture::Press);
    r.app.on_gesture(Gesture::Release);
    wait(|| {
        r.app.state() == State::Idle
            && r.events
                .lock()
                .iter()
                .any(|e| matches!(e, Event::Notice { .. }))
    });
    assert!(r.inj.typed.lock().is_empty());
}

#[test]
fn send_presses_enter_without_trailing_space() {
    let r = rig(Config::default(), stt("Ship it."), None, None);
    assert!(r.app.begin(ochre_core::events::Trigger::Wake));
    assert_eq!(r.app.state(), State::Handsfree);
    r.mic.speak(8000);
    r.app.end(true, false);
    wait(|| *r.inj.enters.lock() == 1);
    assert_eq!(r.inj.typed.lock()[0], "Ship it.");
}

#[test]
fn second_press_while_busy_is_ignored() {
    let r = rig(Config::default(), stt("One."), None, None);
    assert!(r.app.begin(ochre_core::events::Trigger::Ui));
    assert!(!r.app.begin(ochre_core::events::Trigger::Ui));
}

fn partials(r: &Rig) -> Vec<String> {
    r.events
        .lock()
        .iter()
        .filter_map(|e| match e {
            Event::Partial { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

const SCRIPT: &[&str] = &[
    "Um, ship",
    "Um, ship it after",
    "Um, ship it after the tests, uh",
];

#[test]
fn partials_stream_in_order_and_are_never_injected() {
    let r = rig_seg(
        Config::default(),
        stt("Ship it."),
        None,
        None,
        Arc::new(Scripted(SCRIPT)),
    );
    r.app.on_gesture(Gesture::Press);
    for _ in SCRIPT {
        r.mic.speak(1600);
        std::thread::sleep(ochre::live::MIN_INTERVAL * 3); // let each one through the coalescer
    }
    wait(|| partials(&r).len() == SCRIPT.len());
    r.app.on_gesture(Gesture::Release);
    wait(|| r.app.state() == State::Idle && result_of(&r).is_some());
    // In order, hesitations stripped so "um" never flashes in the HUD.
    assert_eq!(
        partials(&r),
        vec!["Ship", "Ship it after", "Ship it after the tests,"]
    );
    // Only the final transcript reaches the injector; nothing live comes after the result.
    assert_eq!(*r.inj.typed.lock(), vec!["Ship it. "]);
    let ev = r.events.lock();
    let result_at = ev
        .iter()
        .position(|e| matches!(e, Event::Result { .. }))
        .unwrap();
    assert!(
        !ev[result_at..]
            .iter()
            .any(|e| matches!(e, Event::Partial { .. }))
    );
}

#[test]
fn partials_off_emits_none() {
    let mut cfg = Config::default();
    cfg.ui.show_partials = false;
    let r = rig_seg(cfg, stt("Quiet."), None, None, Arc::new(Scripted(SCRIPT)));
    r.app.on_gesture(Gesture::Press);
    r.mic.speak(1600);
    r.mic.speak(1600);
    r.app.on_gesture(Gesture::Release);
    wait(|| r.app.state() == State::Idle && result_of(&r).is_some());
    std::thread::sleep(ochre::live::MIN_INTERVAL * 2);
    assert!(partials(&r).is_empty());
    assert_eq!(*r.inj.typed.lock(), vec!["Quiet. "]);
}

#[test]
fn cancel_stops_live_text() {
    let r = rig_seg(
        Config::default(),
        stt("x"),
        None,
        None,
        Arc::new(Scripted(SCRIPT)),
    );
    r.app.on_gesture(Gesture::Press);
    r.mic.speak(1600);
    wait(|| partials(&r).len() == 1);
    r.app.on_gesture(Gesture::Cancel);
    let n = r.events.lock().len();
    r.mic.speak(1600); // the mic sink is gone; nothing may repaint the HUD
    std::thread::sleep(ochre::live::MIN_INTERVAL * 3);
    assert!(
        !r.events.lock()[n..]
            .iter()
            .any(|e| matches!(e, Event::Partial { .. }))
    );
    assert!(r.inj.typed.lock().is_empty());
}

/// Uppercases and counts its calls.
struct Counting(Arc<std::sync::atomic::AtomicUsize>);
impl Refiner for Counting {
    fn info(&self) -> EngineInfo {
        info("counting", "local")
    }
    fn load(&mut self, _: ochre_core::stt::ProgressFn) -> Result<()> {
        Ok(())
    }
    fn refine(&self, text: &str, _: &RefineContext, _: Duration) -> Result<String> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(text.to_uppercase())
    }
}

fn notices(r: &Rig) -> Vec<String> {
    r.events
        .lock()
        .iter()
        .filter_map(|e| match e {
            Event::Notice { message } => Some(message.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn paste_last_retypes_the_final_text_without_refining_or_saving() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let db = std::env::temp_dir().join(format!("ochre-paste-last-{}.sqlite3", std::process::id()));
    let _ = std::fs::remove_file(&db);
    let refines = Arc::new(AtomicUsize::new(0));
    let r = rig_full(
        Config::default(),
        stt("make it loud"),
        None,
        Some(Arc::new(Counting(refines.clone()))),
        Arc::new(OnePhrase),
        History::open(&db).unwrap(),
    );
    let rows = || History::open(&db).unwrap().query("", 50).unwrap().len();

    // Nothing dictated yet: a notice, nothing typed.
    r.app.on_gesture(Gesture::PasteLast);
    std::thread::sleep(Duration::from_millis(30));
    assert!(r.inj.typed.lock().is_empty());
    assert!(notices(&r).iter().any(|n| n == "Nothing to paste yet"));

    r.app.on_gesture(Gesture::Press);
    r.mic.speak(8000);
    r.app.on_gesture(Gesture::Release);
    wait(|| r.app.state() == State::Idle && rows() == 1);
    assert_eq!(*r.inj.typed.lock(), vec!["MAKE IT LOUD "]);
    assert_eq!(refines.load(Ordering::SeqCst), 1);
    let results = |r: &Rig| {
        r.events
            .lock()
            .iter()
            .filter(|e| matches!(e, Event::Result { .. }))
            .count()
    };

    // The chord and the tray / CLI command both type the refined text again, as typed.
    r.app.on_gesture(Gesture::PasteLast);
    wait(|| r.inj.typed.lock().len() == 2);
    r.app.command(ochre_core::events::Command::PasteLast);
    wait(|| r.inj.typed.lock().len() == 3);
    assert_eq!(r.inj.typed.lock()[1], "MAKE IT LOUD ");
    assert_eq!(r.inj.typed.lock()[2], "MAKE IT LOUD ");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(refines.load(Ordering::SeqCst), 1, "never re-refined");
    assert_eq!(rows(), 1, "no new history entry");
    assert_eq!(results(&r), 1, "no new result event");

    // While a take is live, pasting would land in the middle of it: refused.
    r.app.on_gesture(Gesture::Press);
    r.app.on_gesture(Gesture::PasteLast);
    assert!(
        notices(&r)
            .iter()
            .any(|n| n == "Finish the current dictation first")
    );
    // The chord's own order: Abort (silent) first, then PasteLast.
    r.app.on_gesture(Gesture::Abort);
    r.app.on_gesture(Gesture::PasteLast);
    wait(|| r.inj.typed.lock().len() == 4);
    assert_eq!(rows(), 1);
    drop(r);

    // A fresh start (nothing in memory) falls back to the newest history entry.
    let r2 = rig_full(
        Config::default(),
        stt("unused"),
        None,
        None,
        Arc::new(OnePhrase),
        History::open(&db).unwrap(),
    );
    assert_eq!(r2.app.last_transcript().as_deref(), Some("MAKE IT LOUD"));
    r2.app.command(ochre_core::events::Command::PasteLast);
    wait(|| r2.inj.typed.lock().len() == 1);
    assert_eq!(r2.inj.typed.lock()[0], "MAKE IT LOUD ");
    drop(r2);
    let _ = std::fs::remove_file(&db);
}

#[test]
fn paste_last_works_with_history_disabled() {
    let r = rig(Config::default(), stt("Kept in memory."), None, None);
    r.app.on_gesture(Gesture::Press);
    r.mic.speak(8000);
    r.app.on_gesture(Gesture::Release);
    wait(|| r.app.state() == State::Idle && r.inj.typed.lock().len() == 1);
    r.app.on_gesture(Gesture::PasteLast);
    wait(|| r.inj.typed.lock().len() == 2);
    assert_eq!(r.inj.typed.lock()[1], "Kept in memory. ");
    // Pasting makes no sound of its own.
    assert_eq!(*r.cues.lock(), vec![Cue::Start, Cue::Stop]);
}

/// One push-to-talk dictation; waits until `n` insertions have been typed and the app is idle.
fn dictate(r: &Rig, n: usize) {
    r.app.on_gesture(Gesture::Press);
    r.mic.speak(8000);
    r.app.on_gesture(Gesture::Release);
    wait(|| r.inj.typed.lock().len() == n && r.app.state() == State::Idle);
}

const LONG: &str = "We shipped the first build to the staging servers this morning. \
    Maria checked the login flow and the settings page on her phone. \
    The search results still load slowly when the filters are applied. \
    Tom thinks the database index on the orders table is missing again. \
    New paragraph. The release notes need a short section about the new export button. \
    Please send me your comments on the draft before lunch tomorrow. \
    We will decide on the final release date at the Thursday meeting. \
    Bring the latest crash numbers from the dashboard to that meeting.";

#[test]
fn long_dictations_are_refined_in_pieces_when_chunking_is_on() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = Arc::new(AtomicUsize::new(0));
    let mut cfg = Config::default();
    cfg.refine.chunk_long = "on".into();
    cfg.inject.trailing_space = false;
    let r = rig_full(
        cfg,
        stt(LONG),
        None,
        Some(Arc::new(Counting(calls.clone()))),
        Arc::new(OnePhrase),
        History::disabled(),
    );
    dictate(&r, 1);
    let pieces = ochre_refine::chunk::split(LONG, 80, ochre_refine::chunk::TARGET_WORDS);
    assert_eq!(pieces.len(), 2, "{pieces:?}"); // one per paragraph (45 words each)
    assert_eq!(calls.load(Ordering::SeqCst), pieces.len());
    // Each piece refined on its own, the voice command turned into the paragraph break.
    let typed = r.inj.typed.lock()[0].clone();
    let (first, second) = typed.split_once("\n\n").expect("paragraph break");
    assert!(first.starts_with("WE SHIPPED") && first.ends_with("MISSING AGAIN."));
    assert!(second.starts_with("THE RELEASE NOTES") && second.ends_with("THAT MEETING."));
    assert!(!typed.contains("NEW PARAGRAPH"));
}

#[test]
fn chunking_auto_leaves_unknown_and_cloud_refiners_alone() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = Arc::new(AtomicUsize::new(0));
    let r = rig_full(
        Config::default(), // chunk_long = "auto"; this refiner names no bundled model
        stt(LONG),
        None,
        Some(Arc::new(Counting(calls.clone()))),
        Arc::new(OnePhrase),
        History::disabled(),
    );
    dictate(&r, 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn saying_it_again_after_a_cleanup_types_it_as_heard() {
    let db = std::env::temp_dir().join(format!("ochre-redictate-{}.sqlite3", std::process::id()));
    let _ = std::fs::remove_file(&db);
    let mut cfg = Config::default();
    cfg.inject.trailing_space = false;
    let r = rig_full(
        cfg,
        stt("make it loud"),
        None,
        Some(Arc::new(Upper(false))),
        Arc::new(OnePhrase),
        History::open(&db).unwrap(),
    );
    dictate(&r, 1);
    assert_eq!(r.inj.typed.lock()[0], "MAKE IT LOUD");
    // Same words again within the window, and the cleanup had changed them: typed as heard,
    // and only inserted (the earlier text is never deleted).
    dictate(&r, 2);
    assert_eq!(r.inj.typed.lock()[1], " make it loud");
    assert!(
        notices(&r)
            .iter()
            .any(|n| n == "Said again, so typed as heard")
    );
    let rows = || History::open(&db).unwrap().query("", 10).unwrap();
    wait(|| rows().len() == 2);
    let h = rows();
    assert_eq!(
        (
            h[0].text.as_str(),
            h[0].note.as_str(),
            h[0].refiner.as_str()
        ),
        ("make it loud", "redictation", "")
    );
    assert_eq!(
        (h[1].text.as_str(), h[1].note.as_str()),
        ("MAKE IT LOUD", "")
    );
    // The raw insert changed nothing, so a third take is refined again.
    dictate(&r, 3);
    assert_eq!(r.inj.typed.lock()[2], " MAKE IT LOUD");
    drop(r);
    let _ = std::fs::remove_file(&db);
}

#[test]
fn redictation_rule_can_be_turned_off_and_expires() {
    for (on, window) in [(false, 60.0), (true, 0.0)] {
        let mut cfg = Config::default();
        cfg.inject.trailing_space = false;
        cfg.refine.redictation_raw = on;
        cfg.refine.redictation_window_s = window;
        let r = rig(cfg, stt("make it loud"), None, Some(Upper(false)));
        dictate(&r, 1);
        dictate(&r, 2);
        assert_eq!(
            *r.inj.typed.lock(),
            vec!["MAKE IT LOUD", " MAKE IT LOUD"],
            "{on} {window}"
        );
    }
}

#[test]
fn an_unchanged_dictation_said_again_is_refined_as_usual() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let calls = Arc::new(AtomicUsize::new(0));
    // Counting uppercases; already-uppercase text comes back unchanged.
    let r = rig_full(
        Config::default(),
        stt("OK SHIP IT"),
        None,
        Some(Arc::new(Counting(calls.clone()))),
        Arc::new(OnePhrase),
        History::disabled(),
    );
    dictate(&r, 1);
    dictate(&r, 2);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn a_restart_said_three_times_is_typed_once() {
    let r = rig(
        Config::default(),
        stt("The 2.8B long. The 2.8B long. The 2.8B long."),
        None,
        None,
    );
    dictate(&r, 1);
    assert_eq!(r.inj.typed.lock()[0], "The 2.8B long. ");
}
