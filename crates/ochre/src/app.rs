//! Session state machine. Threads:
//! - hotkey / UI / CLI callers: only flip state and enqueue work; they never block;
//! - the Mic's reader thread: feeds audio into the live segment session;
//! - one session worker: runs the tail (finish decodes -> refine -> history -> inject);
//! - a loader thread: (re)loads engines at startup and on settings changes;
//! - while hands-free is on, the wake listener (`ochre-wake`, see handsfree.rs).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use crossbeam_channel::{Sender, unbounded};
use parking_lot::{Condvar, Mutex};
use tracing::{info, warn};

use ochre_core::config::Config;
use ochre_core::events::{Bus, Command, Event, State, Timings, Trigger};
use ochre_core::history::{History, NewEntry};
use ochre_core::platform::{FocusInfo, Gesture, Injector};
use ochre_core::refine::{RefineContext, Refiner};
use ochre_core::stt::{Progress, SttEngine, SttOptions, SttResult};
use ochre_core::{Result, secrets};
use ochre_refine::chunk;

use crate::handsfree::{Ctl, WakeFactory, WakeHandle, WakeSlot};
use crate::live::{LiveSession, LiveText};
use crate::seams::{Mic, PartialFn, PhraseFn, SegmentSession, SegmenterFactory, TranscribeFn};
use crate::stream::StreamSession;
use crate::text::{prepass, same_words_key, strip_hesitations};

/// Audible cues; the binary maps them to earcons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cue {
    Start,
    Stop,
    Cancel,
    Error,
}

/// Everything a session needs, built by the `Loader` from the config. Swapped atomically on
/// settings changes, so a running session keeps the engines it started with.
pub struct Engines {
    pub stt: Arc<dyn SttEngine>,
    /// Local engine used when a cloud engine fails (only if its model is already on disk).
    pub fallback_stt: Option<Arc<dyn SttEngine>>,
    pub refiner: Option<Arc<dyn Refiner>>,
    pub injector: Arc<dyn Injector>,
    pub mic: Arc<dyn Mic>,
    pub segmenter: Arc<dyn SegmenterFactory>,
    /// Safety net over refiner output: returns `raw` when the refined text looks wrong.
    pub guard: fn(raw: &str, refined: &str, mode: &str) -> String,
    pub cue: Option<Arc<dyn Fn(Cue) + Send + Sync>>,
}

pub type Loader =
    Arc<dyn Fn(&Config, &(dyn Fn(Progress) + Send + Sync)) -> Result<Engines> + Send + Sync>;

type Job = Box<dyn FnOnce() + Send>;
type RecordingHook = Arc<dyn Fn(bool) + Send + Sync>;

struct Session {
    id: String,
    trigger: Trigger,
    focus: FocusInfo,
    engines: Arc<Engines>,
    seg: Arc<Mutex<Option<Box<dyn SegmentSession>>>>,
    /// Live HUD text (display only); closed once the text heads for insertion, or on cancel.
    live: Option<LiveSession>,
    raw: bool,
    send: bool,
    ending: bool,
    /// Hands-free: the controller's session id and the wake thread's control channel.
    hf: Option<(u64, Sender<Ctl>)>,
}

struct St {
    state: State,
    session: Option<Session>,
}

/// The previous dictation, for the re-dictation rule: the same words said again soon after Ochre
/// changed them means the cleanup got them wrong.
struct LastDictation {
    at: Instant,
    /// `same_words_key` of its raw text (after hesitations, repeats and dictionary).
    key: String,
    /// What was typed differed from the raw text.
    changed: bool,
}

/// Remembers the last insertion so consecutive dictations into the same field join with a space.
#[derive(Default)]
struct JoinMemory {
    window_id: String,
    ended_with_space: bool,
    at: Option<Instant>,
}

struct Shared {
    bus: Bus,
    /// Coalescing, non-blocking pump for `Event::Partial` (see live.rs).
    live: LiveText,
    cfg: Mutex<Config>,
    /// Where settings changes are saved. None (tests, embedders) never touches the user's file.
    cfg_path: Mutex<Option<std::path::PathBuf>>,
    engines: Mutex<Option<Arc<Engines>>>,
    history: Arc<History>,
    /// History rows and the timing log are written here, after injection (docs/latency.md
    /// rule 6: nothing but typing sits between the decode and the text appearing).
    persist: Sender<Job>,
    loader: Loader,
    st: Mutex<St>,
    join: Mutex<JoinMemory>,
    /// The final text of the last dictation (what was typed, or would have been), for
    /// "paste last transcript". Kept in memory so it works with history turned off.
    last_text: Mutex<Option<String>>,
    last_dictation: Mutex<Option<LastDictation>>,
    worker: Sender<Job>,
    /// Tells the hotkey layer a session is live (so Escape is swallowed only while recording).
    recording_hook: Mutex<Option<RecordingHook>>,
    quit: (Mutex<bool>, Condvar),
    /// The wake listener thread (hands-free), when enabled and running.
    wake: WakeSlot,
    wake_factory: Mutex<WakeFactory>,
    /// The wake detector is loaded, tapped into the mic and not paused for a call.
    armed: AtomicBool,
}

#[derive(Clone)]
pub struct App {
    s: Arc<Shared>,
}

/// What the wake thread holds, so it never keeps the app alive.
#[derive(Clone)]
pub(crate) struct WeakApp(Weak<Shared>);

impl WeakApp {
    pub(crate) fn upgrade(&self) -> Option<App> {
        self.0.upgrade().map(|s| App { s })
    }
}

impl App {
    pub fn new(cfg: Config, bus: Bus, history: History, loader: Loader) -> Self {
        let (tx, rx) = unbounded::<Job>();
        let (ptx, prx) = unbounded::<Job>();
        std::thread::Builder::new()
            .name("ochre-persist".into())
            .spawn(move || {
                for job in prx {
                    job();
                }
            })
            .expect("spawn persist worker");
        std::thread::Builder::new()
            .name("ochre-session".into())
            .spawn(move || {
                for job in rx {
                    job();
                }
            })
            .expect("spawn session worker");
        Self {
            s: Arc::new(Shared {
                live: LiveText::new(bus.clone()),
                bus,
                cfg: Mutex::new(cfg),
                cfg_path: Mutex::new(None),
                engines: Mutex::new(None),
                history: Arc::new(history),
                persist: ptx,
                loader,
                st: Mutex::new(St {
                    state: State::Loading,
                    session: None,
                }),
                join: Mutex::new(JoinMemory::default()),
                last_text: Mutex::new(None),
                last_dictation: Mutex::new(None),
                worker: tx,
                recording_hook: Mutex::new(None),
                quit: (Mutex::new(false), Condvar::new()),
                wake: WakeSlot::default(),
                wake_factory: Mutex::new(crate::handsfree::default_factory()),
                armed: AtomicBool::new(false),
            }),
        }
    }

    /// Save settings changes to `path` (the app passes the user's config file). Without this,
    /// changes stay in memory: a test run must never overwrite the real config.
    pub fn persist_config_to(&self, path: std::path::PathBuf) {
        *self.s.cfg_path.lock() = Some(path);
    }

    /// Replace how the hands-free controller is built (tests use a fake detector). Takes effect
    /// the next time the wake listener starts.
    pub fn set_wake_factory(&self, f: WakeFactory) {
        *self.s.wake_factory.lock() = f;
    }

    /// The wake detector is actually listening (loaded, tapped into the mic, not paused).
    pub fn handsfree_armed(&self) -> bool {
        self.s.armed.load(Ordering::SeqCst)
    }

    /// The wake listener thread is alive.
    pub fn wake_running(&self) -> bool {
        self.s
            .wake
            .handle
            .lock()
            .as_ref()
            .is_some_and(WakeHandle::running)
    }

    /// The binary installs this after starting the hotkey listener.
    pub fn set_recording_hook(&self, f: impl Fn(bool) + Send + Sync + 'static) {
        *self.s.recording_hook.lock() = Some(Arc::new(f));
    }

    fn recording(&self, on: bool) {
        if let Some(f) = self.s.recording_hook.lock().clone() {
            f(on);
        }
    }

    pub fn bus(&self) -> &Bus {
        &self.s.bus
    }

    pub fn config(&self) -> Config {
        self.s.cfg.lock().clone()
    }

    pub fn state(&self) -> State {
        self.s.st.lock().state
    }

    // ------------------------------------------------------------------ lifecycle

    /// Emit hello + config, then load engines on a background thread (Loading -> Idle).
    pub fn start(&self) {
        self.s.bus.emit(Event::Hello {
            version: env!("CARGO_PKG_VERSION").into(),
            platform: std::env::consts::OS.into(),
        });
        self.emit_config();
        self.reload("Loading models…");
    }

    fn reload(&self, detail: &str) {
        self.set_state(State::Loading, None, detail);
        let app = self.clone();
        std::thread::Builder::new()
            .name("ochre-load".into())
            .spawn(move || {
                let cfg = app.config();
                let bus = app.s.bus.clone();
                let progress = move |p: Progress| bus.emit(Event::Download { item: p.item, done: p.done, total: p.total });
                let t0 = Instant::now();
                match (app.s.loader)(&cfg, &progress) {
                    Ok(engines) => {
                        info!(ms = t0.elapsed().as_millis() as u64, stt = %engines.stt.info().id, "engines ready");
                        *app.s.engines.lock() = Some(Arc::new(engines));
                        app.set_state(State::Idle, None, "");
                        app.sync_wake();
                    }
                    Err(e) => app.fail(&format!("Setup failed: {e}"), e.code()),
                }
            })
            .expect("spawn loader");
    }

    pub fn request_quit(&self) {
        *self.s.quit.0.lock() = true;
        self.s.quit.1.notify_all();
    }

    /// Blocks until a Quit command (headless CLI main loop).
    pub fn wait_for_quit(&self) {
        let mut q = self.s.quit.0.lock();
        while !*q {
            self.s.quit.1.wait(&mut q);
        }
    }

    /// A dictation is probably about to start (a chord's first modifier): open a released mic.
    pub fn prime_mic(&self) {
        if let Some(e) = self.s.engines.lock().clone() {
            e.mic.prime();
        }
    }

    /// Reopen the microphone (macOS: a stream opened before the grant may deliver silence).
    pub fn reopen_mic(&self) {
        if let Some(e) = self.s.engines.lock().clone() {
            e.mic.reopen();
        }
    }

    pub fn shutdown(&self) {
        self.stop_wake();
        self.cancel(false);
        *self.s.engines.lock() = None;
    }

    // ------------------------------------------------------------------ hands-free

    /// Start, restart or stop the wake listener to match the config and the current mic.
    fn sync_wake(&self) {
        let _serial = self.s.wake.sync.lock();
        let cfg = self.config();
        let engines = self.s.engines.lock().clone();
        let want = engines.as_ref().filter(|_| cfg.handsfree.enabled);
        let keep = match (&*self.s.wake.handle.lock(), want) {
            (Some(h), Some(e)) => h.running() && h.matches(&cfg, &e.mic),
            _ => false,
        };
        if keep {
            return;
        }
        let old = self.s.wake.handle.lock().take();
        if let Some(old) = old {
            old.stop(); // joins; the thread cancels a running hands-free session
        }
        self.set_armed(false);
        if let Some(e) = want {
            let factory = self.s.wake_factory.lock().clone();
            let h = WakeHandle::spawn(
                WeakApp(Arc::downgrade(&self.s)),
                cfg,
                e.mic.clone(),
                factory,
            );
            *self.s.wake.handle.lock() = Some(h);
        }
    }

    fn stop_wake(&self) {
        let _serial = self.s.wake.sync.lock();
        let old = self.s.wake.handle.lock().take();
        if let Some(old) = old {
            old.stop();
        }
        self.set_armed(false);
    }

    /// Wake thread: the detector is (not) listening. Re-emits the state when it changes.
    pub(crate) fn set_armed(&self, armed: bool) {
        if self.s.armed.swap(armed, Ordering::SeqCst) == armed {
            return;
        }
        // Re-announce the current state with the new flag (never write the state from here: a
        // session may be starting on another thread).
        let ev = match self.s.bus.last_state() {
            Some(Event::State {
                state,
                trigger,
                detail,
                ..
            }) => Event::State {
                state,
                trigger,
                handsfree_armed: armed,
                detail,
            },
            _ => Event::State {
                state: self.state(),
                trigger: None,
                handsfree_armed: armed,
                detail: String::new(),
            },
        };
        self.s.bus.emit(ev);
    }

    pub(crate) fn wake_failed(&self, message: &str) {
        warn!("{message}");
        self.set_armed(false);
        self.notice(message);
    }

    pub(crate) fn notice_msg(&self, message: &str) {
        self.notice(message);
    }

    /// Wake thread: may the detector listen now? (no session of any kind, nothing in flight)
    pub(crate) fn idle_for_wake(&self) -> bool {
        let st = self.s.st.lock();
        st.session.is_none() && matches!(st.state, State::Idle | State::Error)
    }

    /// Wake thread: the wake word was heard. Start a `Trigger::Wake` session whose audio begins at
    /// absolute sample `from` of the mic tap (the wake word's first syllable), so nothing said
    /// right after it is lost. False if the app is busy.
    pub(crate) fn begin_wake(&self, hf_id: u64, from: u64, ctl: Sender<Ctl>) -> bool {
        self.begin_inner(Trigger::Wake, Some((hf_id, from, ctl)))
    }

    fn current_hf(&self) -> Option<u64> {
        self.s
            .st
            .lock()
            .session
            .as_ref()
            .and_then(|s| s.hf.as_ref().map(|h| h.0))
    }

    /// Wake thread: hands-free session `hf_id` ended with text to insert (stop, send, idle).
    pub(crate) fn wake_finish(&self, hf_id: u64, send: bool) {
        if self.current_hf() == Some(hf_id) {
            self.end(send, false);
        }
    }

    /// Wake thread: hands-free session `hf_id` was canceled or was a false wake.
    pub(crate) fn wake_cancel(&self, hf_id: u64, audible: bool) {
        if self.current_hf() == Some(hf_id) {
            self.cancel(audible);
        }
    }

    // ------------------------------------------------------------------ triggers

    /// Called on the hotkey thread: returns immediately.
    pub fn on_gesture(&self, g: Gesture) {
        match g {
            Gesture::Press => {
                self.begin(Trigger::Hotkey);
            }
            Gesture::Lock => {
                let st = self.s.st.lock();
                if st.session.is_some() && st.state == State::Recording {
                    drop(st);
                    self.set_state(State::Locked, Some(Trigger::Hotkey), "");
                }
            }
            Gesture::Release | Gesture::Finish => self.end(false, false),
            Gesture::FinishRaw => self.end(false, true),
            Gesture::Abort => self.cancel(false),
            Gesture::Cancel => self.cancel(true),
            Gesture::PasteLast => self.paste_last(),
        }
    }

    /// Type the last transcript again into the focused field, through the normal injector
    /// (long text uses the paste fallback, which restores the clipboard). No refinement, no new
    /// history row. Returns immediately; the typing runs on the session worker.
    pub fn paste_last(&self) {
        let Some(engines) = self.s.engines.lock().clone() else {
            self.notice("Still loading the speech model, one moment");
            return;
        };
        if self.s.st.lock().session.is_some() {
            self.notice("Finish the current dictation first");
            return;
        }
        let Some(text) = self.last_transcript() else {
            self.notice("Nothing to paste yet");
            return;
        };
        let app = self.clone();
        let _ = self.s.worker.send(Box::new(move || {
            let cfg = app.config();
            let t0 = Instant::now();
            if app.insert(&engines, &cfg, &text, false) && cfg.debug.log_timings {
                info!(
                    ms = ms(t0.elapsed()),
                    chars = text.len(),
                    "pasted last transcript"
                );
            }
        }));
    }

    /// The most recent final text: this run's last dictation, else the newest history row.
    pub fn last_transcript(&self) -> Option<String> {
        let mem = self.s.last_text.lock().clone();
        mem.or_else(|| {
            self.s
                .history
                .query("", 1)
                .ok()?
                .into_iter()
                .next()
                .map(|e| e.text)
        })
        .filter(|t| !t.trim().is_empty())
    }

    fn notice(&self, message: &str) {
        self.s.bus.emit(Event::Notice {
            message: message.into(),
        });
    }

    /// Start a session. Returns false if busy, loading, or the mic could not start.
    pub fn begin(&self, trigger: Trigger) -> bool {
        self.begin_inner(trigger, None)
    }

    fn begin_inner(&self, trigger: Trigger, wake: Option<(u64, u64, Sender<Ctl>)>) -> bool {
        let Some(engines) = self.s.engines.lock().clone() else {
            // Never ignore a key press silently (docs/latency.md rule 8).
            self.s.bus.emit(Event::Notice {
                message: "Still loading the speech model, one moment".into(),
            });
            return false;
        };
        let mut st = self.s.st.lock();
        if st.session.is_some() || !matches!(st.state, State::Idle | State::Error) {
            return false;
        }
        let cfg = self.s.cfg.lock().clone();
        engines.stt.prewarm();
        if let Some(r) = &engines.refiner {
            r.prewarm();
        }
        let focus = engines.injector.focus();
        let transcribe = self.transcribe_fn(&engines, &cfg);
        // Every finished phrase also goes to the hands-free controller (control phrases, false
        // wakes, scratch that); the final text comes back from it at the tail. Live-text previews
        // must not: the controller would keep each one as another phrase.
        let on_phrase: Option<PhraseFn> = wake.as_ref().map(|(hf_id, _, ctl)| {
            let (ctl, hf_id) = (ctl.clone(), *hf_id);
            Arc::new(move |res: &SttResult| {
                if res.text.trim().is_empty() {
                    return; // the speech gate skipped it: nothing was said
                }
                let _ = ctl.send(Ctl::Phrase {
                    hf_id,
                    text: strip_hesitations(&res.text),
                });
            }) as PhraseFn
        });
        // Streaming engines (Soniox real-time, gpt-live-transcribe, Gemini Live) send audio while
        // the user talks and bring their own live partials; the rest go through the segmenter.
        // Partials are display only and never block the decode thread: the pump in live.rs
        // strips hesitations, coalesces to ~30 Hz and emits them. Batch cloud engines get none
        // (each preview would be another paid request).
        // Hands-free always segments: control phrases are found per phrase transcript.
        let streaming = engines.stt.streaming() && wake.is_none();
        let live = (cfg.ui.show_partials && (streaming || engines.stt.info().kind == "local"))
            .then(|| self.s.live.open());
        let mut partial: Option<PartialFn> = live.as_ref().map(LiveSession::partial_fn);
        if let (Some(inner), true) = (partial.clone(), wake.is_some()) {
            // The live text starts with the wake word; show what will be typed instead.
            let phrase = cfg.handsfree.phrase.clone();
            partial = Some(Arc::new(move |text: &str, stable: usize| {
                let shown = ochre_wake::commands::strip_leading_phrase(text, &phrase, &[], 1);
                let cut = text.len().saturating_sub(shown.len());
                let mut stable = stable.saturating_sub(cut).min(shown.len());
                while !shown.is_char_boundary(stable) {
                    stable -= 1;
                }
                inner(&shown, stable);
            }));
        }
        let session: Box<dyn SegmentSession> = if streaming {
            Box::new(StreamSession::start(
                engines.stt.clone(),
                stt_options(&cfg),
                partial,
                transcribe,
            ))
        } else {
            match on_phrase {
                Some(f) => engines.segmenter.start_with_phrases(transcribe, partial, f),
                None => engines.segmenter.start(transcribe, partial),
            }
        };
        let seg = Arc::new(Mutex::new(Some(session)));
        let sink_seg = seg.clone();
        let bus = self.s.bus.clone();
        let level = Arc::new(move |rms: f32| bus.emit(Event::Level { rms }));
        let preroll_ms = if trigger == Trigger::Wake {
            (cfg.handsfree.preroll_s * 1000.0) as u32
        } else {
            150
        };
        let sink = Box::new(move |pcm: &[f32]| {
            if let Some(s) = sink_seg.lock().as_mut() {
                s.feed(pcm);
            }
        });
        let started = match &wake {
            Some((_, from, _)) => engines.mic.begin_from(*from, sink, level),
            None => engines.mic.begin(preroll_ms, sink, level),
        };
        if let Err(e) = started {
            drop(st);
            if let Some(l) = &live {
                l.close();
            }
            if let Some(s) = seg.lock().take() {
                s.cancel();
            }
            self.fail(&format!("Could not start the microphone: {e}"), e.code());
            return false;
        }
        st.session = Some(Session {
            id: new_id(),
            trigger,
            focus,
            engines: engines.clone(),
            seg,
            live,
            raw: false,
            send: false,
            ending: false,
            hf: wake.map(|(id, _, ctl)| (id, ctl)),
        });
        drop(st);
        self.recording(true);
        self.set_state(
            if trigger == Trigger::Wake {
                State::Handsfree
            } else {
                State::Recording
            },
            Some(trigger),
            "",
        );
        cue(&engines, Cue::Start);
        true
    }

    /// Finish the session: flush audio, then hand the tail to the session worker.
    pub fn end(&self, send: bool, raw: bool) {
        let released = Instant::now();
        let (id, engines, seg) = {
            let mut st = self.s.st.lock();
            let Some(sess) = st.session.as_mut() else {
                return;
            };
            if sess.ending {
                return;
            }
            sess.ending = true;
            sess.send = send;
            sess.raw = raw;
            (sess.id.clone(), sess.engines.clone(), sess.seg.clone())
        };
        // Flushes the last captured audio into the segment session (outside our state lock).
        self.recording(false);
        let audio_ms = engines.mic.end().unwrap_or(0);
        if let Some(s) = seg.lock().as_mut() {
            s.release();
        }
        self.set_state(State::Transcribing, None, "");
        cue(&engines, Cue::Stop);
        let app = self.clone();
        let _ = self
            .s
            .worker
            .send(Box::new(move || app.tail(&id, seg, released, audio_ms)));
    }

    pub fn cancel(&self, audible: bool) {
        let sess = self.s.st.lock().session.take();
        let Some(sess) = sess else { return };
        if let Some((hf_id, ctl)) = &sess.hf {
            let _ = ctl.send(Ctl::Cancel { hf_id: *hf_id });
        }
        if let Some(l) = &sess.live {
            l.close();
        }
        self.recording(false);
        sess.engines.mic.cancel();
        if let Some(s) = sess.seg.lock().take() {
            s.cancel();
        }
        self.set_state(State::Idle, None, "");
        if audible {
            cue(&sess.engines, Cue::Cancel);
        }
    }

    // ------------------------------------------------------------------ the tail

    fn tail(
        &self,
        id: &str,
        seg: Arc<Mutex<Option<Box<dyn SegmentSession>>>>,
        released: Instant,
        audio_ms: u64,
    ) {
        // Post-roll: the mic keeps feeding the session for ~150-300 ms after release while the
        // tail decode (started by release()) runs; wait for it to close before finishing.
        let live = self
            .s
            .st
            .lock()
            .session
            .as_ref()
            .filter(|s| s.id == id)
            .map(|s| s.engines.clone());
        if let Some(e) = live {
            e.mic.wait_closed(Duration::from_millis(600));
        }
        let Some(seg) = seg.lock().take() else { return }; // canceled
        let decoded = seg.finish();
        let stt_tail_ms = ms(released.elapsed());
        let (trigger, focus, engines, mut send, raw, hf) = {
            let st = self.s.st.lock();
            match st.session.as_ref().filter(|s| s.id == id) {
                Some(s) => {
                    // Every phrase is decoded: the live text is done (it is never inserted).
                    if let Some(l) = &s.live {
                        l.close();
                    }
                    (
                        s.trigger,
                        s.focus.clone(),
                        s.engines.clone(),
                        s.send,
                        s.raw,
                        s.hf.clone(),
                    )
                }
                None => return, // canceled while decoding
            }
        };
        let results = match decoded {
            Ok(r) => r,
            Err(e) => {
                return self
                    .finish_session(id, Some((format!("Transcription failed: {e}"), e.code())));
            }
        };
        let stt_total_ms: u64 = results.iter().map(|r| r.processing_ms).sum();
        let joined = match &hf {
            // Hands-free: the controller's text (wake word, control phrases and scratched
            // phrases removed), once it has seen every phrase decoded above.
            Some((hf_id, ctl)) => match crate::handsfree::close(ctl, *hf_id, send) {
                Some(end) if end.insert() => {
                    send |= end.press_enter();
                    end.text
                }
                _ => return self.finish_session(id, None), // canceled, false wake, nothing said
            },
            None => results
                .iter()
                .map(|r| r.text.trim())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
        };
        let cfg = self.s.cfg.lock().clone();
        let raw_text = prepass(&joined, &cfg.dictionary);
        if raw_text.trim().is_empty() {
            self.s.bus.emit(Event::Notice {
                message: "Didn't catch that".into(),
            });
            return self.finish_session(id, None);
        }

        let key = same_words_key(&raw_text);
        let redictated = !raw && engines.refiner.is_some() && self.redictated(&cfg, &key);
        if redictated {
            // Replacing the earlier text as well was considered and left out: nothing here can
            // prove the field still ends with it (docs/refine-chunking.md, "Re-dictation").
            info!("same words dictated again after a cleanup changed them: typing them as heard");
            self.notice("Said again, so typed as heard");
        }
        let t_refine = Instant::now();
        let (text, refined, chunks) = if raw || redictated {
            (raw_text.clone(), false, 0)
        } else {
            self.refine(&engines, &cfg, &raw_text, &focus)
        };
        let refine_ms = if refined || engines.refiner.is_some() && !raw && !redictated {
            ms(t_refine.elapsed())
        } else {
            0
        };
        *self.s.last_dictation.lock() = Some(LastDictation {
            at: Instant::now(),
            key,
            changed: text != raw_text,
        });

        self.set_state(State::Inserting, None, "");
        let t_inject = Instant::now();
        let inserted = self.insert(&engines, &cfg, &text, send);
        let inject_ms = ms(t_inject.elapsed());
        *self.s.last_text.lock() = Some(text.clone());
        let timings = Timings {
            audio_ms,
            stt_tail_ms,
            stt_total_ms,
            refine_ms,
            inject_ms,
            release_to_insert_ms: ms(released.elapsed()),
        };
        if cfg.debug.log_timings {
            info!(?trigger, ?timings, chars = text.len(), "dictation");
        }
        // Persist after the text is on screen; the Result event below also carries it to the UI.
        let refiner_id = if refined {
            engines
                .refiner
                .as_ref()
                .map(|r| r.info().id)
                .unwrap_or_default()
        } else {
            String::new()
        };
        let row = (
            raw_text.clone(),
            text.clone(),
            focus.app_name.clone(),
            engines.stt.info().id,
            refiner_id,
        );
        let note = if redictated { "redictation" } else { "" };
        let history = self.s.history.clone();
        let log_line = cfg.debug.log_timings.then(|| {
            serde_json::json!({"trigger": trigger, "timings": &timings, "chars": text.chars().count(),
                               "stt": &row.3, "refiner": &row.4, "chunks": chunks, "note": note})
        });
        let _ = self.s.persist.send(Box::new(move || {
            let (raw, text, app, stt, refiner) = row;
            match history.add(NewEntry {
                raw: &raw,
                text: &text,
                app: &app,
                stt: &stt,
                refiner: &refiner,
                duration_ms: audio_ms,
                note,
            }) {
                Ok(Some(id)) if inserted => {
                    let _ = history.mark_inserted(id);
                }
                Ok(_) => {}
                Err(e) => warn!("history write failed: {e}"),
            }
            if let Some(line) = log_line {
                append_timing_log(&line);
            }
        }));
        self.s.bus.emit(Event::Result {
            id: id.into(),
            raw: raw_text,
            text,
            inserted,
            refined,
            timings,
        });
        self.finish_session(id, None);
    }

    /// The re-dictation rule: the previous dictation, within the window, had the same words and
    /// Ochre typed something other than its raw text.
    fn redictated(&self, cfg: &Config, key: &str) -> bool {
        if !cfg.refine.redictation_raw || key.is_empty() {
            return false;
        }
        self.s.last_dictation.lock().as_ref().is_some_and(|p| {
            p.changed
                && p.key == key
                && p.at.elapsed().as_secs_f64() < cfg.refine.redictation_window_s
        })
    }

    /// Refine `raw`: (text to type, whether it differs from raw, pieces when chunked or 0).
    fn refine(
        &self,
        engines: &Engines,
        cfg: &Config,
        raw: &str,
        focus: &FocusInfo,
    ) -> (String, bool, usize) {
        let Some(refiner) = &engines.refiner else {
            return (raw.to_string(), false, 0);
        };
        self.set_state(State::Refining, None, "");
        let ctx = RefineContext {
            mode: cfg.refine.mode.clone(),
            app_name: focus.app_name.clone(),
            window_title: focus.window_title.clone(),
            style: cfg
                .refine
                .app_styles
                .get(&focus.app_name)
                .cloned()
                .unwrap_or_default(),
            dictionary: cfg.dictionary.words.clone(),
            language: cfg.stt.language.clone(),
        };
        let timeout = Duration::from_millis(if refiner.info().kind == "local" {
            cfg.refine.timeout_ms_local
        } else {
            cfg.refine.timeout_ms_cloud
        });
        let guarded = |raw: &str| match refiner.refine(raw, &ctx, timeout) {
            Ok(out) => Some((engines.guard)(raw, &out, &cfg.refine.mode)),
            Err(e) => {
                warn!("refinement skipped: {e}");
                None
            }
        };
        // Long dictations go piece by piece (chunk.rs): one request per piece, in order (the
        // local server has one slot, and each request reuses the cached system prompt), each
        // piece guarded on its own and falling back to its own raw text.
        let pieces = if chunk::enabled(&cfg.refine.chunk_long, refiner.model_name().as_deref()) {
            chunk::split(raw, cfg.refine.chunk_min_words, chunk::TARGET_WORDS)
        } else {
            Vec::new()
        };
        if pieces.len() >= 2 {
            let outs: Vec<Option<String>> = pieces.iter().map(|p| guarded(&p.text)).collect();
            if outs.iter().all(Option::is_none) {
                return (raw.to_string(), false, pieces.len()); // the refiner is down: as before
            }
            let text = chunk::join(
                pieces
                    .iter()
                    .zip(&outs)
                    .map(|(p, o)| (p.sep, o.as_deref().unwrap_or(&p.text))),
            );
            let changed = text != raw;
            return (text, changed, pieces.len());
        }
        match guarded(raw) {
            Some(text) => {
                let changed = text != raw;
                (text, changed, 0)
            }
            None => (raw.to_string(), false, 0),
        }
    }

    fn insert(&self, engines: &Engines, cfg: &Config, text: &str, send: bool) -> bool {
        let focus = engines.injector.focus();
        let payload = {
            let mut join = self.s.join.lock();
            let recent = join
                .at
                .is_some_and(|t| t.elapsed().as_secs_f64() < cfg.inject.join_window_s);
            let lead = recent
                && join.window_id == focus.window_id
                && !join.ended_with_space
                && !text.starts_with([' ', '.', ',', '!', '?', ';', ':', '\n']);
            let mut p = String::with_capacity(text.len() + 2);
            if lead {
                p.push(' ');
            }
            p.push_str(text);
            if cfg.inject.trailing_space && !send && !p.ends_with([' ', '\n']) {
                p.push(' ');
            }
            *join = JoinMemory {
                window_id: focus.window_id.clone(),
                ended_with_space: p.ends_with([' ', '\n']),
                at: Some(Instant::now()),
            };
            p
        };
        let typed = if cfg.inject.method == "paste"
            || payload.chars().count() > cfg.inject.paste_over_chars
        {
            engines.injector.paste_text(&payload)
        } else {
            engines.injector.type_text(&payload)
        };
        let result = typed.and_then(|()| {
            if send {
                engines.injector.press_enter()
            } else {
                Ok(())
            }
        });
        match result {
            Ok(()) => true,
            Err(e) => {
                self.s.bus.emit(Event::Notice {
                    message: format!("{e}. Saved to history."),
                });
                false
            }
        }
    }

    fn finish_session(&self, id: &str, error: Option<(String, &str)>) {
        {
            let mut st = self.s.st.lock();
            if st.session.as_ref().is_some_and(|s| s.id == id)
                && let Some(l) = st.session.take().and_then(|s| s.live)
            {
                l.close();
            }
        }
        match error {
            Some((msg, code)) => self.fail(&msg, code),
            None => self.set_state(State::Idle, None, ""),
        }
    }

    /// STT with automatic local fallback for retryable cloud failures.
    fn transcribe_fn(&self, engines: &Arc<Engines>, cfg: &Config) -> TranscribeFn {
        let engines = engines.clone();
        let bus = self.s.bus.clone();
        let opts = stt_options(cfg);
        Arc::new(move |pcm: &[f32]| -> Result<SttResult> {
            match engines.stt.transcribe(pcm, &opts) {
                Err(e) if e.is_fallback_worthy() && engines.fallback_stt.is_some() => {
                    warn!(code = e.code(), "cloud STT failed; using local fallback");
                    bus.emit(Event::Notice {
                        message: format!(
                            "{} failed ({}), used local transcription",
                            engines.stt.info().label,
                            e.code()
                        ),
                    });
                    engines
                        .fallback_stt
                        .as_ref()
                        .unwrap()
                        .transcribe(pcm, &opts)
                }
                other => other,
            }
        })
    }

    // ------------------------------------------------------------------ commands from UI / CLI / tray

    pub fn command(&self, cmd: Command) {
        match cmd {
            Command::Start => {
                self.begin(Trigger::Ui);
            }
            Command::Stop => self.end(false, false),
            Command::Cancel => self.cancel(true),
            Command::Toggle => {
                if self.s.st.lock().session.is_some() {
                    self.end(false, false);
                } else {
                    self.begin(Trigger::Cli);
                }
            }
            Command::GetConfig => self.emit_config(),
            Command::SetConfig { patch } => match self.config().patched(&patch) {
                Ok(new) => self.apply_config(new),
                Err(e) => self.s.bus.emit(Event::Error {
                    message: e.to_string(),
                    code: e.code().into(),
                }),
            },
            Command::SetSecret { provider, key } => {
                if let Err(e) = secrets::store(&provider, key.as_deref()) {
                    // "keyring": the settings window shows it beside the key field.
                    self.s.bus.emit(Event::Error {
                        message: e.to_string(),
                        code: "keyring".into(),
                    });
                }
                self.emit_config();
            }
            Command::HistoryQuery { q, limit } => match self.s.history.query(&q, limit) {
                Ok(items) => self.s.bus.emit(Event::History { items }),
                Err(e) => warn!("history query failed: {e}"),
            },
            Command::InsertText { text } => {
                if let Some(engines) = self.s.engines.lock().clone() {
                    let app = self.clone();
                    let _ = self.s.worker.send(Box::new(move || {
                        let cfg = app.config();
                        app.insert(&engines, &cfg, &text, false);
                    }));
                }
            }
            Command::SetHandsfree { enabled } => {
                if let Ok(new) = self
                    .config()
                    .patched(&serde_json::json!({"handsfree": {"enabled": enabled}}))
                {
                    self.apply_config(new);
                }
            }
            Command::HistoryClear => {
                if let Err(e) = self.s.history.clear() {
                    warn!("history clear failed: {e}");
                }
                self.s.bus.emit(Event::History { items: Vec::new() });
            }
            Command::PasteLast => self.paste_last(),
            Command::Quit => self.request_quit(),
            // Handled by the binary (they need engine registries / windows): TestProvider,
            // DownloadModel, TrainWake, OpenSettings, permissions, devices, hotkey capture.
            Command::TestProvider { .. }
            | Command::DownloadModel { .. }
            | Command::TrainWake { .. }
            | Command::OpenSettings
            | Command::CheckPermissions
            | Command::OpenPermissionSettings { .. }
            | Command::ListDevices
            | Command::ShowMicModes
            | Command::HotkeyCapture { .. }
            | Command::CancelDownload { .. } => {}
        }
    }

    pub fn apply_config(&self, new: Config) {
        let old = std::mem::replace(&mut *self.s.cfg.lock(), new.clone());
        if let Some(path) = self.s.cfg_path.lock().as_ref()
            && let Err(e) = new.save_to(path)
        {
            warn!("could not save config: {e}");
        }
        self.emit_config();
        let engine_changed = old.stt != new.stt
            || old.refine.provider != new.refine.provider
            || old.refine.model != new.refine.model
            || old.refine.base_url != new.refine.base_url
            || old.audio.device != new.audio.device
            || old.audio.warm_mic != new.audio.warm_mic
            || old.audio.voice_processing != new.audio.voice_processing
            || old.audio.avoid_bluetooth_mic != new.audio.avoid_bluetooth_mic
            || old.audio.warm_idle_release_s != new.audio.warm_idle_release_s;
        if engine_changed {
            if old.audio.device != new.audio.device
                || old.audio.warm_mic != new.audio.warm_mic
                || old.audio.voice_processing != new.audio.voice_processing
                || old.audio.avoid_bluetooth_mic != new.audio.avoid_bluetooth_mic
                || old.audio.warm_idle_release_s != new.audio.warm_idle_release_s
            {
                self.stop_wake(); // let go of the old microphone; the reload restarts listening
            }
            self.reload("Applying settings…");
        } else if old.handsfree != new.handsfree
            || old.audio.max_session_s != new.audio.max_session_s
        {
            self.sync_wake();
        }
    }

    fn emit_config(&self) {
        let config = serde_json::to_value(&*self.s.cfg.lock()).unwrap_or_default();
        self.s.bus.emit(Event::Config {
            config,
            secrets: secrets::presence(),
        });
    }

    // ------------------------------------------------------------------ helpers

    fn set_state(&self, state: State, trigger: Option<Trigger>, detail: &str) {
        self.s.st.lock().state = state;
        let armed = self.s.armed.load(Ordering::SeqCst);
        self.s.bus.emit(Event::State {
            state,
            trigger,
            handsfree_armed: armed,
            detail: detail.into(),
        });
    }

    fn fail(&self, message: &str, code: &str) {
        warn!(code, "{message}");
        self.s.bus.emit(Event::Error {
            message: message.into(),
            code: code.into(),
        });
        self.set_state(State::Error, None, message);
        if let Some(e) = self.s.engines.lock().clone() {
            cue(&e, Cue::Error);
        }
    }
}

fn stt_options(cfg: &Config) -> SttOptions {
    SttOptions {
        language: cfg.stt.language.clone(),
        vocabulary: cfg.dictionary.words.clone(),
    }
}

fn cue(engines: &Engines, c: Cue) {
    if let Some(f) = &engines.cue {
        f(c);
    }
}

fn ms(d: Duration) -> u64 {
    d.as_millis() as u64
}

fn new_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(1);
    format!(
        "{:x}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    )
}

/// Rolling per-session timing log (docs/latency.md rule 12). Kept small: rotated at 1 MB.
fn append_timing_log(line: &serde_json::Value) {
    use std::io::Write;
    let path = ochre_core::paths::data_dir().join("timings.jsonl");
    if std::fs::metadata(&path).is_ok_and(|m| m.len() > 1_000_000) {
        let _ = std::fs::rename(&path, path.with_extension("jsonl.1"));
    }
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let _ = writeln!(f, "{line}");
    }
}
