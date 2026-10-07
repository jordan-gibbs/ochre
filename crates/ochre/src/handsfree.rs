//! Hands-free wiring: the always-on wake listener thread between the mic tap and the
//! orchestrator (SPEC §6.3, `docs/wakeword.md` §7).
//!
//! ```text
//! capture pump ──tap (try_send, drops on overflow)──► bounded queue ──► `ochre-wake` thread
//!     HandsFree::feed (VAD-gated detector, pre-roll ring, idle timer, call pause)
//!       └─ Wake ──► App::begin_wake (Trigger::Wake, audio from the wake word's first sample)
//! segmenter decode ──phrase text──► control queue ──► HandsFree::on_phrase ──► finish / send /
//!     cancel / scratch / false wake ──► App
//! session tail ──Close──► the controller's final text (wake word and control phrases stripped)
//! ```
//!
//! The thread owns the controller, so nothing locks it; everything else talks to it through one
//! FIFO control channel, which keeps phrase order and lets the tail ask for the final text only
//! after every phrase was handled. It runs at normal priority and never touches the capture
//! callback: the tap is called on the capture pump and only `try_send`s a copy of the block.
//!
//! While the app is busy with anything else (a hotkey dictation, the tail of a session, loading)
//! the audio is dropped and the detector is held when listening resumes, so dictated words never
//! wake it. During a hands-free session the detector keeps running: a wake hit there arms control
//! phrases, and its VAD drives the idle timeout.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use crossbeam_channel::{Receiver, Sender, bounded, select, unbounded};
use parking_lot::Mutex;
use tracing::{debug, info, warn};

use ochre_core::Result;
use ochre_core::config::{Config, HandsFreeConfig};
use ochre_wake::{EndReason, FRAME, HandsFree, HfEvent, PhraseAction, SessionEnd, SpeechFn};

use crate::app::{App, WeakApp};
use crate::seams::{Mic, Tap};

/// Audio queue between the tap and the wake thread, in capture blocks (~10 ms each): ~2.5 s.
const AUDIO_QUEUE: usize = 256;
/// Timer tick when no audio arrives (idle timeout, app gone).
const TICK: Duration = Duration::from_millis(250);

/// A ready controller and a line for the log.
pub struct Listener {
    pub hf: HandsFree,
    /// "model=…, threshold=…, vad=…"
    pub desc: String,
}

/// Builds the controller from the config, on the wake thread (it may download the front end).
pub type WakeFactory = Arc<dyn Fn(&Config) -> Result<Listener> + Send + Sync>;

/// The real thing: the compiled-in "transcribe" head (or `handsfree.model`), the openWakeWord
/// front end (downloaded once into `<models_dir>/openwakeword`), Silero as the VAD gate (an energy
/// gate if Silero can't load), and the call monitor when `pause_on_calls`.
pub fn default_factory() -> WakeFactory {
    Arc::new(|cfg: &Config| {
        let h = &cfg.handsfree;
        let dir = ochre_wake::models::bundled_dir();
        if h.model.is_empty() {
            ochre_wake::models::install_bundled(&dir)?;
        }
        let model = ochre_wake::resolve_model(&h.phrase, &h.model, &dir)?;
        let (vad, vad_name) = speech_fn();
        let hf = HandsFree::from_config(h, cfg.audio.max_session_s as f64, &dir, Some(vad))?;
        Ok(Listener {
            hf,
            desc: describe(&model, h.threshold, vad_name),
        })
    })
}

fn describe(model: &Path, threshold: f32, vad: &str) -> String {
    format!(
        "model={}, threshold={threshold:.2}, vad={vad}",
        model.display()
    )
}

/// Silero (probability >= 0.5 on any 32 ms chunk of the 80 ms frame; chunks below an RMS floor
/// skip inference), else an RMS gate.
fn speech_fn() -> (SpeechFn, &'static str) {
    match ochre_audio::SileroVad::load(&|_, _, _| {}, &AtomicBool::new(false)) {
        Ok(vad) => {
            let mut vad = vad.with_energy_floor(0.002);
            (
                Box::new(move |f: &[f32]| vad.prob(f).map(|p| p >= 0.5).unwrap_or(true)),
                "silero",
            )
        }
        Err(e) => {
            warn!("silero VAD unavailable for hands-free ({e}); using an energy gate");
            (
                Box::new(|f: &[f32]| {
                    (f.iter().map(|v| v * v).sum::<f32>() / f.len().max(1) as f32).sqrt() > 0.005
                }),
                "energy",
            )
        }
    }
}

/// Messages to the wake thread. One FIFO, so a session's phrases are always handled before the
/// tail's `Close`.
pub(crate) enum Ctl {
    /// A finished phrase transcript of hands-free session `hf_id`.
    Phrase {
        hf_id: u64,
        text: String,
    },
    /// The session tail wants the final text: the session's end if it already ended, else end it
    /// now (`finish(send)`, e.g. the hotkey or the HUD stopped it).
    Close {
        hf_id: u64,
        send: bool,
        reply: Sender<Option<SessionEnd>>,
    },
    /// The orchestrator canceled the session (Escape, HUD ×).
    Cancel {
        hf_id: u64,
    },
    Stop,
}

/// A running wake thread.
pub(crate) struct WakeHandle {
    ctl: Sender<Ctl>,
    thread: Option<JoinHandle<()>>,
    mic: Arc<dyn Mic>,
    key: (HandsFreeConfig, u64),
}

fn key(cfg: &Config) -> (HandsFreeConfig, u64) {
    (cfg.handsfree.clone(), cfg.audio.max_session_s)
}

impl WakeHandle {
    pub(crate) fn spawn(
        app: WeakApp,
        cfg: Config,
        mic: Arc<dyn Mic>,
        factory: WakeFactory,
    ) -> Self {
        let (tx, rx) = unbounded();
        let k = key(&cfg);
        let thread = {
            let mic = mic.clone();
            let tx = tx.clone();
            std::thread::Builder::new()
                .name("ochre-wake".into())
                .spawn(move || run(app, cfg, mic, factory, tx, rx))
                .map_err(|e| warn!("could not start the wake thread: {e}"))
                .ok()
        };
        WakeHandle {
            ctl: tx,
            thread,
            mic,
            key: k,
        }
    }

    /// Same settings and the same microphone: nothing to restart.
    pub(crate) fn matches(&self, cfg: &Config, mic: &Arc<dyn Mic>) -> bool {
        self.key == key(cfg) && Arc::as_ptr(&self.mic) as *const () == Arc::as_ptr(mic) as *const ()
    }

    /// Stop listening, cancel a running session, remove the tap and wait for the thread.
    pub(crate) fn stop(mut self) {
        let _ = self.ctl.send(Ctl::Stop);
        if let Some(t) = self.thread.take()
            && t.thread().id() != std::thread::current().id()
        {
            let _ = t.join();
        }
        let _ = self.mic.set_tap(None);
    }

    pub(crate) fn running(&self) -> bool {
        self.thread.as_ref().is_some_and(|t| !t.is_finished())
    }
}

fn run(
    weak: WeakApp,
    cfg: Config,
    mic: Arc<dyn Mic>,
    factory: WakeFactory,
    ctl_tx: Sender<Ctl>,
    ctl_rx: Receiver<Ctl>,
) {
    let listener = match factory(&cfg) {
        Ok(l) => l,
        Err(e) => {
            if let Some(app) = weak.upgrade() {
                app.wake_failed(&format!("Hands-free couldn't start: {e}"));
            }
            return;
        }
    };
    let (atx, arx) = bounded::<(u64, Vec<f32>)>(AUDIO_QUEUE);
    let dropped = Arc::new(AtomicU64::new(0));
    let tap: Tap = {
        let dropped = dropped.clone();
        Box::new(move |at: u64, block: &[f32]| {
            if atx.try_send((at, block.to_vec())).is_err() {
                dropped.fetch_add(1, Ordering::Relaxed);
            }
        })
    };
    if let Err(e) = mic.set_tap(Some(tap)) {
        if let Some(app) = weak.upgrade() {
            app.wake_failed(&format!("Hands-free couldn't start: {e}"));
        }
        return;
    }
    info!("wake listening ({})", listener.desc);
    let mut r = Runner {
        hf: listener.hf,
        ctl: ctl_tx,
        ended: None,
        held: false,
        dropped,
        reported_drops: 0,
    };
    if let Some(app) = weak.upgrade() {
        app.set_armed(true);
    }
    loop {
        select! {
            recv(ctl_rx) -> m => match m {
                Ok(Ctl::Stop) | Err(_) => break,
                Ok(c) => {
                    let Some(app) = weak.upgrade() else { break };
                    r.control(&app, c);
                }
            },
            recv(arx) -> b => {
                let Ok((at, block)) = b else { break };
                let Some(app) = weak.upgrade() else { break };
                r.audio(&app, at, &block);
            },
            default(TICK) => {
                let Some(app) = weak.upgrade() else { break };
                let evs = r.hf.tick();
                r.events(&app, evs, 0);
            },
        }
    }
    let _ = mic.set_tap(None);
    if let Some(app) = weak.upgrade() {
        let evs = r.hf.set_enabled(false); // cancels a running session
        r.events(&app, evs, 0);
        app.set_armed(false);
    }
    info!("wake listener stopped");
}

struct Runner {
    hf: HandsFree,
    /// Our own control channel, handed to wake sessions for their phrases.
    ctl: Sender<Ctl>,
    /// The last session end, for the tail's `Close`.
    ended: Option<SessionEnd>,
    /// Audio is being dropped while the app is busy; hold the detector when it resumes.
    held: bool,
    dropped: Arc<AtomicU64>,
    reported_drops: u64,
}

impl Runner {
    fn audio(&mut self, app: &App, at: u64, block: &[f32]) {
        let d = self.dropped.load(Ordering::Relaxed);
        if d > self.reported_drops {
            warn!(
                blocks = d - self.reported_drops,
                "wake listener fell behind; audio dropped"
            );
            self.reported_drops = d;
        }
        if !self.hf.in_session() && !app.idle_for_wake() {
            self.held = true; // a hotkey dictation, a session's tail, loading: don't listen
            return;
        }
        if self.held {
            self.held = false;
            self.hf.hold();
        }
        // Feed at most one frame per call, so a wake's pre-roll ends exactly at `pos`.
        let (mut rest, mut pos) = (block, at);
        while !rest.is_empty() {
            let n = (FRAME - self.hf.pending_samples()).min(rest.len());
            let evs = self.hf.feed(&rest[..n]);
            rest = &rest[n..];
            pos += n as u64;
            self.events(app, evs, pos);
        }
    }

    /// `pos`: absolute sample just after the audio fed so far (0 = not from audio).
    fn events(&mut self, app: &App, evs: Vec<HfEvent>, pos: u64) {
        for ev in evs {
            match ev {
                HfEvent::Wake(w) => {
                    let back = (self.hf.pending_samples() + w.preroll.len()) as u64;
                    let from = pos.saturating_sub(back);
                    info!(
                        "wake hit: score {:.2}, session {}, pre-roll {:.2} s",
                        w.score,
                        w.session_id,
                        w.preroll.len() as f64 / ochre_wake::SAMPLE_RATE as f64
                    );
                    if !app.begin_wake(w.session_id, from, self.ctl.clone()) {
                        info!("wake ignored: the app is busy");
                        self.ended = self.hf.cancel();
                    }
                }
                HfEvent::End(end) => self.ended(app, end),
                HfEvent::Paused(apps) => {
                    app.set_armed(false);
                    app.notice_msg(&format!(
                        "Hands-free paused: {} is using the microphone",
                        apps.join(", ")
                    ));
                }
                HfEvent::Resumed => app.set_armed(true),
                HfEvent::State(_) => {}
            }
        }
    }

    fn ended(&mut self, app: &App, end: SessionEnd) {
        info!(
            "hands-free session {} ended: {:?} ({} chars{})",
            end.session_id,
            end.reason,
            end.text.chars().count(),
            if end.press_enter() {
                ", then Enter"
            } else {
                ""
            }
        );
        let (id, reason, enter) = (end.session_id, end.reason, end.press_enter());
        self.ended = Some(end);
        match reason {
            EndReason::Cancel => app.wake_cancel(id, true),
            EndReason::FalseWake => app.wake_cancel(id, false),
            _ => app.wake_finish(id, enter),
        }
    }

    fn control(&mut self, app: &App, c: Ctl) {
        match c {
            Ctl::Phrase { hf_id, text } => match self.hf.on_phrase_for(&text, Some(hf_id)) {
                PhraseAction::Finish(e)
                | PhraseAction::Send(e)
                | PhraseAction::Cancel(e)
                | PhraseAction::FalseWake(e) => self.ended(app, e),
                PhraseAction::Scratch { dropped } => debug!("scratch that: {dropped:?}"),
                PhraseAction::Continue { .. } | PhraseAction::Ignored => {}
            },
            Ctl::Close { hf_id, send, reply } => {
                let end = match &self.ended {
                    Some(e) if e.session_id == hf_id => Some(e.clone()),
                    _ if self.hf.session_id() == Some(hf_id) => {
                        let e = self.hf.finish(send);
                        if let Some(e) = &e {
                            info!(
                                "hands-free session {} ended: {:?} (stopped from the app)",
                                e.session_id, e.reason
                            );
                        }
                        self.ended = e.clone();
                        e
                    }
                    _ => None,
                };
                let _ = reply.send(end);
            }
            Ctl::Cancel { hf_id } => {
                if self.hf.session_id() == Some(hf_id) {
                    self.ended = self.hf.cancel();
                }
            }
            Ctl::Stop => {}
        }
    }
}

/// Ask the wake thread for a session's final text (after all its phrases were handled).
pub(crate) fn close(ctl: &Sender<Ctl>, hf_id: u64, send: bool) -> Option<SessionEnd> {
    let (tx, rx) = bounded(1);
    ctl.send(Ctl::Close {
        hf_id,
        send,
        reply: tx,
    })
    .ok()?;
    rx.recv_timeout(Duration::from_secs(3)).ok().flatten()
}

/// The wake thread slot in the app, plus the lock that serializes start/stop.
#[derive(Default)]
pub(crate) struct WakeSlot {
    pub(crate) handle: Mutex<Option<WakeHandle>>,
    pub(crate) sync: Mutex<()>,
}
