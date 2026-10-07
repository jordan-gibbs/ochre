//! Microphone capture with an always-warm stream and pre-roll (SPEC §3.1 rules 2 and 4).
//!
//! Threads:
//! - **cpal callback**: converts samples to f32 and copies them into a lock-free SPSC ring
//!   (`rtrb`), stamps the time, and unparks the pump. No allocation, no locks, no syscalls beyond
//!   the unpark.
//! - **pump** (`ochre-capture`): owns the cpal stream (opening, closing, reopening after device
//!   loss). Woken by each callback, it drains the ring, downmixes to mono, resamples once to
//!   16 kHz (rubato FFT resampler), keeps the last `history` of 16 kHz audio for pre-roll, and
//!   forwards blocks to the active session over a channel. It also meters the level for the HUD
//!   (~15 Hz), enforces the session cap, and watches for a stalled or lost device.
//!
//! With `warm` on, the stream runs while the app runs, so `begin()` has zero device-open latency
//! and the session starts `preroll_ms` *before* the press. With `warm` off, `begin()` opens the
//! device (no pre-roll) and the stream closes when the session ends.
//!
//! **Idle release** (`idle_release`, macOS default 5 min): a warm stream with no session and no
//! tap for that long is closed, so the OS mic indicator goes away; the next `begin()` (or
//! [`Capture::prime`]) opens the fastest path at once (cpal, ~60–120 ms on an M5 Pro) and the
//! stream is warm again. On macOS a voice-processed stream (`voice_mac.rs`, ~1 s to start) then
//! replaces the plain one in the background, between sessions, with the history kept.
//!
//! **Tap** ([`Capture::set_tap`]): an always-on subscriber (hands-free listening) that gets every
//! 16 kHz block on the pump thread, sessions or not, tagged with its absolute sample position. It
//! keeps the device open while installed (even with `warm` off). [`Capture::begin_sink_from`]
//! starts a session at such a position (from the history), so a wake word found on the tap hands
//! over to dictation with no gap and no duplicate audio.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{JoinHandle, Thread};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use ochre_core::{Error, Result, SAMPLE_RATE};

use crate::resample::StreamResampler;

/// Pre-roll kept in memory while warm (SPEC: ≥ 1.5 s).
pub const DEFAULT_HISTORY_MS: u32 = 1500;
pub const DEFAULT_PREROLL_MS: u32 = 150;
/// No callback for this long while the stream should be running = device lost.
const STALL: Duration = Duration::from_millis(1500);
const REOPEN_EVERY: Duration = Duration::from_millis(1000);
const DEFAULT_CHECK_EVERY: Duration = Duration::from_secs(5);
/// HUD meter cadence (at least 30 Hz, docs/latency.md rule 11).
const LEVEL_EVERY: Duration = Duration::from_millis(33);
/// Raw ring capacity in seconds of device audio (the pump normally drains every ~10 ms).
const RING_SECONDS: usize = 2;
const LEVEL_FLOOR_DB: f32 = -60.0;
const LEVEL_CEIL_DB: f32 = -12.0;

/// Map RMS to 0..1 on a dB scale (-60 dBFS → 0, -12 dBFS → 1): what the HUD bars use, because a
/// linear RMS sits near zero for normal speech.
pub fn level_from_rms(rms: f32) -> f32 {
    if rms <= 0.0 {
        return 0.0;
    }
    ((20.0 * rms.log10() - LEVEL_FLOOR_DB) / (LEVEL_CEIL_DB - LEVEL_FLOOR_DB)).clamp(0.0, 1.0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct InputDevice {
    pub name: String,
    pub is_default: bool,
    pub channels: u16,
    pub sample_rate: u32,
}

/// Input devices on the default host, the default first.
pub fn list_devices() -> Vec<InputDevice> {
    let host = cpal::default_host();
    let default = host.default_input_device().and_then(|d| d.name().ok());
    let mut out: Vec<InputDevice> = host
        .input_devices()
        .map(|it| {
            it.filter_map(|d| {
                let name = d.name().ok()?;
                let cfg = d.default_input_config().ok()?;
                Some(InputDevice {
                    is_default: default.as_deref() == Some(name.as_str()),
                    name,
                    channels: cfg.channels(),
                    sample_rate: cfg.sample_rate().0,
                })
            })
            .collect()
        })
        .unwrap_or_default();
    out.sort_by_key(|d| !d.is_default);
    out
}

/// Config stores a device *name* (indexes change across reboots): exact match first, then a
/// case-insensitive substring; unknown names fall back to the default with a warning.
fn resolve_device(host: &cpal::Host, name: Option<&str>) -> Option<cpal::Device> {
    if let Some(want) = name.filter(|n| !n.is_empty()) {
        let devices: Vec<_> = host
            .input_devices()
            .map(|i| i.collect())
            .unwrap_or_default();
        let named = |d: &cpal::Device| d.name().unwrap_or_default();
        if let Some(d) = devices.iter().find(|d| named(d) == want) {
            return Some(d.clone());
        }
        let low = want.to_lowercase();
        if let Some(d) = devices
            .iter()
            .find(|d| named(d).to_lowercase().contains(&low))
        {
            return Some(d.clone());
        }
        tracing::warn!(
            device = want,
            "input device not found; using the system default"
        );
    }
    host.default_input_device()
}

#[derive(Debug, Clone, PartialEq)]
pub enum CaptureEvent {
    /// Stream opened (or reopened after a loss).
    Opened {
        device: String,
        sample_rate: u32,
        channels: u16,
    },
    /// The device disappeared or stopped delivering; reopening is automatic.
    DeviceLost { device: String },
    /// Could not open any input device; retried every second while needed.
    OpenFailed { message: String },
    /// The active session hit `max_session`; its reader has ended.
    LimitReached,
}

pub type LevelFn = Arc<dyn Fn(f32) + Send + Sync>;
pub type EventFn = Arc<dyn Fn(&CaptureEvent) + Send + Sync>;

#[derive(Clone)]
pub struct CaptureOptions {
    /// Device name; None = system default (followed when the default changes).
    pub device: Option<String>,
    pub warm: bool,
    /// macOS: capture through Apple's voice processing (echo cancellation, noise suppression,
    /// the system Mic Mode menu) when `device` is the system default and the stream opens ahead of
    /// time (warm or tapped; it takes ~1 s to start). Ignored elsewhere.
    pub voice_processing: bool,
    /// macOS: with no device configured and a Bluetooth headset as the default input, record from
    /// the built-in mic instead (keeps the headset out of its call profile).
    pub avoid_bluetooth_mic: bool,
    /// Close a warm stream after this long with no session (None = never).
    pub idle_release: Option<Duration>,
    pub history_ms: u32,
    pub max_session: Duration,
    /// Normalized 0..1 level, ~30 Hz, only while a session is active.
    pub on_level: Option<LevelFn>,
    pub on_event: Option<EventFn>,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            device: None,
            warm: true,
            voice_processing: false,
            avoid_bluetooth_mic: false,
            idle_release: None,
            history_ms: DEFAULT_HISTORY_MS,
            max_session: Duration::from_secs(600),
            on_level: None,
            on_event: None,
        }
    }
}

impl CaptureOptions {
    pub fn from_config(cfg: &ochre_core::config::AudioConfig) -> Self {
        Self {
            device: cfg.device.clone(),
            warm: cfg.warm_mic,
            voice_processing: cfg.voice_processing,
            avoid_bluetooth_mic: cfg.avoid_bluetooth_mic,
            idle_release: (cfg.warm_idle_release_s > 0)
                .then(|| Duration::from_secs(cfg.warm_idle_release_s)),
            max_session: Duration::from_secs(cfg.max_session_s.max(1)),
            ..Self::default()
        }
    }
}

/// Same shape as the orchestrator's `ochre::seams::Sink`: called on the capture thread.
pub type Sink = Box<dyn FnMut(&[f32]) + Send>;

/// Always-on subscriber: `(absolute index of the block's first sample, 16 kHz mono block)`, called
/// on the pump thread for every block. It must return at once (e.g. `try_send` into a bounded
/// channel and drop on overflow); it never runs on the real-time cpal callback.
pub type Tap = Box<dyn FnMut(u64, &[f32]) + Send>;

/// Where a session's audio starts.
enum Start {
    /// This many samples of history before now.
    Preroll(usize),
    /// At this absolute sample (see [`Tap`]), as far back as the history reaches.
    At(u64),
}

/// Post-roll after release: at least this much more audio...
pub const POST_ROLL: Duration = Duration::from_millis(150);
/// ...extended while the speaker is still voicing, up to this.
pub const POST_ROLL_MAX: Duration = Duration::from_millis(300);

enum Out {
    Channel(Sender<Vec<f32>>),
    Sink(Sink),
}

enum Cmd {
    Begin {
        start: Start,
        out: Out,
        limit_hit: Arc<AtomicBool>,
        level: Option<LevelFn>,
    },
    /// Deliver everything captured so far and end the session now (channel API).
    End,
    /// Deliver everything captured so far, reply with the session length, then post-roll.
    Release {
        reply: Sender<u64>,
    },
    /// Drop the session immediately, no flush, no post-roll.
    Cancel,
    SetWarm(bool),
    SetDevice(Option<String>),
    /// Close and reopen the device (macOS: a stream opened before the microphone grant).
    Reopen,
    /// A dictation is probably coming (a chord's first modifier went down): open a released mic.
    Prime,
    SetTap(Option<Tap>),
    Shutdown,
}

/// State shared with the input callbacks (cpal, or the macOS voice-processing tap).
pub(crate) struct CallbackShared {
    pub(crate) epoch: Instant,
    /// Microseconds since `epoch` of the last data callback.
    pub(crate) last_cb_us: AtomicU64,
    pub(crate) lost: AtomicBool,
    pub(crate) overruns: AtomicU64,
}

struct Shared {
    cmds: Mutex<VecDeque<Cmd>>,
    pump: Thread,
    /// (device name, rate, channels) of the open stream.
    current: Mutex<Option<(String, u32, u16)>>,
    level: Mutex<f32>,
    /// No session is delivering (the sink has been dropped). Paired with `closed_cv`.
    closed: Mutex<bool>,
    closed_cv: Condvar,
}

/// The capture service. One per process; cheap to share behind an `Arc`.
pub struct Capture {
    shared: Arc<Shared>,
    handle: Option<JoinHandle<()>>,
    history_ms: u32,
}

impl Capture {
    /// Start the pump (and, when `warm`, open the device in the background). Returns at once.
    pub fn start(opts: CaptureOptions) -> Result<Self> {
        let (ttx, trx) = mpsc::channel::<Arc<Shared>>();
        let history_ms = opts.history_ms.max(DEFAULT_PREROLL_MS);
        let handle = std::thread::Builder::new()
            .name("ochre-capture".into())
            .spawn(move || {
                // The pump feeds the segmenter: keep it responsive under load too.
                crate::priority::boost_current_thread();
                let shared = trx.recv().expect("capture shared state");
                Pump::new(opts, shared).run();
            })
            .map_err(|e| Error::Audio(format!("spawn capture thread: {e}")))?;
        let shared = Arc::new(Shared {
            cmds: Mutex::new(VecDeque::new()),
            pump: handle.thread().clone(),
            current: Mutex::new(None),
            level: Mutex::new(0.0),
            closed: Mutex::new(true),
            closed_cv: Condvar::new(),
        });
        ttx.send(shared.clone())
            .map_err(|_| Error::Audio("capture thread died".into()))?;
        Ok(Self {
            shared,
            handle: Some(handle),
            history_ms,
        })
    }

    fn send(&self, cmd: Cmd) {
        self.shared.cmds.lock().unwrap().push_back(cmd);
        self.shared.pump.unpark();
    }

    fn preroll_samples(&self, preroll_ms: u32) -> usize {
        (preroll_ms.min(self.history_ms) as usize * SAMPLE_RATE as usize) / 1000
    }

    fn mark_open(&self) {
        *self.shared.closed.lock().unwrap() = false;
    }

    /// Start a session whose audio begins `preroll_ms` before now (bounded by the history kept).
    /// Ends any active session. The reader yields 16 kHz mono blocks until `end()`, the session
    /// cap, or `Capture` drop.
    pub fn begin(&self, preroll_ms: u32) -> CaptureSession {
        let (tx, rx) = mpsc::channel();
        let limit_hit = Arc::new(AtomicBool::new(false));
        self.mark_open();
        self.send(Cmd::Begin {
            start: Start::Preroll(self.preroll_samples(preroll_ms)),
            out: Out::Channel(tx),
            limit_hit: limit_hit.clone(),
            level: None,
        });
        CaptureSession {
            rx,
            limit_hit,
            started: Instant::now(),
        }
    }

    /// Like `begin`, but audio goes straight to `sink` on the capture thread (no extra hop), and
    /// `level` gets this session's meter (~30 Hz). Returns the session-cap flag.
    pub fn begin_sink(
        &self,
        preroll_ms: u32,
        sink: Sink,
        level: Option<LevelFn>,
    ) -> Arc<AtomicBool> {
        self.begin_sink_at(
            Start::Preroll(self.preroll_samples(preroll_ms)),
            sink,
            level,
        )
    }

    /// Like `begin_sink`, but the session starts at absolute sample `from` of the tap stream
    /// (clamped to the history kept): a hands-free wake hands over without a gap.
    pub fn begin_sink_from(
        &self,
        from: u64,
        sink: Sink,
        level: Option<LevelFn>,
    ) -> Arc<AtomicBool> {
        self.begin_sink_at(Start::At(from), sink, level)
    }

    fn begin_sink_at(&self, start: Start, sink: Sink, level: Option<LevelFn>) -> Arc<AtomicBool> {
        let limit_hit = Arc::new(AtomicBool::new(false));
        self.mark_open();
        self.send(Cmd::Begin {
            start,
            out: Out::Sink(sink),
            limit_hit: limit_hit.clone(),
            level,
        });
        limit_hit
    }

    /// Install (or with `None` remove) the always-on [`Tap`]. While installed the device stays
    /// open, whatever `warm` says.
    pub fn set_tap(&self, tap: Option<Tap>) {
        self.send(Cmd::SetTap(tap));
    }

    /// End the active session now: audio captured so far is delivered, then the reader ends.
    pub fn end(&self) {
        self.send(Cmd::End);
    }

    /// Release: deliver everything captured up to now and return the session length in ms (the
    /// pump is woken directly, so this takes a few ms at most), then keep delivering a post-roll
    /// of `POST_ROLL`, extended up to `POST_ROLL_MAX` while the speaker is still voicing, and drop
    /// the sink. Pair with `wait_closed`.
    pub fn release(&self) -> Result<u64> {
        let (tx, rx) = mpsc::channel();
        self.send(Cmd::Release { reply: tx });
        rx.recv_timeout(Duration::from_millis(250))
            .map_err(|_| Error::Audio("capture thread did not respond".into()))
    }

    /// Drop the active session immediately (no flush, no post-roll).
    pub fn cancel(&self) {
        self.send(Cmd::Cancel);
    }

    /// Block until no session is delivering (post-roll done, sink dropped) or `timeout`.
    /// Returns whether it closed.
    pub fn wait_closed(&self, timeout: Duration) -> bool {
        let g = self.shared.closed.lock().unwrap();
        let (g, _) = self
            .shared
            .closed_cv
            .wait_timeout_while(g, timeout, |closed| !*closed)
            .unwrap();
        *g
    }

    pub fn set_warm(&self, warm: bool) {
        self.send(Cmd::SetWarm(warm));
    }

    pub fn set_device(&self, device: Option<String>) {
        self.send(Cmd::SetDevice(device));
    }

    /// Close the device and open it again (if it is wanted open).
    pub fn reopen(&self) {
        self.send(Cmd::Reopen);
    }

    /// A dictation is probably about to start: open the mic now if idle release closed it.
    pub fn prime(&self) {
        self.send(Cmd::Prime);
    }

    /// (name, native rate, channels) of the open stream, if any.
    pub fn current_device(&self) -> Option<(String, u32, u16)> {
        self.shared.current.lock().unwrap().clone()
    }

    /// Latest level (0..1), also delivered through `on_level`.
    pub fn level(&self) -> f32 {
        *self.shared.level.lock().unwrap()
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.send(Cmd::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The orchestrator's `Mic` seam over `Capture` (ochre adapts it with a thin `impl Mic`).
pub struct WarmMic {
    capture: Capture,
}

impl WarmMic {
    pub fn start(opts: CaptureOptions) -> Result<Self> {
        Ok(Self {
            capture: Capture::start(opts)?,
        })
    }

    pub fn capture(&self) -> &Capture {
        &self.capture
    }

    /// Mirrors `Mic::begin`: audio from `preroll_ms` before now goes to `sink` on the capture
    /// thread; `level` gets ~30 Hz meter values. Returns immediately.
    pub fn begin(&self, preroll_ms: u32, sink: Sink, level: LevelFn) -> Result<()> {
        self.capture.begin_sink(preroll_ms, sink, Some(level));
        Ok(())
    }

    /// Mirrors `Mic::begin_from`: like `begin`, from absolute sample `from` of the tap stream.
    pub fn begin_from(&self, from: u64, sink: Sink, level: LevelFn) -> Result<()> {
        self.capture.begin_sink_from(from, sink, Some(level));
        Ok(())
    }

    /// Mirrors `Mic::set_tap`.
    pub fn set_tap(&self, tap: Option<Tap>) -> Result<()> {
        self.capture.set_tap(tap);
        Ok(())
    }

    /// Mirrors `Mic::end`: flush the pre-release audio, return its duration (ms); the post-roll
    /// keeps flowing into the sink, then the sink is dropped.
    pub fn end(&self) -> Result<u64> {
        self.capture.release()
    }

    /// Mirrors `Mic::wait_closed`.
    pub fn wait_closed(&self, timeout: Duration) {
        self.capture.wait_closed(timeout);
    }

    /// Mirrors `Mic::cancel`: drop the sink immediately, no post-roll.
    pub fn cancel(&self) {
        self.capture.cancel();
    }

    /// Mirrors `Mic::reopen`.
    pub fn reopen(&self) {
        self.capture.reopen();
    }

    /// Mirrors `Mic::prime`.
    pub fn prime(&self) {
        self.capture.prime();
    }
}

/// Reader for one session: 16 kHz mono blocks (~10 ms each; the first carries the pre-roll).
pub struct CaptureSession {
    rx: Receiver<Vec<f32>>,
    limit_hit: Arc<AtomicBool>,
    started: Instant,
}

impl CaptureSession {
    /// Next block; None once the session has ended and everything was delivered.
    pub fn recv(&self) -> Option<Vec<f32>> {
        self.rx.recv().ok()
    }

    /// `Ok(None)` = ended; `Err(())` = nothing within `timeout`.
    #[allow(clippy::result_unit_err)]
    pub fn recv_timeout(&self, timeout: Duration) -> std::result::Result<Option<Vec<f32>>, ()> {
        match self.rx.recv_timeout(timeout) {
            Ok(b) => Ok(Some(b)),
            Err(RecvTimeoutError::Disconnected) => Ok(None),
            Err(RecvTimeoutError::Timeout) => Err(()),
        }
    }

    /// Everything available right now, without blocking.
    pub fn try_iter(&self) -> impl Iterator<Item = Vec<f32>> + '_ {
        self.rx.try_iter()
    }

    pub fn limit_reached(&self) -> bool {
        self.limit_hit.load(Ordering::SeqCst)
    }

    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
}

impl Iterator for CaptureSession {
    type Item = Vec<f32>;
    fn next(&mut self) -> Option<Vec<f32>> {
        self.recv()
    }
}

/// Index into a history of `len` samples, which holds absolute samples `[total - len, total)`,
/// where a session starting at `start` begins.
fn history_start(start: &Start, len: usize, total: u64) -> usize {
    match *start {
        Start::Preroll(n) => len.saturating_sub(n),
        Start::At(at) => {
            let oldest = total.saturating_sub(len as u64);
            at.saturating_sub(oldest).min(len as u64) as usize
        }
    }
}

// ---------------------------------------------------------------------------------------------

enum InputStream {
    Cpal(cpal::Stream),
    #[cfg(target_os = "macos")]
    Voice(#[allow(dead_code)] crate::voice_mac::VoiceStream),
}

impl InputStream {
    fn pause(&self) {
        match self {
            InputStream::Cpal(s) => {
                let _ = s.pause();
            }
            // Stopped by its Drop.
            #[cfg(target_os = "macos")]
            InputStream::Voice(_) => {}
        }
    }

    fn voice_processing(&self) -> bool {
        match self {
            InputStream::Cpal(_) => false,
            #[cfg(target_os = "macos")]
            InputStream::Voice(_) => true,
        }
    }
}

struct Open {
    stream: InputStream,
    consumer: rtrb::Consumer<f32>,
    cb: Arc<CallbackShared>,
    name: String,
    channels: usize,
    resampler: Option<StreamResampler>,
    opened_at: Instant,
}

struct Session {
    out: Out,
    limit_hit: Arc<AtomicBool>,
    level_fn: Option<LevelFn>,
    samples: usize,
    level_sq: f64,
    level_n: usize,
    last_level: Instant,
    peak_rms: f32,
    /// Set at release: post-roll in progress.
    released: Option<Instant>,
    voicing: bool,
}

impl Session {
    /// Deliver a block; false when the receiver is gone.
    fn deliver(&mut self, block: &[f32]) -> bool {
        match &mut self.out {
            Out::Channel(tx) => tx.send(block.to_vec()).is_ok(),
            Out::Sink(f) => {
                f(block);
                true
            }
        }
    }
}

struct Pump {
    opts: CaptureOptions,
    shared: Arc<Shared>,
    open: Option<Open>,
    history: VecDeque<f32>,
    history_cap: usize,
    /// Samples ever appended to `history`: it holds `[total - len, total)`.
    total: u64,
    tap: Option<Tap>,
    session: Option<Session>,
    next_open: Instant,
    last_default_check: Instant,
    mono: Vec<f32>,
    out: Vec<f32>,
    raw: Vec<f32>,
    /// Idle release closed the warm stream; the next session or prime reopens it.
    released: bool,
    /// Open on the fastest path next time (someone is waiting for audio).
    fast_open: bool,
    /// The last session's start or end.
    last_active: Instant,
    /// Voice processing failed to start: plain capture until the next `Capture`.
    voice_failed: bool,
    /// macOS: a voice-processed stream starting in the background, then waiting for a gap
    /// between sessions to replace the plain one.
    #[cfg(target_os = "macos")]
    upgrade: Option<Receiver<Result<VoiceOpened>>>,
    #[cfg(target_os = "macos")]
    upgrade_ready: Option<Open>,
}

#[cfg(target_os = "macos")]
struct VoiceOpened {
    stream: crate::voice_mac::VoiceStream,
    rate: u32,
    consumer: rtrb::Consumer<f32>,
    cb: Arc<CallbackShared>,
    name: String,
}

impl Pump {
    fn new(opts: CaptureOptions, shared: Arc<Shared>) -> Self {
        let history_cap =
            (opts.history_ms.max(DEFAULT_PREROLL_MS) as usize * SAMPLE_RATE as usize) / 1000;
        Self {
            opts,
            shared,
            open: None,
            history: VecDeque::with_capacity(history_cap + 4096),
            history_cap,
            total: 0,
            tap: None,
            session: None,
            next_open: Instant::now(),
            last_default_check: Instant::now(),
            mono: Vec::new(),
            out: Vec::new(),
            raw: Vec::new(),
            released: false,
            fast_open: false,
            last_active: Instant::now(),
            voice_failed: false,
            #[cfg(target_os = "macos")]
            upgrade: None,
            #[cfg(target_os = "macos")]
            upgrade_ready: None,
        }
    }

    fn emit(&self, e: CaptureEvent) {
        if let Some(f) = &self.opts.on_event {
            f(&e);
        }
    }

    fn want_open(&self) -> bool {
        (self.opts.warm && !self.released) || self.session.is_some() || self.tap.is_some()
    }

    /// Someone is about to need audio: undo an idle release, on the fast path.
    fn wake(&mut self) {
        if self.released {
            self.released = false;
            self.fast_open = true;
        }
        self.last_active = Instant::now();
    }

    /// Voice processing is wanted for a stream opened ahead of time (warm, or hands-free).
    fn voice_wanted(&self) -> bool {
        cfg!(target_os = "macos")
            && self.opts.voice_processing
            && !self.voice_failed
            && (self.opts.warm || self.tap.is_some())
    }

    /// (cpal device name to open, voice-processing input: None = not allowed, Some(None) = the
    /// system default, Some(Some(id)) = this device).
    fn route(&self) -> (Option<String>, Option<Option<u32>>) {
        if let Some(d) = &self.opts.device {
            return (Some(d.clone()), None);
        }
        #[cfg(target_os = "macos")]
        if let Some(r) = crate::devices_mac::default_route(self.opts.avoid_bluetooth_mic)
            && r.avoided_bluetooth
        {
            return (Some(r.name), Some(Some(r.id)));
        }
        (None, Some(None))
    }

    /// The name of the device we should be capturing from when none is configured.
    fn wanted_default_name(&self) -> Option<String> {
        #[cfg(target_os = "macos")]
        {
            crate::devices_mac::default_route(self.opts.avoid_bluetooth_mic).map(|r| r.name)
        }
        #[cfg(not(target_os = "macos"))]
        {
            cpal::default_host()
                .default_input_device()
                .and_then(|d| d.name().ok())
        }
    }

    /// Idle release: close a warm stream nobody has used for `idle_release`.
    fn idle_check(&mut self) {
        let Some(idle) = self.opts.idle_release else {
            return;
        };
        if self.opts.warm
            && !self.released
            && self.open.is_some()
            && self.session.is_none()
            && self.tap.is_none()
            && self.last_active.elapsed() >= idle
        {
            tracing::info!(idle_s = idle.as_secs(), "microphone released after idle");
            self.released = true;
            self.close();
        }
    }

    fn run(mut self) {
        loop {
            let cmds: Vec<Cmd> = self.shared.cmds.lock().unwrap().drain(..).collect();
            for cmd in cmds {
                match cmd {
                    Cmd::Shutdown => {
                        self.end_session();
                        self.close();
                        return;
                    }
                    Cmd::Begin {
                        start,
                        out,
                        limit_hit,
                        level,
                    } => {
                        self.wake();
                        self.drain(); // bring history up to "now" first
                        self.end_session();
                        let start = history_start(&start, self.history.len(), self.total);
                        let pre: Vec<f32> = self.history.range(start..).copied().collect();
                        let mut s = Session {
                            out,
                            limit_hit,
                            level_fn: level,
                            samples: pre.len(),
                            level_sq: 0.0,
                            level_n: 0,
                            last_level: Instant::now(),
                            peak_rms: 0.0,
                            released: None,
                            voicing: false,
                        };
                        *self.shared.closed.lock().unwrap() = false;
                        if !pre.is_empty() && !s.deliver(&pre) {
                            self.end_session();
                            continue;
                        }
                        self.session = Some(s);
                        self.next_open = Instant::now();
                    }
                    Cmd::End => {
                        self.drain();
                        self.end_session();
                    }
                    Cmd::Release { reply } => {
                        self.drain();
                        match self.session.as_mut() {
                            Some(s) if s.released.is_none() => {
                                let _ = reply.send((s.samples as u64 * 1000) / SAMPLE_RATE as u64);
                                s.released = Some(Instant::now());
                                if self.open.is_none() {
                                    self.end_session(); // nothing to post-roll from
                                }
                            }
                            _ => {
                                let _ = reply.send(0);
                            }
                        }
                    }
                    Cmd::Cancel => self.end_session(),
                    Cmd::SetWarm(w) => self.opts.warm = w,
                    Cmd::SetTap(t) => self.tap = t,
                    Cmd::Reopen => {
                        self.close();
                        self.next_open = Instant::now();
                    }
                    Cmd::Prime => self.wake(),
                    Cmd::SetDevice(d) => {
                        if d != self.opts.device {
                            self.opts.device = d;
                            self.close();
                            self.next_open = Instant::now();
                        }
                    }
                }
            }

            // Open / close to match what is wanted.
            if self.want_open() && self.open.is_none() && Instant::now() >= self.next_open {
                self.try_open();
            } else if !self.want_open() && self.open.is_some() {
                self.close();
            }

            self.drain();
            self.post_roll_check();
            self.watchdog();
            self.idle_check();
            #[cfg(target_os = "macos")]
            self.upgrade_check();

            let wait = match (&self.open, self.want_open()) {
                // Audio callbacks wake us; the timeout only drives the post-roll deadline and the
                // device watchdog.
                (Some(_), _) => match self.session.as_ref().and_then(|s| s.released) {
                    Some(at) => POST_ROLL_MAX
                        .saturating_sub(at.elapsed())
                        .max(Duration::from_millis(1)),
                    None => Duration::from_millis(200),
                },
                (None, true) => self
                    .next_open
                    .saturating_duration_since(Instant::now())
                    .max(Duration::from_millis(10)),
                (None, false) => Duration::from_secs(3600),
            };
            std::thread::park_timeout(wait);
        }
    }

    /// End the post-roll once `POST_ROLL` has passed and the speaker is quiet, or at
    /// `POST_ROLL_MAX`.
    fn post_roll_check(&mut self) {
        let Some(s) = &self.session else { return };
        let Some(at) = s.released else { return };
        let t = at.elapsed();
        if t >= POST_ROLL_MAX || (t >= POST_ROLL && !s.voicing) || self.open.is_none() {
            self.end_session();
        }
    }

    fn try_open(&mut self) {
        // Voice processing takes ~1 s to start: only for a stream that opens ahead of time (warm,
        // or hands-free listening), never while someone waits for audio. Then it comes in the
        // background (`upgrade_check`).
        let (device, voice_input) = self.route();
        let voice = (self.voice_wanted() && !self.fast_open && self.session.is_none())
            .then_some(voice_input)
            .flatten();
        let tried_voice = voice.is_some();
        self.fast_open = false;
        match open_stream(device.as_deref(), voice, self.shared.pump.clone()) {
            Ok(open) => {
                let rate = open
                    .resampler
                    .as_ref()
                    .map_or(SAMPLE_RATE, |r| r.input_rate());
                let voice = open.stream.voice_processing();
                if tried_voice && !voice {
                    self.voice_failed = true;
                }
                tracing::info!(device = %open.name, rate, channels = open.channels, voice_processing = voice, "capture opened");
                *self.shared.current.lock().unwrap() =
                    Some((open.name.clone(), rate, open.channels as u16));
                self.emit(CaptureEvent::Opened {
                    device: open.name.clone(),
                    sample_rate: rate,
                    channels: open.channels as u16,
                });
                self.history.clear();
                self.open = Some(open);
                self.last_default_check = Instant::now();
            }
            Err(e) => {
                tracing::warn!(error = %e, "could not open input device; retrying");
                self.emit(CaptureEvent::OpenFailed {
                    message: e.to_string(),
                });
                self.next_open = Instant::now() + REOPEN_EVERY;
            }
        }
    }

    fn close(&mut self) {
        #[cfg(target_os = "macos")]
        {
            self.upgrade = None;
            self.upgrade_ready = None;
        }
        if let Some(open) = self.open.take() {
            open.stream.pause();
            drop(open);
        }
        *self.shared.current.lock().unwrap() = None;
        self.history.clear();
    }

    /// Drop the session's sink/sender and signal `wait_closed`.
    fn end_session(&mut self) {
        if self.session.is_some() {
            self.last_active = Instant::now();
        }
        drop(self.session.take()); // the sink (or channel) goes away here
        *self.shared.level.lock().unwrap() = 0.0;
        *self.shared.closed.lock().unwrap() = true;
        self.shared.closed_cv.notify_all();
    }

    fn watchdog(&mut self) {
        let Some(open) = &self.open else { return };
        let since = (open.cb.epoch.elapsed().as_micros() as u64)
            .saturating_sub(open.cb.last_cb_us.load(Ordering::Relaxed));
        let stalled = since > STALL.as_micros() as u64 && open.opened_at.elapsed() > STALL;
        if open.cb.lost.load(Ordering::Relaxed) || stalled {
            let name = open.name.clone();
            tracing::warn!(device = %name, stalled, "input device lost; reopening");
            self.close();
            self.emit(CaptureEvent::DeviceLost { device: name });
            self.next_open = Instant::now() + Duration::from_millis(200);
            return;
        }
        // Follow the system default device when none is configured (checked while idle only).
        if self.opts.device.is_none()
            && self.session.is_none()
            && self.last_default_check.elapsed() > DEFAULT_CHECK_EVERY
        {
            self.last_default_check = Instant::now();
            // (on macOS the built-in mic stands in for a Bluetooth default: see `route`)
            if self.wanted_default_name().is_some_and(|d| d != open.name) {
                tracing::info!("default input device changed; reopening");
                self.close();
                self.next_open = Instant::now();
            }
        }
    }

    /// Move everything in the ring through downmix + resample into history and the session.
    fn drain(&mut self) {
        let Some(open) = self.open.as_mut() else {
            return;
        };
        let n = open.consumer.slots();
        if n == 0 {
            return;
        }
        let ch = open.channels;
        let n = n - n % ch;
        self.raw.clear();
        if let Ok(chunk) = open.consumer.read_chunk(n) {
            let (a, b) = chunk.as_slices();
            self.raw.extend_from_slice(a);
            self.raw.extend_from_slice(b);
            chunk.commit_all();
        }
        self.mono.clear();
        if ch == 1 {
            self.mono.extend_from_slice(&self.raw);
        } else {
            let inv = 1.0 / ch as f32;
            self.mono.extend(
                self.raw
                    .chunks_exact(ch)
                    .map(|f| f.iter().sum::<f32>() * inv),
            );
        }
        self.out.clear();
        match open.resampler.as_mut() {
            Some(r) => r.process(&self.mono, &mut self.out),
            None => self.out.extend_from_slice(&self.mono),
        }
        if self.out.is_empty() {
            return;
        }
        // History (pre-roll).
        self.history.extend(self.out.iter().copied());
        let excess = self.history.len().saturating_sub(self.history_cap);
        self.history.drain(..excess);
        let at = self.total;
        self.total += self.out.len() as u64;
        if let Some(tap) = self.tap.as_mut() {
            tap(at, &self.out);
        }

        // Session.
        let max = (self.opts.max_session.as_secs_f64() * SAMPLE_RATE as f64) as usize;
        let mut ended = false;
        let mut limit = false;
        let mut level_out = None;
        if let Some(s) = self.session.as_mut() {
            let take = self.out.len().min(max.saturating_sub(s.samples));
            if take > 0 {
                let block = &self.out[..take];
                s.samples += take;
                let sq = block.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>();
                s.level_sq += sq;
                s.level_n += take;
                let rms = (sq / take as f64).sqrt() as f32;
                if s.released.is_none() {
                    s.peak_rms = s.peak_rms.max(rms);
                }
                // Voicing during the post-roll: same relative threshold as the segmenter's pauses.
                let quiet = crate::segmenter::QUIET_CAP
                    .min(s.peak_rms * crate::segmenter::QUIET_RELATIVE)
                    .max(1e-5);
                s.voicing = rms > quiet;
                if !s.deliver(block) {
                    ended = true; // reader dropped
                }
            }
            if s.samples >= max && s.released.is_none() {
                s.limit_hit.store(true, Ordering::SeqCst);
                ended = true;
                limit = true;
            }
            if s.last_level.elapsed() >= LEVEL_EVERY && s.level_n > 0 {
                let level = level_from_rms(((s.level_sq / s.level_n as f64).sqrt()) as f32);
                s.level_sq = 0.0;
                s.level_n = 0;
                s.last_level = Instant::now();
                if let Some(f) = &s.level_fn {
                    f(level);
                }
                level_out = Some(level);
            }
        }
        if let Some(level) = level_out {
            *self.shared.level.lock().unwrap() = level;
            if let Some(f) = &self.opts.on_level {
                f(level);
            }
        }
        if limit {
            self.emit(CaptureEvent::LimitReached);
        }
        if ended {
            self.end_session();
        }
    }
}

#[cfg(target_os = "macos")]
impl Pump {
    /// Bring a plain stream up to voice processing without a gap: start it on a helper thread
    /// while the plain one keeps capturing, then swap between sessions, keeping the history.
    fn upgrade_check(&mut self) {
        let plain = matches!(&self.open, Some(o) if !o.stream.voice_processing());
        if !plain || self.released || !self.voice_wanted() {
            self.upgrade = None;
            self.upgrade_ready = None;
            return;
        }
        if let Some(rx) = &self.upgrade {
            match rx.try_recv() {
                Ok(Ok(v)) => {
                    self.upgrade = None;
                    let resampler = if v.rate == SAMPLE_RATE {
                        None
                    } else {
                        StreamResampler::new(v.rate, SAMPLE_RATE).ok()
                    };
                    self.upgrade_ready = Some(Open {
                        stream: InputStream::Voice(v.stream),
                        consumer: v.consumer,
                        cb: v.cb,
                        name: v.name,
                        channels: 1,
                        resampler,
                        opened_at: Instant::now(),
                    });
                }
                Ok(Err(e)) => {
                    self.upgrade = None;
                    self.voice_failed = true;
                    tracing::warn!(error = %e, "voice processing unavailable; keeping the plain microphone");
                }
                Err(_) => {}
            }
        } else if self.upgrade_ready.is_none() && self.session.is_none() {
            let (device, voice_input) = self.route();
            let Some(voice_input) = voice_input else {
                return; // a named device: plain capture only
            };
            let name = device
                .or_else(|| self.open.as_ref().map(|o| o.name.clone()))
                .unwrap_or_default();
            let pump = self.shared.pump.clone();
            let (tx, rx) = mpsc::channel();
            let spawned = std::thread::Builder::new()
                .name("ochre-voice-open".into())
                .spawn(move || {
                    let (producer, consumer) = rtrb::RingBuffer::<f32>::new(48_000 * RING_SECONDS);
                    let cb = new_callback_shared();
                    let r = crate::voice_mac::open(voice_input, producer, cb.clone(), pump.clone())
                        .map(|(stream, rate)| VoiceOpened {
                            stream,
                            rate,
                            consumer,
                            cb,
                            name,
                        });
                    let _ = tx.send(r);
                    pump.unpark();
                });
            if spawned.is_ok() {
                self.upgrade = Some(rx);
            }
        }
        if self.session.is_none()
            && let Some(new) = self.upgrade_ready.take()
        {
            self.drain(); // everything the plain stream has goes into the history first
            if let Some(old) = self.open.replace(new) {
                old.stream.pause();
            }
            if let Some(o) = &self.open {
                tracing::info!(device = %o.name, "capture switched to voice processing");
                *self.shared.current.lock().unwrap() = Some((o.name.clone(), 48_000, 1));
            }
        }
    }
}

fn new_callback_shared() -> Arc<CallbackShared> {
    Arc::new(CallbackShared {
        epoch: Instant::now(),
        last_cb_us: AtomicU64::new(0),
        lost: AtomicBool::new(false),
        overruns: AtomicU64::new(0),
    })
}

/// macOS voice processing on the default input; None (after a warning) falls back to cpal.
#[cfg(target_os = "macos")]
fn open_voice(name: &str, device: Option<u32>, pump: &Thread) -> Option<Open> {
    let rate = 48_000usize;
    let (producer, consumer) = rtrb::RingBuffer::<f32>::new(rate * RING_SECONDS);
    let cb = new_callback_shared();
    match crate::voice_mac::open(device, producer, cb.clone(), pump.clone()) {
        Ok((stream, rate)) => {
            let resampler = if rate == SAMPLE_RATE {
                None
            } else {
                Some(StreamResampler::new(rate, SAMPLE_RATE).ok()?)
            };
            Some(Open {
                stream: InputStream::Voice(stream),
                consumer,
                cb,
                name: name.to_string(),
                channels: 1,
                resampler,
                opened_at: Instant::now(),
            })
        }
        Err(e) => {
            tracing::warn!(error = %e, "voice processing unavailable; using the plain microphone");
            None
        }
    }
}

/// `voice`: try voice processing first (Some(None) = the default input, Some(Some(id)) = that
/// device), falling back to cpal.
fn open_stream(device: Option<&str>, voice: Option<Option<u32>>, pump: Thread) -> Result<Open> {
    let host = cpal::default_host();
    let dev =
        resolve_device(&host, device).ok_or_else(|| Error::Audio("no input device".into()))?;
    let name = dev.name().unwrap_or_else(|_| "input".into());
    #[cfg(target_os = "macos")]
    if let Some(voice_device) = voice
        && let Some(open) = open_voice(&name, voice_device, &pump)
    {
        return Ok(open);
    }
    #[cfg(not(target_os = "macos"))]
    let _ = voice;
    let supported = dev
        .default_input_config()
        .map_err(|e| Error::Audio(format!("{name}: {e}")))?;
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.into();
    let channels = config.channels as usize;
    let rate = config.sample_rate.0;
    let (producer, consumer) =
        rtrb::RingBuffer::<f32>::new(rate as usize * channels * RING_SECONDS);
    let cb = new_callback_shared();
    let stream = match format {
        SampleFormat::F32 => build::<f32>(&dev, &config, producer, cb.clone(), pump),
        SampleFormat::I16 => build::<i16>(&dev, &config, producer, cb.clone(), pump),
        SampleFormat::U16 => build::<u16>(&dev, &config, producer, cb.clone(), pump),
        SampleFormat::I32 => build::<i32>(&dev, &config, producer, cb.clone(), pump),
        SampleFormat::I8 => build::<i8>(&dev, &config, producer, cb.clone(), pump),
        SampleFormat::U8 => build::<u8>(&dev, &config, producer, cb.clone(), pump),
        SampleFormat::F64 => build::<f64>(&dev, &config, producer, cb.clone(), pump),
        other => {
            return Err(Error::Audio(format!(
                "{name}: unsupported sample format {other:?}"
            )));
        }
    }
    .map_err(|e| Error::Audio(format!("{name}: {e}")))?;
    stream
        .play()
        .map_err(|e| Error::Audio(format!("{name}: {e}")))?;
    let resampler = if rate == SAMPLE_RATE {
        None
    } else {
        Some(StreamResampler::new(rate, SAMPLE_RATE)?)
    };
    Ok(Open {
        stream: InputStream::Cpal(stream),
        consumer,
        cb,
        name,
        channels,
        resampler,
        opened_at: Instant::now(),
    })
}

fn build<T>(
    dev: &cpal::Device,
    config: &cpal::StreamConfig,
    mut producer: rtrb::Producer<f32>,
    cb: Arc<CallbackShared>,
    pump: Thread,
) -> std::result::Result<cpal::Stream, cpal::BuildStreamError>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let err_cb = cb.clone();
    let err_pump = pump.clone();
    dev.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            // Real-time path: convert + copy into the ring, stamp, wake the pump. Nothing else.
            let n = data.len().min(producer.slots());
            if n < data.len() {
                cb.overruns.fetch_add(1, Ordering::Relaxed);
            }
            if let Ok(mut chunk) = producer.write_chunk_uninit(n) {
                let (a, b) = chunk.as_mut_slices();
                let (d1, d2) = data[..n].split_at(a.len());
                for (dst, s) in a.iter_mut().zip(d1) {
                    dst.write(s.to_sample::<f32>());
                }
                for (dst, s) in b.iter_mut().zip(d2) {
                    dst.write(s.to_sample::<f32>());
                }
                // SAFETY: every slot of both slices was written above (`a.len() + b.len() == n`).
                unsafe { chunk.commit_all() };
            }
            cb.last_cb_us
                .store(cb.epoch.elapsed().as_micros() as u64, Ordering::Relaxed);
            pump.unpark();
        },
        move |e| {
            tracing::warn!(error = %e, "input stream error");
            if matches!(e, cpal::StreamError::DeviceNotAvailable) {
                err_cb.lost.store(true, Ordering::Relaxed);
                err_pump.unpark();
            }
        },
        None,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_start_in_history() {
        // history holds absolute samples [900, 1000)
        assert_eq!(history_start(&Start::Preroll(30), 100, 1000), 70);
        assert_eq!(history_start(&Start::Preroll(500), 100, 1000), 0);
        assert_eq!(history_start(&Start::At(950), 100, 1000), 50);
        assert_eq!(history_start(&Start::At(10), 100, 1000), 0); // older than the history
        assert_eq!(history_start(&Start::At(5000), 100, 1000), 100); // the future: from now
        assert_eq!(history_start(&Start::At(0), 0, 0), 0);
    }

    #[test]
    fn level_mapping() {
        assert_eq!(level_from_rms(0.0), 0.0);
        assert_eq!(level_from_rms(0.0005), 0.0); // -66 dBFS
        assert!((level_from_rms(10f32.powf(-36.0 / 20.0)) - 0.5).abs() < 1e-3);
        assert_eq!(level_from_rms(1.0), 1.0);
    }

    /// Opens the real default microphone (no audio output, no focus change). Checks the pre-roll
    /// and that blocks keep flowing at ~16 kHz.
    #[test]
    #[ignore = "hardware: opens the default microphone"]
    fn warm_capture_preroll_and_rate() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let ev = events.clone();
        let levels = Arc::new(Mutex::new(0usize));
        let lv = levels.clone();
        let cap = Capture::start(CaptureOptions {
            on_event: Some(Arc::new(move |e| ev.lock().unwrap().push(e.clone()))),
            on_level: Some(Arc::new(move |_| *lv.lock().unwrap() += 1)),
            ..Default::default()
        })
        .unwrap();
        std::thread::sleep(Duration::from_millis(1500)); // warm up and fill history
        assert!(
            cap.current_device().is_some(),
            "events: {:?}",
            events.lock().unwrap()
        );
        let t = Instant::now();
        let session = cap.begin(150);
        let first = session.recv().unwrap();
        let first_ms = t.elapsed().as_millis();
        assert!(
            (2300..=2500).contains(&first.len()),
            "pre-roll block {} samples",
            first.len()
        );
        std::thread::sleep(Duration::from_millis(1000));
        cap.end();
        let rest: usize = session.map(|b| b.len()).sum();
        let total = first.len() + rest;
        eprintln!(
            "first block after {first_ms} ms; {total} samples for ~1.15 s; {:?}",
            cap.current_device()
        );
        assert!((17_000..=20_500).contains(&total), "got {total} samples");
        assert!(
            *levels.lock().unwrap() >= 10,
            "level callbacks: {}",
            levels.lock().unwrap()
        );
    }

    /// The orchestrator seam: end() returns fast, the post-roll keeps flowing, then the sink is
    /// dropped and wait_closed() returns; levels arrive at ≥ 30 Hz; cancel drops at once.
    #[test]
    #[ignore = "hardware: opens the default microphone"]
    fn warm_mic_release_post_roll_and_cancel() {
        struct Counter(Arc<Mutex<(usize, bool)>>);
        impl Drop for Counter {
            fn drop(&mut self) {
                self.0.lock().unwrap().1 = true; // sink dropped
            }
        }
        let mic = WarmMic::start(CaptureOptions::default()).unwrap();
        std::thread::sleep(Duration::from_millis(1200));
        let state = Arc::new(Mutex::new((0usize, false)));
        let counter = Counter(state.clone());
        let levels = Arc::new(Mutex::new(0usize));
        let lv = levels.clone();
        mic.begin(
            150,
            Box::new(move |pcm: &[f32]| counter.0.lock().unwrap().0 += pcm.len()),
            Arc::new(move |_| *lv.lock().unwrap() += 1),
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(1000));
        let t = Instant::now();
        let ms = mic.end().unwrap();
        let end_ms = t.elapsed().as_secs_f64() * 1000.0;
        let at_release = state.lock().unwrap().0;
        assert!(
            !state.lock().unwrap().1,
            "sink still alive during post-roll"
        );
        mic.wait_closed(Duration::from_millis(600));
        let closed_ms = t.elapsed().as_secs_f64() * 1000.0;
        let (total, dropped) = *state.lock().unwrap();
        let post = (total - at_release) as f64 / 16.0;
        let hz = *levels.lock().unwrap() as f64 / 1.0;
        eprintln!(
            "end() {end_ms:.2} ms, session {ms} ms, post-roll {post:.0} ms of audio, closed after {closed_ms:.0} ms, level {hz:.0}/s"
        );
        assert!(end_ms < 5.0, "end() took {end_ms} ms");
        assert!(dropped);
        assert!((100.0..=320.0).contains(&post), "post-roll {post} ms");
        assert!((140.0..=360.0).contains(&closed_ms));
        assert!(hz >= 28.0, "level rate {hz}");

        let state = Arc::new(Mutex::new((0usize, false)));
        let counter = Counter(state.clone());
        mic.begin(
            150,
            Box::new(move |pcm: &[f32]| counter.0.lock().unwrap().0 += pcm.len()),
            Arc::new(|_| {}),
        )
        .unwrap();
        std::thread::sleep(Duration::from_millis(300));
        let t = Instant::now();
        mic.cancel();
        mic.wait_closed(Duration::from_millis(600));
        eprintln!(
            "cancel → sink dropped in {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
        assert!(state.lock().unwrap().1);
        assert!(t.elapsed() < Duration::from_millis(30));
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "hardware: opens the default microphone"]
    fn mac_idle_release_and_cold_start() {
        let cap = Capture::start(CaptureOptions {
            voice_processing: true,
            idle_release: Some(Duration::from_millis(1500)),
            ..Default::default()
        })
        .unwrap();
        let wait_for = |f: &dyn Fn() -> bool, secs: u64| {
            let t = Instant::now();
            while !f() {
                assert!(t.elapsed() < Duration::from_secs(secs), "timed out");
                std::thread::sleep(Duration::from_millis(10));
            }
        };
        wait_for(&|| cap.current_device().is_some(), 5);
        let mut gaps = Vec::new();
        for _ in 0..20 {
            // idle release closes it
            wait_for(&|| cap.current_device().is_none(), 5);
            std::thread::sleep(Duration::from_millis(200));
            // key down: a cold session; time to the first captured sample
            let t = Instant::now();
            let session = cap.begin(150);
            let first = session.recv().unwrap();
            gaps.push(t.elapsed().as_millis() as u64);
            assert!(!first.is_empty());
            std::thread::sleep(Duration::from_millis(400));
            cap.end();
            for _ in session {}
        }
        gaps.sort();
        eprintln!(
            "cold start after idle release, key-down -> first sample: p50 {} ms, p95 {} ms, max {} ms ({gaps:?})",
            gaps[gaps.len() / 2],
            gaps[gaps.len() * 95 / 100],
            gaps[gaps.len() - 1]
        );
        assert!(gaps[gaps.len() / 2] < 150);
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "hardware: opens the default microphone"]
    fn mac_cold_session_upgrades_to_voice_processing() {
        let cap = Capture::start(CaptureOptions {
            voice_processing: true,
            idle_release: Some(Duration::from_millis(800)),
            ..Default::default()
        })
        .unwrap();
        let t = Instant::now();
        while cap.current_device().is_none() {
            assert!(t.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(10));
        }
        std::thread::sleep(Duration::from_millis(1500)); // released
        assert!(cap.current_device().is_none(), "idle release");
        let session = cap.begin(150);
        let _ = session.recv().unwrap();
        cap.end();
        for _ in session {}
        // between sessions the plain stream is replaced by a voice-processed one
        std::thread::sleep(Duration::from_millis(600));
        assert!(
            cap.current_device().is_some(),
            "warm again after a dictation"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "hardware: opens the default microphone"]
    fn mac_voice_processing_captures() {
        let cap = Capture::start(CaptureOptions {
            voice_processing: true,
            ..Default::default()
        })
        .unwrap();
        let t = Instant::now();
        while cap.current_device().is_none() {
            assert!(t.elapsed() < Duration::from_secs(5), "never opened");
            std::thread::sleep(Duration::from_millis(10));
        }
        let opened = t.elapsed();
        let session = cap.begin(150);
        std::thread::sleep(Duration::from_millis(1000));
        cap.end();
        let total: usize = session.map(|b| b.len()).sum();
        eprintln!(
            "voice processing: open {} ms, {:?}, {total} samples for ~1.15 s",
            opened.as_millis(),
            cap.current_device()
        );
        assert!((16_000..21_000).contains(&total), "{total}");
    }

    #[test]
    #[ignore = "hardware: opens the default microphone"]
    fn cold_capture_opens_on_begin() {
        let cap = Capture::start(CaptureOptions {
            warm: false,
            ..Default::default()
        })
        .unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            cap.current_device().is_none(),
            "cold mode must not open the mic while idle"
        );
        let t = Instant::now();
        let session = cap.begin(150);
        let first = session.recv().unwrap();
        eprintln!(
            "cold open → first audio in {} ms ({} samples)",
            t.elapsed().as_millis(),
            first.len()
        );
        std::thread::sleep(Duration::from_millis(500));
        cap.end();
        let total: usize = first.len() + session.map(|b| b.len()).sum::<usize>();
        assert!(total > 4_000, "{total}");
        std::thread::sleep(Duration::from_millis(300));
        assert!(
            cap.current_device().is_none(),
            "closed again after the session"
        );
    }

    #[test]
    #[ignore = "hardware: enumerates input devices"]
    fn lists_devices() {
        let d = list_devices();
        eprintln!("{d:#?}");
        assert!(!d.is_empty());
    }
}
