//! Decode while the user is talking, so release waits on a short tail only
//! (docs/latency.md rules 2 and 9).
//!
//! Audio is cut into phrases during capture and each phrase is decoded at once on the decode
//! thread. The default [`CutConfig`] cuts:
//!
//! * at any pause of ≥ 200 ms once ≥ 2 s has accumulated;
//! * after 8 s, at any pause of ≥ 100 ms;
//! * a phrase never exceeds 20 s: someone who never pauses is cut at the quietest 20 ms frame of
//!   the last 3 s. (Cutting through speech costs Parakeet a lot of accuracy, so this is a safety
//!   net, not the normal path.)
//!
//! Phrases with less than 60 ms of non-quiet audio are dropped, not decoded: Parakeet
//! hallucinates "Yeah." / "Okay." on near-silence.
//!
//! On release ([`Segmenter::release`]) everything up to the last quiet point is submitted at
//! once, so the tail decode overlaps the mic's post-roll; [`Segmenter::finish`] then decodes the
//! remainder only if it contains speech.
//!
//! "Quiet" is relative to the utterance's peak frame RMS (3 %), capped at -54 dBFS, which keeps
//! softly spoken words; with a VAD attached, a frame the VAD scores as non-speech is quiet too.
//! [`CutConfig::long_pause`] keeps the older, more conservative 5 s / 20 s / 30 s rules for
//! comparison.
//!
//! [`DecodeThread`] is the single process-wide decode thread (raised priority, jobs over a
//! channel, nothing polls). [`PhraseSegmenter`] is the factory the orchestrator's
//! `SegmenterFactory` seam wraps.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ochre_core::stt::{SttEngine, SttOptions, SttResult};
use ochre_core::{Error, Result, SAMPLE_RATE};

use crate::vad::SileroVad;

pub const FRAME_S: f32 = 0.02;
/// -54 dBFS.
pub const QUIET_CAP: f32 = 0.002;
/// Of the utterance's peak frame RMS.
pub const QUIET_RELATIVE: f32 = 0.03;
/// VAD probability below which a frame counts as a pause.
pub const VAD_QUIET_BELOW: f32 = 0.3;
/// A remainder needs at least this many non-quiet frames (60 ms) to be worth decoding.
const MIN_SPEECH_FRAMES: usize = 3;

/// Where phrases are cut.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutConfig {
    /// A pause only ends a phrase once the phrase is at least this long.
    pub min_phrase_s: f32,
    /// Pause length that ends a phrase.
    pub pause_s: f32,
    /// After this, a shorter pause (`soft_pause_s`) is enough (e.g. a 20 s rule).
    pub soft_phrase_s: Option<f32>,
    pub soft_pause_s: f32,
    /// Hard limit: cut at the quietest frame of the last `cut_window_s`.
    pub max_phrase_s: f32,
    pub cut_window_s: f32,
    /// On release, a quiet frame this recent is "the last quiet point"; otherwise the quietest
    /// frame of this window is used.
    pub release_window_s: f32,
}

impl Default for CutConfig {
    fn default() -> Self {
        // Measured with `ochre-stt/examples/segment_eval` on LibriSpeech (Parakeet v3 int8): vs
        // whole-utterance decoding this costs +0.4 WER points on 6 s clips and leaves a tail of
        // p95 1.6 s (65 ms decode). Forced cuts through speech every few seconds cost 3-18 points
        // and overlapped re-decoding was worse still, so the hard cap is a rare safety net only.
        Self {
            min_phrase_s: 2.0,
            pause_s: 0.2,
            soft_phrase_s: Some(8.0),
            soft_pause_s: 0.10,
            max_phrase_s: 20.0,
            cut_window_s: 3.0,
            release_window_s: 1.0,
        }
    }
}

impl CutConfig {
    /// Conservative long-pause rules: 5 s + 320 ms pause, 20 s + 120 ms, hard cut by 30 s.
    pub fn long_pause() -> Self {
        Self {
            min_phrase_s: 5.0,
            pause_s: 0.32,
            soft_phrase_s: Some(20.0),
            soft_pause_s: 0.12,
            max_phrase_s: 30.0,
            cut_window_s: 5.0,
            release_window_s: 1.0,
        }
    }

    /// Never cut during capture (whole-utterance decode, for accuracy baselines).
    pub fn never() -> Self {
        Self {
            min_phrase_s: f32::INFINITY,
            max_phrase_s: f32::INFINITY,
            ..Self::default()
        }
    }
}

/// Per-session speech gate for the cutter: Silero probabilities mapped onto 20 ms frames.
struct VadGate {
    vad: Arc<Mutex<SileroVad>>,
    prob: f32,
}

/// Splits a stream of 16 kHz mono f32 blocks into phrases. No threads, no I/O.
pub struct PhraseCutter {
    cfg: CutConfig,
    rate: usize,
    width: usize,
    /// The open phrase (processed frames followed by a partial-frame carry).
    buf: Vec<f32>,
    /// Samples of `buf` already processed in whole frames.
    pos: usize,
    levels: Vec<f32>,
    quiet_flags: Vec<bool>,
    quiet: usize,
    peak_rms: f32,
    vad: Option<VadGate>,
}

impl PhraseCutter {
    pub fn new(rate: u32) -> Self {
        Self::with_config(rate, CutConfig::default())
    }

    pub fn with_config(rate: u32, cfg: CutConfig) -> Self {
        let rate = rate as usize;
        Self {
            cfg,
            rate,
            width: ((rate as f32 * FRAME_S).round() as usize).max(1),
            buf: Vec::with_capacity(rate * 8),
            pos: 0,
            levels: Vec::new(),
            quiet_flags: Vec::new(),
            quiet: 0,
            peak_rms: 0.0,
            vad: None,
        }
    }

    /// Use Silero as an additional pause detector (reset for this session).
    pub fn with_vad(mut self, vad: Arc<Mutex<SileroVad>>) -> Self {
        vad.lock().unwrap().reset();
        self.vad = Some(VadGate { vad, prob: 1.0 });
        self
    }

    fn samples(&self, s: f32) -> usize {
        if s.is_finite() {
            (self.rate as f32 * s) as usize
        } else {
            usize::MAX
        }
    }

    /// Add audio; returns any phrases completed by it.
    pub fn push(&mut self, block: &[f32]) -> Vec<Vec<f32>> {
        self.buf.extend_from_slice(block);
        let mut out = Vec::new();
        while self.buf.len() - self.pos >= self.width {
            let frame = &self.buf[self.pos..self.pos + self.width];
            let rms = (frame.iter().map(|x| x * x).sum::<f32>() / frame.len() as f32).sqrt();
            let peak = frame.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            if let Some(g) = self.vad.as_mut()
                && let Ok(p) = g.vad.lock().unwrap().probs(frame)
                && let Some(&last) = p.last()
            {
                g.prob = last;
            }
            self.pos += self.width;
            self.levels.push(rms);
            self.peak_rms = self.peak_rms.max(rms);
            let threshold = QUIET_CAP.min(self.peak_rms * QUIET_RELATIVE);
            let energy_quiet = rms <= threshold && peak <= (threshold * 4.0).max(1e-6);
            let quiet = energy_quiet || self.vad.as_ref().is_some_and(|g| g.prob < VAD_QUIET_BELOW);
            self.quiet_flags.push(quiet);
            self.quiet = if quiet { self.quiet + self.width } else { 0 };
            let len = self.pos;
            let c = self.cfg;
            if (len >= self.samples(c.min_phrase_s) && self.quiet >= self.samples(c.pause_s))
                || c.soft_phrase_s.is_some_and(|s| {
                    len >= self.samples(s) && self.quiet >= self.samples(c.soft_pause_s)
                })
            {
                out.extend(self.take_speech(self.levels.len()));
            } else if len >= self.samples(c.max_phrase_s) {
                let window = self
                    .levels
                    .len()
                    .min(((c.cut_window_s / FRAME_S).round() as usize).max(1));
                let cut = self.quietest_in_last(window);
                out.extend(self.take_speech(cut + 1));
            }
        }
        out
    }

    fn quietest_in_last(&self, window: usize) -> usize {
        let start = self.levels.len() - window.min(self.levels.len());
        let mut best = start;
        for i in start..self.levels.len() {
            if self.levels[i] < self.levels[best] {
                best = i;
            }
        }
        best
    }

    /// Cut the first `count` processed frames off; None (the audio is dropped) when they hold
    /// less than 60 ms of non-quiet audio: decoding near-silence makes Parakeet hallucinate
    /// ("Yeah.", "Okay.").
    fn take_speech(&mut self, count: usize) -> Option<Vec<f32>> {
        let loud = self.quiet_flags[..count].iter().filter(|q| !**q).count();
        let p = self.take(count);
        (loud >= MIN_SPEECH_FRAMES).then_some(p)
    }

    /// `flush`, but None when the open phrase holds no speech.
    pub fn flush_speech(&mut self) -> Option<Vec<f32>> {
        let speech = self.pending_has_speech();
        let p = self.flush();
        if speech { p } else { None }
    }

    /// Cut the first `count` processed frames off as a phrase.
    fn take(&mut self, count: usize) -> Vec<f32> {
        let n = count * self.width;
        let phrase = self.buf[..n].to_vec();
        self.buf.drain(..n);
        self.pos -= n;
        self.levels.drain(..count);
        self.quiet_flags.drain(..count);
        self.quiet = 0;
        phrase
    }

    /// Release: everything up to the last quiet point (a quiet frame within `release_window_s`,
    /// else the quietest frame of that window). Whatever follows stays open for the post-roll.
    /// When the audio already ends in quiet, that is everything. None if nothing is open.
    pub fn release_cut(&mut self) -> Option<Vec<f32>> {
        if self.levels.is_empty() {
            return self.flush_speech();
        }
        let window = self
            .levels
            .len()
            .min(((self.cfg.release_window_s / FRAME_S).round() as usize).max(1));
        let start = self.levels.len() - window;
        let last_quiet = (start..self.levels.len())
            .rev()
            .find(|&i| self.quiet_flags[i]);
        let cut = match last_quiet {
            Some(i) if i + 1 == self.levels.len() => return self.flush_speech(),
            Some(i) => i,
            None => self.quietest_in_last(window),
        };
        self.take_speech(cut + 1)
    }

    /// The open phrase holds enough non-quiet audio to be worth decoding.
    pub fn pending_has_speech(&self) -> bool {
        let loud = self.quiet_flags.iter().filter(|q| !**q).count();
        if loud >= MIN_SPEECH_FRAMES {
            return true;
        }
        // The partial-frame carry is not classified yet; judge it by energy alone.
        let carry = &self.buf[self.pos..];
        let threshold = QUIET_CAP.min(self.peak_rms * QUIET_RELATIVE).max(1e-6);
        loud + usize::from(carry.iter().any(|x| x.abs() > threshold * 4.0)) >= MIN_SPEECH_FRAMES
    }

    /// Copy of the open (not yet cut) phrase, for live partials.
    pub fn pending(&self) -> Vec<f32> {
        self.buf.clone()
    }

    pub fn pending_seconds(&self) -> f32 {
        self.buf.len() as f32 / self.rate as f32
    }

    /// Everything not yet cut (end of capture), or None if empty.
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        self.levels.clear();
        self.quiet_flags.clear();
        self.pos = 0;
        self.quiet = 0;
        if self.buf.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buf))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Decode thread
// ---------------------------------------------------------------------------------------------

/// The engine slot owned by the decode thread.
pub type EngineSlot = Option<Box<dyn SttEngine>>;
type Job = Box<dyn FnOnce(&mut EngineSlot) + Send>;

struct DecodeInner {
    tx: Mutex<Option<Sender<Job>>>,
    /// Jobs queued or running.
    pending: AtomicUsize,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl Drop for DecodeInner {
    fn drop(&mut self) {
        self.tx.lock().unwrap().take();
        if let Some(h) = self.handle.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

/// The one thread that runs every decode, at raised priority (docs/latency.md rule 1).
/// Engines are not re-entrant and a single worker keeps CPU use predictable, so a finishing
/// session and a new one never decode concurrently. It can own an engine (`EngineSlot`) or run
/// any transcribe closure. Cheap to clone; the thread exits when the last handle drops.
#[derive(Clone)]
pub struct DecodeThread {
    inner: Arc<DecodeInner>,
}

impl DecodeThread {
    pub fn spawn(engine: EngineSlot) -> Self {
        let (tx, rx) = mpsc::channel::<Job>();
        let handle = std::thread::Builder::new()
            .name("ochre-decode".into())
            .spawn(move || {
                if !crate::priority::boost_current_thread() {
                    tracing::debug!("could not raise decode thread priority");
                }
                let mut slot = engine;
                while let Ok(job) = rx.recv() {
                    job(&mut slot);
                }
            })
            .expect("spawn decode thread");
        Self {
            inner: Arc::new(DecodeInner {
                tx: Mutex::new(Some(tx)),
                pending: AtomicUsize::new(0),
                handle: Mutex::new(Some(handle)),
            }),
        }
    }

    /// Run `f` on the decode thread with the engine slot; the result arrives on the receiver.
    /// Use it to load or swap the engine (`*slot = Some(new)`) without racing a decode.
    pub fn exec<R: Send + 'static>(
        &self,
        f: impl FnOnce(&mut EngineSlot) -> R + Send + 'static,
    ) -> Receiver<R> {
        let (rtx, rrx) = mpsc::channel();
        let inner = Arc::downgrade(&self.inner);
        self.inner.pending.fetch_add(1, Ordering::SeqCst);
        let job: Job = Box::new(move |slot| {
            let r = f(slot);
            if let Some(i) = inner.upgrade() {
                i.pending.fetch_sub(1, Ordering::SeqCst);
            }
            let _ = rtx.send(r);
        });
        let sent = self
            .inner
            .tx
            .lock()
            .unwrap()
            .as_ref()
            .map(|tx| tx.send(job).is_ok())
            .unwrap_or(false);
        if !sent {
            self.inner.pending.fetch_sub(1, Ordering::SeqCst);
        }
        rrx
    }

    /// Transcribe with the slot's engine on the decode thread.
    pub fn transcribe(&self, pcm: Vec<f32>, opts: SttOptions) -> Receiver<Result<SttResult>> {
        self.exec(move |slot| match slot {
            Some(e) => e.transcribe(&pcm, &opts),
            None => Err(Error::Model("no speech-to-text engine loaded".into())),
        })
    }

    /// No job queued or running.
    pub fn is_idle(&self) -> bool {
        self.inner.pending.load(Ordering::SeqCst) == 0
    }
}

// ---------------------------------------------------------------------------------------------
// Segmenter
// ---------------------------------------------------------------------------------------------

/// Same shape as the orchestrator's `ochre::seams::TranscribeFn`.
pub type TranscribeFn = Arc<dyn Fn(&[f32]) -> Result<SttResult> + Send + Sync>;
/// `(text, stable_chars)`: the HUD bubble text; the first `stable_chars` will not change.
pub type PartialFn = Arc<dyn Fn(&str, usize) + Send + Sync>;
/// Called on the decode thread as each phrase is decoded (hands-free spots control phrases here).
pub type SegmentFn = Arc<dyn Fn(usize, &SttResult) + Send + Sync>;
/// Speech gate run on the decode thread before a phrase is decoded (e.g. Silero VAD); a phrase
/// it rejects becomes an empty result, which keeps Whisper from hallucinating on silence.
pub type SpeechCheckFn = Arc<dyn Fn(&[f32]) -> bool + Send + Sync>;

/// What a phrase is decoded with.
#[derive(Clone)]
pub enum Transcriber {
    /// A closure (the orchestrator's engine call), run on the decode thread.
    Fn(TranscribeFn),
    /// The engine in the decode thread's slot, with these options.
    Engine(SttOptions),
}

#[derive(Clone, Default)]
pub struct SegmenterOptions {
    pub cut: CutConfig,
    pub on_segment: Option<SegmentFn>,
    pub on_partial: Option<PartialFn>,
    /// Decode the open phrase for the HUD at most this often, and only while the decode thread is
    /// idle (a preview never delays a real phrase). None = no opportunistic partials.
    pub partial_interval: Option<Duration>,
    pub speech_check: Option<SpeechCheckFn>,
    /// Silero as an extra pause detector for the cutter.
    pub vad: Option<Arc<Mutex<SileroVad>>>,
}

/// The joined result of a session plus the latency numbers the orchestrator logs.
#[derive(Debug, Clone, Default)]
pub struct SessionTranscript {
    pub text: String,
    pub phrases: Vec<SttResult>,
    pub audio_ms: u64,
    /// Audio decoded after release (the release cut plus any remainder).
    pub tail_audio_ms: u64,
    /// `release()` (or `finish()` without one) → last phrase decoded.
    pub stt_tail_ms: u64,
    /// Decode work across all phrases (mostly overlapped with speech).
    pub stt_total_ms: u64,
}

struct Shared {
    canceled: AtomicBool,
    finished: AtomicBool,
    committed: Mutex<Vec<String>>,
}

/// One dictation session at 16 kHz mono. This is also the orchestrator's `SegmentSession`.
pub struct Segmenter {
    decoder: DecodeThread,
    transcriber: Transcriber,
    opts: SegmenterOptions,
    cutter: PhraseCutter,
    jobs: Vec<Receiver<Result<Option<SttResult>>>>,
    shared: Arc<Shared>,
    samples: usize,
    last_partial: Option<Instant>,
    released: Option<Instant>,
    tail_samples: usize,
}

impl Segmenter {
    pub fn new(decoder: DecodeThread, transcriber: Transcriber, opts: SegmenterOptions) -> Self {
        let mut cutter = PhraseCutter::with_config(SAMPLE_RATE, opts.cut);
        if let Some(v) = &opts.vad {
            cutter = cutter.with_vad(v.clone());
        }
        Self {
            decoder,
            transcriber,
            opts,
            cutter,
            jobs: Vec::new(),
            shared: Arc::new(Shared {
                canceled: AtomicBool::new(false),
                finished: AtomicBool::new(false),
                committed: Mutex::new(Vec::new()),
            }),
            samples: 0,
            last_partial: None,
            released: None,
            tail_samples: 0,
        }
    }

    /// Feed capture audio (16 kHz mono). Cheap: a few float ops per sample plus, when a phrase
    /// completes, one channel send.
    pub fn feed(&mut self, block: &[f32]) {
        self.samples += block.len();
        for phrase in self.cutter.push(block) {
            self.submit(phrase);
        }
        if self.released.is_none() {
            self.maybe_partial();
        }
    }

    /// Seconds of audio fed so far.
    pub fn seconds(&self) -> f32 {
        self.samples as f32 / SAMPLE_RATE as f32
    }

    /// The user released: submit everything up to the last quiet point now, so its decode
    /// overlaps the post-roll. Call once, right after the mic's `end()`.
    pub fn release(&mut self) {
        if self.released.is_some() {
            return;
        }
        self.released = Some(Instant::now());
        self.shared.finished.store(true, Ordering::SeqCst); // no more partial previews
        if let Some(p) = self.cutter.release_cut() {
            self.submit(p);
        }
    }

    fn submit(&mut self, phrase: Vec<f32>) {
        if self.released.is_some() {
            self.tail_samples += phrase.len();
        }
        let idx = self.jobs.len();
        let shared = self.shared.clone();
        let opts = self.opts.clone();
        let transcriber = self.transcriber.clone();
        let rx = self.decoder.exec(move |slot| -> Result<Option<SttResult>> {
            if shared.canceled.load(Ordering::SeqCst) {
                return Ok(None);
            }
            let duration_ms = (phrase.len() as u64 * 1000) / SAMPLE_RATE as u64;
            let result = if opts
                .speech_check
                .as_ref()
                .is_some_and(|check| !check(&phrase))
            {
                SttResult {
                    text: String::new(),
                    duration_ms,
                    processing_ms: 0,
                    language: None,
                }
            } else {
                match &transcriber {
                    Transcriber::Fn(f) => f(&phrase)?,
                    Transcriber::Engine(o) => slot
                        .as_ref()
                        .ok_or_else(|| Error::Model("no speech-to-text engine loaded".into()))?
                        .transcribe(&phrase, o)?,
                }
            };
            if shared.canceled.load(Ordering::SeqCst) {
                return Ok(None);
            }
            let text = {
                let mut c = shared.committed.lock().unwrap();
                let t = result.text.trim();
                if !t.is_empty() {
                    c.push(t.to_string());
                }
                c.join(" ")
            };
            if let Some(f) = &opts.on_segment {
                f(idx, &result);
            }
            if let Some(f) = &opts.on_partial {
                f(&text, text.len());
            }
            Ok(Some(result))
        });
        self.jobs.push(rx);
    }

    fn maybe_partial(&mut self) {
        let (Some(f), Some(interval)) = (self.opts.on_partial.clone(), self.opts.partial_interval)
        else {
            return;
        };
        if self.last_partial.is_some_and(|t| t.elapsed() < interval)
            || self.cutter.pending_seconds() < 1.0
        {
            return;
        }
        if !self.decoder.is_idle() {
            return; // never delay a real phrase for a preview
        }
        self.last_partial = Some(Instant::now());
        let audio = self.cutter.pending();
        let shared = self.shared.clone();
        let transcriber = self.transcriber.clone();
        drop(self.decoder.exec(move |slot| {
            if shared.canceled.load(Ordering::SeqCst) || shared.finished.load(Ordering::SeqCst) {
                return;
            }
            let r = match &transcriber {
                Transcriber::Fn(f) => f(&audio),
                Transcriber::Engine(o) => match slot.as_ref() {
                    Some(e) => e.transcribe(&audio, o),
                    None => return,
                },
            };
            let Ok(r) = r else { return };
            if shared.canceled.load(Ordering::SeqCst) || shared.finished.load(Ordering::SeqCst) {
                return;
            }
            let stable = shared.committed.lock().unwrap().join(" ");
            let interim = r.text.trim();
            let text = if interim.is_empty() {
                stable.clone()
            } else if stable.is_empty() {
                interim.to_string()
            } else {
                format!("{stable} {interim}")
            };
            f(&text, stable.len());
        }));
    }

    /// Decode what remains (only if it contains speech) and wait for every phrase, in order.
    /// Returns the first decode error.
    pub fn finish(mut self) -> Result<SessionTranscript> {
        let tail_start = *self.released.get_or_insert_with(Instant::now);
        self.shared.finished.store(true, Ordering::SeqCst);
        if self.cutter.pending_has_speech() {
            if let Some(rest) = self.cutter.flush() {
                self.tail_samples += rest.len();
                self.submit_counted(rest);
            }
        } else {
            self.cutter.flush();
        }
        let mut phrases = Vec::with_capacity(self.jobs.len());
        for rx in std::mem::take(&mut self.jobs) {
            match rx.recv() {
                Ok(Ok(Some(r))) => phrases.push(r),
                Ok(Ok(None)) => {}
                Ok(Err(e)) => return Err(e),
                Err(_) => return Err(Error::Other("decode thread stopped".into())),
            }
        }
        let text = phrases
            .iter()
            .map(|r| r.text.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(SessionTranscript {
            text,
            stt_total_ms: phrases.iter().map(|r| r.processing_ms).sum(),
            phrases,
            audio_ms: (self.samples as u64 * 1000) / SAMPLE_RATE as u64,
            tail_audio_ms: (self.tail_samples as u64 * 1000) / SAMPLE_RATE as u64,
            stt_tail_ms: tail_start.elapsed().as_millis() as u64,
        })
    }

    /// `submit` without double-counting tail samples (already counted by the caller).
    fn submit_counted(&mut self, phrase: Vec<f32>) {
        let n = phrase.len();
        self.submit(phrase);
        self.tail_samples -= n;
    }

    /// Drop everything; queued decodes become no-ops. Returns immediately.
    pub fn cancel(self) {
        self.shared.canceled.store(true, Ordering::SeqCst);
    }
}

/// Factory for dictation sessions: the orchestrator's `SegmenterFactory` seam. Owns (a handle
/// to) the process-wide decode thread.
#[derive(Clone)]
pub struct PhraseSegmenter {
    decoder: DecodeThread,
    cut: CutConfig,
    partial_interval: Option<Duration>,
    vad: Option<Arc<Mutex<SileroVad>>>,
    speech_check: Option<SpeechCheckFn>,
}

impl Default for PhraseSegmenter {
    fn default() -> Self {
        Self::new(DecodeThread::spawn(None))
    }
}

impl PhraseSegmenter {
    pub fn new(decoder: DecodeThread) -> Self {
        Self {
            decoder,
            cut: CutConfig::default(),
            partial_interval: Some(Duration::from_millis(700)),
            vad: None,
            speech_check: None,
        }
    }

    pub fn with_cut(mut self, cut: CutConfig) -> Self {
        self.cut = cut;
        self
    }

    /// None disables opportunistic partial decodes of the open phrase.
    pub fn with_partial_interval(mut self, every: Option<Duration>) -> Self {
        self.partial_interval = every;
        self
    }

    /// Silero for pause detection in the cutter.
    pub fn with_vad(mut self, vad: Arc<Mutex<SileroVad>>) -> Self {
        self.vad = Some(vad);
        self
    }

    /// Skip decoding phrases this rejects (e.g. a VAD `has_speech`).
    pub fn with_speech_check(mut self, check: SpeechCheckFn) -> Self {
        self.speech_check = Some(check);
        self
    }

    pub fn decoder(&self) -> &DecodeThread {
        &self.decoder
    }

    /// Mirrors `SegmenterFactory::start`.
    pub fn start(&self, transcribe: TranscribeFn, on_partial: Option<PartialFn>) -> Segmenter {
        self.start_inner(transcribe, on_partial, None)
    }

    /// `start`, plus `on_segment` for every cut phrase (never for a live preview).
    pub fn start_with_segments(
        &self,
        transcribe: TranscribeFn,
        on_partial: Option<PartialFn>,
        on_segment: SegmentFn,
    ) -> Segmenter {
        self.start_inner(transcribe, on_partial, Some(on_segment))
    }

    fn start_inner(
        &self,
        transcribe: TranscribeFn,
        on_partial: Option<PartialFn>,
        on_segment: Option<SegmentFn>,
    ) -> Segmenter {
        Segmenter::new(
            self.decoder.clone(),
            Transcriber::Fn(transcribe),
            SegmenterOptions {
                cut: self.cut,
                on_segment,
                on_partial,
                partial_interval: self.partial_interval,
                vad: self.vad.clone(),
                speech_check: self.speech_check.clone(),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ochre_core::events::EngineInfo;

    const R: usize = SAMPLE_RATE as usize;

    fn speech(secs: f32) -> Vec<f32> {
        (0..(secs * R as f32) as usize)
            .map(|i| (i as f32 * 0.07).sin() * 0.2 * (1.5 + (i as f32 * 0.0003).sin()))
            .collect()
    }

    fn silence(secs: f32) -> Vec<f32> {
        vec![0.0; (secs * R as f32) as usize]
    }

    fn secs(v: &[f32]) -> f32 {
        v.len() as f32 / R as f32
    }

    #[test]
    fn short_utterance_is_one_phrase_at_flush() {
        let mut c = PhraseCutter::new(SAMPLE_RATE);
        let mut audio = speech(1.5);
        audio.extend(silence(0.3)); // pause ends before 2 s: no cut
        audio.extend(speech(1.0));
        assert!(c.push(&audio).is_empty());
        assert_eq!(c.flush().unwrap().len(), audio.len());
    }

    #[test]
    fn cuts_at_first_200ms_pause_after_2s_and_keeps_every_sample() {
        let mut c = PhraseCutter::new(SAMPLE_RATE);
        let mut audio = speech(2.5);
        audio.extend(silence(0.25));
        audio.extend(speech(1.0));
        let mut phrases = Vec::new();
        for block in audio.chunks(160) {
            phrases.extend(c.push(block));
        }
        assert_eq!(phrases.len(), 1);
        assert!(
            (2.68..2.72).contains(&secs(&phrases[0])),
            "cut at {}",
            secs(&phrases[0])
        );
        let rest = c.flush().unwrap();
        assert_eq!(phrases[0].len() + rest.len(), audio.len());
    }

    #[test]
    fn a_pause_shorter_than_200ms_does_not_cut() {
        let mut c = PhraseCutter::new(SAMPLE_RATE);
        let mut audio = speech(2.5);
        audio.extend(silence(0.14));
        audio.extend(speech(1.0));
        assert!(c.push(&audio).is_empty());
    }

    #[test]
    fn nonstop_talker_is_cut_at_the_quietest_frame_by_max_phrase() {
        let mut c = PhraseCutter::new(SAMPLE_RATE);
        let mut audio = speech(18.4);
        // A dip (quiet but not a pause) at 18.4 s.
        audio.extend(speech(0.04).iter().map(|x| x * 0.05));
        audio.extend(speech(10.0));
        let phrases = c.push(&audio);
        assert_eq!(phrases.len(), 1);
        assert!(
            (18.4..18.46).contains(&secs(&phrases[0])),
            "cut at {}",
            secs(&phrases[0])
        );
        assert!(c.pending_seconds() <= 20.0);
    }

    #[test]
    fn after_8s_a_100ms_pause_is_enough() {
        let mut c = PhraseCutter::new(SAMPLE_RATE);
        let mut audio = speech(5.0);
        audio.extend(silence(0.12)); // too short before 8 s
        audio.extend(speech(4.0));
        audio.extend(silence(0.12)); // enough after 8 s
        audio.extend(speech(1.0));
        let phrases = c.push(&audio);
        assert_eq!(phrases.len(), 1);
        assert!(
            (9.2..9.25).contains(&secs(&phrases[0])),
            "cut at {}",
            secs(&phrases[0])
        );
    }

    #[test]
    fn near_silent_phrases_are_dropped_not_decoded() {
        let mut c = PhraseCutter::new(SAMPLE_RATE);
        let mut audio = speech(2.5);
        audio.extend(silence(3.0));
        c.push(&audio);
        let tail = c.release_cut();
        assert!(tail.is_none(), "trailing silence must not become a phrase");
    }

    #[test]
    fn long_pause_rules_still_available() {
        let mut c = PhraseCutter::with_config(SAMPLE_RATE, CutConfig::long_pause());
        let mut audio = speech(6.0);
        audio.extend(silence(0.4));
        audio.extend(speech(2.0));
        let phrases = c.push(&audio);
        assert_eq!(phrases.len(), 1);
        assert!((6.3..6.4).contains(&secs(&phrases[0])));
    }

    #[test]
    fn release_cut_takes_everything_when_audio_ends_quiet() {
        let mut c = PhraseCutter::new(SAMPLE_RATE);
        let mut audio = speech(1.5);
        audio.extend(silence(0.1));
        c.push(&audio);
        assert_eq!(c.release_cut().unwrap().len(), audio.len());
        assert!(c.release_cut().is_none());
        assert!(!c.pending_has_speech());
    }

    #[test]
    fn release_cut_stops_at_the_last_quiet_point_when_still_talking() {
        let mut c = PhraseCutter::new(SAMPLE_RATE);
        let mut audio = speech(1.0);
        audio.extend(silence(0.1));
        audio.extend(speech(0.3)); // still talking at release
        c.push(&audio);
        let cut = c.release_cut().unwrap();
        assert!((1.09..1.11).contains(&secs(&cut)), "cut at {}", secs(&cut));
        assert!(c.pending_has_speech());
        assert!((0.29..0.31).contains(&c.pending_seconds()));
    }

    /// Fake engine: text = "p<seconds>", sleeps 30 ms per call.
    struct Fake;
    impl SttEngine for Fake {
        fn info(&self) -> EngineInfo {
            EngineInfo {
                id: "fake".into(),
                label: "fake".into(),
                kind: "local".into(),
                models: vec![],
                default_model: String::new(),
                needs_key: false,
                note: String::new(),
                languages: String::new(),
            }
        }
        fn load(&mut self, _: ochre_core::stt::ProgressFn) -> Result<()> {
            Ok(())
        }
        fn transcribe(&self, pcm: &[f32], _: &SttOptions) -> Result<SttResult> {
            std::thread::sleep(Duration::from_millis(30));
            Ok(SttResult {
                text: format!("p{}", pcm.len() / R),
                duration_ms: 0,
                processing_ms: 30,
                language: None,
            })
        }
    }

    fn fake_fn() -> TranscribeFn {
        Arc::new(|pcm: &[f32]| Fake.transcribe(pcm, &SttOptions::default()))
    }

    #[test]
    fn segment_hook_skips_live_previews() {
        // Hands-free counts phrases off this hook; a preview counted as a phrase repeats text.
        let calls = Arc::new(AtomicUsize::new(0));
        let c2 = calls.clone();
        let f: TranscribeFn = Arc::new(move |pcm: &[f32]| {
            c2.fetch_add(1, Ordering::SeqCst);
            Fake.transcribe(pcm, &SttOptions::default())
        });
        let segments = Arc::new(Mutex::new(Vec::new()));
        let s2 = segments.clone();
        let partial: PartialFn = Arc::new(|_: &str, _: usize| {});
        let mut seg = PhraseSegmenter::new(DecodeThread::spawn(None))
            .with_partial_interval(Some(Duration::from_millis(1)))
            .start_with_segments(
                f,
                Some(partial),
                Arc::new(move |_, r| s2.lock().unwrap().push(r.text.clone())),
            );
        let mut audio = speech(3.0);
        audio.extend(silence(0.25));
        audio.extend(speech(2.5));
        for b in audio.chunks(1600) {
            seg.feed(b);
            std::thread::sleep(Duration::from_millis(2));
        }
        seg.release();
        let out = seg.finish().unwrap();
        // Exactly the cut phrases, each once.
        let phrases: Vec<String> = out.phrases.iter().map(|r| r.text.clone()).collect();
        assert_eq!(*segments.lock().unwrap(), phrases);
        assert!(out.text.starts_with("p3 p2"), "{}", out.text);
        assert!(calls.load(Ordering::SeqCst) > 2, "previews ran too");
    }

    #[test]
    fn segmenter_decodes_phrases_during_capture_and_joins_in_order() {
        let decoder = DecodeThread::spawn(Some(Box::new(Fake)));
        let segments = Arc::new(Mutex::new(Vec::new()));
        let s2 = segments.clone();
        let partials = Arc::new(Mutex::new(Vec::<(String, usize)>::new()));
        let p2 = partials.clone();
        let mut seg = Segmenter::new(
            decoder.clone(),
            Transcriber::Engine(SttOptions::default()),
            SegmenterOptions {
                on_segment: Some(Arc::new(move |i, r| {
                    s2.lock().unwrap().push((i, r.text.clone()))
                })),
                on_partial: Some(Arc::new(move |t, s| {
                    p2.lock().unwrap().push((t.to_string(), s))
                })),
                partial_interval: Some(Duration::from_millis(1)),
                ..Default::default()
            },
        );
        let mut audio = speech(3.0);
        audio.extend(silence(0.25));
        audio.extend(speech(2.5));
        audio.extend(silence(0.25));
        audio.extend(speech(1.0));
        for b in audio.chunks(1600) {
            seg.feed(b);
        }
        // The two completed phrases are decoded before finish() is called.
        let deadline = Instant::now() + Duration::from_secs(2);
        while segments.lock().unwrap().len() < 2 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        let n = segments.lock().unwrap().len();
        assert_eq!(n, 2);
        seg.release();
        let out = seg.finish().unwrap();
        assert_eq!(out.phrases.len(), 3);
        assert_eq!(out.text, "p3 p2 p1");
        assert!(
            out.stt_tail_ms < 200,
            "only the tail is on the release path: {}",
            out.stt_tail_ms
        );
        assert!(
            (900..=1100).contains(&out.tail_audio_ms),
            "{}",
            out.tail_audio_ms
        );
        assert!(!partials.lock().unwrap().is_empty());
        assert!(decoder.is_idle());
    }

    #[test]
    fn silent_post_roll_is_not_decoded() {
        let decoder = DecodeThread::spawn(None);
        let calls = Arc::new(AtomicUsize::new(0));
        let c2 = calls.clone();
        let f: TranscribeFn = Arc::new(move |pcm: &[f32]| {
            c2.fetch_add(1, Ordering::SeqCst);
            Fake.transcribe(pcm, &SttOptions::default())
        });
        let mut seg = PhraseSegmenter::new(decoder)
            .with_partial_interval(None)
            .start(f, None);
        seg.feed(&speech(1.5));
        seg.feed(&silence(0.05));
        seg.release();
        seg.feed(&silence(0.2)); // post-roll
        let out = seg.finish().unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(out.text, "p1");
    }

    #[test]
    fn spoken_post_roll_is_decoded_after_the_release_cut() {
        let decoder = DecodeThread::spawn(None);
        let mut seg = PhraseSegmenter::new(decoder)
            .with_partial_interval(None)
            .start(fake_fn(), None);
        seg.feed(&speech(1.5));
        seg.feed(&silence(0.1));
        seg.feed(&speech(0.5));
        seg.release(); // cuts at the pause; the last word stays open
        seg.feed(&speech(1.0)); // post-roll still has voice
        let out = seg.finish().unwrap();
        assert_eq!(out.phrases.len(), 2);
        assert_eq!(out.text, "p1 p1");
    }

    #[test]
    fn cancel_makes_queued_work_a_no_op() {
        let decoder = DecodeThread::spawn(Some(Box::new(Fake)));
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        let mut seg = Segmenter::new(
            decoder.clone(),
            Transcriber::Engine(SttOptions::default()),
            SegmenterOptions {
                on_segment: Some(Arc::new(move |_, _| {
                    h.fetch_add(1, Ordering::SeqCst);
                })),
                ..Default::default()
            },
        );
        for _ in 0..4 {
            seg.feed(&speech(2.5));
            seg.feed(&silence(0.25));
        }
        seg.cancel();
        decoder.exec(|_| ()).recv().unwrap(); // barrier
        assert!(
            hits.load(Ordering::SeqCst) <= 1,
            "at most the in-flight phrase completes"
        );
    }

    #[test]
    fn decode_errors_surface_and_missing_engine_is_an_error() {
        let decoder = DecodeThread::spawn(None);
        let mut seg = Segmenter::new(
            decoder.clone(),
            Transcriber::Engine(SttOptions::default()),
            SegmenterOptions::default(),
        );
        seg.feed(&speech(1.0));
        assert!(seg.finish().is_err());
        decoder
            .exec(|slot| *slot = Some(Box::new(Fake) as Box<dyn SttEngine>))
            .recv()
            .unwrap();
        let mut seg = Segmenter::new(
            decoder,
            Transcriber::Engine(SttOptions::default()),
            SegmenterOptions::default(),
        );
        seg.feed(&speech(1.0));
        assert_eq!(seg.finish().unwrap().text, "p1");
    }

    #[test]
    fn speech_check_skips_rejected_phrases() {
        let decoder = DecodeThread::spawn(None);
        let mut seg = PhraseSegmenter::new(decoder)
            .with_partial_interval(None)
            .with_speech_check(Arc::new(|_: &[f32]| false))
            .start(fake_fn(), None);
        seg.feed(&speech(1.0));
        let out = seg.finish().unwrap();
        assert_eq!(out.text, "");
        assert_eq!(out.phrases.len(), 1);
    }
}
