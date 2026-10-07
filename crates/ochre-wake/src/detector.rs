//! Streaming wake detector: front end + head + gate, VAD-gated so a silent room costs only the VAD.
//! Port of `src/openwhisprflow/wake/detector.py` (`docs/wakeword.md` §5-6).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use ochre_core::{Error, Result};

use crate::frontend::{WakeFrontend, WakeHead};
use crate::{FRAME, FRAME_S};

/// A gate firing. Frame indices count 80 ms frames since the detector was created (first = 1).
#[derive(Debug, Clone, PartialEq)]
pub struct WakeHit {
    /// Frame on which the gate fired (end of the word + ~0-400 ms).
    pub frame: u64,
    /// First frame of the run at or above `0.6 x threshold`: roughly the end of the spoken word.
    pub rise_frame: u64,
    /// Peak score of the run.
    pub score: f32,
    /// Head name (model file stem).
    pub model: String,
}

/// Threshold + N consecutive frames + cooldown on a per-frame score stream (`docs/wakeword.md` §5).
#[derive(Debug, Clone)]
pub struct WakeGate {
    pub threshold: f32,
    pub consecutive: u32,
    pub cooldown_frames: i64,
    run: u32,
    last_fire: Option<i64>,
    /// Peak score of the current run.
    pub peak: f32,
}

impl WakeGate {
    pub fn new(threshold: f32, consecutive: u32, cooldown_frames: i64) -> Self {
        WakeGate {
            threshold,
            consecutive: consecutive.max(1),
            cooldown_frames,
            run: 0,
            last_fire: None,
            peak: 0.0,
        }
    }

    pub fn reset_run(&mut self) {
        self.run = 0;
        self.peak = 0.0;
    }

    /// Start a cooldown now (after a session ends, so its last words can't re-wake).
    pub fn hold(&mut self, frame: i64) {
        self.last_fire = Some(frame);
        self.reset_run();
    }

    pub fn update(&mut self, score: f32, frame: i64) -> bool {
        if score < self.threshold {
            self.reset_run();
            return false;
        }
        self.run += 1;
        self.peak = self.peak.max(score);
        if self.run < self.consecutive {
            return false;
        }
        if let Some(last) = self.last_fire
            && frame - last < self.cooldown_frames
        {
            return false; // the run is NOT reset here
        }
        self.last_fire = Some(frame);
        self.run = 0;
        true
    }
}

/// What the detector runs per inferred frame: push one frame (int16 scale), return the head score.
/// [`OnnxScorer`] is the real one; tests script scores with fakes.
pub trait FrameScorer: Send {
    fn push_and_score(&mut self, frame: &[f32]) -> Result<f32>;
    fn reset(&mut self);
    fn name(&self) -> &str {
        ""
    }
}

/// The openWakeWord front end plus one ONNX head.
pub struct OnnxScorer {
    pub frontend: WakeFrontend,
    pub head: WakeHead,
}

impl OnnxScorer {
    /// Load the head and the front end from `frontend_dir` (downloaded there on first use).
    pub fn load(model_path: &Path, frontend_dir: &Path) -> Result<Self> {
        if !model_path.exists() {
            return Err(Error::Model(format!(
                "no wake word model at {}",
                model_path.display()
            )));
        }
        let (mel, emb) = crate::models::ensure_frontend_models(frontend_dir, None)?;
        Ok(OnnxScorer {
            frontend: WakeFrontend::new(&mel, &emb)?,
            head: WakeHead::new(model_path)?,
        })
    }
}

impl FrameScorer for OnnxScorer {
    fn push_and_score(&mut self, frame: &[f32]) -> Result<f32> {
        self.frontend.push_frame(frame)?;
        self.head.score(&self.frontend)
    }
    fn reset(&mut self) {
        self.frontend.reset();
    }
    fn name(&self) -> &str {
        &self.head.name
    }
}

/// Speech predicate for one 1280-sample frame of [-1, 1] floats (e.g. Silero from `ochre-audio`).
pub type SpeechFn = Box<dyn FnMut(&[f32]) -> bool + Send>;

/// Tunables (`docs/wakeword.md` §5-6). `Default` is the shipped behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct DetectorOptions {
    pub threshold: f32,
    pub consecutive: u32,
    pub cooldown_s: f32,
    /// Keep inferring this long after the last voiced frame.
    pub hangover_s: f32,
    /// Skipped (unvoiced) frames kept for catch-up when speech starts.
    pub catchup_s: f32,
    /// Rise level; `None` = `0.6 x threshold`.
    pub rise_threshold: Option<f32>,
    /// Where the mel/embedding models live; `None` = [`crate::models::frontend_dir`].
    pub frontend_dir: Option<PathBuf>,
}

impl Default for DetectorOptions {
    fn default() -> Self {
        DetectorOptions {
            threshold: 0.5,
            consecutive: 2,
            cooldown_s: 1.5,
            hangover_s: 1.0,
            catchup_s: 2.0,
            rise_threshold: None,
            frontend_dir: None,
        }
    }
}

/// Counters for logs and the debug UI.
#[derive(Debug, Clone, Default)]
pub struct DetectorStats {
    /// 80 ms frames seen.
    pub frames: u64,
    /// Frames that ran the front end + head.
    pub inferred: u64,
    pub speech_frames: u64,
    pub hits: u64,
    /// Inference errors (each scores 0).
    pub errors: u64,
    /// Recent head scores (last 64).
    pub scores: VecDeque<f32>,
}

fn frames_for(s: f32) -> i64 {
    (s / FRAME_S).round() as i64
}

/// `process(block) -> Option<WakeHit>` for a live 16 kHz mono stream of any block size.
pub struct WakeDetector {
    scorer: Box<dyn FrameScorer>,
    gate: WakeGate,
    rise_threshold: Option<f32>,
    is_speech: Option<SpeechFn>,
    hangover: i64,
    catchup: usize,
    pending: Vec<f32>,
    idx: i64,
    fed: i64,
    last_voice: i64,
    rise: Option<i64>,
    backlog: VecDeque<(i64, Vec<f32>)>,
    pub stats: DetectorStats,
    /// Head score of the most recent inferred frame.
    pub last_score: f32,
    /// VAD decision for the most recent frame (true without a VAD).
    pub last_voiced: bool,
}

impl WakeDetector {
    /// Load `model_path` (an ONNX head) with the default options and the given threshold. The
    /// front-end models are downloaded into [`crate::models::frontend_dir`] on first use.
    pub fn new(model_path: impl AsRef<Path>, threshold: f32) -> Result<Self> {
        Self::with_options(
            model_path,
            DetectorOptions {
                threshold,
                ..Default::default()
            },
        )
    }

    pub fn with_options(model_path: impl AsRef<Path>, opts: DetectorOptions) -> Result<Self> {
        let dir = opts
            .frontend_dir
            .clone()
            .unwrap_or_else(crate::models::frontend_dir);
        let scorer = OnnxScorer::load(model_path.as_ref(), &dir)?;
        Ok(Self::with_scorer(Box::new(scorer), opts))
    }

    /// Any scorer (shared front end, tests).
    pub fn with_scorer(scorer: Box<dyn FrameScorer>, opts: DetectorOptions) -> Self {
        WakeDetector {
            scorer,
            gate: WakeGate::new(
                opts.threshold,
                opts.consecutive,
                frames_for(opts.cooldown_s).max(1),
            ),
            rise_threshold: opts.rise_threshold,
            is_speech: None,
            hangover: frames_for(opts.hangover_s).max(0),
            catchup: frames_for(opts.catchup_s).max(1) as usize,
            pending: Vec::new(),
            idx: 0,
            fed: 0,
            last_voice: i64::MIN / 2,
            rise: None,
            backlog: VecDeque::new(),
            stats: DetectorStats::default(),
            last_score: 0.0,
            last_voiced: true,
        }
    }

    /// Gate inference on a speech predicate (one 1280-sample [-1, 1] frame -> voiced?).
    pub fn with_vad(mut self, is_speech: impl FnMut(&[f32]) -> bool + Send + 'static) -> Self {
        self.is_speech = Some(Box::new(is_speech));
        self
    }

    pub fn set_vad(&mut self, is_speech: Option<SpeechFn>) {
        self.is_speech = is_speech;
    }

    pub fn has_vad(&self) -> bool {
        self.is_speech.is_some()
    }

    pub fn threshold(&self) -> f32 {
        self.gate.threshold
    }

    pub fn set_threshold(&mut self, t: f32) {
        self.gate.threshold = t;
    }

    /// Index of the last frame seen (1-based; 0 before any audio).
    pub fn frame_index(&self) -> u64 {
        self.idx as u64
    }

    pub fn model_name(&self) -> &str {
        self.scorer.name()
    }

    /// Forget audio context (front end, gate run, backlog). Keeps the cooldown and counters.
    pub fn reset(&mut self) {
        self.scorer.reset();
        self.gate.reset_run();
        self.pending.clear();
        self.backlog.clear();
        self.rise = None;
        self.fed = self.idx;
        self.last_score = 0.0;
    }

    /// Cooldown from now.
    pub fn hold(&mut self) {
        self.gate.hold(self.idx);
    }

    /// Feed 16 kHz mono audio as [-1, 1] floats, any block size. Returns the last hit that fired
    /// inside the block (with the 1.5 s cooldown, at most one per realistic block).
    pub fn process(&mut self, block: &[f32]) -> Option<WakeHit> {
        let mut hit = None;
        self.pending.extend(block.iter().map(|v| v * 32767.0));
        let n = self.pending.len() / FRAME;
        for k in 0..n {
            let frame: Vec<f32> = self.pending[k * FRAME..(k + 1) * FRAME].to_vec();
            if let Some(h) = self.frame_scaled(frame) {
                hit = Some(h);
            }
        }
        self.pending.drain(..n * FRAME);
        hit
    }

    /// Same as [`process`](Self::process) for int16 samples (bit-exact with the Python reference).
    pub fn process_i16(&mut self, block: &[i16]) -> Option<WakeHit> {
        let mut hit = None;
        self.pending.extend(block.iter().map(|&v| v as f32));
        let n = self.pending.len() / FRAME;
        for k in 0..n {
            let frame: Vec<f32> = self.pending[k * FRAME..(k + 1) * FRAME].to_vec();
            if let Some(h) = self.frame_scaled(frame) {
                hit = Some(h);
            }
        }
        self.pending.drain(..n * FRAME);
        hit
    }

    /// One whole 1280-sample frame of [-1, 1] floats (what the hands-free controller calls).
    pub fn process_frame(&mut self, frame: &[f32]) -> Option<WakeHit> {
        assert_eq!(
            frame.len(),
            FRAME,
            "process_frame needs exactly {FRAME} samples"
        );
        self.frame_scaled(frame.iter().map(|v| v * 32767.0).collect())
    }

    fn frame_scaled(&mut self, frame: Vec<f32>) -> Option<WakeHit> {
        self.idx += 1;
        let idx = self.idx;
        self.stats.frames += 1;
        let voiced = match self.is_speech.as_mut() {
            None => true,
            Some(f) => {
                let unit: Vec<f32> = frame.iter().map(|v| v / 32767.0).collect();
                f(&unit)
            }
        };
        self.last_voiced = voiced;
        if voiced {
            self.stats.speech_frames += 1;
            self.last_voice = idx;
        }
        if idx - self.last_voice > self.hangover {
            // Silence: no inference. Remember the frame in case speech starts right after it.
            if self.backlog.len() == self.catchup {
                self.backlog.pop_front();
            }
            self.backlog.push_back((idx, frame));
            return None;
        }
        let fed = self.fed;
        let mut todo: Vec<(i64, Vec<f32>)> =
            self.backlog.drain(..).filter(|(i, _)| *i > fed).collect();
        todo.push((idx, frame));
        let mut hit = None;
        for (i, f) in todo {
            let h = self.infer(i, &f);
            if hit.is_none() {
                hit = h;
            }
        }
        hit
    }

    fn infer(&mut self, i: i64, frame: &[f32]) -> Option<WakeHit> {
        if i != self.fed + 1 {
            // a gap since the last inference: the score run is not continuous
            self.gate.reset_run();
            self.rise = None;
        }
        self.fed = i;
        let s = match self.scorer.push_and_score(frame) {
            Ok(s) => s,
            Err(e) => {
                if self.stats.errors == 0 {
                    tracing::warn!("wake inference failed: {e}");
                }
                self.stats.errors += 1;
                0.0
            }
        };
        self.last_score = s;
        self.stats.inferred += 1;
        if self.stats.scores.len() == 64 {
            self.stats.scores.pop_front();
        }
        self.stats.scores.push_back(s);
        let rise_t = self.rise_threshold.unwrap_or(self.gate.threshold * 0.6);
        if s >= rise_t {
            self.rise.get_or_insert(i);
        } else {
            self.rise = None;
        }
        if self.gate.update(s, i) {
            self.stats.hits += 1;
            return Some(WakeHit {
                frame: i as u64,
                rise_frame: self.rise.unwrap_or(i) as u64,
                score: self.gate.peak.max(s),
                model: self.scorer.name().to_string(),
            });
        }
        None
    }
}

/// `<model>.json` next to a head (threshold table, stats), if present and valid.
pub fn load_meta(model_path: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(model_path.with_extension("json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// `recommended.threshold` from `<model>.json`.
pub fn recommended_threshold(model_path: &Path) -> Option<f32> {
    load_meta(model_path)?
        .get("recommended")?
        .get("threshold")?
        .as_f64()
        .map(|v| v as f32)
}

/// `handsfree.model` (a path) wins; else `<bundled_dir>/<phrase>.onnx` ("hey computer" ->
/// `hey_computer.onnx`). The orchestrator decides `bundled_dir` (e.g. the app's `assets/wake`).
pub fn resolve_model(phrase: &str, model: &str, bundled_dir: &Path) -> Result<PathBuf> {
    let p = if model.is_empty() {
        bundled_dir.join(format!(
            "{}.onnx",
            phrase.trim().to_lowercase().replace(' ', "_")
        ))
    } else {
        PathBuf::from(model)
    };
    if p.exists() {
        Ok(p)
    } else {
        Err(Error::Model(format!(
            "no wake word model for {phrase:?}: {} (train one: ochre-cli train-wake)",
            p.display()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn gate_needs_consecutive_frames_and_cools_down() {
        let mut g = WakeGate::new(0.5, 2, 5);
        assert!(!g.update(0.9, 1));
        assert!(!g.update(0.1, 2));
        assert!(!g.update(0.9, 3));
        assert!(g.update(0.8, 4));
        assert!(!g.update(0.9, 5) && !g.update(0.9, 6));
        assert!(!g.update(0.9, 8));
        assert!(g.update(0.9, 9));
        g.hold(20);
        assert!(!g.update(0.9, 21) && !g.update(0.9, 22));
        assert!(g.update(0.9, 25));
    }

    #[test]
    fn gate_hold_resets_run() {
        let mut g = WakeGate::new(0.5, 2, 3);
        g.update(0.9, 1);
        g.hold(1);
        assert!(!g.update(0.9, 2));
        assert!(!g.update(0.9, 3));
        assert!(g.update(0.9, 4));
    }

    /// Score = script[pushed] (default 0), like the Python FakeFrontend + ScriptedHead.
    struct Scripted {
        pushed: Arc<Mutex<u32>>,
        script: Arc<Mutex<Vec<(u32, f32)>>>,
    }

    impl FrameScorer for Scripted {
        fn push_and_score(&mut self, _frame: &[f32]) -> Result<f32> {
            let mut p = self.pushed.lock().unwrap();
            *p += 1;
            Ok(self
                .script
                .lock()
                .unwrap()
                .iter()
                .find(|(i, _)| *i == *p)
                .map(|x| x.1)
                .unwrap_or(0.0))
        }
        fn reset(&mut self) {}
    }

    type Handles = (Arc<Mutex<u32>>, Arc<Mutex<Vec<(u32, f32)>>>);

    fn det(script: &[(u32, f32)], vad: bool, opts: DetectorOptions) -> (WakeDetector, Handles) {
        let pushed = Arc::new(Mutex::new(0));
        let sc = Arc::new(Mutex::new(script.to_vec()));
        let d = WakeDetector::with_scorer(
            Box::new(Scripted {
                pushed: pushed.clone(),
                script: sc.clone(),
            }),
            opts,
        );
        let d = if vad {
            d.with_vad(|f: &[f32]| f.iter().map(|v| v.abs()).sum::<f32>() / f.len() as f32 > 0.003)
        } else {
            d
        };
        (d, (pushed, sc))
    }

    const LOUD: [f32; FRAME] = [3000.0 / 32767.0; FRAME];
    const QUIET: [f32; FRAME] = [0.0; FRAME];

    #[test]
    fn fires_and_reports_rise() {
        let (mut d, _) = det(
            &[(5, 0.4), (6, 0.7), (7, 0.9)],
            false,
            DetectorOptions::default(),
        );
        let hits: Vec<_> = (0..10).filter_map(|_| d.process(&LOUD)).collect();
        assert_eq!(hits.len(), 1);
        assert_eq!((hits[0].frame, hits[0].rise_frame), (7, 5));
        assert!((hits[0].score - 0.9).abs() < 1e-6);
        assert_eq!((d.stats.hits, d.stats.inferred), (1, 10));
    }

    #[test]
    fn vad_gate_skips_inference_in_silence() {
        let (mut d, (pushed, _)) = det(
            &[],
            true,
            DetectorOptions {
                hangover_s: 0.4,
                ..Default::default()
            },
        );
        for _ in 0..100 {
            d.process(&QUIET);
        }
        assert_eq!((*pushed.lock().unwrap(), d.stats.inferred), (0, 0));
        d.process(&LOUD);
        assert_eq!(*pushed.lock().unwrap(), 26); // 25 caught-up frames + this one
        for _ in 0..5 {
            d.process(&QUIET);
        }
        assert_eq!(*pushed.lock().unwrap(), 31);
        d.process(&QUIET);
        assert_eq!(*pushed.lock().unwrap(), 31);
        assert!(!d.last_voiced && d.has_vad());
    }

    #[test]
    fn catchup_detects_word_that_started_before_vad_onset() {
        let (mut d, (_, script)) = det(&[], true, DetectorOptions::default());
        for _ in 0..30 {
            d.process(&QUIET);
        }
        *script.lock().unwrap() = vec![(24, 0.8), (25, 0.9)];
        let h = d.process(&LOUD).expect("hit");
        assert_eq!(h.frame, 30);
    }

    #[test]
    fn gap_breaks_the_run() {
        let opts = DetectorOptions {
            hangover_s: 0.0,
            catchup_s: 0.08,
            ..Default::default()
        };
        let (mut d, (pushed, _)) = det(&[(2, 0.9), (3, 0.9)], true, opts);
        d.process(&LOUD);
        d.process(&LOUD);
        d.process(&QUIET);
        d.process(&QUIET);
        assert!(d.process(&LOUD).is_none());
        assert_eq!(*pushed.lock().unwrap(), 4);
    }

    #[test]
    fn hold_suppresses_hits() {
        let (mut d, _) = det(
            &[(3, 0.9), (4, 0.9)],
            false,
            DetectorOptions {
                cooldown_s: 1.0,
                ..Default::default()
            },
        );
        d.process(&LOUD);
        d.hold();
        assert!((0..5).all(|_| d.process(&LOUD).is_none()));
    }

    #[test]
    fn odd_block_sizes_frame_correctly() {
        let (mut d, (pushed, _)) = det(&[], false, DetectorOptions::default());
        let audio = vec![0.1f32; FRAME * 5 + 100];
        for c in audio.chunks(777) {
            d.process(c);
        }
        assert_eq!(*pushed.lock().unwrap(), 5);
        assert_eq!(d.frame_index(), 5);
    }

    #[test]
    fn resolve_model_paths() {
        let dir = std::env::temp_dir();
        assert!(resolve_model("no such phrase", "", &dir).is_err());
        assert!(load_meta(Path::new("nope.onnx")).is_none());
    }
}
