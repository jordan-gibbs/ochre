//! Streaming openWakeWord front end (mel spectrogram + speech embedding) and the ONNX head, on
//! `ort`. Exact port of `src/openwhisprflow/wake/frontend.py` (`docs/wakeword.md` §2-4).
//!
//! Audio here is **int16-scale** f32 (a full-scale sine reaches about ±32767). The public detector
//! API takes [-1, 1] floats and scales them; see [`crate::detector::WakeDetector::process`].

use std::path::Path;

use ochre_core::{Error, Result};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{TensorRef, ValueType};

use crate::{EMBED_DIM, FRAME, MEL_BINS, MEL_CONTEXT, MEL_WINDOW, SAMPLE_RATE};

/// Embeddings kept in the streaming window (heads read 16; headroom for wider heads).
pub const KEEP_EMBEDDINGS: usize = 64;

pub(crate) fn model_err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Model(format!("{what}: {e}"))
}

/// Single-threaded, non-spinning CPU session (`docs/wakeword.md` §1): ORT's default spinning pool
/// would burn a core while idle-listening.
pub fn make_session(path: &Path) -> Result<Session> {
    let what = path.display().to_string();
    let e = |err: ort::Error| model_err(&what, err);
    Session::builder()
        .map_err(e)?
        .with_intra_threads(1)
        .map_err(|x| e(x.into()))?
        .with_inter_threads(1)
        .map_err(|x| e(x.into()))?
        .with_parallel_execution(false)
        .map_err(|x| e(x.into()))?
        .with_intra_op_spinning(false)
        .map_err(|x| e(x.into()))?
        .with_inter_op_spinning(false)
        .map_err(|x| e(x.into()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|x| e(x.into()))?
        .commit_from_file(path)
        .map_err(e)
}

/// The mel + embedding models with the streaming state.
pub struct WakeFrontend {
    mel: Session,
    emb: Session,
    raw: Vec<f32>,      // last FRAME + MEL_CONTEXT samples
    pending: Vec<f32>,  // samples not yet forming a frame
    mel_buf: Vec<f32>,  // MEL_WINDOW x MEL_BINS, row-major
    features: Vec<f32>, // n x EMBED_DIM, row-major, n <= KEEP_EMBEDDINGS
    warm: Vec<f32>,     // features after reset (computed once; deterministic)
    /// Embeddings produced since construction (not reset), for profiling.
    pub frames: u64,
}

impl WakeFrontend {
    pub fn new(mel_path: &Path, embed_path: &Path) -> Result<Self> {
        let mut fe = WakeFrontend {
            mel: make_session(mel_path)?,
            emb: make_session(embed_path)?,
            raw: vec![0.0; FRAME + MEL_CONTEXT],
            pending: Vec::new(),
            mel_buf: vec![1.0; MEL_WINDOW * MEL_BINS],
            features: Vec::new(),
            warm: Vec::new(),
            frames: 0,
        };
        // Warm-up: embeddings of 4 s of weak noise, numpy default_rng(0).integers(-1000, 1000).
        let noise: Vec<f32> = crate::rng::numpy_integers(0, -1000, 1000, SAMPLE_RATE * 4)
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let mut warm = fe.embed_clip(&noise)?;
        let n = warm.len() / EMBED_DIM;
        if n > KEEP_EMBEDDINGS {
            warm.drain(..(n - KEEP_EMBEDDINGS) * EMBED_DIM);
        }
        fe.warm = warm;
        fe.reset();
        Ok(fe)
    }

    /// Back to the start state (zero audio context, ones mel buffer, warm-up embeddings).
    pub fn reset(&mut self) {
        self.raw.iter_mut().for_each(|x| *x = 0.0);
        self.pending.clear();
        self.mel_buf.iter_mut().for_each(|x| *x = 1.0);
        self.features.clone_from(&self.warm);
    }

    /// `melspectrogram(x) / 10 + 2`, flattened rows of `MEL_BINS`.
    pub fn melspec(&mut self, x: &[f32]) -> Result<Vec<f32>> {
        let t = TensorRef::from_array_view(([1usize, x.len()], x))
            .map_err(|e| model_err("mel input", e))?;
        let out = self
            .mel
            .run(ort::inputs![t])
            .map_err(|e| model_err("mel", e))?;
        let (_, data) = out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| model_err("mel output", e))?;
        Ok(data.iter().map(|v| v / 10.0 + 2.0).collect())
    }

    /// Embeddings for `n` stacked `MEL_WINDOW x MEL_BINS` windows; flattened `n x EMBED_DIM`.
    fn embed(&mut self, windows: &[f32], n: usize) -> Result<Vec<f32>> {
        let t = TensorRef::from_array_view(([n, MEL_WINDOW, MEL_BINS, 1], windows))
            .map_err(|e| model_err("embedding input", e))?;
        let out = self
            .emb
            .run(ort::inputs![t])
            .map_err(|e| model_err("embedding", e))?;
        let (_, data) = out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| model_err("embedding output", e))?;
        Ok(data.to_vec())
    }

    /// Batch path (tests, warm-up): whole clip at int16 scale -> one embedding per 80 ms.
    pub fn embed_clip(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        let spec = self.melspec(audio)?;
        let rows = spec.len() / MEL_BINS;
        if rows < MEL_WINDOW {
            return Ok(Vec::new());
        }
        let starts: Vec<usize> = (0..=rows - MEL_WINDOW).step_by(8).collect();
        let mut wins = Vec::with_capacity(starts.len() * MEL_WINDOW * MEL_BINS);
        for &s in &starts {
            wins.extend_from_slice(&spec[s * MEL_BINS..(s + MEL_WINDOW) * MEL_BINS]);
        }
        self.embed(&wins, starts.len())
    }

    /// Feed int16-scale audio of any length; returns how many new embeddings were produced.
    pub fn push(&mut self, audio: &[f32]) -> Result<usize> {
        self.pending.extend_from_slice(audio);
        let n = self.pending.len() / FRAME;
        for k in 0..n {
            let chunk: Vec<f32> = self.pending[k * FRAME..(k + 1) * FRAME].to_vec();
            self.push_frame(&chunk)?;
        }
        self.pending.drain(..n * FRAME);
        Ok(n)
    }

    /// One exact 1280-sample frame (int16 scale).
    pub fn push_frame(&mut self, chunk: &[f32]) -> Result<()> {
        debug_assert_eq!(chunk.len(), FRAME);
        self.raw.copy_within(FRAME.., 0);
        self.raw[MEL_CONTEXT..].copy_from_slice(chunk);
        let raw = std::mem::take(&mut self.raw);
        let mel = self.melspec(&raw);
        self.raw = raw;
        let mel = mel?;
        // mel_buf = (mel_buf ++ mel)[last MEL_WINDOW rows]
        let total = self.mel_buf.len() + mel.len();
        let keep = MEL_WINDOW * MEL_BINS;
        let mut joined = Vec::with_capacity(total);
        joined.extend_from_slice(&self.mel_buf);
        joined.extend_from_slice(&mel);
        self.mel_buf.copy_from_slice(&joined[total - keep..]);
        let buf = std::mem::take(&mut self.mel_buf);
        let emb = self.embed(&buf, 1);
        self.mel_buf = buf;
        self.features.extend_from_slice(&emb?);
        let n = self.features.len() / EMBED_DIM;
        if n > KEEP_EMBEDDINGS {
            self.features.drain(..(n - KEEP_EMBEDDINGS) * EMBED_DIM);
        }
        self.frames += 1;
        Ok(())
    }

    /// The last `n` embeddings, flattened `n x 96` (fewer if fewer exist).
    pub fn window(&self, n: usize) -> &[f32] {
        let have = self.features.len() / EMBED_DIM;
        &self.features[(have - n.min(have)) * EMBED_DIM..]
    }

    /// The last `MEL_WINDOW` transformed mel rows, flattened (tests).
    pub fn mel_buffer(&self) -> &[f32] {
        &self.mel_buf
    }
}

/// One ONNX classifier head over `[1, n_frames, 96]` embeddings.
pub struct WakeHead {
    session: Session,
    input_name: String,
    /// Embedding frames the head reads (dim 1 of its input; 16 if dynamic).
    pub n_frames: usize,
    /// File stem, e.g. "transcribe".
    pub name: String,
}

impl WakeHead {
    pub fn new(path: &Path) -> Result<Self> {
        let session = make_session(path)?;
        let inp = session
            .inputs()
            .first()
            .ok_or_else(|| model_err(&path.display().to_string(), "no inputs"))?;
        let input_name = inp.name().to_string();
        let shape: Vec<i64> = match inp.dtype() {
            ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
            other => {
                return Err(model_err(
                    &path.display().to_string(),
                    format!("input is not a tensor: {other:?}"),
                ));
            }
        };
        if shape.len() != 3 || (shape[2] > 0 && shape[2] as usize != EMBED_DIM) {
            return Err(model_err(
                &path.display().to_string(),
                format!("expected (batch, frames, 96) embeddings input, got {shape:?}"),
            ));
        }
        let n_frames = if shape[1] > 0 { shape[1] as usize } else { 16 };
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        Ok(WakeHead {
            session,
            input_name,
            n_frames,
            name,
        })
    }

    /// Score of the newest window; a value outside [0, 1] is a logit and gets squashed.
    pub fn score(&mut self, frontend: &WakeFrontend) -> Result<f32> {
        let w = frontend.window(self.n_frames);
        let n = w.len() / EMBED_DIM;
        let t = TensorRef::from_array_view(([1usize, n, EMBED_DIM], w))
            .map_err(|e| model_err("head input", e))?;
        let out = self
            .session
            .run(ort::inputs![self.input_name.as_str() => t])
            .map_err(|e| model_err("head", e))?;
        let (_, data) = out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| model_err("head output", e))?;
        let v = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        Ok(if (0.0..=1.0).contains(&v) {
            v
        } else {
            1.0 / (1.0 + (-v).exp())
        })
    }
}
