//! NVIDIA Parakeet TDT 0.6B (v3 / v2 / "ultra" post-train), int8 ONNX, run through ONNX Runtime
//! with the mel front end and the TDT greedy decode implemented here (SPEC Â§4.2).
//!
//! Pipeline: `mel::MelFrontend` (128 NeMo log-mels) â†’ `encoder-model.int8.onnx`
//! (`[1,128,T]` â†’ `[1,1024,T/8]`) â†’ greedy token-and-duration (TDT) decode over
//! `decoder_joint-model.int8.onnx`, one call per step. The duration head lets the decoder skip
//! encoder frames, so a 10 s clip takes roughly one decoder call per emitted token plus a few
//! blanks, rather than one per frame. (transcribe-rs, Handy's engine, ignores the duration head and
//! steps every frame; we follow onnx-asr, which is the reference for the decode loop.)
//!
//! Files come from istupakov's onnx-asr exports, pinned to a revision and SHA-256.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use ochre_core::events::EngineInfo;
use ochre_core::stt::{Progress, ProgressFn, SttEngine, SttOptions, SttResult};
use ochre_core::{Error, Result, SAMPLE_RATE};
use ochre_models::{ModelFile, ModelManifest};
use ort::session::Session;
use ort::value::TensorRef;

use crate::chunk::split_at_pauses;
use crate::mel::MelFrontend;
use crate::onnx::{self, Device, Workload, model_err};

pub const ENGINE_ID: &str = "parakeet";
/// docs/benchmarks.md: Ultra int8 keeps fp32 accuracy; istupakov's v3 int8 loses ~1.1 WER.
pub const DEFAULT_MODEL: &str = "parakeet-ultra";
/// The encoder's cost grows with length; longer input is split at pauses first.
pub const MAX_SEGMENT_S: f32 = 30.0;

const ENCODER: &str = "encoder-model.int8.onnx";
const DECODER: &str = "decoder_joint-model.int8.onnx";
const ENCODER_FP32: &str = "encoder-model.onnx";
const VOCAB: &str = "vocab.txt";
const CONFIG: &str = "config.json";

pub struct Variant {
    pub name: &'static str,
    pub label: &'static str,
    pub languages: &'static str,
    /// Weights precision; int8 is the CPU default, fp32 is for GPUs (int8 graphs run poorly on
    /// the CUDA execution provider).
    pub precision: &'static str,
    repo: &'static str,
    rev: &'static str,
    encoder: &'static str,
    decoder: &'static str,
    /// (local name, path in repo, size, sha256)
    files: &'static [(&'static str, &'static str, u64, &'static str)],
}

const CONFIG_FILE: (&str, &str, u64, &str) = (
    CONFIG,
    CONFIG,
    97,
    "666903c76b9798caf2c210afd4f6cd60b08a8dbf9800ec8d7a3bc0d2148ac466",
);
const VOCAB_V3: (&str, &str, u64, &str) = (
    VOCAB,
    VOCAB,
    93939,
    "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
);

pub static VARIANTS: [Variant; 4] = [
    Variant {
        name: "parakeet-tdt-0.6b-v3",
        label: "Parakeet TDT 0.6B v3",
        languages: "en + 24 European languages (auto)",
        precision: "int8",
        repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
        rev: "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
        encoder: ENCODER,
        decoder: DECODER,
        files: &[
            CONFIG_FILE,
            VOCAB_V3,
            (
                DECODER,
                DECODER,
                18202004,
                "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
            ),
            (
                ENCODER,
                ENCODER,
                652183999,
                "6139d2fa7e1b086097b277c7149725edbab89cc7c7ae64b23c741be4055aff09",
            ),
        ],
    },
    Variant {
        name: "parakeet-ultra",
        label: "Parakeet Ultra (v3 post-train)",
        languages: "en + 24 European languages (auto)",
        precision: "int8",
        repo: "Olicorne/parakeet-tdt-0.6b-v3-ultra-onnx",
        rev: "3fd3b4d9772b2e595a9162f91f929f16bb4ab4cd",
        encoder: ENCODER,
        decoder: DECODER,
        files: &[
            CONFIG_FILE,
            VOCAB_V3,
            (
                DECODER,
                "int8/decoder_joint-model.int8.onnx",
                18203490,
                "f7e2db395a3b738cb2893cfb853d25863cebcc5282583a2e86b5762559e0bd32",
            ),
            (
                ENCODER,
                "int8/encoder-model.int8.onnx",
                649537325,
                "8a2b47169cf3f1b114010e12c0221bc6f07203289477a65a44847b9e3f1e01ed",
            ),
        ],
    },
    Variant {
        name: "parakeet-tdt-0.6b-v2",
        label: "Parakeet TDT 0.6B v2 (English)",
        languages: "en",
        precision: "int8",
        repo: "istupakov/parakeet-tdt-0.6b-v2-onnx",
        rev: "0bbb45a3365852604aef28b538a8f066f4ccaa85",
        encoder: ENCODER,
        decoder: DECODER,
        files: &[
            CONFIG_FILE,
            (
                VOCAB,
                VOCAB,
                9384,
                "ec182b70dd42113aff6c5372c75cac58c952443eb22322f57bbd7f53977d497d",
            ),
            (
                DECODER,
                DECODER,
                8998286,
                "a449f49acd68979d418651dd2dcb737cc0f1bf0225e009e29ee326354edbf7d3",
            ),
            (
                ENCODER,
                ENCODER,
                652184014,
                "3e0581fda6ab843888b51e56d7ee78b6d5bc3237ec113af1f732d1d5286aa155",
            ),
        ],
    },
    Variant {
        name: "parakeet-tdt-0.6b-v3-fp32",
        label: "Parakeet TDT 0.6B v3 (fp32 encoder, for GPUs)",
        languages: "en + 24 European languages (auto)",
        precision: "fp32",
        repo: "istupakov/parakeet-tdt-0.6b-v3-onnx",
        rev: "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce",
        encoder: ENCODER_FP32,
        // The decoder runs on the CPU either way, and the int8 one steps ~4x faster than fp32.
        decoder: DECODER,
        files: &[
            CONFIG_FILE,
            VOCAB_V3,
            (
                DECODER,
                DECODER,
                18202004,
                "eea7483ee3d1a30375daedc8ed83e3960c91b098812127a0d99d1c8977667a70",
            ),
            (
                ENCODER_FP32,
                ENCODER_FP32,
                41770866,
                "98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1",
            ),
            (
                "encoder-model.onnx.data",
                "encoder-model.onnx.data",
                2435420160,
                "9a22d372c51455c34f13405da2520baefb7125bd16981397561423ed32d24f36",
            ),
        ],
    },
];

impl Variant {
    pub fn get(name: &str) -> Result<&'static Variant> {
        let name = if name.is_empty() { DEFAULT_MODEL } else { name };
        VARIANTS.iter().find(|v| v.name == name).ok_or_else(|| {
            let all: Vec<_> = VARIANTS.iter().map(|v| v.name).collect();
            Error::Config(format!(
                "unknown Parakeet model {name:?}; choose one of {}",
                all.join(", ")
            ))
        })
    }

    /// Download manifest (pinned revision + SHA-256 for every file).
    pub fn manifest(&self) -> ModelManifest {
        ModelManifest {
            // Directory names match the Python prototype's (`<model>-int8`).
            id: if self.precision == "int8" {
                format!("{}-int8", self.name)
            } else {
                self.name.to_string()
            },
            files: self
                .files
                .iter()
                .map(|&(local, remote, size, sha)| {
                    let mut f = ModelFile::hf(self.repo, self.rev, remote, Some(sha), Some(size));
                    f.name = local.to_string();
                    f
                })
                .collect(),
        }
    }

    pub fn download_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.2).sum()
    }
}

pub fn info() -> EngineInfo {
    let v = Variant::get(DEFAULT_MODEL).expect("default exists");
    EngineInfo {
        id: ENGINE_ID.into(),
        label: "Parakeet TDT 0.6B (local)".into(),
        kind: "local".into(),
        models: VARIANTS.iter().map(|v| v.name.to_string()).collect(),
        default_model: DEFAULT_MODEL.into(),
        needs_key: false,
        note: format!(
            "{} MB download, runs on CPU",
            v.download_bytes() / 1_000_000
        ),
        languages: "en + 24 European languages (v2: English only)".into(),
    }
}

/// The local Parakeet engine. `load()` downloads (if needed), opens the sessions and runs one
/// warm-up inference; `transcribe()` is then ready at full speed.
pub struct Parakeet {
    variant: &'static Variant,
    device: Device,
    models_root: Option<PathBuf>,
    cancel: AtomicBool,
    model: Option<Mutex<Model>>,
    used_device: Option<Device>,
    /// Load timings, for benches and logs.
    pub load_ms: u64,
    pub warmup_ms: u64,
}

impl Parakeet {
    /// `model`: "" for the default; `device`: an `SttConfig::device` string.
    pub fn new(model: &str, device: &str) -> Result<Self> {
        Ok(Self {
            variant: Variant::get(model)?,
            device: Device::parse(device),
            models_root: None,
            cancel: AtomicBool::new(false),
            model: None,
            used_device: None,
            load_ms: 0,
            warmup_ms: 0,
        })
    }

    /// Use `root` instead of `ochre_core::paths::models_dir()` (tests, benches).
    pub fn with_models_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.models_root = Some(root.into());
        self
    }

    pub fn variant(&self) -> &'static Variant {
        self.variant
    }

    /// The device the encoder actually runs on (after any fallback). None before `load()`.
    pub fn device(&self) -> Option<Device> {
        self.used_device
    }

    /// Abort an in-progress download (the partial file is kept for resume).
    pub fn cancel_download(&self) {
        self.cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn root(&self) -> PathBuf {
        self.models_root
            .clone()
            .unwrap_or_else(ochre_core::paths::models_dir)
    }

    pub fn is_downloaded(&self) -> bool {
        self.variant.manifest().is_present_in(&self.root())
    }

    /// Open already-downloaded model files from `dir` (no download, no hash check).
    pub fn load_from_dir(&mut self, dir: &Path) -> Result<()> {
        // Warm-up: one full pass (front end, encoder, decoder) over 2 s of faint noise, so ORT's
        // allocations, kernel selection and (on GPU) algorithm search happen now, not on the first
        // dictation. A GPU that opens fine but cannot run the graph (e.g. no kernels for this
        // architecture) fails here, and we fall back to the CPU.
        let noise: Vec<f32> = (0..2 * SAMPLE_RATE as usize)
            .map(|i| (((i as u32).wrapping_mul(2654435761) >> 16) as f32 / 65536.0 - 0.5) * 0.01)
            .collect();
        let t0 = Instant::now();
        let (mut model, mut used) = Model::open(dir, self.variant, self.device)?;
        self.load_ms = t0.elapsed().as_millis() as u64;
        let mut t1 = Instant::now();
        if let Err(e) = model.recognize(&noise) {
            if used == Device::Cpu {
                return Err(e);
            }
            tracing::warn!(device = used.as_str(), error = %e, "warm-up failed on the accelerator; falling back to CPU");
            drop(model);
            let t = Instant::now();
            (model, used) = Model::open(dir, self.variant, Device::Cpu)?;
            self.load_ms += t.elapsed().as_millis() as u64;
            t1 = Instant::now();
            model.recognize(&noise)?;
        }
        self.warmup_ms = t1.elapsed().as_millis() as u64;
        self.used_device = Some(used);
        let model = Mutex::new(model);
        tracing::info!(
            model = self.variant.name,
            device = used.as_str(),
            load_ms = self.load_ms,
            warmup_ms = self.warmup_ms,
            "parakeet ready"
        );
        self.model = Some(model);
        Ok(())
    }
}

impl SttEngine for Parakeet {
    fn info(&self) -> EngineInfo {
        info()
    }

    fn load(&mut self, progress: ProgressFn) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }
        let manifest = self.variant.manifest();
        let report = |item: &str, done: u64, total: u64| {
            progress(Progress {
                item: item.to_string(),
                done,
                total,
            })
        };
        let dir = manifest.ensure_in(&self.root(), &report, &self.cancel)?;
        self.load_from_dir(&dir)
    }

    fn transcribe(&self, pcm: &[f32], _opts: &SttOptions) -> Result<SttResult> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::Model("parakeet: load() was not called".into()))?;
        let t0 = Instant::now();
        let duration_ms = (pcm.len() as u64 * 1000) / SAMPLE_RATE as u64;
        let mut text = String::new();
        if is_speech_like(pcm) {
            let mut m = model.lock().unwrap_or_else(|p| p.into_inner());
            let max = (MAX_SEGMENT_S * SAMPLE_RATE as f32) as usize;
            for chunk in split_at_pauses(pcm, SAMPLE_RATE as usize, max) {
                let part = m.recognize(chunk)?;
                if !part.is_empty() {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&part);
                }
            }
        }
        Ok(SttResult {
            text,
            duration_ms,
            processing_ms: t0.elapsed().as_millis() as u64,
            language: None,
        })
    }
}

/// Seconds per encoder frame (8x subsampling of 10 ms mel frames); the resolution of TDT
/// timestamps.
pub const FRAME_SECONDS: f32 = 0.08;

/// Default (lead, trail) silence padding per segment, in seconds; see `Model::pad`.
pub const DEFAULT_PAD: (f32, f32) = (0.0, 0.0);

/// A recognized word with its time span inside the decoded audio.
#[derive(Debug, Clone, PartialEq)]
pub struct TimedWord {
    /// The word with any attached punctuation (", " / "." etc. join the previous word).
    pub text: String,
    pub start_s: f32,
    pub end_s: f32,
}

/// Join words back into text (the inverse of the grouping in `transcribe_words`).
pub fn words_to_text(words: &[TimedWord]) -> String {
    words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

impl Parakeet {
    /// Like `transcribe`, but with per-word timestamps (TDT frame resolution, 80 ms). Used for
    /// overlapped chunk decoding: decode a window with left context and keep only the words that
    /// start after the context. `pcm` must be â‰¤ 30 s.
    pub fn transcribe_words(&self, pcm: &[f32]) -> Result<Vec<TimedWord>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| Error::Model("parakeet: load() was not called".into()))?;
        if !is_speech_like(pcm) {
            return Ok(Vec::new());
        }
        let mut m = model.lock().unwrap_or_else(|p| p.into_inner());
        let pieces = m.recognize_pieces(pcm)?;
        let mut words: Vec<TimedWord> = Vec::new();
        for (i, &(tok, frame)) in pieces.iter().enumerate() {
            let piece = m.vocab[tok].as_str();
            let start = (frame as f32 * FRAME_SECONDS - m.pad.0).max(0.0);
            let next_frame = pieces.get(i + 1).map_or(frame + 1, |p| p.1.max(frame + 1));
            let end = (next_frame as f32 * FRAME_SECONDS - m.pad.0).max(0.0);
            let body = piece.trim_start();
            let starts_word = piece.starts_with(char::is_whitespace)
                && body
                    .chars()
                    .next()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_');
            match words.last_mut() {
                Some(w) if !starts_word => {
                    w.text.push_str(body);
                    w.end_s = end;
                }
                _ if body.is_empty() => {}
                _ => words.push(TimedWord {
                    text: body.to_string(),
                    start_s: start,
                    end_s: end,
                }),
            }
        }
        Ok(words)
    }
}

/// Cheap guard so pure silence never reaches the decoder: at least 100 ms, a peak
/// above -54 dBFS and some energy.
pub fn is_speech_like(pcm: &[f32]) -> bool {
    if pcm.len() < SAMPLE_RATE as usize / 10 {
        return false;
    }
    let peak = pcm.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let rms = (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt();
    peak > 0.002 && rms > 0.0003
}

/// Per-stage timings of the last `recognize` call (for the bench).
#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimes {
    pub features_us: u64,
    pub encoder_us: u64,
    pub decoder_us: u64,
    pub decoder_steps: u32,
}

struct Model {
    encoder: Session,
    decoder: Session,
    mel: MelFrontend,
    vocab: Vec<String>,
    blank: usize,
    vocab_size: usize,
    /// (layers, hidden) of each LSTM state tensor `[layers, 1, hidden]`.
    state_shape: [usize; 2],
    max_tokens_per_step: usize,
    last: StageTimes,
    /// Silence (seconds) added before/after every segment. Parakeet can emit nothing at all for
    /// audio that starts abruptly mid-speech (onnx-asr does the same); leading silence fixes it.
    pad: (f32, f32),
}

fn read_config(dir: &Path) -> (usize, usize) {
    // config.json is tiny and flat: {"features_size": 128, "subsampling_factor": 8, ...}.
    let text = std::fs::read_to_string(dir.join(CONFIG)).unwrap_or_default();
    let num = |key: &str, default: usize| -> usize {
        text.find(&format!("\"{key}\""))
            .and_then(|i| text[i..].split(':').nth(1))
            .and_then(|s| s.trim().split(|c: char| !c.is_ascii_digit()).next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(default)
    };
    (num("features_size", 128), num("max_tokens_per_step", 10))
}

fn load_vocab(path: &Path) -> Result<(Vec<String>, usize)> {
    let text = std::fs::read_to_string(path).map_err(|e| model_err("vocab.txt", e))?;
    let mut pairs = Vec::new();
    for line in text.lines() {
        let Some((tok, id)) = line.rsplit_once(' ') else {
            continue;
        };
        let id: usize = id.trim().parse().map_err(|e| model_err("vocab.txt", e))?;
        pairs.push((id, tok.replace('\u{2581}', " ")));
    }
    let n = pairs.iter().map(|p| p.0 + 1).max().unwrap_or(0);
    let mut vocab = vec![String::new(); n];
    for (id, tok) in pairs {
        vocab[id] = tok;
    }
    let blank = vocab
        .iter()
        .position(|t| t == "<blk>")
        .ok_or_else(|| Error::Model("vocab.txt has no <blk>".into()))?;
    Ok((vocab, blank))
}

impl Model {
    fn open(dir: &Path, variant: &Variant, device: Device) -> Result<(Self, Device)> {
        let (n_mels, max_tokens_per_step) = read_config(dir);
        let (vocab, blank) = load_vocab(&dir.join(VOCAB))?;
        // Both sessions build in parallel (session creation, not inference, dominates startup).
        // The decoder step is tiny (an LSTM step + joint): on a GPU the launch overhead would
        // dominate, so it always runs single-threaded on the CPU.
        let dec_path = dir.join(variant.decoder);
        let ((encoder, used), (decoder, _)) = std::thread::scope(|s| {
            let dec = s.spawn(|| onnx::session(&dec_path, Device::Cpu, Workload::Tiny));
            let enc = onnx::session(&dir.join(variant.encoder), device, Workload::Heavy);
            let dec = dec
                .join()
                .unwrap_or_else(|_| Err(Error::Model("decoder session panicked".into())));
            Ok::<_, Error>((enc?, dec?))
        })?;
        let state_shape = decoder
            .inputs()
            .iter()
            .find(|i| i.name() == "input_states_1")
            .and_then(|i| {
                i.dtype()
                    .tensor_shape()
                    .map(|s| [s[0].max(1) as usize, s[2].max(1) as usize])
            })
            .unwrap_or([2, 640]);
        let vocab_size = vocab.len();
        Ok((
            Self {
                encoder,
                decoder,
                mel: MelFrontend::new(n_mels, SAMPLE_RATE),
                vocab,
                blank,
                vocab_size,
                state_shape,
                max_tokens_per_step,
                last: StageTimes::default(),
                pad: std::env::var("OCHRE_PAD")
                    .ok()
                    .and_then(|v| {
                        let (a, b) = v.split_once(',')?;
                        Some((a.parse().ok()?, b.parse().ok()?))
                    })
                    .unwrap_or(DEFAULT_PAD),
            },
            used,
        ))
    }

    /// One segment (â‰¤ 30 s) to text.
    fn recognize(&mut self, pcm: &[f32]) -> Result<String> {
        let pieces = self.recognize_pieces(pcm)?;
        Ok(detokenize(
            pieces.iter().map(|&(t, _)| self.vocab[t].as_str()),
        ))
    }

    /// One segment (â‰¤ 30 s) to (token, encoder frame) pairs.
    fn recognize_pieces(&mut self, pcm: &[f32]) -> Result<Vec<(usize, usize)>> {
        let t0 = Instant::now();
        let n_mels = self.mel.n_mels();
        let padded;
        let pcm = if self.pad.0 > 0.0 || self.pad.1 > 0.0 {
            let (lead, trail) = (
                (self.pad.0 * SAMPLE_RATE as f32) as usize,
                (self.pad.1 * SAMPLE_RATE as f32) as usize,
            );
            let mut v = Vec::with_capacity(lead + pcm.len() + trail);
            v.resize(lead, 0.0);
            v.extend_from_slice(pcm);
            v.resize(lead + pcm.len() + trail, 0.0);
            padded = v;
            &padded[..]
        } else {
            pcm
        };
        let feats = self.mel.compute(pcm);
        let frames = MelFrontend::frames_for(pcm.len());
        let valid = MelFrontend::valid_frames_for(pcm.len()) as i64;
        let t1 = Instant::now();

        let (enc, enc_len, dim) = {
            let outputs = self
                .encoder
                .run(ort::inputs![
                    "audio_signal" => TensorRef::from_array_view(([1usize, n_mels, frames], &feats[..])).map_err(|e| model_err("encoder input", e))?,
                    "length" => TensorRef::from_array_view(([1usize], &[valid][..])).map_err(|e| model_err("encoder input", e))?,
                ])
                .map_err(|e| model_err("encoder", e))?;
            let (shape, data) = outputs["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(|e| model_err("encoder output", e))?;
            let (dim, t) = (shape[1] as usize, shape[2] as usize);
            let (_, lens) = outputs["encoded_lengths"]
                .try_extract_tensor::<i64>()
                .map_err(|e| model_err("encoder output", e))?;
            let len = (lens[0].max(0) as usize).min(t);
            // [1, dim, T] â†’ [T][dim] so each decoder step reads one contiguous frame.
            let mut tr = vec![0.0f32; len * dim];
            for c in 0..dim {
                let row = &data[c * t..c * t + len];
                for (f, &v) in row.iter().enumerate() {
                    tr[f * dim + c] = v;
                }
            }
            (tr, len, dim)
        };
        let t2 = Instant::now();
        let (tokens, steps) = self.decode(&enc, enc_len, dim)?;
        let t3 = Instant::now();
        self.last = StageTimes {
            features_us: (t1 - t0).as_micros() as u64,
            encoder_us: (t2 - t1).as_micros() as u64,
            decoder_us: (t3 - t2).as_micros() as u64,
            decoder_steps: steps,
        };
        Ok(tokens)
    }

    /// Greedy TDT decode (onnx-asr `_AsrWithTransducerDecoding._decoding`).
    fn decode(
        &mut self,
        enc: &[f32],
        len: usize,
        dim: usize,
    ) -> Result<(Vec<(usize, usize)>, u32)> {
        let [layers, hidden] = self.state_shape;
        let state_len = layers * hidden;
        let mut s1 = vec![0.0f32; state_len];
        let mut s2 = vec![0.0f32; state_len];
        let mut n1 = vec![0.0f32; state_len];
        let mut n2 = vec![0.0f32; state_len];
        let mut tokens: Vec<(usize, usize)> = Vec::new();
        let (mut t, mut emitted, mut steps) = (0usize, 0usize, 0u32);
        while t < len {
            let prev = tokens.last().map_or(self.blank, |&(tok, _)| tok) as i32;
            let frame = &enc[t * dim..(t + 1) * dim];
            let (token, skip) = {
                let outputs = self
                    .decoder
                    .run(ort::inputs![
                        "encoder_outputs" => TensorRef::from_array_view(([1usize, dim, 1], frame)).map_err(|e| model_err("decoder input", e))?,
                        "targets" => TensorRef::from_array_view(([1usize, 1], &[prev][..])).map_err(|e| model_err("decoder input", e))?,
                        "target_length" => TensorRef::from_array_view(([1usize], &[1i32][..])).map_err(|e| model_err("decoder input", e))?,
                        "input_states_1" => TensorRef::from_array_view(([layers, 1, hidden], &s1[..])).map_err(|e| model_err("decoder input", e))?,
                        "input_states_2" => TensorRef::from_array_view(([layers, 1, hidden], &s2[..])).map_err(|e| model_err("decoder input", e))?,
                    ])
                    .map_err(|e| model_err("decoder", e))?;
                let (_, logits) = outputs["outputs"]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| model_err("decoder output", e))?;
                let token = argmax(&logits[..self.vocab_size]);
                let skip = if logits.len() > self.vocab_size {
                    argmax(&logits[self.vocab_size..])
                } else {
                    0
                };
                if token != self.blank {
                    let (_, o1) = outputs["output_states_1"]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| model_err("decoder output", e))?;
                    let (_, o2) = outputs["output_states_2"]
                        .try_extract_tensor::<f32>()
                        .map_err(|e| model_err("decoder output", e))?;
                    n1.copy_from_slice(&o1[..state_len]);
                    n2.copy_from_slice(&o2[..state_len]);
                }
                (token, skip)
            };
            steps += 1;
            if token != self.blank {
                std::mem::swap(&mut s1, &mut n1);
                std::mem::swap(&mut s2, &mut n2);
                tokens.push((token, t));
                emitted += 1;
            }
            if skip > 0 {
                t += skip;
                emitted = 0;
            } else if token == self.blank || emitted == self.max_tokens_per_step {
                t += 1;
                emitted = 0;
            }
        }
        Ok((tokens, steps))
    }
}

fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

/// Join SentencePiece pieces (`â–` already mapped to a space) the way onnx-asr does with
/// `re.sub(r"\A\s|\s\B|(\s)\b", ...)`: drop a leading space and any space not followed by a word
/// character (so no space before punctuation); keep spaces that start a word.
pub fn detokenize<'a>(pieces: impl Iterator<Item = &'a str>) -> String {
    let joined: String = pieces.collect();
    let chars: Vec<char> = joined.chars().collect();
    let mut out = String::with_capacity(joined.len());
    for (i, &c) in chars.iter().enumerate() {
        if c.is_whitespace() {
            let next_is_word = chars
                .get(i + 1)
                .is_some_and(|&n| n.is_alphanumeric() || n == '_');
            if i > 0 && next_is_word {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Stage timings of the most recent segment (bench support).
pub fn last_stage_times(engine: &Parakeet) -> Option<StageTimes> {
    engine
        .model
        .as_ref()
        .map(|m| m.lock().unwrap_or_else(|p| p.into_inner()).last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detokenize_matches_onnx_asr_spacing() {
        let p = [" Hello", ",", " wor", "ld", " .", " it", "'s", " 3", " %"];
        assert_eq!(detokenize(p.into_iter()), "Hello, world. it's 3%");
        assert_eq!(detokenize(std::iter::empty()), "");
    }

    #[test]
    fn manifests_are_pinned() {
        for v in &VARIANTS {
            let m = v.manifest();
            assert!(m.files.len() >= 4);
            assert!(
                m.files
                    .iter()
                    .all(|f| f.sha256.as_ref().is_some_and(|s| s.len() == 64) && f.size.is_some())
            );
            assert!(m.files.iter().all(|f| f.url.contains(v.rev)));
        }
        assert_eq!(Variant::get("").unwrap().name, DEFAULT_MODEL);
        assert!(Variant::get("nope").is_err());
    }

    #[test]
    fn silence_is_skipped() {
        assert!(!is_speech_like(&vec![0.0; 16_000]));
        assert!(!is_speech_like(&[0.5; 100]));
        let tone: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.1).sin() * 0.1).collect();
        assert!(is_speech_like(&tone));
    }

    /// Full model on real speech. Needs the downloaded model (OCHRE_MODELS_DIR or the default
    /// models dir) and OCHRE_REF_DIR with wavs/ls00.wav.
    #[test]
    #[ignore = "needs the 670 MB Parakeet model on disk and a LibriSpeech WAV (OCHRE_REF_DIR)"]
    fn transcribes_librispeech() {
        let dir = std::path::PathBuf::from(std::env::var("OCHRE_REF_DIR").expect("OCHRE_REF_DIR"));
        let mut r = hound::WavReader::open(dir.join("wavs/ls00.wav")).unwrap();
        let pcm: Vec<f32> = r
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let mut e = Parakeet::new("", "cpu").unwrap();
        e.load(&|_| {}).unwrap();
        let out = e.transcribe(&pcm, &SttOptions::default()).unwrap();
        assert_eq!(
            out.text,
            "mister Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
        );
    }
}
