//! Silero VAD (ONNX, MIT) for speech gating: hands-free idle listening and the "is there any
//! speech in this phrase?" silence guard before decoding.
//!
//! The model is the v5-interface graph (`input` = 64 samples of context + 512 new samples at
//! 16 kHz, recurrent `state` `[2,1,128]`, scalar `sr`), currently the v6.2 weights from
//! `istupakov/silero-vad-onnx`, pinned by revision and SHA-256. It uses a single-
//! threaded, non-spinning ORT session (spinning pools burn idle CPU) and an optional RMS floor
//! that skips inference on near-silent chunks, so the VAD costs almost nothing while the room
//! is quiet.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use ochre_core::{Error, Result, SAMPLE_RATE};
use ochre_models::{ModelFile, ModelManifest};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

pub const CHUNK: usize = 512;
pub const CONTEXT: usize = 64;
const STATE_LEN: usize = 2 * 128;
const REPO: &str = "istupakov/silero-vad-onnx";
const REVISION: &str = "b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6";
const FILE: &str = "silero_vad.onnx";
const SIZE: u64 = 2_327_524;
const SHA256: &str = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3";

pub fn manifest() -> ModelManifest {
    ModelManifest {
        id: "silero-vad".into(),
        files: vec![ModelFile::hf(
            REPO,
            REVISION,
            FILE,
            Some(SHA256),
            Some(SIZE),
        )],
    }
}

fn err(e: impl std::fmt::Display) -> Error {
    Error::Model(format!("silero vad: {e}"))
}

/// Streaming speech probability for 16 kHz mono f32 of any block size. Not `Sync`; wrap it in a
/// `Mutex` to share it (e.g. as a `Segmenter` speech check).
pub struct SileroVad {
    session: Session,
    state: Vec<f32>,
    input: Vec<f32>,
    rem: Vec<f32>,
    energy_floor: f32,
    gap: usize,
    pub inferences: u64,
    pub skipped: u64,
}

impl SileroVad {
    /// Download (once, verified) and open the model from `models_dir()`.
    pub fn load(progress: &dyn Fn(&str, u64, u64), cancel: &AtomicBool) -> Result<Self> {
        let dir = manifest().ensure(progress, cancel)?;
        Self::open(&dir.join(FILE))
    }

    pub fn open(path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| err(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| err(e.to_string()))?
            .with_inter_threads(1)
            .map_err(|e| err(e.to_string()))?
            .with_intra_op_spinning(false)
            .map_err(|e| err(e.to_string()))?
            .with_inter_op_spinning(false)
            .map_err(|e| err(e.to_string()))?
            .commit_from_file(path)
            .map_err(err)?;
        Ok(Self {
            session,
            state: vec![0.0; STATE_LEN],
            input: vec![0.0; CONTEXT + CHUNK],
            rem: Vec::with_capacity(CHUNK),
            energy_floor: 0.0,
            gap: 0,
            inferences: 0,
            skipped: 0,
        })
    }

    /// Chunks with RMS below `floor` score 0 without running the model (0 = always run).
    pub fn with_energy_floor(mut self, floor: f32) -> Self {
        self.energy_floor = floor;
        self
    }

    pub fn reset(&mut self) {
        self.state.iter_mut().for_each(|x| *x = 0.0);
        self.input.iter_mut().for_each(|x| *x = 0.0);
        self.rem.clear();
        self.gap = 0;
    }

    fn chunk(&mut self, c: &[f32]) -> Result<f32> {
        let rms = (c.iter().map(|x| x * x).sum::<f32>() / c.len() as f32).sqrt();
        // Slide the 64-sample context: input = [context | chunk].
        let skip = self.energy_floor > 0.0 && rms < self.energy_floor;
        if skip {
            self.skipped += 1;
            self.gap += 1;
            self.input[..CONTEXT].copy_from_slice(&c[CHUNK - CONTEXT..]);
            return Ok(0.0);
        }
        if self.gap > 30 {
            // ~1 s skipped: the recurrent state is stale.
            self.state.iter_mut().for_each(|x| *x = 0.0);
        }
        self.gap = 0;
        self.input[CONTEXT..].copy_from_slice(c);
        self.inferences += 1;
        let sr = [SAMPLE_RATE as i64];
        let outputs = self
            .session
            .run(ort::inputs![
                "input" => TensorRef::from_array_view(([1usize, CONTEXT + CHUNK], &self.input[..])).map_err(err)?,
                "state" => TensorRef::from_array_view(([2usize, 1, 128], &self.state[..])).map_err(err)?,
                "sr" => TensorRef::from_array_view((Vec::<usize>::new(), &sr[..])).map_err(err)?,
            ])
            .map_err(err)?;
        let (_, p) = outputs["output"].try_extract_tensor::<f32>().map_err(err)?;
        let p = p[0];
        let (_, s) = outputs["stateN"].try_extract_tensor::<f32>().map_err(err)?;
        self.state.copy_from_slice(&s[..STATE_LEN]);
        drop(outputs);
        let (head, tail) = self.input.split_at_mut(CONTEXT);
        head.copy_from_slice(&tail[CHUNK - CONTEXT..]);
        Ok(p)
    }

    /// Probabilities for each 32 ms chunk completed by `block` (leftovers carry over).
    pub fn probs(&mut self, block: &[f32]) -> Result<Vec<f32>> {
        let mut out = Vec::with_capacity((self.rem.len() + block.len()) / CHUNK);
        let mut rest = block;
        if !self.rem.is_empty() {
            let need = CHUNK - self.rem.len();
            let take = need.min(rest.len());
            self.rem.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.rem.len() == CHUNK {
                let c = std::mem::take(&mut self.rem);
                out.push(self.chunk(&c)?);
                self.rem = c;
                self.rem.clear();
            }
        }
        let (chunks, tail) = rest.as_chunks::<CHUNK>();
        for c in chunks {
            out.push(self.chunk(c)?);
        }
        self.rem.extend_from_slice(tail);
        Ok(out)
    }

    /// Max probability over the chunks completed by `block` (0 if none completed).
    pub fn prob(&mut self, block: &[f32]) -> Result<f32> {
        Ok(self.probs(block)?.into_iter().fold(0.0, f32::max))
    }

    /// Fraction of 32 ms chunks at or above `threshold` in a whole clip (state reset around it).
    pub fn speech_ratio(&mut self, audio: &[f32], threshold: f32) -> Result<f32> {
        self.reset();
        let p = self.probs(audio)?;
        self.reset();
        Ok(if p.is_empty() {
            0.0
        } else {
            p.iter().filter(|&&x| x >= threshold).count() as f32 / p.len() as f32
        })
    }

    /// At least `min_chunks` chunks (3 ≈ 100 ms) of speech: the silence guard for a phrase.
    pub fn has_speech(&mut self, audio: &[f32], threshold: f32, min_chunks: usize) -> Result<bool> {
        self.reset();
        let hits = self
            .probs(audio)?
            .iter()
            .filter(|&&x| x >= threshold)
            .count();
        self.reset();
        Ok(hits >= min_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_wav(p: &Path) -> Vec<f32> {
        let mut r = hound::WavReader::open(p).unwrap();
        assert_eq!(r.spec().sample_rate, 16_000);
        r.samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect()
    }

    #[test]
    #[ignore = "network/model: downloads the 2.3 MB Silero model; OCHRE_REF_DIR for a speech WAV"]
    fn speech_vs_silence() {
        let mut vad = SileroVad::load(&|_, _, _| {}, &AtomicBool::new(false)).unwrap();
        let silence = vec![0.0f32; 16_000];
        assert!(!vad.has_speech(&silence, 0.5, 3).unwrap());
        let noise: Vec<f32> = (0..16_000)
            .map(|i| (((i as u32).wrapping_mul(2654435761) >> 16) as f32 / 65536.0 - 0.5) * 0.02)
            .collect();
        assert!(vad.speech_ratio(&noise, 0.5).unwrap() < 0.1);
        if let Ok(dir) = std::env::var("OCHRE_REF_DIR") {
            let speech = read_wav(&Path::new(&dir).join("wavs/ls00.wav"));
            let ratio = vad.speech_ratio(&speech, 0.5).unwrap();
            eprintln!("speech ratio on LibriSpeech clip: {ratio:.2}");
            assert!(ratio > 0.5);
            // Streaming in odd-sized blocks gives the same chunk count as one call.
            vad.reset();
            let n: usize = speech
                .chunks(333)
                .map(|b| vad.probs(b).unwrap().len())
                .sum();
            assert_eq!(n, speech.len() / CHUNK);
        }
        // The energy floor skips inference on silence.
        let mut vad = vad.with_energy_floor(0.001);
        vad.probs(&silence).unwrap();
        assert!(vad.skipped >= 30);
    }
}
