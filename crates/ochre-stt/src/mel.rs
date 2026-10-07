//! NeMo-style log-mel features (the `AudioToMelSpectrogramPreprocessor` used by Parakeet),
//! bit-for-bit in structure with onnx-asr's `NemoPreprocessorNumpy`:
//!
//! pre-emphasis 0.97 → zero-pad n_fft/2 each side → 512-point frames every 160 samples with a
//! symmetric 400-sample Hann window centred in the frame → |rfft|² → Slaney mel filterbank
//! (0–8 kHz) → ln(x + 2⁻²⁴) → per-feature mean/variance normalisation over the valid frames
//! (unbiased variance, `/(std + 1e-5)`), frames past the valid length zeroed.
//!
//! Output layout is `[n_mels, frames]` row-major, i.e. the `[1, n_mels, T]` tensor the encoder
//! takes. All buffers are reused between calls, so steady-state feature extraction allocates
//! nothing but the output.

use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex, num_complex::Complex32};

pub const N_FFT: usize = 512;
pub const WIN_LENGTH: usize = 400;
pub const HOP: usize = 160;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 5.960_464_5e-8; // 2^-24

pub struct MelFrontend {
    n_mels: usize,
    /// `[n_bins][n_mels]` (bin-major so the inner loop runs over contiguous mels), with each
    /// bin's non-zero mel range in `ranges` so the matmul skips the zeros.
    fbank: Vec<f32>,
    ranges: Vec<(usize, usize)>,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
    scratch: Vec<Complex32>,
    frame: Vec<f32>,
    spec: Vec<Complex32>,
    padded: Vec<f32>,
}

impl MelFrontend {
    pub fn new(n_mels: usize, sample_rate: u32) -> Self {
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        let scratch = fft.make_scratch_vec();
        let spec = fft.make_output_vec();
        let n_bins = N_FFT / 2 + 1;
        let fb = slaney_mel_filterbank(
            n_mels,
            N_FFT,
            sample_rate as f64,
            0.0,
            sample_rate as f64 / 2.0,
        );
        let ranges = (0..n_bins)
            .map(|b| {
                let row = &fb[b * n_mels..(b + 1) * n_mels];
                let lo = row.iter().position(|&w| w != 0.0).unwrap_or(0);
                let hi = row.iter().rposition(|&w| w != 0.0).map_or(0, |i| i + 1);
                (lo, hi.max(lo))
            })
            .collect();
        Self {
            n_mels,
            fbank: fb,
            ranges,
            window: centred_hann(),
            fft,
            scratch,
            frame: vec![0.0; N_FFT],
            spec,
            padded: Vec::new(),
        }
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    /// Number of frames in the output for `n` samples (`n / hop + 1`).
    pub fn frames_for(n: usize) -> usize {
        n / HOP + 1
    }

    /// Valid (normalised, non-zero) frames for `n` samples (`n / hop`), the encoder `length`.
    pub fn valid_frames_for(n: usize) -> usize {
        n / HOP
    }

    /// Features for mono 16 kHz PCM. Returns `[n_mels * frames]`, row-major `[n_mels][frames]`.
    pub fn compute(&mut self, pcm: &[f32]) -> Vec<f32> {
        let n = pcm.len();
        let frames = Self::frames_for(n);
        let valid = Self::valid_frames_for(n);
        let pad = N_FFT / 2;

        // Pre-emphasis straight into the zero-padded buffer.
        self.padded.clear();
        self.padded.resize(n + 2 * pad, 0.0);
        let mut prev = 0.0f32;
        for (dst, &x) in self.padded[pad..pad + n].iter_mut().zip(pcm) {
            *dst = x - PREEMPH * prev;
            prev = x;
        }

        let n_mels = self.n_mels;
        let mut out = vec![0.0f32; n_mels * frames];
        let mut mel = vec![0.0f32; n_mels];
        for t in 0..frames {
            let start = t * HOP;
            for ((f, &x), &w) in self
                .frame
                .iter_mut()
                .zip(&self.padded[start..start + N_FFT])
                .zip(&self.window)
            {
                *f = x * w;
            }
            self.fft
                .process_with_scratch(&mut self.frame, &mut self.spec, &mut self.scratch)
                .expect("fft sizes are fixed");
            mel.iter_mut().for_each(|m| *m = 0.0);
            for (b, c) in self.spec.iter().enumerate() {
                let p = c.re * c.re + c.im * c.im;
                let (lo, hi) = self.ranges[b];
                let row = &self.fbank[b * n_mels..(b + 1) * n_mels];
                for (m, &w) in mel[lo..hi].iter_mut().zip(&row[lo..hi]) {
                    *m += p * w;
                }
            }
            for (m, &e) in mel.iter().enumerate() {
                out[m * frames + t] = (e + LOG_GUARD).ln();
            }
        }

        // Per-feature normalisation over the valid frames; everything after is zero.
        for m in 0..n_mels {
            let row = &mut out[m * frames..(m + 1) * frames];
            if valid == 0 {
                row.iter_mut().for_each(|x| *x = 0.0);
                continue;
            }
            let mean = row[..valid].iter().map(|&x| x as f64).sum::<f64>() / valid as f64;
            let var = row[..valid]
                .iter()
                .map(|&x| (x as f64 - mean).powi(2))
                .sum::<f64>()
                / (valid as f64 - 1.0);
            let denom = (var.sqrt() + 1e-5) as f32;
            let mean = mean as f32;
            for x in &mut row[..valid] {
                *x = (*x - mean) / denom;
            }
            row[valid..].iter_mut().for_each(|x| *x = 0.0);
        }
        out
    }
}

/// `np.hanning(400)` (symmetric) zero-padded to 512, centred.
fn centred_hann() -> Vec<f32> {
    let mut w = vec![0.0f32; N_FFT];
    let off = (N_FFT - WIN_LENGTH) / 2;
    for i in 0..WIN_LENGTH {
        let v = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (WIN_LENGTH - 1) as f64).cos();
        w[off + i] = v as f32;
    }
    w
}

fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// librosa `filters.mel(htk=False, norm="slaney")`, transposed to `[n_bins][n_mels]`.
pub fn slaney_mel_filterbank(
    n_mels: usize,
    n_fft: usize,
    sr: f64,
    fmin: f64,
    fmax: f64,
) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let (mmin, mmax) = (hz_to_mel(fmin), hz_to_mel(fmax));
    let hz: Vec<f64> = (0..n_mels + 2)
        .map(|i| mel_to_hz(mmin + (mmax - mmin) * i as f64 / (n_mels + 1) as f64))
        .collect();
    let mut fb = vec![0.0f32; n_bins * n_mels];
    for m in 0..n_mels {
        let (lo, ce, hi) = (hz[m], hz[m + 1], hz[m + 2]);
        let enorm = 2.0 / (hi - lo);
        for b in 0..n_bins {
            let f = sr / 2.0 * b as f64 / (n_bins - 1) as f64;
            let w = ((f - lo) / (ce - lo)).min((hi - f) / (hi - ce)).max(0.0);
            fb[b * n_mels + m] = (w * enorm) as f32;
        }
    }
    fb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes_and_normalisation() {
        let mut fe = MelFrontend::new(128, 16_000);
        let pcm: Vec<f32> = (0..16_000)
            .map(|i| (i as f32 * 0.05).sin() * 0.1 + ((i * 7919) % 13) as f32 * 1e-3)
            .collect();
        let f = fe.compute(&pcm);
        let frames = MelFrontend::frames_for(pcm.len());
        assert_eq!(frames, 101);
        assert_eq!(f.len(), 128 * frames);
        let valid = MelFrontend::valid_frames_for(pcm.len());
        for m in [0, 40, 127] {
            let row = &f[m * frames..m * frames + valid];
            let mean: f32 = row.iter().sum::<f32>() / valid as f32;
            assert!(mean.abs() < 1e-3, "row {m} mean {mean}");
            assert_eq!(f[m * frames + frames - 1], 0.0);
        }
        assert!(f.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn filterbank_rows_are_triangles() {
        let fb = slaney_mel_filterbank(128, 512, 16_000.0, 0.0, 8_000.0);
        assert_eq!(fb.len(), 257 * 128);
        for m in 0..128 {
            assert!((0..257).any(|b| fb[b * 128 + m] > 0.0), "empty mel {m}");
        }
    }

    /// Compares against onnx-asr's shipped `nemo128` filterbank and its NumPy features for a
    /// LibriSpeech clip. Needs files exported by the reference script (see the bench docs):
    /// OCHRE_REF_DIR with nemo128_fbank.f32, ls00_feat.f32 and wavs/ls00.wav.
    #[test]
    #[ignore = "needs reference files exported from the Python onnx-asr package (OCHRE_REF_DIR)"]
    fn matches_onnx_asr_reference() {
        let dir = std::path::PathBuf::from(std::env::var("OCHRE_REF_DIR").expect("OCHRE_REF_DIR"));
        let read_f32 = |p: &std::path::Path| -> Vec<f32> {
            std::fs::read(p)
                .unwrap()
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        };
        let want_fb = read_f32(&dir.join("nemo128_fbank.f32"));
        let fb = slaney_mel_filterbank(128, 512, 16_000.0, 0.0, 8_000.0);
        let max_fb = want_fb
            .iter()
            .zip(&fb)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_fb < 1e-6, "filterbank max abs diff {max_fb}");

        let mut r = hound::WavReader::open(dir.join("wavs/ls00.wav")).unwrap();
        let pcm: Vec<f32> = r
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect();
        let want = read_f32(&dir.join("ls00_feat.f32"));
        let got = MelFrontend::new(128, 16_000).compute(&pcm);
        assert_eq!(want.len(), got.len());
        let max = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean = want
            .iter()
            .zip(&got)
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / want.len() as f32;
        eprintln!("features: max abs diff {max:.2e}, mean {mean:.2e}");
        assert!(
            max < 2e-3 && mean < 1e-4,
            "features differ: max {max}, mean {mean}"
        );
    }
}
