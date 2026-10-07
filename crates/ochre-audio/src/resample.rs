//! High-quality sample-rate conversion to the engines' 16 kHz (rubato's FFT resampler:
//! BlackmanHarris2 anti-aliasing, fixed ratio). Done once, on the capture pump thread.

use audioadapter_buffers::direct::InterleavedSlice;
use ochre_core::{Error, Result};
use rubato::{Fft, FixedSync, Resampler};

/// Streaming mono resampler: feed arbitrary-length blocks, get output as soon as a 10 ms chunk
/// is complete. Allocation-free after construction.
pub struct StreamResampler {
    inner: Fft<f32>,
    from: u32,
    pending: Vec<f32>,
    out: Vec<f32>,
}

impl StreamResampler {
    pub fn new(from: u32, to: u32) -> Result<Self> {
        let chunk = (from as usize / 100).max(64);
        let inner = Fft::<f32>::new(from as usize, to as usize, chunk, 1, FixedSync::Input)
            .map_err(|e| Error::Audio(format!("resampler {from}->{to}: {e}")))?;
        let out = vec![0.0; inner.output_frames_max()];
        Ok(Self {
            pending: Vec::with_capacity(chunk * 4),
            inner,
            from,
            out,
        })
    }

    pub fn input_rate(&self) -> u32 {
        self.from
    }

    /// Resample `input`, appending output to `out`.
    pub fn process(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.pending.extend_from_slice(input);
        let need = self.inner.input_frames_next();
        let mut pos = 0;
        while self.pending.len() - pos >= need {
            let n_out = self.inner.output_frames_next();
            let inp =
                InterleavedSlice::new(&self.pending[pos..pos + need], 1, need).expect("sizes");
            let mut o = InterleavedSlice::new_mut(&mut self.out[..n_out], 1, n_out).expect("sizes");
            match self.inner.process_into_buffer(&inp, &mut o, None) {
                Ok((used, produced)) => {
                    out.extend_from_slice(&self.out[..produced]);
                    pos += used;
                }
                Err(e) => {
                    tracing::error!(error = %e, "resampler failed; dropping a chunk");
                    pos += need;
                }
            }
        }
        self.pending.drain(..pos);
    }
}

/// Offline mono resample of a whole clip (e.g. a WAV file), delay-compensated.
pub fn resample(pcm: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to || pcm.is_empty() {
        return Ok(pcm.to_vec());
    }
    let mut r = Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Input)
        .map_err(|e| Error::Audio(format!("resampler {from}->{to}: {e}")))?;
    let mut out = vec![0.0f32; r.process_all_needed_output_len(pcm.len())];
    let inp = InterleavedSlice::new(pcm, 1, pcm.len()).map_err(|e| Error::Audio(e.to_string()))?;
    let n = out.len();
    let mut o =
        InterleavedSlice::new_mut(&mut out, 1, n).map_err(|e| Error::Audio(e.to_string()))?;
    let (_, produced) = r
        .process_all_into_buffer(&inp, &mut o, pcm.len(), None)
        .map_err(|e| Error::Audio(e.to_string()))?;
    out.truncate(produced);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f32, secs: f32) -> Vec<f32> {
        (0..(rate as f32 * secs) as usize)
            .map(|i| (2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin() * 0.5)
            .collect()
    }

    /// Correlation-based amplitude of `hz` in `x`.
    fn amp(x: &[f32], rate: u32, hz: f32) -> f32 {
        let (mut s, mut c) = (0.0f64, 0.0f64);
        for (i, v) in x.iter().enumerate() {
            let ph = 2.0 * std::f64::consts::PI * hz as f64 * i as f64 / rate as f64;
            s += *v as f64 * ph.sin();
            c += *v as f64 * ph.cos();
        }
        (2.0 * (s * s + c * c).sqrt() / x.len() as f64) as f32
    }

    #[test]
    fn streaming_matches_length_keeps_passband_and_kills_aliases() {
        for from in [48_000u32, 44_100, 32_000] {
            let mut r = StreamResampler::new(from, 16_000).unwrap();
            // 1 kHz (keep) + 11 kHz (must not alias down to 5 kHz at 16 kHz).
            let x: Vec<f32> = tone(from, 1000.0, 2.0)
                .iter()
                .zip(tone(from, 11_000.0, 2.0))
                .map(|(a, b)| a + b)
                .collect();
            let mut out = Vec::new();
            for block in x.chunks(441) {
                r.process(block, &mut out);
            }
            let expected = 32_000usize;
            assert!(
                out.len() + 1000 >= expected && out.len() <= expected,
                "{from}: {} samples",
                out.len()
            );
            let steady = &out[4000..out.len() - 1000];
            let keep = amp(steady, 16_000, 1000.0);
            let alias = amp(steady, 16_000, 16_000.0 - 11_000.0);
            assert!(
                (keep - 0.5).abs() < 0.01,
                "{from}: passband amplitude {keep}"
            );
            assert!(alias < 0.001, "{from}: alias amplitude {alias}");
        }
    }

    #[test]
    fn offline_resample_is_delay_compensated() {
        let mut x = vec![0.0f32; 48_000];
        x[24_000] = 1.0; // impulse at 0.5 s
        let y = resample(&x, 48_000, 16_000).unwrap();
        assert!((y.len() as i64 - 16_000).abs() <= 2, "{}", y.len());
        let peak = y
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .unwrap()
            .0;
        assert!((peak as i64 - 8_000).abs() <= 1, "impulse moved to {peak}");
    }
}
