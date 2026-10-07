//! Short, quiet UI sounds: start, stop, cancel, error.
//!
//! Tones are synthesized at the output device's own rate (no resampling, no asset files): soft
//! sine partials with a 4 ms attack and exponential decay, peaking around -22 dBFS, the same
//! recipe as the prototype's `earcons.py`. `play()` only sends on a channel; a dedicated
//! `ochre-earcons` thread opens the output stream, plays, and closes it, so a sound can never delay
//! the hotkey path or mic capture.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Earcon {
    Start,
    Stop,
    Cancel,
    Error,
}

const PEAK_DBFS: f32 = -22.0;

/// (start_s, freq_hz, dur_s, gain): small, warm intervals rather than beeps.
fn notes(e: Earcon) -> &'static [(f32, f32, f32, f32)] {
    match e {
        Earcon::Start => &[(0.00, 659.25, 0.16, 0.8), (0.07, 987.77, 0.20, 1.0)], // E5 -> B5
        Earcon::Stop => &[(0.00, 987.77, 0.16, 0.9), (0.07, 659.25, 0.22, 1.0)],  // B5 -> E5
        Earcon::Cancel => &[(0.00, 523.25, 0.12, 1.0), (0.05, 392.00, 0.18, 0.8)], // C5 -> G4
        Earcon::Error => &[(0.00, 311.13, 0.14, 1.0), (0.16, 293.66, 0.22, 1.0)], // Eb4 -> D4
    }
}

/// Mono samples for `e` at `rate`, peak-normalized to -22 dBFS with a 10 ms fade-out.
pub fn synth(e: Earcon, rate: u32) -> Vec<f32> {
    let ns = notes(e);
    let r = rate as f32;
    let total = ns.iter().map(|n| n.0 + n.2).fold(0.0, f32::max) + 0.02;
    let mut out = vec![0.0f32; (total * r) as usize];
    for &(start, freq, dur, gain) in ns {
        let i0 = (start * r) as usize;
        for i in 0..(dur * r) as usize {
            let t = i as f32 / r;
            let attack = (t / 0.004).min(1.0);
            let decay = (-t / (dur / 4.5)).exp();
            let w = 2.0 * std::f32::consts::PI * freq * t;
            // Fundamental plus a faint octave: rounder than a pure sine, nowhere near a buzzer.
            out[i0 + i] += gain * attack * decay * (w.sin() + 0.18 * (2.0 * w).sin());
        }
    }
    let peak = out.iter().fold(1e-9f32, |m, x| m.max(x.abs()));
    let scale = 10f32.powf(PEAK_DBFS / 20.0) / peak;
    let fade = (0.01 * r) as usize;
    let n = out.len();
    for (i, x) in out.iter_mut().enumerate() {
        *x *= scale;
        if i + fade >= n {
            *x *= (n - i) as f32 / fade as f32;
        }
    }
    out
}

/// Fire-and-forget earcon player.
pub struct Earcons {
    tx: Mutex<Sender<Earcon>>,
    enabled: Arc<AtomicBool>,
}

impl Earcons {
    pub fn new(enabled: bool) -> Self {
        let (tx, rx) = mpsc::channel::<Earcon>();
        let _ = std::thread::Builder::new()
            .name("ochre-earcons".into())
            .spawn(move || {
                while let Ok(e) = rx.recv() {
                    // Coalesce a burst (e.g. start immediately followed by cancel): play the latest.
                    let e = rx.try_iter().last().unwrap_or(e);
                    if let Err(err) = play_blocking(e) {
                        tracing::debug!(error = %err, "earcon playback failed");
                    }
                }
            });
        Self {
            tx: Mutex::new(tx),
            enabled: Arc::new(AtomicBool::new(enabled)),
        }
    }

    pub fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    /// Returns immediately.
    pub fn play(&self, e: Earcon) {
        if self.enabled.load(Ordering::Relaxed) {
            let _ = self.tx.lock().unwrap().send(e);
        }
    }
}

/// Open the default output, play `e` once, close. Blocks for the sound's duration.
pub fn play_blocking(e: Earcon) -> Result<(), String> {
    let dev = cpal::default_host()
        .default_output_device()
        .ok_or("no output device")?;
    let cfg = dev.default_output_config().map_err(|e| e.to_string())?;
    let format = cfg.sample_format();
    let config: cpal::StreamConfig = cfg.into();
    let pcm: Arc<[f32]> = synth(e, config.sample_rate.0).into();
    let secs = pcm.len() as f32 / config.sample_rate.0 as f32;
    let stream = match format {
        SampleFormat::F32 => out_stream::<f32>(&dev, &config, pcm),
        SampleFormat::I16 => out_stream::<i16>(&dev, &config, pcm),
        SampleFormat::U16 => out_stream::<u16>(&dev, &config, pcm),
        SampleFormat::I32 => out_stream::<i32>(&dev, &config, pcm),
        other => return Err(format!("unsupported output format {other:?}")),
    }?;
    stream.play().map_err(|e| e.to_string())?;
    std::thread::sleep(Duration::from_secs_f32(secs + 0.05));
    Ok(())
}

fn out_stream<T>(
    dev: &cpal::Device,
    config: &cpal::StreamConfig,
    pcm: Arc<[f32]>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let ch = config.channels as usize;
    let pos = AtomicUsize::new(0);
    dev.build_output_stream(
        config,
        move |data: &mut [T], _| {
            let mut p = pos.load(Ordering::Relaxed);
            for frame in data.chunks_mut(ch) {
                let v = pcm.get(p).copied().unwrap_or(0.0);
                p += 1;
                for s in frame {
                    *s = T::from_sample(v);
                }
            }
            pos.store(p, Ordering::Relaxed);
        },
        |e| tracing::debug!(error = %e, "earcon stream error"),
        None,
    )
    .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tones_are_short_and_quiet() {
        for e in [Earcon::Start, Earcon::Stop, Earcon::Cancel, Earcon::Error] {
            let x = synth(e, 48_000);
            let secs = x.len() as f32 / 48_000.0;
            assert!((0.15..0.45).contains(&secs), "{e:?}: {secs}s");
            let peak = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                (20.0 * peak.log10() + 22.0).abs() < 0.1,
                "{e:?} peak {peak}"
            );
            assert!(x.last().unwrap().abs() < 1e-3, "fades out");
        }
    }

    #[test]
    fn play_never_blocks_the_caller() {
        let ec = Earcons::new(false); // disabled: nothing is sent, nothing is heard
        let t = std::time::Instant::now();
        for _ in 0..100 {
            ec.play(Earcon::Start);
        }
        assert!(t.elapsed() < Duration::from_millis(20));
    }
}
