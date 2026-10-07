//! Stream a WAV (or the live mic) through the wake detector; print scores and wake events.
//!
//! ```text
//! cargo run -p ochre-wake --example wake_stream -- --wav clip.wav [--model assets/wake/transcribe.onnx]
//! cargo run -p ochre-wake --example wake_stream -- --mic --seconds 60
//! ```
//!
//! Options: `--model PATH` (default `assets/wake/transcribe.onnx`), `--threshold T` (default the
//! model's `recommended.threshold`, else 0.5), `--vad energy|none` (default energy: a stand-in for
//! Silero, RMS above `--vad-rms`, default 0.005), `--all` (print every frame's score, else only
//! frames scoring >= 0.1), `--realtime` (pace a WAV in real time, to measure idle CPU), `--pad S`
//! (silence before and after a WAV, default 1 s), `--loop N` (repeat the WAV N times).

#[path = "../tests/support/wav.rs"]
mod wav;

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use ochre_wake::{DetectorOptions, FRAME, WakeDetector, recommended_threshold};

struct Args {
    model: PathBuf,
    threshold: Option<f32>,
    wav: Option<PathBuf>,
    mic: bool,
    seconds: f64,
    vad: bool,
    vad_rms: f32,
    all: bool,
    realtime: bool,
    pad: f64,
    repeat: usize,
}

fn parse() -> Args {
    let mut a = Args {
        model: PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/wake/transcribe.onnx"
        )),
        threshold: None,
        wav: None,
        mic: false,
        seconds: 30.0,
        vad: true,
        vad_rms: 0.005,
        all: false,
        realtime: false,
        pad: 1.0,
        repeat: 1,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().unwrap_or_else(|| panic!("{k} needs a value"));
        match k.as_str() {
            "--model" => a.model = v().into(),
            "--threshold" => a.threshold = Some(v().parse().expect("threshold")),
            "--wav" => a.wav = Some(v().into()),
            "--mic" => a.mic = true,
            "--seconds" => a.seconds = v().parse().expect("seconds"),
            "--vad" => a.vad = v() != "none",
            "--vad-rms" => a.vad_rms = v().parse().expect("vad-rms"),
            "--all" => a.all = true,
            "--realtime" => a.realtime = true,
            "--pad" => a.pad = v().parse().expect("pad"),
            "--loop" => a.repeat = v().parse().expect("loop"),
            "-h" | "--help" => {
                println!("see the doc comment at the top of examples/wake_stream.rs");
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
    }
    if a.wav.is_none() && !a.mic {
        panic!("give --wav PATH or --mic");
    }
    a
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse();
    let threshold = args
        .threshold
        .or_else(|| recommended_threshold(&args.model))
        .unwrap_or(0.5);
    let t_load = Instant::now();
    let mut det = WakeDetector::with_options(
        &args.model,
        DetectorOptions {
            threshold,
            ..Default::default()
        },
    )?;
    if args.vad {
        let rms_t = args.vad_rms;
        det = det.with_vad(move |f: &[f32]| {
            (f.iter().map(|v| v * v).sum::<f32>() / f.len() as f32).sqrt() > rms_t
        });
    }
    println!(
        "model {} (threshold {threshold:.2}, vad {}), loaded in {} ms",
        args.model.display(),
        if args.vad { "energy" } else { "off" },
        t_load.elapsed().as_millis()
    );

    let mut busy = Duration::ZERO;
    let on_block = |det: &mut WakeDetector, block: &[f32], busy: &mut Duration| {
        let before = det.stats.inferred;
        let t = Instant::now();
        let hit = det.process(block);
        *busy += t.elapsed();
        let frame = det.frame_index();
        let ts = frame as f64 * 0.08;
        if det.stats.inferred > before && (args.all || det.last_score >= 0.1) {
            println!("  t={ts:7.2}s frame={frame:5} score={:.4}", det.last_score);
        }
        if let Some(h) = hit {
            println!(
                "WAKE t={ts:.2}s frame={} rise_frame={} (word end ~{:.2}s) score={:.4} model={}",
                h.frame,
                h.rise_frame,
                h.rise_frame as f64 * 0.08,
                h.score,
                h.model
            );
        }
    };

    let wall = Instant::now();
    let audio_s;
    if let Some(path) = &args.wav {
        let (rate, x) = wav::read_wav(path)?;
        let x = wav::to_16k(rate, &x);
        let pad = vec![0i16; (args.pad * 16_000.0) as usize];
        let mut stream: Vec<f32> = Vec::new();
        for _ in 0..args.repeat {
            stream.extend(
                pad.iter()
                    .chain(&x)
                    .chain(&pad)
                    .map(|&v| v as f32 / 32767.0),
            );
        }
        audio_s = stream.len() as f64 / 16_000.0;
        println!(
            "{}: {:.2} s of audio ({} Hz source)",
            path.display(),
            audio_s,
            rate
        );
        let start = Instant::now();
        for (k, block) in stream.chunks(FRAME).enumerate() {
            on_block(&mut det, block, &mut busy);
            if args.realtime {
                let due = start + Duration::from_secs_f64((k + 1) as f64 * 0.08);
                if let Some(d) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(d);
                }
            }
        }
    } else {
        audio_s = mic(&args, |block| on_block(&mut det, block, &mut busy))?;
    }

    let s = &det.stats;
    let per = if s.inferred > 0 {
        busy.as_secs_f64() * 1000.0 / s.inferred as f64
    } else {
        0.0
    };
    println!(
        "\n{} frames ({:.1} s audio), {} speech, {} inferred, {} hits; detector busy {:.1} ms total = {:.2}% of one core in real time ({per:.2} ms per inferred frame); wall {:.2} s",
        s.frames,
        audio_s,
        s.speech_frames,
        s.inferred,
        s.hits,
        busy.as_secs_f64() * 1000.0,
        100.0 * busy.as_secs_f64() / audio_s.max(1e-9),
        wall.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Default input device -> mono 16 kHz [-1, 1] blocks -> `f`, for `args.seconds`. Returns seconds.
fn mic(args: &Args, mut f: impl FnMut(&[f32])) -> Result<f64, Box<dyn std::error::Error>> {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let host = cpal::default_host();
    let dev = host.default_input_device().ok_or("no input device")?;
    let cfg = dev.default_input_config()?;
    let rate = cfg.sample_rate().0 as f64;
    let ch = cfg.channels() as usize;
    println!(
        "mic: {} @ {rate} Hz x{ch} ({:?}); listening {} s",
        dev.name()?,
        cfg.sample_format(),
        args.seconds
    );
    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let mut pos = 0.0f64; // fractional read position for the linear resampler
    let mut prev = 0.0f32;
    let step = rate / 16_000.0;
    let mut convert = move |mono: Vec<f32>| {
        let mut out = Vec::with_capacity((mono.len() as f64 / step) as usize + 1);
        while pos < mono.len() as f64 {
            let j = pos.floor() as isize;
            let fr = (pos - j as f64) as f32;
            let a = if j < 0 { prev } else { mono[j as usize] };
            let b = mono.get((j + 1) as usize).copied().unwrap_or(a);
            out.push(a + (b - a) * fr);
            pos += step;
        }
        pos -= mono.len() as f64;
        prev = *mono.last().unwrap_or(&prev);
        out
    };
    let err = |e| eprintln!("stream error: {e}");
    let stream = match cfg.sample_format() {
        cpal::SampleFormat::F32 => dev.build_input_stream(
            &cfg.clone().into(),
            move |d: &[f32], _: &_| {
                let mono = d
                    .chunks(ch)
                    .map(|c| c.iter().sum::<f32>() / ch as f32)
                    .collect();
                let _ = tx.send(convert(mono));
            },
            err,
            None,
        )?,
        cpal::SampleFormat::I16 => dev.build_input_stream(
            &cfg.clone().into(),
            move |d: &[i16], _: &_| {
                let mono = d
                    .chunks(ch)
                    .map(|c| c.iter().map(|&v| v as f32 / 32768.0).sum::<f32>() / ch as f32)
                    .collect();
                let _ = tx.send(convert(mono));
            },
            err,
            None,
        )?,
        other => return Err(format!("unsupported sample format {other:?}").into()),
    };
    stream.play()?;
    let end = Instant::now() + Duration::from_secs_f64(args.seconds);
    let mut n = 0usize;
    while let Some(left) = end.checked_duration_since(Instant::now()) {
        if let Ok(block) = rx.recv_timeout(left) {
            n += block.len();
            f(&block);
        }
    }
    Ok(n as f64 / 16_000.0)
}
