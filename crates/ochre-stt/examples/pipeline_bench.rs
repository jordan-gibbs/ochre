//! Release → text through the real pipeline pieces: `ochre_audio::PhraseSegmenter` (decode thread
//! at raised priority) + Parakeet, fed in real time.
//!
//! ```text
//! cargo run -p ochre-stt --example pipeline_bench --release -- <wav...|dir> [--background-load N]
//! ```
//!
//! Each 16 kHz clip is fed in 10 ms blocks at real-time pace (as the mic would), then `release()`
//! is called at the end of the clip, 150 ms of post-roll silence is fed (in real time, as the
//! mic's post-roll would arrive), and `finish()` waits for the result. Reported per clip:
//! release → text (the number the user feels, minus injection), the tail audio decoded after
//! release, and the phrase count.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ochre_audio::segmenter::{DecodeThread, PhraseSegmenter, TranscribeFn};
use ochre_core::stt::{SttEngine, SttOptions};

fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    if s.is_empty() {
        0.0
    } else {
        s[((s.len() - 1) as f64 * p).round() as usize]
    }
}

fn feed_realtime(seg: &mut ochre_audio::Segmenter, pcm: &[f32]) {
    let start = Instant::now();
    for (i, block) in pcm.chunks(160).enumerate() {
        let due = start + Duration::from_millis(10 * i as u64);
        if let Some(d) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(d);
        }
        seg.feed(block);
    }
}

fn run(
    label: &str,
    factory: &PhraseSegmenter,
    transcribe: &TranscribeFn,
    clips: &[(String, Vec<f32>)],
) -> Vec<f64> {
    println!("== {label}");
    println!(
        "{:<12} {:>7} {:>8} {:>16} {:>7}  text",
        "file", "audio_s", "phrases", "release→text ms", "tail_s"
    );
    let mut out = Vec::new();
    for (name, pcm) in clips {
        let mut seg = factory.start(transcribe.clone(), None);
        feed_realtime(&mut seg, pcm);
        let released = Instant::now();
        seg.release();
        feed_realtime(&mut seg, &vec![0.0; 2400]); // 150 ms post-roll
        let r = seg.finish().expect("finish");
        let ms = released.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{:<12} {:>7.2} {:>8} {:>16.0} {:>7.2}  {}",
            name,
            pcm.len() as f64 / 16_000.0,
            r.phrases.len(),
            ms,
            r.tail_audio_ms as f64 / 1000.0,
            r.text
        );
        out.push(ms);
    }
    println!(
        "-- {label}: release→text p50 {:.0} ms, p95 {:.0} ms (includes the 150 ms post-roll)",
        pct(&out, 0.5),
        pct(&out, 0.95)
    );
    out
}

fn main() {
    let mut files = Vec::new();
    let mut load = 0usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--background-load" {
            load = args.next().unwrap().parse().unwrap();
            continue;
        }
        let p = PathBuf::from(a);
        if p.is_dir() {
            let mut v: Vec<_> = std::fs::read_dir(&p)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "wav"))
                .collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(p);
        }
    }
    let clips: Vec<(String, Vec<f32>)> = files
        .iter()
        .map(|f| {
            let mut r = hound::WavReader::open(f).unwrap();
            assert_eq!(r.spec().sample_rate, 16_000);
            (
                f.file_name().unwrap().to_string_lossy().to_string(),
                r.samples::<i16>()
                    .map(|s| s.unwrap() as f32 / 32768.0)
                    .collect(),
            )
        })
        .collect();

    let mut engine = ochre_stt::Parakeet::new("", "cpu").unwrap();
    let t = Instant::now();
    engine.load(&|_| {}).unwrap();
    println!("parakeet ready in {} ms", t.elapsed().as_millis());
    let engine = Arc::new(engine);
    let e2 = engine.clone();
    let transcribe: TranscribeFn =
        Arc::new(move |pcm: &[f32]| e2.transcribe(pcm, &SttOptions::default()));
    let factory = PhraseSegmenter::new(DecodeThread::spawn(None)).with_partial_interval(None);

    let idle = run("idle", &factory, &transcribe, &clips);
    if load > 0 {
        let stop = Arc::new(AtomicBool::new(false));
        let spinners: Vec<_> = (0..load)
            .map(|_| {
                let stop = stop.clone();
                std::thread::spawn(move || {
                    let mut x = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        x = std::hint::black_box(
                            x.wrapping_mul(6364136223846793005).wrapping_add(1),
                        );
                    }
                })
            })
            .collect();
        let loaded = run(
            &format!("background load: {load} busy threads"),
            &factory,
            &transcribe,
            &clips,
        );
        stop.store(true, Ordering::Relaxed);
        for s in spinners {
            let _ = s.join();
        }
        println!(
            "GATE release→text p95 under load ≤ 2× idle: idle {:.0} ms, loaded {:.0} ms → {:.2}× → {}",
            pct(&idle, 0.95),
            pct(&loaded, 0.95),
            pct(&loaded, 0.95) / pct(&idle, 0.95),
            if pct(&loaded, 0.95) <= 2.0 * pct(&idle, 0.95) {
                "PASS"
            } else {
                "FAIL"
            }
        );
    }
}
