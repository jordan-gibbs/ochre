//! Latency harness for local STT engines.
//!
//! ```text
//! cargo run -p ochre-stt --example bench --release -- [--engine parakeet|whisper] [--model M]
//!     [--device cpu|cuda|directml|coreml] [--repeat N] [--ref ref.json]
//!     [--background-load N] [--no-boost] <wav or dir>...
//! ```
//!
//! Prints load and warm-up time, then per-utterance decode ms and RTF (with the front end /
//! encoder / decoder split for Parakeet), and a summary with p50/p95.
//!
//! - `--ref` takes `{ "<file name>": { "text": "..." } }` (e.g. onnx-asr's output for the same
//!   WAVs) and reports identical transcripts and the WER between the two.
//! - `--background-load N` runs the set twice: idle, then with N busy-spinning threads at normal
//!   priority (the release gate is p95 under load ≤ 2× idle).
//! - The decode runs on the main thread, boosted like the app's decode thread; `--no-boost`
//!   leaves it at normal priority for comparison.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use ochre_core::stt::{SttEngine, SttOptions};

fn read_wav(path: &Path) -> Result<Vec<f32>, String> {
    let mut r = hound::WavReader::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let spec = r.spec();
    if spec.sample_rate != 16_000 {
        return Err(format!(
            "{}: {} Hz (bench needs 16 kHz input)",
            path.display(),
            spec.sample_rate
        ));
    }
    let ch = spec.channels as usize;
    let raw: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>()
                .map(|s| s.unwrap() as f32 * scale)
                .collect()
        }
    };
    Ok(raw
        .chunks(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect())
}

fn words(s: &str) -> Vec<String> {
    s.to_lowercase()
        .split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn edit_distance(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, y) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + usize::from(x != y))
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    if s.is_empty() {
        return 0.0;
    }
    s[((s.len() - 1) as f64 * p).round() as usize]
}

struct Row {
    name: String,
    secs: f64,
    ms: f64,
    text: String,
}

fn stage_columns(parakeet: Option<&ochre_stt::Parakeet>) -> String {
    match parakeet.and_then(ochre_stt::parakeet::last_stage_times) {
        Some(s) => format!(
            "{:>8.1} {:>8.1} {:>8.1} {:>6}",
            s.features_us as f64 / 1000.0,
            s.encoder_us as f64 / 1000.0,
            s.decoder_us as f64 / 1000.0,
            s.decoder_steps
        ),
        None => format!("{:>8} {:>8} {:>8} {:>6}", "-", "-", "-", "-"),
    }
}

fn run_pass(
    label: &str,
    engine: &dyn SttEngine,
    parakeet: Option<&ochre_stt::Parakeet>,
    clips: &[(String, Vec<f32>)],
    repeat: usize,
) -> Vec<Row> {
    println!("== {label}");
    println!(
        "{:<12} {:>7} {:>9} {:>7} {:>8} {:>8} {:>8} {:>6}  text",
        "file", "audio_s", "decode_ms", "rtf", "feat_ms", "enc_ms", "dec_ms", "steps"
    );
    let opts = SttOptions::default();
    let mut rows = Vec::new();
    for (name, pcm) in clips {
        let secs = pcm.len() as f64 / 16_000.0;
        let mut best = f64::INFINITY;
        let mut text = String::new();
        for _ in 0..repeat {
            let t = Instant::now();
            let r = engine.transcribe(pcm, &opts).expect("transcribe");
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
            text = r.text;
        }
        println!(
            "{:<12} {:>7.2} {:>9.1} {:>7.4} {}  {}",
            name,
            secs,
            best,
            best / 1000.0 / secs,
            stage_columns(parakeet),
            text
        );
        rows.push(Row {
            name: name.clone(),
            secs,
            ms: best,
            text,
        });
    }
    let audio: f64 = rows.iter().map(|r| r.secs).sum();
    let decode: f64 = rows.iter().map(|r| r.ms).sum();
    let per10: Vec<f64> = rows
        .iter()
        .filter(|r| r.secs >= 2.0)
        .map(|r| r.ms / r.secs * 10.0)
        .collect();
    let ms: Vec<f64> = rows.iter().map(|r| r.ms).collect();
    println!(
        "-- {label}: {:.1} s audio in {:.0} ms → RTF {:.4}; per-clip decode p50 {:.0} ms, p95 {:.0} ms; normalized to 10 s audio p50 {:.0} ms, p95 {:.0} ms",
        audio,
        decode,
        decode / 1000.0 / audio,
        pct(&ms, 0.5),
        pct(&ms, 0.95),
        pct(&per10, 0.5),
        pct(&per10, 0.95)
    );
    rows
}

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let mut args = std::env::args().skip(1);
    let (mut engine_id, mut model, mut device, mut repeat) = (
        "parakeet".to_string(),
        String::new(),
        "cpu".to_string(),
        1usize,
    );
    let (mut load_threads, mut boost) = (0usize, true);
    let mut reference: Option<serde_json::Value> = None;
    let mut files: Vec<PathBuf> = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--engine" => engine_id = args.next().expect("--engine value"),
            "--model" => model = args.next().expect("--model value"),
            "--device" => device = args.next().expect("--device value"),
            "--repeat" => {
                repeat = args
                    .next()
                    .expect("--repeat value")
                    .parse()
                    .expect("number")
            }
            "--background-load" => {
                load_threads = args
                    .next()
                    .expect("--background-load value")
                    .parse()
                    .expect("number")
            }
            "--no-boost" => boost = false,
            "--ref" => {
                let p = args.next().expect("--ref path");
                reference = Some(
                    serde_json::from_str(&std::fs::read_to_string(p).expect("read ref"))
                        .expect("ref json"),
                );
            }
            _ => {
                let p = PathBuf::from(&a);
                if p.is_dir() {
                    let mut v: Vec<_> = std::fs::read_dir(&p)
                        .unwrap()
                        .filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("wav")))
                        .collect();
                    v.sort();
                    files.extend(v);
                } else {
                    files.push(p);
                }
            }
        }
    }
    if files.is_empty() {
        eprintln!(
            "usage: bench [--engine E] [--model M] [--device D] [--repeat N] [--ref ref.json] [--background-load N] [--no-boost] <wav|dir>..."
        );
        std::process::exit(2);
    }
    let clips: Vec<(String, Vec<f32>)> = files
        .iter()
        .filter_map(|f| match read_wav(f) {
            Ok(p) => Some((f.file_name().unwrap().to_string_lossy().to_string(), p)),
            Err(e) => {
                eprintln!("skip {e}");
                None
            }
        })
        .collect();
    if boost {
        ochre_stt::priority::boost_current_thread();
    }
    println!(
        "engine {engine_id} model {:?} device requested {device}; decode thread boosted: {boost}; ORT intra-op threads {} of {} physical cores",
        if model.is_empty() { "default" } else { &model },
        ochre_stt::onnx::heavy_threads(),
        ochre_stt::priority::physical_cores()
    );

    let t0 = Instant::now();
    let progress = |p: ochre_core::stt::Progress| {
        eprint!(
            "\rdownload {}: {}/{} MB   ",
            p.item,
            p.done >> 20,
            p.total >> 20
        )
    };
    let mut parakeet: Option<ochre_stt::Parakeet> = None;
    let mut other: Option<Box<dyn SttEngine>> = None;
    if engine_id == "parakeet" {
        let mut p = ochre_stt::Parakeet::new(&model, &device).expect("engine");
        p.load(&progress).expect("load");
        println!(
            "load: {} ms to ready (sessions {} ms, warm-up {} ms) on {}",
            t0.elapsed().as_millis(),
            p.load_ms,
            p.warmup_ms,
            p.device().unwrap().as_str()
        );
        parakeet = Some(p);
    } else {
        let mut e = ochre_stt::create(&engine_id, &model, &device).expect("engine");
        e.load(&progress).expect("load");
        println!(
            "load: {} ms to ready (includes warm-up)",
            t0.elapsed().as_millis()
        );
        other = Some(e);
    }
    let engine: &dyn SttEngine = match (&parakeet, &other) {
        (Some(p), _) => p,
        (_, Some(e)) => e.as_ref(),
        _ => unreachable!(),
    };

    let idle = run_pass("idle", engine, parakeet.as_ref(), &clips, repeat);

    if let Some(reference) = &reference {
        let (mut identical, mut compared, mut errs, mut ref_words) = (0, 0, 0usize, 0usize);
        for r in &idle {
            if let Some(want) = reference
                .get(&r.name)
                .and_then(|v| v.get("text"))
                .and_then(|t| t.as_str())
            {
                compared += 1;
                if want == r.text {
                    identical += 1;
                } else {
                    println!("   diff {}: ref: {want}", r.name);
                }
                let (w, g) = (words(want), words(&r.text));
                errs += edit_distance(&w, &g);
                ref_words += w.len();
            }
        }
        println!(
            "vs reference: {identical}/{compared} identical, WER between the two {:.2}% ({errs} edits / {ref_words} words)",
            errs as f64 * 100.0 / ref_words.max(1) as f64
        );
    }

    if load_threads > 0 {
        let stop = Arc::new(AtomicBool::new(false));
        let spinners: Vec<_> = (0..load_threads)
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
        let loaded = run_pass(
            &format!("background load: {load_threads} busy threads"),
            engine,
            parakeet.as_ref(),
            &clips,
            repeat,
        );
        stop.store(true, Ordering::Relaxed);
        for s in spinners {
            let _ = s.join();
        }
        let a: Vec<f64> = idle.iter().map(|r| r.ms).collect();
        let b: Vec<f64> = loaded.iter().map(|r| r.ms).collect();
        let (p50a, p95a, p50b, p95b) = (pct(&a, 0.5), pct(&a, 0.95), pct(&b, 0.5), pct(&b, 0.95));
        println!(
            "GATE (p95 under load ≤ 2× idle): idle p50 {p50a:.0} / p95 {p95a:.0} ms; loaded p50 {p50b:.0} / p95 {p95b:.0} ms → ratio p50 {:.2}×, p95 {:.2}× → {}",
            p50b / p50a,
            p95b / p95a,
            if p95b <= 2.0 * p95a { "PASS" } else { "FAIL" }
        );
    }
}
