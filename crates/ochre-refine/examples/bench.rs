//! Refinement latency + output benchmark (SPEC §5.2 target: < 1 s for a ~100-word paragraph).
//!
//! ```text
//! cargo run -p ochre-refine --example bench --release                      # quill-0.8b, auto accel
//! cargo run -p ochre-refine --example bench --release -- --accel cpu --threads 16
//! cargo run -p ochre-refine --example bench --release -- --model quill-2b --reps 5
//! cargo run -p ochre-refine --example bench --release -- --cloud openai   # a cloud refiner (needs its key)
//! cargo run -p ochre-refine --example bench --release -- --cloud openai --samples raw.tsv  # name<TAB>text lines
//! ```
//!
//! Local runs go through the real `LocalRefiner` (same binary/model resolution, flags, warm-up and
//! priming as the app). Samples are run round-robin so consecutive requests always differ, as in
//! real use; between requests the bench waits for the background prime, which in real use happens
//! during the seconds between dictations. Reported per sample: median wall time and llama.cpp's own
//! split (prompt tokens evaluated / reused from cache, prompt ms, generated tokens, generation ms).

use std::time::{Duration, Instant};

use ochre_core::config::RefineConfig;
use ochre_core::refine::{RefineContext, Refiner};
use ochre_core::stt::Progress;
use ochre_refine::local::LocalRefiner;

/// Realistic raw ASR output: lower-case, no punctuation, fillers, repeats, self-corrections.
const SAMPLES: &[(&str, &str)] = &[
    (
        "short",
        "um can you send me the the file when you get a chance",
    ),
    (
        "paragraph",
        "so um i wanted to give everyone a quick update on the project uh we finished the first round of user \
interviews last week and the main thing we heard was that people find the onboarding way too long like they drop off \
before they even see the dashboard so uh what we're thinking is we cut it down to three steps instead of seven and we \
move the integrations stuff to later you know after they've actually seen some value um i'll share the full notes in \
the doc by thursday and if anyone has concerns just let me know before then",
    ),
    (
        "self_correction",
        "let's meet on monday no wait make that tuesday at three thirty in the small conference room",
    ),
    (
        "list",
        "okay the grocery list is first eggs second uh milk third bread and fourth some some coffee beans",
    ),
    (
        "question",
        "what time does the pharmacy on main street close today",
    ),
    (
        "email",
        "my email is john dot smith at gmail dot com and the site is example dot com slash pricing",
    ),
];

struct Args {
    model: String,
    accel: String,
    threads: u32,
    reps: usize,
    cloud: Option<String>,
    cloud_model: String,
    samples: Option<String>,
}

fn args() -> Args {
    let mut a = Args {
        model: "quill-0.8b".into(),
        accel: "auto".into(),
        threads: 0,
        reps: 3,
        cloud: None,
        cloud_model: String::new(),
        samples: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut v = || it.next().unwrap_or_default();
        match k.as_str() {
            "--model" => a.model = v(),
            "--accel" => a.accel = v(),
            "--threads" => a.threads = v().parse().unwrap_or(0),
            "--reps" => a.reps = v().parse().unwrap_or(3).max(1),
            "--cloud" => a.cloud = Some(v()),
            "--cloud-model" => a.cloud_model = v(),
            "--samples" => a.samples = Some(v()),
            other => {
                eprintln!("unknown arg {other}; see the doc comment in examples/bench.rs");
                std::process::exit(2);
            }
        }
    }
    a
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn progress(p: Progress) {
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);
    if p.total == 0 {
        return;
    }
    let decile = p.done * 10 / p.total;
    if LAST.swap(decile, std::sync::atomic::Ordering::Relaxed) != decile {
        eprintln!(
            "  downloading {}: {} / {} MB",
            p.item,
            p.done >> 20,
            p.total >> 20
        );
    }
}

fn main() {
    let a = args();
    let ctx = RefineContext {
        mode: "clean".into(),
        ..Default::default()
    };
    if let Some(provider) = &a.cloud {
        let owned: Vec<(String, String)> = match &a.samples {
            Some(path) => std::fs::read_to_string(path)
                .expect("read --samples file")
                .lines()
                .filter_map(|l| l.split_once('\t'))
                .map(|(n, t)| (n.trim().to_string(), t.trim().to_string()))
                .collect(),
            None => SAMPLES
                .iter()
                .map(|(n, t)| (n.to_string(), t.to_string()))
                .collect(),
        };
        return cloud(provider, &a.cloud_model, &ctx, a.reps, &owned);
    }
    let cfg = RefineConfig {
        provider: "local".into(),
        model: a.model.clone(),
        local_accel: a.accel.clone(),
        local_threads: a.threads,
        ..Default::default()
    };
    let mut r = LocalRefiner::new(&cfg).expect("config");
    let t0 = Instant::now();
    r.load(&progress).expect("load");
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let s = r.server().expect("server");
    println!(
        "## {} / {} / threads={} / ngram-spec={}  (load incl. spawn, /health, warm-up, prime: {load_ms:.0} ms)\n",
        a.model, s.accel, s.options.threads, s.options.ngram_spec
    );
    let mut runs: Vec<Vec<ochre_refine::local::LocalOutput>> =
        SAMPLES.iter().map(|_| vec![]).collect();
    for rep in 0..=a.reps {
        for (i, (_, text)) in SAMPLES.iter().enumerate() {
            let out = r
                .refine_detailed(text, &ctx, Duration::from_secs(60))
                .expect("refine");
            // Wait for the background prime (in real use it runs between dictations).
            std::thread::sleep(Duration::from_millis(if s.accel == "cpu" {
                600
            } else {
                250
            }));
            if rep > 0 {
                runs[i].push(out); // rep 0 is a warm-up pass
            }
        }
    }
    println!(
        "| sample | words | wall ms | prompt tok (cached) | prompt ms | gen tok | gen ms | gen tok/s | http+other ms | guard |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|");
    let mut outputs = vec![];
    for ((name, text), rs) in SAMPLES.iter().zip(&runs) {
        let wall = median(rs.iter().map(|o| o.wall_ms).collect());
        let o = rs
            .iter()
            .min_by(|x, y| {
                (x.wall_ms - wall)
                    .abs()
                    .partial_cmp(&(y.wall_ms - wall).abs())
                    .unwrap()
            })
            .unwrap();
        let t = &o.timings;
        let tps = if t.predicted_ms > 0.0 {
            t.predicted_n as f64 / t.predicted_ms * 1000.0
        } else {
            0.0
        };
        let guard = ochre_refine::guard::check(text, &o.text, "clean")
            .map(|r| r.as_str())
            .unwrap_or("ok");
        println!(
            "| {name} | {} | {wall:.0} | {} ({}) | {:.0} | {} | {:.0} | {tps:.0} | {:.0} | {guard} |",
            text.split_whitespace().count(),
            t.prompt_n,
            t.cache_n,
            t.prompt_ms,
            t.predicted_n,
            t.predicted_ms,
            o.wall_ms - t.prompt_ms - t.predicted_ms
        );
        outputs.push((name, o.text.clone()));
    }
    let para = median(runs[1].iter().map(|o| o.wall_ms).collect());
    println!(
        "\nparagraph (~100 words): {para:.0} ms -> {} (< 1000 ms)\n",
        if para < 1000.0 { "PASS" } else { "FAIL" }
    );
    for (name, text) in outputs {
        println!("[{name}] {text}");
    }
}

fn cloud(
    provider: &str,
    model: &str,
    ctx: &RefineContext,
    reps: usize,
    samples: &[(String, String)],
) {
    let cfg = RefineConfig {
        provider: provider.into(),
        model: model.into(),
        ..Default::default()
    };
    let mut r = ochre_refine::create(&cfg).expect("create");
    let t0 = Instant::now();
    r.load(&progress).expect("load (key missing or rejected?)");
    println!(
        "## cloud {provider} / {}  (load = key check + TLS prewarm: {:.0} ms)\n",
        r.info().default_model,
        t0.elapsed().as_secs_f64() * 1000.0
    );
    println!("| sample | ms (each run) | guard |\n|---|---|---|");
    let mut outputs = vec![];
    for (name, text) in samples {
        let mut times = vec![];
        let mut last = None;
        for _ in 0..reps {
            let o = ochre_refine::refine_text(Some(r.as_ref()), text, ctx, Duration::from_secs(10));
            times.push(o.ms.to_string());
            last = Some(o);
        }
        let o = last.unwrap();
        println!(
            "| {name} | {} | {} |",
            times.join(" / "),
            o.reason.as_deref().unwrap_or("ok")
        );
        outputs.push((name, o.text));
    }
    println!();
    for (name, text) in outputs {
        println!("[{name}] {text}");
    }
}
