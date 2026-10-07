//! Live smoke test for the cloud engines (spends a fraction of a cent per run).
//!
//! ```text
//! cargo run -p ochre-cloud --example cloud_smoke --release -- <clip.wav> <engine> [model] [--stream]
//! ```
//!
//! `<clip.wav>` must be 16 kHz mono 16-bit PCM. Runs `load()` (key check + TLS prewarm), then three
//! `transcribe()` calls, printing latency and text. `--stream` (any streaming engine: soniox,
//! openai with gpt-live-transcribe, google with gemini-3.5-transcribe-live) instead feeds the clip
//! through `SttEngine::stream` in 100 ms frames at real-time pace, the way the app would while the
//! user speaks, and reports socket-open and release-to-final latency; `--burst` sends it all at
//! once. `OCHRE_DELAY=minimal|low|...` sets gpt-live-transcribe's `delay`; `--runs N` repeats.

use std::time::{Duration, Instant};

use ochre_core::config::SttConfig;
use ochre_core::stt::{SttEngine, SttOptions};

fn read_wav(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read wav");
    // Find the "data" chunk.
    let mut i = 12;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let len = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        if id == b"data" {
            return bytes[i + 8..(i + 8 + len).min(bytes.len())]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
                .collect();
        }
        i += 8 + len + (len & 1);
    }
    panic!("no data chunk");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (path, id) = (&args[0], args[1].as_str());
    let model = args
        .get(2)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_default();
    let pcm = read_wav(path);
    let opts = SttOptions {
        language: Some("en".into()),
        vocabulary: vec!["Kubernetes".into()],
    };
    let cfg = SttConfig {
        engine: id.into(),
        model,
        ..Default::default()
    };
    println!("clip: {:.2} s", pcm.len() as f64 / 16_000.0);

    let runs: usize = args
        .iter()
        .position(|a| a == "--runs")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse().ok())
        .unwrap_or(3);
    if args.iter().any(|a| a == "--stream" || a == "--burst") {
        let burst = args.iter().any(|a| a == "--burst");
        let eng: Box<dyn SttEngine> = if id == "openai" {
            let mut o = ochre_cloud::openai::OpenAiStt::openai(&cfg);
            if let Ok(d) = std::env::var("OCHRE_DELAY") {
                o.delay = d;
            }
            Box::new(o)
        } else {
            ochre_cloud::create(id, &cfg).expect("create")
        };
        for run in 0..runs {
            let t0 = Instant::now();
            let partials = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let p2 = partials.clone();
            let mut s = eng
                .stream(
                    &opts,
                    Some(Box::new(move |_t: &str, _s: usize| {
                        p2.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    })),
                )
                .expect("not a streaming engine/model")
                .expect("open");
            let open_ms = t0.elapsed().as_millis();
            let start = Instant::now();
            for (k, chunk) in pcm.chunks(1600).enumerate() {
                s.send(chunk).expect("send");
                if !burst {
                    let due = start + Duration::from_millis(100 * (k as u64 + 1));
                    if let Some(w) = due.checked_duration_since(Instant::now()) {
                        std::thread::sleep(w);
                    }
                }
            }
            let r = s.finish().expect("finish");
            println!(
                "{id} stream run {run} ({}): open {open_ms} ms, release->final {} ms, {} partials
  {}",
                if burst { "burst" } else { "real-time pace" },
                r.processing_ms,
                partials.load(std::sync::atomic::Ordering::Relaxed),
                r.text
            );
        }
        return;
    }

    let mut eng = ochre_cloud::create(id, &cfg).expect("create");
    let t0 = Instant::now();
    eng.load(&|_| {}).expect("load");
    println!(
        "{} / {}: load (key + TLS prewarm) {} ms",
        id,
        eng.info().default_model,
        t0.elapsed().as_millis()
    );
    for i in 0..runs {
        let t = Instant::now();
        match eng.transcribe(&pcm, &opts) {
            Ok(r) => println!(
                "  run {i}: {} ms (engine {} ms)  {:?}  [{}]",
                t.elapsed().as_millis(),
                r.processing_ms,
                r.text,
                r.language.unwrap_or_default()
            ),
            Err(e) => println!("  run {i}: error {} ({})", e, e.code()),
        }
    }
}
