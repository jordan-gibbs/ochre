//! Live connector measurements (network + an OpenAI key; a fraction of a cent per run):
//!
//! ```text
//! cargo test -p ochre-cloud --release live_connector -- --ignored --nocapture
//! ```
//!
//! Keys are read from a dotenv file (`OCHRE_LIVE_ENV`, default the workspace root's `.env`) into
//! memory and passed with `with_key`: never put in the environment or written anywhere. The ~7 s
//! clip is synthesized with OpenAI TTS (24 kHz PCM, resampled to 16 kHz).

use std::time::{Duration, Instant};

use ochre_core::config::SttConfig;
use ochre_core::stt::{SttEngine, SttOptions};

use crate::openai::OpenAiStt;

pub const DEFAULT_ENV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../.env");
const LINE: &str = "Hey Sam, can we move the design review to Thursday at three thirty? I need another day to finish the Kubernetes migration.";

/// `name` from the dotenv file, if present.
pub fn key(name: &str) -> Option<String> {
    let path = std::env::var("OCHRE_LIVE_ENV").unwrap_or_else(|_| DEFAULT_ENV.into());
    std::fs::read_to_string(path).ok()?.lines().find_map(|l| {
        let (k, v) = l.split_once('=')?;
        (k.trim() == name)
            .then(|| v.trim().trim_matches('"').trim_matches('\'').to_string())
            .filter(|v| !v.is_empty())
    })
}

fn tts(key: &str) -> Vec<f32> {
    let bytes = crate::common::client()
        .post("https://api.openai.com/v1/audio/speech")
        .bearer_auth(key)
        .json(&serde_json::json!({"model": "gpt-4o-mini-tts", "voice": "alloy", "input": LINE, "response_format": "pcm"}))
        .timeout(Duration::from_secs(60))
        .send()
        .expect("tts")
        .bytes()
        .expect("tts body");
    let x: Vec<f32> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
        .collect();
    // 24 kHz -> 16 kHz, linear.
    (0..x.len() * 2 / 3)
        .map(|i| {
            let p = i as f64 * 1.5;
            let (j, f) = (p as usize, (p - p.floor()) as f32);
            let a = x[j];
            let b = *x.get(j + 1).unwrap_or(&a);
            a + (b - a) * f
        })
        .collect()
}

fn opts() -> SttOptions {
    SttOptions {
        language: Some("en".into()),
        vocabulary: vec!["Kubernetes".into()],
    }
}

#[test]
#[ignore = "network + OpenAI key"]
fn live_connector_openai_stt_latency() {
    let Some(k) = key("OPENAI_API_KEY") else {
        eprintln!("no OPENAI_API_KEY in the live env file; skipped");
        return;
    };
    let pcm = tts(&k);
    eprintln!("clip {:.2} s", pcm.len() as f64 / 16_000.0);

    let mut batch = OpenAiStt::openai(&SttConfig::default()).with_key(&k);
    batch.load(&|_| {}).unwrap();
    let mut ms = vec![];
    for _ in 0..3 {
        let r = batch.transcribe(&pcm, &opts()).unwrap();
        assert!(r.text.contains("Kubernetes"), "{}", r.text);
        ms.push(r.processing_ms);
    }
    eprintln!("gpt-transcribe batch: {ms:?} ms");

    let live = OpenAiStt::openai(&SttConfig {
        model: "gpt-live-transcribe".into(),
        ..Default::default()
    })
    .with_key(&k);
    let mut ms = vec![];
    for _ in 0..3 {
        let t0 = Instant::now();
        let mut s = live.stream(&opts(), None).unwrap().unwrap();
        let open = t0.elapsed();
        let start = Instant::now();
        for (i, c) in pcm.chunks(1600).enumerate() {
            s.send(c).unwrap();
            let due = start + Duration::from_millis(100 * (i as u64 + 1));
            if let Some(w) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(w);
            }
        }
        let r = s.finish().unwrap();
        assert!(r.text.contains("Kubernetes"), "{}", r.text);
        eprintln!("  open {} ms: {}", open.as_millis(), r.text);
        ms.push(r.processing_ms);
    }
    eprintln!("gpt-live-transcribe release->final (real-time pace): {ms:?} ms");
}
