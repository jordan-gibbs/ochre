//! Numerical parity with the Python reference (`src/openwhisprflow/wake/`). These need the
//! openWakeWord front-end models (downloaded on first run), so they are `#[ignore]`:
//!
//! ```text
//! CARGO_TARGET_DIR=target/ochre-wake cargo test -p ochre-wake --test parity -- --ignored --nocapture
//! ```
//!
//! Fixtures in `tests/fixtures/` were made by `tests/fixtures/make_reference.py` (run with the
//! repo's `.venv`): per-frame Python values for TTS clips streamed from a fresh `reset()` in
//! 1280-sample frames without a VAD. The reference head is not shipped with this repo; set
//! `OCHRE_TEST_WAKE_MODEL` or keep `../wake-head/wake_head.onnx` next to it.

#[path = "support/wav.rs"]
mod wav;

use std::path::{Path, PathBuf};

use ochre_wake::frontend::{WakeFrontend, WakeHead};
use ochre_wake::models::{ensure_frontend_models, frontend_dir};
use ochre_wake::{DetectorOptions, FRAME, MEL_BINS, WakeDetector, WakeGate};
use serde_json::Value;

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_dir() -> PathBuf {
    crate_dir().join("../..")
}

fn fixtures() -> PathBuf {
    crate_dir().join("tests/fixtures")
}

fn frontend() -> WakeFrontend {
    let (mel, emb) = ensure_frontend_models(&frontend_dir(), None).expect("front-end models");
    WakeFrontend::new(&mel, &emb).expect("front end")
}

fn reference_head() -> Option<PathBuf> {
    let p = std::env::var_os("OCHRE_TEST_WAKE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| repo_dir().join("../wake-head/wake_head.onnx"));
    p.exists().then_some(p)
}

fn floats(v: &Value) -> Vec<f32> {
    match v {
        Value::Array(a) => a.iter().flat_map(floats).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap() as f32],
        _ => panic!("not numeric: {v}"),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "downloads the openWakeWord front-end models (~2.4 MB)"]
fn download_frontend_models() {
    let (mel, emb) = ensure_frontend_models(
        &frontend_dir(),
        Some(&|n, d, t| {
            if d == t {
                eprintln!("{n}: {d} bytes");
            }
        }),
    )
    .unwrap();
    assert_eq!(std::fs::metadata(mel).unwrap().len(), 1_087_958);
    assert_eq!(std::fs::metadata(emb).unwrap().len(), 1_326_578);
}

/// Our streaming windows equal openWakeWord's own for the deterministic chirp, frames 24..30.
#[test]
#[ignore = "needs the front-end models"]
fn frontend_matches_openwakeword_chirp() {
    let (_, chirp) = wav::read_wav(&fixtures().join("chirp.wav")).unwrap();
    let want: Value = serde_json::from_str(
        &std::fs::read_to_string(fixtures().join("chirp_windows.json")).unwrap(),
    )
    .unwrap();
    let want = floats(&want["windows"]); // 6 x 16 x 96, frames 24..30
    let mut fe = frontend();
    let mut got = Vec::new();
    for (k, f) in chirp.chunks_exact(FRAME).enumerate() {
        let x: Vec<f32> = f.iter().map(|&v| v as f32).collect();
        assert_eq!(fe.push(&x).unwrap(), 1);
        if (24..30).contains(&k) {
            got.extend_from_slice(fe.window(16));
        }
    }
    let d = max_abs_diff(&got, &want);
    eprintln!("chirp windows vs openWakeWord: max |diff| = {d:.3e}");
    assert!(d < 1e-3);
}

/// Per-frame mel rows, embeddings, scores and gate hits vs the Python reference for one fixture.
fn check_fixture(json: &Path, model: &Path) {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(json).unwrap()).unwrap();
    let wav_path = json.with_extension("wav");
    let (rate, x) = wav::read_wav(&wav_path).unwrap();
    assert_eq!(rate, 16_000);
    let threshold = r["threshold"].as_f64().unwrap() as f32;
    let mut fe = frontend();
    let mut head = WakeHead::new(model).unwrap();
    let mut gate = WakeGate::new(threshold, 2, 19);
    let emb_ref = r
        .get("embedding_first_frames")
        .or_else(|| r.get("embedding"))
        .map(|v| v.as_array().unwrap().clone());
    let (mut scores, mut hits, mut emb_d, mut mel_d) = (Vec::new(), Vec::new(), 0f32, 0f32);
    for (k, f) in x.chunks_exact(FRAME).enumerate() {
        let xf: Vec<f32> = f.iter().map(|&v| v as f32).collect();
        fe.push(&xf).unwrap();
        if k == 0 {
            let mb = fe.mel_buffer();
            mel_d = max_abs_diff(
                &mb[mb.len() - 8 * MEL_BINS..],
                &floats(&r["mel_rows_frame0"]),
            );
        }
        if let Some(e) = emb_ref.as_ref().and_then(|e| e.get(k)) {
            let w = fe.window(1);
            emb_d = emb_d.max(max_abs_diff(w, &floats(e)));
        }
        let s = head.score(&fe).unwrap();
        scores.push(s);
        if gate.update(s, k as i64) {
            hits.push(k as i64);
        }
    }
    let score_d = max_abs_diff(&scores, &floats(&r["scores"]));
    let want_hits: Vec<i64> = r["hits"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["frame"].as_i64().unwrap())
        .collect();
    let peak = scores.iter().copied().fold(0.0, f32::max);
    eprintln!(
        "{}: {} frames, max |diff| mel {mel_d:.2e}, embedding {emb_d:.2e}, score {score_d:.2e}; peak {peak:.4}; hits {hits:?} (python {want_hits:?})",
        json.file_name().unwrap().to_string_lossy(),
        scores.len()
    );
    assert!(mel_d < 1e-4 && emb_d < 1e-4 && score_d < 1e-3);
    assert_eq!(hits, want_hits);

    // and through the full detector (VAD-less), as the app streams it: Python score_clip hits
    if let Some(det_hits) = r.get("detector_hits") {
        let mut det = WakeDetector::with_options(
            model,
            DetectorOptions {
                threshold,
                ..Default::default()
            },
        )
        .unwrap();
        let pad = vec![0i16; 16_000];
        let stream: Vec<i16> = pad.iter().chain(&x).chain(&pad).copied().collect();
        let mut got = Vec::new();
        for b in stream.chunks(FRAME) {
            if let Some(h) = det.process_i16(b) {
                got.push((h.frame, h.rise_frame));
            }
        }
        let want: Vec<(u64, u64)> = det_hits
            .as_array()
            .unwrap()
            .iter()
            .map(|h| {
                (
                    h["frame"].as_u64().unwrap(),
                    h["rise_frame"].as_u64().unwrap(),
                )
            })
            .collect();
        eprintln!("  detector hits {got:?} (python {want:?})");
        assert_eq!(got, want);
    }
}

#[test]
#[ignore = "needs the front-end models and the reference head"]
fn reference_clips_match_python() {
    let Some(model) = reference_head() else {
        eprintln!("skipped: no reference head (set OCHRE_TEST_WAKE_MODEL)");
        return;
    };
    for name in ["positive_tts.json", "negative_tts.json"] {
        check_fixture(&fixtures().join(name), &model);
    }
}

/// The shipped "transcribe" model against the training engineer's fixture (docs/wakeword.md §10).
#[test]
#[ignore = "needs the front-end models and assets/wake/transcribe.onnx"]
fn transcribe_sample_matches_python() {
    let model = repo_dir().join("assets/wake/transcribe.onnx");
    let json = repo_dir().join("tests/fixtures/wake/transcribe_sample.json");
    if !model.exists() || !json.exists() {
        eprintln!(
            "skipped: {} or {} not present yet",
            model.display(),
            json.display()
        );
        return;
    }
    check_fixture(&json, &model);
}

/// End to end: the real detector inside the controller wakes on the shipped sample ("Transcribe,
/// hello there. Transcribe send."), then the transcript of that audio ends the session with Send.
#[test]
#[ignore = "needs the front-end models and assets/wake/transcribe.onnx"]
fn handsfree_end_to_end_on_sample() {
    use ochre_core::config::HandsFreeConfig;
    use ochre_wake::{HandsFree, HfEvent, PhraseAction};

    let model = repo_dir().join("assets/wake/transcribe.onnx");
    let wav_path = repo_dir().join("tests/fixtures/wake/transcribe_sample.wav");
    if !model.exists() || !wav_path.exists() {
        eprintln!("skipped: shipped model or sample not present");
        return;
    }
    let threshold = ochre_wake::recommended_threshold(&model).unwrap();
    let det = WakeDetector::new(&model, threshold)
        .unwrap()
        .with_vad(|f: &[f32]| {
            (f.iter().map(|v| v * v).sum::<f32>() / f.len() as f32).sqrt() > 0.005
        });
    let cfg = HandsFreeConfig {
        enabled: true,
        threshold,
        ..Default::default()
    };
    let mut hf = HandsFree::new(&cfg, Box::new(det));
    let (_, x) = wav::read_wav(&wav_path).unwrap();
    let audio: Vec<f32> = std::iter::repeat_n(0.0, 16_000)
        .chain(x.iter().map(|&v| v as f32 / 32767.0))
        .chain(std::iter::repeat_n(0.0, 16_000))
        .collect();
    let mut wakes = Vec::new();
    for block in audio.chunks(512) {
        for ev in hf.feed(block) {
            if let HfEvent::Wake(w) = ev {
                wakes.push(w);
            }
        }
    }
    assert_eq!(
        wakes.len(),
        1,
        "one session (the second 'transcribe' only arms commands)"
    );
    let w = &wakes[0];
    eprintln!(
        "wake: score {:.3}, pre-roll {:.2} s, clean {}",
        w.score,
        w.preroll.len() as f64 / 16_000.0,
        w.clean_start
    );
    assert!(w.preroll.len() >= 16_000 / 2);
    match hf.on_phrase("Transcribe, hello there. Transcribe send.") {
        PhraseAction::Send(end) => {
            assert_eq!(end.text, "Hello there.");
            assert!(end.press_enter());
        }
        other => panic!("expected Send, got {other:?}"),
    }
}
