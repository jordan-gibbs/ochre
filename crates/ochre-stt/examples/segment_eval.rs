//! Does live phrase cutting cost accuracy, and how long is the tail at release?
//!
//! ```text
//! cargo run -p ochre-stt --example segment_eval --release -- <wav dir with meta.json> [--vad] [--concat N]
//! ```
//!
//! For each clip (16 kHz WAV; `meta.json` maps file name → `{"ref": "..."}`), the audio is fed
//! through `ochre_audio::PhraseCutter` in 10 ms blocks exactly as capture would, then released at
//! the end of the audio. Every phrase is decoded with Parakeet and joined. Reported per cut
//! config: WER vs the reference for whole-utterance decoding and for joined phrases, the tail
//! audio length after release, and the tail decode time (what release actually waits for).
//! `--concat N` also builds utterances from N consecutive clips back to back (long dictations).

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use ochre_audio::segmenter::{CutConfig, PhraseCutter};
use ochre_core::stt::{SttEngine, SttOptions};

fn norm(s: &str) -> Vec<String> {
    s.to_lowercase()
        .replace("mister", "mr")
        .split_whitespace()
        .map(|w| {
            w.chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn edits(a: &[String], b: &[String]) -> usize {
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
        0.0
    } else {
        s[((s.len() - 1) as f64 * p).round() as usize]
    }
}

const SR: f32 = 16_000.0;
const EPS: f32 = 0.04;

fn slice(pcm: &[f32], a: f32, b: f32) -> &[f32] {
    let i = ((a * SR) as usize).min(pcm.len());
    let j = ((b * SR) as usize).min(pcm.len()).max(i);
    &pcm[i..j]
}

/// (A) Pause cuts (2 s / 180 ms, no forced cuts), each phrase decoded with `ctx` s of left
/// context; words starting inside the context are dropped. Returns (text, tail_s, tail_ms).
fn ctx_pause(engine: &ochre_stt::Parakeet, pcm: &[f32], ctx: f32) -> (String, f64, f64) {
    let cfg = CutConfig {
        max_phrase_s: 30.0,
        cut_window_s: 5.0,
        ..CutConfig::default()
    };
    let mut cutter = PhraseCutter::with_config(16_000, cfg);
    let mut bounds = Vec::new(); // (start_s, end_s)
    let mut pos = 0usize;
    for block in pcm.chunks(160) {
        for p in cutter.push(block) {
            bounds.push((pos as f32 / SR, (pos + p.len()) as f32 / SR));
            pos += p.len();
        }
    }
    let mut tail_bounds = Vec::new();
    if let Some(p) = cutter.release_cut() {
        tail_bounds.push((pos as f32 / SR, (pos + p.len()) as f32 / SR));
        pos += p.len();
    }
    if cutter.pending_has_speech()
        && let Some(p) = cutter.flush()
    {
        tail_bounds.push((pos as f32 / SR, (pos + p.len()) as f32 / SR));
    }
    let mut words = Vec::new();
    let decode = |(a, b): (f32, f32), words: &mut Vec<ochre_stt::parakeet::TimedWord>| {
        let w0 = (a - ctx).max(0.0);
        let ws = engine.transcribe_words(slice(pcm, w0, b)).unwrap();
        for w in ws {
            if (w.start_s + w.end_s) / 2.0 + w0 >= a - EPS || a == 0.0 {
                words.push(w);
            }
        }
    };
    for b in bounds {
        decode(b, &mut words);
    }
    let t = Instant::now();
    let mut tail_s = 0.0;
    for b in tail_bounds {
        tail_s += (b.1 - b.0.max(ctx) + ctx.min(b.0)) as f64;
        decode(b, &mut words);
    }
    (
        ochre_stt::parakeet::words_to_text(&words),
        tail_s,
        t.elapsed().as_secs_f64() * 1000.0,
    )
}

/// (B) Pause-independent streaming: every `step` s of new audio decode [committed - ctx, now]
/// and commit words that end before now - `guard`; at release decode the rest.
fn streaming(
    engine: &ochre_stt::Parakeet,
    pcm: &[f32],
    step: f32,
    ctx: f32,
    guard: f32,
) -> (String, f64, f64) {
    let total = pcm.len() as f32 / SR;
    let mut committed = 0.0f32;
    let mut words = Vec::new();
    let mut now = 0.0f32;
    loop {
        now += step;
        if now >= total {
            break;
        }
        if now - committed < step + guard {
            continue;
        }
        let w0 = (committed - ctx).max(0.0);
        let mut advanced = false;
        for w in engine.transcribe_words(slice(pcm, w0, now)).unwrap() {
            let (s, e) = (w.start_s + w0, w.end_s + w0);
            if (s + e) / 2.0 >= committed - EPS && e <= now - guard {
                committed = e;
                words.push(w);
                advanced = true;
            }
        }
        if !advanced {
            committed = committed.max(now - guard - ctx); // silence: move on, keep some context
        }
    }
    let w0 = (committed - ctx).max(0.0);
    let t = Instant::now();
    for w in engine.transcribe_words(slice(pcm, w0, total)).unwrap() {
        if (w.start_s + w.end_s) / 2.0 + w0 >= committed - EPS {
            words.push(w);
        }
    }
    (
        ochre_stt::parakeet::words_to_text(&words),
        (total - w0) as f64,
        t.elapsed().as_secs_f64() * 1000.0,
    )
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("wav dir"));
    let (mut use_vad, mut concat) = (false, 0usize);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--vad" => use_vad = true,
            "--concat" => concat = args.next().unwrap().parse().unwrap(),
            _ => panic!("unknown arg {a}"),
        }
    }
    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let mut names: Vec<String> = meta.as_object().unwrap().keys().cloned().collect();
    names.sort();
    let mut clips: Vec<(String, Vec<f32>, String)> = names
        .iter()
        .map(|n| {
            let mut r = hound::WavReader::open(dir.join(n)).unwrap();
            let mut pcm: Vec<f32> = r
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect();
            // LibriSpeech clips end on the last word; a dictating user releases a moment after.
            pcm.extend(std::iter::repeat_n(0.0, 4000));
            (n.clone(), pcm, meta[n]["ref"].as_str().unwrap().to_string())
        })
        .collect();
    if concat > 1 {
        let joined: Vec<_> = clips
            .chunks(concat)
            .filter(|c| c.len() == concat)
            .map(|c| {
                (
                    format!("{}+{}", c[0].0, c.len() - 1),
                    c.iter()
                        .flat_map(|x| x.1.iter().copied())
                        .collect::<Vec<f32>>(),
                    c.iter().map(|x| x.2.as_str()).collect::<Vec<_>>().join(" "),
                )
            })
            .collect();
        clips = joined;
    }
    ochre_stt::priority::boost_current_thread();
    let mut engine = ochre_stt::Parakeet::new("", "cpu").unwrap();
    engine.load(&|_| {}).unwrap();
    let opts = SttOptions::default();
    let vad = use_vad.then(|| {
        Arc::new(Mutex::new(
            ochre_audio::SileroVad::load(&|_, _, _| {}, &std::sync::atomic::AtomicBool::new(false))
                .unwrap(),
        ))
    });

    let forced = |max: f32, pause: f32| CutConfig {
        pause_s: pause,
        soft_phrase_s: None,
        max_phrase_s: max,
        cut_window_s: 1.0,
        ..CutConfig::default()
    };
    let configs: Vec<(&str, CutConfig)> = vec![
        ("whole (no cutter)", CutConfig::never()),
        ("long-pause 5s/320ms/30s", CutConfig::long_pause()),
        ("default 2s/200ms soft 8s/100", CutConfig::default()),
        ("2s/180ms pause only", forced(30.0, 0.18)),
        ("2s/300ms pause only", forced(30.0, 0.3)),
        ("2s/180ms forced max 6s", forced(6.0, 0.18)),
        ("2s/180ms forced max 4s", forced(4.0, 0.18)),
        ("2s/180ms forced max 3s", forced(3.0, 0.18)),
    ];
    let total_audio: f64 = clips.iter().map(|c| c.1.len() as f64 / 16_000.0).sum();
    println!(
        "{} utterances, {:.0} s audio, median {:.1} s; vad: {use_vad}",
        clips.len(),
        total_audio,
        {
            let d: Vec<f64> = clips.iter().map(|c| c.1.len() as f64 / 16_000.0).collect();
            pct(&d, 0.5)
        }
    );
    println!(
        "{:<28} {:>7} {:>8} {:>9} {:>9} {:>10} {:>10} {:>10}",
        "config", "WER%", "phrases", "tail_s50", "tail_s95", "tail_max_s", "tailms_50", "tailms_95"
    );
    type Policy<'a> = Box<dyn Fn(&[f32]) -> (String, f64, f64) + 'a>;
    let policies: Vec<(&str, Policy)> = vec![
        (
            "A: pause cuts + 1s ctx",
            Box::new(|p: &[f32]| ctx_pause(&engine, p, 1.0)),
        ),
        (
            "A: pause cuts + 2s ctx",
            Box::new(|p: &[f32]| ctx_pause(&engine, p, 2.0)),
        ),
        (
            "B: stream 2s, ctx 1s, guard .5",
            Box::new(|p: &[f32]| streaming(&engine, p, 2.0, 1.0, 0.5)),
        ),
        (
            "B: stream 2s, ctx 2s, guard .5",
            Box::new(|p: &[f32]| streaming(&engine, p, 2.0, 2.0, 0.5)),
        ),
        (
            "B: stream 1.5s, ctx 1.5s, g .4",
            Box::new(|p: &[f32]| streaming(&engine, p, 1.5, 1.5, 0.4)),
        ),
    ];
    for (label, f) in &policies {
        let (mut errs, mut words) = (0usize, 0usize);
        let (mut tail_s, mut tail_ms) = (Vec::new(), Vec::new());
        for (_, pcm, reference) in &clips {
            let (text, ts, tm) = f(pcm);
            tail_s.push(ts);
            tail_ms.push(tm);
            let r = norm(reference);
            errs += edits(&r, &norm(&text));
            words += r.len();
        }
        println!(
            "{:<28} {:>7.2} {:>8} {:>9.2} {:>9.2} {:>10.2} {:>10.0} {:>10.0}",
            label,
            errs as f64 * 100.0 / words as f64,
            "-",
            pct(&tail_s, 0.5),
            pct(&tail_s, 0.95),
            tail_s.iter().cloned().fold(0.0, f64::max),
            pct(&tail_ms, 0.5),
            pct(&tail_ms, 0.95)
        );
    }
    for (label, cfg) in configs {
        let (mut errs, mut words, mut phrases) = (0usize, 0usize, 0usize);
        let (mut tail_s, mut tail_ms) = (Vec::new(), Vec::new());
        for (_, pcm, reference) in &clips {
            if label.starts_with("whole") {
                let t = Instant::now();
                let text = engine.transcribe(pcm, &opts).unwrap().text;
                tail_ms.push(t.elapsed().as_secs_f64() * 1000.0);
                tail_s.push(pcm.len() as f64 / 16_000.0);
                phrases += 1;
                let r = norm(reference);
                errs += edits(&r, &norm(&text));
                words += r.len();
                continue;
            }
            let mut cutter = PhraseCutter::with_config(16_000, cfg);
            if let Some(v) = &vad {
                cutter = cutter.with_vad(v.clone());
            }
            let mut parts: Vec<Vec<f32>> = Vec::new();
            for block in pcm.chunks(160) {
                parts.extend(cutter.push(block));
            }
            let live = parts.len();
            // Release at the end of the audio: release cut + remainder (when it has speech).
            let mut tail: Vec<Vec<f32>> = cutter.release_cut().into_iter().collect();
            if cutter.pending_has_speech() {
                tail.extend(cutter.flush());
            }
            let mut texts = Vec::new();
            for p in &parts {
                texts.push(engine.transcribe(p, &opts).unwrap().text);
            }
            let t = Instant::now();
            for p in &tail {
                texts.push(engine.transcribe(p, &opts).unwrap().text);
            }
            tail_ms.push(t.elapsed().as_secs_f64() * 1000.0);
            tail_s.push(tail.iter().map(|p| p.len()).sum::<usize>() as f64 / 16_000.0);
            phrases += live + tail.len();
            let hyp = norm(&texts.join(" "));
            let r = norm(reference);
            errs += edits(&r, &hyp);
            words += r.len();
        }
        println!(
            "{:<28} {:>7.2} {:>8} {:>9.2} {:>9.2} {:>10.2} {:>10.0} {:>10.0}",
            label,
            errs as f64 * 100.0 / words as f64,
            phrases,
            pct(&tail_s, 0.5),
            pct(&tail_s, 0.95),
            tail_s.iter().cloned().fold(0.0, f64::max),
            pct(&tail_ms, 0.5),
            pct(&tail_ms, 0.95)
        );
    }
}
