//! Parity harness for `src/openwhisprflow/refine/chunk.py` (the eval harness's port of
//! `chunk.rs`). Reads JSON lines `{"text": "...", "min": 80, "target": 40}` on stdin and writes
//! `{"pieces": [[sep, text], ...]}` per line.
//!
//! ```text
//! cargo run -p ochre-refine --example chunk_parity < cases.jsonl     # tools/eval/chunk_parity.py
//! ```

use std::io::{BufRead, Write};

use ochre_refine::chunk;

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).expect("valid JSON line");
        let text = v["text"].as_str().unwrap_or_default();
        let n = |k: &str, d: usize| v[k].as_u64().map_or(d, |x| x as usize);
        let pieces: Vec<(&str, String)> = chunk::split(
            text,
            n("min", chunk::MIN_WORDS),
            n("target", chunk::TARGET_WORDS),
        )
        .into_iter()
        .map(|p| (p.sep.as_str(), p.text))
        .collect();
        writeln!(out, "{}", serde_json::json!({ "pieces": pieces })).expect("write stdout");
    }
}
