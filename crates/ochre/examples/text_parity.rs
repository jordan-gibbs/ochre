//! Parity harness for tools/refine-data/owf_text.py (the Python port of `text.rs` that builds
//! training `raw`). Reads JSON lines `{"text": "...", "words": [...]}` on stdin and writes
//! `{"strip": ..., "collapse": ..., "full": ...}` per line, where `full` is the app's exact
//! pre-pass (`text::prepass`): `apply_corrections(collapse_repeats(strip_hesitations(text)), dictionary)`.
//!
//! `cargo run -p ochre --example text_parity < cases.jsonl`

use std::io::{BufRead, Write};

use ochre::text::{collapse_repeats, prepass, strip_hesitations};
use ochre_core::config::DictionaryConfig;

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line.expect("stdin");
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(&line).expect("json line");
        let text = v["text"].as_str().unwrap_or_default();
        let words = v["words"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|w| w.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        let dict = DictionaryConfig {
            words,
            ..Default::default()
        };
        let strip = strip_hesitations(text);
        let collapse = collapse_repeats(text);
        let full = prepass(text, &dict);
        let rec = serde_json::json!({"strip": strip, "collapse": collapse, "full": full});
        writeln!(out, "{rec}").expect("stdout");
    }
}
