//! Render refinement examples through the production local prompt path, for the fine-tuning
//! byte-parity check (`training/refine/render.py --check-rust`).
//!
//! ```text
//! CARGO_TARGET_DIR=target/finetune cargo run -p ochre-refine --example render < in.jsonl > out.jsonl
//! ```
//!
//! Input: one JSON object per line with `raw`, `mode`, `style`, `dictionary` (as in the training
//! JSONL). Output: one line per input, `{"prompt": build_prompt(raw, ctx)}`.

use std::io::{BufRead, Write};

use ochre_core::refine::RefineContext;
use ochre_refine::local::build_prompt;

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line.expect("read stdin");
        if line.trim().is_empty() {
            continue;
        }
        let ex: serde_json::Value = serde_json::from_str(&line).expect("valid JSON line");
        let s = |k: &str| ex.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        let ctx = RefineContext {
            mode: if s("mode").is_empty() {
                "clean".into()
            } else {
                s("mode")
            },
            style: s("style"),
            dictionary: ex
                .get("dictionary")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|w| w.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default(),
            ..Default::default()
        };
        let prompt = build_prompt(&s("raw"), &ctx);
        writeln!(out, "{}", serde_json::json!({ "prompt": prompt })).expect("write stdout");
    }
}
