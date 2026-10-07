//! Parity harness for tools/refine-data/asr.py `phrases_like_app` (the Python port of
//! `PhraseCutter` used to build training data). For each raw little-endian f32 16 kHz mono file
//! given on the command line, feeds it through `PhraseCutter::new` (CutConfig::default, no VAD) in
//! 20 ms blocks, then `release_cut`, then the `finish` flush, and prints one JSON line with the
//! phrase lengths in samples.
//!
//! `cargo run -p ochre-audio --example cutter_parity -- a.f32 b.f32`

use ochre_audio::PhraseCutter;

fn main() {
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).expect("read input");
        let pcm: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut c = PhraseCutter::new(16_000);
        let mut lens: Vec<usize> = Vec::new();
        for block in pcm.chunks(320) {
            lens.extend(c.push(block).iter().map(Vec::len));
        }
        if let Some(p) = c.release_cut() {
            lens.push(p.len());
        }
        if c.pending_has_speech() {
            if let Some(rest) = c.flush() {
                lens.push(rest.len());
            }
        }
        let list = lens
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!(
            "{{\"file\":\"{}\",\"phrases\":[{list}]}}",
            path.replace('\\', "/")
        );
    }
}
