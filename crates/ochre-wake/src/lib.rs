//! Hands-free mode for Ochre (SPEC §6.3, runtime contract in `docs/wakeword.md`).
//!
//! Three independent pieces, none of which records, transcribes or types:
//!
//! 1. **[`WakeDetector`]**: a streaming wake-word detector on `ort`. Per 80 ms (1280 samples at
//!    16 kHz) it runs openWakeWord's mel model (1760 samples with 480 of left context -> 8 mel rows),
//!    the embedding model (76x32 mel window -> 96-d embedding) and an ONNX head over the last 16
//!    embeddings, then a gate (threshold, 2 consecutive frames, 1.5 s cooldown). It matches the
//!    Python reference (`src/openwhisprflow/wake/`) to ~1e-6. The mel and embedding models download
//!    once into `<models_dir>/openwakeword` (sha256-checked). An optional speech predicate gates
//!    inference, so a silent room costs only the VAD.
//! 2. **[`commands`]**: fuzzy detection and stripping of a trailing control phrase ("transcribe
//!    stop|done|send|cancel|scratch that"), robust to ASR spellings, never mid-sentence.
//! 3. **[`HandsFree`]**: the session controller. It takes mic blocks, keeps the pre-roll, starts a
//!    session on wake, decides per transcribed phrase, and handles idle timeout, the session cap and
//!    pausing during calls ([`calls`]; Windows consent store, a no-op elsewhere for now).
//!
//! # API for the orchestrator
//!
//! ```no_run
//! use ochre_core::config::HandsFreeConfig;
//! use ochre_wake::{HandsFree, HfEvent, PhraseAction, WakeDetector, commands};
//!
//! # fn silero(_: &[f32]) -> bool { true }
//! # fn demo(cfg: HandsFreeConfig, mic_block: &[f32]) -> ochre_core::Result<()> {
//! // Detector alone: [-1, 1] f32 blocks of any size in, at most one hit per call out.
//! let mut det = WakeDetector::new("assets/wake/transcribe.onnx", 0.55)?
//!     .with_vad(|frame: &[f32]| silero(frame)); // one 1280-sample [-1, 1] frame -> voiced?
//! if let Some(hit) = det.process(mic_block) {
//!     println!("wake at frame {} (score {:.2})", hit.frame, hit.score);
//! }
//!
//! // Controller: owns a detector, emits events, decides per phrase.
//! let mut hf = HandsFree::new(&cfg, Box::new(det))
//!     .with_max_session_s(600.0)
//!     .on_wake(|w| println!("earcon; seed STT with {} pre-roll samples", w.preroll.len()));
//! for ev in hf.feed(mic_block) {
//!     if let HfEvent::End(end) = ev { /* idle timeout / cap: insert end.text if end.insert() */ }
//! }
//! match hf.on_phrase("Transcribe, hey Sarah. Transcribe send.") {
//!     PhraseAction::Send(end) => { /* insert end.text, press Enter */ }
//!     PhraseAction::Finish(end) => { /* insert end.text */ }
//!     PhraseAction::Cancel(_) | PhraseAction::FalseWake(_) => { /* discard */ }
//!     PhraseAction::Scratch { .. } | PhraseAction::Continue { .. } | PhraseAction::Ignored => {}
//! }
//!
//! // Control phrases on their own:
//! let (text, ctl) = commands::parse_trailing("See you at five. Transcribe stop.", "transcribe");
//! assert_eq!(text, "See you at five.");
//! assert_eq!(ctl, Some(commands::Control::Finish));
//! # Ok(()) }
//! ```
//!
//! * [`WakeDetector::new(model_path, threshold)`](WakeDetector::new),
//!   [`with_options`](WakeDetector::with_options) for the tunables, [`process`](WakeDetector::process)
//!   (also `process_i16`, `process_frame`), `reset`, `hold`, and `stats` / `last_score` for a debug UI.
//!   [`recommended_threshold`] reads `recommended.threshold` from the model's `.json`.
//! * [`HandsFree::new(cfg, detector)`](HandsFree::new) (any [`WakeSource`]) or
//!   [`HandsFree::from_config`]; [`feed`](HandsFree::feed) / [`tick`](HandsFree::tick) return
//!   [`HfEvent`]s (also delivered to the `on_wake` / `on_end` / `on_pause` / `on_state` callbacks);
//!   [`on_phrase`](HandsFree::on_phrase) returns a [`PhraseAction`]; plus `finish(send)`, `cancel()`
//!   and `set_enabled`. Every session end, whatever the cause, is emitted once as [`HfEvent::End`].
//! * [`commands::parse_trailing(text, phrase)`](commands::parse_trailing) ->
//!   `(String, Option<Control>)`, [`commands::parse_control`] for aliases, the armed mode and the
//!   matched span, plus `starts_with_phrase` / `strip_leading_phrase` / `ends_with_phrase`.
//!
//! Audio convention: 16 kHz mono f32 in [-1, 1] everywhere in the public API. The front end
//! internally works at int16 scale (x32767), as openWakeWord was trained.
//!
//! Threading: one detector/controller per audio thread; nothing here locks. ORT sessions are
//! single-threaded with spinning disabled, so idle listening never busy-waits.

pub mod calls;
pub mod commands;
pub mod detector;
pub mod frontend;
pub mod handsfree;
pub mod models;
mod rng;

pub use calls::{MicMonitor, MicUsageMonitor};
pub use commands::{Control, parse_trailing};
pub use detector::{
    DetectorOptions, DetectorStats, FrameScorer, OnnxScorer, SpeechFn, WakeDetector, WakeGate,
    WakeHit, load_meta, recommended_threshold, resolve_model,
};
pub use handsfree::{
    EndReason, HandsFree, HfEvent, HfState, PhraseAction, SessionEnd, WakeSource, WakeStart,
};

pub const SAMPLE_RATE: usize = 16_000;
/// 80 ms at 16 kHz: one embedding per frame.
pub const FRAME: usize = 1280;
pub const FRAME_S: f32 = 0.08;
/// Extra left context for the mel model (3 hops of 10 ms).
pub const MEL_CONTEXT: usize = 480;
/// Mel rows per embedding window.
pub const MEL_WINDOW: usize = 76;
pub const MEL_BINS: usize = 32;
pub const EMBED_DIM: usize = 96;
