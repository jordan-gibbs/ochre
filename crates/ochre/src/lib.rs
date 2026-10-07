//! The Ochre orchestrator: one dictation session at a time, every stage wired
//! together (SPEC §3). The GUI binary (app/src-tauri) and the headless `ochre-cli` binary both embed it.
//!
//! Flow: trigger (hotkey gesture / wake word / UI / CLI) -> warm mic with pre-roll -> segmenter
//! (phrases decode while the user talks) -> corrections -> optional refinement (guarded,
//! timeout -> raw) -> history (before insert) -> injection. Every state change goes out on the Bus.

pub mod app;
pub mod connectors;
pub mod handsfree;
pub mod live;
pub mod seams;
pub mod stream;
pub mod text;
pub mod wiring;

pub use app::{App, Engines, Loader};
