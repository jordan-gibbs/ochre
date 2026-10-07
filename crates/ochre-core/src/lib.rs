//! Shared contracts for Ochre: config, the core <-> UI protocol, and the traits every
//! engine implements (speech-to-text, refinement, hotkeys, injection). No heavy dependencies live
//! here, so every other crate can depend on it cheaply.

pub mod config;
pub mod error;
pub mod events;
pub mod history;
#[cfg(target_os = "linux")]
pub mod linux_child;
#[cfg(target_os = "linux")]
pub mod linux_priority;
pub mod paths;
pub mod platform;
pub mod refine;
pub mod secrets;
pub mod stt;

pub use error::{Error, Result};

/// Every engine works on mono f32 PCM at this rate; capture resamples once, up front.
pub const SAMPLE_RATE: u32 = 16_000;
