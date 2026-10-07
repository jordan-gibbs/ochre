//! Local speech-to-text engines (SPEC §4.2). Every engine implements
//! `ochre_core::stt::SttEngine`: `load()` downloads through `ochre-models`, opens the model and runs a
//! warm-up inference; `transcribe()` takes mono f32 PCM at 16 kHz.
//!
//! Adding an engine: put the model manifest (pinned URLs + SHA-256) and the inference in a new
//! module, reuse `onnx::session` (device selection with CPU fallback) and `mel` if it is an
//! ONNX model with NeMo-style features, then add it to `local_engines()` and `create()`.

pub mod chunk;
pub mod mel;
pub mod onnx;
pub mod parakeet;
pub mod priority;
#[cfg(feature = "whisper")]
pub mod whisper;

use ochre_core::events::EngineInfo;
use ochre_core::stt::SttEngine;
use ochre_core::{Error, Result};

pub use onnx::Device;
pub use parakeet::Parakeet;

/// Every local engine id Ochre knows about, compiled into this build or not (Whisper sits behind
/// the `whisper` cargo feature). Lets the registry tell "not in this build" apart from "unknown",
/// so a local id is never handed to the cloud factory.
pub const KNOWN_LOCAL: &[&str] = &["parakeet", "whisper"];

/// The display name of a known local engine, for messages.
pub fn known_local_label(id: &str) -> Option<&'static str> {
    match id {
        "parakeet" => Some("Parakeet"),
        "whisper" => Some("Whisper"),
        _ => None,
    }
}

/// Metadata for every local engine compiled into this build.
pub fn local_engines() -> Vec<EngineInfo> {
    #[allow(unused_mut)]
    let mut v = vec![parakeet::info()];
    #[cfg(feature = "whisper")]
    v.push(whisper::info());
    v
}

/// Construct (but do not load) a local engine. `model` "" = the engine's default; `device` is an
/// `SttConfig::device` string. Returns `Error::Config` for an unknown id, so the caller's
/// registry can try cloud providers next.
pub fn create(engine: &str, model: &str, device: &str) -> Result<Box<dyn SttEngine>> {
    match engine {
        parakeet::ENGINE_ID => Ok(Box::new(Parakeet::new(model, device)?)),
        #[cfg(feature = "whisper")]
        whisper::ENGINE_ID => Ok(Box::new(whisper::Whisper::new(model, device)?)),
        other => Err(Error::Config(format!("unknown local STT engine {other:?}"))),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn registry() {
        assert!(super::local_engines().iter().any(|e| e.id == "parakeet"));
        assert!(super::create("parakeet", "", "cpu").is_ok());
        assert!(super::create("nope", "", "cpu").is_err());
    }

    #[test]
    fn compiled_engines_are_known() {
        for e in super::local_engines() {
            assert!(super::KNOWN_LOCAL.contains(&e.id.as_str()), "{}", e.id);
            assert!(super::known_local_label(&e.id).is_some(), "{}", e.id);
        }
        assert_eq!(
            super::local_engines().iter().any(|e| e.id == "whisper"),
            cfg!(feature = "whisper")
        );
    }
}
