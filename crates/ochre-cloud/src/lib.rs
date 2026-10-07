//! BYOK cloud speech-to-text engines (SPEC §4.3): Soniox, Deepgram, OpenAI, Groq, ElevenLabs
//! Scribe, AssemblyAI and Google (Gemini 3.5 Transcribe). Plain blocking `reqwest` (rustls) and
//! `tungstenite`, no vendor SDKs.
//!
//! Streaming engines (`SttEngine::streaming()`): Soniox real-time, OpenAI `gpt-live-transcribe`
//! and Google `gemini-3.5-transcribe-live` send audio while the user talks, so release waits only
//! for the final. The rest are batch per phrase. `prewarm()` (key-down) re-opens the pooled TLS /
//! HTTP/2 connection off-thread.
//!
//! Every engine keeps one pooled HTTPS client (TLS stays warm between phrases), pre-warms it in
//! `load()` with a free authenticated GET (which also rejects a bad key early), honours
//! `stt.cloud_timeout_ms` as a total budget per phrase, passes the dictionary as key terms or a
//! prompt, and maps failures to `Error::{Auth, Quota, Network, Timeout, MissingKey}` so the
//! orchestrator can fall back to the local engine (`Error::is_fallback_worthy`).
//! Keys come from `ochre_core::secrets::get(<engine id>)`.

pub mod assemblyai;
pub mod common;
pub mod deepgram;
pub mod elevenlabs;
pub mod google;
pub mod openai;
pub mod openai_rt;
pub mod soniox;
pub mod ws;

#[cfg(test)]
mod live_tests;
#[cfg(test)]
mod testutil;

use ochre_core::config::SttConfig;
use ochre_core::events::EngineInfo;
use ochre_core::stt::SttEngine;
use ochre_core::{Error, Result};

pub const IDS: [&str; 7] = [
    "soniox",
    "deepgram",
    "openai",
    "google",
    "groq",
    "elevenlabs",
    "assemblyai",
];

/// Every cloud engine, in display order.
pub fn engines() -> Vec<EngineInfo> {
    vec![
        soniox::info(),
        deepgram::info(),
        openai::openai_info(),
        google::info(),
        openai::groq_info(),
        elevenlabs::info(),
        assemblyai::info(),
    ]
}

/// Build (not load) the engine `id` with this config.
pub fn create(id: &str, cfg: &SttConfig) -> Result<Box<dyn SttEngine>> {
    Ok(match id {
        "soniox" => Box::new(soniox::Soniox::new(cfg)),
        "deepgram" => Box::new(deepgram::Deepgram::new(cfg)),
        "openai" => Box::new(openai::OpenAiStt::openai(cfg)),
        "google" => Box::new(google::Google::new(cfg)),
        "groq" => Box::new(openai::OpenAiStt::groq(cfg)),
        "elevenlabs" => Box::new(elevenlabs::ElevenLabs::new(cfg)),
        "assemblyai" => Box::new(assemblyai::AssemblyAi::new(cfg)),
        other => {
            return Err(Error::Config(format!(
                "unknown cloud speech engine {other:?}"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry() {
        let infos = engines();
        assert_eq!(infos.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), IDS);
        for id in IDS {
            let e = create(id, &SttConfig::default()).unwrap();
            assert_eq!(e.info().id, id);
            assert!(e.info().needs_key && e.info().kind == "cloud");
        }
        let custom = SttConfig {
            model: "gpt-4o-mini-transcribe".into(),
            ..Default::default()
        };
        assert_eq!(
            create("openai", &custom).unwrap().info().default_model,
            "gpt-4o-mini-transcribe"
        );
        assert!(create("parakeet", &SttConfig::default()).is_err());
        // Streaming follows the model: gpt-transcribe is batch, gpt-live-transcribe streams.
        let live = |id: &str, model: &str| {
            create(
                id,
                &SttConfig {
                    model: model.into(),
                    ..Default::default()
                },
            )
            .unwrap()
            .streaming()
        };
        assert!(!live("openai", "") && live("openai", "gpt-live-transcribe"));
        assert!(!live("groq", "") && !live("google", ""));
        assert!(live("google", "gemini-3.5-transcribe-live") && live("soniox", ""));
        assert_eq!(
            create("openai", &SttConfig::default())
                .unwrap()
                .info()
                .default_model,
            "gpt-transcribe"
        );
    }

    /// Live: every engine whose key is configured loads (key check + prewarm) and transcribes 1 s
    /// of near-silence without error. `cargo test -p ochre-cloud -- --ignored`
    #[test]
    #[ignore = "network + API keys"]
    fn live_engines_with_keys() {
        let pcm: Vec<f32> = (0..16_000)
            .map(|i| ((i as f32) * 0.05).sin() * 0.001)
            .collect();
        for id in IDS {
            if !ochre_core::secrets::has(id) {
                eprintln!("{id}: no key, skipped");
                continue;
            }
            let mut e = create(id, &SttConfig::default()).unwrap();
            e.load(&|_| {})
                .unwrap_or_else(|err| panic!("{id} load: {err}"));
            let r = e.transcribe(&pcm, &ochre_core::stt::SttOptions::default());
            eprintln!(
                "{id}: {:?}",
                r.as_ref().map(|r| (r.processing_ms, r.text.clone()))
            );
            r.unwrap_or_else(|err| panic!("{id} transcribe: {err}"));
        }
    }
}
