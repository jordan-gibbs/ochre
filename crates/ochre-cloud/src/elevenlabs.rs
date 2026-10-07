//! ElevenLabs Scribe (`POST /v1/speech-to-text`), batch per phrase.
//!
//! Multipart with raw 16 kHz PCM and `file_format=pcm_s16le_16`, which skips server-side decoding.
//! Audio-event tags ("(laughter)") are off because they would be typed into the user's text box.
//! Dictionary words go in as repeated `keyterms` (< 50 chars, <= 5 words, none of `<>{}[]\`;
//! 100+ terms bills a 20 s minimum, so we cap at 99; keyterms add a ~20% surcharge).
//! `scribe_v2` is current; `scribe_v1` is deprecated (Oct 2026). ~$0.22-0.40/h by plan.
//! NOT live-verified (no key): request shape from the docs, pinned by tests.

use std::time::Instant;

use ochre_core::Result;
use ochre_core::config::SttConfig;
use ochre_core::events::EngineInfo;
use ochre_core::stt::{ProgressFn, SttEngine, SttOptions, SttResult};
use reqwest::blocking::multipart::{Form, Part};

use crate::common::{self, Core};

pub const BASE: &str = "https://api.elevenlabs.io/v1";

pub fn info() -> EngineInfo {
    EngineInfo {
        id: "elevenlabs".into(),
        label: "ElevenLabs Scribe".into(),
        kind: "cloud".into(),
        models: vec!["scribe_v2".into()],
        default_model: "scribe_v2".into(),
        needs_key: true,
        note: "scribe_v2 ~$0.22-0.40/h".into(),
        languages: "multilingual".into(),
    }
}

pub struct ElevenLabs {
    pub core: Core,
    base: String,
}

impl ElevenLabs {
    pub fn new(cfg: &SttConfig) -> Self {
        Self {
            core: Core::new("elevenlabs", "scribe_v2", cfg),
            base: BASE.into(),
        }
    }
    pub fn with_base(mut self, base: &str) -> Self {
        self.base = base.trim_end_matches('/').to_string();
        self
    }
    pub fn with_key(mut self, key: &str) -> Self {
        self.core.key_override = Some(key.into());
        self
    }

    pub fn keyterms(vocab: &[String]) -> Vec<String> {
        common::terms(vocab, 1000, 49)
            .into_iter()
            .filter(|t| {
                t.split_whitespace().count() <= 5
                    && !t.contains(['<', '>', '{', '}', '[', ']', '\\'])
            })
            .take(99)
            .collect()
    }
}

impl SttEngine for ElevenLabs {
    fn info(&self) -> EngineInfo {
        let mut i = info();
        i.default_model = self.core.model.clone();
        i
    }

    fn load(&mut self, _progress: ProgressFn) -> Result<()> {
        let key = self.core.key()?;
        self.core
            .prewarm(&format!("{}/models", self.base), &[("xi-api-key", key)])
    }

    fn prewarm(&self) {
        if let Ok(key) = self.core.key() {
            self.core
                .prewarm_async(format!("{}/models", self.base), vec![("xi-api-key", key)]);
        }
    }

    fn transcribe(&self, pcm: &[f32], opts: &SttOptions) -> Result<SttResult> {
        let key = self.core.key()?;
        let t0 = Instant::now();
        let lang = self.core.lang(opts.language.as_deref());
        let mut form = Form::new()
            .text("model_id", self.core.model.clone())
            .text("file_format", "pcm_s16le_16")
            .text("tag_audio_events", "false")
            .text("timestamps_granularity", "word")
            .text("diarize", "false")
            .part(
                "file",
                Part::bytes(common::pcm16(pcm))
                    .file_name("audio.pcm")
                    .mime_str("application/octet-stream")
                    .expect("mime"),
            );
        if let Some(l) = &lang {
            form = form.text("language_code", l.clone());
        }
        for t in Self::keyterms(&opts.vocabulary) {
            form = form.text("keyterms", t);
        }
        let req = self
            .core
            .client
            .post(format!("{}/speech-to-text", self.base))
            .header("xi-api-key", key)
            .multipart(form);
        let body = self.core.send(req, self.core.timeout)?;
        let text = body["text"]
            .as_str()
            .ok_or_else(|| common::bad_response("elevenlabs"))?;
        Ok(SttResult {
            text: text.trim().to_string(),
            duration_ms: common::duration_ms(pcm),
            processing_ms: t0.elapsed().as_millis() as u64,
            language: body["language_code"].as_str().map(str::to_string).or(lang),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{MockServer, Reply};
    use serde_json::json;

    #[test]
    fn request_shape() {
        let srv = MockServer::start(vec![Reply::json(
            200,
            json!({"text": "hi", "language_code": "en", "words": []}),
        )]);
        let e = ElevenLabs::new(&SttConfig::default())
            .with_base(&srv.url("/v1"))
            .with_key("test-key");
        let opts = SttOptions {
            language: None,
            vocabulary: vec!["Acme".into(), "a b c d e f".into(), "bad[x]".into()],
        };
        assert_eq!(e.transcribe(&vec![0.0; 1600], &opts).unwrap().text, "hi");
        let req = &srv.requests()[0];
        assert_eq!(req.path, "/v1/speech-to-text");
        assert_eq!(req.header("xi-api-key"), Some("test-key"));
        let body = req.body_str();
        for f in [
            "name=\"model_id\"\r\n\r\nscribe_v2",
            "name=\"file_format\"\r\n\r\npcm_s16le_16",
            "name=\"keyterms\"\r\n\r\nAcme",
            "name=\"tag_audio_events\"\r\n\r\nfalse",
        ] {
            assert!(body.contains(f), "missing {f}");
        }
        assert_eq!(body.matches("name=\"keyterms\"").count(), 1);
        assert!(!body.contains("RIFF")); // raw PCM, no WAV header
    }
}
