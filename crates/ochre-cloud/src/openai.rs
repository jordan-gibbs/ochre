//! OpenAI (`POST /v1/audio/transcriptions`) and Groq (same API at `/openai/v1`), batch per phrase,
//! plus OpenAI's streaming `gpt-live-transcribe` over the Realtime API (`openai_rt`).
//!
//! OpenAI default `gpt-transcribe` (released 2026-07-28, batch): measured 0.86-1.13 s on an 8.5 s
//! clip vs 1.6-2.0 s for `gpt-4o-mini-transcribe`, and the most accurate. `prompt` is free text,
//! so the dictionary goes there (capped at 800 chars; whisper-style models read ~224 tokens).
//! `gpt-live-transcribe` is not served by `/audio/transcriptions`; selecting it makes the engine
//! stream while the user talks (`stream()`), and a finished buffer is pushed through the same
//! Realtime socket. Cost (Oct 2026): gpt-transcribe $0.0045/min, gpt-live-transcribe $0.017/min,
//! gpt-4o-mini-transcribe $0.003/min, gpt-4o-transcribe $0.006/min.
//!
//! Groq default `whisper-large-v3-turbo` ($0.04/h; 10 s minimum billed per request).

use std::time::Instant;

use ochre_core::Result;
use ochre_core::config::SttConfig;
use ochre_core::events::EngineInfo;
use ochre_core::stt::{PartialSink, ProgressFn, SttEngine, SttOptions, SttResult, SttStream};
use reqwest::blocking::multipart::{Form, Part};

use crate::common::{self, Core};
use crate::openai_rt::{self, LIVE_MODEL, OpenAiLive};

pub const OPENAI_BASE: &str = "https://api.openai.com/v1";
pub const GROQ_BASE: &str = "https://api.groq.com/openai/v1";
const MAX_PROMPT_CHARS: usize = 800;

pub fn openai_info() -> EngineInfo {
    EngineInfo {
        id: "openai".into(),
        label: "OpenAI".into(),
        kind: "cloud".into(),
        models: [
            "gpt-transcribe",
            LIVE_MODEL,
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe",
            "whisper-1",
        ]
        .map(String::from)
        .to_vec(),
        default_model: "gpt-transcribe".into(),
        needs_key: true,
        note: "gpt-transcribe ~$0.27/h; gpt-live-transcribe (streaming) ~$1.02/h".into(),
        languages: "multilingual".into(),
    }
}

pub fn groq_info() -> EngineInfo {
    EngineInfo {
        id: "groq".into(),
        label: "Groq (Whisper)".into(),
        kind: "cloud".into(),
        models: ["whisper-large-v3-turbo", "whisper-large-v3"]
            .map(String::from)
            .to_vec(),
        default_model: "whisper-large-v3-turbo".into(),
        needs_key: true,
        note: "~$0.04/h (10 s minimum per request)".into(),
        languages: "multilingual".into(),
    }
}

/// Any OpenAI-compatible transcription endpoint.
pub struct OpenAiStt {
    pub core: Core,
    base: String,
    info: EngineInfo,
    /// Realtime `delay` for gpt-live-transcribe.
    pub delay: String,
}

impl OpenAiStt {
    pub fn openai(cfg: &SttConfig) -> Self {
        Self {
            core: Core::new("openai", "gpt-transcribe", cfg),
            base: OPENAI_BASE.into(),
            info: openai_info(),
            delay: openai_rt::DEFAULT_DELAY.into(),
        }
    }

    pub fn groq(cfg: &SttConfig) -> Self {
        Self {
            core: Core::new("groq", "whisper-large-v3-turbo", cfg),
            base: GROQ_BASE.into(),
            info: groq_info(),
            delay: openai_rt::DEFAULT_DELAY.into(),
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

    fn is_live(&self) -> bool {
        self.core.provider == "openai" && self.core.model == LIVE_MODEL
    }

    fn open_live(&self, opts: &SttOptions, on_partial: Option<PartialSink>) -> Result<OpenAiLive> {
        let update = openai_rt::session_update(
            &self.core.model,
            self.core.lang(opts.language.as_deref()).as_deref(),
            &opts.vocabulary,
            &self.delay,
        );
        OpenAiLive::open(
            &self.core.key()?,
            &openai_rt::ws_url(&self.base),
            &update,
            self.core.timeout,
            on_partial,
        )
    }
}

impl SttEngine for OpenAiStt {
    fn info(&self) -> EngineInfo {
        let mut i = self.info.clone();
        i.default_model = self.core.model.clone();
        i
    }

    fn load(&mut self, _progress: ProgressFn) -> Result<()> {
        let key = self.core.key()?;
        self.core.prewarm(
            &format!("{}/models", self.base),
            &[("Authorization", format!("Bearer {key}"))],
        )
    }

    fn transcribe(&self, pcm: &[f32], opts: &SttOptions) -> Result<SttResult> {
        if self.is_live() {
            // A finished buffer through the Realtime socket (fallback / Test path).
            let t0 = Instant::now();
            let mut s = Box::new(self.open_live(opts, None)?);
            for chunk in pcm.chunks(ochre_core::SAMPLE_RATE as usize) {
                s.send(chunk)?;
            }
            let mut r = s.finish()?;
            r.processing_ms = t0.elapsed().as_millis() as u64;
            return Ok(r);
        }
        let key = self.core.key()?;
        let t0 = Instant::now();
        let lang = self.core.lang(opts.language.as_deref());
        let mut form = Form::new()
            .text("model", self.core.model.clone())
            .text("response_format", "json")
            .text("temperature", "0")
            .part(
                "file",
                Part::bytes(common::wav(pcm))
                    .file_name("audio.wav")
                    .mime_str("audio/wav")
                    .expect("mime"),
            );
        if let Some(l) = &lang {
            form = form.text("language", l.clone());
        }
        let prompt: String = common::terms(&opts.vocabulary, 200, 100)
            .join(", ")
            .chars()
            .take(MAX_PROMPT_CHARS)
            .collect();
        if !prompt.is_empty() {
            form = form.text("prompt", prompt);
        }
        let req = self
            .core
            .client
            .post(format!("{}/audio/transcriptions", self.base))
            .bearer_auth(key)
            .multipart(form);
        let body = self.core.send(req, self.core.timeout)?;
        let text = body["text"]
            .as_str()
            .ok_or_else(|| common::bad_response(self.core.provider))?;
        Ok(SttResult {
            text: text.trim().to_string(),
            duration_ms: common::duration_ms(pcm),
            processing_ms: t0.elapsed().as_millis() as u64,
            language: body["language"].as_str().map(str::to_string).or(lang),
        })
    }

    fn streaming(&self) -> bool {
        self.is_live()
    }

    fn stream(
        &self,
        opts: &SttOptions,
        on_partial: Option<PartialSink>,
    ) -> Option<Result<Box<dyn SttStream>>> {
        self.is_live().then(|| {
            self.open_live(opts, on_partial)
                .map(|s| Box::new(s) as Box<dyn SttStream>)
        })
    }

    fn prewarm(&self) {
        if let Ok(key) = self.core.key() {
            self.core.prewarm_async(
                format!("{}/models", self.base),
                vec![("Authorization", format!("Bearer {key}"))],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{MockServer, Reply};
    use ochre_core::Error;
    use serde_json::json;
    use std::time::Duration;

    fn cfg() -> SttConfig {
        SttConfig {
            engine: "groq".into(),
            language: Some("en".into()),
            ..Default::default()
        }
    }

    #[test]
    fn request_shape() {
        let srv = MockServer::start(vec![Reply::json(200, json!({"text": " hello world "}))]);
        let e = OpenAiStt::groq(&cfg())
            .with_base(&srv.url("/openai/v1"))
            .with_key("test-key");
        let opts = SttOptions {
            language: None,
            vocabulary: vec!["Kubernetes".into(), "Soniox".into()],
        };
        let r = e.transcribe(&vec![0.0; 16_000], &opts).unwrap();
        assert_eq!(
            (r.text.as_str(), r.duration_ms, r.language.as_deref()),
            ("hello world", 1000, Some("en"))
        );
        let req = &srv.requests()[0];
        assert_eq!(req.path, "/openai/v1/audio/transcriptions");
        assert_eq!(req.header("authorization"), Some("Bearer test-key"));
        let body = req.body_str();
        for f in [
            "name=\"model\"\r\n\r\nwhisper-large-v3-turbo",
            "name=\"language\"\r\n\r\nen",
            "name=\"prompt\"\r\n\r\nKubernetes, Soniox",
            "filename=\"audio.wav\"",
            "RIFF",
        ] {
            assert!(body.contains(f), "missing {f}");
        }
    }

    #[test]
    fn error_mapping() {
        for (status, check) in [
            (
                401,
                (|e: &Error| matches!(e, Error::Auth { .. })) as fn(&Error) -> bool,
            ),
            (402, |e| matches!(e, Error::Quota { .. })),
            (429, |e| matches!(e, Error::Quota { .. })),
            (503, |e| matches!(e, Error::Network { .. })),
        ] {
            let srv = MockServer::start(vec![Reply::json(
                status,
                json!({"error": {"message": "nope"}}),
            )]);
            let e = OpenAiStt::openai(&cfg())
                .with_base(&srv.url("/v1"))
                .with_key("k");
            let err = e
                .transcribe(&[0.0; 160], &SttOptions::default())
                .unwrap_err();
            assert!(check(&err) && err.is_fallback_worthy(), "{status}: {err:?}");
        }
    }

    #[test]
    fn timeout_is_timeout() {
        let srv = MockServer::start(vec![Reply::delay(Duration::from_millis(1500))]);
        let c = SttConfig {
            cloud_timeout_ms: 500,
            ..cfg()
        };
        let e = OpenAiStt::openai(&c)
            .with_base(&srv.url("/v1"))
            .with_key("k");
        let t0 = Instant::now();
        let err = e
            .transcribe(&[0.0; 160], &SttOptions::default())
            .unwrap_err();
        assert!(matches!(err, Error::Timeout(_)), "{err:?}");
        assert!(t0.elapsed() < Duration::from_millis(1400));
    }

    #[test]
    fn missing_key() {
        let mut e = OpenAiStt::openai(&cfg());
        e.core.provider = "nonexistent-provider-for-test";
        assert!(matches!(
            e.transcribe(&[0.0; 160], &SttOptions::default()),
            Err(Error::MissingKey(_))
        ));
    }
}
