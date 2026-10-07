//! Deepgram pre-recorded STT (`POST /v1/listen`), batch per phrase.
//!
//! Raw WAV bytes are the body (no multipart, no upload step), one of the lowest-latency batch APIs.
//! `smart_format` gives punctuation, casing and written-form numbers. Dictionary words go in as
//! repeated `keyterm` params (Nova-3 only). Nova-3 is still Deepgram's prerecorded model as of
//! Oct 2026 (Flux is streaming-only on /v2/listen). ~$0.0043/min ($0.26/h).
//! NOT live-verified (no key): request shape from the docs, pinned by tests.

use std::time::Instant;

use ochre_core::Result;
use ochre_core::config::SttConfig;
use ochre_core::events::EngineInfo;
use ochre_core::stt::{ProgressFn, SttEngine, SttOptions, SttResult};

use crate::common::{self, Core};

pub const BASE: &str = "https://api.deepgram.com/v1";

pub fn info() -> EngineInfo {
    EngineInfo {
        id: "deepgram".into(),
        label: "Deepgram".into(),
        kind: "cloud".into(),
        models: ["nova-3", "nova-3-medical", "nova-2"]
            .map(String::from)
            .to_vec(),
        default_model: "nova-3".into(),
        needs_key: true,
        note: "nova-3 ~$0.26/h".into(),
        languages: "multilingual".into(),
    }
}

pub struct Deepgram {
    pub core: Core,
    base: String,
}

impl Deepgram {
    pub fn new(cfg: &SttConfig) -> Self {
        Self {
            core: Core::new("deepgram", "nova-3", cfg),
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

    pub fn params(&self, lang: Option<&str>, vocab: &[String]) -> Vec<(String, String)> {
        let mut p: Vec<(String, String)> = [
            ("model", self.core.model.as_str()),
            ("smart_format", "true"),
            ("punctuate", "true"),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
        match lang {
            Some(l) => p.push(("language".into(), l.into())),
            None => p.push(("detect_language".into(), "true".into())),
        }
        if self.core.model.starts_with("nova-3") {
            // keyterm total is capped (~500 tokens); 50 short terms stays well inside it.
            p.extend(
                common::terms(vocab, 50, 50)
                    .into_iter()
                    .map(|t| ("keyterm".to_string(), t)),
            );
        }
        p
    }
}

impl SttEngine for Deepgram {
    fn info(&self) -> EngineInfo {
        let mut i = info();
        i.default_model = self.core.model.clone();
        i
    }

    fn load(&mut self, _progress: ProgressFn) -> Result<()> {
        let key = self.core.key()?;
        self.core.prewarm(
            &format!("{}/projects", self.base),
            &[("Authorization", format!("Token {key}"))],
        )
    }

    fn prewarm(&self) {
        if let Ok(key) = self.core.key() {
            self.core.prewarm_async(
                format!("{}/projects", self.base),
                vec![("Authorization", format!("Token {key}"))],
            );
        }
    }

    fn transcribe(&self, pcm: &[f32], opts: &SttOptions) -> Result<SttResult> {
        let key = self.core.key()?;
        let t0 = Instant::now();
        let lang = self.core.lang(opts.language.as_deref());
        let req = self
            .core
            .client
            .post(format!("{}/listen", self.base))
            .query(&self.params(lang.as_deref(), &opts.vocabulary))
            .header("Authorization", format!("Token {key}"))
            .header("Content-Type", "audio/wav")
            .body(common::wav(pcm));
        let body = self.core.send(req, self.core.timeout)?;
        let channel = &body["results"]["channels"][0];
        let text = channel["alternatives"][0]["transcript"]
            .as_str()
            .ok_or_else(|| common::bad_response("deepgram"))?;
        Ok(SttResult {
            text: text.trim().to_string(),
            duration_ms: common::duration_ms(pcm),
            processing_ms: t0.elapsed().as_millis() as u64,
            language: channel["detected_language"]
                .as_str()
                .map(str::to_string)
                .or(lang),
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
            json!({"results": {"channels": [{"detected_language": "en", "alternatives": [{"transcript": "hello world"}]}]}}),
        )]);
        let e = Deepgram::new(&SttConfig::default())
            .with_base(&srv.url("/v1"))
            .with_key("test-key");
        let opts = SttOptions {
            language: None,
            vocabulary: vec!["Kubernetes, Soniox".into()],
        };
        let r = e.transcribe(&vec![0.0; 8000], &opts).unwrap();
        assert_eq!(
            (r.text.as_str(), r.language.as_deref()),
            ("hello world", Some("en"))
        );
        let req = &srv.requests()[0];
        assert_eq!(req.path, "/v1/listen");
        assert_eq!(req.header("authorization"), Some("Token test-key"));
        assert_eq!(req.header("content-type"), Some("audio/wav"));
        assert_eq!(
            req.query,
            "model=nova-3&smart_format=true&punctuate=true&detect_language=true&keyterm=Kubernetes&keyterm=Soniox"
        );
        assert_eq!(&req.body[..4], b"RIFF");
    }

    #[test]
    fn nova2_has_no_keyterms() {
        let cfg = SttConfig {
            model: "nova-2".into(),
            ..Default::default()
        };
        let p = Deepgram::new(&cfg).params(Some("en"), &["X".into()]);
        assert!(
            p.iter().all(|(k, _)| k != "keyterm") && p.contains(&("language".into(), "en".into()))
        );
    }
}
