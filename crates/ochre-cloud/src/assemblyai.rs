//! AssemblyAI pre-recorded STT: upload -> submit -> poll, per phrase.
//!
//! No synchronous batch endpoint exists, so a phrase costs three round trips plus queue time. We
//! poll every 150 ms within the `cloud_timeout_ms` budget and delete the transcript afterwards (in
//! the background) because dictation is private by default. `speech_models` accepts only
//! `universal-3-5-pro` ($0.21/h) and `universal-2` ($0.15/h) as of Oct 2026.
//! NOT live-verified (no key): request shape from the docs, pinned by tests.

use std::time::{Duration, Instant};

use ochre_core::config::SttConfig;
use ochre_core::events::EngineInfo;
use ochre_core::stt::{ProgressFn, SttEngine, SttOptions, SttResult};
use ochre_core::{Error, Result};
use serde_json::{Value, json};

use crate::common::{self, Core};

pub const BASE: &str = "https://api.assemblyai.com/v2";

pub fn info() -> EngineInfo {
    EngineInfo {
        id: "assemblyai".into(),
        label: "AssemblyAI".into(),
        kind: "cloud".into(),
        models: ["universal-3-5-pro", "universal-2"]
            .map(String::from)
            .to_vec(),
        default_model: "universal-3-5-pro".into(),
        needs_key: true,
        note: "universal-3-5-pro ~$0.21/h (async: ~1-2 s per phrase)".into(),
        languages: "multilingual".into(),
    }
}

pub struct AssemblyAi {
    pub core: Core,
    base: String,
    pub poll: Duration,
}

impl AssemblyAi {
    pub fn new(cfg: &SttConfig) -> Self {
        Self {
            core: Core::new("assemblyai", "universal-3-5-pro", cfg),
            base: BASE.into(),
            poll: Duration::from_millis(150),
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

    pub fn job(&self, audio_url: &str, lang: Option<&str>, vocab: &[String]) -> Value {
        let mut job = json!({"audio_url": audio_url, "speech_models": [self.core.model], "punctuate": true, "format_text": true});
        match lang {
            Some(l) => job["language_code"] = json!(l),
            None => job["language_detection"] = json!(true),
        }
        let max = if self.core.model == "universal-2" {
            200
        } else {
            1000
        };
        let terms: Vec<String> = common::terms(vocab, max, 100)
            .into_iter()
            .filter(|t| t.split_whitespace().count() <= 6)
            .collect();
        if !terms.is_empty() {
            job["keyterms_prompt"] = json!(terms);
        }
        job
    }
}

impl SttEngine for AssemblyAi {
    fn info(&self) -> EngineInfo {
        let mut i = info();
        i.default_model = self.core.model.clone();
        i
    }

    fn load(&mut self, _progress: ProgressFn) -> Result<()> {
        let key = self.core.key()?;
        self.core.prewarm(
            &format!("{}/transcript?limit=1", self.base),
            &[("Authorization", key)],
        )
    }

    fn prewarm(&self) {
        if let Ok(key) = self.core.key() {
            self.core.prewarm_async(
                format!("{}/transcript?limit=1", self.base),
                vec![("Authorization", key)],
            );
        }
    }

    fn transcribe(&self, pcm: &[f32], opts: &SttOptions) -> Result<SttResult> {
        let key = self.core.key()?;
        let t0 = Instant::now();
        let deadline = t0 + self.core.timeout;
        let lang = self.core.lang(opts.language.as_deref());
        let c = &self.core;
        let up = c.send(
            c.client
                .post(format!("{}/upload", self.base))
                .header("Authorization", &key)
                .header("Content-Type", "application/octet-stream")
                .body(common::wav(pcm)),
            c.left(deadline)?,
        )?;
        let audio_url = up["upload_url"]
            .as_str()
            .ok_or_else(|| common::bad_response("assemblyai"))?;
        let mut job = c.send(
            c.client
                .post(format!("{}/transcript", self.base))
                .header("Authorization", &key)
                .json(&self.job(audio_url, lang.as_deref(), &opts.vocabulary)),
            c.left(deadline)?,
        )?;
        let id = job["id"]
            .as_str()
            .ok_or_else(|| common::bad_response("assemblyai"))?
            .to_string();
        let cleanup = || {
            c.delete_later(
                vec![format!("{}/transcript/{id}", self.base)],
                ("Authorization", key.clone()),
            )
        };
        while !matches!(job["status"].as_str(), Some("completed") | Some("error")) {
            if Instant::now() + self.poll > deadline {
                cleanup();
                return Err(Error::Timeout(c.timeout));
            }
            std::thread::sleep(self.poll);
            job = match c.send(
                c.client
                    .get(format!("{}/transcript/{id}", self.base))
                    .header("Authorization", &key),
                c.left(deadline)?,
            ) {
                Ok(j) => j,
                Err(e) => {
                    cleanup();
                    return Err(e);
                }
            };
        }
        cleanup();
        if job["status"] == "error" {
            return Err(Error::Other(format!(
                "assemblyai: {}",
                common::redact(job["error"].as_str().unwrap_or("transcription failed"))
            )));
        }
        Ok(SttResult {
            text: job["text"].as_str().unwrap_or("").trim().to_string(),
            duration_ms: common::duration_ms(pcm),
            processing_ms: t0.elapsed().as_millis() as u64,
            language: job["language_code"].as_str().map(str::to_string).or(lang),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{MockServer, Reply};

    #[test]
    fn flow() {
        let srv = MockServer::start(vec![
            Reply::json(200, json!({"upload_url": "https://cdn/x"})),
            Reply::json(200, json!({"id": "t1", "status": "queued"})),
            Reply::json(200, json!({"id": "t1", "status": "processing"})),
            Reply::json(
                200,
                json!({"id": "t1", "status": "completed", "text": "hi there", "language_code": "en"}),
            ),
            Reply::json(200, json!({})),
        ]);
        let mut e = AssemblyAi::new(&SttConfig::default())
            .with_base(&srv.url("/v2"))
            .with_key("test-key");
        e.poll = Duration::from_millis(1);
        let r = e
            .transcribe(
                &vec![0.0; 1600],
                &SttOptions {
                    language: None,
                    vocabulary: vec!["Acme".into()],
                },
            )
            .unwrap();
        assert_eq!(
            (r.text.as_str(), r.language.as_deref()),
            ("hi there", Some("en"))
        );
        // The DELETE runs on a background thread: wait for it (bounded) instead of a fixed sleep.
        for _ in 0..200 {
            if srv.requests().iter().any(|r| r.method == "DELETE") {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let reqs = srv.requests();
        assert_eq!(reqs[0].path, "/v2/upload");
        assert_eq!(reqs[0].header("authorization"), Some("test-key"));
        let job: Value = serde_json::from_slice(&reqs[1].body).unwrap();
        assert_eq!(job["speech_models"], json!(["universal-3-5-pro"]));
        assert_eq!(job["keyterms_prompt"], json!(["Acme"]));
        assert_eq!(job["language_detection"], true);
        assert_eq!(job["audio_url"], "https://cdn/x");
        assert_eq!(
            (reqs[2].method.as_str(), reqs[2].path.as_str()),
            ("GET", "/v2/transcript/t1")
        );
        assert!(
            reqs.iter()
                .any(|r| r.method == "DELETE" && r.path == "/v2/transcript/t1")
        );
    }

    #[test]
    fn times_out_within_budget() {
        let srv = MockServer::start(vec![
            Reply::json(200, json!({"upload_url": "https://cdn/x"})),
            Reply::json(200, json!({"id": "t1", "status": "queued"})),
        ]);
        let cfg = SttConfig {
            cloud_timeout_ms: 600,
            ..Default::default()
        };
        let mut e = AssemblyAi::new(&cfg)
            .with_base(&srv.url("/v2"))
            .with_key("k");
        e.poll = Duration::from_millis(50);
        let t0 = Instant::now();
        assert!(matches!(
            e.transcribe(&[0.0; 160], &SttOptions::default()),
            Err(Error::Timeout(_))
        ));
        assert!(t0.elapsed() < Duration::from_millis(900));
    }
}
