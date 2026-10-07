//! Refinement through the Gemini API (`models/{model}:generateContent`).
//!
//! Default `gemini-3.5-flash-lite` (stable since July 2026; $0.30 / $2.50 per 1M tokens; the 2.5
//! models are closed to new users). Thinking cannot be switched off on Gemini 3.x: its levels are
//! minimal | low | medium | high, and 3.5 Flash-Lite already defaults to `minimal`
//! (ai.google.dev/gemini-api/docs/thinking). We send `thinkingLevel: minimal` explicitly so a
//! future default change cannot slow dictation down. Google recommends leaving Gemini 3
//! `temperature` at its default (lower values can loop), so we only pin it to 0 on older models.
//! Thought parts (`thought: true`) are skipped, so thinking never reaches the output.

use std::time::Duration;

use ochre_core::events::EngineInfo;
use ochre_core::refine::{RefineContext, Refiner};
use ochre_core::stt::ProgressFn;
use ochre_core::{Error, Result};
use serde_json::{Value, json};

use crate::{http, prompts};

pub const BASE: &str = "https://generativelanguage.googleapis.com/v1beta";
pub const DEFAULT_MODEL: &str = "gemini-3.5-flash-lite";

pub fn info() -> EngineInfo {
    EngineInfo {
        id: "gemini".into(),
        label: "Google Gemini".into(),
        kind: "cloud".into(),
        models: vec![
            DEFAULT_MODEL.into(),
            "gemini-3.5-flash".into(),
            "gemini-3.1-flash-lite".into(),
        ],
        default_model: DEFAULT_MODEL.into(),
        needs_key: true,
        note: "gemini-3.5-flash-lite: ~$0.0003 per dictation".into(),
        languages: "multilingual".into(),
    }
}

pub fn generation_config(model: &str, max_tokens: u32) -> Value {
    if model.starts_with("gemini-3") {
        // Headroom in case the model still thinks a little.
        json!({"maxOutputTokens": max_tokens + 512, "thinkingConfig": {"thinkingLevel": "minimal"}})
    } else {
        json!({"maxOutputTokens": max_tokens, "temperature": 0})
    }
}

pub struct GeminiRefiner {
    base: String,
    model: String,
    key_override: Option<String>,
    client: reqwest::blocking::Client,
}

impl GeminiRefiner {
    pub fn new(model: &str, base_url: &str) -> Self {
        Self {
            base: if base_url.is_empty() { BASE } else { base_url }
                .trim_end_matches('/')
                .to_string(),
            model: if model.is_empty() {
                DEFAULT_MODEL
            } else {
                model
            }
            .to_string(),
            key_override: None,
            client: http::client(),
        }
    }

    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key_override = Some(key.into());
        self
    }

    fn key(&self) -> Result<String> {
        self.key_override
            .clone()
            .map(Ok)
            .unwrap_or_else(|| http::require_key("gemini"))
    }

    pub fn build_body(&self, text: &str, ctx: &RefineContext) -> Value {
        json!({
            "systemInstruction": {"parts": [{"text": prompts::system_prompt(ctx)}]},
            "contents": [{"role": "user", "parts": [{"text": prompts::user_message(text)}]}],
            "generationConfig": generation_config(&self.model, http::max_tokens_for(text)),
        })
    }
}

impl Refiner for GeminiRefiner {
    fn info(&self) -> EngineInfo {
        let mut i = info();
        i.default_model = self.model.clone();
        i
    }

    fn load(&mut self, _progress: ProgressFn) -> Result<()> {
        let key = self.key()?;
        http::prewarm(
            &self.client,
            "gemini",
            &format!("{}/models/{}", self.base, self.model),
            &[("x-goog-api-key", &key)],
        )
    }

    fn refine(&self, text: &str, ctx: &RefineContext, timeout: Duration) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(text.to_string());
        }
        let key = self.key()?;
        let url = format!("{}/models/{}:generateContent", self.base, self.model);
        let data = http::post_json(
            &self.client,
            "gemini",
            &url,
            &[("x-goog-api-key", &key)],
            &self.build_body(text, ctx),
            timeout,
        )?;
        let Some(cand) = data["candidates"].get(0) else {
            let reason = data["promptFeedback"]["blockReason"]
                .as_str()
                .unwrap_or("empty response");
            return Err(Error::Other(format!("gemini: no candidate ({reason})")));
        };
        let parts: Vec<&str> = cand["content"]["parts"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|p| p["thought"] != true)
                    .filter_map(|p| p["text"].as_str())
                    .collect()
            })
            .unwrap_or_default();
        if parts.is_empty() {
            return Err(Error::Other(format!(
                "gemini: no text (finishReason={})",
                cand["finishReason"]
            )));
        }
        Ok(parts.concat().trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{MockServer, Reply};

    #[test]
    fn request_shape_skips_thoughts() {
        let srv = MockServer::start(vec![Reply::json(
            200,
            json!({"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": true}, {"text": "Hi."}]}}]}),
        )]);
        let r = GeminiRefiner::new("", &srv.url("/v1beta")).with_key("test-key");
        assert_eq!(
            r.refine("hi", &RefineContext::default(), Duration::from_secs(5))
                .unwrap(),
            "Hi."
        );
        let req = &srv.requests()[0];
        assert_eq!(
            req.path,
            "/v1beta/models/gemini-3.5-flash-lite:generateContent"
        );
        assert_eq!(req.header("x-goog-api-key"), Some("test-key"));
        let b: Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(
            b["generationConfig"]["thinkingConfig"],
            json!({"thinkingLevel": "minimal"})
        );
        assert_eq!(
            b["contents"][0]["parts"][0]["text"],
            "<dictation>\nhi\n</dictation>"
        );
    }

    #[test]
    fn older_models_pin_temperature() {
        assert_eq!(
            generation_config("gemini-2.5-flash-lite", 100),
            json!({"maxOutputTokens": 100, "temperature": 0})
        );
    }
}
