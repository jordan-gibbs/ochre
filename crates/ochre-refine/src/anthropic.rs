//! Refinement through the Anthropic Messages API (`POST /v1/messages`).
//!
//! Default `claude-haiku-4-5` (Oct 2026: still the fastest/cheapest Claude, $1/$5 per 1M tokens,
//! no thinking unless asked, accepts `temperature`). A typical dictation (~600 prompt + ~60 output
//! tokens) costs about $0.0009. Newer models reject a non-default temperature and think by
//! default, so for them we send `output_config.effort: low` and no temperature.

use std::time::Duration;

use ochre_core::events::EngineInfo;
use ochre_core::refine::{RefineContext, Refiner};
use ochre_core::stt::ProgressFn;
use ochre_core::{Error, Result};
use serde_json::{Value, json};

use crate::{http, prompts};

pub const BASE: &str = "https://api.anthropic.com";
pub const API_VERSION: &str = "2023-06-01";
pub const DEFAULT_MODEL: &str = "claude-haiku-4-5";

pub fn info() -> EngineInfo {
    EngineInfo {
        id: "anthropic".into(),
        label: "Anthropic (Claude)".into(),
        kind: "cloud".into(),
        models: vec![DEFAULT_MODEL.into(), "claude-sonnet-5-5".into()],
        default_model: DEFAULT_MODEL.into(),
        needs_key: true,
        note: "claude-haiku-4-5: ~$0.0009 per dictation".into(),
        languages: "multilingual".into(),
    }
}

pub struct AnthropicRefiner {
    base: String,
    model: String,
    key_override: Option<String>,
    client: reqwest::blocking::Client,
}

impl AnthropicRefiner {
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
            .unwrap_or_else(|| http::require_key("anthropic"))
    }

    pub fn build_body(&self, text: &str, ctx: &RefineContext) -> Value {
        let haiku = self.model.starts_with("claude-haiku");
        let mut body = json!({
            "model": self.model,
            "max_tokens": http::max_tokens_for(text) + if haiku { 0 } else { 2048 },
            "system": prompts::system_prompt(ctx),
            "messages": [{"role": "user", "content": prompts::user_message(text)}],
        });
        if haiku {
            body["temperature"] = json!(0);
        } else {
            body["output_config"] = json!({"effort": "low"});
        }
        body
    }
}

impl Refiner for AnthropicRefiner {
    fn info(&self) -> EngineInfo {
        let mut i = info();
        i.default_model = self.model.clone();
        i
    }

    fn load(&mut self, _progress: ProgressFn) -> Result<()> {
        let key = self.key()?;
        http::prewarm(
            &self.client,
            "anthropic",
            &format!("{}/v1/models", self.base),
            &[("x-api-key", &key), ("anthropic-version", API_VERSION)],
        )
    }

    fn refine(&self, text: &str, ctx: &RefineContext, timeout: Duration) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(text.to_string());
        }
        let key = self.key()?;
        let headers = [
            ("x-api-key", key.as_str()),
            ("anthropic-version", API_VERSION),
        ];
        let data = http::post_json(
            &self.client,
            "anthropic",
            &format!("{}/v1/messages", self.base),
            &headers,
            &self.build_body(text, ctx),
            timeout,
        )?;
        if data["stop_reason"] == "refusal" {
            return Err(Error::Other("anthropic: request declined".into()));
        }
        let parts: Vec<&str> = data["content"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter(|b| b["type"] == "text")
                    .filter_map(|b| b["text"].as_str())
                    .collect()
            })
            .unwrap_or_default();
        if parts.is_empty() {
            return Err(Error::Other("anthropic: no text in response".into()));
        }
        Ok(parts.concat().trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{MockServer, Reply};

    #[test]
    fn request_shape() {
        let srv = MockServer::start(vec![Reply::json(
            200,
            json!({"content": [{"type": "text", "text": "Hi."}], "stop_reason": "end_turn"}),
        )]);
        let r = AnthropicRefiner::new("", &srv.url("")).with_key("test-key");
        assert_eq!(
            r.refine("hi", &RefineContext::default(), Duration::from_secs(5))
                .unwrap(),
            "Hi."
        );
        let req = &srv.requests()[0];
        assert_eq!(req.path, "/v1/messages");
        assert_eq!(req.header("x-api-key"), Some("test-key"));
        assert_eq!(req.header("anthropic-version"), Some("2023-06-01"));
        let b: Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(b["model"], "claude-haiku-4-5");
        assert_eq!(b["temperature"], 0);
        assert!(
            b["system"]
                .as_str()
                .unwrap()
                .starts_with("You clean up dictated text.")
        );
    }

    #[test]
    fn auth_error() {
        let srv = MockServer::start(vec![Reply::json(
            401,
            json!({"error": {"message": "bad key"}}),
        )]);
        let r = AnthropicRefiner::new("", &srv.url("")).with_key("k");
        assert!(matches!(
            r.refine("hi", &RefineContext::default(), Duration::from_secs(5)),
            Err(Error::Auth { .. })
        ));
    }

    #[test]
    fn non_haiku_uses_effort_not_temperature() {
        let b = AnthropicRefiner::new("claude-sonnet-5-5", "")
            .build_body("hi", &RefineContext::default());
        assert!(b.get("temperature").is_none() && b["output_config"]["effort"] == "low");
    }
}
