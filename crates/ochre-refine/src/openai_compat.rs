//! Refinement through any OpenAI-compatible `/chat/completions` endpoint: OpenAI, Groq,
//! OpenRouter, Cerebras and any custom base URL (Ollama, LM Studio, vLLM, Together, ...).
//!
//! Defaults (verified against provider docs, Oct 2026), picked for latency first, then cost:
//! - openai: `gpt-5.6-luna` (GPT-5.6's fast, cost-sensitive tier; $0.20 / $1.20 per 1M tokens)
//!   with `reasoning_effort: "none"`. Luna's efforts are none | low | medium (default) | high |
//!   xhigh | max: `"minimal"` is rejected with HTTP 400 on both Chat Completions and Responses
//!   (verified live 2026-10-04), and `none` is its zero-reasoning setting (0 reasoning tokens).
//!   Chat Completions returns only the final answer in `message.content`; reasoning is never in it.
//! - groq: `openai/gpt-oss-20b` (~1000 tok/s) with `reasoning_effort: low` (its minimum).
//!   `llama-3.1-8b-instant` was shut down on 2026-08-16.
//! - openrouter: `openai/gpt-4.1-nano`.
//! - cerebras: `gpt-oss-120b` with `reasoning_effort: low` (`llama3.1-8b` is no longer listed).
//!
//! Reasoning models reject `temperature` unless reasoning is off, so params depend on the model,
//! and a 400 naming one of our optional params is retried once without them.

use std::time::Duration;

use ochre_core::events::EngineInfo;
use ochre_core::refine::{RefineContext, Refiner};
use ochre_core::stt::ProgressFn;
use ochre_core::{Error, Result};
use serde_json::{Value, json};

use crate::http;
use crate::prompts;

pub struct Preset {
    pub id: &'static str,
    pub label: &'static str,
    pub base_url: &'static str,
    pub default_model: &'static str,
    pub models: &'static [&'static str],
    pub note: &'static str,
}

pub const PRESETS: &[Preset] = &[
    Preset {
        id: "openai",
        label: "OpenAI",
        base_url: "https://api.openai.com/v1",
        default_model: "gpt-6-luna",
        models: &[
            "gpt-6-luna",
            "gpt-5.6-luna",
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-5.4-nano",
        ],
        note: "gpt-6-luna, no reasoning: ~$0.0001 per dictation",
    },
    Preset {
        id: "groq",
        label: "Groq",
        base_url: "https://api.groq.com/openai/v1",
        default_model: "openai/gpt-oss-20b",
        models: &["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        note: "gpt-oss-20b, ~1000 tok/s: ~$0.00006 per dictation",
    },
    Preset {
        id: "openrouter",
        label: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        default_model: "openai/gpt-4.1-nano",
        models: &[
            "openai/gpt-4.1-nano",
            "google/gemini-3.5-flash-lite",
            "anthropic/claude-haiku-4-5",
            "openai/gpt-oss-20b",
        ],
        note: "any OpenRouter model slug",
    },
    Preset {
        id: "cerebras",
        label: "Cerebras",
        base_url: "https://api.cerebras.ai/v1",
        default_model: "gpt-oss-120b",
        models: &["gpt-oss-120b", "qwen-3.8-27b"],
        note: "gpt-oss-120b, very fast",
    },
    Preset {
        id: "custom",
        label: "Custom (OpenAI-compatible)",
        base_url: "",
        default_model: "",
        models: &[],
        note: "Ollama, LM Studio, vLLM, ... via refine.base_url",
    },
];

pub fn preset(id: &str) -> &'static Preset {
    PRESETS
        .iter()
        .find(|p| p.id == id)
        .unwrap_or(&PRESETS[PRESETS.len() - 1])
}

pub fn info(p: &Preset) -> EngineInfo {
    EngineInfo {
        id: p.id.into(),
        label: p.label.into(),
        kind: "cloud".into(),
        models: p.models.iter().map(|m| m.to_string()).collect(),
        default_model: p.default_model.into(),
        needs_key: p.id != "custom",
        note: p.note.into(),
        languages: "multilingual".into(),
    }
}

fn bare(model: &str) -> &str {
    model.rsplit('/').next().unwrap_or(model)
}

/// True for models that think before answering (and so reject `temperature` by default).
pub fn is_reasoning(model: &str) -> bool {
    let m = bare(model);
    m.starts_with("gpt-5")
        || m.starts_with("gpt-6")
        || (m.starts_with('o') && m[1..].starts_with(|c: char| c.is_ascii_digit()))
        || m.contains("gpt-oss")
}

/// Sampling / reasoning params a given model accepts.
pub fn model_params(model: &str) -> serde_json::Map<String, Value> {
    let m = bare(model);
    let mut p = serde_json::Map::new();
    if m.contains("gpt-oss") {
        // gpt-oss accepts low/medium/high only.
        p.insert("reasoning_effort".into(), json!("low"));
        p.insert("temperature".into(), json!(0));
    } else if m.starts_with('o') && m[1..].starts_with(|c: char| c.is_ascii_digit()) {
        p.insert("reasoning_effort".into(), json!("low"));
    } else if m == "gpt-5" || m.starts_with("gpt-5-") {
        // The original gpt-5 family's lowest effort is "minimal".
        p.insert("reasoning_effort".into(), json!("minimal"));
    } else if m.starts_with("gpt-5") || m.starts_with("gpt-6") {
        // gpt-5.x / gpt-6 (Luna included): "none" is the lowest effort and turns reasoning off;
        // they reject "minimal". No temperature (docs/refinement.md Â§7.1).
        p.insert("reasoning_effort".into(), json!("none"));
    } else {
        p.insert("temperature".into(), json!(0));
    }
    p
}

pub struct OpenAiCompatRefiner {
    provider: String,
    kind: &'static str,
    base_url: String,
    model: String,
    key_override: Option<String>,
    key_optional: bool,
    client: reqwest::blocking::Client,
}

impl OpenAiCompatRefiner {
    pub fn new(provider: &str, model: &str, base_url: &str) -> Result<Self> {
        let p = preset(provider);
        let base_url = if base_url.is_empty() {
            p.base_url
        } else {
            base_url
        }
        .trim_end_matches('/')
        .to_string();
        let model = if model.is_empty() {
            p.default_model
        } else {
            model
        }
        .to_string();
        if base_url.is_empty() {
            return Err(Error::Config(format!("{provider}: set refine.base_url")));
        }
        if model.is_empty() {
            return Err(Error::Config(format!("{provider}: set refine.model")));
        }
        let local = base_url.contains("://127.0.0.1")
            || base_url.contains("://localhost")
            || base_url.contains("://[::1]");
        Ok(Self {
            provider: provider.to_string(),
            kind: if local { "local" } else { "cloud" },
            key_optional: p.id == "custom" || local,
            base_url,
            model,
            key_override: None,
            client: if local {
                http::local_client()
            } else {
                http::client()
            },
        })
    }

    /// Use this key instead of the keyring/env lookup (tests, "Test" button with an unsaved key).
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.key_override = Some(key.into());
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn key(&self) -> Result<Option<String>> {
        if let Some(k) = &self.key_override {
            return Ok(Some(k.clone()));
        }
        // "custom" keys are stored under the provider id too (e.g. a key for a hosted vLLM).
        match ochre_core::secrets::get(&self.provider) {
            Some(k) => Ok(Some(k)),
            None if self.key_optional => Ok(None),
            None => Err(Error::MissingKey(self.provider.clone())),
        }
    }

    pub fn build_body(&self, text: &str, ctx: &RefineContext) -> Value {
        let mut cap = http::max_tokens_for(text);
        if is_reasoning(&self.model) {
            cap += 1024;
        }
        let token_field = if self.base_url.contains("api.openai.com") {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        let mut body = json!({
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompts::system_prompt(ctx)},
                {"role": "user", "content": prompts::user_message(text)},
            ],
            "stream": false,
        });
        body[token_field] = json!(cap);
        let obj = body.as_object_mut().unwrap();
        obj.extend(model_params(&self.model));
        body
    }
}

impl Refiner for OpenAiCompatRefiner {
    fn info(&self) -> EngineInfo {
        let mut i = info(preset(&self.provider));
        i.kind = self.kind.into();
        i.default_model = self.model.clone();
        i
    }

    fn load(&mut self, _progress: ProgressFn) -> Result<()> {
        let key = self.key()?;
        let auth = key.map(|k| format!("Bearer {k}"));
        let headers: Vec<(&str, &str)> = auth
            .as_deref()
            .map(|a| vec![("Authorization", a)])
            .unwrap_or_default();
        http::prewarm(
            &self.client,
            &self.provider,
            &format!("{}/models", self.base_url),
            &headers,
        )
    }

    fn refine(&self, text: &str, ctx: &RefineContext, timeout: Duration) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(text.to_string());
        }
        let auth = self.key()?.map(|k| format!("Bearer {k}"));
        let headers: Vec<(&str, &str)> = auth
            .as_deref()
            .map(|a| vec![("Authorization", a)])
            .unwrap_or_default();
        let url = format!("{}/chat/completions", self.base_url);
        let mut body = self.build_body(text, ctx);
        let data =
            match http::post_json(&self.client, &self.provider, &url, &headers, &body, timeout) {
                Ok(d) => d,
                Err(e)
                    if e.status == Some(400) && {
                        let b = e.body.to_lowercase();
                        b.contains("temperature") || b.contains("reasoning")
                    } =>
                {
                    // A model that rejects one of our optional params: retry once without them.
                    let obj = body.as_object_mut().unwrap();
                    obj.remove("temperature");
                    obj.remove("reasoning_effort");
                    http::post_json(&self.client, &self.provider, &url, &headers, &body, timeout)?
                }
                Err(e) => return Err(e.into()),
            };
        // Only `message.content` is the answer: reasoning models keep their reasoning out of it
        // (some OpenAI-compatible servers put it in `reasoning` / `reasoning_content`, which we
        // never read). A reasoning model that spent its budget thinking returns empty content
        // with finish_reason "length": that is an error, so the caller inserts the raw text.
        let choice = &data["choices"][0];
        let content = choice["message"]["content"].as_str().unwrap_or("");
        if content.trim().is_empty() {
            return Err(Error::Other(format!(
                "{}: no answer (finish_reason={})",
                self.provider, choice["finish_reason"]
            )));
        }
        Ok(content.trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{MockServer, Reply};

    #[test]
    fn params_per_model() {
        assert_eq!(
            Value::Object(model_params("gpt-4.1-nano")),
            json!({"temperature": 0})
        );
        assert_eq!(
            Value::Object(model_params("gpt-6-luna")),
            json!({"reasoning_effort": "none"})
        );
        assert_eq!(
            Value::Object(model_params("gpt-5.6-luna")),
            json!({"reasoning_effort": "none"})
        );
        assert_eq!(
            Value::Object(model_params("gpt-5.4-nano")),
            json!({"reasoning_effort": "none"})
        );
        assert_eq!(
            Value::Object(model_params("gpt-5-nano")),
            json!({"reasoning_effort": "minimal"})
        );
        assert_eq!(
            model_params("openai/gpt-oss-20b")["reasoning_effort"],
            "low"
        );
        assert_eq!(model_params("o4-mini")["reasoning_effort"], "low");
        assert!(
            is_reasoning("gpt-oss-120b")
                && !is_reasoning("gpt-4.1-nano")
                && !is_reasoning("openrouter/auto")
        );
    }

    #[test]
    fn request_shape_and_retry_without_rejected_param() {
        let srv = MockServer::start(vec![
            Reply::json(
                400,
                json!({"error": {"message": "Unsupported parameter: 'temperature'"}}),
            ),
            Reply::json(200, json!({"choices": [{"message": {"content": " Hi. "}}]})),
        ]);
        let r = OpenAiCompatRefiner::new("groq", "some-model", &srv.url("/v1"))
            .unwrap()
            .with_key("test-key");
        assert_eq!(
            r.refine("hi", &RefineContext::default(), Duration::from_secs(5))
                .unwrap(),
            "Hi."
        );
        let reqs = srv.requests();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].path, "/v1/chat/completions");
        assert_eq!(reqs[0].header("authorization"), Some("Bearer test-key"));
        let b0: Value = serde_json::from_slice(&reqs[0].body).unwrap();
        let b1: Value = serde_json::from_slice(&reqs[1].body).unwrap();
        assert_eq!(
            b0["messages"][1]["content"],
            "<dictation>\nhi\n</dictation>"
        );
        assert!(
            b0.get("max_tokens").is_some()
                && b0["temperature"] == 0
                && b1.get("temperature").is_none()
        );
    }

    #[test]
    fn openai_uses_max_completion_tokens() {
        let r = OpenAiCompatRefiner::new("openai", "", "").unwrap();
        let b = r.build_body("hello there", &RefineContext::default());
        assert!(b.get("max_completion_tokens").is_some() && b.get("max_tokens").is_none());
        assert_eq!(b["model"], "gpt-6-luna");
        // Luna: no reasoning, no temperature.
        assert_eq!(b["reasoning_effort"], "none");
        assert!(b.get("temperature").is_none());
        assert_eq!(
            b["messages"][1]["content"],
            "<dictation>\nhello there\n</dictation>"
        );
    }

    #[test]
    fn luna_request_shape_and_reasoning_never_leaks() {
        // A response carrying reasoning beside the answer: only `content` is used. An empty
        // answer (budget spent on reasoning) is an error, so the raw text is inserted.
        let srv = MockServer::start(vec![
            Reply::json(
                200,
                json!({"choices": [{"message": {"content": "Send me the file.", "reasoning_content": "The user wants..."}, "finish_reason": "stop"}]}),
            ),
            Reply::json(
                200,
                json!({"choices": [{"message": {"content": "", "reasoning": "thinking..."}, "finish_reason": "length"}]}),
            ),
        ]);
        let r = OpenAiCompatRefiner::new("openai", "", &srv.url("/v1"))
            .unwrap()
            .with_key("k");
        let ctx = RefineContext {
            mode: "clean".into(),
            ..Default::default()
        };
        assert_eq!(
            r.refine("um send me the file", &ctx, Duration::from_secs(5))
                .unwrap(),
            "Send me the file."
        );
        let b: Value = serde_json::from_slice(&srv.requests()[0].body).unwrap();
        assert_eq!(b["model"], "gpt-6-luna");
        assert_eq!(b["reasoning_effort"], "none");
        assert_eq!(
            b["messages"][1]["content"],
            "<dictation>\num send me the file\n</dictation>"
        );
        let o = crate::refine_text(
            Some(&r),
            "um send me the file",
            &ctx,
            Duration::from_secs(5),
        );
        assert!(!o.refined && o.text == "um send me the file", "{o:?}");
    }

    #[test]
    fn errors_map() {
        let srv = MockServer::start(vec![Reply::json(
            401,
            json!({"error": {"message": "bad key"}}),
        )]);
        let r = OpenAiCompatRefiner::new("openai", "", &srv.url("/v1"))
            .unwrap()
            .with_key("k");
        let e = r
            .refine("hi", &RefineContext::default(), Duration::from_secs(5))
            .unwrap_err();
        assert!(matches!(e, Error::Auth { .. }), "{e:?}");
        let srv = MockServer::start(vec![Reply::delay(Duration::from_millis(800))]);
        let r = OpenAiCompatRefiner::new("openai", "", &srv.url("/v1"))
            .unwrap()
            .with_key("k");
        let e = r
            .refine("hi", &RefineContext::default(), Duration::from_millis(200))
            .unwrap_err();
        assert!(matches!(e, Error::Timeout(_)), "{e:?}");
    }

    #[test]
    fn local_base_url_needs_no_key() {
        let srv = MockServer::start(vec![Reply::json(
            200,
            json!({"choices": [{"message": {"content": "Ok."}}]}),
        )]);
        let r = OpenAiCompatRefiner::new("custom", "llama3", &srv.url("/v1")).unwrap();
        assert_eq!(r.info().kind, "local");
        assert_eq!(
            r.refine("ok", &RefineContext::default(), Duration::from_secs(5))
                .unwrap(),
            "Ok."
        );
        assert_eq!(srv.requests()[0].header("authorization"), None);
    }

    /// Live: GPT-5.6 Luna at its lowest effort (`none`; `minimal` is rejected) on a short line and
    /// a ~100-word paragraph. The key is read from a dotenv file (`OCHRE_LIVE_ENV`, default the
    /// workspace root's `.env`) into memory only.
    /// `cargo test -p ochre-refine --release live_luna -- --ignored --nocapture`
    #[test]
    #[ignore = "network + OpenAI key"]
    fn live_luna_latency() {
        let path = std::env::var("OCHRE_LIVE_ENV")
            .unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../../.env").into());
        let Some(key) = std::fs::read_to_string(path).ok().and_then(|t| {
            t.lines().find_map(|l| {
                l.strip_prefix("OPENAI_API_KEY=")
                    .map(|v| v.trim().trim_matches('"').to_string())
            })
        }) else {
            eprintln!("no OPENAI_API_KEY in the live env file; skipped");
            return;
        };
        let mut r = OpenAiCompatRefiner::new("openai", "", "")
            .unwrap()
            .with_key(key);
        assert_eq!(r.model(), "gpt-6-luna");
        r.load(&|_| {}).unwrap();
        let ctx = RefineContext {
            mode: "clean".into(),
            ..Default::default()
        };
        let short = "um can you send me the the file when you get a chance";
        let para = "so um i wanted to give everyone a quick update on the project uh we finished the first round of user \
interviews last week and the main thing we heard was that people find the onboarding way too long like they drop off \
before they even see the dashboard so uh what we're thinking is we cut it down to three steps instead of seven and we \
move the integrations stuff to later you know after they've actually seen some value um i'll share the full notes in \
the doc by thursday and if anyone has concerns just let me know before then";
        for (name, text) in [("short", short), ("paragraph", para)] {
            let mut ms = vec![];
            let mut last = String::new();
            for _ in 0..3 {
                let o = crate::refine_text(Some(&r), text, &ctx, Duration::from_secs(10));
                assert!(o.refined, "{name}: {o:?}");
                assert!(!o.text.contains("<think>") && !o.text.contains("dictation>"));
                ms.push(o.ms);
                last = o.text;
            }
            let words = text.split_whitespace().count();
            eprintln!("luna {name} ({words} words): {ms:?} ms\n  {last}");
        }
    }
}
