//! Refinement (SPEC §5): shared prompts, the safety guard, the deterministic normalize scaffold,
//! the local Quill refiner (managed llama-server sidecar) and the cloud LLM refiners.
//!
//! The orchestrator calls [`create`] once, `load()` at startup (keeps the model warm), then
//! [`refine_text`] per dictation, which never fails: any error, timeout or guard rejection returns
//! the raw text.

pub mod anthropic;
pub mod chunk;
pub mod gemini;
pub mod guard;
pub mod http;
pub mod local;
pub mod normalize;
pub mod openai_compat;
pub mod prompts;

#[cfg(test)]
mod testutil;

use std::time::{Duration, Instant};

use ochre_core::config::RefineConfig;
use ochre_core::events::EngineInfo;
use ochre_core::refine::{RefineContext, Refiner};
use ochre_core::{Error, Result};

/// Every provider the settings UI can offer, in display order.
pub fn engines() -> Vec<EngineInfo> {
    let mut out = vec![local::info()];
    out.extend(
        openai_compat::PRESETS
            .iter()
            .filter(|p| p.id != "custom")
            .map(openai_compat::info),
    );
    out.push(anthropic::info());
    out.push(gemini::info());
    out.push(openai_compat::info(openai_compat::preset("custom")));
    out
}

/// Build (not load) the configured refiner. `provider = "off"` (or mode "off") is a config error:
/// the orchestrator should not create a refiner then.
pub fn create(cfg: &RefineConfig) -> Result<Box<dyn Refiner>> {
    match cfg.provider.as_str() {
        "" | "off" => Err(Error::Config("refinement is off".into())),
        _ if cfg.mode == "off" => Err(Error::Config("refinement is off".into())),
        "local" => Ok(Box::new(local::LocalRefiner::new(cfg)?)),
        "anthropic" => Ok(Box::new(anthropic::AnthropicRefiner::new(
            &cfg.model,
            &cfg.base_url,
        ))),
        "gemini" => Ok(Box::new(gemini::GeminiRefiner::new(
            &cfg.model,
            &cfg.base_url,
        ))),
        p if openai_compat::PRESETS.iter().any(|x| x.id == p) => Ok(Box::new(
            openai_compat::OpenAiCompatRefiner::new(p, &cfg.model, &cfg.base_url)?,
        )),
        p => Err(Error::Config(format!("unknown refinement provider {p:?}"))),
    }
}

/// The per-dictation context: mode from config; style from `app_styles` matched against the focused
/// app's process name ("Slack.exe" -> "slack" -> "casual"), exact key first, then substring.
pub fn context_for(
    cfg: &RefineConfig,
    app_name: &str,
    window_title: &str,
    dictionary: &[String],
    language: Option<String>,
) -> RefineContext {
    let app = app_name.to_lowercase();
    let app = app
        .strip_suffix(".exe")
        .or_else(|| app.strip_suffix(".app"))
        .unwrap_or(&app)
        .to_string();
    let style = cfg
        .app_styles
        .get(&app)
        .or_else(|| {
            cfg.app_styles
                .iter()
                .find(|(k, _)| !k.is_empty() && app.contains(k.as_str()))
                .map(|(_, v)| v)
        })
        .cloned()
        .unwrap_or_default();
    let mode = if matches!(cfg.mode.as_str(), "clean" | "polish" | "off") {
        cfg.mode.clone()
    } else {
        "clean".into()
    };
    RefineContext {
        mode,
        app_name: app_name.to_string(),
        window_title: window_title.to_string(),
        style,
        dictionary: dictionary.to_vec(),
        language,
    }
}

/// The configured timeout for a refiner of this kind.
pub fn timeout_for(cfg: &RefineConfig, refiner: &dyn Refiner) -> Duration {
    Duration::from_millis(if refiner.info().kind == "local" {
        cfg.timeout_ms_local
    } else {
        cfg.timeout_ms_cloud
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// What to insert.
    pub text: String,
    /// False -> the raw text was used.
    pub refined: bool,
    /// Why raw was used: a guard rule (`Reject::as_str`) or an error code (`Error::code`).
    pub reason: Option<String>,
    pub ms: u64,
}

/// Refine with every safety net: errors, timeouts and guard rejections all return `raw`. Never
/// fails, so the caller can always insert something.
pub fn refine_text(
    refiner: Option<&dyn Refiner>,
    raw: &str,
    ctx: &RefineContext,
    timeout: Duration,
) -> Outcome {
    let t0 = Instant::now();
    let ms = |t0: Instant| t0.elapsed().as_millis() as u64;
    let Some(refiner) = refiner else {
        return Outcome {
            text: raw.to_string(),
            refined: false,
            reason: None,
            ms: 0,
        };
    };
    if ctx.mode == "off" || raw.trim().is_empty() {
        return Outcome {
            text: raw.to_string(),
            refined: false,
            reason: None,
            ms: 0,
        };
    }
    let out = match refiner.refine(raw, ctx, timeout) {
        Ok(out) => out,
        Err(e) => {
            tracing::warn!("refinement failed ({}): {e}", e.code());
            return Outcome {
                text: raw.to_string(),
                refined: false,
                reason: Some(e.code().to_string()),
                ms: ms(t0),
            };
        }
    };
    let elapsed = ms(t0);
    if elapsed > timeout.as_millis() as u64 {
        return Outcome {
            text: raw.to_string(),
            refined: false,
            reason: Some("timeout".into()),
            ms: elapsed,
        };
    }
    match guard::check(raw, &out, prompts::mode_of(ctx)) {
        Some(rule) => {
            tracing::info!("refinement rejected by guard: {}", rule.as_str());
            Outcome {
                text: raw.to_string(),
                refined: false,
                reason: Some(rule.as_str().to_string()),
                ms: elapsed,
            }
        }
        None => Outcome {
            text: guard::tidy(&out),
            refined: true,
            reason: None,
            ms: elapsed,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ochre_core::stt::ProgressFn;

    struct Fake(Result<String, &'static str>);
    impl Refiner for Fake {
        fn info(&self) -> EngineInfo {
            local::info()
        }
        fn load(&mut self, _: ProgressFn) -> Result<()> {
            Ok(())
        }
        fn refine(&self, _: &str, _: &RefineContext, _: Duration) -> Result<String> {
            self.0.clone().map_err(|m| Error::Network {
                provider: "x".into(),
                message: m.into(),
            })
        }
    }

    #[test]
    fn registry() {
        let ids: Vec<String> = engines().into_iter().map(|e| e.id).collect();
        assert_eq!(
            ids,
            [
                "local",
                "openai",
                "groq",
                "openrouter",
                "cerebras",
                "anthropic",
                "gemini",
                "custom"
            ]
        );
        for p in [
            "local",
            "openai",
            "groq",
            "openrouter",
            "cerebras",
            "anthropic",
            "gemini",
        ] {
            let cfg = RefineConfig {
                provider: p.into(),
                ..Default::default()
            };
            assert_eq!(create(&cfg).unwrap().info().id, p);
        }
        assert!(create(&RefineConfig::default()).is_err()); // provider "off"
        assert!(
            create(&RefineConfig {
                provider: "custom".into(),
                ..Default::default()
            })
            .is_err()
        ); // needs base_url
        assert!(
            create(&RefineConfig {
                provider: "nope".into(),
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn context_styles() {
        let cfg = RefineConfig::default();
        assert_eq!(
            context_for(&cfg, "Slack.exe", "", &[], None).style,
            "casual"
        );
        assert_eq!(
            context_for(&cfg, "OUTLOOK.EXE", "", &[], None).style,
            "formal"
        );
        assert_eq!(
            context_for(&cfg, "Code - Insiders.exe", "", &[], None).style,
            "literal"
        );
        assert_eq!(context_for(&cfg, "notepad.exe", "", &[], None).style, "");
    }

    #[test]
    fn refine_text_falls_back() {
        let ctx = RefineContext {
            mode: "clean".into(),
            ..Default::default()
        };
        let t = Duration::from_secs(1);
        let ok = refine_text(
            Some(&Fake(Ok("Send me the file.".into()))),
            "send me the file",
            &ctx,
            t,
        );
        assert!(ok.refined && ok.text == "Send me the file.");
        let err = refine_text(Some(&Fake(Err("down"))), "send me the file", &ctx, t);
        assert_eq!(
            (err.text.as_str(), err.reason.as_deref()),
            ("send me the file", Some("network"))
        );
        let rej = refine_text(Some(&Fake(Ok("".into()))), "send me the file", &ctx, t);
        assert_eq!(
            (rej.text.as_str(), rej.reason.as_deref()),
            ("send me the file", Some("empty"))
        );
        let none = refine_text(None, "x", &ctx, t);
        assert!(!none.refined && none.reason.is_none());
    }

    /// Live: the local Quill sidecar (downloads llama.cpp + the model on first run) refines a short
    /// dictation under the local timeout. `cargo test -p ochre-refine --release -- --ignored`
    #[test]
    #[ignore = "downloads llama.cpp + Quill, starts llama-server"]
    fn live_local_quill() {
        let cfg = RefineConfig {
            provider: "local".into(),
            ..Default::default()
        };
        let mut r = create(&cfg).unwrap();
        r.load(&|_| {}).unwrap();
        let ctx = context_for(&cfg, "notepad.exe", "", &[], None);
        let o = refine_text(
            Some(r.as_ref()),
            "um can you send me the the file",
            &ctx,
            timeout_for(&cfg, r.as_ref()),
        );
        assert!(o.refined, "{o:?}");
        assert_eq!(o.text, "Can you send me the file?");
    }

    /// Live: every cloud refiner whose key is configured cleans one dictation.
    #[test]
    #[ignore = "network + API keys"]
    fn live_cloud_with_keys() {
        for p in [
            "openai",
            "groq",
            "openrouter",
            "cerebras",
            "anthropic",
            "gemini",
        ] {
            if !ochre_core::secrets::has(p) {
                eprintln!("{p}: no key, skipped");
                continue;
            }
            let cfg = RefineConfig {
                provider: p.into(),
                ..Default::default()
            };
            let mut r = create(&cfg).unwrap();
            r.load(&|_| {}).unwrap_or_else(|e| panic!("{p} load: {e}"));
            let ctx = context_for(&cfg, "", "", &[], None);
            let o = refine_text(
                Some(r.as_ref()),
                "what time does the pharmacy close today",
                &ctx,
                Duration::from_secs(10),
            );
            eprintln!("{p}: {o:?}");
            assert!(o.refined && o.text.ends_with('?'), "{p}: {o:?}");
        }
    }
}
