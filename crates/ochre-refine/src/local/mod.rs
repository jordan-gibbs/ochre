//! The `local` refiner: Quill (or any GGUF) on a managed llama-server sidecar (SPEC §5.2,
//! docs/refinement.md §3).
//!
//! Why a sidecar and not in-process `llama-cpp-2`: per-request loopback HTTP+JSON costs 2-6 ms
//! (measured, both here and in the Python bench), cold start is model load either way, and the
//! sidecar lets us download upstream's prebuilt CUDA/Vulkan/Metal/CPU builds on demand (the
//! 577 MB CUDA runtime never ships in our installer), isolates GPU-driver crashes, and gives us
//! n-gram speculative decoding, slots and priority for free.
//!
//! Prompting: the shared `prompts::system_prompt` + `<dictation>` user message rendered as raw
//! ChatML with the assistant turn pre-seeded with an empty think block, sent to `/completion`,
//! greedy, never `--jinja`. (Quill's own card prompt answers dictated questions; see
//! docs/refinement.md §2.4.) After each request the system-prompt prefix is re-primed in the
//! background so the next dictation only evaluates its own tokens.
//!
//! Quill 0.8B is verbatim-only, so output always goes through `normalize`.
//!
//! The default model is "auto": Ochre Refine 4B / 2B / 0.8B by GPU memory, or the same-size Quill
//! if that download fails (`auto.rs`).
//!
//! With `refine.base_url` set, this instead talks to an existing OpenAI-compatible server
//! (Ollama, LM Studio, your own llama-server) through `/chat/completions`.

pub mod auto;
pub mod install;
pub mod server;

use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use ochre_core::config::RefineConfig;
use ochre_core::events::EngineInfo;
use ochre_core::refine::{RefineContext, Refiner};
use ochre_core::stt::ProgressFn;
use ochre_core::{Error, Result};
use regex::Regex;

use crate::openai_compat::OpenAiCompatRefiner;
use crate::{guard, normalize, prompts};
use server::{LlamaServer, ServerOptions};

/// The catalog entry. `note` starts with what "auto" picks on this machine (default accelerator):
/// `auto: <model id> (<hardware>)` — the settings UI reads the id from it.
pub fn info() -> EngineInfo {
    let (pick, hw) = auto::resolve("auto");
    info_with(pick.name, &hw.describe())
}

pub fn info_with(auto_pick: &str, hardware: &str) -> EngineInfo {
    let mut models = vec![auto::AUTO.to_string()];
    models.extend(install::MODELS.iter().map(|m| m.name.to_string()));
    EngineInfo {
        id: "local".into(),
        label: "Local (Ochre Refine / Quill on llama.cpp)".into(),
        kind: "local".into(),
        models,
        default_model: install::DEFAULT_MODEL.into(),
        needs_key: false,
        note: format!(
            "auto: {auto_pick} ({hardware}). Ochre Refine 4B 2.7 GB / 2B 1.2 GB (GPU), \
             0.8B 529 MB (CPU), fine-tuned for dictation; \
             Quill 4B / 2B / 0.8B as alternatives"
        ),
        languages: "en".into(),
    }
}

/// llama.cpp's own split of one request.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LlamaTimings {
    pub prompt_n: u64,
    pub cache_n: u64,
    pub prompt_ms: f64,
    pub predicted_n: u64,
    pub predicted_ms: f64,
}

#[derive(Debug, Clone, Default)]
pub struct LocalOutput {
    /// Cleaned and normalized text (what `refine` returns).
    pub text: String,
    /// The model's raw `content`.
    pub raw: String,
    pub timings: LlamaTimings,
    pub wall_ms: f64,
}

/// `<|im_start|>system\n{system}<|im_end|>\n<|im_start|>`: ends on a special token, so it tokenizes
/// identically alone and inside the full prompt.
pub fn priming_prefix(system: &str) -> String {
    format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>")
}

pub fn build_prompt(text: &str, ctx: &RefineContext) -> String {
    prompts::chatml(&prompts::system_prompt(ctx), &prompts::user_message(text))
}

/// `max(32, chars / 3.2 * 2 + 16)`: the guard rejects > 2x anyway.
pub fn n_predict(text: &str) -> u32 {
    ((text.chars().count() as f64 / 3.2 * 2.0) as u32 + 16).max(32)
}

static LEFTOVER: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"<\|im_(?:start|end)\|>|<\|endoftext\|>").unwrap());

/// Strip special tokens and any think block that slipped through (should not happen with the
/// pre-seeded empty block, but a leaked chain of thought must never be typed).
pub fn clean_output(content: &str) -> String {
    let out = LEFTOVER.replace_all(content, "");
    let out = match out.split_once("</think>") {
        Some((_, after)) => after.to_string(),
        None => out.into_owned(),
    };
    guard::tidy(&out)
}

pub struct LocalRefiner {
    /// As configured ("auto" or a model id / .gguf path).
    model: String,
    /// What `model` resolved to at load ("auto" -> e.g. "ochre-refine-4b").
    resolved: Option<String>,
    accel: String,
    threads: u32,
    external: Option<OpenAiCompatRefiner>,
    server: Option<Arc<LlamaServer>>,
    primer: Mutex<Option<Sender<String>>>,
}

impl LocalRefiner {
    pub fn new(cfg: &RefineConfig) -> Result<Self> {
        let external = if cfg.base_url.is_empty() {
            None
        } else {
            // "auto" means our own models; an external server gets its own default.
            let model = if auto::is_auto(&cfg.model) {
                "default"
            } else {
                &cfg.model
            };
            Some(OpenAiCompatRefiner::new("custom", model, &cfg.base_url)?)
        };
        Ok(LocalRefiner {
            model: if cfg.model.is_empty() {
                install::DEFAULT_MODEL.into()
            } else {
                cfg.model.clone()
            },
            accel: if cfg.local_accel.is_empty() {
                "auto".into()
            } else {
                cfg.local_accel.clone()
            },
            threads: cfg.local_threads,
            external,
            resolved: None,
            server: None,
            primer: Mutex::new(None),
        })
    }

    pub fn server(&self) -> Option<&Arc<LlamaServer>> {
        self.server.as_ref()
    }

    fn start(
        &self,
        model: &std::path::Path,
        accel: &str,
        progress: ProgressFn,
    ) -> Result<Arc<LlamaServer>> {
        let (exe, got) = install::ensure_binary(accel, progress)?;
        let server = Arc::new(LlamaServer::new(
            &exe,
            model,
            &got,
            ServerOptions::tuned(&got, self.threads),
        ));
        server.start(Duration::from_secs(90))?;
        Ok(server)
    }

    /// One priming worker per server: coalesces requests, runs off the latency path.
    fn spawn_primer(server: Arc<LlamaServer>) -> Sender<String> {
        let (tx, rx) = channel::<String>();
        std::thread::Builder::new()
            .name("llama-prime".into())
            .spawn(move || {
                while let Ok(mut prefix) = rx.recv() {
                    while let Ok(newer) = rx.try_recv() {
                        prefix = newer;
                    }
                    if server.alive()
                        && let Err(e) = server.prime(&prefix)
                    {
                        tracing::debug!("prime failed: {e}");
                    }
                }
            })
            .expect("spawn primer");
        tx
    }

    /// Refine and return llama.cpp's timing split too (used by the bench).
    pub fn refine_detailed(
        &self,
        text: &str,
        ctx: &RefineContext,
        timeout: Duration,
    ) -> Result<LocalOutput> {
        let server = self
            .server
            .as_ref()
            .ok_or_else(|| Error::Model("local refiner not loaded".into()))?;
        if !server.ensure_running() {
            return Err(Error::Network {
                provider: "local".into(),
                message: "llama-server is restarting".into(),
            });
        }
        let t0 = Instant::now();
        let data = server.completion(&build_prompt(text, ctx), n_predict(text), timeout);
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        if let Some(tx) = self.primer.lock().unwrap().as_ref() {
            let _ = tx.send(priming_prefix(&prompts::system_prompt(ctx)));
        }
        let data = data.inspect_err(|_| {
            server.ensure_running();
        })?;
        let raw = data["content"].as_str().unwrap_or("").to_string();
        let t = &data["timings"];
        let timings = LlamaTimings {
            prompt_n: t["prompt_n"].as_u64().unwrap_or(0),
            cache_n: t["cache_n"].as_u64().unwrap_or(0),
            prompt_ms: t["prompt_ms"].as_f64().unwrap_or(0.0),
            predicted_n: t["predicted_n"].as_u64().unwrap_or(0),
            predicted_ms: t["predicted_ms"].as_f64().unwrap_or(0.0),
        };
        Ok(LocalOutput {
            text: normalize::normalize(&clean_output(&raw)),
            raw,
            timings,
            wall_ms,
        })
    }

    /// Prime and wait (bench / warm-up), instead of the background worker.
    pub fn prime_now(&self, ctx: &RefineContext) -> Result<()> {
        match &self.server {
            Some(s) => s.prime(&priming_prefix(&prompts::system_prompt(ctx))),
            None => Ok(()),
        }
    }
}

impl Refiner for LocalRefiner {
    fn info(&self) -> EngineInfo {
        let mut i = info();
        i.default_model = self.model.clone();
        if let Some(s) = &self.server {
            let name = self.resolved.as_deref().unwrap_or(&self.model);
            i.note = if auto::is_auto(&self.model) {
                format!(
                    "auto: {name} on llama.cpp ({}, {} threads)",
                    s.accel, s.options.threads
                )
            } else {
                format!(
                    "{name} on llama.cpp ({}, {} threads)",
                    s.accel, s.options.threads
                )
            };
        }
        i
    }

    /// Download (first run) and start the server, then warm it: one tiny refinement (CUDA graphs,
    /// first-request setup) and a prime of the default-context prefix. Blocking: ~1.5-4 s when
    /// everything is on disk, minutes on first run. Idempotent.
    fn load(&mut self, progress: ProgressFn) -> Result<()> {
        if let Some(ext) = self.external.as_mut() {
            return ext.load(progress);
        }
        if self.server.as_ref().is_some_and(|s| s.alive()) {
            return Ok(());
        }
        let model = if auto::is_auto(&self.model) {
            let (path, m) = auto::ensure(&self.accel, progress)?;
            self.resolved = Some(m.name.to_string());
            path
        } else {
            self.resolved = Some(self.model.clone());
            install::ensure_model(&self.model, progress)?
        };
        let accel = install::pick_accel(&self.accel);
        let server = match self.start(&model, &accel, progress) {
            Ok(s) => s,
            Err(e) if accel != "cpu" => {
                tracing::warn!(
                    "accelerated llama-server ({accel}) failed ({e}); falling back to the CPU build"
                );
                self.start(&model, "cpu", progress)?
            }
            Err(e) => return Err(e),
        };
        *self.primer.lock().unwrap() = Some(Self::spawn_primer(server.clone()));
        self.server = Some(server);
        let ctx = RefineContext::default();
        if let Err(e) =
            self.refine_detailed("okay so this is a warm up", &ctx, Duration::from_secs(30))
        {
            tracing::warn!("local refiner warm-up failed: {e}");
        }
        self.prime_now(&ctx)?;
        Ok(())
    }

    fn model_name(&self) -> Option<String> {
        if self.external.is_some() {
            None
        } else {
            self.resolved.clone()
        }
    }

    fn refine(&self, text: &str, ctx: &RefineContext, timeout: Duration) -> Result<String> {
        if text.trim().is_empty() {
            return Ok(text.to_string());
        }
        if let Some(ext) = &self.external {
            return Ok(normalize::normalize(&clean_output(
                &ext.refine(text, ctx, timeout)?,
            )));
        }
        Ok(self.refine_detailed(text, ctx, timeout)?.text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_and_prefix_line_up() {
        let ctx = RefineContext::default();
        let p = build_prompt("um hi", &ctx);
        assert!(p.starts_with(&priming_prefix(&prompts::system_prompt(&ctx))));
        assert!(p.ends_with("<|im_start|>user\n<dictation>\num hi\n</dictation><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert_eq!(n_predict("hi"), 32);
        assert_eq!(n_predict(&"a".repeat(320)), 216);
    }

    #[test]
    fn catalog_offers_auto_first() {
        let i = info_with("ochre-refine-4b", "NVIDIA GPU, 16 GB");
        assert_eq!(i.default_model, "auto");
        assert_eq!(i.models[0], "auto");
        for m in install::MODELS {
            assert!(i.models.iter().any(|x| x == m.name), "{}", m.name);
        }
        assert!(
            i.note
                .starts_with("auto: ochre-refine-4b (NVIDIA GPU, 16 GB).")
        );
        assert!(!i.note.contains("  "), "{}", i.note);
        let r = LocalRefiner::new(&RefineConfig::default()).unwrap();
        assert_eq!(r.model, "auto");
    }

    #[test]
    fn cleans_leaks() {
        assert_eq!(
            clean_output("reasoning here</think>\n\nSend it.<|im_end|>"),
            "Send it."
        );
        assert_eq!(clean_output(" Hello there. "), "Hello there.");
    }

    #[test]
    fn external_base_url_uses_chat_completions() {
        use crate::testutil::{MockServer, Reply};
        let srv = MockServer::start(vec![Reply::json(
            200,
            serde_json::json!({"choices": [{"message": {"content": "Meet at three thirty pm."}}]}),
        )]);
        let cfg = RefineConfig {
            provider: "local".into(),
            base_url: srv.url("/v1"),
            model: "quill".into(),
            ..Default::default()
        };
        let r = LocalRefiner::new(&cfg).unwrap();
        let out = r
            .refine(
                "meet at three thirty pm",
                &RefineContext::default(),
                Duration::from_secs(5),
            )
            .unwrap();
        assert_eq!(out, "Meet at 3:30 PM.");
        assert_eq!(srv.requests()[0].path, "/v1/chat/completions");
    }
}
