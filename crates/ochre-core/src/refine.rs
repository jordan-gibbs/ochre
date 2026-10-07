//! Refinement contract (SPEC §5.1). Refiners clean dictated text; they never answer it.

use std::time::Duration;

use crate::Result;
use crate::events::EngineInfo;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct RefineContext {
    /// "clean" | "polish"
    pub mode: String,
    /// Lowercase exe / bundle name of the focused app.
    pub app_name: String,
    pub window_title: String,
    /// Per-app style hint: "casual" | "formal" | "literal" | "".
    pub style: String,
    pub dictionary: Vec<String>,
    pub language: Option<String>,
}

pub trait Refiner: Send + Sync {
    fn info(&self) -> EngineInfo;

    /// Start/connect (local server, key check). Idempotent; must leave the model warm.
    fn load(&mut self, progress: crate::stt::ProgressFn) -> Result<()>;

    /// Returns refined text or an error; the core applies the safety guard and falls back to the
    /// raw text on any error or timeout.
    fn refine(&self, text: &str, ctx: &RefineContext, timeout: Duration) -> Result<String>;

    /// Called on key-down so a cloud connection is warm by the time refinement runs. Must not block.
    fn prewarm(&self) {}

    /// The bundled local model this refiner runs, once loaded ("ochre-refine-2b"); None for cloud
    /// and external servers. Per-size behaviour ("auto" settings) keys off it.
    fn model_name(&self) -> Option<String> {
        None
    }
}
