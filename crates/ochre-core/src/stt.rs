//! Speech-to-text contract shared by every local and cloud engine (SPEC §4.1).

use crate::Result;
use crate::events::EngineInfo;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SttOptions {
    /// None = auto / engine default.
    pub language: Option<String>,
    /// Vocabulary to bias toward (dictionary words), where the engine supports it.
    pub vocabulary: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SttResult {
    pub text: String,
    pub duration_ms: u64,
    pub processing_ms: u64,
    pub language: Option<String>,
}

/// Download progress for one file (bytes).
#[derive(Debug, Clone, PartialEq)]
pub struct Progress {
    pub item: String,
    pub done: u64,
    pub total: u64,
}

pub type ProgressFn<'a> = &'a (dyn Fn(Progress) + Send + Sync);

/// (text, stable_chars) for live partials.
pub type PartialSink = Box<dyn FnMut(&str, usize) + Send>;

/// A live recognition stream: audio is sent while the user talks, so release only waits for the
/// provider's final (Soniox real-time: 75-137 ms release -> final, measured).
pub trait SttStream: Send {
    /// Mono f32 at `crate::SAMPLE_RATE`.
    fn send(&mut self, pcm: &[f32]) -> Result<()>;
    fn finish(self: Box<Self>) -> Result<SttResult>;
    fn cancel(self: Box<Self>);
}

pub trait SttEngine: Send + Sync {
    fn info(&self) -> EngineInfo;

    /// Download / verify models (local) or check configuration (cloud). Idempotent. Local engines
    /// must also run one warm-up inference here so the first real dictation is not slow.
    fn load(&mut self, progress: ProgressFn) -> Result<()>;

    /// Mono f32 PCM at `crate::SAMPLE_RATE`, values in [-1, 1]. Blocking. Called from one decode
    /// thread at a time; engines need not be re-entrant.
    fn transcribe(&self, pcm: &[f32], opts: &SttOptions) -> Result<SttResult>;

    /// True when [`SttEngine::stream`] returns a stream for the configured model. Cheap (no
    /// network): the orchestrator checks it on key-down to choose between streaming and the phrase
    /// segmenter before opening anything.
    fn streaming(&self) -> bool {
        false
    }

    /// Engines that can recognize while the user talks return a stream; the orchestrator then
    /// bypasses the phrase segmenter. `None` = batch only. Opening may block on the network
    /// (TLS + WebSocket handshake), so callers open it off the hotkey thread.
    fn stream(
        &self,
        _opts: &SttOptions,
        _on_partial: Option<PartialSink>,
    ) -> Option<Result<Box<dyn SttStream>>> {
        None
    }

    /// Called on key-down: warm connections (TLS, HTTP/2) so the first request after release
    /// isn't cold (OpenAI measured 1.7 s cold vs ~0.8 s warm). Must not block.
    fn prewarm(&self) {}
}
