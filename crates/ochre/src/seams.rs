//! The two audio seams the orchestrator drives. ochre-audio implements them for real; tests use
//! fakes. Keeping them here (not in ochre-core) lets the session logic be tested without a mic.

use std::sync::Arc;

use ochre_core::Result;
use ochre_core::stt::SttResult;

/// Mono f32 at `ochre_core::SAMPLE_RATE`.
pub type Sink = Box<dyn FnMut(&[f32]) + Send>;
pub type LevelFn = Arc<dyn Fn(f32) + Send + Sync>;
/// Hands-free tap: `(absolute index of the block's first sample, block)`, every captured block,
/// on the capture thread. Must not block.
pub type Tap = Box<dyn FnMut(u64, &[f32]) + Send>;

pub trait Mic: Send + Sync {
    /// Start delivering audio to `sink`, beginning `preroll_ms` *before now* (from the warm ring
    /// buffer) so the first syllable is never clipped. Returns immediately.
    fn begin(&self, preroll_ms: u32, sink: Sink, level: LevelFn) -> Result<()>;

    /// Release: flush all pre-release audio into the sink and return at once (<= 5 ms). A short
    /// post-roll (~150-300 ms, longer while VAD hears voice) may keep flowing into the sink
    /// afterwards; the mic then drops the sink. Returns the session's audio duration in ms.
    fn end(&self) -> Result<u64>;

    /// Block until the post-roll is done and the sink has been dropped (or `timeout`). Called on
    /// the session worker, never on the hotkey thread. Default: nothing to wait for.
    fn wait_closed(&self, _timeout: std::time::Duration) {}

    /// Drop the session without flushing.
    fn cancel(&self);

    /// Close the device and open it again (macOS: after the microphone grant arrives).
    fn reopen(&self) {}

    /// A dictation is probably about to start: open a mic that idle release closed.
    fn prime(&self) {}

    /// Hands-free: install (or remove with `None`) an always-on subscriber to every captured
    /// block, sessions or not. The device stays open while one is installed.
    fn set_tap(&self, _tap: Option<Tap>) -> Result<()> {
        Err(ochre_core::Error::Audio(
            "this microphone has no always-on tap".into(),
        ))
    }

    /// Like `begin`, but starting at absolute sample `from` of the tap stream (as far back as
    /// the history reaches): the wake word and the words after it reach the session without a gap.
    fn begin_from(&self, _from: u64, sink: Sink, level: LevelFn) -> Result<()> {
        self.begin(1500, sink, level)
    }
}

pub type TranscribeFn = Arc<dyn Fn(&[f32]) -> Result<SttResult> + Send + Sync>;
/// `(text, stable)` for the HUD's live text: `stable` is a byte offset into `text` (the prefix that
/// will not change). Display only; must be cheap (it runs on the decode thread).
pub type PartialFn = Arc<dyn Fn(&str, usize) + Send + Sync>;
/// One finished phrase's result, in order, on the decode thread. Never a live-text preview.
pub type PhraseFn = Arc<dyn Fn(&SttResult) + Send + Sync>;

pub trait SegmenterFactory: Send + Sync {
    /// One dictation: phrases are cut at pauses and decoded on the decode thread while the user
    /// is still talking, so release only waits on the tail phrase.
    fn start(
        &self,
        transcribe: TranscribeFn,
        on_partial: Option<PartialFn>,
    ) -> Box<dyn SegmentSession>;

    /// `start`, plus `on_phrase` for every finished phrase (hands-free control phrases). Live-text
    /// previews also go through `transcribe`, so this must not be built by wrapping it in a
    /// segmenter that makes previews. Default: for segmenters without previews (test fakes).
    fn start_with_phrases(
        &self,
        transcribe: TranscribeFn,
        on_partial: Option<PartialFn>,
        on_phrase: PhraseFn,
    ) -> Box<dyn SegmentSession> {
        let wrapped: TranscribeFn = Arc::new(move |pcm: &[f32]| {
            let r = transcribe(pcm);
            if let Ok(res) = &r {
                on_phrase(res);
            }
            r
        });
        self.start(wrapped, on_partial)
    }
}

pub trait SegmentSession: Send {
    fn feed(&mut self, pcm: &[f32]);
    /// Called right after `Mic::end()`: cut at the last quiet point and start decoding everything
    /// fed so far, so the tail decode overlaps the post-roll. Default: no-op.
    fn release(&mut self) {}
    /// After the post-roll: decode whatever remains (skipped when it is silent) and return every
    /// phrase result in order.
    fn finish(self: Box<Self>) -> Result<Vec<SttResult>>;
    fn cancel(self: Box<Self>);
}
