//! Audio for Ochre (SPEC §3, §3.1):
//!
//! - [`capture`]: cpal input with an always-warm stream, a lock-free ring buffer, pre-roll, one
//!   high-quality resample to 16 kHz off the callback thread, HUD levels, device-loss recovery and
//!   the session cap.
//! - [`segmenter`]: phrase cutting during capture (short phrases cut at pauses) and the single
//!   decode thread that owns the STT engine.
//! - [`vad`]: Silero VAD (ONNX) for hands-free gating and the silence guard.
//! - [`earcons`]: short start/stop/cancel/error tones, fire-and-forget.
//! - [`resample`]: the resampler, also for WAV files.
//! - `voice_mac` (macOS): the microphone through Apple's voice processing, and the system Mic Mode
//!   menu ([`show_mic_modes`], [`mic_mode`]).

pub mod capture;
#[cfg(target_os = "macos")]
mod devices_mac;
pub mod earcons;
pub mod priority;
pub mod resample;
pub mod segmenter;
pub mod vad;
#[cfg(target_os = "macos")]
mod voice_mac;

pub use capture::{
    Capture, CaptureEvent, CaptureOptions, CaptureSession, InputDevice, Tap, WarmMic, list_devices,
};
pub use earcons::{Earcon, Earcons};
pub use segmenter::{
    CutConfig, DecodeThread, PhraseCutter, PhraseSegmenter, Segmenter, SegmenterOptions,
    SessionTranscript, Transcriber,
};
pub use vad::SileroVad;
#[cfg(target_os = "macos")]
pub use voice_mac::{mic_mode, show_mic_modes};

#[cfg(test)]
mod tests {
    /// The orchestrator's seams (`Mic: Send + Sync`, `SegmenterFactory: Send + Sync`,
    /// `SegmentSession: Send`) are implemented over these types.
    #[test]
    fn seam_types_are_thread_safe() {
        fn send_sync<T: Send + Sync>() {}
        fn send<T: Send>() {}
        send_sync::<super::WarmMic>();
        send_sync::<super::PhraseSegmenter>();
        send::<super::Segmenter>();
    }
}
