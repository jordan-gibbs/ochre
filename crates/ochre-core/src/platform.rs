//! Platform contracts: hotkey gestures, text injection, focused-app info (SPEC §6).

use serde::{Deserialize, Serialize};

use crate::Result;

/// What the hotkey layer reports. The gesture state machine (ochre-platform) turns raw key
/// down/up into these, so the orchestrator never sees timing logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Gesture {
    /// Voice key went down: start recording immediately (hold mode, or the first tap of a double).
    Press,
    /// Released after a hold: finish.
    Release,
    /// Second press within the double-tap window: keep recording hands-off.
    Lock,
    /// Single tap while locked: finish.
    Finish,
    /// Finish, but skip refinement (raw modifier held). Sent instead of Release / Finish.
    FinishRaw,
    /// A short tap that turned out not to be a double tap: discard the session silently.
    Abort,
    /// Escape while recording.
    Cancel,
    /// The paste-last chord (Voice key + Down by default, or a standalone combo): type the last
    /// transcript again. Never re-refined, never saved as a new history entry. When the chord
    /// interrupts a take the Voice key just started, an `Abort` comes first.
    PasteLast,
}

pub type GestureFn = Box<dyn Fn(Gesture) + Send + Sync>;

pub trait HotkeyListener: Send {
    /// Install the hook and start delivering gestures. `on_gesture` runs on a dedicated dispatcher
    /// thread (never the hook thread itself); return promptly.
    fn start(&mut self, on_gesture: GestureFn) -> Result<()>;
    fn set_key(&mut self, key: &str) -> Result<()>;
    /// Escape is swallowed only while recording, so it keeps working everywhere else. The
    /// orchestrator calls `set_recording(true)` when any session starts (hotkey, wake word, UI)
    /// and `set_recording(false)` when it ends, is canceled or aborted.
    fn set_recording(&self, recording: bool);
    fn stop(&mut self);
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FocusInfo {
    /// Lowercase exe / bundle / WM_CLASS: "slack", "chrome", "code".
    pub app_name: String,
    pub window_title: String,
    /// Opaque id for the leading-space join rule.
    pub window_id: String,
    /// Windows: target runs elevated and we do not (UIPI blocks synthesized input).
    pub elevated: bool,
}

pub trait Injector: Send + Sync {
    fn focus(&self) -> FocusInfo;
    fn type_text(&self, text: &str) -> Result<()>;
    /// Clipboard fallback with save/restore.
    fn paste_text(&self, text: &str) -> Result<()>;
    fn press_enter(&self) -> Result<()>;
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PermissionIssue {
    /// "microphone" | "accessibility" | "input_monitoring" | "input_group" | ...
    pub name: String,
    /// Human-readable fix, shown in onboarding.
    pub fix: String,
}
