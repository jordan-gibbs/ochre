//! The core <-> UI protocol (SPEC §7.1), single source of truth for message shapes.
//!
//! Core -> UI: `Event`, serialized as `{"event": "<snake_case>", ...fields}` and delivered to the
//! Tauri webviews as the `ochre://event` event. UI -> core: `Command`, `{"op": "<snake_case>", ...}`,
//! received through the `ochre_command` Tauri command. The CLI (`ochre toggle`) reaches a running
//! instance through the single-instance plugin, which forwards its argv as a Command.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    /// Models loading or downloading; `Download` events carry progress.
    Loading,
    /// Ready. Hands-free may be armed (see `handsfree_armed`).
    Idle,
    /// Hold-to-talk.
    Recording,
    /// Double-tapped: recording until a single tap.
    Locked,
    /// Wake-word session; ends on "transcribe stop/send/cancel".
    Handsfree,
    Transcribing,
    Refining,
    Inserting,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trigger {
    Hotkey,
    Wake,
    Ui,
    Tray,
    Cli,
}

/// Per-stage latency of one session, in milliseconds. `release_to_insert_ms` is the number the
/// user feels; everything else explains it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Timings {
    pub audio_ms: u64,
    /// Release -> last phrase decoded (only the tail phrase should be on this path).
    pub stt_tail_ms: u64,
    /// Total decode work across all phrases (most of it overlapped with speech).
    pub stt_total_ms: u64,
    pub refine_ms: u64,
    pub inject_ms: u64,
    pub release_to_insert_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EngineInfo {
    pub id: String,
    pub label: String,
    /// "local" | "cloud"
    pub kind: String,
    pub models: Vec<String>,
    pub default_model: String,
    pub needs_key: bool,
    /// Short human hint: size for local engines, approximate price for cloud ones.
    pub note: String,
    pub languages: String,
}

/// One stage of a connector: an engine/provider id plus a model ("" = the engine's default).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StageChoice {
    /// `stt.engine` or `refine.provider`.
    pub id: String,
    pub model: String,
    /// Refinement mode ("clean" | "polish"); empty for speech-to-text.
    #[serde(default)]
    pub mode: String,
}

/// A cloud connector: one provider, one API key, both stages configured with fast defaults.
/// Choosing one in settings writes `stt.{engine,model}` and `refine.{provider,model,mode}` through
/// `set_config`; the catalog itself lives in `ochre::connectors`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConnectorInfo {
    pub id: String,
    pub label: String,
    /// The `secrets` provider the one key is saved under ("" = no key, e.g. local only).
    pub key_provider: String,
    /// Further keys a mixed connector needs (e.g. `soniox+openai` also needs `openai`).
    #[serde(default)]
    pub extra_keys: Vec<String>,
    pub stt: StageChoice,
    pub refine: StageChoice,
    pub notes: String,
    /// Approximate USD per hour of dictated audio (speech-to-text + refinement), from list prices
    /// and ~400 dictations of ~600 prompt / ~60 output tokens per hour.
    pub est_cost_per_hour: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Hello {
        version: String,
        platform: String,
    },
    State {
        state: State,
        trigger: Option<Trigger>,
        handsfree_armed: bool,
        detail: String,
    },
    /// 0..1 normalized mic level, ~15/s while recording.
    Level {
        rms: f32,
    },
    /// Live text for the HUD while the user talks (display only, never inserted; hesitations
    /// already stripped). The first `stable_chars` UTF-16 code units (a JS string index) will
    /// not change. At most ~30 per second; a new session starts from scratch.
    Partial {
        text: String,
        stable_chars: usize,
    },
    Result {
        id: String,
        raw: String,
        text: String,
        inserted: bool,
        refined: bool,
        timings: Timings,
    },
    Error {
        message: String,
        code: String,
    },
    Notice {
        message: String,
    },
    /// Model / binary download progress in bytes.
    Download {
        item: String,
        done: u64,
        total: u64,
    },
    Config {
        config: serde_json::Value,
        secrets: std::collections::BTreeMap<String, bool>,
    },
    History {
        items: Vec<crate::history::Entry>,
    },
    TestResult {
        stage: String,
        provider: String,
        ok: bool,
        message: String,
        ms: u64,
    },
    Engines {
        stt: Vec<EngineInfo>,
        refine: Vec<EngineInfo>,
        /// engine id -> models of it already on disk (local engines only).
        #[serde(default)]
        installed: std::collections::BTreeMap<String, Vec<String>>,
    },
    /// The cloud connector catalog (settings: "Cloud connector" picker). Sent once at startup,
    /// next to `Engines`.
    Connectors {
        items: Vec<ConnectorInfo>,
    },
    Permissions {
        missing: Vec<crate::platform::PermissionIssue>,
    },
    /// Input devices for the microphone picker; `default` is the system default's name.
    Devices {
        items: Vec<String>,
        default: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Command {
    Start,
    Stop,
    Cancel,
    Toggle,
    GetConfig,
    SetConfig {
        patch: serde_json::Value,
    },
    SetSecret {
        provider: String,
        key: Option<String>,
    },
    TestProvider {
        stage: String,
        provider: String,
    },
    DownloadModel {
        stage: String,
        name: String,
    },
    HistoryQuery {
        q: String,
        limit: usize,
    },
    InsertText {
        text: String,
    },
    SetHandsfree {
        enabled: bool,
    },
    TrainWake {
        word: String,
    },
    OpenSettings,
    HistoryClear,
    CheckPermissions,
    OpenPermissionSettings {
        name: String,
    },
    ListDevices,
    /// macOS: open the system Mic Mode menu (Standard / Voice Isolation / Wide Spectrum).
    ShowMicModes,
    /// The settings key-capture field is active: the hook must pass the Voice key through.
    HotkeyCapture {
        active: bool,
    },
    CancelDownload {
        item: String,
    },
    /// Type the most recent transcript again into the focused field (tray, CLI `paste-last`, or
    /// the paste-last chord). No refinement, no new history row; the clipboard is left as it was.
    PasteLast,
    Quit,
}

type Listener = Box<dyn Fn(&Event) + Send + Sync>;

/// Thread-safe fan-out of core events. Listeners must be cheap (the Tauri emitter, a logger);
/// a slow listener would delay the pipeline, so heavy work belongs on the listener's own thread.
#[derive(Clone, Default)]
pub struct Bus {
    listeners: Arc<Mutex<Vec<(u64, Listener)>>>,
    next: Arc<std::sync::atomic::AtomicU64>,
    last_state: Arc<Mutex<Option<Event>>>,
}

impl Bus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn subscribe(&self, f: impl Fn(&Event) + Send + Sync + 'static) -> u64 {
        let id = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.listeners.lock().unwrap().push((id, Box::new(f)));
        id
    }

    pub fn unsubscribe(&self, id: u64) {
        self.listeners.lock().unwrap().retain(|(i, _)| *i != id);
    }

    pub fn emit(&self, event: Event) {
        if matches!(event, Event::State { .. }) {
            *self.last_state.lock().unwrap() = Some(event.clone());
        }
        for (_, f) in self.listeners.lock().unwrap().iter() {
            f(&event);
        }
    }

    /// Replayed to a UI that connects late (e.g. the settings window opening).
    pub fn last_state(&self) -> Option<Event> {
        self.last_state.lock().unwrap().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_format() {
        let e = Event::State {
            state: State::Recording,
            trigger: Some(Trigger::Hotkey),
            handsfree_armed: false,
            detail: String::new(),
        };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["event"], "state");
        assert_eq!(v["state"], "recording");
        assert_eq!(v["trigger"], "hotkey");
        let c: Command =
            serde_json::from_str(r#"{"op":"set_config","patch":{"stt":{"engine":"groq"}}}"#)
                .unwrap();
        assert!(matches!(c, Command::SetConfig { .. }));
        assert!(serde_json::from_str::<Command>(r#"{"op":"rm_rf"}"#).is_err());
    }

    #[test]
    fn bus_fanout_and_last_state() {
        let bus = Bus::new();
        let seen = Arc::new(Mutex::new(0));
        let s = seen.clone();
        let id = bus.subscribe(move |_| *s.lock().unwrap() += 1);
        bus.emit(Event::Notice {
            message: "x".into(),
        });
        bus.emit(Event::State {
            state: State::Idle,
            trigger: None,
            handsfree_armed: false,
            detail: String::new(),
        });
        assert_eq!(*seen.lock().unwrap(), 2);
        assert!(bus.last_state().is_some());
        bus.unsubscribe(id);
        bus.emit(Event::Notice {
            message: "y".into(),
        });
        assert_eq!(*seen.lock().unwrap(), 2);
    }
}
