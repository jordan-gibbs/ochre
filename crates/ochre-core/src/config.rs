//! Typed config stored as TOML (SPEC §8). Every field has a default, so a partial or older file
//! always loads; unknown top-level keys are kept and written back.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{Error, Result, paths};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HotkeyConfig {
    /// "right_alt" (Right Option on macOS), "right_ctrl", "caps_lock", "f13", "ctrl+shift+space", ...
    pub key: String,
    /// A second press within this window locks recording (double-tap mode).
    pub double_tap_ms: u64,
    /// A press shorter than this counts as a tap, not a hold.
    pub hold_min_ms: u64,
    /// Held when finishing: insert raw text, skip refinement for this one.
    pub raw_modifier: String,
    /// Paste the last transcript again. A plain key name ("down") means *Voice key + that key*
    /// (hold the Voice key, press Down); a chord ("ctrl+alt+v") is a standalone shortcut;
    /// "" turns it off.
    pub paste_last: String,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        Self {
            key: "right_alt".into(),
            double_tap_ms: 350,
            hold_min_ms: 250,
            raw_modifier: "shift".into(),
            paste_last: "down".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// Input device name; None = system default.
    pub device: Option<String>,
    pub max_session_s: u64,
    pub earcons: bool,
    /// Keep the input stream open while idle so recording starts with zero device-open latency
    /// and a pre-roll covers the first syllable. Costs a mic-in-use indicator on some OSes.
    /// Hands-free keeps the stream open while it is enabled, whatever this says.
    pub warm_mic: bool,
    /// macOS: capture through Apple's voice processing (echo cancellation, noise suppression and
    /// the system Mic Mode menu, whose Voice Isolation removes other people's voices). Applies to
    /// the system default microphone; ignored on other OSes.
    pub voice_processing: bool,
    /// macOS: with no device chosen, record from the built-in mic instead of a Bluetooth headset's
    /// (capturing the headset mic drops its audio to call quality). Ignored on other OSes.
    pub avoid_bluetooth_mic: bool,
    /// Close the warm microphone after this many seconds without a dictation, so the OS mic
    /// indicator goes away (0 = keep it open; hands-free keeps it open regardless). Default 300 on
    /// macOS, 0 elsewhere.
    pub warm_idle_release_s: u64,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            device: None,
            max_session_s: 600,
            earcons: true,
            warm_mic: true,
            voice_processing: true,
            avoid_bluetooth_mic: true,
            warm_idle_release_s: if cfg!(target_os = "macos") { 300 } else { 0 },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SttConfig {
    /// Local engine id or cloud provider id (see ochre-stt registry).
    pub engine: String,
    /// "" = the engine's default model.
    pub model: String,
    /// None = auto / engine default.
    pub language: Option<String>,
    /// "auto" | "cpu" | "cuda" | "directml" | "coreml"
    pub device: String,
    pub cloud_timeout_ms: u64,
    /// Cloud failure -> local engine, when its model is already downloaded.
    pub fallback_local: bool,
}

impl Default for SttConfig {
    fn default() -> Self {
        Self {
            engine: "parakeet".into(),
            model: String::new(),
            language: None,
            device: "auto".into(),
            cloud_timeout_ms: 8000,
            fallback_local: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RefineConfig {
    /// "off" | "local" | "openai" | "groq" | "openrouter" | "anthropic" | "gemini" | "custom"
    pub provider: String,
    /// "clean" | "polish"
    pub mode: String,
    pub model: String,
    /// Custom / external OpenAI-compatible server (Ollama, LM Studio, ...).
    pub base_url: String,
    pub timeout_ms_local: u64,
    pub timeout_ms_cloud: u64,
    /// Local llama-server accelerator: "auto" | "cuda" | "vulkan" | "metal" | "cpu".
    pub local_accel: String,
    /// CPU threads for local refinement; 0 = all logical cores (measured fastest).
    pub local_threads: u32,
    /// app name (lowercase exe / bundle) -> style hint
    pub app_styles: BTreeMap<String, String>,
    /// Refine long dictations in sentence/paragraph pieces (docs/refine-chunking.md):
    /// "auto" (the local model sizes where it measurably helps) | "on" | "off".
    pub chunk_long: String,
    /// Dictations shorter than this many words are always refined in one piece.
    pub chunk_min_words: usize,
    /// Saying the same thing again within `redictation_window_s` after Ochre changed it means the
    /// cleanup got it wrong: the repeat is typed as heard (hesitations and dictionary still apply).
    pub redictation_raw: bool,
    pub redictation_window_s: f64,
}

impl Default for RefineConfig {
    fn default() -> Self {
        let app_styles = [
            ("slack", "casual"),
            ("discord", "casual"),
            ("outlook", "formal"),
            ("mail", "formal"),
            ("code", "literal"),
            ("windowsterminal", "literal"),
            ("terminal", "literal"),
            ("iterm2", "literal"),
        ]
        .into_iter()
        .map(|(a, s)| (a.to_string(), s.to_string()))
        .collect();
        Self {
            provider: "off".into(),
            mode: "clean".into(),
            model: String::new(),
            base_url: String::new(),
            timeout_ms_local: 1500,
            timeout_ms_cloud: 3000,
            local_accel: "auto".into(),
            local_threads: 0,
            app_styles,
            chunk_long: "auto".into(),
            chunk_min_words: 80,
            redictation_raw: true,
            redictation_window_s: 60.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InjectConfig {
    /// "type" (synthesized Unicode keystrokes) | "paste" (clipboard, restored afterwards)
    pub method: String,
    pub paste_over_chars: usize,
    pub join_window_s: f64,
    pub trailing_space: bool,
}

impl Default for InjectConfig {
    fn default() -> Self {
        Self {
            method: "type".into(),
            paste_over_chars: 200,
            join_window_s: 20.0,
            trailing_space: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HandsFreeConfig {
    pub enabled: bool,
    pub phrase: String,
    /// "" = bundled model for the phrase
    pub model: String,
    pub threshold: f32,
    pub preroll_s: f32,
    pub idle_timeout_s: f32,
    pub pause_on_calls: bool,
}

impl Default for HandsFreeConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            phrase: "transcribe".into(),
            model: String::new(),
            threshold: 0.45, // transcribe.onnx: 0.37 false wakes/h, 95% streamed recall (assets/wake/transcribe.json)
            preroll_s: 1.5,
            idle_timeout_s: 45.0,
            pause_on_calls: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DictionaryConfig {
    /// Preferred spellings / vocabulary (also used as STT bias where supported).
    pub words: Vec<String>,
    /// Whole-phrase replacements: said -> written.
    pub replacements: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    /// "light" (default) | "dark" | "system" (follow the OS). The tray art ignores this: it
    /// follows the taskbar's own theme for contrast.
    pub theme: String,
    pub show_partials: bool,
    pub start_at_login: bool,
    pub onboarded: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: "light".into(),
            show_partials: true,
            start_at_login: false,
            onboarded: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HistoryConfig {
    pub enabled: bool,
    pub keep_days: u32,
}

impl Default for HistoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            keep_days: 90,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugConfig {
    /// Writes every session's audio to data_dir/debug-audio. Off by default; never on in release defaults.
    pub keep_audio: bool,
    /// Logs a per-stage latency breakdown for every session (see ochre::timing).
    pub log_timings: bool,
    pub log_level: String,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            keep_audio: false,
            log_timings: true,
            log_level: "info".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub hotkey: HotkeyConfig,
    pub audio: AudioConfig,
    pub stt: SttConfig,
    pub refine: RefineConfig,
    pub inject: InjectConfig,
    pub handsfree: HandsFreeConfig,
    pub dictionary: DictionaryConfig,
    pub ui: UiConfig,
    pub history: HistoryConfig,
    pub debug: DebugConfig,
    /// Unknown top-level keys, round-tripped untouched.
    #[serde(flatten)]
    pub extra: toml::Table,
}

impl Config {
    pub fn load() -> Result<Self> {
        Self::load_from(&paths::config_path())
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            // Editors on Windows (PowerShell's `Set-Content -Encoding utf8`) write a BOM.
            Ok(text) => toml::from_str(text.strip_prefix('\u{feff}').unwrap_or(&text))
                .map_err(|e| Error::Config(format!("{}: {e}", path.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(&paths::config_path())
    }

    /// Atomic write (temp file + rename) so a crash never leaves a half-written config.
    pub fn save_to(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let text = toml::to_string_pretty(self).map_err(|e| Error::Config(e.to_string()))?;
        let tmp = path.with_extension("toml.tmp");
        std::fs::write(&tmp, text)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Deep-merge a partial JSON object (as sent by the UI) and return the new config.
    pub fn patched(&self, patch: &serde_json::Value) -> Result<Self> {
        let mut base = serde_json::to_value(self).map_err(|e| Error::Config(e.to_string()))?;
        merge(&mut base, patch);
        serde_json::from_value(base).map_err(|e| Error::Config(e.to_string()))
    }
}

fn merge(base: &mut serde_json::Value, patch: &serde_json::Value) {
    match (base, patch) {
        (serde_json::Value::Object(b), serde_json::Value::Object(p)) => {
            for (k, v) in p {
                match b.get_mut(k) {
                    Some(slot) if slot.is_object() && v.is_object() => merge(slot, v),
                    _ => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, p) => *b = p.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_file_loads_with_defaults_and_keeps_unknown_keys() {
        let dir = std::env::temp_dir().join(format!("ochre-cfg-{}", uuid::Uuid::new_v4()));
        let path = dir.join("config.toml");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            &path,
            "[stt]\nengine = \"groq\"\nbogus = 1\n\n[future]\nx = 1\n",
        )
        .unwrap();
        let cfg = Config::load_from(&path).unwrap();
        assert_eq!(cfg.stt.engine, "groq");
        assert_eq!(cfg.stt.cloud_timeout_ms, 8000);
        assert!(cfg.extra.contains_key("future"));
        cfg.save_to(&path).unwrap();
        let again = Config::load_from(&path).unwrap();
        assert_eq!(again, cfg);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn patch_is_deep() {
        let cfg = Config::default();
        let new = cfg
            .patched(&serde_json::json!({"refine": {"provider": "local"}, "dictionary": {"words": ["Ochre"]}}))
            .unwrap();
        assert_eq!(new.refine.provider, "local");
        assert_eq!(new.refine.mode, "clean");
        assert_eq!(new.dictionary.words, vec!["Ochre"]);
        assert_eq!(cfg.refine.provider, "off");
    }

    #[test]
    fn bom_is_ignored() {
        let path = std::env::temp_dir().join(format!("ochre-bom-{}.toml", std::process::id()));
        std::fs::write(&path, "\u{feff}[hotkey]\nkey = \"right_ctrl\"\n").unwrap();
        let cfg = Config::load_from(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(cfg.unwrap().hotkey.key, "right_ctrl");
    }

    #[test]
    fn missing_file_is_default() {
        let cfg = Config::load_from(Path::new("/definitely/not/here.toml")).unwrap();
        assert_eq!(cfg, Config::default());
    }
}
