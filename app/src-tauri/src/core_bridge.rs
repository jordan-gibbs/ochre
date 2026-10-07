//! The seam between the shell and the dictation core.
//!
//! The shell needs only `start()`, `command(Command)`, `on_gesture(Gesture)` and `shutdown()`
//! from a core; everything else flows back as `Event`s on the shared `Bus`.
//!
//! - [`OchreCore`] (default): the real orchestrator, `ochre::App`. The binary handles the commands the
//!   orchestrator leaves to it (TestProvider, DownloadModel, TrainWake; OpenSettings and Quit are
//!   handled in main.rs) and advertises the engine catalog and permissions.
//! - [`crate::demo::DemoCore`] (`--demo`): scripted events for every state, in-memory config.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ochre_core::config::Config;
use ochre_core::events::{Bus, Command, EngineInfo, Event};
use ochre_core::history::History;
use ochre_core::platform::Gesture;

use crate::cli::Args;

pub trait Core: Send + Sync + 'static {
    fn start(&self);
    fn command(&self, cmd: Command);
    /// From the hotkey thread; must return immediately. (Called once the lead wires hotkeys.)
    #[allow(dead_code)]
    fn on_gesture(&self, _g: Gesture) {}
    fn shutdown(&self);
}

pub fn create(args: &Args, bus: Bus) -> Box<dyn Core> {
    if args.demo {
        Box::new(crate::demo::DemoCore::new(
            bus,
            args.demo_hold.clone(),
            !args.demo_once,
        ))
    } else {
        Box::new(OchreCore::new(bus))
    }
}

pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

pub fn platform() -> String {
    std::env::consts::OS.to_string()
}

#[cfg(test)]
pub fn state_event(state: ochre_core::events::State, armed: bool, detail: &str) -> Event {
    Event::State {
        state,
        trigger: None,
        handsfree_armed: armed,
        detail: detail.to_string(),
    }
}

pub fn config_event(cfg: &Config, secrets: BTreeMap<String, bool>) -> Event {
    Event::Config {
        config: serde_json::to_value(cfg).unwrap_or_default(),
        secrets,
    }
}

fn info(
    id: &str,
    label: &str,
    kind: &str,
    models: &[&str],
    needs_key: bool,
    note: &str,
    languages: &str,
) -> EngineInfo {
    EngineInfo {
        id: id.into(),
        label: label.into(),
        kind: kind.into(),
        models: models.iter().map(|m| m.to_string()).collect(),
        default_model: models.first().map(|m| m.to_string()).unwrap_or_default(),
        needs_key,
        note: note.into(),
        languages: languages.into(),
    }
}

/// The engine catalog Settings and onboarding offer. The labels and notes are curated here; the
/// speech list is filtered to the engines this build can actually run
/// (`ochre::wiring::stt_available`), so an engine behind a cargo feature that is off (Whisper) is
/// never offered.
pub fn engines() -> Event {
    let stt: Vec<EngineInfo> = vec![
        info(
            "parakeet",
            "Parakeet TDT 0.6B v3",
            "local",
            &["parakeet-tdt-0.6b-v3-int8"],
            false,
            "670 MB · fast on any modern CPU",
            "25 European languages",
        ),
        info(
            "whisper",
            "Whisper large-v3-turbo",
            "local",
            &["large-v3-turbo", "small.en"],
            false,
            "1.6 GB · best with a GPU",
            "99 languages",
        ),
        info(
            "groq",
            "Groq",
            "cloud",
            &["whisper-large-v3-turbo"],
            true,
            "≈ $0.04 per hour of audio",
            "99 languages",
        ),
        info(
            "openai",
            "OpenAI",
            "cloud",
            &[
                "gpt-transcribe",
                "gpt-live-transcribe",
                "gpt-4o-mini-transcribe",
                "gpt-4o-transcribe",
            ],
            true,
            "≈ $0.27 per hour of audio; live streaming ≈ $1.02",
            "57 languages",
        ),
        info(
            "google",
            "Google (Gemini 3.5 Transcribe)",
            "cloud",
            &["gemini-3.5-transcribe", "gemini-3.5-transcribe-live"],
            true,
            "≈ $0.30 per hour of audio; removes fillers itself (preview)",
            "85+ languages",
        ),
        info(
            "soniox",
            "Soniox",
            "cloud",
            &["stt-async-v5"],
            true,
            "≈ $0.10 per hour of audio",
            "60 languages",
        ),
        info(
            "deepgram",
            "Deepgram",
            "cloud",
            &["nova-3"],
            true,
            "≈ $0.26 per hour of audio",
            "36 languages",
        ),
        info(
            "elevenlabs",
            "ElevenLabs Scribe",
            "cloud",
            &["scribe_v1"],
            true,
            "≈ $0.40 per hour of audio",
            "99 languages",
        ),
        info(
            "assemblyai",
            "AssemblyAI",
            "cloud",
            &["universal"],
            true,
            "≈ $0.15 per hour of audio",
            "99 languages",
        ),
    ]
    .into_iter()
    .filter(|e| ochre::wiring::stt_available(&e.id))
    .collect();
    // The local entry is the real one: its model list, and its note says what "auto" picks on
    // this machine (`auto: <id> (<hardware>)`), which Settings shows.
    let local = ochre::wiring::refine_engines()
        .into_iter()
        .find(|e| e.id == "local")
        .unwrap_or_else(|| {
            info(
                "local",
                "Ochre Refine (on this computer)",
                "local",
                &[
                    "auto",
                    "ochre-refine-4b",
                    "ochre-refine-2b",
                    "ochre-refine-0.8b",
                    "quill-4b",
                    "quill-2b",
                    "quill-0.8b",
                ],
                false,
                "auto: quill-0.8b (no usable GPU)",
                "English",
            )
        });
    let refine = vec![
        EngineInfo {
            label: "Ochre Refine (on this computer)".into(),
            languages: "English".into(),
            ..local
        },
        info(
            "openai",
            "OpenAI",
            "cloud",
            &["gpt-5.6-luna", "gpt-6-luna", "gpt-4.1-nano", "gpt-4.1-mini"],
            true,
            "≈ $0.02 per 100 dictations",
            "Any",
        ),
        info(
            "groq",
            "Groq",
            "cloud",
            &["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
            true,
            "≈ $0.01 per 100 dictations",
            "Any",
        ),
        info(
            "anthropic",
            "Anthropic",
            "cloud",
            &["claude-haiku-4-5"],
            true,
            "≈ $0.05 per 100 dictations",
            "Any",
        ),
        info(
            "gemini",
            "Gemini",
            "cloud",
            &["gemini-3.5-flash-lite", "gemini-3.5-flash"],
            true,
            "≈ $0.03 per 100 dictations",
            "Any",
        ),
        info(
            "openrouter",
            "OpenRouter",
            "cloud",
            &[],
            true,
            "Price depends on the model",
            "Any",
        ),
        info(
            "custom",
            "Custom server (OpenAI-compatible)",
            "cloud",
            &[],
            false,
            "Ollama, LM Studio, vLLM…",
            "Any",
        ),
    ];
    Event::Engines {
        stt,
        refine,
        installed: Default::default(),
    }
}

// --------------------------------------------------------------------------------------------
// Snapshot: what a window that opens late (settings, onboarding, a reloaded HUD) needs to catch up.

#[derive(Default)]
pub struct Snapshot {
    hello: Option<Event>,
    config: Option<Event>,
    engines: Option<Event>,
    connectors: Option<Event>,
    permissions: Option<Event>,
    result: Option<Event>,
    state: Option<Event>,
    downloads: BTreeMap<String, Event>,
}

impl Snapshot {
    pub fn record(&mut self, e: &Event) {
        let slot = match e {
            Event::Hello { .. } => &mut self.hello,
            Event::Config { .. } => &mut self.config,
            Event::Engines { .. } => &mut self.engines,
            Event::Connectors { .. } => &mut self.connectors,
            Event::Permissions { .. } => &mut self.permissions,
            Event::Result { .. } => &mut self.result,
            Event::State { .. } => &mut self.state,
            Event::Download { item, done, total } => {
                if total > &0 && done >= total {
                    self.downloads.remove(item);
                } else {
                    self.downloads.insert(item.clone(), e.clone());
                }
                return;
            }
            _ => return,
        };
        *slot = Some(e.clone());
    }

    /// Replay order: identity and settings first, the live state last.
    pub fn events(&self) -> Vec<Event> {
        [
            &self.hello,
            &self.config,
            &self.engines,
            &self.connectors,
            &self.permissions,
            &self.result,
        ]
        .into_iter()
        .flatten()
        .cloned()
        .chain(self.downloads.values().cloned())
        .chain(self.state.iter().cloned())
        .collect()
    }

    pub fn config(&self) -> Option<&serde_json::Value> {
        match &self.config {
            Some(Event::Config { config, .. }) => Some(config),
            _ => None,
        }
    }
}

// --------------------------------------------------------------------------------------------
// OchreCore: the real orchestrator.

/// macOS: fn / Globe as the Voice key also fires whatever "Press 🌐 key to" is set to.
fn globe_notice(bus: &Bus, key: &str) {
    #[cfg(target_os = "macos")]
    if key == "fn"
        && let Some(action) = ochre_platform::macos::globe_key_action()
    {
        bus.emit(Event::Notice {
            message: format!(
                "fn (Globe) will also {action}. Set System Settings › Keyboard › “Press 🌐 key to” \
to “Do Nothing”, or pick another Voice key."
            ),
        });
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (bus, key);
}

fn loader() -> ochre::Loader {
    ochre::wiring::default_loader()
}

type Hotkeys = Arc<parking_lot::Mutex<Option<Box<dyn ochre_core::platform::HotkeyListener>>>>;

pub struct OchreCore {
    app: ochre::App,
    bus: Bus,
    hotkeys: Hotkeys,
    /// The settings key-capture field owns the keyboard: don't (re)install the hook.
    capturing: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

impl OchreCore {
    pub fn new(bus: Bus) -> Self {
        let cfg = Config::load().unwrap_or_else(|e| {
            // Running on defaults means the next settings save overwrites the file, so keep the
            // unreadable one for the user to recover from.
            let path = ochre_core::paths::config_path();
            let keep = path.with_extension("toml.unreadable");
            let _ = std::fs::copy(&path, &keep);
            tracing::error!(
                "config unreadable, using defaults (copy kept at {}): {e}",
                keep.display()
            );
            Config::default()
        });
        let history = if cfg.history.enabled {
            History::open(&ochre_core::paths::data_dir().join("history.sqlite3"))
                .unwrap_or_else(|_| History::disabled())
        } else {
            History::disabled()
        };
        let app = ochre::App::new(cfg, bus.clone(), history, loader());
        app.persist_config_to(ochre_core::paths::config_path());
        let hotkeys: Hotkeys = Arc::new(parking_lot::Mutex::new(None));
        // Escape is swallowed only while a session is live.
        let primed = app.clone();
        ochre_platform::set_prime_hook(move || primed.prime_mic());
        let hk = hotkeys.clone();
        app.set_recording_hook(move |on| {
            if let Some(l) = hk.lock().as_ref() {
                l.set_recording(on);
            }
        });
        Self {
            app,
            bus,
            hotkeys,
            capturing: Arc::new(AtomicBool::new(false)),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    fn start_hotkeys(&self) {
        start_hotkeys(&self.app, &self.bus, &self.hotkeys);
    }

    fn emit_permissions(&self) {
        self.bus.emit(Event::Permissions {
            missing: ochre_platform::permissions::check(),
        });
    }

    /// macOS grants (Accessibility, Input Monitoring, Microphone) are switched on in System
    /// Settings while we run, and nothing tells us. Poll them (cheap calls, 1 s while something is
    /// missing, 5 s otherwise), publish changes for onboarding, and install the Voice key the
    /// moment it can work, so no restart is needed.
    #[cfg(target_os = "macos")]
    fn watch_permissions(&self) {
        let (app, bus, hotkeys) = (self.app.clone(), self.bus.clone(), self.hotkeys.clone());
        let (capturing, stop) = (self.capturing.clone(), self.stop.clone());
        let mut last = ochre_platform::permissions::check();
        let spawned = std::thread::Builder::new()
            .name("ochre-permissions".into())
            .spawn(move || {
                while !stop.load(Ordering::Acquire) {
                    let wait = if last.is_empty() { 5 } else { 1 };
                    std::thread::park_timeout(std::time::Duration::from_secs(wait));
                    if stop.load(Ordering::Acquire) {
                        return;
                    }
                    let now = ochre_platform::permissions::check();
                    if now == last {
                        continue;
                    }
                    tracing::info!(
                        "permissions changed; missing: {:?}",
                        now.iter().map(|p| p.name.as_str()).collect::<Vec<_>>()
                    );
                    let gained = |name: &str| {
                        last.iter().any(|p| p.name == name) && !now.iter().any(|p| p.name == name)
                    };
                    let keys = gained("accessibility") || gained("input_monitoring");
                    let mic = gained("microphone");
                    bus.emit(Event::Permissions {
                        missing: now.clone(),
                    });
                    if keys && !capturing.load(Ordering::Acquire) {
                        // Reinstall even if a tap is up: one made before Input Monitoring was
                        // granted may never see a key.
                        start_hotkeys(&app, &bus, &hotkeys);
                    }
                    if mic {
                        // A stream opened before the grant may keep delivering silence.
                        app.reopen_mic();
                    }
                    last = now;
                }
            });
        if let Err(e) = spawned {
            tracing::warn!("permission watcher: {e}");
        }
    }
}

/// Install the Voice-key hook. A failure (e.g. macOS permission missing) is reported, and the
/// app keeps working from the tray and the CLI.
fn start_hotkeys(app: &ochre::App, bus: &Bus, hotkeys: &Hotkeys) {
    let cfg = app.config();
    let mut slot = hotkeys.lock();
    if let Some(mut old) = slot.take() {
        old.stop();
    }
    let error = |e: &ochre_core::Error| {
        bus.emit(Event::Error {
            message: format!("The Voice key couldn't be installed: {e}"),
            code: e.code().into(),
        })
    };
    match ochre_platform::hotkeys(&cfg.hotkey) {
        Ok(mut listener) => {
            let app = app.clone();
            match listener.start(Box::new(move |g| app.on_gesture(g))) {
                Ok(()) => {
                    tracing::info!("Voice key installed ({})", cfg.hotkey.key);
                    *slot = Some(listener);
                    globe_notice(bus, &cfg.hotkey.key);
                }
                Err(e) => error(&e),
            }
        }
        Err(e) => error(&e),
    }
}

impl Core for OchreCore {
    fn start(&self) {
        self.app.start();
        self.bus.emit(engines());
        self.bus.emit(ochre::connectors::event());
        self.emit_permissions();
        self.start_hotkeys();
        #[cfg(target_os = "macos")]
        self.watch_permissions();
    }

    fn command(&self, cmd: Command) {
        match cmd {
            Command::TestProvider { stage, provider } => {
                self.bus.emit(Event::TestResult { stage, provider, ok: false, message: "Not wired yet.".into(), ms: 0 })
            }
            Command::DownloadModel { .. } => {
                let cfg = self.app.config();
                let bus = self.bus.clone();
                std::thread::spawn(move || {
                    let progress = {
                        let bus = bus.clone();
                        move |p: ochre_core::stt::Progress| bus.emit(Event::Download { item: p.item, done: p.done, total: p.total })
                    };
                    if let Err(e) = ochre::wiring::download_stt(&cfg, &progress) {
                        bus.emit(Event::Error { message: format!("Download failed: {e}"), code: e.code().into() });
                    }
                    bus.emit(engines());
                });
            }
            Command::TrainWake { word } => self.bus.emit(Event::Notice {
                message: format!("Run `python -m training.wakeword.pipeline all --word {word}` to train a wake word."),
            }),
            Command::CheckPermissions => self.emit_permissions(),
            // macOS: the system prompt (when it can still be shown) plus the System Settings pane.
            Command::OpenPermissionSettings { name } => {
                if cfg!(target_os = "macos") {
                    ochre_platform::permissions::request(&name);
                } else {
                    let _ = ochre_platform::permissions::open_settings(&name);
                }
                self.emit_permissions();
            }
            Command::ShowMicModes => {
                #[cfg(target_os = "macos")]
                ochre_audio::show_mic_modes();
            }
            Command::ListDevices => {
                let (items, default) = ochre::wiring::input_devices();
                self.bus.emit(Event::Devices { items, default });
            }
            // The settings key-capture field needs the real key, so the hook steps aside.
            Command::HotkeyCapture { active } => {
                self.capturing.store(active, Ordering::Release);
                if active {
                    if let Some(mut l) = self.hotkeys.lock().take() {
                        l.stop();
                    }
                } else {
                    self.start_hotkeys();
                }
            }
            Command::SetConfig { .. } => {
                let before = self.app.config().hotkey;
                self.app.command(cmd);
                let after = self.app.config().hotkey;
                if before != after {
                    self.start_hotkeys();
                }
            }
            cmd => self.app.command(cmd),
        }
    }

    fn on_gesture(&self, g: Gesture) {
        self.app.on_gesture(g);
    }

    fn shutdown(&self) {
        self.stop.store(true, Ordering::Release);
        if let Some(mut l) = self.hotkeys.lock().take() {
            l.stop();
        }
        self.app.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ochre_core::events::State;

    #[test]
    fn snapshot_replays_state_last_and_drops_finished_downloads() {
        let mut s = Snapshot::default();
        s.record(&state_event(State::Loading, false, ""));
        s.record(&Event::Hello {
            version: "1".into(),
            platform: "windows".into(),
        });
        s.record(&Event::Download {
            item: "a".into(),
            done: 1,
            total: 10,
        });
        s.record(&Event::Download {
            item: "b".into(),
            done: 5,
            total: 5,
        });
        s.record(&Event::Level { rms: 0.3 });
        let ev = s.events();
        assert_eq!(ev.len(), 3);
        assert!(matches!(ev[0], Event::Hello { .. }));
        assert!(matches!(ev[2], Event::State { .. }));
    }

    #[test]
    fn catalog_has_every_connector_engine() {
        let Event::Engines { stt, refine, .. } = engines() else {
            unreachable!()
        };
        for c in ochre::connectors::all() {
            assert!(stt.iter().any(|e| e.id == c.stt.id), "{}", c.stt.id);
            assert!(
                refine.iter().any(|e| e.id == c.refine.id),
                "{}",
                c.refine.id
            );
            for (list, choice) in [(&stt, &c.stt), (&refine, &c.refine)] {
                let e = list.iter().find(|e| e.id == choice.id).unwrap();
                let m = &choice.model;
                assert!(m.is_empty() || e.models.contains(m), "{}/{m}", choice.id);
            }
        }
        let mut s = Snapshot::default();
        s.record(&ochre::connectors::event());
        assert!(matches!(s.events()[0], Event::Connectors { .. }));
    }

    #[test]
    fn local_refine_catalog_offers_auto() {
        let Event::Engines { refine, .. } = engines() else {
            unreachable!()
        };
        let local = refine.iter().find(|e| e.id == "local").unwrap();
        assert_eq!(local.default_model, "auto");
        assert_eq!(local.models[0], "auto");
        for m in [
            "ochre-refine-4b",
            "ochre-refine-2b",
            "ochre-refine-0.8b",
            "quill-0.8b",
        ] {
            assert!(local.models.iter().any(|x| x == m), "{m}");
        }
        assert!(local.note.starts_with("auto: "), "{}", local.note);
    }

    #[test]
    fn catalog_offers_only_engines_this_build_can_run() {
        let Event::Engines { stt, .. } = engines() else {
            unreachable!()
        };
        for e in &stt {
            assert!(ochre::wiring::stt_available(&e.id), "{}", e.id);
        }
        // every engine the build has is offered
        for e in ochre::wiring::stt_engines() {
            assert!(stt.iter().any(|x| x.id == e.id), "{} missing", e.id);
        }
    }

    #[test]
    fn catalog_serializes() {
        let v = serde_json::to_value(engines()).unwrap();
        assert_eq!(v["event"], "engines");
        assert!(
            v["stt"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["id"] == "parakeet")
        );
    }
}
