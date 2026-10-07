//! Tray / menu-bar icon. The icon reflects the core state (idle / recording / busy / error;
//! template images on macOS so the menu bar can tint them), and the menu offers status,
//! Start/Stop dictation, Paste last transcript, Hands-free, a Refinement quick switch,
//! Settings…, History… and Quit.

use std::sync::Mutex;

use ochre_core::events::{Command, Event, State};
use tauri::image::Image;
use tauri::menu::{CheckMenuItem, Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::tray::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Manager, Wry};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Idle,
    Recording,
    Busy,
    Error,
}

pub fn variant(state: State) -> Variant {
    match state {
        State::Recording | State::Locked | State::Handsfree => Variant::Recording,
        State::Loading | State::Transcribing | State::Refining | State::Inserting => Variant::Busy,
        State::Error => Variant::Error,
        State::Idle => Variant::Idle,
    }
}

/// Which taskbar / panel the icon sits on. This follows the OS shell's theme (taskbar contrast),
/// not the app's `ui.theme`: a light app on a dark taskbar still needs the light-on-dark art.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bar {
    Light,
    Dark,
}

#[cfg(not(target_os = "macos"))]
macro_rules! tray_png {
    ($v:literal, $bar:literal) => {
        include_bytes!(concat!(
            "../icons/tray/ochre-tray-",
            $v,
            "-",
            $bar,
            "-16@2x.png"
        ))
    };
}

/// (bytes, is_template). Windows / Linux: a 32 px render (16 pt @2x) per taskbar theme.
/// macOS: template images (the menu bar tints them) except recording, the one colour image.
fn icon_bytes(v: Variant, bar: Bar) -> (&'static [u8], bool) {
    #[cfg(target_os = "macos")]
    {
        let _ = bar;
        match v {
            Variant::Idle => (
                include_bytes!("../icons/tray/macos/ochre-template-idle-20@2x.png"),
                true,
            ),
            Variant::Busy => (
                include_bytes!("../icons/tray/macos/ochre-template-busy-20@2x.png"),
                true,
            ),
            Variant::Error => (
                include_bytes!("../icons/tray/macos/ochre-template-error-20@2x.png"),
                true,
            ),
            Variant::Recording => (
                include_bytes!("../icons/tray/macos/ochre-recording-20@2x.png"),
                false,
            ),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let b: &'static [u8] = match (v, bar) {
            (Variant::Idle, Bar::Light) => tray_png!("idle", "light"),
            (Variant::Idle, Bar::Dark) => tray_png!("idle", "dark"),
            (Variant::Recording, Bar::Light) => tray_png!("recording", "light"),
            (Variant::Recording, Bar::Dark) => tray_png!("recording", "dark"),
            (Variant::Busy, Bar::Light) => tray_png!("busy", "light"),
            (Variant::Busy, Bar::Dark) => tray_png!("busy", "dark"),
            (Variant::Error, Bar::Light) => tray_png!("error", "light"),
            (Variant::Error, Bar::Dark) => tray_png!("error", "dark"),
        };
        (b, false)
    }
}

/// The OS taskbar theme. Windows: `SystemUsesLightTheme` (the taskbar / tray, which can differ
/// from the apps theme). Linux: the HUD window's GTK theme. macOS: unused (template images).
pub fn taskbar(app: &AppHandle) -> Bar {
    #[cfg(windows)]
    {
        let _ = app;
        win_taskbar()
    }
    #[cfg(not(windows))]
    {
        match app.get_webview_window("hud").and_then(|w| w.theme().ok()) {
            Some(tauri::Theme::Light) => Bar::Light,
            _ => Bar::Dark,
        }
    }
}

#[cfg(windows)]
fn win_taskbar() -> Bar {
    use windows_sys::Win32::System::Registry::{HKEY_CURRENT_USER, RRF_RT_REG_DWORD, RegGetValueW};
    let wide = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
    let key = wide(r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize");
    let name = wide("SystemUsesLightTheme");
    let mut val: u32 = 0;
    let mut len = std::mem::size_of::<u32>() as u32;
    // SAFETY: valid NUL-terminated wide strings and a u32 out-buffer of the stated size.
    let r = unsafe {
        RegGetValueW(
            HKEY_CURRENT_USER,
            key.as_ptr(),
            name.as_ptr(),
            RRF_RT_REG_DWORD,
            std::ptr::null_mut(),
            (&mut val as *mut u32).cast(),
            &mut len,
        )
    };
    // Windows 10/11 default to a dark taskbar when the value is missing
    if r == 0 && val == 1 {
        Bar::Light
    } else {
        Bar::Dark
    }
}

/// "right_alt" -> "Right Alt" (Windows / Linux) or "Right Option" (macOS); chords join with " + ".
pub fn key_label(name: &str) -> String {
    let mac = cfg!(target_os = "macos");
    let meta = if mac {
        "Command"
    } else if cfg!(windows) {
        "Win"
    } else {
        "Super"
    };
    name.split('+')
        .map(|p| match p.trim() {
            "ctrl" => (if mac { "Control" } else { "Ctrl" }).to_string(),
            "alt" => (if mac { "Option" } else { "Alt" }).to_string(),
            "shift" => "Shift".to_string(),
            "meta" => meta.to_string(),
            "right_alt" => (if mac { "Right Option" } else { "Right Alt" }).to_string(),
            "right_ctrl" => (if mac { "Right Control" } else { "Right Ctrl" }).to_string(),
            "right_shift" => "Right Shift".to_string(),
            "right_meta" => format!("Right {meta}"),
            "caps_lock" => "Caps Lock".to_string(),
            "scroll_lock" => "Scroll Lock".to_string(),
            "space" => "Space".to_string(),
            p if p.len() > 1
                && p.starts_with('f')
                && p[1..].chars().all(|c| c.is_ascii_digit()) =>
            {
                p.to_uppercase()
            }
            p => {
                let mut c = p.chars();
                c.next()
                    .map(|f| f.to_uppercase().chain(c).collect())
                    .unwrap_or_default()
            }
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

/// The tray item for `hotkey.paste_last`, with its shortcut as a hint: a plain key pairs with
/// the Voice key ("Right Ctrl + Down"), a chord stands alone, "" has no shortcut.
pub fn paste_label(voice_key: &str, paste_last: &str) -> String {
    let p = paste_last.trim();
    let hint = if p.is_empty() || matches!(p, "off" | "none" | "disabled") {
        return "Paste last transcript".into();
    } else if p.contains('+') {
        key_label(p)
    } else {
        format!("{} + {}", key_label(voice_key), key_label(p))
    };
    format!("Paste last transcript ({hint})")
}

pub fn status_line(state: State, detail: &str, armed: bool, key: &str, phrase: &str) -> String {
    match state {
        State::Loading if !detail.is_empty() => {
            detail.trim_end_matches(['.', '…']).to_string() + "…"
        }
        State::Loading => "Loading models…".into(),
        State::Recording | State::Locked => "Dictating…".into(),
        State::Handsfree => "Listening hands-free…".into(),
        State::Transcribing => "Transcribing…".into(),
        State::Refining => "Refining…".into(),
        State::Inserting => "Inserting…".into(),
        State::Error if !detail.is_empty() => format!("Error: {detail}"),
        State::Error => "Something went wrong".into(),
        State::Idle if armed => format!("Ready · hold {key} or say “{phrase}”"),
        State::Idle => format!("Ready · hold {key} to dictate"),
    }
}

fn refine_choice(provider: &str) -> &'static str {
    match provider {
        "" | "off" => "off",
        "local" => "local",
        _ => "cloud",
    }
}

struct View {
    state: State,
    detail: String,
    armed: bool,
    key: String,
    /// Raw config values for the paste-last label.
    key_name: String,
    paste_last: String,
    phrase: String,
    handsfree: bool,
    refine: String,
    /// Last cloud refiner chosen, so "Cloud" in the quick switch goes back to it.
    cloud_refiner: String,
    variant: Option<Variant>,
    bar: Bar,
}

pub struct Tray {
    icon: TrayIcon,
    status: MenuItem<Wry>,
    start_stop: MenuItem<Wry>,
    paste: MenuItem<Wry>,
    handsfree: CheckMenuItem<Wry>,
    refine_off: CheckMenuItem<Wry>,
    refine_local: CheckMenuItem<Wry>,
    refine_cloud: CheckMenuItem<Wry>,
    view: Mutex<View>,
}

/// What a menu click asks for.
pub enum Action {
    Core(Command),
    /// Give the keyboard back to the user's window, then type the last transcript.
    PasteLast,
    Open(&'static str),
    Quit,
}

impl Tray {
    pub fn create(
        app: &AppHandle,
        on_action: impl Fn(&AppHandle, Action) + Send + Sync + 'static,
    ) -> tauri::Result<Self> {
        let status = MenuItem::with_id(app, "status", "Starting…", false, None::<&str>)?;
        let start_stop = MenuItem::with_id(app, "toggle", "Start dictation", true, None::<&str>)?;
        let paste = MenuItem::with_id(
            app,
            "paste_last",
            paste_label("right_alt", "down"),
            true,
            None::<&str>,
        )?;
        let handsfree = CheckMenuItem::with_id(
            app,
            "handsfree",
            "Hands-free listening",
            true,
            false,
            None::<&str>,
        )?;
        let refine_off =
            CheckMenuItem::with_id(app, "refine:off", "Off", true, true, None::<&str>)?;
        let refine_local = CheckMenuItem::with_id(
            app,
            "refine:local",
            "Local (on this computer)",
            true,
            false,
            None::<&str>,
        )?;
        let refine_cloud =
            CheckMenuItem::with_id(app, "refine:cloud", "Cloud", true, false, None::<&str>)?;
        let refine = Submenu::with_items(
            app,
            "Refinement",
            true,
            &[&refine_off, &refine_local, &refine_cloud],
        )?;
        let settings = MenuItem::with_id(app, "settings", "Settings…", true, None::<&str>)?;
        let history = MenuItem::with_id(app, "history", "History…", true, None::<&str>)?;
        let quit = MenuItem::with_id(app, "quit", "Quit Ochre", true, None::<&str>)?;
        let sep = || PredefinedMenuItem::separator(app);
        let menu = Menu::with_items(
            app,
            &[
                &status,
                &sep()?,
                &start_stop,
                &paste,
                &handsfree,
                &refine,
                &sep()?,
                &settings,
                &history,
                &sep()?,
                &quit,
            ],
        )?;

        let bar = taskbar(app);
        let (bytes, template) = icon_bytes(Variant::Idle, bar);
        let on_action = std::sync::Arc::new(on_action);
        let on_menu = on_action.clone();
        let icon = TrayIconBuilder::with_id("main")
            .icon(Image::from_bytes(bytes)?)
            .icon_as_template(template)
            .tooltip("Ochre")
            .menu(&menu)
            // macOS: a menu-bar click opens the menu; Windows / Linux: left click opens Settings
            .show_menu_on_left_click(cfg!(target_os = "macos"))
            .on_menu_event(move |app, ev| {
                let id = ev.id().as_ref().to_string();
                let tray = app.state::<std::sync::Arc<Tray>>();
                if let Some(a) = tray.action_for(&id) {
                    on_menu(app, a);
                }
            })
            .on_tray_icon_event(move |icon, ev| {
                if cfg!(target_os = "macos") {
                    return;
                }
                if let TrayIconEvent::Click {
                    button: MouseButton::Left,
                    button_state: MouseButtonState::Up,
                    ..
                } = ev
                {
                    on_action(icon.app_handle(), Action::Open("settings"));
                }
            })
            .build(app)?;

        Ok(Self {
            icon,
            status,
            start_stop,
            paste,
            handsfree,
            refine_off,
            refine_local,
            refine_cloud,
            view: Mutex::new(View {
                state: State::Loading,
                detail: String::new(),
                armed: false,
                key: key_label("right_alt"),
                key_name: "right_alt".into(),
                paste_last: "down".into(),
                phrase: "transcribe".into(),
                handsfree: false,
                refine: "off".into(),
                cloud_refiner: "openai".into(),
                variant: None,
                bar,
            }),
        })
    }

    fn action_for(&self, id: &str) -> Option<Action> {
        let v = self.view.lock().unwrap();
        let a = match id {
            "toggle" => Action::Core(match v.state {
                State::Recording | State::Locked | State::Handsfree => Command::Stop,
                _ => Command::Start,
            }),
            "paste_last" => Action::PasteLast,
            "handsfree" => Action::Core(Command::SetHandsfree {
                enabled: !v.handsfree,
            }),
            "refine:off" | "refine:local" | "refine:cloud" => {
                let provider = match id {
                    "refine:off" => "off".to_string(),
                    "refine:local" => "local".to_string(),
                    _ => v.cloud_refiner.clone(),
                };
                Action::Core(Command::SetConfig {
                    patch: serde_json::json!({ "refine": { "provider": provider } }),
                })
            }
            "settings" => Action::Open("settings"),
            "history" => Action::Open("history"),
            "quit" => Action::Quit,
            _ => return None,
        };
        drop(v);
        // a CheckMenuItem flips itself on click; put the truth back until the core confirms
        self.sync_checks();
        Some(a)
    }

    pub fn on_event(&self, e: &Event) {
        match e {
            Event::State {
                state,
                handsfree_armed,
                detail,
                ..
            } => {
                {
                    let mut v = self.view.lock().unwrap();
                    v.state = *state;
                    v.armed = *handsfree_armed;
                    v.detail = detail.clone();
                }
                self.refresh();
            }
            Event::Config { config, .. } => {
                {
                    let mut v = self.view.lock().unwrap();
                    let s = |p: &str| {
                        config
                            .pointer(p)
                            .and_then(|x| x.as_str())
                            .map(str::to_string)
                    };
                    if let Some(k) = s("/hotkey/key") {
                        v.key = key_label(&k);
                        v.key_name = k;
                    }
                    if let Some(p) = s("/hotkey/paste_last") {
                        v.paste_last = p;
                    }
                    if let Some(p) = s("/handsfree/phrase") {
                        v.phrase = p;
                    }
                    v.handsfree = config
                        .pointer("/handsfree/enabled")
                        .and_then(|x| x.as_bool())
                        .unwrap_or(false);
                    if let Some(r) = s("/refine/provider") {
                        if refine_choice(&r) == "cloud" {
                            v.cloud_refiner = r.clone();
                        }
                        v.refine = r;
                    }
                }
                self.refresh();
            }
            _ => {}
        }
    }

    fn sync_checks(&self) {
        let v = self.view.lock().unwrap();
        let choice = refine_choice(&v.refine);
        let _ = self.handsfree.set_checked(v.handsfree);
        let _ = self.refine_off.set_checked(choice == "off");
        let _ = self.refine_local.set_checked(choice == "local");
        let _ = self.refine_cloud.set_checked(choice == "cloud");
        let _ = self
            .refine_cloud
            .set_text(format!("Cloud ({})", cloud_label(&v.cloud_refiner)));
    }

    /// Re-check the taskbar theme (cheap; called from a slow poll) and swap the art if it flipped.
    #[cfg(not(target_os = "macos"))]
    pub fn check_taskbar(&self) {
        let bar = taskbar(self.icon.app_handle());
        let changed = {
            let mut v = self.view.lock().unwrap();
            let changed = v.bar != bar;
            v.bar = bar;
            if changed {
                v.variant = None;
            }
            changed
        };
        if changed {
            self.refresh();
        }
    }

    fn refresh(&self) {
        let (status, recording, usable, variant, changed, paste, bar) = {
            let mut v = self.view.lock().unwrap();
            let status = status_line(v.state, &v.detail, v.armed, &v.key, &v.phrase);
            let recording = matches!(v.state, State::Recording | State::Locked | State::Handsfree);
            let usable = matches!(v.state, State::Idle | State::Error) || recording;
            let variant = variant(v.state);
            let changed = v.variant != Some(variant);
            v.variant = Some(variant);
            let paste = paste_label(&v.key_name, &v.paste_last);
            (status, recording, usable, variant, changed, paste, v.bar)
        };
        let _ = self.status.set_text(&status);
        let _ = self.start_stop.set_text(if recording {
            "Stop dictation"
        } else {
            "Start dictation"
        });
        let _ = self.start_stop.set_enabled(usable);
        let _ = self.paste.set_text(&paste);
        let _ = self.paste.set_enabled(usable && !recording);
        let _ = self.icon.set_tooltip(Some(format!("Ochre · {status}")));
        if changed {
            let (bytes, template) = icon_bytes(variant, bar);
            if let Ok(img) = Image::from_bytes(bytes) {
                let _ = self.icon.set_icon(Some(img));
                let _ = self.icon.set_icon_as_template(template);
            }
        }
        self.sync_checks();
    }
}

fn cloud_label(id: &str) -> &str {
    match id {
        "openai" => "OpenAI",
        "groq" => "Groq",
        "openrouter" => "OpenRouter",
        "anthropic" => "Anthropic",
        "gemini" => "Gemini",
        "custom" => "custom server",
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        if !cfg!(target_os = "macos") {
            assert_eq!(key_label("right_alt"), "Right Alt");
            assert_eq!(key_label("ctrl+shift+space"), "Ctrl + Shift + Space");
        }
        assert_eq!(key_label("f13"), "F13");
        assert_eq!(key_label("caps_lock"), "Caps Lock");
    }

    #[test]
    fn paste_labels() {
        if !cfg!(target_os = "macos") {
            assert_eq!(
                paste_label("right_ctrl", "down"),
                "Paste last transcript (Right Ctrl + Down)"
            );
            assert_eq!(
                paste_label("right_alt", "ctrl+alt+v"),
                "Paste last transcript (Ctrl + Alt + V)"
            );
        }
        assert_eq!(paste_label("right_ctrl", ""), "Paste last transcript");
        assert_eq!(paste_label("right_ctrl", "off"), "Paste last transcript");
    }

    #[test]
    fn status_and_variant() {
        assert_eq!(variant(State::Locked), Variant::Recording);
        assert_eq!(variant(State::Refining), Variant::Busy);
        assert_eq!(
            status_line(State::Idle, "", false, "Right Alt", "transcribe"),
            "Ready · hold Right Alt to dictate"
        );
        assert_eq!(
            status_line(State::Loading, "Loading models…", false, "", ""),
            "Loading models…"
        );
        assert!(status_line(State::Idle, "", true, "F13", "transcribe").contains("“transcribe”"));
    }
}
