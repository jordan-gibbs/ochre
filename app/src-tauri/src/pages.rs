//! Settings and onboarding windows. Unlike the HUD these are created on demand (opening one isn't
//! latency-critical) and kept hidden until the page has rendered once (`ui_ready`), so there is no
//! white flash in dark mode.

use std::time::Duration;

use tauri::utils::config::Color;
use tauri::webview::WebviewWindowBuilder;
use tauri::{AppHandle, Emitter, Manager, Theme, WebviewUrl};

/// "settings", "history", "transcription", ... -> (window label, page, section)
pub fn route(target: &str) -> (&'static str, &'static str, String) {
    match target {
        "onboarding" => ("onboarding", "onboarding/index.html", String::new()),
        // The main window opens on History; "general" etc. pick a section.
        "settings" | "" => ("settings", "settings/index.html", "history".into()),
        section => ("settings", "settings/index.html", section.to_string()),
    }
}

pub fn theme_of(config: Option<&serde_json::Value>) -> Option<Theme> {
    match config
        .and_then(|c| c.pointer("/ui/theme"))
        .and_then(|t| t.as_str())
    {
        // "system" follows the OS; anything else (missing, "light", unknown) is light, the default
        Some("system") => None,
        Some("dark") => Some(Theme::Dark),
        _ => Some(Theme::Light),
    }
}

pub fn open(app: &AppHandle, target: &str, theme: Option<Theme>) -> tauri::Result<()> {
    let (label, page, section) = route(target);
    if let Some(w) = app.get_webview_window(label) {
        if !section.is_empty() {
            let _ = w.emit_to(label, "ochre://navigate", &section);
        }
        let _ = w.unminimize();
        w.show()?;
        let _ = w.set_focus();
        return Ok(());
    }
    let url = if section.is_empty() {
        page.to_string()
    } else {
        format!("{page}#{section}")
    };
    let dark = match theme {
        Some(Theme::Dark) => true,
        Some(_) => false,
        None => matches!(dark_light_guess(app), Some(Theme::Dark)),
    };
    let bg = if dark {
        Color(0x1b, 0x1a, 0x19, 0xff)
    } else {
        Color(0xf7, 0xf5, 0xf2, 0xff)
    };
    let (w, h, min_w, min_h) = if label == "onboarding" {
        (760.0, 640.0, 640.0, 560.0)
    } else {
        (980.0, 700.0, 780.0, 540.0)
    };
    let win = WebviewWindowBuilder::new(app, label, WebviewUrl::App(url.into()))
        .title("Ochre")
        .inner_size(w, h)
        .min_inner_size(min_w, min_h)
        .center()
        .theme(theme)
        .background_color(bg)
        .visible(false)
        .build()?;
    // fallback, in case the page never reports ready
    let handle = win.clone();
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(1500));
        if !handle.is_visible().unwrap_or(true) {
            let _ = handle.show();
            let _ = handle.set_focus();
        }
    });
    Ok(())
}

/// The page has painted: show its window.
pub fn ready(app: &AppHandle, label: &str) {
    if label == "hud" {
        return;
    }
    if let Some(w) = app.get_webview_window(label)
        && !w.is_visible().unwrap_or(false)
    {
        let _ = w.show();
        let _ = w.set_focus();
    }
}

fn dark_light_guess(app: &AppHandle) -> Option<Theme> {
    app.get_webview_window("hud").and_then(|w| w.theme().ok())
}

pub fn set_theme(app: &AppHandle, theme: Option<Theme>) {
    for label in ["settings", "onboarding"] {
        if let Some(w) = app.get_webview_window(label) {
            let _ = w.set_theme(theme);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes() {
        assert_eq!(
            route("history"),
            ("settings", "settings/index.html", "history".to_string())
        );
        assert_eq!(route("onboarding").0, "onboarding");
        assert_eq!(route("settings").2, "history");
        assert_eq!(route("general").2, "general");
        assert_eq!(
            theme_of(Some(&serde_json::json!({"ui": {"theme": "dark"}}))),
            Some(Theme::Dark)
        );
        assert_eq!(theme_of(None), Some(Theme::Light));
        assert_eq!(
            theme_of(Some(&serde_json::json!({"ui": {"theme": "system"}}))),
            None
        );
    }
}
