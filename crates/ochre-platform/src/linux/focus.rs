//! The focused app on Linux.
//!
//! * X11: `_NET_ACTIVE_WINDOW`, then `WM_CLASS` (class, lowercased) and `_NET_WM_NAME`, over
//!   `x11rb`.
//! * Wayland: there is no portable way to ask. Hyprland (`hyprctl activewindow -j`) and Sway
//!   (`swaymsg -t get_tree`) are supported; elsewhere (GNOME, KDE) `FocusInfo` is empty, which
//!   only means per-app styles and the leading-space join rule fall back to their defaults.

use std::process::Command;

use ochre_core::platform::FocusInfo;
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{AtomEnum, ConnectionExt as _};
use x11rb::rust_connection::RustConnection;

pub fn get_focus() -> FocusInfo {
    if super::session_type() == "wayland" {
        if super::has_tool("hyprctl") {
            return parse_hyprland(&run(&["hyprctl", "activewindow", "-j"]));
        }
        if super::has_tool("swaymsg") {
            return parse_sway_tree(&run(&["swaymsg", "-t", "get_tree"]));
        }
        return FocusInfo::default();
    }
    x11_focus().unwrap_or_default()
}

fn run(args: &[&str]) -> String {
    Command::new(args[0])
        .args(&args[1..])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default()
}

fn x11_focus() -> Option<FocusInfo> {
    let (conn, screen) = RustConnection::connect(None).ok()?;
    let root = conn.setup().roots.get(screen)?.root;
    let atom = |name: &[u8]| {
        conn.intern_atom(false, name)
            .ok()?
            .reply()
            .ok()
            .map(|r| r.atom)
    };
    let active = atom(b"_NET_ACTIVE_WINDOW")?;
    let utf8 = atom(b"UTF8_STRING")?;
    let net_name = atom(b"_NET_WM_NAME")?;
    let reply = conn
        .get_property(false, root, active, AtomEnum::WINDOW, 0, 1)
        .ok()?
        .reply()
        .ok()?;
    let window = reply.value32()?.next()?;
    if window == 0 {
        return Some(FocusInfo::default());
    }
    let class = conn
        .get_property(false, window, AtomEnum::WM_CLASS, AtomEnum::STRING, 0, 256)
        .ok()
        .and_then(|c| c.reply().ok())
        .map(|r| parse_wm_class(&r.value))
        .unwrap_or_default();
    let mut title = conn
        .get_property(false, window, net_name, utf8, 0, 1024)
        .ok()
        .and_then(|c| c.reply().ok())
        .map(|r| String::from_utf8_lossy(&r.value).into_owned())
        .unwrap_or_default();
    if title.is_empty() {
        title = conn
            .get_property(false, window, AtomEnum::WM_NAME, AtomEnum::STRING, 0, 1024)
            .ok()
            .and_then(|c| c.reply().ok())
            .map(|r| String::from_utf8_lossy(&r.value).into_owned())
            .unwrap_or_default();
    }
    Some(FocusInfo {
        app_name: class,
        window_title: title,
        window_id: format!("x11:{window:x}"),
        elevated: false,
    })
}

/// `WM_CLASS` is "instance\0Class\0": the class, lowercased.
pub fn parse_wm_class(raw: &[u8]) -> String {
    raw.split(|b| *b == 0)
        .rfind(|p| !p.is_empty())
        .map(|p| String::from_utf8_lossy(p).to_lowercase())
        .unwrap_or_default()
}

fn find_focused(node: &serde_json::Value) -> Option<&serde_json::Value> {
    if node.get("focused").and_then(|f| f.as_bool()) == Some(true) {
        return Some(node);
    }
    ["nodes", "floating_nodes"]
        .iter()
        .filter_map(|k| node.get(*k).and_then(|n| n.as_array()))
        .flatten()
        .find_map(find_focused)
}

fn text(v: Option<&serde_json::Value>) -> String {
    v.and_then(|v| v.as_str()).unwrap_or_default().to_string()
}

/// The focused container of `swaymsg -t get_tree`.
pub fn parse_sway_tree(json: &str) -> FocusInfo {
    let Ok(tree) = serde_json::from_str::<serde_json::Value>(json) else {
        return FocusInfo::default();
    };
    let Some(node) = find_focused(&tree) else {
        return FocusInfo::default();
    };
    let mut app = text(node.get("app_id"));
    if app.is_empty() {
        app = text(node.get("window_properties").and_then(|p| p.get("class")));
    }
    FocusInfo {
        app_name: app.to_lowercase(),
        window_title: text(node.get("name")),
        window_id: format!(
            "sway:{}",
            node.get("id").and_then(|i| i.as_i64()).unwrap_or_default()
        ),
        elevated: false,
    }
}

/// `hyprctl activewindow -j`.
pub fn parse_hyprland(json: &str) -> FocusInfo {
    let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
        return FocusInfo::default();
    };
    if !v.is_object() {
        return FocusInfo::default();
    }
    FocusInfo {
        app_name: text(v.get("class")).to_lowercase(),
        window_title: text(v.get("title")),
        window_id: format!("hypr:{}", text(v.get("address"))),
        elevated: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wm_class() {
        assert_eq!(parse_wm_class(b"slack\0Slack\0"), "slack");
        assert_eq!(parse_wm_class(b"code\0Code\0"), "code");
        assert_eq!(parse_wm_class(b""), "");
    }

    #[test]
    fn hyprland() {
        let f = parse_hyprland(r#"{"address": "0x55d1", "class": "kitty", "title": "vim \"x\""}"#);
        assert_eq!(f.app_name, "kitty");
        assert_eq!(f.window_title, "vim \"x\"");
        assert_eq!(f.window_id, "hypr:0x55d1");
        assert_eq!(parse_hyprland("garbage"), FocusInfo::default());
    }

    #[test]
    fn sway() {
        let tree = r#"{"id": 1, "name": "root", "rect": {"x": 0}, "focused": false, "nodes": [
            {"id": 42, "rect": {"x": 0, "y": 0}, "name": "Inbox - Mail", "app_id": "Thunderbird", "focused": true, "nodes": []}]}"#;
        let f = parse_sway_tree(tree);
        assert_eq!(f.app_name, "thunderbird");
        assert_eq!(f.window_title, "Inbox - Mail");
        assert_eq!(f.window_id, "sway:42");
        let xwayland = r#"{"id": 7, "name": "Slack", "app_id": null, "window_properties": {"class": "Slack"}, "focused": true}"#;
        assert_eq!(parse_sway_tree(xwayland).app_name, "slack");
    }
}
