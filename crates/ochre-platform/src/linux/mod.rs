//! Linux: X11 (XInput2 raw key events) and evdev (Wayland) hotkeys, typing through the
//! standard tools (xdotool, wtype, kwtype, dotool, ydotool), paste via wl-copy / xclip.
//!
//! Neither backend can *swallow* a key: the Voice key and Escape also reach the focused app.
//! Pick a key that does nothing on its own (Right Ctrl, F13-F24, Pause, Scroll Lock). Right Alt
//! is fine on most layouts (it is AltGr / ISO_Level3_Shift and does nothing alone), but a lone
//! Alt tap may highlight the menu bar in some apps.
//!
//! **Compositor-shortcut fallback (any Wayland desktop, no permissions):** bind `ochre toggle` to
//! a shortcut in the desktop's keyboard settings (GNOME Settings > Keyboard > Custom Shortcuts,
//! `bindsym` in Sway, `bind` in Hyprland, KDE Shortcuts). The single-instance plugin forwards it
//! to the running app, which maps it with [`crate::toggle_gestures`]: a locked recording starts,
//! the next toggle finishes it.
//!
//! Verified on Ubuntu 24.04 / GNOME 46, X11 and Wayland (evdev Voice key, ydotool 0.1 typing
//! through `ydotoold`, wl-copy paste). KDE, Sway and Hyprland paths are untested.

pub mod evdev;
pub mod focus;
pub mod inject;
pub mod x11;

use ochre_core::config::HotkeyConfig;
use ochre_core::platform::{HotkeyListener, PermissionIssue};
use ochre_core::{Error, Result};

pub use inject::LinuxInjector;

/// `"wayland"`, `"x11"` or `""` (no graphical session).
pub fn session_type() -> &'static str {
    let st = std::env::var("XDG_SESSION_TYPE")
        .unwrap_or_default()
        .to_lowercase();
    if st == "wayland" {
        return "wayland";
    }
    if st == "x11" {
        return "x11";
    }
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        return "wayland";
    }
    if std::env::var_os("DISPLAY").is_some() {
        return "x11";
    }
    ""
}

/// `XDG_CURRENT_DESKTOP`, lowercased ("gnome", "kde", "sway", ...).
pub fn desktop() -> String {
    std::env::var("XDG_CURRENT_DESKTOP")
        .unwrap_or_default()
        .to_lowercase()
}

/// Whether an executable is on PATH.
pub fn has_tool(name: &str) -> bool {
    let Some(path) = std::env::var_os("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| {
        let p = dir.join(name);
        p.is_file()
            && std::fs::metadata(&p).is_ok_and(|m| {
                use std::os::unix::fs::PermissionsExt;
                m.permissions().mode() & 0o111 != 0
            })
    })
}

/// Where ydotool looks for its daemon: `$YDOTOOL_SOCKET`, `/tmp` (0.1.x and early 1.x) and the
/// runtime dir (later 1.x).
pub fn ydotoold_sockets() -> Vec<String> {
    // SAFETY: getuid has no failure mode.
    let uid = unsafe { libc::getuid() };
    [
        std::env::var("YDOTOOL_SOCKET").unwrap_or_default(),
        "/tmp/.ydotool_socket".to_string(),
        format!("/run/user/{uid}/.ydotool_socket"),
    ]
    .into_iter()
    .filter(|c| !c.is_empty())
    .collect()
}

/// A daemon answers on `path`: 0.1.x listens on a stream socket, 1.x on a datagram one. A
/// socket file left behind by a daemon that died refuses both.
pub fn socket_alive(path: &str) -> bool {
    use std::os::unix::net::{UnixDatagram, UnixStream};
    UnixStream::connect(path).is_ok()
        || UnixDatagram::unbound().is_ok_and(|s| s.connect(path).is_ok())
}

pub fn ydotoold_running() -> bool {
    ydotoold_sockets().iter().any(|c| socket_alive(c))
}

pub fn permission_issues() -> Vec<PermissionIssue> {
    let tool = |t: &str| has_tool(t);
    let desktop = desktop();
    crate::permissions::linux_issues(&crate::permissions::LinuxEnv {
        session: session_type(),
        has_tool: &tool,
        input_readable: evdev::any_readable(),
        ydotoold_running: ydotoold_running(),
        desktop: &desktop,
        uinput_writable: uinput_writable(),
        ydotool_legacy: has_tool("ydotool") && inject::ydotool_is_legacy(),
    })
}

/// This user may write `/dev/uinput` (udev `uaccess` rule, or a group with access).
pub fn uinput_writable() -> bool {
    // SAFETY: valid NUL-terminated path.
    unsafe { libc::access(c"/dev/uinput".as_ptr(), libc::W_OK) == 0 }
}

/// The key backend for this session. `OCHRE_HOTKEY_BACKEND=evdev|x11` overrides. Wayland uses
/// evdev (X11 clients only see keys while an XWayland window has focus); X11 uses XInput2
/// and falls back to evdev when the X server lacks it.
pub fn hotkeys(cfg: &HotkeyConfig) -> Result<Box<dyn HotkeyListener>> {
    let forced = std::env::var("OCHRE_HOTKEY_BACKEND").unwrap_or_default();
    match forced.as_str() {
        "evdev" => return Ok(Box::new(evdev::EvdevHotkeys::new(cfg)?)),
        "x11" => return Ok(Box::new(x11::X11Hotkeys::new(cfg)?)),
        "" => {}
        other => {
            return Err(Error::Config(format!(
                "OCHRE_HOTKEY_BACKEND must be evdev or x11, not {other:?}"
            )));
        }
    }
    if session_type() == "x11" && x11::available() {
        Ok(Box::new(x11::X11Hotkeys::new(cfg)?))
    } else {
        Ok(Box::new(evdev::EvdevHotkeys::new(cfg)?))
    }
}

fn validate(spec: &crate::keys::KeySpec) -> std::result::Result<(), String> {
    match crate::keys::evdev_code(spec.key) {
        Some(_) => Ok(()),
        None => Err(format!(
            "{:?} cannot be used as the Voice key on Linux",
            spec.key
        )),
    }
}
