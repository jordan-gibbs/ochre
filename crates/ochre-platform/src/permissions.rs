//! What the OS still has to grant before hotkeys and typing work, with fix instructions.
//!
//! [`check`] returns one [`PermissionIssue`] per missing grant; empty means good to go. The
//! onboarding UI lists them and, on macOS, offers a button per entry that calls
//! [`open_settings`] (and [`request`], which shows the system prompt).
//!
//! * **Windows** needs no grants. Elevated (admin) targets are reported per insertion.
//! * **macOS:** `microphone`, `accessibility` (swallowing the Voice key, typing) and
//!   `input_monitoring` (seeing the Voice key). The grant belongs to the binary that runs us (the packaged app,
//!   or the terminal when started from a shell).
//! * **Linux:** `display` (no graphical session), `input_group` (Wayland: reading
//!   `/dev/input`), `typing_tool` / `ydotoold` / `clipboard_tool` (the helper programs we drive).

use ochre_core::platform::PermissionIssue;

fn issue(name: &str, fix: impl Into<String>) -> PermissionIssue {
    PermissionIssue {
        name: name.into(),
        fix: fix.into(),
    }
}

/// Everything missing on this machine right now.
pub fn check() -> Vec<PermissionIssue> {
    #[cfg(target_os = "macos")]
    {
        macos_check()
    }
    #[cfg(target_os = "linux")]
    {
        crate::linux::permission_issues()
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        Vec::new()
    }
}

/// macOS System Settings deep links per permission name.
pub const MAC_PANES: [(&str, &str); 4] = [
    (
        "accessibility",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    ),
    (
        "input_monitoring",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent",
    ),
    (
        "screen_recording",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    ),
    (
        "microphone",
        "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone",
    ),
];

pub fn mac_pane_url(permission: &str) -> Option<&'static str> {
    MAC_PANES
        .iter()
        .find(|(n, _)| *n == permission)
        .map(|(_, u)| *u)
}

/// macOS: open System Settings at the pane for `permission`. Returns false elsewhere, for an
/// unknown permission, or when `open` fails.
pub fn open_settings(permission: &str) -> bool {
    let Some(url) = mac_pane_url(permission) else {
        return false;
    };
    if !cfg!(target_os = "macos") {
        return false;
    }
    std::process::Command::new("open")
        .arg(url)
        .status()
        .is_ok_and(|s| s.success())
}

/// macOS: show the system prompt for a permission (no-op elsewhere or when already granted),
/// and open its System Settings pane, where the switch is. macOS shows each prompt only once
/// per app, so the pane is what the user needs from the second try on.
pub fn request(permission: &str) {
    #[cfg(target_os = "macos")]
    {
        match permission {
            "microphone" => {
                if !crate::macos::request_microphone() {
                    open_settings(permission);
                }
            }
            "accessibility" => {
                if !crate::macos::accessibility_trusted() {
                    crate::macos::request_accessibility();
                    open_settings(permission);
                }
            }
            "input_monitoring" => {
                if !objc2_core_graphics::CGRequestListenEventAccess() {
                    open_settings(permission);
                }
            }
            _ => {
                open_settings(permission);
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = permission;
    }
}

const MAC_WHO: &str = "the app that runs Ochre (the packaged app, or your Terminal / iTerm when started from a shell)";

#[cfg(target_os = "macos")]
fn macos_check() -> Vec<PermissionIssue> {
    let mut out = Vec::new();
    if crate::macos::microphone_access() != crate::macos::MicAccess::Granted {
        out.push(issue("microphone", mac_microphone_fix()));
    }
    if !crate::macos::accessibility_trusted() {
        out.push(issue("accessibility", mac_accessibility_fix()));
    }
    if !objc2_core_graphics::CGPreflightListenEventAccess() {
        out.push(issue("input_monitoring", mac_input_monitoring_fix()));
    }
    out
}

pub fn mac_microphone_fix() -> String {
    "Allow the microphone: System Settings > Privacy & Security > Microphone, then turn on Ochre. \
Needed to hear you."
        .to_string()
}

pub fn mac_accessibility_fix() -> String {
    format!(
        "Allow Accessibility: System Settings > Privacy & Security > Accessibility, then turn on {MAC_WHO}. \
Needed to use the Voice key and to type text."
    )
}

pub fn mac_input_monitoring_fix() -> String {
    format!(
        "Allow Input Monitoring: System Settings > Privacy & Security > Input Monitoring, then turn on {MAC_WHO}. \
Needed to see the Voice key."
    )
}

/// Linux checks over an injected environment (the unit-test seam).
pub struct LinuxEnv<'a> {
    pub session: &'a str,
    pub has_tool: &'a dyn Fn(&str) -> bool,
    /// Some readable `/dev/input/event*` exists (None: no devices at all).
    pub input_readable: Option<bool>,
    pub ydotoold_running: bool,
    pub desktop: &'a str,
    /// This user may write `/dev/uinput` (dotool and ydotool 0.1.x type through it directly).
    pub uinput_writable: bool,
    /// The installed `ydotool` is 0.1.x (Debian 12, Ubuntu 24.04), which needs no daemon.
    pub ydotool_legacy: bool,
}

pub const UINPUT_FIX: &str = "Let Ochre type: your login needs write access to /dev/uinput. The Ochre .deb installs a udev rule for this; otherwise run `echo 'KERNEL==\"uinput\", TAG+=\"uaccess\"' | sudo tee /etc/udev/rules.d/60-ochre-uinput.rules && sudo udevadm trigger /dev/uinput`, then restart Ochre.";

pub const TOGGLE_HINT: &str = "Or bind `ochre toggle` to a shortcut in your desktop's keyboard settings: it starts and \
finishes a recording without any special permission.";

pub fn linux_issues(env: &LinuxEnv<'_>) -> Vec<PermissionIssue> {
    let tool = env.has_tool;
    let mut out = Vec::new();
    match env.session {
        "wayland" => {
            if env.input_readable == Some(false) {
                out.push(issue(
                    "input_group",
                    format!(
                        "To use the Voice key on Wayland, add yourself to the input group: \
`sudo usermod -aG input $USER`, then log out and back in. {TOGGLE_HINT}"
                    ),
                ));
            }
            let desktop = env.desktop.to_lowercase();
            let wtype_works = !(desktop.contains("gnome") || desktop.contains("kde"));
            // Typing through a compositor protocol: no permissions needed.
            let direct = (wtype_works && tool("wtype")) || tool("kwtype");
            // dotool and ydotool 0.1.x open /dev/uinput themselves; ydotool 1.x needs ydotoold,
            // which Ochre starts itself when it is installed and /dev/uinput is writable.
            let via_uinput =
                tool("dotool") || (tool("ydotool") && (env.ydotool_legacy || tool("ydotoold")));
            let ready = direct
                || (via_uinput && env.uinput_writable)
                || (tool("ydotool") && env.ydotoold_running);
            if !direct && !tool("dotool") && !tool("ydotool") {
                out.push(issue(
                    "typing_tool",
                    "Install a typing tool: `wtype` (Sway, Hyprland and other wlroots desktops), `kwtype` (KDE), \
or `ydotool` / `dotool` (GNOME and everything else), e.g. `sudo apt install ydotool`.",
                ));
            } else if !ready && via_uinput {
                out.push(issue("uinput", UINPUT_FIX));
            } else if !ready {
                out.push(issue(
                    "ydotoold",
                    "Start the ydotool daemon: `systemctl --user enable --now ydotool` (or run `ydotoold`); \
ydotool needs it to type.",
                ));
            }
            if !tool("wl-copy") {
                out.push(issue(
                    "clipboard_tool",
                    "Install `wl-clipboard` for the paste fallback.",
                ));
            }
        }
        "x11" => {
            if !tool("xdotool") {
                out.push(issue(
                    "typing_tool",
                    "Install `xdotool` so dictated text can be typed.",
                ));
            }
            if !(tool("xclip") || tool("xsel")) {
                out.push(issue(
                    "clipboard_tool",
                    "Install `xclip` for the paste fallback.",
                ));
            }
        }
        _ => out.push(issue(
            "display",
            "No graphical session found (neither WAYLAND_DISPLAY nor DISPLAY is set).",
        )),
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: &[PermissionIssue]) -> Vec<&str> {
        v.iter().map(|i| i.name.as_str()).collect()
    }

    #[test]
    fn linux_wayland_matrix() {
        let none = |_: &str| false;
        let env = LinuxEnv {
            session: "wayland",
            has_tool: &none,
            input_readable: Some(false),
            ydotoold_running: false,
            desktop: "sway",
            uinput_writable: false,
            ydotool_legacy: false,
        };
        assert_eq!(
            names(&linux_issues(&env)),
            ["input_group", "typing_tool", "clipboard_tool"]
        );

        let wtype = |t: &str| matches!(t, "wtype" | "wl-copy");
        let env = LinuxEnv {
            has_tool: &wtype,
            input_readable: Some(true),
            ..env
        };
        assert!(linux_issues(&env).is_empty());

        // wtype does not work on GNOME (no virtual-keyboard protocol): ydotool needs its daemon.
        let gnome = |t: &str| matches!(t, "wtype" | "ydotool" | "wl-copy");
        let env = LinuxEnv {
            has_tool: &gnome,
            desktop: "GNOME",
            ..env
        };
        assert_eq!(names(&linux_issues(&env)), ["ydotoold"]);
        let env = LinuxEnv {
            ydotoold_running: true,
            ..env
        };
        assert!(linux_issues(&env).is_empty());

        // Ubuntu 24.04: ydotool 0.1.x writes /dev/uinput itself, so it needs the udev rule.
        let env = LinuxEnv {
            ydotoold_running: false,
            ydotool_legacy: true,
            ..env
        };
        assert_eq!(names(&linux_issues(&env)), ["uinput"]);
        let env = LinuxEnv {
            uinput_writable: true,
            ..env
        };
        assert!(linux_issues(&env).is_empty());
        // dotool alone also goes through uinput.
        let dotool = |t: &str| matches!(t, "dotool" | "wl-copy");
        let env = LinuxEnv {
            has_tool: &dotool,
            uinput_writable: false,
            ..env
        };
        assert_eq!(names(&linux_issues(&env)), ["uinput"]);
    }

    #[test]
    fn linux_x11_and_headless() {
        let none = |_: &str| false;
        let env = LinuxEnv {
            session: "x11",
            has_tool: &none,
            input_readable: None,
            ydotoold_running: false,
            desktop: "",
            uinput_writable: false,
            ydotool_legacy: false,
        };
        assert_eq!(
            names(&linux_issues(&env)),
            ["typing_tool", "clipboard_tool"]
        );
        let all = |_: &str| true;
        let env = LinuxEnv {
            has_tool: &all,
            ..env
        };
        assert!(linux_issues(&env).is_empty());
        let env = LinuxEnv { session: "", ..env };
        assert_eq!(names(&linux_issues(&env)), ["display"]);
    }

    #[test]
    fn mac_panes() {
        assert!(
            mac_pane_url("accessibility")
                .unwrap()
                .contains("Privacy_Accessibility")
        );
        assert!(
            mac_pane_url("input_monitoring")
                .unwrap()
                .contains("ListenEvent")
        );
        assert!(mac_pane_url("bogus").is_none());
        if !cfg!(target_os = "macos") {
            assert!(!open_settings("accessibility"));
        }
        assert!(mac_accessibility_fix().contains("Accessibility"));
        assert!(mac_input_monitoring_fix().contains("Input Monitoring"));
        assert!(mac_microphone_fix().contains("Microphone"));
    }

    #[test]
    fn check_runs() {
        let issues = check();
        if cfg!(windows) {
            assert!(issues.is_empty());
        }
    }
}
