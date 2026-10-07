//! Linux text injection through the standard command-line tools (SPEC §6.2), one process per
//! result (§3.1: one batched call).
//!
//! * **X11:** `xdotool type --clearmodifiers` (Unicode-capable; lifts modifiers the user still
//!   holds and puts them back; types `\n` as Return and `\t` as Tab), then `ydotool`.
//! * **Wayland:** `wtype` (virtual-keyboard protocol: Sway, Hyprland and other wlroots
//!   compositors; **not GNOME or KDE**, which lack the protocol, see Handy #2006), `kwtype` on
//!   KDE, then `dotool`, then `ydotool` (both through `/dev/uinput`, so they work everywhere but
//!   need write access to it: the packages ship a udev rule that grants it to the logged-in
//!   user, see `packaging/linux/60-ochre-uinput.rules`). ydotool types US-layout key codes only,
//!   so non-ASCII text, or any text on a non-US layout, is pasted instead. Two incompatible
//!   ydotool generations are in the wild: 0.1.x (Debian 12, Ubuntu 24.04) names keys and opens
//!   uinput itself; 1.x takes raw key codes and needs `ydotoold`. Both are handled.
//! * **Paste fallback:** `wl-copy` / `xclip` / `xsel`, then Ctrl+V through the typing tool. Only
//!   the text flavour of the previous clipboard is restored (on a background thread). Terminals
//!   usually paste with Ctrl+Shift+V instead, so typing is preferred there.
//!
//! Every command takes an argument list (never a shell), so dictated text cannot run commands.
//! The native virtual-keyboard protocol was not implemented: it would only cover the wlroots
//! compositors where `wtype` already works.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

use ochre_core::platform::{FocusInfo, Injector};
use ochre_core::{Error, Result};

use super::focus;
use crate::text::{Segment, plan};

pub const NO_TOOL_WAYLAND: &str = "No typing tool found for Wayland. Install `wtype` (Sway, Hyprland), `kwtype` (KDE), \
or `dotool` / `ydotool` with its daemon (GNOME and everything else). The text is in History.";
pub const NO_TOOL_X11: &str = "No typing tool found. Install `xdotool`. The text is in History.";
pub const NO_CLIPBOARD: &str = "No clipboard tool found: install `wl-clipboard` (Wayland) or `xclip` (X11). The text is in History.";

/// One command: argv plus optional stdin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cmd {
    pub args: Vec<String>,
    pub stdin: Option<String>,
}

fn cmd(args: &[&str]) -> Cmd {
    Cmd {
        args: args.iter().map(|s| s.to_string()).collect(),
        stdin: None,
    }
}

/// The typing tool for a session: `xdotool`, `wtype`, `kwtype`, `dotool`, `ydotool`, or None.
pub fn typing_tool(
    session: &str,
    desktop: &str,
    has: &dyn Fn(&str) -> bool,
) -> Option<&'static str> {
    let gnome = desktop.contains("gnome");
    let kde = desktop.contains("kde");
    let order: &[&'static str] = if session == "wayland" {
        if kde {
            &["kwtype", "dotool", "ydotool"]
        } else if gnome {
            &["dotool", "ydotool"]
        } else {
            &["wtype", "dotool", "ydotool"]
        }
    } else {
        &["xdotool", "ydotool"]
    };
    order.iter().copied().find(|t| has(t))
}

/// `ydotool` 0.1.x as a separate tool name: it takes different arguments than 1.x. Without
/// `ydotoold` it creates a fresh uinput device per call.
pub const YDOTOOL_LEGACY: &str = "ydotool-0.1";
/// `ydotool` 0.1.x talking to a running `ydotoold` (one persistent device: no start-up delay).
pub const YDOTOOL_LEGACY_DAEMON: &str = "ydotool-0.1+d";

/// Whether the usage text that `ydotool` prints without arguments is the 0.1.x one (it lists
/// a `recorder` command that 1.x dropped).
pub fn is_legacy_ydotool(usage: &str) -> bool {
    usage.contains("recorder")
}

pub fn ydotool_is_legacy() -> bool {
    static LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LEGACY.get_or_init(|| {
        Command::new("ydotool")
            .stdin(Stdio::null())
            .output()
            .is_ok_and(|o| {
                is_legacy_ydotool(&String::from_utf8_lossy(&o.stdout))
                    || is_legacy_ydotool(&String::from_utf8_lossy(&o.stderr))
            })
    })
}

/// `ydotool` resolved to its generation, starting `ydotoold` first when it can.
fn resolve_tool(tool: &'static str) -> &'static str {
    if tool != "ydotool" {
        return tool;
    }
    let daemon = ensure_ydotoold();
    match (ydotool_is_legacy(), daemon) {
        (true, true) => YDOTOOL_LEGACY_DAEMON,
        (true, false) => YDOTOOL_LEGACY,
        (false, _) => "ydotool",
    }
}

/// Make sure `ydotoold` is serving: if it isn't, and it is installed and this user may write
/// `/dev/uinput`, start it as our child (it dies with us). Without it ydotool 1.x can't type at
/// all and 0.1.x has to wait for a new device on every call.
fn ensure_ydotoold() -> bool {
    /// pid of the ydotoold we started, while it runs.
    static OURS: std::sync::Mutex<Option<u32>> = std::sync::Mutex::new(None);
    if super::ydotoold_running() {
        return true;
    }
    if !super::has_tool("ydotoold") || !super::uinput_writable() {
        return false;
    }
    let mut ours = OURS.lock().unwrap_or_else(|e| e.into_inner());
    if ours.is_some() {
        return false; // ours is running but has no socket: don't start another
    }
    // A socket file left by a daemon that died makes the new one fail to bind.
    for path in super::ydotoold_sockets() {
        let p = std::path::Path::new(&path);
        if p.exists() && !super::socket_alive(&path) {
            let _ = std::fs::remove_file(p);
        }
    }
    // PR_SET_PDEATHSIG fires when the *thread* that forked the child exits, so the daemon is
    // forked from a thread that stays alive, waiting on it, for as long as the daemon runs.
    let (tx, rx) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new()
        .name("ochre-ydotoold".into())
        .spawn(move || {
            let mut command = Command::new("ydotoold");
            command
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            {
                use std::os::unix::process::CommandExt;
                // SAFETY: prctl is async-signal-safe; it only asks for SIGTERM when this
                // thread (so, in practice, Ochre) goes away.
                unsafe {
                    command.pre_exec(|| {
                        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGTERM);
                        Ok(())
                    });
                }
            }
            match command.spawn() {
                Ok(mut child) => {
                    let _ = tx.send(Ok(child.id()));
                    let _ = child.wait();
                    *OURS.lock().unwrap_or_else(|e| e.into_inner()) = None;
                    tracing::warn!("ydotoold exited");
                }
                Err(e) => {
                    let _ = tx.send(Err(e));
                }
            }
        });
    if spawned.is_err() {
        return false;
    }
    match rx.recv() {
        Ok(Ok(pid)) => {
            tracing::info!("started ydotoold (pid {pid})");
            *ours = Some(pid);
        }
        Ok(Err(e)) => {
            tracing::warn!("could not start ydotoold: {e}");
            return false;
        }
        Err(_) => return false,
    }
    drop(ours);
    for _ in 0..50 {
        if super::ydotoold_running() {
            // The desktop drops keys from a virtual keyboard it hasn't picked up yet.
            std::thread::sleep(Duration::from_millis(300));
            return true;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    false
}

/// Whether `tool` can type `text` on keyboard layout `layout` ("" = unknown, assumed US).
/// ydotool sends US key codes, so it can only type ASCII on a US layout.
pub fn can_type(tool: &str, text: &str, layout: &str) -> bool {
    if tool.starts_with("ydotool") {
        text.is_ascii() && (layout.is_empty() || layout == "us")
    } else {
        true
    }
}

/// The active XKB layout ("us", "de", ...; "" when unknown). `XKB_DEFAULT_LAYOUT` wins; on GNOME
/// it is the first input source (`mru-sources`, else `sources`).
pub fn current_layout(desktop: &str) -> String {
    if let Ok(l) = std::env::var("XKB_DEFAULT_LAYOUT") {
        return l.split(',').next().unwrap_or("").trim().to_string();
    }
    if !desktop.contains("gnome") {
        return String::new();
    }
    ["mru-sources", "sources"]
        .iter()
        .find_map(|key| {
            let out = Command::new("gsettings")
                .args(["get", "org.gnome.desktop.input-sources", key])
                .stdin(Stdio::null())
                .output()
                .ok()?;
            parse_gnome_sources(&String::from_utf8_lossy(&out.stdout))
        })
        .unwrap_or_default()
}

/// First xkb layout in a GNOME input-sources value: `[('xkb', 'de+nodeadkeys'), ('ibus', 'x')]`
/// -> "de". None for an empty list or a non-xkb first source.
pub fn parse_gnome_sources(value: &str) -> Option<String> {
    let first = value.split(')').next()?;
    let mut parts = first.split('\'').skip(1).step_by(2);
    let (kind, id) = (parts.next()?, parts.next()?);
    (kind == "xkb").then(|| id.split('+').next().unwrap_or(id).to_string())
}

/// Whether an X11 WM_CLASS / app name is a terminal emulator (Ctrl+V doesn't paste there).
pub fn is_terminal(app: &str) -> bool {
    const TERMINALS: [&str; 15] = [
        "terminal",
        "konsole",
        "xterm",
        "urxvt",
        "rxvt",
        "alacritty",
        "kitty",
        "terminator",
        "tilix",
        "wezterm",
        "foot",
        "kgx",
        "org.gnome.console",
        "st-256color",
        "ghostty",
    ];
    let app = app.to_lowercase();
    TERMINALS.iter().any(|t| app.contains(t))
}

/// The command that types `text` with `tool` (None: the tool cannot type this text).
pub fn type_command(tool: &str, text: &str) -> Option<Cmd> {
    Some(match tool {
        "xdotool" => cmd(&[
            "xdotool",
            "type",
            "--clearmodifiers",
            "--delay",
            "1",
            "--",
            text,
        ]),
        "wtype" => cmd(&["wtype", "--", text]),
        "kwtype" => cmd(&["kwtype", "--", text]),
        "dotool" => {
            let mut script = String::new();
            for seg in plan(text) {
                match seg {
                    Segment::Text(run) => {
                        script.push_str("type ");
                        script.push_str(run);
                        script.push('\n');
                    }
                    Segment::Enter => script.push_str("key enter\n"),
                    Segment::Tab => script.push_str("key tab\n"),
                }
            }
            Cmd {
                args: vec!["dotool".into()],
                stdin: Some(script),
            }
        }
        "ydotool" if text.is_ascii() => cmd(&["ydotool", "type", "--key-delay", "1", "--", text]),
        // 0.1.x parses options out of the text, so the text goes through stdin.
        YDOTOOL_LEGACY | YDOTOOL_LEGACY_DAEMON if text.is_ascii() => Cmd {
            args: [
                "ydotool",
                "type",
                "--delay",
                legacy_delay(tool),
                "--key-delay",
                "1",
                "--file",
                "-",
            ]
            .map(String::from)
            .to_vec(),
            stdin: Some(text.to_string()),
        },
        _ => return None,
    })
}

/// ydotool 0.1.x without its daemon creates a fresh uinput device per call, and the desktop
/// drops keys sent before it has picked the device up (measured on GNOME 46: 20-40 ms lost the
/// first characters). 100 ms is ydotool's own default.
fn legacy_delay(tool: &str) -> &'static str {
    if tool == YDOTOOL_LEGACY_DAEMON {
        "0"
    } else {
        "100"
    }
}

pub fn enter_command(tool: &str) -> Cmd {
    match tool {
        YDOTOOL_LEGACY | YDOTOOL_LEGACY_DAEMON => {
            cmd(&["ydotool", "key", "--delay", legacy_delay(tool), "Enter"])
        }
        "xdotool" => cmd(&["xdotool", "key", "--clearmodifiers", "Return"]),
        "wtype" => cmd(&["wtype", "-k", "Return"]),
        "kwtype" => cmd(&["kwtype", "--", "\n"]),
        "dotool" => Cmd {
            args: vec!["dotool".into()],
            stdin: Some("key enter\n".into()),
        },
        _ => cmd(&["ydotool", "key", "28:1", "28:0"]), // evdev KEY_ENTER
    }
}

pub fn paste_command(tool: &str) -> Cmd {
    match tool {
        YDOTOOL_LEGACY | YDOTOOL_LEGACY_DAEMON => {
            cmd(&["ydotool", "key", "--delay", legacy_delay(tool), "ctrl+v"])
        }
        "xdotool" => cmd(&["xdotool", "key", "--clearmodifiers", "ctrl+v"]),
        "wtype" => cmd(&["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]),
        "dotool" => Cmd {
            args: vec!["dotool".into()],
            stdin: Some("key ctrl+v\n".into()),
        },
        _ => cmd(&["ydotool", "key", "29:1", "47:1", "47:0", "29:0"]), // KEY_LEFTCTRL, KEY_V
    }
}

/// (copy-from-stdin, read-to-stdout) clipboard commands.
pub fn clipboard_commands(session: &str, has: &dyn Fn(&str) -> bool) -> Option<(Cmd, Option<Cmd>)> {
    if session == "wayland" && has("wl-copy") {
        let read = has("wl-paste").then(|| cmd(&["wl-paste", "--no-newline", "--type", "text"]));
        return Some((cmd(&["wl-copy"]), read));
    }
    if has("xclip") {
        return Some((
            cmd(&["xclip", "-selection", "clipboard", "-in"]),
            Some(cmd(&["xclip", "-selection", "clipboard", "-out"])),
        ));
    }
    if has("xsel") {
        return Some((
            cmd(&["xsel", "--clipboard", "--input"]),
            Some(cmd(&["xsel", "--clipboard", "--output"])),
        ));
    }
    None
}

fn failure(c: &Cmd, detail: impl std::fmt::Display) -> Error {
    Error::Inject(format!(
        "{} failed: {detail}. The text is in History.",
        c.args[0]
    ))
}

/// Run a command to completion. Clipboard writers fork a process that keeps serving the
/// selection and holds inherited pipes open, so output is only captured when `capture`.
fn run(c: &Cmd, capture: bool) -> Result<String> {
    let mut command = Command::new(&c.args[0]);
    command.args(&c.args[1..]);
    command.stdin(if c.stdin.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    if capture {
        command.stdout(Stdio::piped()).stderr(Stdio::piped());
    } else {
        command.stdout(Stdio::null()).stderr(Stdio::null());
    }
    let mut child = command.spawn().map_err(|e| failure(c, e))?;
    if let (Some(input), Some(mut pipe)) = (&c.stdin, child.stdin.take()) {
        pipe.write_all(input.as_bytes())
            .map_err(|e| failure(c, e))?;
    }
    let out = child.wait_with_output().map_err(|e| failure(c, e))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(failure(
            c,
            if err.is_empty() {
                out.status.to_string()
            } else {
                err
            },
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

#[derive(Debug, Clone)]
pub struct LinuxInjector {
    pub session: &'static str,
    pub desktop: String,
    /// How long the target gets to read the clipboard before it is restored.
    pub restore_delay: Duration,
}

impl LinuxInjector {
    pub fn new() -> Result<Self> {
        let me = Self {
            session: super::session_type(),
            desktop: super::desktop(),
            restore_delay: Duration::from_millis(750),
        };
        // Start ydotoold now rather than on the first dictation (it costs ~300 ms once).
        if typing_tool(me.session, &me.desktop, &super::has_tool) == Some("ydotool") {
            std::thread::spawn(|| resolve_tool("ydotool"));
        }
        Ok(me)
    }

    fn tool(&self) -> Result<&'static str> {
        typing_tool(self.session, &self.desktop, &super::has_tool)
            .map(resolve_tool)
            .ok_or_else(|| {
                Error::Inject(
                    if self.session == "wayland" {
                        NO_TOOL_WAYLAND
                    } else {
                        NO_TOOL_X11
                    }
                    .into(),
                )
            })
    }
}

impl Injector for LinuxInjector {
    fn focus(&self) -> FocusInfo {
        focus::get_focus()
    }

    fn type_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let tool = self.tool()?;
        if tool.starts_with("ydotool") && !can_type(tool, text, &current_layout(&self.desktop)) {
            return self.paste_text(text); // US key codes only
        }
        // xdotool types characters missing from the keymap by remapping a spare key code per
        // character, and apps often read the key before the remap lands: on Ubuntu 24.04 "wörld
        // café" came out as "wrld caf" most of the time. Paste those, except into terminals
        // (Ctrl+V doesn't paste there).
        if tool == "xdotool" && !text.is_ascii() && !is_terminal(&focus::get_focus().app_name) {
            return self.paste_text(text);
        }
        match type_command(tool, text) {
            Some(c) => run(&c, true).map(|_| ()),
            None => self.paste_text(text),
        }
    }

    fn paste_text(&self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let tool = self.tool()?;
        let (copy, read) = clipboard_commands(self.session, &super::has_tool)
            .ok_or_else(|| Error::Inject(NO_CLIPBOARD.into()))?;
        let saved = read.as_ref().and_then(|r| run(r, true).ok());
        run(
            &Cmd {
                stdin: Some(text.to_string()),
                ..copy.clone()
            },
            false,
        )?;
        if copy.args[0] == "wl-copy" {
            // wl-copy returns before the compositor has the new selection (GNOME 46: an
            // immediate Ctrl+V pasted nothing 3 times in 5; 50 ms later, 5 in 5).
            std::thread::sleep(Duration::from_millis(100));
        }
        let result = run(&paste_command(tool), true).map(|_| ());
        if let Some(saved) = saved {
            let delay = self.restore_delay;
            let pasted = text.to_string();
            std::thread::spawn(move || {
                std::thread::sleep(delay);
                // Only restore while the clipboard still holds our text.
                let current = read.as_ref().and_then(|r| run(r, true).ok());
                if current.is_none_or(|c| c == pasted) {
                    let _ = run(
                        &Cmd {
                            stdin: Some(saved),
                            ..copy
                        },
                        false,
                    );
                }
            });
        }
        result
    }

    fn press_enter(&self) -> Result<()> {
        run(&enter_command(self.tool()?), true).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_choice_per_session() {
        let all = |_: &str| true;
        assert_eq!(typing_tool("x11", "", &all), Some("xdotool"));
        assert_eq!(typing_tool("wayland", "sway", &all), Some("wtype"));
        assert_eq!(typing_tool("wayland", "gnome", &all), Some("dotool"));
        assert_eq!(typing_tool("wayland", "kde", &all), Some("kwtype"));
        let ydo = |t: &str| t == "ydotool";
        assert_eq!(typing_tool("wayland", "gnome", &ydo), Some("ydotool"));
        assert_eq!(typing_tool("x11", "", &ydo), Some("ydotool"));
        assert_eq!(typing_tool("wayland", "", &|_: &str| false), None);
    }

    #[test]
    fn commands_never_go_through_a_shell() {
        let evil = "$(rm -rf ~); `x` -n";
        let c = type_command("xdotool", evil).unwrap();
        assert_eq!(c.args.last().unwrap(), evil);
        assert_eq!(c.args[c.args.len() - 2], "--");
        let c = type_command("wtype", "-k Return").unwrap();
        assert_eq!(c.args, ["wtype", "--", "-k Return"]);
    }

    #[test]
    fn dotool_script_and_ydotool_ascii_only() {
        let c = type_command("dotool", "Hi\nthere\tyou").unwrap();
        assert_eq!(
            c.stdin.unwrap(),
            "type Hi\nkey enter\ntype there\nkey tab\ntype you\n"
        );
        assert!(type_command("ydotool", "café").is_none());
        assert!(type_command("ydotool", "cafe").is_some());
    }

    #[test]
    fn ydotool_generations() {
        assert!(is_legacy_ydotool(
            "Usage: ydotool <cmd> <args>
Available commands:
  type
  recorder
  mousemove
"
        ));
        assert!(!is_legacy_ydotool(
            "Usage: ydotool <cmd> <args>
Available commands:
  click
  mousemove
  type
  key
  debug
  bakers
  stdin
"
        ));
        let c = type_command(YDOTOOL_LEGACY, "-x hi").unwrap();
        assert_eq!(c.args.last().unwrap(), "-");
        assert_eq!(c.stdin.as_deref(), Some("-x hi"));
        assert!(type_command(YDOTOOL_LEGACY, "café").is_none());
        assert_eq!(paste_command(YDOTOOL_LEGACY).args.last().unwrap(), "ctrl+v");
        assert_eq!(enter_command(YDOTOOL_LEGACY).args.last().unwrap(), "Enter");
        assert_eq!(enter_command(YDOTOOL_LEGACY).args[3], "100");
        assert_eq!(paste_command(YDOTOOL_LEGACY_DAEMON).args[3], "0");
    }

    #[test]
    fn terminals() {
        for t in [
            "gnome-terminal-server",
            "org.gnome.Console",
            "kgx",
            "Alacritty",
            "kitty",
            "XTerm",
        ] {
            assert!(is_terminal(t), "{t}");
        }
        assert!(!is_terminal("firefox") && !is_terminal("gnome-text-editor") && !is_terminal(""));
    }

    #[test]
    fn layouts() {
        assert_eq!(
            parse_gnome_sources("[('xkb', 'de+nodeadkeys'), ('xkb', 'us')]").as_deref(),
            Some("de")
        );
        assert_eq!(
            parse_gnome_sources(
                "[('xkb', 'us')]
"
            )
            .as_deref(),
            Some("us")
        );
        assert_eq!(parse_gnome_sources("@a(ss) []"), None);
        assert_eq!(parse_gnome_sources("[('ibus', 'mozc-jp')]"), None);
        assert!(can_type("ydotool", "hi", "us") && can_type("ydotool", "hi", ""));
        assert!(!can_type("ydotool", "hi", "de") && !can_type(YDOTOOL_LEGACY, "é", "us"));
        assert!(can_type("dotool", "é", "de") && can_type("xdotool", "é", ""));
    }

    #[test]
    fn clipboard_tools() {
        let wl = |t: &str| matches!(t, "wl-copy" | "wl-paste");
        let (copy, read) = clipboard_commands("wayland", &wl).unwrap();
        assert_eq!(copy.args, ["wl-copy"]);
        assert!(read.is_some());
        let x = |t: &str| t == "xsel";
        assert_eq!(clipboard_commands("x11", &x).unwrap().0.args[0], "xsel");
        assert!(clipboard_commands("x11", &|_: &str| false).is_none());
        assert_eq!(
            paste_command("wtype").args,
            ["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]
        );
        assert_eq!(
            enter_command("ydotool").args,
            ["ydotool", "key", "28:1", "28:0"]
        );
    }
}
