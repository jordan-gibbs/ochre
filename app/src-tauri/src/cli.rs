//! Command-line arguments, both for this launch and for a second launch whose argv the
//! single-instance plugin forwards to the running app (`ochre toggle`, which is also how
//! Wayland users bind a compositor shortcut).

use ochre_core::events::Command;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Args {
    /// Scripted events instead of the real core (`--demo`).
    pub demo: bool,
    /// Play the demo once instead of looping (`--demo-once`).
    pub demo_once: bool,
    /// Hold the demo at a labelled step (`--demo-hold=recording`), for checking window behaviour.
    pub demo_hold: Option<String>,
    /// Launched by the OS at login (`--autostart`): stay quiet, no windows.
    pub autostart: bool,
    /// Turn off capture exclusion on the HUD (`--no-protect`), for screenshots of the real window.
    pub no_protect: bool,
    /// Window to open at startup (`--open=settings|history|onboarding`).
    pub open: Option<String>,
    /// A verb for the running instance (`toggle|start|stop|cancel|paste-last|settings|history|quit`).
    pub forward: Option<Forward>,
}

/// What a second launch asks the running instance to do.
#[derive(Debug, Clone, PartialEq)]
pub enum Forward {
    Command(Command),
    Open(String),
}

pub fn parse<I, S>(args: I) -> Args
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut out = Args::default();
    for a in args {
        let a = a.as_ref();
        let (key, val) = match a.split_once('=') {
            Some((k, v)) => (k, Some(v)),
            None => (a, None),
        };
        match key {
            "--demo" => out.demo = true,
            "--demo-once" => {
                out.demo = true;
                out.demo_once = true;
            }
            "--demo-hold" => {
                out.demo = true;
                out.demo_hold = val.map(str::to_string);
            }
            "--autostart" => out.autostart = true,
            "--no-protect" => out.no_protect = true,
            "--open" => out.open = val.map(str::to_string),
            _ => {
                if out.forward.is_none() {
                    out.forward = verb(a);
                }
            }
        }
    }
    out
}

/// `toggle`, `--toggle`, `start`, ... -> what the running app should do.
pub fn verb(arg: &str) -> Option<Forward> {
    let v = arg.trim_start_matches('-').to_ascii_lowercase();
    Some(match v.as_str() {
        "toggle" => Forward::Command(Command::Toggle),
        "start" => Forward::Command(Command::Start),
        "stop" => Forward::Command(Command::Stop),
        "cancel" => Forward::Command(Command::Cancel),
        "paste-last" | "paste_last" | "pastelast" => Forward::Command(Command::PasteLast),
        "quit" => Forward::Command(Command::Quit),
        "settings" | "history" | "onboarding" => Forward::Open(v),
        _ => return None,
    })
}

/// The argv of a second launch (argv[0] is the executable).
pub fn forwarded(argv: &[String]) -> Option<Forward> {
    argv.iter().skip(1).find_map(|a| verb(a))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_and_verbs() {
        let a = parse(["--demo-hold=recording", "--no-protect"]);
        assert!(a.demo && a.no_protect);
        assert_eq!(a.demo_hold.as_deref(), Some("recording"));
        assert_eq!(
            parse(["toggle"]).forward,
            Some(Forward::Command(Command::Toggle))
        );
        assert_eq!(parse(["--open=settings"]).open.as_deref(), Some("settings"));
        assert!(parse(["--bogus"]).forward.is_none());
    }

    #[test]
    fn second_launch_argv() {
        let argv = vec!["C:/x/ochre.exe".to_string(), "--cancel".to_string()];
        assert_eq!(forwarded(&argv), Some(Forward::Command(Command::Cancel)));
        let argv = vec!["ochre".to_string(), "history".to_string()];
        assert_eq!(forwarded(&argv), Some(Forward::Open("history".into())));
        assert_eq!(forwarded(&["ochre".to_string()]), None);
        let argv = vec!["ochre".to_string(), "paste-last".to_string()];
        assert_eq!(forwarded(&argv), Some(Forward::Command(Command::PasteLast)));
        assert_eq!(
            parse(["--paste-last"]).forward,
            Some(Forward::Command(Command::PasteLast))
        );
    }
}
