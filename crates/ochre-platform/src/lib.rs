//! Ochre platform layer (SPEC §6.1, §6.2): the Voice key gesture state machine,
//! global hotkey listeners, text injection into the focused field, focused-app info, the
//! leading-space join rule and OS permission checks, for Windows, macOS and Linux.
//!
//! Entry points:
//!
//! * [`hotkeys`] gives the OS's [`HotkeyListener`]. It reports [`Gesture`]s through the
//!   `on_gesture` callback, which runs on a dedicated dispatcher thread (never the hook thread)
//!   and should still return quickly.
//! * [`injector`] gives the OS's [`Injector`]: one batched injection per result (§3.1).
//! * [`spacing::JoinMemory`] applies the leading/trailing space rule before injecting.
//! * [`permissions::check`] lists what the OS still has to grant.
//!
//! The orchestrator must call [`HotkeyListener::set_recording`]`(true)` when a session starts
//! (from any source) and `(false)` when it ends or is discarded: Escape is swallowed only while
//! recording, and a session started from the UI is finished by one press of the Voice key.
//!
//! [`Gesture`]: ochre_core::platform::Gesture

pub mod driver;
pub mod focus;
pub mod gesture;
pub mod keys;
pub mod permissions;
pub mod spacing;
pub mod text;

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(windows)]
pub mod windows;

use ochre_core::Result;
use ochre_core::config::HotkeyConfig;
use ochre_core::platform::{HotkeyListener, Injector};

pub use driver::toggle_gestures;

type PrimeHook = std::sync::Arc<dyn Fn() + Send + Sync>;
static PRIME: std::sync::Mutex<Option<PrimeHook>> = std::sync::Mutex::new(None);

/// Called when a dictation is probably about to start, before the Voice key completes (macOS: the
/// first modifier of a chord Voice key went down), so a released microphone can open early. Must
/// return at once: it runs on the hotkey thread.
pub fn set_prime_hook(f: impl Fn() + Send + Sync + 'static) {
    *PRIME.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::sync::Arc::new(f));
}

#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) fn prime() {
    let hook = PRIME.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if let Some(f) = hook {
        f();
    }
}
pub use spacing::{JoinMemory, join_text};

/// The Voice key listener for this OS. Validates `cfg` (unknown or unsafe keys are
/// `Error::Config`); nothing is hooked until [`HotkeyListener::start`].
pub fn hotkeys(cfg: &HotkeyConfig) -> Result<Box<dyn HotkeyListener>> {
    #[cfg(windows)]
    {
        Ok(Box::new(windows::WindowsHotkeys::new(cfg)?))
    }
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(macos::MacHotkeys::new(cfg)?))
    }
    #[cfg(target_os = "linux")]
    {
        linux::hotkeys(cfg)
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        let _ = cfg;
        Err(ochre_core::Error::Other(
            "no hotkey support on this OS".into(),
        ))
    }
}

/// The text injector for this OS.
pub fn injector() -> Result<Box<dyn Injector>> {
    #[cfg(windows)]
    {
        Ok(Box::new(windows::WindowsInjector::new()))
    }
    #[cfg(target_os = "macos")]
    {
        Ok(Box::new(macos::MacInjector::new()))
    }
    #[cfg(target_os = "linux")]
    {
        Ok(Box::new(linux::LinuxInjector::new()?))
    }
    #[cfg(not(any(windows, target_os = "macos", target_os = "linux")))]
    {
        Err(ochre_core::Error::Other(
            "no text injection on this OS".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factories_validate_config() {
        let bad = HotkeyConfig {
            key: "q".into(),
            ..HotkeyConfig::default()
        };
        assert_eq!(hotkeys(&bad).err().map(|e| e.code()), Some("config"));
        #[cfg(windows)]
        {
            assert!(hotkeys(&HotkeyConfig::default()).is_ok());
            assert!(injector().is_ok());
        }
    }
}
