//! The focused app on macOS: a short app name, window title and window number.
//!
//! The frontmost normal window comes from `CGWindowListCopyWindowInfo` (front to back, layer
//! 0), which stays current even off the main thread, unlike `NSWorkspace.frontmostApplication`
//! (updated through main-run-loop notifications). Since macOS 10.15 other apps' window titles
//! are only visible with the Screen Recording permission, so without it the title is empty;
//! the app name and window id still work for the join rule and per-app styles.

use objc2::runtime::AnyObject;
use objc2_app_kit::{NSRunningApplication, NSWorkspace};
use objc2_core_foundation::CFArray;
use objc2_core_graphics::{
    CGWindowListCopyWindowInfo, CGWindowListOption, kCGNullWindowID, kCGWindowLayer, kCGWindowName,
    kCGWindowNumber, kCGWindowOwnerPID,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSString};
use ochre_core::platform::FocusInfo;

/// Bundle ids whose names do not match the keys in `RefineConfig.app_styles`.
const FRIENDLY_BUNDLES: &[(&str, &str)] = &[
    ("com.microsoft.vscode", "code"),
    ("com.microsoft.vscodeinsiders", "code"),
    ("com.apple.mail", "mail"),
    ("com.apple.terminal", "terminal"),
    ("com.googlecode.iterm2", "iterm2"),
    ("com.tinyspeck.slackmacgap", "slack"),
    ("com.hnc.discord", "discord"),
    ("com.microsoft.outlook", "outlook"),
    ("com.google.chrome", "chrome"),
    ("com.microsoft.edgemac", "edge"),
    ("org.mozilla.firefox", "firefox"),
    ("company.thebrowser.browser", "arc"),
    ("com.apple.safari", "safari"),
    ("com.apple.mobilesms", "messages"),
    ("notion.id", "notion"),
    ("us.zoom.xos", "zoom"),
    ("dev.warp.warp-stable", "terminal"),
    ("com.mitchellh.ghostty", "terminal"),
    ("net.kovidgoyal.kitty", "terminal"),
    ("io.alacritty", "terminal"),
];

/// Friendly name for known bundles, else the localized name squeezed to `[a-z0-9]`, else the
/// bundle id's last component.
pub fn app_key(bundle_id: &str, localized_name: &str) -> String {
    let bid = bundle_id.to_lowercase();
    if let Some((_, name)) = FRIENDLY_BUNDLES.iter().find(|(b, _)| *b == bid) {
        return (*name).to_string();
    }
    let squeezed: String = localized_name
        .to_lowercase()
        .chars()
        .filter(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        .collect();
    if squeezed.len() >= 2 {
        return squeezed.chars().take(24).collect();
    }
    bid.rsplit('.').next().unwrap_or("").to_string()
}

fn number(dict: &NSDictionary<NSString, AnyObject>, key: &NSString) -> Option<i64> {
    let value = dict.objectForKey(key)?;
    value.downcast_ref::<NSNumber>().map(|n| n.as_i64())
}

fn string(dict: &NSDictionary<NSString, AnyObject>, key: &NSString) -> String {
    dict.objectForKey(key)
        .and_then(|v| v.downcast_ref::<NSString>().map(|s| s.to_string()))
        .unwrap_or_default()
}

fn cf_key(key: &objc2_core_foundation::CFString) -> &NSString {
    // SAFETY: CFString is toll-free bridged to NSString.
    unsafe { &*(key as *const objc2_core_foundation::CFString as *const NSString) }
}

/// (pid, window number, title) of the frontmost normal window.
fn front_window() -> Option<(i32, i64, String)> {
    let options =
        CGWindowListOption::OptionOnScreenOnly | CGWindowListOption::ExcludeDesktopElements;
    let list = CGWindowListCopyWindowInfo(options, kCGNullWindowID)?;
    // SAFETY: CFArray of CFDictionary is toll-free bridged to NSArray of NSDictionary.
    let array: &NSArray<NSDictionary<NSString, AnyObject>> = unsafe {
        &*(&*list as *const CFArray as *const NSArray<NSDictionary<NSString, AnyObject>>)
    };
    // SAFETY: reading immutable CoreGraphics constants.
    let (k_pid, k_layer, k_number, k_name) = unsafe {
        (
            cf_key(kCGWindowOwnerPID),
            cf_key(kCGWindowLayer),
            cf_key(kCGWindowNumber),
            cf_key(kCGWindowName),
        )
    };
    for info in array.iter() {
        if number(&info, k_layer) == Some(0)
            && let Some(pid) = number(&info, k_pid)
        {
            return Some((
                pid as i32,
                number(&info, k_number).unwrap_or(0),
                string(&info, k_name),
            ));
        }
    }
    None
}

pub fn get_focus() -> FocusInfo {
    let (pid, window, title) = match front_window() {
        Some(w) => w,
        None => {
            let Some(app) = NSWorkspace::sharedWorkspace().frontmostApplication() else {
                return FocusInfo::default();
            };
            (app.processIdentifier(), 0, String::new())
        }
    };
    let (bundle, name) = match NSRunningApplication::runningApplicationWithProcessIdentifier(pid) {
        Some(app) => (
            app.bundleIdentifier()
                .map(|s| s.to_string())
                .unwrap_or_default(),
            app.localizedName()
                .map(|s| s.to_string())
                .unwrap_or_default(),
        ),
        None => (String::new(), String::new()),
    };
    FocusInfo {
        app_name: app_key(&bundle, &name),
        window_title: title,
        window_id: format!("{pid}:{window}"),
        elevated: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_keys() {
        assert_eq!(app_key("com.microsoft.VSCode", "Code"), "code");
        assert_eq!(app_key("com.example.thing", "Some App 2"), "someapp2");
        assert_eq!(app_key("com.example.thing", "X"), "thing");
        assert_eq!(app_key("", ""), "");
    }
}
