//! macOS: CGEventTap hotkeys, Unicode CGEvent typing, NSPasteboard paste, focus.
//!

pub mod focus;
pub mod inject;
pub mod layout;
pub mod tap;

use objc2::runtime::Bool;
use objc2::{class, msg_send};
use objc2_foundation::{NSDictionary, NSNumber, NSString};

pub use inject::MacInjector;
pub use tap::MacHotkeys;

#[link(name = "ApplicationServices", kind = "framework")]
unsafe extern "C" {
    fn AXIsProcessTrusted() -> u8;
    fn AXIsProcessTrustedWithOptions(options: *const std::ffi::c_void) -> u8;
}

#[link(name = "AVFoundation", kind = "framework")]
unsafe extern "C" {
    static AVMediaTypeAudio: &'static NSString;
}

#[link(name = "Carbon", kind = "framework")]
unsafe extern "C" {
    fn IsSecureEventInputEnabled() -> u8;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFPreferencesCopyAppValue(
        key: *const std::ffi::c_void,
        app: *const std::ffi::c_void,
    ) -> *const std::ffi::c_void;
    fn CFNumberGetValue(
        number: *const std::ffi::c_void,
        kind: isize,
        out: *mut std::ffi::c_void,
    ) -> u8;
    fn CFRelease(cf: *const std::ffi::c_void);
}

/// Accessibility is granted to the process (needed to swallow keys and to post events).
pub fn accessibility_trusted() -> bool {
    // SAFETY: no arguments.
    unsafe { AXIsProcessTrusted() != 0 }
}

/// Show the system "allow Accessibility" prompt (no-op when already granted).
pub fn request_accessibility() {
    let key = NSString::from_str("AXTrustedCheckOptionPrompt"); // kAXTrustedCheckOptionPrompt
    let yes = NSNumber::new_bool(true);
    let options = NSDictionary::from_slices(&[&*key], &[&*yes]);
    // SAFETY: NSDictionary is toll-free bridged to CFDictionaryRef and outlives the call.
    unsafe {
        AXIsProcessTrustedWithOptions(
            &*options as *const NSDictionary<NSString, NSNumber> as *const _,
        );
    }
}

/// Secure input is on (a password field, or a terminal with "Secure Keyboard Entry"): macOS
/// silently drops synthetic keystrokes.
pub fn secure_input_enabled() -> bool {
    // SAFETY: no arguments.
    unsafe { IsSecureEventInputEnabled() != 0 }
}

/// `AVAuthorizationStatus` for the microphone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MicAccess {
    /// Never asked: macOS shows its prompt the first time the microphone opens.
    NotDetermined,
    /// Blocked by a profile (MDM / Screen Time); the user can't change it.
    Restricted,
    Denied,
    Granted,
}

pub fn microphone_access() -> MicAccess {
    // SAFETY: a class method taking an AVMediaType constant, returning an NSInteger.
    let status: isize = unsafe {
        msg_send![class!(AVCaptureDevice), authorizationStatusForMediaType: AVMediaTypeAudio]
    };
    match status {
        0 => MicAccess::NotDetermined,
        1 => MicAccess::Restricted,
        2 => MicAccess::Denied,
        _ => MicAccess::Granted,
    }
}

/// Show the system microphone prompt when it was never answered. Returns false when there is
/// no prompt to show (already denied or restricted: only System Settings can change it).
pub fn request_microphone() -> bool {
    match microphone_access() {
        MicAccess::Granted => true,
        MicAccess::NotDetermined => {
            let done = block2::RcBlock::new(|granted: Bool| {
                tracing::info!("microphone access: granted={}", granted.as_bool());
            });
            // SAFETY: the block is copied by AVFoundation and called once on its own queue.
            let () = unsafe {
                msg_send![class!(AVCaptureDevice), requestAccessForMediaType: AVMediaTypeAudio, completionHandler: &*done]
            };
            true
        }
        MicAccess::Denied | MicAccess::Restricted => false,
    }
}

/// What macOS does with the fn / Globe key (System Settings > Keyboard > "Press 🌐 key to"), when
/// that is anything but "Do Nothing". macOS acts on Globe below our event tap, so as the Voice key
/// it would also open the emoji picker, switch input source or start Apple Dictation.
pub fn globe_key_action() -> Option<&'static str> {
    let key = NSString::from_str("AppleFnUsageType");
    let domain = NSString::from_str("com.apple.HIToolbox");
    // SAFETY: NSString is toll-free bridged to CFString; Copy rule: released below.
    let value = unsafe {
        CFPreferencesCopyAppValue(
            &*key as *const NSString as *const _,
            &*domain as *const NSString as *const _,
        )
    };
    let mut usage: i64 = -1; // unset: the system default (emoji picker or input source switch)
    if !value.is_null() {
        // SAFETY: the value is a CFNumber; kCFNumberSInt64Type = 4.
        unsafe {
            CFNumberGetValue(value, 4, &mut usage as *mut i64 as *mut _);
            CFRelease(value);
        }
    }
    match usage {
        0 => None,
        1 => Some("change the input source"),
        3 => Some("start Apple Dictation"),
        _ => Some("show the emoji picker"),
    }
}
