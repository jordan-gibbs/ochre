//! macOS: which input device to capture from when none is configured.
//!
//! Capturing a Bluetooth headset's microphone switches it from its music profile to the call
//! profile (HFP): everything the user hears drops to phone quality for as long as the mic is open,
//! and the headset mic is worse for dictation anyway. So when the system default input is a
//! Bluetooth device, Ochre records from the Mac's built-in microphone instead (if it has one),
//! unless the user turned that off or picked the headset by name.
//!
//! CoreAudio queries only (no CoreFoundation strings beyond device names); cheap enough for the
//! capture watchdog's periodic check.

use std::ffi::c_void;

use objc2_foundation::NSString;

#[repr(C)]
struct Address {
    selector: u32,
    scope: u32,
    element: u32,
}

const fn fourcc(s: &[u8; 4]) -> u32 {
    u32::from_be_bytes(*s)
}

const SYSTEM_OBJECT: u32 = 1;
const SCOPE_GLOBAL: u32 = fourcc(b"glob");
const SCOPE_INPUT: u32 = fourcc(b"inpt");
const DEVICES: u32 = fourcc(b"dev#");
const DEFAULT_INPUT: u32 = fourcc(b"dIn ");
const TRANSPORT: u32 = fourcc(b"tran");
const STREAMS: u32 = fourcc(b"stm#");
const NAME: u32 = fourcc(b"lnam");

pub const TRANSPORT_BUILT_IN: u32 = fourcc(b"bltn");
pub const TRANSPORT_BLUETOOTH: u32 = fourcc(b"blue");
pub const TRANSPORT_BLUETOOTH_LE: u32 = fourcc(b"blea");

#[link(name = "CoreAudio", kind = "framework")]
unsafe extern "C" {
    fn AudioObjectGetPropertyDataSize(
        obj: u32,
        addr: *const Address,
        qsize: u32,
        qdata: *const c_void,
        size: *mut u32,
    ) -> i32;
    fn AudioObjectGetPropertyData(
        obj: u32,
        addr: *const Address,
        qsize: u32,
        qdata: *const c_void,
        size: *mut u32,
        data: *mut c_void,
    ) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFRelease(cf: *const c_void);
}

fn addr(selector: u32, scope: u32) -> Address {
    Address {
        selector,
        scope,
        element: 0,
    }
}

fn get_u32(obj: u32, selector: u32) -> Option<u32> {
    let mut v = 0u32;
    let mut size = 4u32;
    // SAFETY: a 4-byte property read into a u32.
    let st = unsafe {
        AudioObjectGetPropertyData(
            obj,
            &addr(selector, SCOPE_GLOBAL),
            0,
            std::ptr::null(),
            &mut size,
            &mut v as *mut u32 as *mut c_void,
        )
    };
    (st == 0).then_some(v)
}

fn has_input(id: u32) -> bool {
    let mut size = 0u32;
    // SAFETY: size query.
    let st = unsafe {
        AudioObjectGetPropertyDataSize(
            id,
            &addr(STREAMS, SCOPE_INPUT),
            0,
            std::ptr::null(),
            &mut size,
        )
    };
    st == 0 && size > 0
}

/// The device's name, as cpal reports it (`kAudioObjectPropertyName`).
pub fn name(id: u32) -> Option<String> {
    let mut cf: *const c_void = std::ptr::null();
    let mut size = std::mem::size_of::<*const c_void>() as u32;
    // SAFETY: the property is a CFStringRef we own (Copy rule) and release below.
    let st = unsafe {
        AudioObjectGetPropertyData(
            id,
            &addr(NAME, SCOPE_GLOBAL),
            0,
            std::ptr::null(),
            &mut size,
            &mut cf as *mut *const c_void as *mut c_void,
        )
    };
    if st != 0 || cf.is_null() {
        return None;
    }
    // SAFETY: CFString is toll-free bridged to NSString.
    let s = unsafe { (*(cf as *const NSString)).to_string() };
    // SAFETY: we own the reference.
    unsafe { CFRelease(cf) };
    Some(s)
}

pub fn transport(id: u32) -> Option<u32> {
    get_u32(id, TRANSPORT)
}

pub fn is_bluetooth(transport: u32) -> bool {
    transport == TRANSPORT_BLUETOOTH || transport == TRANSPORT_BLUETOOTH_LE
}

pub fn default_input() -> Option<u32> {
    get_u32(SYSTEM_OBJECT, DEFAULT_INPUT).filter(|&id| id != 0)
}

fn input_devices() -> Vec<u32> {
    let a = addr(DEVICES, SCOPE_GLOBAL);
    let mut size = 0u32;
    // SAFETY: size query on the system object.
    if unsafe { AudioObjectGetPropertyDataSize(SYSTEM_OBJECT, &a, 0, std::ptr::null(), &mut size) }
        != 0
    {
        return Vec::new();
    }
    let mut ids = vec![0u32; size as usize / 4];
    // SAFETY: `ids` holds `size` bytes.
    let st = unsafe {
        AudioObjectGetPropertyData(
            SYSTEM_OBJECT,
            &a,
            0,
            std::ptr::null(),
            &mut size,
            ids.as_mut_ptr() as *mut c_void,
        )
    };
    if st != 0 {
        return Vec::new();
    }
    ids.truncate(size as usize / 4);
    ids.retain(|&id| has_input(id));
    ids
}

/// The Mac's own microphone, if it has one (Mac mini / Studio / Pro have none).
pub fn built_in_input() -> Option<u32> {
    input_devices()
        .into_iter()
        .find(|&id| transport(id) == Some(TRANSPORT_BUILT_IN))
}

/// Where to capture from when no device is configured.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Route {
    pub id: u32,
    pub name: String,
    /// The default input is a Bluetooth headset and this is the built-in mic instead.
    pub avoided_bluetooth: bool,
}

/// The default input, or the built-in mic in its place when the default is Bluetooth and
/// `avoid_bluetooth` is on. None when there is no input at all.
pub fn default_route(avoid_bluetooth: bool) -> Option<Route> {
    let def = default_input()?;
    let def_name = name(def).unwrap_or_default();
    if avoid_bluetooth && transport(def).is_some_and(is_bluetooth) {
        match built_in_input() {
            Some(b) => {
                return Some(Route {
                    id: b,
                    name: name(b).unwrap_or_default(),
                    avoided_bluetooth: true,
                });
            }
            None => tracing::info!(
                device = %def_name,
                "the default input is a Bluetooth headset and this Mac has no built-in microphone: using the headset"
            ),
        }
    }
    Some(Route {
        id: def,
        name: def_name,
        avoided_bluetooth: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_route_is_a_named_input() {
        // Runs on any Mac (CI has a virtual or no input; then None is fine).
        if let Some(r) = default_route(true) {
            assert!(!r.name.is_empty());
            assert!(has_input(r.id));
            if r.avoided_bluetooth {
                assert_eq!(transport(r.id), Some(TRANSPORT_BUILT_IN));
            }
        }
        assert!(is_bluetooth(TRANSPORT_BLUETOOTH_LE) && !is_bluetooth(TRANSPORT_BUILT_IN));
    }
}
