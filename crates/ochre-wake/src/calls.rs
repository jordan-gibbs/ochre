//! Is another app using the microphone (a call)? Hands-free listening pauses while one is.
//! Port of `src/openwhisprflow/wake/calls.py`.
//!
//! Windows records every microphone user in the privacy consent store,
//! `HKCU\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone`
//! (and the same path under HKLM). Each packaged app has a subkey (`Microsoft.Teams_8wekyb3d8bbwe`);
//! each desktop app has a subkey of `NonPackaged` named by its exe path with `\` written as `#`.
//! `LastUsedTimeStart` / `LastUsedTimeStop` are FILETIMEs; an app holds the mic now when `Stop` is 0
//! or older than `Start`. Browsers show up as the browser (Meet in Chrome = chrome.exe).
//!
//! Limits: it says *that* an app holds the mic, not that it is a call (`ignore_apps` lists always-on
//! ones such as mic-effects apps); a crashed app can leave a stale entry, so entries started more
//! than [`STALE_H`] hours ago are ignored. Our own exe is excluded. Registry reads only; no admin.
//!
//! **macOS** (14+): CoreAudio lists every process doing audio
//! (`kAudioHardwarePropertyProcessObjectList`) and whether it is capturing right now
//! (`kAudioProcessPropertyIsRunningInput`). The device-wide "running somewhere" flag is useless here:
//! our own warm mic keeps it on. System daemons that always hold the input (`corespeechd` for "Hey
//! Siri", `historicalaudiod`) live under `/System` or `/usr` and never count. Entries are exe paths,
//! so `ignore_apps` matches them as on Windows. Older macOS reports an error once (no pausing).
//!
//! Elsewhere `read_consent_store` returns nothing. TODO: Linux via PipeWire/PulseAudio
//! source-outputs.

use std::time::{SystemTime, UNIX_EPOCH};

pub const CONSENT_KEY: &str =
    r"Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone";
/// An "in use" entry started longer ago than this is a crashed app, not a call.
pub const STALE_H: f64 = 12.0;
const EPOCH_DIFF: f64 = 11_644_473_600.0; // seconds between 1601-01-01 and 1970-01-01

/// One consent-store entry: `app` = package name or exe path (`\` separators); times are FILETIMEs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MicUse {
    pub app: String,
    pub start: u64,
    pub stop: u64,
}

impl MicUse {
    pub fn active(&self) -> bool {
        self.start > 0 && (self.stop == 0 || self.stop < self.start)
    }

    pub fn started_unix(&self) -> f64 {
        if self.start == 0 {
            0.0
        } else {
            self.start as f64 / 1e7 - EPOCH_DIFF
        }
    }

    /// Display name: "Zoom", "chrome", "Microsoft.Teams".
    pub fn short(&self) -> String {
        let norm = self.app.replace('/', "\\");
        let mut a = norm.rsplit('\\').next().unwrap_or("").to_string();
        if a.to_lowercase().ends_with(".exe") {
            a.truncate(a.len() - 4);
        }
        if norm.contains('\\') {
            a
        } else {
            a.split('_').next().unwrap_or("").to_string()
        }
    }
}

/// What the hands-free controller polls: display names of other apps holding the mic now.
pub trait MicMonitor: Send {
    fn in_use(&mut self) -> Vec<String>;
}

pub type Reader = Box<dyn FnMut() -> std::io::Result<Vec<MicUse>> + Send>;

/// The real monitor (consent store on Windows; never reports anything elsewhere).
pub struct MicUsageMonitor {
    reader: Reader,
    ignore: Vec<String>,
    own: Vec<String>,
    now: Box<dyn Fn() -> f64 + Send>,
    warned: bool,
}

fn norm_path(p: &str) -> String {
    p.replace('/', "\\").to_lowercase()
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

impl Default for MicUsageMonitor {
    fn default() -> Self {
        Self::new(&[])
    }
}

impl MicUsageMonitor {
    /// `ignore_apps`: case-insensitive substrings of app ids that never count (mic-effects apps).
    pub fn new(ignore_apps: &[String]) -> Self {
        let own = std::env::current_exe()
            .ok()
            .map(|p| vec![norm_path(&p.to_string_lossy())])
            .unwrap_or_default();
        Self::with_reader(
            Box::new(read_consent_store),
            ignore_apps,
            own,
            Box::new(unix_now),
        )
    }

    pub fn with_reader(
        reader: Reader,
        ignore_apps: &[String],
        own_paths: Vec<String>,
        now: Box<dyn Fn() -> f64 + Send>,
    ) -> Self {
        MicUsageMonitor {
            reader,
            ignore: ignore_apps
                .iter()
                .filter(|a| !a.is_empty())
                .map(|a| a.to_lowercase())
                .collect(),
            own: own_paths.iter().map(|p| norm_path(p)).collect(),
            now,
            warned: false,
        }
    }

    /// Entries of other apps holding the mic right now.
    pub fn active_entries(&mut self) -> Vec<MicUse> {
        let entries = match (self.reader)() {
            Ok(e) => e,
            Err(e) => {
                if !self.warned {
                    self.warned = true;
                    tracing::warn!(
                        "microphone usage unavailable ({e}): hands-free won't pause for calls"
                    );
                }
                return Vec::new();
            }
        };
        let stale = (self.now)() - STALE_H * 3600.0;
        entries
            .into_iter()
            .filter(|e| e.active() && e.started_unix() >= stale)
            .filter(|e| {
                let low = e.app.to_lowercase();
                !self.own.contains(&norm_path(&e.app))
                    && !self.ignore.iter().any(|i| low.contains(i))
            })
            .collect()
    }
}

impl MicMonitor for MicUsageMonitor {
    fn in_use(&mut self) -> Vec<String> {
        self.active_entries().iter().map(MicUse::short).collect()
    }
}

/// All microphone consent-store entries (HKCU + HKLM, packaged + NonPackaged).
#[cfg(windows)]
pub fn read_consent_store() -> std::io::Result<Vec<MicUse>> {
    use winreg::RegKey;
    use winreg::enums::{HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE, KEY_READ};

    fn entry(parent: &RegKey, name: &str, app: String, out: &mut Vec<MicUse>) {
        let Ok(k) = parent.open_subkey_with_flags(name, KEY_READ) else {
            return;
        };
        let start: u64 = k.get_value("LastUsedTimeStart").unwrap_or(0);
        let stop: u64 = k.get_value("LastUsedTimeStop").unwrap_or(0);
        if start != 0 || stop != 0 {
            out.push(MicUse { app, start, stop });
        }
    }

    let mut out = Vec::new();
    for hive in [HKEY_CURRENT_USER, HKEY_LOCAL_MACHINE] {
        let Ok(root) = RegKey::predef(hive).open_subkey_with_flags(CONSENT_KEY, KEY_READ) else {
            continue;
        };
        for sub in root.enum_keys().filter_map(|k| k.ok()) {
            if sub == "NonPackaged" {
                if let Ok(np) = root.open_subkey_with_flags(&sub, KEY_READ) {
                    for exe in np.enum_keys().filter_map(|k| k.ok()) {
                        let app = exe.replace('#', "\\");
                        entry(&np, &exe, app, &mut out);
                    }
                }
            } else {
                entry(&root, &sub, sub.clone(), &mut out);
            }
        }
    }
    Ok(out)
}

/// macOS: the processes other than us capturing audio now, as active entries (exe paths).
#[cfg(target_os = "macos")]
pub fn read_consent_store() -> std::io::Result<Vec<MicUse>> {
    let now = ((unix_now() + EPOCH_DIFF) * 1e7) as u64;
    let me = std::process::id() as i32;
    Ok(mac::capturing_pids()?
        .into_iter()
        .filter(|&pid| pid != me)
        .filter_map(mac::exe_path)
        .filter(|path| mac::counts(path))
        .map(|app| MicUse {
            app,
            start: now,
            stop: 0,
        })
        .collect())
}

/// Linux: never reports a call (see the module docs for the TODO).
#[cfg(not(any(windows, target_os = "macos")))]
pub fn read_consent_store() -> std::io::Result<Vec<MicUse>> {
    Ok(Vec::new())
}

/// System processes that hold the input all the time ("Hey Siri") are not calls.
#[cfg(any(target_os = "macos", test))]
mod mac_rules {
    pub fn counts(path: &str) -> bool {
        !(path.is_empty() || path.starts_with("/System/") || path.starts_with("/usr/"))
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use std::ffi::c_void;
    use std::io::{Error, Result};

    pub use super::mac_rules::counts;

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
    const PROCESS_OBJECT_LIST: u32 = fourcc(b"prs#");
    const PROCESS_PID: u32 = fourcc(b"ppid");
    const PROCESS_RUNNING_INPUT: u32 = fourcc(b"piri");

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

    fn addr(selector: u32) -> Address {
        Address {
            selector,
            scope: SCOPE_GLOBAL,
            element: 0,
        }
    }

    fn get<T: Copy + Default>(obj: u32, selector: u32) -> Option<T> {
        let mut v = T::default();
        let mut size = std::mem::size_of::<T>() as u32;
        // SAFETY: `v` is a plain value of the size we pass.
        let st = unsafe {
            AudioObjectGetPropertyData(
                obj,
                &addr(selector),
                0,
                std::ptr::null(),
                &mut size,
                &mut v as *mut T as *mut c_void,
            )
        };
        (st == 0).then_some(v)
    }

    pub fn capturing_pids() -> Result<Vec<i32>> {
        let a = addr(PROCESS_OBJECT_LIST);
        let mut size = 0u32;
        // SAFETY: size query on the system object.
        let st = unsafe {
            AudioObjectGetPropertyDataSize(SYSTEM_OBJECT, &a, 0, std::ptr::null(), &mut size)
        };
        if st != 0 {
            return Err(Error::other(format!(
                "CoreAudio process list unavailable (OSStatus {st}; needs macOS 14)"
            )));
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
            return Err(Error::other(format!(
                "CoreAudio process list: OSStatus {st}"
            )));
        }
        ids.truncate(size as usize / 4);
        Ok(ids
            .into_iter()
            .filter(|&id| get::<u32>(id, PROCESS_RUNNING_INPUT).unwrap_or(0) != 0)
            .filter_map(|id| get::<i32>(id, PROCESS_PID))
            .collect())
    }

    pub fn exe_path(pid: i32) -> Option<String> {
        let mut buf = vec![0u8; libc::PROC_PIDPATHINFO_MAXSIZE as usize];
        // SAFETY: `buf` is PROC_PIDPATHINFO_MAXSIZE bytes.
        let n =
            unsafe { libc::proc_pidpath(pid, buf.as_mut_ptr() as *mut c_void, buf.len() as u32) };
        (n > 0).then(|| String::from_utf8_lossy(&buf[..n as usize]).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ft(unix: f64) -> u64 {
        ((unix + EPOCH_DIFF) * 1e7) as u64
    }

    #[test]
    fn mac_system_daemons_never_count() {
        assert!(!mac_rules::counts(
            "/System/Library/PrivateFrameworks/CoreSpeech.framework/corespeechd"
        ));
        assert!(!mac_rules::counts("/usr/libexec/historicalaudiod"));
        assert!(!mac_rules::counts(""));
        assert!(mac_rules::counts(
            "/Applications/zoom.us.app/Contents/MacOS/zoom.us"
        ));
        let m = MicUse {
            app: "/Applications/zoom.us.app/Contents/MacOS/zoom.us".into(),
            start: 1,
            stop: 0,
        };
        assert_eq!(m.short(), "zoom.us");
    }

    #[test]
    fn short_names() {
        let m = |a: &str| {
            MicUse {
                app: a.into(),
                start: 1,
                stop: 0,
            }
            .short()
        };
        assert_eq!(m(r"C:\Program Files\Zoom\bin\Zoom.exe"), "Zoom");
        assert_eq!(m("Microsoft.Teams_8wekyb3d8bbwe"), "Microsoft.Teams");
        assert_eq!(m(r"C:\x\chrome.EXE"), "chrome");
    }

    #[test]
    fn filters_inactive_stale_own_and_ignored() {
        let now = 1_800_000_000.0;
        let entries = vec![
            MicUse {
                app: r"C:\Zoom\Zoom.exe".into(),
                start: ft(now - 60.0),
                stop: 0,
            },
            MicUse {
                app: r"C:\old\Crashed.exe".into(),
                start: ft(now - 13.0 * 3600.0),
                stop: 0,
            },
            MicUse {
                app: r"C:\done\Done.exe".into(),
                start: ft(now - 600.0),
                stop: ft(now - 500.0),
            },
            MicUse {
                app: r"C:\me\ochre.exe".into(),
                start: ft(now - 5.0),
                stop: 0,
            },
            MicUse {
                app: r"C:\nv\NVIDIA Broadcast.exe".into(),
                start: ft(now - 5.0),
                stop: 0,
            },
            MicUse {
                app: "Microsoft.Teams_8wekyb3d8bbwe".into(),
                start: ft(now - 30.0),
                stop: ft(now - 3000.0),
            },
        ];
        let mut m = MicUsageMonitor::with_reader(
            Box::new(move || Ok(entries.clone())),
            &["nvidia broadcast".to_string()],
            vec!["c:/me/OCHRE.exe".into()],
            Box::new(move || now),
        );
        assert_eq!(
            m.in_use(),
            vec!["Zoom".to_string(), "Microsoft.Teams".to_string()]
        );
    }

    #[test]
    fn reader_error_means_no_call() {
        let mut m = MicUsageMonitor::with_reader(
            Box::new(|| Err(std::io::Error::other("nope"))),
            &[],
            vec![],
            Box::new(unix_now),
        );
        assert!(m.in_use().is_empty());
    }

    #[test]
    fn real_store_reads() {
        // smoke test: must not fail on this machine (empty off Windows)
        let entries = read_consent_store().expect("consent store");
        let in_use = MicUsageMonitor::default().in_use();
        eprintln!(
            "{} consent-store entries; in use: {in_use:?}",
            entries.len()
        );
    }
}
