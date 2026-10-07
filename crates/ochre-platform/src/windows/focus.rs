//! The focused app on Windows: executable name, window title, window id, elevation.

use std::sync::OnceLock;

use ochre_core::platform::FocusInfo;
use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND, LPARAM};
use windows::Win32::Security::{GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation};
use windows::Win32::System::Threading::{
    GetCurrentProcess, OpenProcess, OpenProcessToken, PROCESS_NAME_WIN32,
    PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumChildWindows, GUITHREADINFO, GetForegroundWindow, GetGUIThreadInfo, GetWindowTextLengthW,
    GetWindowTextW, GetWindowThreadProcessId,
};
use windows::core::{BOOL, PWSTR};

/// Elevation of a process: `Some(true)` elevated, `Some(false)` not, `None` when its token
/// cannot be read (protected or higher-integrity processes).
fn token_elevated(process: HANDLE) -> Option<bool> {
    let mut token = HANDLE::default();
    // SAFETY: valid out-pointers; the token handle is closed below.
    unsafe {
        OpenProcessToken(process, TOKEN_QUERY, &mut token).ok()?;
        let mut elevation = TOKEN_ELEVATION::default();
        let mut len = 0u32;
        let ok = GetTokenInformation(
            token,
            TokenElevation,
            Some(&mut elevation as *mut _ as *mut _),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut len,
        );
        let _ = CloseHandle(token);
        ok.ok()?;
        Some(elevation.TokenIsElevated != 0)
    }
}

/// Whether this process runs elevated (cached).
pub fn self_elevated() -> bool {
    static SELF: OnceLock<bool> = OnceLock::new();
    // SAFETY: pseudo-handle of our own process.
    *SELF.get_or_init(|| token_elevated(unsafe { GetCurrentProcess() }).unwrap_or(false))
}

/// Full image path and elevation of a process id.
fn process_info(pid: u32) -> (String, Option<bool>) {
    // SAFETY: the handle is closed before returning; buffers are sized by `len`.
    unsafe {
        let Ok(process) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return (String::new(), None);
        };
        let mut buf = vec![0u16; 32768];
        let mut len = buf.len() as u32;
        let path = match QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            PWSTR(buf.as_mut_ptr()),
            &mut len,
        ) {
            Ok(()) => String::from_utf16_lossy(&buf[..len as usize]),
            Err(_) => String::new(),
        };
        let elevated = token_elevated(process);
        let _ = CloseHandle(process);
        (path, elevated)
    }
}

/// `C:\...\Slack.exe` -> `slack` (matches the keys in `RefineConfig.app_styles`).
pub fn app_name_from_path(path: &str) -> String {
    let name = path.rsplit(['\\', '/']).next().unwrap_or("").to_lowercase();
    name.strip_suffix(".exe")
        .map(str::to_string)
        .unwrap_or(name)
}

fn pid_of(hwnd: HWND) -> u32 {
    let mut pid = 0u32;
    // SAFETY: valid out-pointer.
    unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
    pid
}

fn title_of(hwnd: HWND) -> String {
    // SAFETY: buffer sized from GetWindowTextLengthW.
    unsafe {
        let n = GetWindowTextLengthW(hwnd);
        if n <= 0 {
            return String::new();
        }
        let mut buf = vec![0u16; n as usize + 1];
        let got = GetWindowTextW(hwnd, &mut buf);
        String::from_utf16_lossy(&buf[..got.max(0) as usize])
    }
}

/// The control with keyboard focus inside a window's thread.
pub fn focused_control(hwnd: HWND) -> HWND {
    // SAFETY: valid struct pointer with cbSize set.
    unsafe {
        let tid = GetWindowThreadProcessId(hwnd, None);
        let mut info = GUITHREADINFO {
            cbSize: std::mem::size_of::<GUITHREADINFO>() as u32,
            ..Default::default()
        };
        if tid != 0 && GetGUIThreadInfo(tid, &mut info).is_ok() {
            info.hwndFocus
        } else {
            HWND::default()
        }
    }
}

/// UWP apps live inside ApplicationFrameHost; the real app owns a child CoreWindow.
fn uwp_child_pid(hwnd: HWND, host_pid: u32) -> u32 {
    struct Search {
        host: u32,
        found: u32,
    }
    unsafe extern "system" fn visit(child: HWND, lparam: LPARAM) -> BOOL {
        // SAFETY: lparam is the &mut Search passed below, alive for the call.
        let search = unsafe { &mut *(lparam.0 as *mut Search) };
        let pid = pid_of(child);
        if pid != 0 && pid != search.host {
            search.found = pid;
            return BOOL(0);
        }
        BOOL(1)
    }
    let mut search = Search {
        host: host_pid,
        found: 0,
    };
    // SAFETY: the callback only touches `search`, which outlives the call.
    unsafe {
        let _ = EnumChildWindows(
            Some(hwnd),
            Some(visit),
            LPARAM(&mut search as *mut Search as isize),
        );
    }
    search.found
}

pub fn get_focus() -> FocusInfo {
    // SAFETY: no arguments.
    let hwnd = unsafe { GetForegroundWindow() };
    if hwnd.is_invalid() {
        return FocusInfo::default();
    }
    let mut pid = pid_of(hwnd);
    let (mut path, mut elevated) = process_info(pid);
    if app_name_from_path(&path) == "applicationframehost" {
        let child = uwp_child_pid(hwnd, pid);
        if child != 0 {
            pid = child;
            let (p, e) = process_info(child);
            if !p.is_empty() {
                path = p;
            }
            elevated = e;
        }
    }
    // A token we may not even read belongs to a higher-integrity process: UIPI blocks it too.
    let target_elevated = pid != 0 && elevated.unwrap_or(true);
    FocusInfo {
        app_name: app_name_from_path(&path),
        window_title: title_of(hwnd),
        window_id: format!(
            "{:x}:{:x}",
            hwnd.0 as usize,
            focused_control(hwnd).0 as usize
        ),
        elevated: target_elevated && !self_elevated(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_names() {
        assert_eq!(
            app_name_from_path(r"C:\Program Files\Slack\Slack.exe"),
            "slack"
        );
        assert_eq!(
            app_name_from_path(r"C:\x\WindowsTerminal.exe"),
            "windowsterminal"
        );
        assert_eq!(app_name_from_path("code"), "code");
        assert_eq!(app_name_from_path(""), "");
    }

    #[test]
    fn focus_of_this_desktop_does_not_panic() {
        let f = get_focus();
        assert!(f.window_id.is_empty() || f.window_id.contains(':'));
    }
}
