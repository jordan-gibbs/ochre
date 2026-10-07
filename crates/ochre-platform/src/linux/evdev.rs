//! Wayland (and fallback) Voice key listener: reads `/dev/input/event*` directly.
//!
//! Wayland gives clients no global key access, so we read the kernel's input devices. That
//! needs membership in the `input` group (`sudo usermod -aG input $USER`, then log out and back
//! in), which is a real privilege: any member can read every keystroke. Devices are opened
//! without a grab (keys still reach the desktop) and re-scanned every 5 s for hot-plugging. Our
//! own virtual devices (ydotoold, dotool) are skipped so injected text never looks like typing.

use std::ffi::CString;
use std::os::fd::RawFd;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ochre_core::config::HotkeyConfig;
use ochre_core::platform::{GestureFn, HotkeyListener};
use ochre_core::{Error, Result};

use crate::driver::{Driver, monotonic_ms};
use crate::keys;
use crate::permissions::TOGGLE_HINT;

const DEVICE_DIR: &str = "/dev/input";
const RESCAN: Duration = Duration::from_secs(5);
const EV_KEY: u16 = 1;
const SKIP_NAMES: [&str; 4] = ["ydotoold", "dotool", "ochre", "wtype"];

/// One EV_KEY event out of a raw `struct input_event` buffer: (code, value) where value is
/// 0 up, 1 down, 2 auto-repeat.
pub fn parse_events(buf: &[u8]) -> Vec<(u16, i32)> {
    let size = std::mem::size_of::<libc::input_event>();
    let mut out = Vec::new();
    for chunk in buf.chunks_exact(size) {
        // SAFETY: the chunk is exactly one input_event; read_unaligned copes with alignment.
        let ev: libc::input_event =
            unsafe { std::ptr::read_unaligned(chunk.as_ptr() as *const libc::input_event) };
        if ev.type_ == EV_KEY {
            out.push((ev.code, ev.value));
        }
    }
    out
}

fn event_paths() -> Vec<String> {
    let Ok(dir) = std::fs::read_dir(DEVICE_DIR) else {
        return Vec::new();
    };
    let mut out: Vec<String> = dir
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("event"))
        })
        .filter_map(|p| p.to_str().map(str::to_string))
        .collect();
    out.sort();
    out
}

/// Some(true) when at least one input device is readable, Some(false) when devices exist but
/// none is, None when there are none.
pub fn any_readable() -> Option<bool> {
    let paths = event_paths();
    if paths.is_empty() {
        return None;
    }
    Some(paths.iter().any(|p| {
        CString::new(p.as_str()).is_ok_and(|c| {
            // SAFETY: valid NUL-terminated path.
            unsafe { libc::access(c.as_ptr(), libc::R_OK) == 0 }
        })
    }))
}

fn device_name(fd: RawFd) -> String {
    let mut buf = [0u8; 256];
    // EVIOCGNAME(len) = _IOC(_IOC_READ, 'E', 0x06, len)
    let request = (2u64 << 30) | ((buf.len() as u64) << 16) | ((b'E' as u64) << 8) | 0x06;
    // SAFETY: the buffer is as long as the request says.
    let n = unsafe { libc::ioctl(fd, request as _, buf.as_mut_ptr()) };
    if n <= 0 {
        return String::new();
    }
    let end = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).to_lowercase()
}

struct Devices {
    open: Vec<(String, RawFd)>,
}

impl Devices {
    fn rescan(&mut self) {
        for path in event_paths() {
            if self.open.iter().any(|(p, _)| *p == path) {
                continue;
            }
            let Ok(c) = CString::new(path.as_str()) else {
                continue;
            };
            // SAFETY: valid path; the fd is closed in `close`/`drop`.
            let fd = unsafe {
                libc::open(
                    c.as_ptr(),
                    libc::O_RDONLY | libc::O_NONBLOCK | libc::O_CLOEXEC,
                )
            };
            if fd < 0 {
                continue;
            }
            let name = device_name(fd);
            if SKIP_NAMES.iter().any(|s| name.contains(s)) {
                // SAFETY: our own fd.
                unsafe { libc::close(fd) };
                continue;
            }
            self.open.push((path, fd));
        }
    }

    fn close(&mut self, fd: RawFd) {
        self.open.retain(|(_, f)| *f != fd);
        // SAFETY: our own fd.
        unsafe { libc::close(fd) };
    }
}

impl Drop for Devices {
    fn drop(&mut self) {
        for (_, fd) in self.open.drain(..) {
            // SAFETY: our own fds.
            unsafe { libc::close(fd) };
        }
    }
}

struct Running {
    thread: JoinHandle<()>,
    stop_w: RawFd,
}

/// [`HotkeyListener`] reading `/dev/input/event*`.
pub struct EvdevHotkeys {
    driver: Arc<Driver>,
    running: Mutex<Option<Running>>,
}

impl EvdevHotkeys {
    pub fn new(cfg: &HotkeyConfig) -> Result<Self> {
        Ok(Self {
            driver: Driver::new(cfg, monotonic_ms, super::validate)?,
            running: Mutex::new(None),
        })
    }

    /// Fails with a fix-it message unless at least one input device is readable.
    pub fn check() -> Result<()> {
        match any_readable() {
            None => Err(Error::Permission(format!(
                "No input devices found in {DEVICE_DIR}. {TOGGLE_HINT}"
            ))),
            Some(false) => Err(Error::Permission(format!(
                "Reading the Voice key on Wayland needs access to {DEVICE_DIR}. Run `sudo usermod -aG input $USER`, \
then log out and back in. {TOGGLE_HINT}"
            ))),
            Some(true) => Ok(()),
        }
    }
}

impl HotkeyListener for EvdevHotkeys {
    fn start(&mut self, on_gesture: GestureFn) -> Result<()> {
        let mut slot = self.running.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_some() {
            return Ok(());
        }
        Self::check()?;
        let mut pipe = [0 as RawFd; 2];
        // SAFETY: valid array of two fds.
        if unsafe { libc::pipe(pipe.as_mut_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        self.driver.start(on_gesture)?;
        let driver = Arc::clone(&self.driver);
        let stop_r = pipe[0];
        let thread = std::thread::Builder::new()
            .name("ochre-evdev".into())
            .spawn(move || run(driver, stop_r))?;
        *slot = Some(Running {
            thread,
            stop_w: pipe[1],
        });
        Ok(())
    }

    fn set_key(&mut self, key: &str) -> Result<()> {
        self.driver.set_key(key)
    }

    fn set_recording(&self, recording: bool) {
        self.driver.set_recording(recording);
    }

    fn stop(&mut self) {
        let running = self
            .running
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(r) = running {
            // SAFETY: writing one byte to our own pipe, then closing it.
            unsafe {
                libc::write(r.stop_w, b"x".as_ptr().cast(), 1);
            }
            let _ = r.thread.join();
            // SAFETY: our own fd.
            unsafe { libc::close(r.stop_w) };
            self.driver.stop();
            self.driver.reset();
        }
    }
}

impl Drop for EvdevHotkeys {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(driver: Arc<Driver>, stop_r: RawFd) {
    let mut devices = Devices { open: Vec::new() };
    devices.rescan();
    let mut next_scan = Instant::now() + RESCAN;
    let stopped = AtomicBool::new(false);
    let mut buf = vec![0u8; std::mem::size_of::<libc::input_event>() * 64];
    while !stopped.load(Ordering::Relaxed) {
        let mut fds: Vec<libc::pollfd> = devices
            .open
            .iter()
            .map(|(_, fd)| libc::pollfd {
                fd: *fd,
                events: libc::POLLIN,
                revents: 0,
            })
            .collect();
        fds.push(libc::pollfd {
            fd: stop_r,
            events: libc::POLLIN,
            revents: 0,
        });
        let timeout = next_scan
            .saturating_duration_since(Instant::now())
            .as_millis() as i32;
        // SAFETY: valid pollfd array.
        let n = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as _, timeout.max(1)) };
        if n < 0 {
            if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                continue;
            }
            break;
        }
        let mut dead = Vec::new();
        for p in &fds {
            if p.revents == 0 {
                continue;
            }
            if p.fd == stop_r {
                stopped.store(true, Ordering::Relaxed);
                break;
            }
            if p.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                dead.push(p.fd); // unplugged
                continue;
            }
            // SAFETY: reading into our buffer from our own fd.
            let got = unsafe { libc::read(p.fd, buf.as_mut_ptr().cast(), buf.len()) };
            if got <= 0 {
                if got < 0
                    && std::io::Error::last_os_error().kind() != std::io::ErrorKind::WouldBlock
                {
                    dead.push(p.fd);
                }
                continue;
            }
            for (code, value) in parse_events(&buf[..got as usize]) {
                driver.feed(keys::evdev_name(code), value != 0, monotonic_ms());
            }
        }
        for fd in dead {
            devices.close(fd);
        }
        if Instant::now() >= next_scan {
            next_scan = Instant::now() + RESCAN;
            devices.rescan();
        }
    }
    // SAFETY: our own fd.
    unsafe { libc::close(stop_r) };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_key_events_only() {
        let mk = |type_: u16, code: u16, value: i32| {
            let ev = libc::input_event {
                time: libc::timeval {
                    tv_sec: 0,
                    tv_usec: 0,
                },
                type_,
                code,
                value,
            };
            // SAFETY: plain-old-data struct to bytes.
            unsafe {
                std::slice::from_raw_parts(
                    &ev as *const _ as *const u8,
                    std::mem::size_of::<libc::input_event>(),
                )
                .to_vec()
            }
        };
        let mut buf = mk(EV_KEY, 100, 1);
        buf.extend(mk(0, 0, 0)); // EV_SYN
        buf.extend(mk(EV_KEY, 100, 0));
        assert_eq!(parse_events(&buf), vec![(100, 1), (100, 0)]);
        assert_eq!(keys::evdev_name(100), "right_alt");
    }
}
