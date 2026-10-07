//! X11 Voice key listener: XInput2 raw key events on the root window (`x11rb`, pure Rust, no
//! libX11 linking). Raw events report every key on every keyboard without a grab and without
//! permissions, but cannot swallow anything (see the module docs of [`crate::linux`]).

use std::os::fd::{AsRawFd, RawFd};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use ochre_core::config::HotkeyConfig;
use ochre_core::platform::{GestureFn, HotkeyListener};
use ochre_core::{Error, Result};
use x11rb::connection::Connection;
use x11rb::protocol::Event;
use x11rb::protocol::xinput::{self, ConnectionExt as _, EventMask, XIEventMask};
use x11rb::rust_connection::RustConnection;

use crate::driver::{Driver, monotonic_ms};
use crate::keys;

/// XIAllMasterDevices.
const ALL_MASTER_DEVICES: u16 = 1;

/// The X server is reachable and speaks XInput 2.
pub fn available() -> bool {
    connect().is_ok()
}

fn connect() -> Result<(RustConnection, u32)> {
    let (conn, screen) = RustConnection::connect(None)
        .map_err(|e| Error::Other(format!("cannot open the X display: {e}")))?;
    let root = conn
        .setup()
        .roots
        .get(screen)
        .map(|s| s.root)
        .ok_or_else(|| Error::Other("no X screen".into()))?;
    // Ask for XI 2.2: an XI 2.0 client gets no raw events at all while any client holds a
    // keyboard grab (the X server filters them), so the Voice key went unseen whenever GNOME
    // Shell or an app had grabbed the keyboard. From 2.1 raw events are delivered regardless.
    let version = conn
        .xinput_xi_query_version(2, 2)
        .map_err(|e| Error::Other(e.to_string()))?
        .reply()
        .map_err(|e| Error::Other(format!("the X server has no XInput 2: {e}")))?;
    if version.major_version < 2 {
        return Err(Error::Other("the X server has no XInput 2".into()));
    }
    Ok((conn, root))
}

struct Running {
    thread: JoinHandle<()>,
    stop_w: RawFd,
}

/// [`HotkeyListener`] on XInput2 raw key events.
pub struct X11Hotkeys {
    driver: Arc<Driver>,
    running: Mutex<Option<Running>>,
}

impl X11Hotkeys {
    pub fn new(cfg: &HotkeyConfig) -> Result<Self> {
        Ok(Self {
            driver: Driver::new(cfg, monotonic_ms, super::validate)?,
            running: Mutex::new(None),
        })
    }
}

impl HotkeyListener for X11Hotkeys {
    fn start(&mut self, on_gesture: GestureFn) -> Result<()> {
        let mut slot = self.running.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_some() {
            return Ok(());
        }
        let (conn, root) = connect()?;
        let mask = EventMask {
            deviceid: ALL_MASTER_DEVICES,
            mask: vec![XIEventMask::RAW_KEY_PRESS | XIEventMask::RAW_KEY_RELEASE],
        };
        conn.xinput_xi_select_events(root, &[mask])
            .map_err(|e| Error::Other(e.to_string()))?
            .check()
            .map_err(|e| Error::Other(format!("XISelectEvents failed: {e}")))?;
        let mut pipe = [0 as RawFd; 2];
        // SAFETY: valid array of two fds.
        if unsafe { libc::pipe(pipe.as_mut_ptr()) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        self.driver.start(on_gesture)?;
        let driver = Arc::clone(&self.driver);
        let stop_r = pipe[0];
        let thread = std::thread::Builder::new()
            .name("ochre-x11keys".into())
            .spawn(move || run(conn, driver, stop_r))?;
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
            // SAFETY: one byte to our own pipe, then close it.
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

impl Drop for X11Hotkeys {
    fn drop(&mut self) {
        self.stop();
    }
}

fn handle(driver: &Driver, event: Event) {
    let (keycode, down) = match event {
        Event::XinputRawKeyPress(e) => (e.detail, true),
        Event::XinputRawKeyRelease(e) => (e.detail, false),
        _ => return,
    };
    // X keycodes are evdev codes + 8.
    let name = keys::evdev_name(keycode.saturating_sub(8) as u16);
    driver.feed(name, down, monotonic_ms());
}

fn run(conn: RustConnection, driver: Arc<Driver>, stop_r: RawFd) {
    let fd = conn.stream().as_raw_fd();
    'outer: loop {
        // Drain what x11rb already buffered before sleeping on the socket.
        loop {
            match conn.poll_for_event() {
                Ok(Some(event)) => handle(&driver, event),
                Ok(None) => break,
                Err(e) => {
                    tracing::error!("X11 connection lost: {e}");
                    break 'outer;
                }
            }
        }
        let mut fds = [
            libc::pollfd {
                fd,
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: stop_r,
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        // SAFETY: valid pollfd array.
        let n = unsafe { libc::poll(fds.as_mut_ptr(), 2, -1) };
        if n < 0 && std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted {
            break;
        }
        if fds[1].revents != 0 {
            break;
        }
    }
    // SAFETY: our own fd.
    unsafe { libc::close(stop_r) };
    let _ = xinput::ConnectionExt::xinput_xi_select_events(&conn, conn.setup().roots[0].root, &[]);
}
