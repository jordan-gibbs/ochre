//! The HUD window: transparent, undecorated, always on top, skip taskbar, never focusable,
//! click-through except its own controls, capture-excluded where the OS allows, center-bottom above
//! the taskbar / dock on the monitor you're working on.
//!
//! It is created hidden at startup and only ever shown / hidden, so a `state` event puts it on screen
//! within a frame: Rust shows it the moment an active state is emitted (before the webview has even
//! seen the event), and the renderer hides it once its exit animation has finished (`hud_set_mode`).
//! A hide request is ignored while the core is still in an active state, which settles the race
//! between an old fade-out and a new session.
//!
//! Click-through: the window ignores the mouse except while the cursor is over one of the rects the
//! renderer reports (its × buttons). A 40 ms cursor poll runs only while the HUD is visible and has
//! buttons, and also catches a press over a button that Windows routed to the window below.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ochre_core::events::{Event, State};
use serde::{Deserialize, Serialize};
#[cfg(windows)]
use tauri::Manager;
use tauri::utils::config::BackgroundThrottlingPolicy;
use tauri::webview::WebviewWindowBuilder;
use tauri::{AppHandle, Emitter, PhysicalPosition, PhysicalSize, WebviewUrl, WebviewWindow};

/// Logical size of the HUD window. Content is bottom-anchored inside it; the rest is transparent
/// and click-through.
pub const HUD_W: f64 = 760.0;
pub const HUD_H: f64 = 320.0;
/// CSS px between the window's bottom edge and the pill: room for its soft (hero) shadow.
/// Matches `.hud { bottom }` in theme/hud.css.
pub const PAD: f64 = 30.0;
/// Logical px from the work area's bottom (taskbar / dock) to the pill's bottom edge.
pub const BOTTOM_GAP: f64 = 36.0;
const POLL: Duration = Duration::from_millis(40);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Hidden,
    Pip,
    Hud,
}

#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

pub fn hit(rects: &[Rect], p: Option<(f64, f64)>) -> bool {
    let Some((x, y)) = p else { return false };
    rects
        .iter()
        .any(|r| x >= r.x && y >= r.y && x < r.x + r.w && y < r.y + r.h)
}

/// Screen point (physical) -> CSS px inside a window at `origin` (physical) with `scale`.
pub fn to_css(
    cursor: (f64, f64),
    origin: (f64, f64),
    size: (f64, f64),
    scale: f64,
) -> Option<(f64, f64)> {
    let (dx, dy) = (cursor.0 - origin.0, cursor.1 - origin.1);
    if dx < 0.0 || dy < 0.0 || dx >= size.0 || dy >= size.1 {
        return None;
    }
    let s = if scale > 0.0 { scale } else { 1.0 };
    Some((dx / s, dy / s))
}

/// Window bounds (physical) for a work area (physical) at `scale`.
pub fn bounds(work: (i32, i32, u32, u32), scale: f64) -> (i32, i32, u32, u32) {
    let w = (HUD_W * scale).round() as u32;
    let h = (HUD_H * scale).round() as u32;
    let (wx, wy, ww, wh) = work;
    let x = wx + (ww as i32 - w as i32) / 2;
    let bottom = wy + wh as i32 - ((BOTTOM_GAP - PAD) * scale).round() as i32;
    (x, bottom - h as i32, w, h)
}

/// States in which the HUD must be on screen, whatever the renderer thinks.
pub fn is_active(state: State) -> bool {
    !matches!(state, State::Idle | State::Error)
}

struct Inner {
    mode: Mode,
    rects: Vec<Rect>,
    ignoring: bool,
    core_state: State,
}

pub struct Hud {
    app: AppHandle,
    win: WebviewWindow,
    inner: Mutex<Inner>,
    polling: AtomicBool,
    poller: Mutex<Option<std::thread::Thread>>,
}

impl Hud {
    pub fn create(app: &AppHandle, protect: bool) -> tauri::Result<Arc<Self>> {
        let win = WebviewWindowBuilder::new(app, "hud", WebviewUrl::App("hud/index.html".into()))
            .title("Ochre HUD")
            .inner_size(HUD_W, HUD_H)
            .resizable(false)
            .maximizable(false)
            .minimizable(false)
            .closable(false)
            .decorations(false)
            .transparent(true)
            .shadow(false)
            .always_on_top(true)
            .visible_on_all_workspaces(true)
            .skip_taskbar(true)
            .focused(false)
            .focusable(false)
            .content_protected(protect)
            .background_throttling(BackgroundThrottlingPolicy::Disabled)
            .visible(false)
            .build()?;
        // On Linux a window that was never shown has no GdkWindow yet and tao's click-through
        // call unwraps it (panic at startup), so there it is applied in `show_window`.
        #[cfg(not(target_os = "linux"))]
        win.set_ignore_cursor_events(true)?;

        #[cfg(target_os = "macos")]
        if let Ok(ns) = win.ns_window() {
            let ns = ns as usize;
            let _ =
                app.run_on_main_thread(move || crate::native::mac_harden(ns as *mut _, protect));
        }
        #[cfg(windows)]
        if let Ok(h) = win.hwnd() {
            let aff = crate::native::display_affinity(h.0 as _);
            eprintln!("[hud] content_protection={protect} display_affinity={aff:#x?}");
        }

        let hud = Arc::new(Self {
            app: app.clone(),
            win,
            inner: Mutex::new(Inner {
                mode: Mode::Hidden,
                rects: vec![],
                ignoring: true,
                core_state: State::Loading,
            }),
            polling: AtomicBool::new(false),
            poller: Mutex::new(None),
        });
        hud.spawn_poller();
        Ok(hud)
    }

    /// Called from the bus listener for every core event, before the webviews see it.
    pub fn on_event(&self, e: &Event) {
        match e {
            Event::State {
                state,
                handsfree_armed,
                ..
            } => {
                self.inner.lock().unwrap().core_state = *state;
                if is_active(*state) || *state == State::Error {
                    self.show(Mode::Hud);
                } else if *handsfree_armed {
                    self.show(Mode::Pip);
                }
            }
            Event::Error { .. } | Event::Notice { .. } => self.show(Mode::Hud),
            _ => {}
        }
    }

    /// The renderer's request, after its enter / exit animation.
    pub fn set_mode(&self, mode: Mode) {
        if mode == Mode::Hidden {
            self.hide();
        } else {
            self.show(mode);
        }
    }

    pub fn set_hit_rects(&self, rects: Vec<Rect>) {
        let wake = {
            let mut g = self.inner.lock().unwrap();
            g.rects = rects
                .into_iter()
                .filter(|r| r.w.is_finite() && r.h.is_finite() && r.w > 0.0)
                .collect();
            g.mode != Mode::Hidden && !g.rects.is_empty()
        };
        self.set_polling(wake);
    }

    fn show(&self, mode: Mode) {
        let was = {
            let mut g = self.inner.lock().unwrap();
            let was = g.mode;
            // pip never downgrades a full HUD; the renderer decides when to shrink
            if !(was == Mode::Hud && mode == Mode::Pip) {
                g.mode = mode;
            }
            was
        };
        if was != Mode::Hidden {
            return;
        }
        self.place();
        self.show_window();
        let has_rects = !self.inner.lock().unwrap().rects.is_empty();
        self.set_polling(has_rects);
    }

    fn hide(&self) {
        {
            let mut g = self.inner.lock().unwrap();
            if is_active(g.core_state) || g.mode == Mode::Hidden {
                return;
            }
            g.mode = Mode::Hidden;
        }
        self.set_polling(false);
        self.set_ignore(true);
        let _ = self.win.hide();
    }

    fn show_window(&self) {
        #[cfg(windows)]
        if let Ok(h) = self.win.hwnd() {
            crate::native::show_no_activate(h.0 as _);
            return;
        }
        #[cfg(target_os = "macos")]
        if let Ok(ns) = self.win.ns_window() {
            let ns = ns as usize;
            let _ = self
                .app
                .run_on_main_thread(move || crate::native::mac_show_no_activate(ns as *mut _));
            return;
        }
        #[allow(unreachable_code)]
        {
            let _ = self.win.show();
            #[cfg(target_os = "linux")]
            if self.inner.lock().unwrap().ignoring {
                let _ = self.win.set_ignore_cursor_events(true);
            }
        }
    }

    /// Center-bottom of the monitor with the focused window, else the cursor's, else the primary.
    fn place(&self) {
        let own: Vec<isize> = self.own_handles();
        let point = crate::native::foreground_center(&own)
            .or_else(crate::native::cursor)
            .or_else(|| self.app.cursor_position().ok().map(|p| (p.x, p.y)));
        let monitor = point
            .and_then(|(x, y)| self.app.monitor_from_point(x, y).ok().flatten())
            .or_else(|| self.app.primary_monitor().ok().flatten());
        let Some(m) = monitor else { return };
        let wa = m.work_area();
        let (x, y, w, h) = bounds(
            (wa.position.x, wa.position.y, wa.size.width, wa.size.height),
            m.scale_factor(),
        );
        let _ = self.win.set_size(PhysicalSize::new(w, h));
        let _ = self.win.set_position(PhysicalPosition::new(x, y));
    }

    fn own_handles(&self) -> Vec<isize> {
        #[cfg(windows)]
        {
            self.app
                .webview_windows()
                .values()
                .filter_map(|w| w.hwnd().ok())
                .map(|h| h.0 as isize)
                .collect()
        }
        #[cfg(not(windows))]
        {
            vec![]
        }
    }

    fn set_ignore(&self, ignore: bool) {
        let changed = {
            let mut g = self.inner.lock().unwrap();
            let c = g.ignoring != ignore;
            g.ignoring = ignore;
            c
        };
        if changed {
            let _ = self.win.set_ignore_cursor_events(ignore);
        }
    }

    fn set_polling(&self, on: bool) {
        let was = self.polling.swap(on, Ordering::SeqCst);
        if on
            && !was
            && let Some(t) = self.poller.lock().unwrap().as_ref()
        {
            t.unpark();
        }
        if !on && was {
            self.set_ignore(true);
        }
    }

    fn cursor_css(&self) -> Option<(f64, f64)> {
        #[cfg(windows)]
        {
            let h = self.win.hwnd().ok()?;
            let (x, y, w, hh) = crate::native::window_rect(h.0 as _)?;
            let scale = self.win.scale_factor().ok()?;
            to_css(
                crate::native::cursor()?,
                (x as f64, y as f64),
                (w as f64, hh as f64),
                scale,
            )
        }
        #[cfg(not(windows))]
        {
            let c = self.app.cursor_position().ok()?;
            let o = self.win.outer_position().ok()?;
            let s = self.win.outer_size().ok()?;
            let scale = self.win.scale_factor().ok()?;
            to_css(
                (c.x, c.y),
                (o.x as f64, o.y as f64),
                (s.width as f64, s.height as f64),
                scale,
            )
        }
    }

    fn spawn_poller(self: &Arc<Self>) {
        let weak = Arc::downgrade(self);
        let handle = std::thread::Builder::new()
            .name("ochre-hud-hit".into())
            .spawn(move || {
                let mut was_down = false;
                loop {
                    let Some(hud) = weak.upgrade() else { return };
                    if !hud.polling.load(Ordering::SeqCst) {
                        was_down = false;
                        drop(hud);
                        std::thread::park();
                        continue;
                    }
                    let p = hud.cursor_css();
                    let over = {
                        let g = hud.inner.lock().unwrap();
                        hit(&g.rects, p)
                    };
                    hud.set_ignore(!over);
                    if let Some(down) = crate::native::left_button_down() {
                        if down
                            && !was_down
                            && over
                            && let Some((x, y)) = p
                        {
                            let _ = hud.app.emit_to(
                                "hud",
                                "ochre://hud-press",
                                serde_json::json!({ "x": x, "y": y }),
                            );
                        }
                        was_down = down;
                    }
                    drop(hud);
                    std::thread::sleep(POLL);
                }
            })
            .expect("spawn hud poller");
        *self.poller.lock().unwrap() = Some(handle.thread().clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hit_testing() {
        let rects = [Rect {
            x: 10.0,
            y: 10.0,
            w: 20.0,
            h: 20.0,
        }];
        assert!(hit(&rects, Some((15.0, 29.0))));
        assert!(!hit(&rects, Some((30.0, 15.0))));
        assert!(!hit(&rects, None));
        assert_eq!(
            to_css((250.0, 150.0), (100.0, 100.0), (400.0, 200.0), 2.0),
            Some((75.0, 25.0))
        );
        assert_eq!(
            to_css((50.0, 150.0), (100.0, 100.0), (400.0, 200.0), 2.0),
            None
        );
    }

    #[test]
    fn placement_is_centered_above_the_work_area_bottom() {
        // 1920x1040 work area (taskbar 40 px) at 150 %
        let (x, y, w, h) = bounds((0, 0, 1920, 1040), 1.5);
        assert_eq!(w, (HUD_W * 1.5) as u32);
        assert_eq!(x, (1920 - w as i32) / 2);
        assert_eq!(
            y + h as i32,
            1040 - ((BOTTOM_GAP - PAD) * 1.5).round() as i32
        );
        // second monitor to the left, negative coordinates
        let (x, _, w, _) = bounds((-2560, 0, 2560, 1400), 1.0);
        assert_eq!(x, -2560 + (2560 - w as i32) / 2);
    }

    #[test]
    fn active_states_pin_the_hud() {
        assert!(
            is_active(State::Recording) && is_active(State::Loading) && is_active(State::Refining)
        );
        assert!(!is_active(State::Idle) && !is_active(State::Error));
    }
}
