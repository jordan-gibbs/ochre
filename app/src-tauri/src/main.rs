//! Ochre: the Tauri v2 shell around the dictation core.
//!
//! - Every core `Event` on the bus is forwarded to the webviews as the Tauri event `ochre://event`;
//!   webviews send `Command`s through the `ochre_command` Tauri command.
//! - The HUD window is created hidden at startup and shown from Rust the moment an active state is
//!   emitted (see `hud.rs`). Settings and onboarding are created on demand (`pages.rs`).
//! - Tray icon and menu reflect the state (`tray.rs`). A second launch with
//!   `toggle|start|stop|cancel|paste-last|settings|history|quit` forwards to the running app.
//! - `--demo` swaps the core for a scripted driver (`demo.rs`).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod cli;
mod core_bridge;
mod demo;
mod hud;
mod native;
mod pages;
mod tray;

use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex, OnceLock};

use ochre_core::events::{Bus, Command, Event};
use tauri::{AppHandle, Emitter, Manager, RunEvent, WebviewWindow};
use tauri_plugin_autostart::{MacosLauncher, ManagerExt};

use crate::cli::{Args, Forward};
use crate::core_bridge::{Core, Snapshot};
use crate::hud::Hud;
use crate::tray::{Action, Tray};

pub const EVENT: &str = "ochre://event";

pub struct Shell {
    core: Box<dyn Core>,
    bus: Bus,
    snapshot: Mutex<Snapshot>,
    args: Args,
    hud: OnceLock<Arc<Hud>>,
}

impl Shell {
    fn theme(&self) -> Option<tauri::Theme> {
        pages::theme_of(self.snapshot.lock().unwrap().config())
    }
}

fn dispatch(app: &AppHandle, shell: &Shell, cmd: Command) {
    match cmd {
        Command::Quit => quit(app, shell),
        Command::OpenSettings => open(app, shell, "settings"),
        cmd => shell.core.command(cmd),
    }
}

fn open(app: &AppHandle, shell: &Shell, target: &str) {
    if let Err(e) = pages::open(app, target, shell.theme()) {
        eprintln!("[shell] couldn't open {target}: {e}");
    }
}

/// The tray menu took the foreground (Windows: the taskbar / Ochre's menu window), so bring the
/// user's window back first, off the main thread, then type into it.
fn paste_last_from_tray(app: &AppHandle, shell: &Arc<Shell>) {
    let (app, shell) = (app.clone(), shell.clone());
    let spawned = std::thread::Builder::new()
        .name("ochre-paste-last".into())
        .spawn(move || {
            let restored =
                ochre_platform::focus::restore_previous(std::time::Duration::from_millis(150));
            if !restored && cfg!(windows) {
                eprintln!("[shell] paste last: couldn't bring the previous window back");
            }
            dispatch(&app, &shell, Command::PasteLast);
        });
    if let Err(e) = spawned {
        eprintln!("[shell] paste last: {e}");
    }
}

fn quit(app: &AppHandle, shell: &Shell) {
    shell.core.shutdown();
    app.exit(0);
}

// ------------------------------------------------------------------------------------ commands

#[tauri::command]
async fn ochre_command(
    app: AppHandle,
    shell: tauri::State<'_, Arc<Shell>>,
    cmd: Command,
) -> Result<(), String> {
    dispatch(&app, &shell, cmd);
    Ok(())
}

/// Everything a window that opened late needs to catch up, oldest first (state last).
#[tauri::command]
async fn ochre_snapshot(shell: tauri::State<'_, Arc<Shell>>) -> Result<Vec<Event>, String> {
    Ok(shell.snapshot.lock().unwrap().events())
}

#[tauri::command]
async fn hud_set_mode(shell: tauri::State<'_, Arc<Shell>>, mode: hud::Mode) -> Result<(), String> {
    if let Some(h) = shell.hud.get() {
        h.set_mode(mode);
    }
    Ok(())
}

#[tauri::command]
async fn hud_set_hit_rects(
    shell: tauri::State<'_, Arc<Shell>>,
    rects: Vec<hud::Rect>,
) -> Result<(), String> {
    if let Some(h) = shell.hud.get() {
        h.set_hit_rects(rects);
    }
    Ok(())
}

#[tauri::command]
async fn ui_ready(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    pages::ready(&app, window.label());
    Ok(())
}

#[tauri::command]
async fn open_page(
    app: AppHandle,
    shell: tauri::State<'_, Arc<Shell>>,
    target: String,
) -> Result<(), String> {
    open(&app, &shell, &target);
    Ok(())
}

#[tauri::command]
async fn close_page(window: WebviewWindow) -> Result<(), String> {
    window.close().map_err(|e| e.to_string())
}

// ------------------------------------------------------------------------------------ wiring

/// Off the bus thread: tray, theme, start-at-login, first-run onboarding. None of it is
/// latency-critical, and tray / window calls may wait on the main thread.
fn shell_thread(app: AppHandle, shell: Arc<Shell>, tray: Arc<Tray>, rx: Receiver<Event>) {
    std::thread::Builder::new()
        .name("ochre-shell".into())
        .spawn(move || {
            let mut onboarding_checked = false;
            let mut last_theme: Option<String> = None;
            for e in rx {
                tray.on_event(&e);
                let Event::Config { config, .. } = &e else {
                    continue;
                };
                let theme = config
                    .pointer("/ui/theme")
                    .and_then(|t| t.as_str())
                    .map(str::to_string);
                if theme != last_theme {
                    pages::set_theme(&app, pages::theme_of(Some(config)));
                    last_theme = theme;
                }
                if !shell.args.demo {
                    sync_autostart(&app, config);
                }
                if !onboarding_checked {
                    onboarding_checked = true;
                    let onboarded = config
                        .pointer("/ui/onboarded")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true);
                    if !onboarded && !shell.args.autostart && shell.args.open.is_none() {
                        open(&app, &shell, "onboarding");
                    }
                }
            }
        })
        .expect("spawn shell thread");
}

/// `ui.start_at_login` is the source of truth; the OS login item follows it.
fn sync_autostart(app: &AppHandle, config: &serde_json::Value) {
    let want = config
        .pointer("/ui/start_at_login")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let auto = app.autolaunch();
    if auto.is_enabled().unwrap_or(false) != want {
        let r = if want { auto.enable() } else { auto.disable() };
        if let Err(e) = r {
            eprintln!("[shell] start at login: {e}");
        }
    }
}

/// Linux Wayland: run our windows through XWayland unless told otherwise. Wayland lets no client
/// place its own windows or refuse focus, so on GNOME the native HUD opens wherever mutter puts
/// it and takes keyboard focus from the app being dictated into (the text then goes nowhere).
/// As an X11 window it sits centred at the bottom and never takes focus. Only GTK's backend
/// changes: the Voice key (evdev), typing and the clipboard still use the Wayland session.
/// `OCHRE_WAYLAND_NATIVE=1` or an explicit `GDK_BACKEND` keeps native Wayland windows.
#[cfg(target_os = "linux")]
fn prefer_xwayland() {
    let wayland = std::env::var("XDG_SESSION_TYPE").is_ok_and(|t| t == "wayland")
        || std::env::var_os("WAYLAND_DISPLAY").is_some();
    if wayland
        && std::env::var_os("DISPLAY").is_some()
        && std::env::var_os("GDK_BACKEND").is_none()
        && std::env::var_os("OCHRE_WAYLAND_NATIVE").is_none()
    {
        // SAFETY: first thing in main, before any other thread exists.
        unsafe { std::env::set_var("GDK_BACKEND", "x11") };
    }
}

fn main() {
    #[cfg(target_os = "linux")]
    prefer_xwayland();
    // Logs go to stderr (visible when launched from a terminal); default: info for our crates.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,ort=warn,tao=warn,wry=warn".into()),
        )
        .with_writer(std::io::stderr)
        .try_init();
    // One-time move of the pre-rename `openwhisprflow` dirs (config, history, models) to `ochre`,
    // before anything below opens them. Logged by `migrate_legacy_dirs`.
    ochre_core::paths::migrate_legacy_dirs();
    let args = cli::parse(std::env::args().skip(1));
    if let Some(h) = &args.demo_hold
        && !demo::LABELS.contains(&h.as_str())
    {
        eprintln!(
            "--demo-hold: unknown label {h:?}; one of {}",
            demo::LABELS.join(", ")
        );
    }
    let bus = Bus::new();
    let shell = Arc::new(Shell {
        core: core_bridge::create(&args, bus.clone()),
        bus,
        snapshot: Mutex::new(Snapshot::default()),
        args,
        hud: OnceLock::new(),
    });

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, argv, _cwd| {
            let shell = app.state::<Arc<Shell>>();
            match cli::forwarded(&argv) {
                Some(Forward::Command(cmd)) => dispatch(app, &shell, cmd),
                Some(Forward::Open(page)) => open(app, &shell, &page),
                None => open(app, &shell, "settings"),
            }
        }))
        .plugin(tauri_plugin_autostart::init(
            MacosLauncher::LaunchAgent,
            Some(vec!["--autostart"]),
        ))
        .manage(shell.clone())
        .invoke_handler(tauri::generate_handler![
            ochre_command,
            ochre_snapshot,
            hud_set_mode,
            hud_set_hit_rects,
            ui_ready,
            open_page,
            close_page
        ])
        .setup(move |app| {
            #[cfg(target_os = "macos")]
            {
                app.set_activation_policy(tauri::ActivationPolicy::Accessory);
                // Cmd+V for the paste fallback follows the keyboard layout (main thread only).
                ochre_platform::macos::layout::track();
            }
            // Where "Paste last transcript" in the tray types (the window before the tray click).
            ochre_platform::focus::track_foreground();

            let handle = app.handle().clone();
            let hud = Hud::create(&handle, !shell.args.no_protect)?;
            let _ = shell.hud.set(hud.clone());

            let tray_shell = shell.clone();
            let tray = Arc::new(Tray::create(&handle, move |app, action| match action {
                Action::Core(cmd) => dispatch(app, &tray_shell, cmd),
                Action::PasteLast => paste_last_from_tray(app, &tray_shell),
                Action::Open(page) => open(app, &tray_shell, page),
                Action::Quit => quit(app, &tray_shell),
            })?);
            app.manage(tray.clone());
            // the taskbar theme can flip at any time (or at sunset); a slow poll is plenty
            #[cfg(not(target_os = "macos"))]
            {
                let poll = tray.clone();
                let _ = std::thread::Builder::new()
                    .name("ochre-tray-theme".into())
                    .spawn(move || {
                        loop {
                            std::thread::sleep(std::time::Duration::from_secs(3));
                            poll.check_taskbar();
                        }
                    });
            }

            let (tx, rx) = mpsc::channel::<Event>();
            shell_thread(handle.clone(), shell.clone(), tray, rx);

            let listener_shell = shell.clone();
            let tx = Mutex::new(tx);
            shell.bus.subscribe(move |e| {
                hud.on_event(e); // first: the HUD must be on screen within a frame
                if !matches!(e, Event::Level { .. }) {
                    listener_shell.snapshot.lock().unwrap().record(e);
                }
                let _ = handle.emit(EVENT, e);
                if matches!(e, Event::State { .. } | Event::Config { .. }) {
                    let _ = tx.lock().unwrap().send(e.clone());
                }
            });

            // LEAD: hotkeys go here, e.g.
            //   let core = shell.clone();
            //   ochre_platform::hotkeys(&cfg.hotkey, move |g| core.core.on_gesture(g));
            // (keep the handle alive in `Shell`; `on_gesture` returns immediately)
            shell.core.start();
            if let Some(page) = shell.args.open.clone() {
                open(app.handle(), &shell, &page);
            }
            if let Some(Forward::Open(page)) = shell.args.forward.clone() {
                open(app.handle(), &shell, &page);
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build Ochre");

    app.run(|app, event| {
        match event {
            // closing the last window keeps the tray app running; only Quit exits
            RunEvent::ExitRequested {
                api, code: None, ..
            } => api.prevent_exit(),
            RunEvent::Exit => app.state::<Arc<Shell>>().core.shutdown(),
            _ => {}
        }
    });
}
