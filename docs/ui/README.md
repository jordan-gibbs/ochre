# Ochre UI (Tauri v2)

The shell is `app/src-tauri` (binary `ochre`) and the web UI is `app/ui`, written in plain
HTML/CSS/ES modules with no bundler. The visual design ("theme") is `app/ui/theme/**` and is documented in
`docs/design.md`. Every page reads its colours, fonts and motion only through those tokens.

```powershell
cargo run -p ochre-app -- --demo                 # every state, scripted, looping
cargo run -p ochre-app -- --demo-hold=locked     # stop at one state (labels: download loading recording
                                                          #   transcribing refining inserted locked pip handsfree notice error)
cargo run -p ochre-app -- --demo --open=settings # or history | onboarding | transcription | ...
ochre toggle|start|stop|cancel|paste-last|settings|history|quit   # forwarded to the running app
node app/scripts/render-shots.mjs [--only=hud-recording,...]    # regenerate the PNGs in this folder
node --test app/test/ui.test.mjs                                # HUD model, key capture, formatting
```

Other flags: `--no-protect` turns off capture exclusion, and `--autostart` (passed by the login item) opens no windows.

## How it fits together

- **Core seam** (`core_bridge.rs`): the `Core` trait wraps `ochre::App` (`OchreCore`) or the scripted
  `DemoCore` (`--demo`). Every bus `Event` goes to the webviews as `ochre://event`. Pages send
  `Command`s through the `ochre_command` Tauri command and catch up with `ochre_snapshot`.
- **HUD** (`hud.rs`, `ui/hud`): created hidden at startup. Rust shows it the moment an active
  `state` event is emitted, before the webview has even seen the event. The renderer hides it after
  its exit animation. A hide request is ignored while the core is still active.
- **Click-through**: the window ignores the mouse except over the rects the renderer reports (its ×).
  A 40 ms cursor poll runs only while those buttons are visible.
- **Tray** (`tray.rs`): the icon changes with the state (idle / recording / busy / error), using template
  images on macOS. The menu has a status line, Start/Stop, Hands-free, Refinement (Off / Local / Cloud),
  Settings…, History… and Quit. On Windows and Linux a left click opens Settings.
- **Start at login** follows `ui.start_at_login` (tauri-plugin-autostart). Demo mode never touches it.

## Window behaviour per OS

| | Windows | macOS | Linux X11 | Linux Wayland |
|---|---|---|---|---|
| Never takes focus | `WS_EX_NOACTIVATE` + `SW_SHOWNOACTIVATE` (verified) | `orderFrontRegardless`; app is `Accessory` (no Dock icon) | `accept_focus=false` | compositor-dependent |
| Always on top | `HWND_TOPMOST` (verified) | `NSStatusWindowLevel`, all Spaces, over full-screen apps | keep-above hint | **not guaranteed**: needs a layer-shell surface |
| Click-through | `WS_EX_TRANSPARENT` toggled by the hit poll (verified) | `ignoresMouseEvents` | input shape | input region (GTK); works on most compositors |
| Placement | monitor of the foreground window, else the cursor's; DPI-aware | cursor's monitor | cursor's monitor | **compositor decides**; often centred |
| Hidden from capture | `WDA_EXCLUDEFROMCAPTURE` (0x11, verified) | `NSWindowSharingNone` (newer ScreenCaptureKit may still see it) | no | no |
| Transparency | yes | yes (`macos-private-api`) | needs a compositor | yes |

On Wayland the hotkey hook may also be unavailable. Bind a compositor shortcut to
`ochre toggle` instead (single-instance forwarding handles it). A proper fix is a
`gtk-layer-shell` HUD surface. That is not done yet.

## Screenshots

`hud-*.png` show the real `hud/index.html` placed exactly as `hud.rs` places the window, with a light app on
the left and a dark app on the right. `settings-*` and `onboarding-*` come in `-light` and `-dark`. All of
them are rendered by headless Edge (the WebView2 engine) from `app/ui` with the browser mock
`ui/dev/mock.js`, which mirrors the Rust demo fixtures. Nothing is shown on screen and the desktop is
never captured.
