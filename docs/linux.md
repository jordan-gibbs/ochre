# Linux: status and setup

Status 2026-10-06. Tested on Ubuntu 24.04.4 with GNOME 46 (X11 and Wayland sessions) in a
VirtualBox VM with no GPU and a virtual microphone (a PipeWire null sink playing recorded
speech). The VM froze under load, so **no latency, accuracy or idle-CPU number from it is
meaningful**. Those rows need real hardware. KDE, Sway and Hyprland are untested.

## Recommended setup

1. Install the `.deb` (or the `.rpm`, or run the AppImage).
   - The `.deb` recommends `xdotool`, `xclip`, `wl-clipboard`, `ydotool` and `ydotoold`.
   - It installs a udev rule (`60-ochre-uinput.rules`, `TAG+="uaccess"`) that lets the user at
     the local seat use `/dev/uinput`, so ydotool can type on Wayland without root. Steam ships
     the same kind of rule for its controllers.
   - Ochre starts `ydotoold` itself.
2. **X11:** nothing else to do. The Voice key (Right Alt by default) uses XInput 2.2.
3. **Wayland (GNOME, KDE):** apps can't watch global keys, so pick one of these:
   - join the `input` group so Ochre can read `/dev/input`:
     `sudo usermod -aG input $USER`, then log out and back in; or
   - bind a keyboard shortcut in your desktop settings to `ochre toggle`. It needs no extra
     permissions and measured 109–163 ms in the VM.

   GNOME 46 has no GlobalShortcuts portal yet; GNOME 48+ and KDE do, and supporting it is
   future work.
4. **GNOME:** the tray icon needs the AppIndicator extension, which Ubuntu ships enabled. Without
   it there is no tray icon. Open Settings with `ochre settings`, or from the app menu entry's
   actions.

## How it works

| Piece | X11 | Wayland |
|---|---|---|
| Voice key | XInput 2.2 raw events | `/dev/input` (evdev, `input` group) or an `ochre toggle` shortcut |
| Typing | `xdotool` | `ydotool` (0.1.x and 1.x), `dotool`, `wtype` (wlroots) or `kwtype` (KDE) |
| Non-ASCII text or non-US layout | pasted (except into terminals) | pasted (`wl-copy`, then Ctrl+V) |
| Windows (HUD, settings) | native | through XWayland, so the HUD never takes focus; `OCHRE_WAYLAND_NATIVE=1` opts out |
| Local cleanup server | `llama-server`, tied to Ochre's lifetime (`PR_SET_PDEATHSIG` from a long-lived thread) | same |
| Thread priority | `setpriority`, falling back to RealtimeKit | same |
| API keys | Secret Service (GNOME Keyring, KWallet, KeePassXC); otherwise `OCHRE_<PROVIDER>_API_KEY` | same |

## Test results (VM, 2026-10-06)

| Check | Result |
|---|---|
| `.deb` installs on a clean system, launches from the app menu, tray icon | ✅ X11 and Wayland |
| Onboarding: speech model download with progress, accurate permission rows | ✅ |
| Hold the Voice key → text lands (Text Editor, GTK4 dialog) | ✅ |
| Esc cancels; Voice key + Down pastes the last transcript | ✅ Wayland |
| Paste path restores the clipboard | ✅ 4/4 |
| German layout ("Grüße yz Straße") | ✅ after the fix (xdotool dropped ö/é) |
| HUD placement, no focus stealing | ✅ after the XWayland change |
| Start at login; no leftover `llama-server` after quit or `kill -9` | ✅ after the fix |
| AppImage launches | ✅ |
| Double-tap lock, then tap to stop | ❓ inconclusive in the VM |
| Hands-free | ⚠️ wake word fires (score 0.88); stop/send/cancel not testable |
| Local cleanup | ⚠️ server loads; cleanup timed out on the VM's CPU; CUDA/Vulkan untested |
| Idle CPU with hands-free on (target ≤ 2%) | ❌ 4.7–7.6% in the VM; recheck on hardware |

## Known issues

- On Wayland, Linux can't swallow keys, so the Voice key also reaches the focused app.
- The onboarding welcome animation keeps a full core busy under software rendering (no GPU).
- The onboarding model step can show "Download" next to "Ready to dictate".
- With fractional scaling, XWayland windows look blurry.
- Hands-free "pause during calls" is not implemented on Linux.
