# Notes for coding agents

## "Install Ochre" / "build the app for me"

When someone clones this repo and asks you to install Ochre, run the installer for their OS from
the repo root. It installs Rust and the OS prerequisites if they're missing, builds a release,
installs the app and (on macOS) starts it.

| OS | Command | Result |
|---|---|---|
| macOS | `./scripts/install.sh` | `Ochre.app` in `/Applications` (or `~/Applications`), started; `ochre` CLI linked into `~/.cargo/bin` |
| Linux | `./scripts/install.sh` | `ochre` in `~/.cargo/bin` |
| Windows | `powershell -ExecutionPolicy Bypass -File scripts/install.ps1` | `ochre.exe` in `%USERPROFILE%\.cargo\bin` |

**The first build takes 3–10 minutes** (later builds about 1 minute). Run it with a long timeout
or in the background and wait for it to exit; don't kill it because it's quiet. Success is exit
code 0 and, on macOS, the line `Installed /Applications/Ochre.app`.

### What only the user can do

Tell them; don't try to work around these.

- **macOS, first time:** if Apple's Command Line Tools are missing, the script opens their
  installer and exits with a message. The user finishes that dialog, then you run the script again.
- **macOS permissions:** the Ochre window that opens asks for **Microphone**, **Accessibility**
  and **Input Monitoring**. For each, the user clicks *Allow…* and switches Ochre on in System
  Settings. The list updates by itself and no restart is needed. Always install the `.app` the script
  builds; never use `cargo run` or the bare binary in `target/` for daily use, because macOS
  would give the permissions to the terminal instead of Ochre.
- **Linux:** the script installs system packages with `sudo`. If you can't answer a password
  prompt, ask the user to run `./scripts/install.sh` in their own terminal, or to run the
  package-install line for their distro from the script and then re-run it with you.
- **Windows:** if the script says the Visual Studio C++ build tools are missing, the user installs
  "Desktop development with C++" from the link it prints, then you re-run it.
- **Linux on Wayland:** if the Voice key isn't seen, the user binds a desktop shortcut to
  `ochre toggle`.

### After installing, tell the user

- Hold **Right Option** (Mac) or **Right Alt** (Windows, Linux), speak, let go: the text is typed
  where the cursor is. The first launch downloads the speech model (~670 MB) with a progress bar.
- **Cleanup is off by default.** To have filler words and self-corrections cleaned up, they turn
  it on in Settings → Refinement ("on this computer" downloads a model sized for their hardware, up
  to 2.6 GB; cloud providers need an API key).
- **macOS, noisy room:** Ochre uses the Mac's voice processing for the microphone. On the last
  setup step (or Settings → General → Microphone mode) they can choose **Voice Isolation** so other
  people's voices are left out.

### Updating

`git pull`, then run the same installer again. On macOS a local build gets a new signature each
time, so the script clears Ochre's old permission entries and reopens the permissions step: the
user clicks *Allow…* again. (Builds signed with a Developer ID, via `APPLE_SIGNING_IDENTITY`,
keep their permissions.)

### When something doesn't work

- Logs go to stderr. On macOS, quit Ochre (`ochre quit`) and run
  `RUST_LOG=info /Applications/Ochre.app/Contents/MacOS/ochre`; every dictation logs per-stage
  timings. Windows and Linux: run `ochre` from a terminal.
- Voice key does nothing on macOS: a permission is missing. `ochre onboarding` opens the setup
  window, whose permissions step shows which (with *Allow…* buttons).
- Config and models: `~/Library/Application Support/ochre/` (macOS),
  `%APPDATA%\ochre\config\` and `%LOCALAPPDATA%\ochre\data\` (Windows), `~/.config/ochre/` and
  `~/.local/share/ochre/` (Linux). History is `history.sqlite3` there.
- Never print or commit API keys (they live in the OS keychain) or the user's transcripts.

Per-OS details: [docs/macos.md](docs/macos.md), [docs/linux.md](docs/linux.md).

## Working on the code

Read `README.md`, `CONTRIBUTING.md` and `SPEC.md` first. Before pushing, run what CI runs (CI
pins Rust 1.95.0; newer clippy versions flag code that 1.95 accepts):

```sh
cargo fmt --all -- --check
cargo clippy --workspace --locked -- -D warnings
cargo test --workspace --locked
node --test app/test/ui.test.mjs
```
