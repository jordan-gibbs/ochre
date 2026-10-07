#!/usr/bin/env sh
# Ochre installer for macOS and Linux: builds from this checkout and installs `ochre`.
set -eu
cd "$(dirname "$0")/.."

have() { command -v "$1" >/dev/null 2>&1; }

if ! have cargo; then
  echo "Installing Rust (rustup)..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  . "$HOME/.cargo/env"
fi

case "$(uname -s)" in
  Linux)
    echo "Installing system packages (needs sudo)..."
    # Build deps, then the typing (xdotool on X11, ydotool on Wayland) and clipboard tools.
    if have apt-get; then
      sudo apt-get update
      sudo apt-get install -y build-essential pkg-config libssl-dev libasound2-dev \
        libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev libxdo-dev libxi-dev \
        xdotool xclip wl-clipboard
      sudo apt-get install -y ydotool || true
      sudo apt-get install -y ydotoold 2>/dev/null || true # a separate package on Debian 12 / Ubuntu 24.04
    elif have dnf; then
      sudo dnf install -y gcc gcc-c++ pkgconf-pkg-config openssl-devel alsa-lib-devel \
        webkit2gtk4.1-devel libappindicator-gtk3-devel librsvg2-devel libxdo-devel libXi-devel \
        xdotool xclip wl-clipboard ydotool
    elif have pacman; then
      sudo pacman -S --needed base-devel openssl alsa-lib webkit2gtk-4.1 libappindicator-gtk3 librsvg \
        libxi xdotool xclip wl-clipboard ydotool
    else
      echo "Unknown distro: install the Tauri v2 prerequisites, ALSA, xdotool, wl-clipboard and ydotool manually." >&2
    fi
    # Wayland: ydotool types through /dev/uinput. This rule (the same one the .deb / .rpm ship)
    # gives the logged-in user access to it, without root or the input group.
    rule=/etc/udev/rules.d/60-ochre-uinput.rules
    if [ ! -e "$rule" ] && [ ! -e /usr/lib/udev/rules.d/60-ochre-uinput.rules ]; then
      echo "Allowing typing on Wayland: installing $rule (needs sudo)..."
      sudo install -m 0644 app/src-tauri/linux/60-ochre-uinput.rules "$rule" &&
        sudo udevadm control --reload-rules && sudo udevadm trigger --name-match=uinput || true
    fi
    if [ "${XDG_SESSION_TYPE:-}" = "wayland" ] && ! id -nG | grep -qw input; then
      echo
      echo "Wayland: the Voice key is read from /dev/input, which needs the 'input' group:"
      echo "    sudo usermod -aG input \$USER    (then log out and back in)"
      echo "  Any member of 'input' can read every keystroke. Instead, you can bind a keyboard"
      echo "  shortcut to 'ochre toggle' in your desktop settings (press once to start, again to stop)."
    fi
    ;;
  Darwin)
    if ! xcode-select -p >/dev/null 2>&1; then
      xcode-select --install >/dev/null 2>&1 || true
      echo "Apple's Command Line Tools are needed: finish the install dialog that just opened," >&2
      echo "then run this script again." >&2
      exit 1
    fi
    ;;
esac

echo "Building Ochre (release). The first build takes a few minutes..."
if [ "$(uname -s)" = Darwin ]; then
  # macOS grants Microphone / Accessibility / Input Monitoring to an app bundle; a bare binary
  # would hand them to your terminal. So build Ochre.app and install it in /Applications.
  if cargo tauri --version >/dev/null 2>&1; then
    tauri() { cargo tauri "$@"; }
  elif have npx; then
    tauri() { npx --yes @tauri-apps/cli@2 "$@"; }
  else
    echo "Installing the Tauri CLI (one time)..."
    cargo install --locked tauri-cli --version "^2"
    tauri() { cargo tauri "$@"; }
  fi
  (cd app/src-tauri && tauri build --bundles app "$@")
  app=target/release/bundle/macos/Ochre.app
  id=io.github.jordangibbs.ochre
  dest=/Applications
  [ -w "$dest" ] || dest="$HOME/Applications"
  mkdir -p "$dest"
  updating=false
  [ -d "$dest/Ochre.app" ] && updating=true
  # Stop a running copy with its own CLI (osascript would ask for Automation permission).
  if pgrep -f "Ochre.app/Contents/MacOS/ochre" >/dev/null 2>&1; then
    "$dest/Ochre.app/Contents/MacOS/ochre" quit >/dev/null 2>&1 || true
    i=0
    while pgrep -f "Ochre.app/Contents/MacOS/ochre" >/dev/null 2>&1 && [ $i -lt 20 ]; do
      sleep 0.5; i=$((i + 1))
    done
  fi
  rm -rf "$dest/Ochre.app"
  cp -R "$app" "$dest/"
  # `ochre toggle`, `ochre paste-last` etc. from a shell talk to the running app.
  mkdir -p "$HOME/.cargo/bin"
  ln -sf "$dest/Ochre.app/Contents/MacOS/ochre" "$HOME/.cargo/bin/ochre"
  echo
  # macOS ties permissions to the exact signature. A local (ad-hoc signed) build gets a new one
  # every time, so after an update the old switches in System Settings stay on but no longer
  # apply. Clear them and reopen the permissions step so they can be granted again.
  if $updating && codesign -dv "$dest/Ochre.app" 2>&1 | grep -q "Signature=adhoc"; then
    for service in Accessibility ListenEvent Microphone; do
      tccutil reset "$service" "$id" >/dev/null 2>&1 || true
    done
    echo "Updated $dest/Ochre.app. This build has a new signature, so macOS needs the"
    echo "Microphone, Accessibility and Input Monitoring permissions again: the Ochre window"
    echo "that opens now walks through them (click Allow… on each)."
    open "$dest/Ochre.app" --args --open=onboarding
  else
    echo "Installed $dest/Ochre.app. Starting it..."
    open "$dest/Ochre.app"
    $updating || echo "The first-run window walks you through the Microphone, Accessibility and Input Monitoring permissions."
  fi
  echo "Hold Right Option to dictate. Settings are in the menu bar."
  exit 0
fi
cargo install --locked --path app/src-tauri "$@"
# Ochre used to be called openwhisprflow: drop that old binary (settings and models carry over on first launch).
if have openwhisprflow; then cargo uninstall openwhisprflow-app >/dev/null 2>&1 || true; fi
echo
echo "Installed. Start it with:  ochre"
echo "Hold Right Alt (Right Option on Mac) to dictate. Settings are in the tray / menu bar."
