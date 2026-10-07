// The only place the UI talks to the shell. In the app: Tauri's global API (`withGlobalTauri`),
// core events arrive as `ochre://event`, commands go out through `ochre_command`. In a plain browser
// (screenshots, `?mock` previews) the same calls are served by ../dev/mock.js.

import { resolveTheme } from './format.js';

const T = window.__TAURI__;
let mock = null;

export const inApp = !!T;

async function getMock() {
  if (!mock) {
    const m = await import('../dev/mock.js');
    mock = m.sharedMock();
  }
  return mock;
}

/** Subscribe to core events, then replay the snapshot (config, engines, state, ...). */
export async function connect(onEvent) {
  if (T) {
    await T.event.listen('ochre://event', (e) => onEvent(e.payload, false));
    const snap = await T.core.invoke('ochre_snapshot');
    for (const ev of snap) onEvent(ev, true);
  } else {
    const m = await getMock();
    m.subscribe(onEvent);
  }
}

/** A `Command` (`{op: ..., ...}`, see ochre-core events.rs). */
export function send(cmd) {
  if (T) return T.core.invoke('ochre_command', { cmd }).catch((e) => console.error('ochre_command', e));
  return getMock().then((m) => m.command(cmd));
}

export function invoke(name, args = {}) {
  if (T) return T.core.invoke(name, args).catch((e) => console.error(name, e));
  return getMock().then((m) => m.invoke(name, args));
}

export function listen(name, fn) {
  if (T) return T.event.listen(name, (e) => fn(e.payload));
  return getMock().then((m) => m.on(name, fn));
}

/** Theme from config `ui.theme`: "light" (the default) | "dark" | "system" (follow the OS, and
 *  keep following it while the page is open). `?theme=` overrides it (previews, screenshots).
 *  Every window (settings, onboarding, HUD) goes through here, so they always agree. */
let themeState = { pref: 'light', root: null, watching: false };
export function applyTheme(pref, root = document.documentElement) {
  const q = new URLSearchParams(location.search).get('theme');
  themeState.pref = q || pref || 'light';
  themeState.root = root;
  const mq = matchMedia('(prefers-color-scheme: dark)');
  if (!themeState.watching) {
    themeState.watching = true;
    mq.addEventListener('change', () => {
      if (themeState.pref === 'system') applyTheme(themeState.pref, themeState.root);
    });
  }
  const t = resolveTheme(themeState.pref, mq.matches);
  root.dataset.theme = t;
  return t;
}

export function platform(hello) {
  const p = (hello && hello.platform) || navigator.userAgent;
  if (/mac|darwin/i.test(p)) return 'macos';
  if (/win/i.test(p)) return 'windows';
  return 'linux';
}
