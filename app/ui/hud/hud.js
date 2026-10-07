// HUD renderer. Folds core events into the model (model.js), maps the view onto the unified
// `.hud[data-state]` pill of the design contract (docs/design.md §7, theme/hud.css), and tells the
// shell the window mode (hidden | pip | hud) and where its buttons are: the window is
// click-through everywhere else. Motion helpers come from theme/hud-kit.js (window.HudKit).

import * as M from './model.js';
import { connect, send, invoke, listen, applyTheme } from '../shared/bridge.js';
import { label as keyLabel } from '../shared/keys.js';

const Kit = window.HudKit;
const $ = (sel) => document.querySelector(sel);
const hud = $('.hud');
const el = {
  pill: $('.hud-pill'), label: $('.hud-label'), hint: $('.hud-hint'), meta: $('.hud-meta'),
  close: $('.hud-close'), action: $('.hud-action'), pip: $('.hud-pip'),
  live: $('.hud-live'), liveText: $('.hud-live-text'),
};
const meter = Kit.oMeter($('.hud-o'));
const LEVEL_STATES = new Set(['recording', 'locked', 'handsfree']);

let ctx = { keyLabel: 'Right Alt', phrase: 'transcribe', showPartials: true };
let platformId = 'windows';
let themePref = 'light';
let S = M.initial();
let mode = 'hidden';
const now = () => performance.now();
const HIT_PAD = 8; // CSS px of forgiveness around every button

// ---------------------------------------------------------------- hit rects (click-through except buttons)

let rectsSig = '';
function measureHitRects() {
  const out = [];
  if (mode === 'hud') {
    for (const b of [el.close, el.action]) {
      if (b.hidden || b.offsetParent === null) continue;
      const r = b.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0) continue;
      out.push({ x: Math.round(r.left) - HIT_PAD, y: Math.round(r.top) - HIT_PAD, w: Math.round(r.width) + 2 * HIT_PAD, h: Math.round(r.height) + 2 * HIT_PAD });
    }
  }
  const sig = JSON.stringify(out);
  if (sig !== rectsSig) {
    rectsSig = sig;
    invoke('hud_set_hit_rects', { rects: out });
  }
}
let rectsTimer = null;
function scheduleHitRects() {
  measureHitRects();
  clearTimeout(rectsTimer);
  rectsTimer = setTimeout(measureHitRects, 420); // again once the width morph has settled
}

// ---------------------------------------------------------------- presence (window mode)

const durMs = (name, dflt) => {
  const v = getComputedStyle(hud).getPropertyValue(name).trim();
  const n = parseFloat(v);
  return Number.isFinite(n) ? (v.endsWith('ms') ? n : n * 1000) : dflt;
};
let hideTimer = null;
function applyMode(want) {
  if (want !== 'hidden' && hideTimer) { // came back before the exit finished
    clearTimeout(hideTimer);
    hideTimer = null;
    hud.setAttribute('data-shown', '');
  }
  if (want === mode) return;
  if (want === 'hidden') {
    if (hideTimer) return;
    // leave first (exit is --dur-2, ease-in), then let the shell hide the window
    hud.removeAttribute('data-shown');
    hideTimer = setTimeout(() => {
      hideTimer = null;
      mode = 'hidden';
      invoke('hud_set_mode', { mode: 'hidden' });
      scheduleHitRects();
    }, durMs('--dur-2', 160) + 40);
    return;
  }
  const was = mode;
  mode = want;
  invoke('hud_set_mode', { mode: want });
  if (was === 'hidden' || !hud.hasAttribute('data-shown')) {
    // the shell already showed the window on the state event; spring in on the next frame
    requestAnimationFrame(() => requestAnimationFrame(() => hud.setAttribute('data-shown', '')));
  }
  scheduleHitRects();
}

// ---------------------------------------------------------------- drawing

function writeText(h) {
  el.label.textContent = h.label;
  el.hint.textContent = '';
  if (h.kbd) {
    const k = document.createElement('kbd');
    k.textContent = h.kbd.kbd;
    el.hint.append(h.kbd.before, k, h.kbd.after);
  } else {
    el.hint.textContent = h.hint;
  }
  el.meta.textContent = h.meta;
  el.close.hidden = !h.close;
  el.close.setAttribute('aria-label', M.LIVE.has(S.state) ? 'Cancel' : 'Dismiss');
}

function render() {
  const v = M.view(S, ctx, now());
  const h = M.hudView(v);

  hud.toggleAttribute('data-armed', h.armed);
  hud.classList.toggle('is-indeterminate', h.indeterminate);
  hud.style.setProperty('--progress', h.progress != null ? h.progress.toFixed(4) : '0');
  if (h.pip) el.pip.textContent = h.pip;

  // Live text swaps the label for a fixed-width line: one width glide when it appears or leaves
  // (with the state change if there is one), none while words stream in.
  const liveChanged = !!h.live !== hud.hasAttribute('data-live');
  const apply = () => { writeText(h); setLive(h.live); };
  if (hud.dataset.state !== h.state) Kit.setState(hud, h.state, apply);
  else if (liveChanged) Kit.morph(hud, apply);
  else apply();

  if (!LEVEL_STATES.has(h.state)) meter.stop();
  applyMode(v.mode);
  scheduleHitRects();
  scheduleExpiry();
}

// ---------------------------------------------------------------- live text line
//
// The line is a run of spans over the text. An update keeps every span inside the common prefix
// untouched (so words already on screen never re-animate), trims the one straddling it, and
// appends the new tail as one fresh span that fades in. A span boundary is kept at the stable
// point so unconfirmed words can be tinted. Spans far off the left edge are dropped, which keeps
// the DOM small on long dictations.

const LIVE_KEEP = 320;      // chars kept in the DOM; the visible line holds ~45
let liveShown = '';         // the full text the spans represent (from `liveDrop` on)
let liveDrop = 0;           // chars dropped off the front
let liveSpans = [];         // {el, text, born}

function liveSpan(text, born) {
  const e = document.createElement('span');
  e.className = 'hud-w';
  e.textContent = text;
  const age = now() - born;
  if (age > 0) e.style.animationDelay = `-${age.toFixed(0)}ms`; // a split keeps its fade phase
  return { el: e, text, born };
}

function clearLive() {
  el.liveText.textContent = '';
  liveShown = '';
  liveDrop = 0;
  liveSpans = [];
}

function setLive(live) {
  hud.toggleAttribute('data-live', !!live);
  if (!live) { clearLive(); return; }
  const text = live.text;
  let p = M.commonPrefix(liveShown, text);
  if (p < liveDrop) { clearLive(); p = 0; } // rewritten off-screen: start over
  // keep / trim / drop existing spans
  let pos = liveDrop;
  const keep = [];
  for (const s of liveSpans) {
    const end = pos + s.text.length;
    if (end <= p) keep.push(s);
    else if (pos < p) { s.text = s.text.slice(0, p - pos); s.el.textContent = s.text; keep.push(s); }
    else s.el.remove();
    pos = end;
  }
  liveSpans = keep;
  if (text.length > p) {
    const s = liveSpan(text.slice(p), now());
    el.liveText.append(s.el);
    liveSpans.push(s);
  }
  liveShown = text;
  // a boundary at the stable point, then tint
  const k = Math.max(liveDrop, Math.min(live.stable, text.length));
  pos = liveDrop;
  for (let i = 0; i < liveSpans.length; i++) {
    const s = liveSpans[i];
    const end = pos + s.text.length;
    if (pos < k && k < end) {
      const rest = liveSpan(s.text.slice(k - pos), s.born);
      s.text = s.text.slice(0, k - pos);
      s.el.textContent = s.text;
      s.el.after(rest.el);
      liveSpans.splice(i + 1, 0, rest);
    }
    s.el.classList.toggle('is-pending', pos >= k);
    pos += s.text.length;
  }
  // prune what has long scrolled off the left
  let total = pos - liveDrop;
  while (total > LIVE_KEEP && liveSpans.length > 1) {
    const s = liveSpans.shift();
    s.el.remove();
    liveDrop += s.text.length;
    total -= s.text.length;
  }
  Kit.fitLive(el.live);
}

let expiryTimer = null;
function scheduleExpiry() {
  clearTimeout(expiryTimer);
  const t = M.nextDeadline(S, now());
  if (t != null) expiryTimer = setTimeout(render, Math.max(16, t - now() + 5));
}

// ---------------------------------------------------------------- ×

let lastPress = -Infinity;
function onClose(e) {
  if (e) { e.preventDefault(); e.stopPropagation(); }
  if (now() - lastPress < 500) return; // one action per physical click (pointerdown, poll press, click)
  lastPress = now();
  if (M.closeAction(S).cancel) {
    send({ op: 'cancel' });
    S = M.reduce(S, { event: '_cancel' }, now());
  }
  S = M.reduce(S, { event: '_dismiss' }, now());
  render();
}
el.close.addEventListener('pointerdown', onClose);
el.close.addEventListener('click', onClose);
// the shell's cursor poll saw a press over a button that the OS routed to the window below
listen('ochre://hud-press', (pt) => {
  const r = el.close.getBoundingClientRect();
  if (!el.close.hidden && pt.x >= r.left - HIT_PAD && pt.x < r.right + HIT_PAD && pt.y >= r.top - HIT_PAD && pt.y < r.bottom + HIT_PAD) onClose();
});

// ---------------------------------------------------------------- wiring

function onConfig(c) {
  ctx = {
    keyLabel: keyLabel((c.hotkey && c.hotkey.key) || 'right_alt', platformId),
    showPartials: !(c.ui && c.ui.show_partials === false),
  };
  themePref = (c.ui && c.ui.theme) || 'light';
  applyTheme(themePref);
}

function onEvent(msg, replay) {
  if (!msg || typeof msg.event !== 'string') return;
  switch (msg.event) {
    case 'level':
      if (LEVEL_STATES.has(hud.dataset.state)) meter.push(msg.rms);
      return;
    case 'hello': platformId = /mac|darwin/i.test(msg.platform) ? 'macos' : /win/i.test(msg.platform) ? 'windows' : 'linux'; return;
    case 'config': onConfig(msg.config || {}); break;
    case 'result': if (replay) return; break;
    default: break;
  }
  S = M.reduce(S, msg, now());
  render();
}

applyTheme(themePref);
render();
connect(onEvent).then(() => {
  // screenshots / previews: drive the HUD directly (after the snapshot replay)
  window.__ochreHud = { emit: (ev) => onEvent(ev, false), level: (rms) => onEvent({ event: 'level', rms }, false) };
});
