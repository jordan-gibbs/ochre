// HUD model: core events (ochre-core events.rs) -> what the HUD shows. Pure and unit-tested
// (app/test/hud-model.test.mjs); hud.js only draws `view()` and re-renders at `nextDeadline()`.
//
//   reduce(S, msg, now)   fold one core event (or a local "_cancel" / "_dismiss") into S
//   view(S, ctx, now)     -> { mode: 'hidden' | 'pip' | 'hud', pill, live, card, toast, pip }
//
// Live text (`partial` events, display only): one line inside the pill, newest words at the
// right, older ones fading out on the left. Shown while recording / locked / hands-free and
// through transcribing; gone from refining on. `ui.show_partials: false` turns it off.
//
// State -> pill (SPEC §7):
//   loading       white pill "Downloading speech model… 312 / 670 MB" + progress line, then "Loading models…"
//   idle          hidden; a tiny pip "Say “transcribe”" when hands-free is armed
//   recording     orange pill "Dictating… release to paste" + level bars + ×
//   locked        orange pill with a lock "Dictating… tap Right Alt to finish" + bars + ×
//   handsfree     orange pill "Listening — say “transcribe stop”" + breathing mic + bars + ×
//   transcribing  white pill "Transcribing" + shimmer and a light sweeping along the accent line
//   refining      white pill with a twinkling sparkle "Refining" in orange (distinct from transcribing)
//   inserting     white pill "Inserting"
//   result        toast "✓ Inserted · 13 words · 212 ms" (or "Not inserted · saved to History"), then fade
//   error         red-dot card with the message + ×, auto-dismissed
//   notice        orange-dot card with the message, auto-dismissed

import { bytesPair, prettyItem, words as countWords, ms as fmtMs } from '../shared/format.js';

export const TIMING = {
  toastMs: 2400,
  cancelToastMs: 1400,
  errorMs: 9000,
  noticeMs: 6000,
  cancelEchoMs: 2500, // a local × is followed by state idle from the core within this window
};

export const RECORDING = new Set(['recording', 'locked', 'handsfree']);
export const WORKING = new Set(['transcribing', 'refining', 'inserting']);
export const LIVE = new Set([...RECORDING, ...WORKING]);

export function initial() {
  return {
    state: 'idle', trigger: null, detail: '', armed: false,
    downloads: {},   // item -> {done, total, at}, for the current loading phase
    partial: null,   // {text, stable}
    toast: null,     // {kind: 'inserted'|'saved'|'cancelled'|'empty', words, refined, ms, until}
    card: null,      // {tone: 'error'|'notice', message, until}
    cancelAt: -Infinity,
  };
}

export function reduce(S, msg, now) {
  if (!msg || typeof msg !== 'object') return S;
  switch (msg.event) {
    case 'state': {
      const prev = S.state;
      const st = String(msg.state || 'idle');
      S.state = st;
      S.trigger = msg.trigger || null;
      S.detail = msg.detail || '';
      if (typeof msg.handsfree_armed === 'boolean') S.armed = msg.handsfree_armed;
      if (st === 'loading' && prev !== 'loading') S.downloads = {};
      // a new session starts clean: no stale words, toast or error from the last one
      if (RECORDING.has(st) && !LIVE.has(prev)) {
        S.partial = null;
        S.toast = null;
        if (S.card && S.card.tone === 'error') S.card = null;
      }
      // an error state without its own error event: show the detail as the card
      if (st === 'error' && S.detail && !(S.card && S.card.tone === 'error' && S.card.until > now)) {
        S.card = { tone: 'error', message: S.detail, until: now + TIMING.errorMs };
      }
      if (st === 'idle') {
        S.partial = null;
        if (now - S.cancelAt < TIMING.cancelEchoMs && LIVE.has(prev)) {
          S.toast = { kind: 'cancelled', until: now + TIMING.cancelToastMs };
        }
        S.cancelAt = -Infinity;
      }
      return S;
    }
    case 'partial':
      S.partial = { text: String(msg.text || ''), stable: Math.max(0, Number(msg.stable_chars) || 0) };
      return S;
    case 'result': {
      const n = countWords(msg.text);
      const t = msg.timings || {};
      const ms = Number.isFinite(t.release_to_insert_ms) ? t.release_to_insert_ms : null;
      S.partial = null;
      S.toast = n === 0
        ? { kind: 'empty', until: now + TIMING.cancelToastMs }
        : { kind: msg.inserted === false ? 'saved' : 'inserted', words: n, refined: !!msg.refined, ms,
          until: now + TIMING.toastMs + (msg.inserted === false ? 2500 : 0) };
      return S;
    }
    case 'error':
      S.card = { tone: 'error', message: String(msg.message || 'Something went wrong.'), until: now + TIMING.errorMs };
      return S;
    case 'notice':
      S.card = { tone: 'notice', message: String(msg.message || ''), until: now + TIMING.noticeMs };
      return S;
    case 'download': {
      const item = String(msg.item || 'model');
      S.downloads[item] = { done: Number(msg.done) || 0, total: Number(msg.total) || 0, at: now };
      return S;
    }
    case 'config': {
      const hf = msg.config && msg.config.handsfree;
      if (hf && hf.enabled === false && S.state === 'idle') S.armed = false;
      return S;
    }
    case '_cancel':
      S.cancelAt = now;
      return S;
    case '_dismiss':
      S.card = null;
      S.toast = null;
      return S;
    default:
      return S;
  }
}

export function downloadSummary(S) {
  const items = Object.entries(S.downloads);
  const active = items.filter(([, d]) => d.total > 0 && d.done < d.total);
  if (!active.length) return null;
  let done = 0;
  let total = 0;
  for (const [, d] of items) {
    if (d.total > 0) { done += Math.min(d.done, d.total); total += d.total; }
  }
  const latest = active.sort((a, b) => b[1].at - a[1].at)[0][0];
  return { item: latest, done, total, fraction: total ? done / total : 0 };
}

export function view(S, ctx = {}, now = 0) {
  const keyLabel = ctx.keyLabel || 'Right Alt';
  const phrase = 'transcribe'; // the bundled wake word; it is not user-configurable
  const showPartials = ctx.showPartials !== false;
  const out = { mode: 'hidden', pill: null, live: null, card: null, toast: null, pip: null };

  const card = S.card && S.card.until > now ? S.card : null;
  const toast = S.toast && S.toast.until > now ? S.toast : null;
  const st = S.state;

  if (st === 'loading') {
    const dl = downloadSummary(S);
    if (dl) {
      const label = /^download/i.test(S.detail) ? S.detail.replace(/[.…]*$/, '…') : `Downloading ${prettyItem(dl.item)}…`;
      out.pill = { kind: 'loading', label, meta: bytesPair(dl.done, dl.total), progress: dl.fraction };
    } else {
      out.pill = { kind: 'loading', label: S.detail && !/^download/i.test(S.detail) ? S.detail : 'Loading models…', progress: null };
    }
  } else if (st === 'recording') {
    out.pill = { kind: 'dictating', variant: 'hold', label: 'Dictating…', hint: 'release to paste', close: true };
  } else if (st === 'locked') {
    out.pill = { kind: 'dictating', variant: 'locked', label: 'Dictating…', hint: `tap ${keyLabel} to finish`, close: true };
  } else if (st === 'handsfree') {
    out.pill = { kind: 'dictating', variant: 'handsfree', label: 'Listening', hint: `say “${phrase} stop”`, close: true };
  } else if (st === 'transcribing') {
    out.pill = { kind: 'transcribing', label: 'Transcribing', close: true };
  } else if (st === 'refining') {
    out.pill = { kind: 'refining', label: 'Refining', close: true };
  } else if (st === 'inserting' && !toast) {
    out.pill = { kind: 'inserting', label: 'Inserting' };
  }

  if (LIVE.has(st) && showPartials && S.partial && S.partial.text.trim()) {
    const t = S.partial.text;
    let k = Math.min(S.partial.stable, t.length);
    // never split a word between confirmed and pending: snap back to the word's start
    while (k > 0 && k < t.length && !/\s/.test(t[k - 1]) && !/\s/.test(t[k])) k--;
    out.live = { text: t, stable: k };
  }
  if (card) out.card = { tone: card.tone, message: card.message };
  if (toast && !RECORDING.has(st)) out.toast = toastView(toast);

  if (out.pill || out.live || out.card || out.toast) out.mode = 'hud';
  // Armed hands-free while idle shows nothing on screen: the tray icon carries that state.
  return out;
}

export function toastView(t) {
  if (t.kind === 'cancelled') return { kind: 'muted', title: 'Cancelled' };
  if (t.kind === 'empty') return { kind: 'muted', title: 'Nothing heard' };
  if (t.kind === 'saved') return { kind: 'saved', title: 'Not inserted', meta: 'saved to History' };
  const parts = [`${t.words} word${t.words === 1 ? '' : 's'}`];
  if (t.refined) parts.push('refined');
  if (t.ms != null) parts.push(fmtMs(t.ms));
  return { kind: 'inserted', title: 'Inserted', meta: parts.join(' · ') };
}

/** Next time the view changes on its own (a toast or card expiring), or null. */
export function nextDeadline(S, now) {
  const ts = [S.toast && S.toast.until, S.card && S.card.until].filter((t) => t && t > now);
  return ts.length ? Math.min(...ts) : null;
}

/** What × does: cancel the session while one runs; always clear what's shown. */
export function closeAction(S) {
  return { cancel: LIVE.has(S.state), dismiss: true };
}

const LIVE_TEXT_STATES = new Set(['recording', 'locked', 'handsfree', 'transcribing']);

/** Length of the common prefix of two strings (the live line only animates what changed). */
export function commonPrefix(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a.charCodeAt(i) === b.charCodeAt(i)) i++;
  // never end inside a surrogate pair
  if (i > 0 && i < a.length && (a.charCodeAt(i - 1) & 0xfc00) === 0xd800) i--;
  return i;
}

/**
 * view() -> the unified pill of the design contract (docs/design.md §7): one `data-state` plus
 * the label / hint / meta to write. `hint` may carry a `kbd` part: {before, kbd, after}.
 * Priority: a live session > an error / notice card > the result toast > the armed pip.
 */
export function hudView(v) {
  const out = { state: 'idle', shown: v.mode !== 'hidden', armed: v.mode === 'pip', label: '', hint: '', kbd: null,
    meta: '', progress: null, indeterminate: false, close: false, live: null, pip: v.pip ? v.pip.label : '' };
  const p = v.pill;
  if (p) {
    out.label = p.label;
    out.hint = p.hint || '';
    out.meta = p.meta || '';
    out.close = !!p.close;
    if (p.kind === 'dictating') out.state = p.variant === 'hold' ? 'recording' : p.variant;
    else out.state = p.kind;
    if (p.kind === 'loading') {
      out.progress = p.progress;
      out.indeterminate = p.progress == null;
    }
    const m = /^tap (.+) to finish$/.exec(out.hint);
    if (m) { out.hint = ''; out.kbd = { before: 'tap ', kbd: m[1], after: ' to finish' }; }
  } else if (v.card) {
    out.state = v.card.tone === 'error' ? 'error' : 'notice';
    out.label = v.card.message;
    out.close = true;
  } else if (v.toast) {
    const t = v.toast;
    if (t.kind === 'inserted') {
      out.state = 'inserted';
      out.label = t.title;
      const parts = (t.meta || '').split(' · ');
      const timing = parts.length > 1 && /\d\s*(ms|s)$/.test(parts[parts.length - 1]) ? parts.pop() : '';
      out.hint = parts.join(' · ');
      out.meta = timing;
    } else {
      out.state = 'notice';
      out.label = t.kind === 'saved' ? 'Saved to History' : t.title;
      out.hint = t.kind === 'saved' ? 'not inserted' : '';
      out.close = t.kind === 'saved';
    }
  }
  if (v.live && LIVE_TEXT_STATES.has(out.state)) out.live = v.live;
  out.label = out.label.replace(/(\.\.\.|…)$/, ''); // the pill's glyph already says "in progress"
  return out;
}
