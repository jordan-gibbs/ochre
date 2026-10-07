// A tiny component kit for settings and onboarding: plain DOM, no framework, emitting the markup
// contract of theme/components.css (docs/design.md §8). Controls bound to a config path register
// an updater, so a fresh `config` event refreshes values in place without rebuilding the page
// (and without stealing focus from a field being typed in).

import { store, setPath, cfg, saveSecret, test } from './store.js';
import { icon } from './icons.js';
import { bytesPair, ms as fmtMs } from './format.js';
import { label as keyLabel, createCapture, presets } from './keys.js';

let updaters = [];
export function resetBindings() { updaters = []; }
export function refreshBindings() { for (const u of updaters) u(); }
function bind(fn) { updaters.push(fn); fn(); }

/** h('div.cls#id', {attrs}, ...children). Strings are text; `html` attr sets innerHTML. */
export function h(sel, attrs, ...children) {
  if (attrs == null || typeof attrs !== 'object' || attrs instanceof Node || Array.isArray(attrs)) {
    if (attrs != null) children.unshift(attrs);
    attrs = {};
  }
  const [, tag = 'div', rest = ''] = /^([a-z0-9-]*)(.*)$/i.exec(sel);
  const el = document.createElement(tag || 'div');
  for (const m of rest.matchAll(/([.#])([^.#]+)/g)) {
    if (m[1] === '.') el.classList.add(m[2]); else el.id = m[2];
  }
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'class') el.className += ` ${v}`;
    else if (k === 'style' && typeof v === 'object') {
      for (const [sk, sv] of Object.entries(v)) el.style.setProperty(sk.startsWith('--') ? sk : sk.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`), sv);
    } else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

/** One element from an HTML / SVG string (for inline icons that must be direct children). */
export function fromHtml(html) {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
}

const svg = (name) => fromHtml(icon(name));

// ---------------------------------------------------------------- layout

export function section(title, lede, ...children) {
  return h('section.page', h('header.page-head', h('h1.page-title', title), lede ? h('p.page-lede', lede) : null), ...children);
}

/** group(title | {title, aside}, ...rows): rows go in a .card unless one is already a .card / bare block. */
export function group(head, ...rows) {
  const title = head && typeof head === 'object' && !(head instanceof Node) ? head.title : head;
  const aside = head && typeof head === 'object' && !(head instanceof Node) ? head.aside : null;
  const flat = rows.flat().filter(Boolean);
  const bare = flat.length === 1 && flat[0].dataset && flat[0].dataset.bare !== undefined;
  return h('div.group',
    title || aside ? h('div.group-head', title ? h('h2.group-title', title) : h('span'), aside ? (aside instanceof Node ? aside : h('span.group-aside', aside)) : null) : null,
    bare ? flat[0] : h('div.card', flat));
}

export function row(label, desc, control, { stacked = false } = {}) {
  return h(`div.row${stacked ? '.is-stacked' : ''}`,
    h('div.row-text', h('span.row-label', label), desc ? h('span.row-desc', desc) : null),
    control ? h('div.row-control', control) : null);
}

/** A padded block inside a card. */
export function block(...children) {
  return h('div.stack', children);
}

export function callout(text, tone) {
  return h('div.callout', { 'data-tone': tone || null, html: '<svg viewBox="0 0 16 16"><circle cx="8" cy="8" r="6.5"/><path d="M8 7.2v4M8 4.9v.1"/></svg>' },
    typeof text === 'string' ? h('span', text) : text);
}

export function badge(text, tone) {
  return h('span.badge', { 'data-tone': tone || null }, text);
}

// ---------------------------------------------------------------- bound controls

export function toggle(path, { onChange, label } = {}) {
  const input = h('input.toggle', { type: 'checkbox', role: 'switch', 'aria-label': label || path });
  input.addEventListener('change', () => (onChange ? onChange(input.checked) : setPath(path, input.checked)));
  bind(() => { input.checked = !!cfg(path, false); });
  return input;
}

/** options: [[value, label]]. The thumb glides to `--i`. */
export function segmented(options, getValue, setValue, { label } = {}) {
  const el = h('div.seg', { role: 'radiogroup', 'aria-label': label || null, style: { '--n': String(options.length), '--i': '0' } });
  const btns = options.map(([v, text]) => {
    const b = h('button', { type: 'button', role: 'radio', onclick: () => setValue(v) }, text);
    b.dataset.value = v;
    el.append(b);
    return b;
  });
  bind(() => {
    const cur = String(getValue());
    btns.forEach((b, i) => {
      const on = b.dataset.value === cur;
      b.setAttribute('aria-checked', String(on));
      if (on) el.style.setProperty('--i', String(i));
    });
  });
  return el;
}

export function select(options, getValue, setValue, attrs = {}) {
  const el = h('select', attrs, options.map(([v, l]) => h('option', { value: v }, l)));
  el.addEventListener('change', () => setValue(el.value));
  bind(() => {
    const v = String(getValue() ?? '');
    if (v && ![...el.options].some((o) => o.value === v)) el.append(h('option', { value: v }, v));
    el.value = v;
  });
  return h('span.select', el);
}

export function selectPath(path, options, attrs) {
  return select(options, () => cfg(path, ''), (v) => setPath(path, v), attrs);
}

/** Commits on blur / Enter, never while typing. */
export function text(path, { placeholder = '', map = (v) => v, label, mono = false } = {}) {
  const el = h(`input.input${mono ? '.mono' : ''}`, { type: 'text', placeholder, spellcheck: 'false', 'aria-label': label || placeholder || path });
  el.addEventListener('change', () => {
    const v = map(el.value);
    if (v !== cfg(path)) setPath(path, v);
  });
  el.addEventListener('keydown', (e) => { if (e.key === 'Enter') el.blur(); });
  bind(() => { if (document.activeElement !== el) el.value = cfg(path, '') ?? ''; });
  return el;
}

/** Range with a fill (`--value` 0..1), a scale under it and the current value. */
export function slider(path, { min, max, step, format = (v) => v, invert = false, scale, label }) {
  const el = h('input.range', { type: 'range', min, max, step, 'aria-label': label || path });
  const out = h('span.small.muted.tnum');
  const show = () => {
    el.style.setProperty('--value', ((el.value - min) / (max - min)).toFixed(3));
    out.textContent = format(Number(el.value));
  };
  el.addEventListener('input', show);
  el.addEventListener('change', () => setPath(path, invert ? +(max + min - Number(el.value)).toFixed(3) : Number(el.value)));
  bind(() => {
    if (document.activeElement === el) return;
    const v = Number(cfg(path, min));
    el.value = invert ? max + min - v : v;
    show();
  });
  const sc = scale ? h('span.range-scale', h('span', scale[0]), out, h('span', scale[1])) : h('span.range-scale', out);
  return h('div.range-wrap', el, sc);
}

export function button(label, onclick, { kind = '', iconName, attrs = {} } = {}) {
  const cls = kind ? kind.split(' ').map((k) => `.btn-${k}`).join('') : '';
  return h(`button.btn${cls}`, { type: 'button', onclick, ...attrs }, iconName ? svg(iconName) : null, label);
}

// ---------------------------------------------------------------- composite controls

/** `.choice-progress` for a model download, fed by `download` events. */
export function progress(item) {
  const bar = h('span.progress', { role: 'progressbar' }, h('i'));
  const num = h('span.num');
  const el = h('span.choice-progress', bar, num);
  bind(() => {
    const d = store.downloads[item];
    el.hidden = !d || (d.total && d.done >= d.total);
    if (!d) return;
    const f = d.total ? Math.min(1, d.done / d.total) : 0;
    bar.style.setProperty('--value', f.toFixed(4));
    bar.setAttribute('aria-valuenow', String(Math.round(f * 100)));
    num.textContent = bytesPair(d.done, d.total);
  });
  return el;
}

/** Model / provider picker: `.card.choices > label.choice`. side(e) -> node for `.choice-side`. */
export function choices(list, selectedId, onPick, { name, side, progressFor, recommended } = {}) {
  return h('div.card.choices', { role: 'radiogroup', 'data-bare': '' }, list.map((e) => {
    const input = h('input', { type: 'radio', name: name || 'choice', checked: e.id === selectedId || null });
    input.addEventListener('change', () => onPick(e));
    return h('label.choice', input,
      h('span.choice-title', e.label, e.id === recommended ? [' ', badge('Recommended', 'accent')] : null),
      h('span.choice-meta', [e.note, e.languages].filter(Boolean).join(' · ')),
      side ? h('span.choice-side', side(e)) : null,
      progressFor ? progressFor(e) : null);
  }));
}

export function keychainName() {
  return store.platform === 'macos' ? 'Keychain' : store.platform === 'windows' ? 'Credential Manager' : 'keyring';
}

const STAGE_NAMES = { stt: 'Transcription', refine: 'Refinement' };

/** One status for several stage tests (a connector's Test checks both stages). */
function combineTests(pairs) {
  const rs = pairs.map(([s, p]) => [s, store.tests[`${s}:${p}`]]).filter(([, r]) => r);
  if (!rs.length) return undefined;
  if (rs.some(([, r]) => r.pending)) return { pending: true };
  const bad = rs.find(([, r]) => !r.ok);
  if (bad) return { ok: false, message: `${STAGE_NAMES[bad[0]] || bad[0]}: ${bad[1].message}`, ms: 0 };
  const what = rs.length === pairs.length && pairs.length > 1 ? 'Both stages work' : `${rs.map(([s]) => STAGE_NAMES[s] || s).join(' and ')} works`;
  return { ok: true, message: what, ms: rs.reduce((n, [, r]) => n + (r.ms || 0), 0) };
}

/** API key + Test: `.key-field` then `.status[data-tone]`. The key leaves the page in the save call and the field is cleared.
 *  `tests()` -> [[stage, provider], …] makes Test check several stages with one key (cloud connectors). */
export function secretField(provider, stage, providerFn = () => provider, { tests } = {}) {
  const input = h('input.input.mono', { type: 'password', autocomplete: 'off', spellcheck: 'false', 'aria-label': `${provider} API key` });
  const reveal = h('button.field-btn', { type: 'button', 'aria-label': 'Show key', html: '<svg viewBox="0 0 16 16"><path d="M1.5 8S4 3.5 8 3.5 14.5 8 14.5 8 12 12.5 8 12.5 1.5 8 1.5 8z"/><circle cx="8" cy="8" r="2"/></svg>' });
  reveal.addEventListener('click', () => { input.type = input.type === 'password' ? 'text' : 'password'; });
  const save = () => {
    const v = input.value.trim();
    if (!v) return;
    saveSecret(provider, v);
    input.value = '';
    input.type = 'password';
    input.blur();
  };
  input.addEventListener('keydown', (e) => { if (e.key === 'Enter') save(); });
  input.addEventListener('change', save);
  const testBtn = button('Test', () => (tests ? tests().forEach(([s, p]) => test(s, p)) : test(stage, providerFn())));
  const status = h('span.status');
  const remove = button('Remove key', () => saveSecret(provider, null), { kind: 'ghost sm' });
  bind(() => {
    const has = !!store.secrets[provider];
    input.placeholder = has ? '•••••••••••• saved · paste to replace' : 'Paste an API key';
    remove.hidden = !has;
    const r = tests ? combineTests(tests()) : store.tests[`${stage}:${providerFn()}`];
    testBtn.disabled = !!(r && r.pending);
    statusInto(status, r, has ? { tone: 'success', text: `Saved in ${keychainName()}` }
      : store.keyringError ? { tone: 'danger', text: store.keyringError } : { tone: '', text: 'No key saved' });
  });
  return h('div', { style: { width: '100%' } }, h('div.key-field', h('div.field', input, reveal), testBtn), h('div.hstack', status, h('span.spacer'), remove));
}

function statusInto(el, r, idle) {
  el.textContent = '';
  if (r && r.pending) { el.dataset.tone = 'busy'; el.append('Testing…'); return; }
  if (r && r.message) {
    el.dataset.tone = r.ok ? 'success' : 'danger';
    el.append(r.message);
    if (r.ms) el.append(h('span.meta', ` · ${fmtMs(r.ms)}`));
    return;
  }
  if (idle.tone) el.dataset.tone = idle.tone; else delete el.dataset.tone;
  el.append(idle.text);
}

/** Test button + status for local engines (no key). */
export function tester(stage, providerFn) {
  const status = h('span.status');
  const btn = button('Test', () => test(stage, providerFn()));
  bind(() => {
    const r = store.tests[`${stage}:${providerFn()}`];
    btn.disabled = !!(r && r.pending);
    status.hidden = !r;
    if (r) statusInto(status, r, { text: '' });
  });
  return h('div.hstack', status, btn);
}

/** "Press a key" Voice key picker: `button.capture[data-capturing] > kbd.keycap + .capture-edit`. */
export function keyPicker(getKey, onPick) {
  const cap = createCapture(store.platform);
  const field = h('button.capture', { type: 'button', 'aria-live': 'polite' });
  const err = h('span.status', { 'data-tone': 'danger', hidden: true });
  let capturing = false;

  const draw = () => {
    field.textContent = '';
    field.toggleAttribute('data-capturing', capturing);
    if (capturing) {
      const pressing = cap.pressing;
      if (pressing.length) for (const c of pressing) field.append(h('kbd.keycap', c.replace(/^(Key|Digit)/, '').replace(/(Left|Right)$/, ' $1').replace('Control', 'Ctrl')));
      else field.append('Press a key…');
      return;
    }
    const parts = keyLabel(getKey(), store.platform).split(' + ');
    parts.forEach((p, i) => {
      if (i) field.append(h('span.capture-plus', '+'));
      field.append(h('kbd.keycap', p));
    });
    field.append(h('span.capture-edit', 'Change'));
    field.setAttribute('aria-label', `Voice key: ${parts.join(' + ')}. Click to change`);
  };
  const stop = () => { capturing = false; cap.reset(); draw(); };
  field.addEventListener('click', () => { if (!capturing) { capturing = true; err.hidden = true; cap.reset(); draw(); } });
  field.addEventListener('blur', stop);
  field.addEventListener('keydown', (e) => {
    if (!capturing) return;
    e.preventDefault();
    if (e.code === 'Escape' && cap.pressing.length === 0) { stop(); return; }
    cap.keydown(e.code);
    draw();
  });
  field.addEventListener('keyup', (e) => {
    if (!capturing) return;
    e.preventDefault();
    const r = cap.keyup(e.code);
    if (!r) { draw(); return; }
    capturing = false;
    if (r.name) onPick(r.name);
    else { err.textContent = r.error; err.hidden = false; }
    draw();
  });
  bind(draw);

  const chips = h('div.hstack.key-presets', { role: 'group', 'aria-label': 'Common Voice keys' }, presets(store.platform).slice(0, 6).map((p) => {
    const b = button(keyLabel(p, store.platform), () => { err.hidden = true; onPick(p); }, { kind: 'ghost sm', attrs: { 'data-key': p } });
    b.prepend(svg('check'));
    return b;
  }));
  // the preset that is the current Voice key reads as selected
  bind(() => {
    const cur = getKey();
    for (const b of chips.children) b.setAttribute('aria-pressed', String(b.dataset.key === cur));
  });
  return h('div.keypicker', field, err, chips);
}

/* The brand o (app/icons/src/ochre-glyph-small.svg), for the mini HUD pills. */
const O_PATH = 'M50.1 81.2Q40.2 81.2 32.8 77.35Q25.4 73.5 21.35 66.45Q17.3 59.4 17.3 49.9Q17.3 40.4 21.35 33.4Q25.4 26.4 32.8 22.6Q40.2 18.8 50.2 18.8Q60.2 18.8 67.45 22.6Q74.7 26.4 78.7 33.4Q82.7 40.4 82.7 50Q82.7 59.5 78.7 66.55Q74.7 73.6 67.4 77.4Q60.1 81.2 50.1 81.2ZM50.1 65.2Q57 65.2 60.65 61.05Q64.3 56.9 64.3 49.9Q64.3 42.9 60.65 38.85Q57 34.8 50.1 34.8Q43.1 34.8 39.4 38.85Q35.7 42.9 35.7 49.9Q35.7 56.9 39.4 61.05Q43.1 65.2 50.1 65.2Z';
const GX_GLYPH = {
  o: `<svg class="gx-o" viewBox="0 0 100 100" aria-hidden="true"><path d="${O_PATH}"/></svg>`,
  check: '<svg class="gx-ico" viewBox="0 0 16 16" aria-hidden="true"><path d="M3.6 8.4l2.9 2.9 5.9-6.3"/></svg>',
  x: '<svg class="gx-ico" viewBox="0 0 16 16" aria-hidden="true"><path d="M5 5l6 6M11 5l-6 6"/></svg>',
};
/* Timeline cues under the key (64 x 12): the key's presses over time. `held` a long press,
   `tap` a short one, `rec` recording that is already going (dashed). */
const GX_TIME = {
  hold: '<path class="t-base" d="M2 6h60"/><rect class="t-held" x="6" y="2.5" width="46" height="7" rx="3.5"/>',
  double: '<path class="t-base" d="M2 6h10"/><rect class="t-tap" x="12" y="2.5" width="7" height="7" rx="3.5"/><rect class="t-tap" x="23" y="2.5" width="7" height="7" rx="3.5"/><path class="t-rec" d="M35 6h27"/>',
  tap: '<path class="t-rec" d="M2 6h26"/><rect class="t-tap" x="31" y="2.5" width="7" height="7" rx="3.5"/><path class="t-base" d="M42 6h20"/>',
  esc: '<path class="t-rec" d="M2 6h26"/><rect class="t-tap t-neutral" x="31" y="2.5" width="7" height="7" rx="3.5"/><path class="t-base" d="M42 6h20"/>',
};

/** The four gestures, each a small scene: the key (a real keycap with the user's key name), when it
    is pressed (a timeline cue), and what the HUD does (a mini pill). Styled in shared/app.css. */
export function gestures(keyName) {
  const k = keyName;
  const long = k.length > 9 ? '.is-long' : '';
  const scene = (kind, key, keyCls, pill) => h(`div.gx.gx-${kind}`, { 'aria-hidden': 'true' },
    h(`kbd.keycap.gx-key${keyCls}${key === 'Esc' ? '' : long}`, key),
    fromHtml(`<svg class="gx-time" viewBox="0 0 64 12" aria-hidden="true">${GX_TIME[kind]}</svg>`),
    pill);
  const pill = (tone, glyph, label) => h(`span.gx-pill`, { 'data-tone': tone },
    h('span.gx-orb', { html: GX_GLYPH[glyph] }),
    h('span.gx-label', label));
  const g = (title, desc, sc) => h('div.gesture', sc, h('div.g-title', title), h('div.g-desc', desc));
  return h('div.gestures',
    g('Hold to talk', `Hold ${k}, speak, release. The text lands where you’re typing.`,
      scene('hold', k, '.is-down', pill('listening', 'o', 'Dictating'))),
    g('Double-tap to lock', `Tap ${k} twice to keep recording hands-off.`,
      scene('double', k, '', pill('locked', 'o', 'Locked'))),
    g('Tap to finish', `While locked, one tap of ${k} stops and inserts.`,
      scene('tap', k, '', pill('done', 'check', 'Inserted'))),
    g('Esc to cancel', 'Press Esc while recording to throw it away.',
      scene('esc', 'Esc', '', pill('cancel', 'x', 'Cancelled'))));
}
