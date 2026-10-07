// First-run onboarding: welcome, permissions, Voice key, speech model, try it. Finishing sets
// `ui.onboarded` and closes the window. `?step=<id>` opens at a step (screenshots).
// Markup: theme/components.css contract (docs/design.md §8); the welcome illustration is the real
// HUD pill (theme/hud.css).

import { store, subscribe, start, setPath, cfg } from '../shared/store.js';
import { send, invoke } from '../shared/bridge.js';
import { label as keyLabel } from '../shared/keys.js';
import { ms as fmtMs } from '../shared/format.js';
import { h, group, row, badge, choices, progress, keyPicker, gestures, button, callout, fromHtml, resetBindings, refreshBindings } from '../shared/ui.js';
import { redrawFor, tryInitial, tryReduce, tryStatus } from './try.js';

const STEPS = ['welcome', 'permissions', 'key', 'model', 'try'];
const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
let step = Math.max(0, STEPS.indexOf(params.get('step') || 'welcome'));
const KEY = () => keyLabel(cfg('hotkey.key', 'right_alt'), store.platform);

function page(kicker, title, lede, ...children) {
  return h('section.page', h('header.page-head', h('div.kicker', kicker), h('h1.page-title', title), lede ? h('p.page-lede', lede) : null), ...children);
}

function welcome() {
  const hud = fromHtml(`<div class="hud" data-state="recording" data-shown>
    <div class="hud-pill" role="presentation">
      <span class="hud-glyph" aria-hidden="true"><span class="hud-o"><svg class="hud-o-glyph" viewBox="0 0 100 100"><path d="M50.1 81.2Q40.2 81.2 32.8 77.35Q25.4 73.5 21.35 66.45Q17.3 59.4 17.3 49.9Q17.3 40.4 21.35 33.4Q25.4 26.4 32.8 22.6Q40.2 18.8 50.2 18.8Q60.2 18.8 67.45 22.6Q74.7 26.4 78.7 33.4Q82.7 40.4 82.7 50Q82.7 59.5 78.7 66.55Q74.7 73.6 67.4 77.4Q60.1 81.2 50.1 81.2ZM50.1 65.2Q57 65.2 60.65 61.05Q64.3 56.9 64.3 49.9Q64.3 42.9 60.65 38.85Q57 34.8 50.1 34.8Q43.1 34.8 39.4 38.85Q35.7 42.9 35.7 49.9Q35.7 56.9 39.4 61.05Q43.1 65.2 50.1 65.2Z"/></svg></span></span>
      <span class="hud-text"><span class="hud-label">Dictating</span><span class="hud-hint">release to paste</span></span>
    </div></div>`);
  const meter = window.HudKit && window.HudKit.oMeter(hud.querySelector('.hud-o'));
  if (meter) {
    // a synthetic voice at the core's ~30 level events per second, so the o grows with it as it would
    const voice = window.HudKit.fakeVoice(11);
    const t0 = performance.now();
    const tick = () => { if (!hud.isConnected) { meter.stop(); return; } meter.push(voice(performance.now() - t0)); setTimeout(tick, 33); };
    setTimeout(tick, 0);
  }
  return h('section.page.welcome',
    h('div.lockup.welcome-lockup', { role: 'img', 'aria-label': 'Ochre' },
      h('img', { src: '../theme/brand/ochre-icon.svg', alt: '' }),
      h('span.wordmark')),
    h('h1.page-title', 'Talk, and your words land where you type.'),
    h('p.page-lede', 'Hold a key, speak naturally, let go. Ochre turns it into clean text in whatever app you’re in, privately, on this computer.'),
    h('div.welcome-hud', hud));
}

const PERMS = {
  windows: [['microphone', 'Microphone', 'Windows asks the first time you dictate. If nothing is heard, check Settings › Privacy & security › Microphone.']],
  linux: [['microphone', 'Microphone', 'Uses your default input (PipeWire / PulseAudio).'],
    ['input_group', 'Voice key', 'Works as is on X11. On Wayland the key is read from /dev/input: join the “input” group, or bind a shortcut to `ochre toggle`.'],
    ['typing_tool', 'Typing', 'Types into the focused app: xdotool on X11; ydotool, dotool, wtype or kwtype on Wayland.']],
  macos: [['microphone', 'Microphone', 'Lets Ochre hear you while you hold the Voice key. It turns the mic off after 5 minutes without a dictation.'],
    ['accessibility', 'Accessibility', 'Lets Ochre type into other apps.'],
    ['input_monitoring', 'Input Monitoring', 'Lets Ochre notice the Voice key in any app.']],
};

// Issues without a row of their own (Linux typing setup, the paste fallback, no display).
const EXTRA_TITLES = { uinput: 'Typing access', ydotoold: 'Typing daemon', clipboard_tool: 'Clipboard tool', display: 'Display' };

function permissions() {
  const missing = new Map(store.permissions.map((p) => [p.name, p.fix]));
  const list = PERMS[store.platform] || PERMS.windows;
  const extra = store.permissions.filter((p) => !list.some(([n]) => n === p.name)).map((p) => [p.name, EXTRA_TITLES[p.name] || p.name.replace(/_/g, ' '), p.fix]);
  const rows = [...list, ...extra].map(([name, title, desc]) => {
    const need = missing.has(name);
    // macOS: show the system prompt and open the pane with the switch; the list updates on its own.
    const allow = need && store.platform === 'macos'
      ? button('Allow…', () => send({ op: 'open_permission_settings', name }), { kind: 'sm' })
      : null;
    return row(title, need ? missing.get(name) : desc, need ? h('div.hstack', badge('Needs access', 'warning'), allow) : badge('Ready', 'success'));
  });
  return page('Step 1 of 4', 'A few permissions', store.platform === 'macos'
    ? 'macOS keeps apps from listening to keys or typing for you until you say so. Turn these on once.'
    : 'Ochre needs to hear you and type for you. Nothing is recorded unless you hold the Voice key.',
  group(null, ...rows),
  store.platform === 'macos' && missing.size ? callout('Click Allow, switch Ochre on in System Settings, then come back here: this list updates on its own, no restart needed.') : null);
}

function key() {
  return page('Step 2 of 4', 'Your Voice key', 'One key does everything. Pick one you never type with.',
    group(null, row('Voice key', null, keyPicker(() => cfg('hotkey.key', 'right_alt'), (name) => setPath('hotkey.key', name)), { stacked: true }),
      h('div.stack', gestures(KEY()))));
}

function model() {
  const local = store.engines.stt.filter((e) => e.kind === 'local');
  const e = store.engines.stt.find((x) => x.id === cfg('stt.engine')) || local[0];
  const name = (e && (cfg('stt.model') || e.default_model)) || '';
  const st = store.state || {};
  const d = store.downloads[name];
  const done = d && d.total && d.done >= d.total;
  let status = null;
  if (st.state === 'loading' && !(d && !done)) status = h('span.status', { 'data-tone': 'busy' }, st.detail || 'Loading models…');
  else if (done || (st.state === 'idle' && !d)) status = h('span.status', { 'data-tone': 'success' }, 'Ready to dictate');
  return page('Step 3 of 4', 'The speech model', 'Runs on this computer, so your voice never leaves it. You can switch to a cloud provider with your own key later.',
    group({ title: 'Local model', aside: 'Stored on this computer' },
      choices(local, e && e.id, (x) => setPath('stt.engine', x.id), {
        name: 'ob-model', recommended: local[0] && local[0].id,
        side: (x) => (x.id === (e && e.id) && !done ? button('Download', () => send({ op: 'download_model', stage: 'stt', name }), { kind: 'sm', iconName: 'download' }) : null),
        progressFor: (x) => progress(x.id === (e && e.id) ? name : x.default_model),
      })),
    h('div.hstack', status, h('span.spacer'),
      h('button.btn.btn-ghost.btn-sm', { type: 'button', onclick: () => invoke('open_page', { target: 'transcription' }) }, 'Prefer the cloud? Choose a provider')));
}

// The box is created once and kept: Ochre types into it, so rebuilding it would lose the text.
let trySt = tryInitial();
let tryBox = null;
const tryLine = h('div.try-result', { 'aria-live': 'polite' });

function drawTryStatus() {
  const st = tryStatus(trySt, KEY());
  tryLine.dataset.tone = st.tone;
  if (tryBox) tryBox.classList.toggle('is-done', st.tone === 'success');
  if (st.tone === 'tip') { tryLine.replaceChildren(h('span.small.muted', st.text)); return; }
  const check = h('span.try-check', { 'aria-hidden': 'true', html: '<svg viewBox="0 0 16 16"><path d="M3.6 8.4l2.9 2.9 5.9-6.3" pathLength="20"/></svg>' });
  tryLine.replaceChildren(h('span.status', { 'data-tone': st.tone === 'success' ? 'success' : 'warning' },
    st.tone === 'success' ? check : null,
    st.tone === 'success' ? h('strong', st.text) : st.text,
    st.ms != null ? h('span.meta', ` ${fmtMs(st.ms)} from letting go to text`) : null));
}

function tryIt() {
  if (!tryBox) {
    tryBox = h('textarea.input.try-box', { placeholder: `Click here, then hold ${KEY()} and say something. Let go when you’re done.`, spellcheck: 'false', 'aria-label': 'Try it' });
  } else {
    tryBox.placeholder = `Click here, then hold ${KEY()} and say something. Let go when you’re done.`;
  }
  drawTryStatus();
  return page('Step 4 of 4', 'Try it', 'Dictate into the box below. It works the same in every app.', tryBox, tryLine, micModeTip());
}

// macOS: Voice Isolation is the user's choice in the system Mic Mode menu (apps can't set it), and
// the menu lists Ochre only while its voice-processed mic runs (warm mic + voice processing).
function micModeTip() {
  if (store.platform !== 'macos' || !cfg('audio.voice_processing', true) || !cfg('audio.warm_mic', true)) return null;
  return callout(h('div.hstack',
    h('span', 'Other voices or noise getting in? Choose Voice Isolation in the Mac’s microphone mode menu. The Mac remembers it for Ochre.'),
    h('span.spacer'),
    button('Microphone mode…', () => send({ op: 'show_mic_modes' }), { kind: 'sm' })));
}

const RENDER = { welcome, permissions, key, model, try: tryIt };

function render() {
  resetBindings();
  $('.ob-step').replaceChildren(RENDER[STEPS[step]]());
  $('.ob-dots').replaceChildren(...STEPS.map((_, i) => h(`i${i <= step ? '.on' : ''}`)));
  $('.back').style.visibility = step === 0 ? 'hidden' : 'visible';
  $('.next').textContent = step === 0 ? 'Get started' : step === STEPS.length - 1 ? 'Finish' : 'Continue';
  $('.foot-hint').textContent = step === 0 ? 'Takes about a minute' : '';
}

$('.back').addEventListener('click', () => { if (step > 0) { step--; render(); } });
$('.next').addEventListener('click', () => {
  if (step < STEPS.length - 1) { step++; render(); return; }
  setPath('ui.onboarded', true);
  invoke('close_page');
});

subscribe((ev) => {
  const id = STEPS[step];
  if (ev.event === 'result') trySt = tryReduce(trySt, ev);
  const done = ev.event === 'download' && store.downloads[ev.item] && ev.done >= ev.total;
  const redraw = redrawFor(id, ev, { downloadDone: !!done });
  if (redraw === 'page') render();
  else if (redraw === 'patch') { drawTryStatus(); refreshBindings(); }
  else refreshBindings();
});

await start();
// only results from this session count (`?step=try` screenshots show the last one as new)
trySt = tryInitial(params.get('step') === 'try' ? null : store.result);
if (params.get('step') === 'try') {
  trySt = tryReduce(trySt, store.result);
  // the mock has no real typing: put the text where Ochre would have typed it
  if (params.has('mock') && trySt.result && trySt.result.inserted) { tryIt(); tryBox.value = trySt.result.text; }
}
render();
await document.fonts.ready;
requestAnimationFrame(() => {
  invoke('ui_ready');
  window.__ochreReady = true;
});
