// Settings window: sidebar nav + one calm column (theme/components.css contract). Every change
// goes out as `set_config{patch}`; the core answers with a fresh `config` event, which refreshes
// the bound controls in place. API keys go through `set_secret` and are never kept in the page.

import { store, subscribe, start, setPath, setPatch, cfg } from '../shared/store.js';
import { send, invoke, listen, inApp } from '../shared/bridge.js';
import { icon } from '../shared/icons.js';
import { label as keyLabel, pasteLabel, hint as keyHint } from '../shared/keys.js';
import { ms as fmtMs, words, appName, LANGUAGES, autoPick, localRefineLabel, READY, readyValue, readyPatch, readyHint } from '../shared/format.js';
import {
  h, section, group, row, block, callout, badge, toggle, segmented, select, selectPath, text, slider, button,
  progress, choices, secretField, tester, keyPicker, gestures, resetBindings, refreshBindings, keychainName, fromHtml,
} from '../shared/ui.js';

// History first: it's the page the window opens on.
const SECTIONS = [
  ['history', 'History', 'history'],
  ['general', 'General', 'general'],
  ['transcription', 'Transcription', 'transcription'],
  ['refinement', 'Refinement', 'refinement'],
  ['handsfree', 'Hands-free', 'handsfree'],
  ['dictionary', 'Dictionary', 'dictionary'],
  null,
  ['about', 'About', 'about'],
];

const $ = (s) => document.querySelector(s);
const HOME = 'history';
let current = (location.hash || `#${HOME}`).slice(1);
let structure = '';
const KEY = () => keyLabel(cfg('hotkey.key', 'right_alt'), store.platform);

// ---------------------------------------------------------------- helpers

const sttEngines = (kind) => store.engines.stt.filter((e) => !kind || e.kind === kind);
const refineEngines = (kind) => store.engines.refine.filter((e) => !kind || e.kind === kind);
const sttEngine = () => store.engines.stt.find((e) => e.id === cfg('stt.engine')) || store.engines.stt[0];
const refineKind = () => {
  const p = cfg('refine.provider', 'off');
  return p === 'off' || !p ? 'off' : p === 'local' ? 'local' : 'cloud';
};
let lastCloudStt = 'groq';
let lastCloudRefine = 'openai';
/** The note under a local refinement choice: what auto picked and why, sizes for the rest. */
function localRefineNote(m, note) {
  if (m === 'auto') {
    const p = autoPick(note);
    return `${p ? `Picked for this PC (${p.hardware}). ` : ''}4B with a GPU of 6 GB or more, 2B on smaller GPUs, 0.8B without a GPU.`;
  }
  return ({ 'ochre-refine-4b': '2.7 GB · GPU', 'ochre-refine-2b': '1.2 GB · GPU', 'ochre-refine-0.8b': '529 MB · runs on CPU', 'quill-4b': '2.6 GB · GPU', 'quill-2b': '1.2 GB · GPU', 'quill-0.8b': '505 MB · runs on CPU' })[m] || '';
}
const modelOptions = (e) => (e && e.models.length ? e.models : ['']).map((m) => [m, m || 'Default']);
const downloadButton = (stage, name) => button('Download', () => send({ op: 'download_model', stage, name }), { kind: 'sm', iconName: 'download' });
// Engine ids that share another provider's key (ochre-core secrets::ALIASES).
const KEY_ALIASES = { google: 'gemini' };
const keyFor = (id) => KEY_ALIASES[id] || id;

// ---------------------------------------------------------------- cloud connectors
// One provider, one key, both stages (ochre::connectors). The picker offers the single-key cloud
// connectors plus Custom (mix stages by hand, under Advanced) and Local only.

let customPinned = false; // "Custom" chosen while the stages still match a connector
let advanced = false;     // per-stage controls revealed under a connector
const sameModel = (a, b) => a === b || !a || !b;
const pickable = () => store.connectors.filter((c) => c.key_provider && !(c.extra_keys || []).length);
const isLocalStt = () => (sttEngine() || {}).kind === 'local';

function connectorId() {
  if (customPinned) return 'custom';
  const refineOff = cfg('refine.provider', 'off') === 'off';
  const c = pickable().find((x) => cfg('stt.engine') === x.stt.id && sameModel(cfg('stt.model') || '', x.stt.model)
    && (refineOff || (cfg('refine.provider') === x.refine.id && sameModel(cfg('refine.model') || '', x.refine.model))));
  if (c) return c.id;
  if (isLocalStt() && ['off', 'local', ''].includes(cfg('refine.provider', 'off'))) return 'local';
  return 'custom';
}
const connector = () => store.connectors.find((c) => c.id === connectorId());
const usesConnector = () => !!connector() && connectorId() !== 'local';

function cost(perHour) {
  if (!perHour) return 'No usage cost: everything runs on this computer.';
  return `About $${perHour.toFixed(2)} per hour of dictation, billed to your key.`;
}

function pickConnector(id) {
  advanced = false;
  if (id === 'custom') { customPinned = true; advanced = true; renderPage(false); return; }
  customPinned = false;
  if (id === 'local') {
    const local = sttEngines('local')[0];
    const refine = cfg('refine.provider', 'off') === 'off' ? 'off' : 'local';
    setPatch({ stt: { engine: local ? local.id : 'parakeet', model: '' }, refine: { provider: refine, model: '' } });
    return;
  }
  const c = store.connectors.find((x) => x.id === id);
  if (c) setPatch({ stt: { engine: c.stt.id, model: c.stt.model }, refine: { provider: c.refine.id, model: c.refine.model, mode: c.refine.mode } });
}

/** "Cloud connector" group, at the top of Transcription and Refinement. */
function connectorGroup() {
  if (!store.connectors.length) return null;
  const id = connectorId();
  const c = connector();
  const options = [...pickable().map((x) => [x.id, x.label]), ['custom', 'Custom'], ['local', 'Local only']];
  const seg = segmented(options, connectorId, pickConnector, { label: 'Cloud connector' });
  seg.classList.add('is-wide');
  seg.style.width = '100%';
  const desc = id === 'custom' ? 'Each stage set by hand below, under Advanced.'
    : id === 'local' ? 'Speech and cleanup both run on this computer.'
      : 'One API key sets up transcription and refinement together.';
  const rows = [row('Cloud connector', desc, seg, { stacked: true })];
  if (c && id !== 'local') {
    const tests = () => [['stt', c.stt.id], ...(cfg('refine.provider', 'off') === 'off' ? [] : [['refine', c.refine.id]])];
    rows.push(row(`${c.label} API key`, `One key for both stages, kept in ${keychainName()}, never in config files.`,
      secretField(c.key_provider, 'stt', () => c.stt.id, { tests }), { stacked: true }));
  }
  rows.push(row(id === 'custom' ? 'Cost' : 'About the cost', id === 'custom' ? 'Depends on the providers you pick; each shows its price.' : `${cost(c ? c.est_cost_per_hour : 0)} ${c && c.notes && id !== 'local' ? c.notes : ''}`.trim(),
    id !== 'custom' && id !== 'local'
      ? button(advanced ? 'Hide advanced' : 'Advanced', () => { advanced = !advanced; renderPage(false); }, { kind: 'ghost sm' })
      : null));
  return group({ title: 'Cloud connector', aside: id === 'custom' || id === 'local' ? null : 'Both stages, one key' }, rows);
}

// ---------------------------------------------------------------- sections

function general() {
  const mod = (k) => ({ ctrl: store.platform === 'macos' ? 'Control' : 'Ctrl', alt: store.platform === 'macos' ? 'Option' : 'Alt' })[k];
  return section('General', 'How you start and stop a dictation, and how Ochre fits into your desktop.',
    group('Dictation',
      row('Voice key', keyHint(store.platform),
        keyPicker(() => cfg('hotkey.key', 'right_alt'), (name) => setPath('hotkey.key', name)), { stacked: true }),
      block(gestures(KEY())),
      row('Paste last transcript', 'Types your last dictation again where you’re typing, without touching your clipboard. Also in the tray menu.',
        pasteLastControl()),
      row('Skip refinement for one dictation', 'Hold this while you finish to insert the raw transcript.',
        selectPath('hotkey.raw_modifier', [['shift', 'Shift'], ['ctrl', mod('ctrl')], ['alt', mod('alt')], ['', 'Off']], { 'aria-label': 'Raw modifier' })),
      row('Double-tap speed', 'How quickly the second tap must follow the first to lock recording.',
        slider('hotkey.double_tap_ms', { min: 200, max: 600, step: 10, format: (v) => `${v} ms`, scale: ['Quick', 'Relaxed'], label: 'Double-tap speed' }))),
    group('System',
      row('Start at login', 'Ochre waits quietly in the tray.', toggle('ui.start_at_login', { label: 'Start at login' })),
      row('Appearance', 'Light by default. System follows your OS.', segmented([['light', 'Light'], ['dark', 'Dark'], ['system', 'System']], () => cfg('ui.theme', 'light'), (v) => setPath('ui.theme', v), { label: 'Appearance' })),
      row('Language', 'What you speak. Detecting automatically works for most people.',
        select(LANGUAGES, () => cfg('stt.language') || '', (v) => setPath('stt.language', v || null), { 'aria-label': 'Language' })),
      row('Show live text', 'Your words appear in the pill as you talk. Nothing is typed until you finish.', toggle('ui.show_partials', { label: 'Show live text' })),
      row('Sounds', 'A soft tick when recording starts and stops.', toggle('audio.earcons', { label: 'Sounds' }))),
    group('Typing',
      row('Insert text by', 'Typing leaves your clipboard alone. Pasting is faster for very long text.',
        segmented([['type', 'Typing'], ['paste', 'Pasting']], () => cfg('inject.method', 'type'), (v) => setPath('inject.method', v), { label: 'Insert text by' })),
      row('Add a space after', 'So the next dictation joins up naturally.', toggle('inject.trailing_space', { label: 'Add a space after' })),
      store.platform === 'macos' ? null
        : row('Keep the microphone ready', 'Recording starts instantly and never clips your first word. Your system may show the mic as in use.', toggle('audio.warm_mic', { label: 'Keep the microphone ready' }))),
    store.platform === 'macos' ? macMicGroup() : null);
}

/** macOS: Apple's voice processing (on by default) and the system Mic Mode menu. */
function macMicGroup() {
  const on = cfg('audio.voice_processing', true);
  const ready = readyValue(cfg('audio.warm_mic', true), cfg('audio.warm_idle_release_s', 300));
  return group('Microphone',
    row('Keep the microphone ready', readyHint(ready, cfg('handsfree.enabled', false)),
      select(READY, () => readyValue(cfg('audio.warm_mic', true), cfg('audio.warm_idle_release_s', 300)), (v) => setPatch(readyPatch(v)), { 'aria-label': 'Keep the microphone ready' })),
    row('Use the built-in mic with headphones', 'With AirPods or other Bluetooth headphones connected, Ochre records with the Mac’s microphone, so your headphones keep their full sound quality (their mic also makes more mistakes). Picking the headphones as the input overrides this.',
      toggle('audio.avoid_bluetooth_mic', { label: 'Use the built-in mic with headphones' })),
    row('Filter background noise', 'Uses the Mac’s own voice processing, like FaceTime: removes echo from your speakers and steady noise while the microphone is kept ready. Turn off for the raw microphone.',
      toggle('audio.voice_processing', { label: 'Filter background noise' })),
    on ? row('Microphone mode', 'Choose Voice Isolation to also remove other people’s voices. The Mac remembers it for Ochre.',
      button('Choose…', () => send({ op: 'show_mic_modes' }), { kind: 'sm' })) : null);
}

/** Keycaps for the paste-last shortcut plus its picker (`hotkey.paste_last`). */
function pasteLastControl() {
  const voice = cfg('hotkey.key', 'right_alt');
  const text = pasteLabel(cfg('hotkey.paste_last', 'down'), voice, store.platform);
  const caps = h('span.paste-keys');
  text.split(' + ').filter(Boolean).forEach((p, i) => {
    if (i) caps.append(h('span.capture-plus', '+'));
    caps.append(h('kbd.keycap', p));
  });
  const options = [['down', `${KEY()} + ↓`], ['up', `${KEY()} + ↑`], ['ctrl+alt+v', keyLabel('ctrl+alt+v', store.platform)], ['', 'Off']];
  return h('div.paste-last', text ? caps : null,
    select(options, () => cfg('hotkey.paste_last', 'down'), (v) => setPath('hotkey.paste_last', v), { 'aria-label': 'Paste last transcript shortcut' }));
}

function transcription() {
  const conn = connectorGroup();
  const e = sttEngine();
  const kind = e ? e.kind : 'local';
  if (kind === 'cloud' && e) lastCloudStt = e.id;
  const pickKind = (k) => {
    if (k === kind) return;
    const target = k === 'cloud' ? (sttEngines('cloud').find((x) => x.id === lastCloudStt) || sttEngines('cloud')[0]) : sttEngines('local')[0];
    if (target) setPatch({ stt: { engine: target.id, model: '' } });
  };
  const model = () => cfg('stt.model') || (e && e.default_model) || '';
  const head = group(null,
    row('Engine', kind === 'local' ? 'On this computer. Audio never leaves it.' : 'Your own API key, billed by the provider. Usually cents a month.',
      segmented([['local', 'Local'], ['cloud', 'Cloud']], () => kind, pickKind, { label: 'Engine' })));
  const pick = (x) => setPatch({ stt: { engine: x.id, model: '' } });
  const lede = 'Turn speech into words, privately on this computer or with a cloud provider you choose.';
  const unreachable = () => group('When the cloud is unreachable',
    row('Fall back to the local model', 'Used only if it’s already downloaded.', toggle('stt.fallback_local', { label: 'Fall back to the local model' })),
    row('Give up after', null, selectPath('stt.cloud_timeout_ms', [['4000', '4 seconds'], ['8000', '8 seconds'], ['15000', '15 seconds']], { 'aria-label': 'Give up after' })));

  if (usesConnector() && !advanced && e && kind === 'cloud') {
    return section('Transcription', lede, conn,
      group('Speech model', row(e.label, [model(), e.languages].filter(Boolean).join(' · '), null)),
      unreachable(),
      callout(`Audio goes only to ${e.label}, and only while you dictate.`));
  }

  if (kind === 'local') {
    const devices = [['auto', 'Automatic'], ['cpu', 'CPU']];
    if (store.platform !== 'macos') devices.push(['cuda', 'NVIDIA GPU (CUDA)']);
    if (store.platform === 'windows') devices.push(['directml', 'Any GPU (DirectML)']);
    if (store.platform === 'macos') devices.push(['coreml', 'Neural Engine (Core ML)']);
    const local = sttEngines('local');
    // the configured engine isn't in this build (e.g. Whisper without its cargo feature)
    const want = cfg('stt.engine');
    const missing = want && store.engines.stt.length && !store.engines.stt.some((x) => x.id === want);
    return section('Transcription', lede, conn, head,
      missing ? callout(`The “${want}” speech engine isn’t included in this build of Ochre. Choose a model below.`, 'warning') : null,
      group({ title: 'Local model', aside: 'Stored on this computer' },
        choices(local, e && e.id, pick, {
          name: 'stt-local', recommended: local[0] && local[0].id,
          side: (x) => (x.id === (e && e.id) ? downloadButton('stt', model()) : null),
          progressFor: (x) => progress(x.id === (e && e.id) ? model() : x.default_model),
        })),
      group('Options',
        row('Model', null, select(modelOptions(e), model, (v) => setPath('stt.model', v), { 'aria-label': 'Model' })),
        row('Run on', 'Automatic picks the fastest device it finds.', selectPath('stt.device', devices, { 'aria-label': 'Run on' })),
        row('Check it works', 'Transcribes a short built-in clip.', tester('stt', () => e && e.id))));
  }

  return section('Transcription', lede, conn, head,
    group('Cloud provider',
      row('Provider', e.note, select(sttEngines('cloud').map((x) => [x.id, x.label]), () => e.id, (v) => pick({ id: v }), { 'aria-label': 'Provider' })),
      row('Model', e.languages, select(modelOptions(e), model, (v) => setPath('stt.model', v), { 'aria-label': 'Model' })),
      row('API key', `Kept in ${keychainName()}, never in config files.`, secretField(keyFor(e.id), 'stt', () => e.id), { stacked: true })),
    unreachable(),
    callout(`Audio goes only to ${e.label}, and only while you dictate.${e.note ? ` Cost: ${e.note}, billed to your key.` : ''}`));
}

function refinement() {
  const conn = connectorGroup();
  const kind = refineKind();
  const provider = cfg('refine.provider', 'off');
  if (kind === 'cloud') lastCloudRefine = provider;
  const e = store.engines.refine.find((x) => x.id === provider);
  const pick = (k) => {
    if (k === kind) return;
    setPatch({ refine: { provider: k === 'off' ? 'off' : k === 'local' ? 'local' : lastCloudRefine, model: '' } });
  };
  const model = () => cfg('refine.model') || (e && e.default_model) || '';
  const mode = cfg('refine.mode', 'clean');
  const head = group(null,
    row('Refine with', 'Removes fillers and false starts, applies “no wait, make that Tuesday”, fixes punctuation. Never adds content.',
      segmented([['off', 'Off'], ['local', 'Local'], ['cloud', 'Cloud']], () => kind, pick, { label: 'Refine with' })),
    kind !== 'off' ? row('Style', mode === 'polish' ? 'A light rewrite for clarity and flow. Still your meaning, never new content.' : 'Fillers, false starts and self-corrections removed. Your wording stays.',
      segmented([['clean', 'Clean'], ['polish', 'Polish']], () => mode, (v) => setPath('refine.mode', v), { label: 'Style' })) : null);
  const lede = 'An optional pass that tidies the transcript, in your own voice, before it’s typed.';
  const styleRow = () => row('Style', mode === 'polish' ? 'A light rewrite for clarity and flow. Still your meaning, never new content.' : 'Fillers, false starts and self-corrections removed. Your wording stays.',
    segmented([['clean', 'Clean'], ['polish', 'Polish']], () => mode, (v) => setPath('refine.mode', v), { label: 'Style' }));

  if (usesConnector() && !advanced) {
    const c = connector();
    const re = store.engines.refine.find((x) => x.id === c.refine.id);
    const on = kind !== 'off';
    const setOn = (v) => setPatch({ refine: v === 'on' ? { provider: c.refine.id, model: c.refine.model, mode: c.refine.mode } : { provider: 'off' } });
    return section('Refinement', lede, conn,
      group(null,
        row(`Refine with ${re ? re.label : c.label}`, on ? `${c.refine.model || (re && re.default_model) || ''}: removes fillers and false starts, applies “no wait, make that Tuesday”, fixes punctuation. Never adds content.` : 'Off: text is typed exactly as transcribed.',
          segmented([['off', 'Off'], ['on', 'On']], () => (on ? 'on' : 'off'), setOn, { label: 'Refine' })),
        on ? styleRow() : null),
      on ? appStyles() : null);
  }

  if (kind === 'off') {
    return section('Refinement', lede, conn, head,
      callout('Refinement is off: text is typed exactly as heard, with the speech model’s own punctuation.'));
  }
  const engine = kind === 'local'
    ? group({ title: 'Local model', aside: 'Runs in a small server that stays warm' },
      choices(refineEngines('local').flatMap((x) => x.models.map((m) => ({ ...x, id: m, label: localRefineLabel(m, x.note), note: localRefineNote(m, x.note), languages: '' }))),
        model(), (x) => setPath('refine.model', x.id), {
          name: 'refine-local', recommended: e && e.default_model,
          side: (x) => (x.id === model() ? downloadButton('refine', x.id) : null),
          progressFor: (x) => progress(x.id),
        }))
    : group('Cloud model',
      row('Provider', e ? e.note : null, select(refineEngines('cloud').map((x) => [x.id, x.label]), () => provider, (v) => setPatch({ refine: { provider: v, model: '' } }), { 'aria-label': 'Provider' })),
      row('Model', null, e && e.models.length ? select(modelOptions(e), model, (v) => setPath('refine.model', v), { 'aria-label': 'Model' }) : text('refine.model', { placeholder: 'model id, e.g. llama3.2:3b', label: 'Model' })),
      provider === 'custom' ? row('Server URL', 'Any OpenAI-compatible endpoint: Ollama, LM Studio, vLLM…', text('refine.base_url', { placeholder: 'http://127.0.0.1:11434/v1', mono: true, label: 'Server URL' })) : null,
      e && e.needs_key ? row('API key', `Kept in ${keychainName()}, never in config files.`, secretField(keyFor(provider), 'refine', () => provider), { stacked: true }) : row('Check it works', 'Sends one test sentence.', tester('refine', () => provider)));
  const localTest = kind === 'local' ? group(null, row('Check it works', 'Cleans a short test sentence.', tester('refine', () => 'local'))) : null;
  return section('Refinement', lede, conn, head, engine, localTest, longAndRepeats(), appStyles());
}

function longAndRepeats() {
  return group({ title: 'Long dictations and repeats' },
    row('Split long dictations', 'Cleans a long dictation a few sentences at a time. Auto turns it on only for models where it measurably helps, which is none of the current ones.',
      segmented([['auto', 'Auto'], ['on', 'On'], ['off', 'Off']], () => cfg('refine.chunk_long', 'auto'), (v) => setPath('refine.chunk_long', v), { label: 'Split long dictations' })),
    row('Said again, typed as heard', 'If you repeat a dictation within a minute of a cleanup, the repeat is typed exactly as heard. The earlier text is left as it is.',
      toggle('refine.redictation_raw', { label: 'Said again, typed as heard' })));
}

function appStyles() {
  const styles = cfg('refine.app_styles', {}) || {};
  const STY = [['casual', 'Casual'], ['neutral', 'Neutral'], ['formal', 'Formal'], ['literal', 'Literal']];
  const DESC = { casual: 'Relaxed, keeps contractions', neutral: 'Plain and clear', formal: 'Full sentences', literal: 'No rewriting near code' };
  const set = (next) => setPatch({ refine: { app_styles: next } });
  const removeSvg = '<svg viewBox="0 0 16 16"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/></svg>';
  const rows = Object.entries(styles).sort(([a], [b]) => a.localeCompare(b)).map(([app, style]) =>
    h('div.app-row',
      h('span.app-icon', appName(app).replace(/\s/g, '').slice(0, 2)),
      h('span.app-name', appName(app), h('small', DESC[style] || style)),
      select(STY, () => style, (v) => set({ ...styles, [app]: v }), { 'aria-label': `${appName(app)} style` }),
      h('button.btn.btn-ghost.btn-icon.btn-sm', { type: 'button', 'aria-label': `Remove ${appName(app)}`, html: removeSvg, onclick: () => { const n = { ...styles }; delete n[app]; set(n); } })));
  const appIn = h('input.input', { placeholder: 'Program name, e.g. slack', spellcheck: 'false', 'aria-label': 'Program name' });
  const add = () => {
    const a = appIn.value.trim().toLowerCase().replace(/\.exe$/, '');
    if (a) set({ ...styles, [a]: 'neutral' });
  };
  appIn.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') add(); });
  return group({ title: 'Per-app styles', aside: 'Matched by program name' }, ...rows,
    h('div.row', h('div.row-text', appIn), h('div.row-control', button('Add app', add, { kind: 'ghost sm', iconName: 'plus' }))));
}

function handsfree() {
  const phrase = 'transcribe';
  const say = (s, does) => [h('span.say-phrase', `“${s}”`), h('span.say-does', does)];
  return section('Hands-free', 'Say “transcribe” to start dictating without touching the keyboard.',
    group(null,
      row('Listen for the wake word', 'The microphone stays on while this is enabled, and your system will show it in use. Detection runs entirely on this computer.',
        toggle('handsfree.enabled', { label: 'Hands-free', onChange: (v) => send({ op: 'set_handsfree', enabled: v }) })),
      row('Sensitivity', 'Higher wakes more easily, and more often by mistake.',
        slider('handsfree.threshold', { min: 0.3, max: 0.9, step: 0.01, invert: true, format: (v) => `${Math.round(((v - 0.3) / 0.6) * 100)}%`, scale: ['Strict', 'Eager'], label: 'Sensitivity' })),
      row('Finish after silence', 'Inserts what you said when you stop talking.', slider('handsfree.idle_timeout_s', { min: 10, max: 120, step: 5, format: (v) => `${v} s`, label: 'Finish after silence' })),
      row('Pause during calls', 'Stops listening while another app uses the mic for a call.', toggle('handsfree.pause_on_calls', { label: 'Pause during calls' }))),
    group('Control phrases',
      block(h('div.says',
        say(phrase, 'starts a dictation'),
        say(`${phrase} stop`, 'finishes and types it'),
        say(`${phrase} send`, 'types it, then presses Enter'),
        say(`${phrase} scratch that`, 'drops the last sentence'),
        say(`${phrase} cancel`, 'throws it away')))));
}

function dictionary() {
  const list = cfg('dictionary.words', []) || [];
  const reps = cfg('dictionary.replacements', {}) || {};
  const setWords = (w) => setPath('dictionary.words', w);
  const wordIn = h('input.tag-input#word-in', { placeholder: 'Add a word, press Enter', 'aria-label': 'Add a word', spellcheck: 'false', 'data-keep': 'word-in' });
  wordIn.addEventListener('keydown', (ev) => {
    const v = wordIn.value.trim();
    if (ev.key === 'Enter' && v && !list.includes(v)) setWords([...list, v]);
    if (ev.key === 'Backspace' && !wordIn.value && list.length) setWords(list.slice(0, -1));
  });
  const tags = h('div.tags', { 'data-bare': '' }, list.map((w) => h('span.tag', w,
    h('button.tag-x', { type: 'button', 'aria-label': `Remove ${w}`, onclick: () => setWords(list.filter((x) => x !== w)) }))), wordIn);

  const removeSvg = '<svg viewBox="0 0 16 16"><path d="M4.5 4.5l7 7M11.5 4.5l-7 7"/></svg>';
  const setReps = (n) => setPath('dictionary.replacements', n);
  const said = h('input.input#said-in', { placeholder: 'Spoken phrase', 'aria-label': 'Spoken', spellcheck: 'false', 'data-keep': 'said-in' });
  const written = h('input.input', { placeholder: 'Replacement', 'aria-label': 'Written', spellcheck: 'false' });
  const addRep = () => {
    const a = said.value.trim().toLowerCase();
    if (a && written.value.trim()) setReps({ ...reps, [a]: written.value.trim() });
  };
  written.addEventListener('keydown', (ev) => { if (ev.key === 'Enter') addRep(); });
  const pairs = h('div.pairs',
    h('div.pairs-head', h('span', 'When I say'), h('span'), h('span', 'Write'), h('span')),
    Object.entries(reps).map(([a, b]) => {
      const ia = h('input.input', { value: a, 'aria-label': 'Spoken', spellcheck: 'false' });
      const ib = h('input.input', { value: b, 'aria-label': 'Written', spellcheck: 'false' });
      const commit = () => {
        const n = { ...reps };
        delete n[a];
        if (ia.value.trim() && ib.value.trim()) n[ia.value.trim().toLowerCase()] = ib.value.trim();
        setReps(n);
      };
      ia.addEventListener('change', commit);
      ib.addEventListener('change', commit);
      return h('div.pair', ia, h('span.pair-arrow'), ib,
        h('button.btn.btn-ghost.btn-icon.btn-sm', { type: 'button', 'aria-label': `Remove ${a}`, html: removeSvg, onclick: () => { const n = { ...reps }; delete n[a]; setReps(n); } }));
    }),
    h('div.pair', said, h('span.pair-arrow'), written, h('span')));
  return section('Dictionary', 'Names and terms Ochre should always get right.',
    group({ title: 'Preferred spellings', aside: `${list.length} word${list.length === 1 ? '' : 's'}` }, tags),
    group({ title: 'Replacements', aside: 'Whole phrases, after transcription' }, h('div.card.stack', { 'data-bare': '' }, pairs)));
}

let historyQ = '';
let historyTimer = null;
const queryHistory = () => send({ op: 'history_query', q: historyQ, limit: 100 });

function dayLabel(d) {
  const today = new Date();
  const y = new Date(today);
  y.setDate(today.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return 'Today';
  if (d.toDateString() === y.toDateString()) return 'Yesterday';
  return d.toLocaleDateString(undefined, { weekday: 'long', month: 'short', day: 'numeric' });
}

function historyItems() {
  const items = store.history;
  if (items == null) return h('div.card.history', h('div.empty', 'Loading…'));
  if (!items.length) return h('div.card.history', h('div.empty', historyQ ? `Nothing matches “${historyQ}”.` : `Nothing yet. Hold ${KEY()} and say something.`));
  const card = h('div.card.history');
  let lastDay = '';
  for (const it of items) {
    const d = new Date(it.created * 1000);
    const day = dayLabel(d);
    if (day !== lastDay) { card.append(h('div.day-label', day)); lastDay = day; }
    const raw = h('div.history-raw', `Heard: ${it.raw}`);
    raw.hidden = true;
    const copy = h('button.btn.btn-ghost.btn-sm.btn-icon', { type: 'button', 'aria-label': 'Copy', html: '<svg viewBox="0 0 16 16"><rect x="5" y="5" width="8.5" height="8.5" rx="2"/><path d="M3 10.5V4a1.5 1.5 0 0 1 1.5-1.5H10"/></svg>' });
    copy.addEventListener('click', async () => { try { await navigator.clipboard.writeText(it.text); } catch { /* clipboard denied */ } });
    const [tagText, tone] = !it.inserted ? ['Saved only', 'warning'] : it.refiner ? ['Refined', 'accent']
      : it.note === 'redictation' ? ['Said again: as heard', null] : ['Raw', null];
    const tag = badge(tagText, tone);
    if (it.refiner) { tag.style.cursor = 'pointer'; tag.title = 'Show what was heard'; tag.addEventListener('click', () => { raw.hidden = !raw.hidden; }); }
    card.append(h('div.history-item',
      h('span.history-time.tnum', d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })),
      h('div', h('div.history-text', it.text), raw,
        h('div.history-meta', tag, h('span', appName(it.app)), h('span', '·'), h('span.tnum', `${words(it.text)} words`))),
      h('div.history-actions', copy, button('Re-insert', () => send({ op: 'insert_text', text: it.text }), { kind: 'ghost sm' }))));
  }
  return card;
}

function historyPage() {
  const search = h('div.field',
    fromHtml('<svg class="icon-lead" viewBox="0 0 16 16"><circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5 14 14"/></svg>'),
    h('input.input#hist-q', {
      placeholder: 'Search history', 'aria-label': 'Search history', value: historyQ, spellcheck: 'false', 'data-keep': 'hist-q',
      oninput: (ev) => { historyQ = ev.target.value; clearTimeout(historyTimer); historyTimer = setTimeout(queryHistory, 160); },
    }));
  return section('History', 'Everything you dictated, stored only on this computer. Text only, never audio.',
    h('div.hstack', { style: { marginBottom: 'var(--space-4)' } }, search),
    h('div#hist-holder', { style: { marginBottom: 'var(--space-6)' } }, historyItems()),
    group(null,
      row('Keep history', 'Saved before it’s typed, so nothing is lost if focus changes.', toggle('history.enabled', { label: 'Keep history' })),
      row('Forget after', null, selectPath('history.keep_days', [['7', '1 week'], ['30', '1 month'], ['90', '3 months'], ['365', '1 year']], { 'aria-label': 'Forget after' }))));
}

function latencyCard() {
  const r = store.result;
  if (!r || !r.timings) return h('span.row-desc', `No dictations yet. Hold ${KEY()} and talk; the timing shows up here.`);
  const t = r.timings;
  const total = Math.max(1, t.release_to_insert_ms);
  const parts = [['Transcribe tail', t.stt_tail_ms, 'lat-a'], ['Refine', t.refine_ms, 'lat-b'], ['Type', t.inject_ms, 'lat-c']];
  parts.push(['Other', Math.max(0, total - parts.reduce((s, [, v]) => s + (v || 0), 0)), 'lat-d']);
  const shown = parts.filter(([, v]) => v > 0);
  return h('div', { style: { display: 'grid', gap: 'var(--space-3)' } },
    h('div.lat-big', h('span.lat-num', fmtMs(t.release_to_insert_ms)), h('span.lat-label', 'from releasing the key to text in the box')),
    h('div.lat-bar', shown.map(([l, v, c]) => h(`i.${c}`, { style: { flexGrow: String(v) }, title: `${l}: ${fmtMs(v)}` }))),
    h('div.lat-legend', shown.map(([l, v, c]) => h('span', h(`i.${c}`), `${l} ${fmtMs(v)}`))),
    h('span.row-desc', `${fmtMs(t.audio_ms)} of speech; ${fmtMs(t.stt_total_ms)} of decoding happened while you were still talking.`));
}

function about() {
  const v = store.hello ? store.hello.version : '…';
  const os = ({ windows: 'Windows', macos: 'macOS', linux: 'Linux' })[store.platform];
  const e = sttEngine();
  return h('section.page',
    h('div.card.about-hero',
      h('img', { src: '../theme/brand/ochre-icon.svg', alt: '' }),
      h('h1.about-wordmark', h('span.wordmark', { role: 'img', 'aria-label': 'Ochre' })),
      h('p.page-lede', 'Hold a key, talk, and polished text lands where you type.'),
      h('p.mono.small.muted', `v${v} · ${os}`),
      h('div.hstack', button('GitHub ↗', () => window.open('https://github.com/jordan-gibbs/ochre'), { kind: 'ghost' }))),
    group('Last dictation', block(latencyCard())),
    group('Privacy',
      row('Audio', 'Processed locally unless you pick a cloud engine. Never stored.', e && e.kind === 'cloud' ? badge(`Sent to ${e.label}`, 'warning') : badge('On device', 'success')),
      row('Keys', `Held by ${keychainName()}, never written to disk by Ochre.`, badge('Keyring')),
      row('Telemetry', 'None. There is nothing to opt out of.', badge('Off'))),
    h('p.small.muted', { style: { textAlign: 'center' } }, 'Open source under the MIT license. Type: Funnel Display, Instrument Sans and Geist Mono (SIL OFL 1.1).'));
}

const RENDER = { general, transcription, refinement, handsfree, dictionary, history: historyPage, about };

function structureKey() {
  switch (current) {
    case 'general': return `${cfg('hotkey.key')}|${cfg('hotkey.paste_last')}|${store.platform}`;
    case 'transcription': return `${cfg('stt.engine')}|${cfg('stt.model')}|${store.engines.stt.length}|${store.platform}|${connectorId()}|${advanced}|${store.connectors.length}|${cfg('refine.provider')}`;
    case 'refinement': return `${cfg('refine.provider')}|${cfg('refine.mode')}|${cfg('refine.model')}|${JSON.stringify(cfg('refine.app_styles'))}|${store.engines.refine.length}|${connectorId()}|${advanced}|${store.connectors.length}`;
    case 'handsfree': return cfg('handsfree.phrase');
    case 'dictionary': return JSON.stringify([cfg('dictionary.words'), cfg('dictionary.replacements')]);
    case 'about': return `${store.hello && store.hello.version}|${store.result && store.result.id}|${cfg('stt.engine')}`;
    default: return '';
  }
}

// ---------------------------------------------------------------- shell

function renderNav() {
  const nav = $('.nav');
  nav.textContent = '';
  for (const s of SECTIONS) {
    if (!s) { nav.append(h('div.nav-sep')); continue; }
    const [id, label, ic] = s;
    const a = h('a.nav-item', { href: `#${id}`, html: icon(ic) }, label);
    if (id === current) a.setAttribute('aria-current', 'page');
    nav.append(a);
  }
}

function renderPage(animate) {
  if (!store.cfg) return;
  const keep = document.activeElement && document.activeElement.dataset ? document.activeElement.dataset.keep : null;
  resetBindings();
  const page = (RENDER[current] || RENDER[HOME])();
  if (!animate) page.style.animation = 'none';
  $('#main').replaceChildren(page);
  if (animate) $('#main').scrollTop = 0;
  structure = structureKey();
  if (keep) { const el = document.getElementById(keep); if (el) el.focus(); }
}

function renderStatus() {
  const s = store.state || {};
  const map = {
    loading: ['Loading models', 'busy'], idle: ['Ready', 'success'], recording: ['Dictating', 'accent'], locked: ['Dictating', 'accent'],
    handsfree: ['Listening', 'companion'], transcribing: ['Transcribing', 'busy'], refining: ['Refining', 'busy'], inserting: ['Inserting', 'busy'],
    error: ['Something went wrong', 'danger'],
  };
  const [label, tone] = map[s.state] || ['Starting…', 'busy'];
  $('.status-text').textContent = label;
  $('.engine-status .status').dataset.tone = tone;
  const r = store.result;
  const e = sttEngine();
  $('.latency').textContent = r && r.timings
    ? `Last dictation ${fmtMs(r.timings.release_to_insert_ms)}`
    : s.state === 'idle' ? (s.handsfree_armed ? `Hold ${KEY()} or say “transcribe”` : `Hold ${KEY()} to dictate`) : (e ? `${e.label} · ${e.kind}` : '');
}

function go(id) {
  if (!RENDER[id]) id = HOME;
  if (id === current && $('#main').childElementCount) return;
  current = id;
  if (location.hash !== `#${id}`) history.replaceState(null, '', `#${id}`);
  renderNav();
  if (id === 'history') queryHistory();
  renderPage(true);
}

window.addEventListener('hashchange', () => { advanced = false; go(location.hash.slice(1)); });
listen('ochre://navigate', (id) => go(id));

subscribe((ev) => {
  if (['state', 'result', 'hello', 'config', 'engines'].includes(ev.event)) renderStatus();
  if (ev.event === 'history' && current === 'history') {
    const holder = document.getElementById('hist-holder');
    if (holder) holder.replaceChildren(historyItems());
    return;
  }
  if (!['config', 'engines', 'connectors', 'download', 'test_result', '_test_pending', 'result', 'hello', 'permissions'].includes(ev.event)) return;
  if (structureKey() !== structure) renderPage(false);
  else refreshBindings();
});

await start();
// screenshots: a variant of a page (e.g. Transcription with a cloud provider)
const params = new URLSearchParams(location.search);
if (!inApp && params.get('variant') === 'cloud') setPatch({ stt: { engine: 'groq', model: '' } });
if (!inApp && params.get('connector')) pickConnector(params.get('connector')); // e.g. ?connector=google
if (!inApp && params.get('advanced')) advanced = true;
renderNav();
renderStatus();
if (current === 'history') queryHistory();
renderPage(false);
await document.fonts.ready;
const scroll = params.get('scroll');
if (scroll) { const c = $('#main'); c.style.scrollBehavior = 'auto'; c.scrollTop = scroll === 'bottom' ? c.scrollHeight : Number(scroll); }
requestAnimationFrame(() => {
  invoke('ui_ready');
  window.__ochreReady = true;
});
