// Small, pure formatting helpers shared by the HUD, settings and onboarding.

const MB = 1024 ** 2;
const GB = 1024 ** 3;

/** "312 / 670 MB": both numbers in the unit of the total. */
export function bytesPair(done, total) {
  if (!total) return done ? `${Math.round(done / MB)} MB` : '';
  if (total >= GB) return `${(done / GB).toFixed(2)} / ${(total / GB).toFixed(2)} GB`;
  const dp = total < 10 * MB ? 1 : 0;
  return `${(done / MB).toFixed(dp)} / ${(total / MB).toFixed(dp)} MB`;
}

export function ms(n) {
  if (n == null || !Number.isFinite(n)) return '';
  return n < 1000 ? `${Math.round(n)} ms` : `${(n / 1000).toFixed(n < 10000 ? 1 : 0)} s`;
}

export function words(text) {
  const t = String(text || '').trim();
  return t ? t.split(/\s+/).length : 0;
}

/** Download item ids -> words people use. */
export function prettyItem(item) {
  const s = String(item || '').toLowerCase();
  if (s.includes('parakeet') || s.includes('nemo')) return 'speech model';
  if (s.includes('whisper')) return 'Whisper';
  if (s.includes('quill') || s.includes('ochre-refine')) return 'refinement model';
  if (s.includes('llama')) return 'llama.cpp';
  if (s.includes('wake') || s.includes('melspec') || s.includes('embedding')) return 'wake word model';
  return String(item || 'model').replace(/\.(onnx|gguf|bin|zip|tar\.gz|json|txt)$/i, '');
}

export function ago(unixSeconds, now = Date.now() / 1000) {
  const d = Math.max(0, now - unixSeconds);
  if (d < 45) return 'just now';
  if (d < 3600) return `${Math.round(d / 60)} min ago`;
  if (d < 86400) return `${Math.round(d / 3600)} h ago`;
  if (d < 2 * 86400) return 'yesterday';
  if (d < 7 * 86400) return `${Math.round(d / 86400)} days ago`;
  return new Date(unixSeconds * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
}

/** "slack" -> "Slack", "windowsterminal" -> "Windows Terminal" for the few we know. */
export function appName(id) {
  const known = {
    slack: 'Slack', discord: 'Discord', outlook: 'Outlook', mail: 'Mail', code: 'VS Code', chrome: 'Chrome',
    windowsterminal: 'Windows Terminal', terminal: 'Terminal', iterm2: 'iTerm2', notion: 'Notion', firefox: 'Firefox',
  };
  const s = String(id || '');
  return known[s.toLowerCase()] || (s ? s[0].toUpperCase() + s.slice(1) : 'Unknown app');
}

/** Read a dotted path from an object. */
export function get(obj, path, dflt) {
  let v = obj;
  for (const k of path.split('.')) {
    if (v == null || typeof v !== 'object') return dflt;
    v = v[k];
  }
  return v === undefined ? dflt : v;
}

/** {a: {b: v}} for "a.b", v: the shape `set_config{patch}` expects. */
export function patchFor(path, value) {
  const keys = path.split('.');
  const out = {};
  let cur = out;
  keys.forEach((k, i) => {
    if (i === keys.length - 1) cur[k] = value;
    else cur = cur[k] = {};
  });
  return out;
}

export function deepMerge(base, patch) {
  const out = Array.isArray(base) ? [...base] : { ...base };
  for (const [k, v] of Object.entries(patch || {})) {
    const b = base ? base[k] : undefined;
    if (v && typeof v === 'object' && !Array.isArray(v) && b && typeof b === 'object' && !Array.isArray(b)) out[k] = deepMerge(b, v);
    else out[k] = v;
  }
  return out;
}

export const LANGUAGES = [
  ['', 'Detect automatically'], ['en', 'English'], ['de', 'German'], ['fr', 'French'], ['es', 'Spanish'],
  ['it', 'Italian'], ['pt', 'Portuguese'], ['nl', 'Dutch'], ['pl', 'Polish'], ['sv', 'Swedish'], ['da', 'Danish'],
  ['fi', 'Finnish'], ['cs', 'Czech'], ['uk', 'Ukrainian'], ['ru', 'Russian'], ['ja', 'Japanese'], ['zh', 'Chinese'],
  ['ko', 'Korean'], ['hi', 'Hindi'],
];

/** Display names for local refinement models (ids stay the config values). */
export const LOCAL_REFINE_LABELS = {
  'ochre-refine-4b': 'Ochre Refine 4B',
  'ochre-refine-2b': 'Ochre Refine 2B',
  'ochre-refine-0.8b': 'Ochre Refine 0.8B (CPU)',
  'quill-4b': 'Quill 4B',
  'quill-2b': 'Quill 2B',
  'quill-0.8b': 'Quill 0.8B',
};

/** What "auto" resolves to on this PC, from the local engine's note
 * (`auto: <model id> (<hardware>). …`, ochre-refine local::info). */
export function autoPick(note) {
  const m = /^auto: (\S+) \(([^)]*)\)/.exec(note || '');
  return m ? { id: m[1], hardware: m[2] } : null;
}

/** "Auto (Ochre Refine 4B on this PC)", or a model's display name. */
export function localRefineLabel(model, note) {
  if (model === 'auto') {
    const p = autoPick(note);
    return p ? `Auto (${LOCAL_REFINE_LABELS[p.id] || p.id} on this PC)` : 'Auto';
  }
  return LOCAL_REFINE_LABELS[model] || model;
}

/** config `ui.theme` -> "light" | "dark". Light is the default (missing or unknown values);
 *  "system" follows the OS (`osDark`). */
export function resolveTheme(pref, osDark) {
  const p = pref === 'dark' || pref === 'system' ? pref : 'light';
  return p === 'dark' || (p === 'system' && osDark) ? 'dark' : 'light';
}

// "Keep the microphone ready" on macOS: audio.warm_mic + audio.warm_idle_release_s as one choice.
export const READY = [['always', 'Always'], ['60', '1 minute after a dictation'], ['300', '5 minutes after a dictation'],
  ['900', '15 minutes after a dictation'], ['3600', '1 hour after a dictation'], ['off', 'Only while dictating']];

export function readyValue(warm, releaseS) {
  if (!warm) return 'off';
  return releaseS > 0 ? String(releaseS) : 'always';
}

export function readyPatch(v) {
  if (v === 'off') return { audio: { warm_mic: false } };
  return { audio: { warm_mic: true, warm_idle_release_s: v === 'always' ? 0 : Number(v) } };
}

export function readyHint(v, handsfree) {
  if (handsfree) return 'Hands-free is on, so the microphone stays on to hear “transcribe”; the orange mic dot is expected.';
  if (v === 'always') return 'Recording is instant and never clips your first word. macOS shows the orange mic dot the whole time Ochre runs.';
  if (v === 'off') return 'The microphone opens when you press the Voice key (about 0.1 s) and closes after. No background filtering in this mode.';
  const label = (READY.find(([k]) => k === v) || [, ''])[1].replace(' after a dictation', '');
  return `The microphone turns off after ${label} without a dictation, and the orange mic dot with it. The next dictation starts it again in about 0.1 s.`;
}
