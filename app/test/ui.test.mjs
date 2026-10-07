// Pure UI logic: HUD model + view mapping, key capture, formatting.   node --test app/test/
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as M from '../ui/hud/model.js';
import { createCapture, label, pasteLabel, presets, hint } from '../ui/shared/keys.js';
import { bytesPair, patchFor, deepMerge, ms, autoPick, localRefineLabel, resolveTheme, readyValue, readyPatch, readyHint } from '../ui/shared/format.js';
import { createRequire } from 'node:module';
import { redrawFor, tryInitial, tryReduce, tryStatus } from '../ui/onboarding/try.js';

const st = (state, extra = {}) => ({ event: 'state', state, trigger: 'hotkey', handsfree_armed: false, detail: '', ...extra });
const run = (events, t = 0) => events.reduce((S, e) => M.reduce(S, e, t), M.initial());
const hud = (S, ctx = {}, t = 0) => M.hudView(M.view(S, ctx, t));

test('every core state maps to a design data-state', () => {
  const cases = {
    recording: 'recording', locked: 'locked', handsfree: 'handsfree', transcribing: 'transcribing',
    refining: 'refining', inserting: 'inserting', loading: 'loading',
  };
  for (const [core, want] of Object.entries(cases)) assert.equal(hud(run([st(core)])).state, want, core);
  assert.equal(hud(run([st('idle')])).shown, false);
});

test('locked hint carries the configured key as a kbd part', () => {
  const h = hud(run([st('locked')]), { keyLabel: 'F13' });
  assert.deepEqual(h.kbd, { before: 'tap ', kbd: 'F13', after: ' to finish' });
  assert.equal(h.label, 'Dictating');
});

test('loading: determinate with downloads, indeterminate without', () => {
  const MB = 1024 * 1024;
  const S = run([st('loading', { detail: 'Downloading speech model' }), { event: 'download', item: 'parakeet', done: 312 * MB, total: 670 * MB }]);
  const h = hud(S);
  assert.equal(h.indeterminate, false);
  assert.ok(Math.abs(h.progress - 312 / 670) < 1e-9);
  assert.equal(h.meta, '312 / 670 MB');
  assert.equal(hud(run([st('loading', { detail: 'Loading models…' })])).indeterminate, true);
});

test('result -> inserted toast with words and timing, then expiry', () => {
  const S = run([st('recording'), st('inserting'),
    { event: 'result', id: '1', raw: 'a b', text: 'Hello there world.', inserted: true, refined: true, timings: { release_to_insert_ms: 212 } }, st('idle')]);
  const h = hud(S);
  assert.equal(h.state, 'inserted');
  assert.equal(h.hint, '3 words · refined');
  assert.equal(h.meta, '212 ms');
  assert.equal(M.view(S, {}, M.TIMING.toastMs + 10).mode, 'hidden');
});

test('not inserted -> notice "Saved to History"; errors win over toasts', () => {
  let S = run([st('recording'), { event: 'result', id: '1', raw: '', text: 'x y', inserted: false, refined: false, timings: {} }, st('idle')]);
  assert.equal(hud(S).state, 'notice');
  assert.equal(hud(S).label, 'Saved to History');
  S = M.reduce(S, { event: 'error', message: 'Mic unavailable', code: 'audio' }, 0);
  assert.equal(hud(S).state, 'error');
  assert.equal(hud(S).close, true);
});

test('partials: live line splits at a word boundary, lasts through transcribing, hides at refining', () => {
  let S = run([st('recording'), { event: 'partial', text: 'Ship it after the tests', stable_chars: 11 }]);
  const l = hud(S).live;
  assert.equal(l.text, 'Ship it after the tests');
  assert.equal(l.text.slice(0, l.stable), 'Ship it '); // snapped back to the start of "after"
  assert.equal(hud(S).state, 'recording');
  S = M.reduce(S, st('transcribing'), 0);
  assert.equal(hud(S).live.text, 'Ship it after the tests');
  S = M.reduce(S, st('refining'), 0);
  assert.equal(hud(S).live, null);
});

test('partials: off in settings shows none; a new session starts empty', () => {
  const S = run([st('recording'), { event: 'partial', text: 'hello there', stable_chars: 5 }]);
  assert.equal(hud(S, { showPartials: false }).live, null);
  const S2 = run([st('recording'), { event: 'partial', text: 'old words', stable_chars: 9 }, st('idle'), st('recording')]);
  assert.equal(hud(S2).live, null);
  // whitespace-only partials never open the line
  assert.equal(hud(run([st('recording'), { event: 'partial', text: '  ', stable_chars: 0 }])).live, null);
});

test('commonPrefix keeps surrogate pairs whole', () => {
  assert.equal(M.commonPrefix('ship it', 'ship it after'), 7);
  assert.equal(M.commonPrefix('ship it', 'shop it'), 2);
  assert.equal(M.commonPrefix('', 'x'), 0);
  assert.equal(M.commonPrefix('a \u{1F600}', 'a \u{1F601}'), 2);
});

test('armed idle shows nothing on screen; × cancels only live sessions', () => {
  const S = run([st('idle', { handsfree_armed: true })]);
  const h = hud(S, { phrase: 'computer' });
  assert.equal(h.shown, false);
  assert.equal(h.pip, '');
  assert.equal(M.closeAction(run([st('recording')])).cancel, true);
  assert.equal(M.closeAction(run([st('idle')])).cancel, false);
});

test('a local cancel echo shows "Cancelled"', () => {
  let S = run([st('recording')]);
  S = M.reduce(S, { event: '_cancel' }, 100);
  S = M.reduce(S, st('idle'), 200);
  assert.equal(hud(S, {}, 200).label, 'Cancelled');
});

test('key capture rules', () => {
  const cap = createCapture();
  cap.keydown('AltRight');
  assert.deepEqual(cap.keyup('AltRight'), { name: 'right_alt' });
  cap.keydown('ControlLeft'); cap.keydown('AltRight'); cap.keyup('AltRight');
  assert.deepEqual(cap.keyup('ControlLeft'), { name: 'right_alt' }); // AltGr
  cap.keydown('KeyA');
  assert.ok(cap.keyup('KeyA').error);
  cap.keydown('F13');
  assert.deepEqual(cap.keyup('F13'), { name: 'f13' });
  cap.keydown('ControlLeft'); cap.keydown('MetaLeft'); cap.keyup('MetaLeft');
  assert.ok(cap.keyup('ControlLeft').error, 'modifier-only chords are refused');
  assert.equal(label('right_alt', 'macos'), 'Right Option');
  assert.equal(label('ctrl+meta', 'windows'), 'Ctrl + Win');
});

test('voice key suggestions fit the OS', () => {
  const mac = presets('macos');
  assert.ok(!mac.includes('caps_lock'), 'macOS cannot hold Caps Lock');
  assert.ok(!mac.some((k) => ['menu', 'insert', 'scroll_lock', 'pause', 'f20'].includes(k)));
  assert.ok(mac.slice(0, 6).includes('fn'), 'fn (Globe) is among the visible Mac choices');
  assert.equal(label('fn', 'macos'), 'fn (Globe)');
  assert.equal(label('right_meta', 'macos'), 'Right Command');
  const win = presets('windows');
  assert.deepEqual(win.slice(0, 3), ['right_alt', 'right_ctrl', 'caps_lock']);
  assert.ok(![...mac, ...win].some((k) => k.includes('+')), 'no modifier-only chords');
  assert.ok(win.includes('caps_lock') && win.includes('f20'));
  assert.match(hint('macos'), /Right Option/);
  assert.match(hint('windows'), /Right Alt/);
  const cap = createCapture('macos');
  cap.keydown('CapsLock');
  assert.match(cap.keyup('CapsLock').error, /macOS can't use Caps Lock.*Right Option/);
  cap.keydown('KeyA');
  assert.match(cap.keyup('KeyA').error, /Right Option/);
  const wcap = createCapture('windows');
  wcap.keydown('CapsLock');
  assert.deepEqual(wcap.keyup('CapsLock'), { name: 'caps_lock' });
});

test('paste-last shortcut labels', () => {
  assert.equal(pasteLabel('down', 'right_ctrl', 'windows'), 'Right Ctrl + ↓');
  assert.equal(pasteLabel('ctrl+alt+v', 'right_alt', 'windows'), 'Ctrl + Alt + V');
  assert.equal(pasteLabel('down', 'right_alt', 'macos'), 'Right Option + ↓');
  assert.equal(pasteLabel('', 'right_ctrl', 'windows'), '');
  assert.equal(pasteLabel(undefined, 'right_ctrl', 'windows'), '');
});

test('format helpers', () => {
  assert.equal(bytesPair(1.5 * 1024 ** 3, 3 * 1024 ** 3), '1.50 / 3.00 GB');
  assert.deepEqual(patchFor('refine.mode', 'polish'), { refine: { mode: 'polish' } });
  assert.deepEqual(deepMerge({ a: { b: 1, c: 2 } }, { a: { b: 3 } }), { a: { b: 3, c: 2 } });
  assert.equal(ms(212), '212 ms');
  assert.equal(ms(1500), '1.5 s');
});

test('local refine labels: auto shows its pick for this PC', () => {
  const note = 'auto: ochre-refine-4b (NVIDIA GPU, 16 GB). Ochre Refine 4B 2.7 GB';
  assert.deepEqual(autoPick(note), { id: 'ochre-refine-4b', hardware: 'NVIDIA GPU, 16 GB' });
  assert.equal(localRefineLabel('auto', note), 'Auto (Ochre Refine 4B on this PC)');
  assert.equal(localRefineLabel('auto', 'auto: quill-0.8b (no usable GPU). x'), 'Auto (Quill 0.8B on this PC)');
  assert.equal(localRefineLabel('auto', ''), 'Auto');
  assert.equal(localRefineLabel('ochre-refine-2b', note), 'Ochre Refine 2B');
  assert.equal(localRefineLabel('ochre-refine-0.8b', note), 'Ochre Refine 0.8B (CPU)');
  assert.equal(localRefineLabel('my.gguf', note), 'my.gguf');
});

test('theme: light by default; system follows the OS', () => {
  assert.equal(resolveTheme(undefined, true), 'light');
  assert.equal(resolveTheme('light', true), 'light');
  assert.equal(resolveTheme('bogus', true), 'light');
  assert.equal(resolveTheme('dark', false), 'dark');
  assert.equal(resolveTheme('system', true), 'dark');
  assert.equal(resolveTheme('system', false), 'light');
});

// ---------------------------------------------------------------- onboarding: try it

test('onboarding try-it: a result never rebuilds the step (the box keeps the dictated text)', () => {
  const result = { event: 'result', id: '7', text: 'Hello there.', inserted: true, timings: { release_to_insert_ms: 212 } };
  // regression: the step used to re-render on `result`, replacing the textarea Ochre had just typed into
  assert.equal(redrawFor('try', result), 'patch');
  for (const ev of [st('recording'), st('inserting'), st('idle'), { event: 'level', rms: 0.4 }, { event: 'config', config: {} }]) {
    assert.notEqual(redrawFor('try', ev), 'page', ev.event);
  }
  assert.equal(redrawFor('model', st('idle')), 'page');
  assert.equal(redrawFor('model', { event: 'download' }, { downloadDone: false }), null);
  assert.equal(redrawFor('permissions', { event: 'permissions' }), 'page');
});

test('onboarding try-it: waits with a tip, then shows "That’s it" with the timing', () => {
  const old = { event: 'result', id: '1', text: 'old', inserted: true, timings: {} };
  let s = tryInitial(old);
  assert.equal(tryStatus(s, 'Right Ctrl').tone, 'tip');
  assert.match(tryStatus(s, 'Right Ctrl').text, /Right Ctrl/);
  s = tryReduce(s, old);                       // the replayed result from before onboarding
  assert.equal(s.result, null);
  s = tryReduce(s, { ...old });                // same id, replayed again
  assert.equal(s.result, null);
  s = tryReduce(s, st('idle'));
  assert.equal(s.result, null);
  const fresh = { event: 'result', id: '2', text: 'Hello there.', inserted: true, timings: { release_to_insert_ms: 212 } };
  s = tryReduce(s, fresh);
  assert.deepEqual(tryStatus(s, 'Right Ctrl'), { tone: 'success', text: 'That’s it.', ms: 212 });
  s = tryReduce(s, { ...fresh, id: '3', inserted: false });
  assert.equal(tryStatus(s, 'Right Ctrl').tone, 'notice');
});

// ---------------------------------------------------------------- the speaking o (hud-kit.js)

function loadKit() {
  globalThis.self = globalThis;
  const require = createRequire(import.meta.url);
  return require('../ui/theme/hud-kit.js');
}

test('fakeVoice: deterministic, in range, speech-shaped (loud syllables and quiet gaps)', () => {
  const Kit = loadKit();
  const a = Kit.fakeVoice(5), b = Kit.fakeVoice(5);
  const xs = [];
  for (let t = 0; t < 6000; t += 33) { const v = a(t); assert.equal(v, b(t)); xs.push(v); }
  assert.ok(xs.every((v) => v >= 0 && v <= 1));
  assert.ok(xs.filter((v) => v > 0.6).length > 15, 'has loud syllables');
  assert.ok(xs.filter((v) => v < 0.3).length > 40, 'has quiet gaps');
});

// A fake browser for oMeter: a manual clock, frames driven by hand, the o's transform readable.
function oRig({ reduce = false } = {}) {
  const Kit = loadKit();
  const env = { now: 0, frame: null, reduce };
  const saved = { now: performance.now, raf: globalThis.requestAnimationFrame, caf: globalThis.cancelAnimationFrame, gcs: globalThis.getComputedStyle, mm: globalThis.matchMedia };
  performance.now = () => env.now;
  globalThis.requestAnimationFrame = (cb) => { env.frame = cb; return 1; };
  globalThis.cancelAnimationFrame = () => { env.frame = null; };
  globalThis.getComputedStyle = () => ({ getPropertyValue: () => '' });
  globalThis.matchMedia = () => ({ matches: env.reduce });
  const glyph = { style: { transform: '' } };
  let animated = 0;
  const el = { querySelector: () => glyph, querySelectorAll: () => [], animate() { animated++; } };
  const m = Kit.oMeter(el);
  const scale = () => { const x = /scale\((.+)\)/.exec(glyph.style.transform); return x ? parseFloat(x[1]) : 1; };
  // advance `ms` of wall time: a level event every 33 ms (if `level` given), a frame every 16 ms
  const run = (ms, level) => {
    const out = [];
    const end = env.now + ms;
    let nextPush = env.now;
    while (env.now < end) {
      if (level != null && env.now >= nextPush) { m.push(typeof level === 'function' ? level(env.now) : level); nextPush += 33; }
      env.now += 16;
      const f = env.frame; env.frame = null;
      if (f) f(env.now);
      out.push(scale());
    }
    return out;
  };
  const restore = () => {
    performance.now = saved.now; globalThis.requestAnimationFrame = saved.raf; globalThis.cancelAnimationFrame = saved.caf;
    globalThis.getComputedStyle = saved.gcs; globalThis.matchMedia = saved.mm;
  };
  return { m, env, glyph, scale, run, restore, animated: () => animated };
}

test('oMeter: silence (room noise) leaves the o still at exactly 1.0, with no frames', () => {
  const r = oRig();
  try {
    const xs = r.run(3000, 0.22);
    assert.ok(xs.every((x) => x === 1), 'no breathing, no pulse');
    assert.equal(r.glyph.style.transform, '');
    assert.equal(r.env.frame, null, 'no frame loop while silent');
  } finally { r.restore(); }
});

test('oMeter: scales up with level (1.0 -> ~1.3), louder is bigger, never past 1.35', () => {
  const r = oRig();
  try {
    r.run(1500, 0.2);                       // settle the noise floor on the room
    const soft = r.run(800, 0.5).at(-1);
    const loud = r.run(800, 0.95).at(-1);
    assert.ok(soft > 1.05, `soft ${soft}`);
    assert.ok(loud > soft, `loud ${loud} > soft ${soft}`);
    assert.ok(loud >= 1.22 && loud <= 1.35, `loud ${loud}`);
    // a speech-like voice never pushes it past the cap and never spawns animations (no rings)
    const voice = loadKit().fakeVoice(9);
    const t0 = r.env.now;
    const xs = r.run(4000, (t) => voice(t - t0));
    assert.ok(Math.max(...xs) <= 1.35);
    assert.equal(r.animated(), 0);
  } finally { r.restore(); }
});

test('oMeter: eased attack (~80 ms) and slower release (~300 ms), no jitter between frames', () => {
  const r = oRig();
  try {
    r.run(1500, 0.2);
    const up = r.run(600, 0.95);
    const top = up.at(-1);
    const t63 = up.findIndex((x) => x - 1 >= 0.63 * (top - 1)) * 16;
    assert.ok(t63 >= 48 && t63 <= 260, `attack ${t63} ms`);
    // the first frames ease in rather than jump
    assert.ok(up[0] - 1 < 0.25 * (top - 1), `first frame ${up[0]}`);
    const down = r.run(1500, 0.2);
    const d63 = down.findIndex((x) => top - x >= 0.63 * (top - 1)) * 16;
    assert.ok(d63 > t63 && d63 >= 200 && d63 <= 700, `release ${d63} ms`);
    assert.equal(down.at(-1), 1, 'back at rest');
    // step size per 16 ms frame stays small: smooth, not a flicker
    const steps = up.concat(down).slice(1).map((x, i) => Math.abs(x - up.concat(down)[i]));
    assert.ok(Math.max(...steps) < 0.06, `max step ${Math.max(...steps)}`);
  } finally { r.restore(); }
});

test('oMeter: reduced motion still follows the voice, with less growth; stop() resets', () => {
  const r = oRig({ reduce: true });
  try {
    r.run(1500, 0.2);
    const loud = r.run(800, 0.95).at(-1);
    assert.ok(loud > 1.05 && loud < 1.16, `reduced ${loud}`);
    r.m.stop();
    assert.equal(r.glyph.style.transform, '');
    assert.equal(r.m.scale, 1);
  } finally { r.restore(); }
});

test('macOS "keep the microphone ready" maps to warm_mic + warm_idle_release_s', () => {
  assert.equal(readyValue(true, 0), 'always');
  assert.equal(readyValue(true, 300), '300');
  assert.equal(readyValue(false, 300), 'off');
  assert.deepEqual(readyPatch('always'), { audio: { warm_mic: true, warm_idle_release_s: 0 } });
  assert.deepEqual(readyPatch('900'), { audio: { warm_mic: true, warm_idle_release_s: 900 } });
  assert.deepEqual(readyPatch('off'), { audio: { warm_mic: false } });
  assert.match(readyHint('300', false), /after 5 minutes without a dictation/);
  assert.match(readyHint('300', true), /Hands-free is on/);
  assert.match(readyHint('always', false), /orange mic dot the whole time/);
});
