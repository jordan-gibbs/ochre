// Drives the real HUD (in two iframes, light and dark) into one state for a screenshot.
//   dev/stage.html?shot=recording     (see SHOTS)
// Sets window.__ochreReady once both HUDs have settled.

import { resultEvent, SENTENCE } from './mock.js';

const MB = 1024 * 1024;
const st = (state, extra = {}) => ({ event: 'state', state, trigger: 'hotkey', handsfree_armed: false, detail: '', ...extra });
const partial = (text, stable) => ({ event: 'partial', text, stable_chars: stable });
const WORDS = 'Thanks for the notes, I will send the deck tonight';

export const SHOTS = {
  'loading-download': [st('loading', { detail: 'Downloading speech model' }),
    { event: 'download', item: 'parakeet-tdt-0.6b-v3-int8', done: 312 * MB, total: 670 * MB }],
  loading: [st('loading', { detail: 'Loading models…' })],
  recording: [st('recording'), partial(WORDS, 'Thanks for the notes, I will send the'.length)],
  locked: [st('locked'), partial('Remind me to water the plants', 'Remind me to water the'.length)],
  handsfree: [st('handsfree', { handsfree_armed: true, trigger: 'wake' }), partial('Ship it after the tests pass', 12)],
  transcribing: [st('recording'), partial(SENTENCE, SENTENCE.length - 22), st('transcribing')],
  refining: [st('recording'), partial(SENTENCE, SENTENCE.length), st('transcribing'), st('refining')],
  inserted: [st('recording'), st('transcribing'), st('inserting'), resultEvent(), st('idle')],
  'not-inserted': [st('recording'), st('transcribing'), resultEvent('Remind me to water the plants on Friday.', false, false, 188), st('idle')],
  error: [{ event: 'error', message: 'Groq rejected the API key. Check it in Settings › Transcription.', code: 'auth' }, st('error', { detail: 'Groq rejected the API key.' })],
  notice: [{ event: 'notice', message: 'That window runs as administrator, so the text was saved to History instead.' }],
  pip: [st('idle', { handsfree_armed: true })],
};

const LIVE = new Set(['recording', 'locked', 'handsfree']);
const params = new URLSearchParams(location.search);
const shot = params.get('shot') || 'recording';
const frames = [...document.querySelectorAll('iframe.hud')];
const themes = ['light', 'dark'];

function loaded(frame, theme) {
  return new Promise((resolve) => {
    frame.addEventListener('load', () => {
      const poll = () => (frame.contentWindow.__ochreHud ? resolve(frame.contentWindow) : setTimeout(poll, 20));
      poll();
    }, { once: true });
    frame.src = `../hud/index.html?mock&theme=${theme}`;
  });
}

const wins = await Promise.all(frames.map((f, i) => loaded(f, themes[i])));
await Promise.all(wins.map((w) => w.document.fonts.ready));
const events = SHOTS[shot] || [];
for (const w of wins) for (const ev of events) w.__ochreHud.emit(ev);
const live = events.some((e) => e.event === 'state' && LIVE.has(e.state));
if (live) {
  // the same speech-like level as the demo, so the o grows with a "voice" in the shot
  const voice = wins[0].HudKit.fakeVoice(7), t0 = performance.now();
  setInterval(() => {
    const rms = voice(performance.now() - t0);
    for (const w of wins) w.__ochreHud.level(rms);
  }, 33);
}
setTimeout(() => { window.__ochreReady = true; }, 900);
