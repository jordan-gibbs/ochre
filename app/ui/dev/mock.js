// Browser-only stand-in for the shell + core, used when the pages are opened outside the app
// (scripts/render-shots.mjs, or any page with `?mock`). It answers commands like the Rust demo
// driver (src-tauri/src/demo.rs) and mirrors its fixtures, so screenshots match `--demo`.
//
//   ?mock            quiet: snapshot only (config, engines, idle), commands answered
//   ?mock=demo       also loop through every HUD state
//   ?platform=macos  pretend to be another OS (key names, permissions)

const MB = 1024 * 1024;
const params = new URLSearchParams(location.search);
const PLATFORM = params.get('platform') || 'windows';

export function demoConfig() {
  return {
    hotkey: { key: 'right_alt', double_tap_ms: 350, hold_min_ms: 250, raw_modifier: 'shift', paste_last: 'down' },
    audio: { device: null, max_session_s: 600, earcons: true, warm_mic: true },
    stt: { engine: 'parakeet', model: '', language: null, device: 'auto', cloud_timeout_ms: 8000, fallback_local: true },
    refine: {
      provider: 'local', mode: 'clean', model: '', base_url: '', timeout_ms_local: 1500, timeout_ms_cloud: 3000,
      chunk_long: 'auto', chunk_min_words: 80, redictation_raw: true, redictation_window_s: 60.0,
      app_styles: { code: 'literal', discord: 'casual', iterm2: 'literal', mail: 'formal', outlook: 'formal', slack: 'casual', terminal: 'literal', windowsterminal: 'literal' },
    },
    inject: { method: 'type', paste_over_chars: 2000, join_window_s: 20.0, trailing_space: true },
    handsfree: { enabled: false, phrase: 'transcribe', model: '', threshold: 0.55, preroll_s: 1.5, idle_timeout_s: 45.0, pause_on_calls: true },
    dictionary: { words: ['Parakeet', 'Ochre', 'Kubernetes', 'Priya Raman'], replacements: { 'my email': 'jordan@example.com', 'oaker': 'Ochre' } },
    ui: { theme: 'light', show_partials: true, start_at_login: false, onboarded: true },
    history: { enabled: true, keep_days: 90 },
    debug: { keep_audio: false, log_timings: true, log_level: 'info' },
  };
}

const info = (id, label, kind, models, needs_key, note, languages) => ({ id, label, kind, models, default_model: models[0] || '', needs_key, note, languages });

export const ENGINES = {
  stt: [
    info('parakeet', 'Parakeet TDT 0.6B v3', 'local', ['parakeet-tdt-0.6b-v3-int8'], false, '670 MB · fast on any modern CPU', '25 European languages'),
    // Whisper is left out, as in a default build (it sits behind ochre-stt's `whisper` feature and
    // the shell only offers engines the build can run: core_bridge::engines)
    info('groq', 'Groq', 'cloud', ['whisper-large-v3-turbo'], true, '≈ $0.04 per hour of audio', '99 languages'),
    info('openai', 'OpenAI', 'cloud', ['gpt-transcribe', 'gpt-live-transcribe', 'gpt-4o-mini-transcribe', 'gpt-4o-transcribe'], true, '≈ $0.27 per hour of audio; live streaming ≈ $1.02', '57 languages'),
    info('google', 'Google (Gemini 3.5 Transcribe)', 'cloud', ['gemini-3.5-transcribe', 'gemini-3.5-transcribe-live'], true, '≈ $0.30 per hour of audio; removes fillers itself (preview)', '85+ languages'),
    info('soniox', 'Soniox', 'cloud', ['stt-async-v5'], true, '≈ $0.10 per hour of audio', '60 languages'),
    info('deepgram', 'Deepgram', 'cloud', ['nova-3'], true, '≈ $0.26 per hour of audio', '36 languages'),
    info('elevenlabs', 'ElevenLabs Scribe', 'cloud', ['scribe_v1'], true, '≈ $0.40 per hour of audio', '99 languages'),
    info('assemblyai', 'AssemblyAI', 'cloud', ['universal'], true, '≈ $0.15 per hour of audio', '99 languages'),
  ],
  refine: [
    info('local', 'Ochre Refine (on this computer)', 'local', ['auto', 'ochre-refine-4b', 'ochre-refine-2b', 'ochre-refine-0.8b', 'quill-4b', 'quill-2b', 'quill-0.8b'], false, 'auto: ochre-refine-4b (NVIDIA GPU, 16 GB). Ochre Refine 4B 2.7 GB / 2B 1.2 GB (GPU), 0.8B 529 MB (CPU)', 'English'),
    info('openai', 'OpenAI', 'cloud', ['gpt-5.6-luna', 'gpt-6-luna', 'gpt-4.1-nano', 'gpt-4.1-mini'], true, '≈ $0.02 per 100 dictations', 'Any'),
    info('groq', 'Groq', 'cloud', ['openai/gpt-oss-20b', 'openai/gpt-oss-120b'], true, '≈ $0.01 per 100 dictations', 'Any'),
    info('anthropic', 'Anthropic', 'cloud', ['claude-haiku-4-5'], true, '≈ $0.05 per 100 dictations', 'Any'),
    info('gemini', 'Gemini', 'cloud', ['gemini-3.5-flash-lite', 'gemini-3.5-flash'], true, '≈ $0.03 per 100 dictations', 'Any'),
    info('openrouter', 'OpenRouter', 'cloud', [], true, 'Price depends on the model', 'Any'),
    info('custom', 'Custom server (OpenAI-compatible)', 'cloud', [], false, 'Ollama, LM Studio, vLLM…', 'Any'),
  ],
};

// Mirrors crates/ochre/src/connectors.rs (the app sends it as the `connectors` event).
const stage = (id, model, mode = '') => ({ id, model, mode });
const connector = (id, label, key_provider, extra_keys, stt, refine, est_cost_per_hour, notes) => ({ id, label, key_provider, extra_keys, stt, refine, est_cost_per_hour, notes });
export const CONNECTORS = [
  connector('openai', 'OpenAI', 'openai', [], stage('openai', 'gpt-live-transcribe'), stage('openai', 'gpt-5.6-luna', 'clean'), 1.1,
    'Streams with gpt-live-transcribe and cleans up with GPT-5.6 Luna (no reasoning). For about a third of the price, pick gpt-transcribe under Advanced (batch, slower release).'),
  connector('google', 'Google', 'gemini', [], stage('google', 'gemini-3.5-transcribe'), stage('gemini', 'gemini-3.5-flash-lite', 'clean'), 0.43,
    'Gemini 3.5 Transcribe (preview) cleans up as it transcribes; Gemini 3.5 Flash-Lite adds a light pass. One Gemini API key from Google AI Studio.'),
  connector('groq', 'Groq', 'groq', [], stage('groq', 'whisper-large-v3-turbo'), stage('groq', 'openai/gpt-oss-20b', 'clean'), 0.13,
    'Cheapest: Whisper large-v3-turbo and gpt-oss-20b on Groq.'),
  connector('soniox+openai', 'Soniox + OpenAI', 'soniox', ['openai'], stage('soniox', ''), stage('openai', 'gpt-5.6-luna', 'clean'), 0.2,
    'Fastest release: Soniox streams while you talk; GPT-5.6 Luna cleans up. Needs a Soniox key and an OpenAI key.'),
  connector('local', 'Local only', '', [], stage('parakeet', ''), stage('local', '', 'clean'), 0, 'Everything runs on this computer; audio never leaves it.'),
];
const KEY_ALIASES = { google: 'gemini' };

export function demoHistory(now = Date.now() / 1000) {
  const e = (id, ago, app, text, raw, refiner) => ({ id, created: now - ago, raw, text, app, stt: 'parakeet', refiner, duration_ms: 4200, inserted: true });
  return [
    e(41, 90, 'slack', 'Sounds good, I can join the design review at two and bring the new HUD screenshots.', 'uh sounds good I can join the design review at two and um bring the new HUD screenshots', 'local'),
    e(40, 1300, 'outlook', "Hi Sam, thanks for the notes. I'll send the deck tonight.", "hi sam thanks for the notes I'll send the deck tonight", 'local'),
    e(39, 5400, 'code', 'TODO: coalesce level events for slow clients.', 'todo coalesce level events for slow clients', ''),
    e(38, 86400, 'notion', 'Parakeet is fast enough on CPU that release-to-text feels instant.', 'parakeet is fast enough on cpu that release to text feels instant', 'local'),
    e(37, 3 * 86400, 'chrome', 'Can we move the sync to Thursday at 3 PM?', 'can we move the sync to tuesday no wait thursday at three pm', 'local'),
  ];
}

export const SENTENCE = 'Thanks for the notes, I will send the deck tonight and loop in Priya on the pricing questions.';

export function resultEvent(text = SENTENCE, inserted = true, refined = true, ms = 212) {
  return {
    event: 'result', id: `demo-${ms}`, raw: text.toLowerCase(), text, inserted, refined,
    timings: { audio_ms: 5400, stt_tail_ms: 118, stt_total_ms: 690, refine_ms: refined ? ms - 200 : 0, inject_ms: 14, release_to_insert_ms: ms },
  };
}

const state = (s, extra = {}) => ({ event: 'state', state: s, trigger: 'hotkey', handsfree_armed: false, detail: '', ...extra });

/** [delay ms, event] for one pass through every state (mirrors demo.rs `showcase`). */
export function script() {
  const steps = [];
  const at = (ms, ev) => steps.push([ms, ev]);
  at(300, state('loading', { detail: 'Downloading speech model' }));
  for (let i = 0; i <= 20; i++) at(110, { event: 'download', item: 'parakeet-tdt-0.6b-v3-int8', done: Math.round(670 * MB * i / 20), total: 670 * MB });
  at(150, state('loading', { detail: 'Loading models…' }));
  at(1300, state('idle'));
  at(1200, state('recording'));
  const w = SENTENCE.split(' ');
  for (let i = 1; i <= w.length; i++) at(170, { event: 'partial', text: w.slice(0, i).join(' '), stable_chars: w.slice(0, Math.max(0, i - 3)).join(' ').length });
  at(400, state('transcribing'));
  at(800, state('refining'));
  at(1000, state('inserting'));
  at(100, resultEvent());
  at(40, state('idle'));
  at(2600, state('locked'));
  at(300, { event: 'partial', text: 'Remind me to water the', stable_chars: 15 });
  at(2300, state('transcribing'));
  at(500, state('inserting'));
  at(80, resultEvent('Remind me to water the plants on Friday.', true, false, 188));
  at(40, state('idle'));
  at(3200, state('idle', { handsfree_armed: true }));
  at(2600, state('handsfree', { handsfree_armed: true, trigger: 'wake' }));
  at(500, { event: 'partial', text: 'Ship it after the tests pass', stable_chars: 12 });
  at(2200, state('transcribing', { handsfree_armed: true, trigger: 'wake' }));
  at(480, resultEvent('Ship it after the tests pass.', true, false, 176));
  at(40, state('idle', { handsfree_armed: true }));
  at(3000, { event: 'notice', message: 'That window runs as administrator, so the text was saved to History instead.' });
  at(5500, { event: 'error', message: 'Groq rejected the API key. Check it in Settings › Transcription.', code: 'auth' });
  at(6000, state('idle'));
  return steps;
}

function deepMerge(base, patch) {
  const out = { ...base };
  for (const [k, v] of Object.entries(patch || {})) {
    const b = base ? base[k] : undefined;
    if (v && typeof v === 'object' && !Array.isArray(v) && b && typeof b === 'object' && !Array.isArray(b)) out[k] = deepMerge(b, v);
    else out[k] = v;
  }
  return out;
}

export function createMock({ loop = false } = {}) {
  let cfg = demoConfig();
  const secrets = { anthropic: false, groq: true, openai: false, soniox: false };
  const subs = new Set();
  const named = new Map();
  let st = state('idle');
  let lastResult = resultEvent();
  let permissions = [];
  if (PLATFORM === 'macos' && params.get('perm') !== 'ok') {
    permissions = [
      { name: 'accessibility', fix: 'System Settings › Privacy & Security › Accessibility: turn on Ochre, so it can type into other apps.' },
      { name: 'input_monitoring', fix: 'System Settings › Privacy & Security › Input Monitoring: turn on Ochre, so it can see the Voice key.' },
    ];
  }

  const emit = (ev) => {
    if (ev.event === 'state') st = ev;
    if (ev.event === 'result') lastResult = ev;
    for (const f of subs) f(ev, false);
  };
  const later = (ms, ev) => setTimeout(() => emit(ev), ms);
  const config = () => ({ event: 'config', config: cfg, secrets: { ...secrets } });

  function play() {
    let t = 0;
    for (const [ms, ev] of script()) { t += ms; later(t, ev); }
    if (loop) setTimeout(play, t + 1500);
  }

  return {
    emit,
    subscribe(fn) {
      subs.add(fn);
      for (const ev of [{ event: 'hello', version: '0.1.0-demo', platform: PLATFORM }, config(), { event: 'engines', ...ENGINES }, { event: 'connectors', items: CONNECTORS },
        { event: 'permissions', missing: permissions }, lastResult, st]) fn(ev, true);
      // ?dl=<MB>: a speech-model download in progress (screenshots)
      if (params.get('dl')) fn({ event: 'download', item: 'parakeet-tdt-0.6b-v3-int8', done: Number(params.get('dl')) * MB, total: 670 * MB }, true);
      if (loop && subs.size === 1) play();
    },
    on(name, fn) { named.set(name, fn); },
    fire(name, payload) { const f = named.get(name); if (f) f(payload); },
    invoke() { return Promise.resolve(); },
    setConfig(patch) { cfg = deepMerge(cfg, patch); emit(config()); },
    command(msg) {
      switch (msg.op) {
        case 'get_config': emit(config()); break;
        case 'set_config': cfg = deepMerge(cfg, msg.patch || {}); emit(config()); break;
        case 'set_handsfree': cfg = deepMerge(cfg, { handsfree: { enabled: !!msg.enabled } }); emit(config()); emit(state('idle', { handsfree_armed: !!msg.enabled })); break;
        case 'set_secret': secrets[KEY_ALIASES[msg.provider] || msg.provider] = !!msg.key; emit(config()); break;
        case 'test_provider': {
          const local = ['local', 'parakeet', 'whisper'].includes(msg.provider);
          const ok = local || !!secrets[KEY_ALIASES[msg.provider] || msg.provider];
          later(700, { event: 'test_result', stage: msg.stage, provider: msg.provider, ok, ms: local ? 184 : 412,
            message: ok ? (msg.stage === 'stt' ? 'Transcribed the test clip: “The quick brown fox jumps over the lazy dog.”' : 'Cleaned the test sentence: “Let’s meet Tuesday at 3 PM.”') : `No API key saved for ${msg.provider}.` });
          break;
        }
        case 'history_query': {
          const q = String(msg.q || '').toLowerCase();
          emit({ event: 'history', items: demoHistory().filter((h) => !q || h.text.toLowerCase().includes(q)).slice(0, msg.limit || 50) });
          break;
        }
        case 'download_model': {
          const total = /whisper|large/.test(msg.name) ? 1600 * MB : 505 * MB;
          for (let i = 1; i <= 25; i++) later(i * 140, { event: 'download', item: msg.name, done: Math.round(total * i / 25), total });
          break;
        }
        case 'start': case 'toggle': emit(state('recording', { trigger: 'ui' })); break;
        case 'stop': case 'cancel': emit(state('idle', { trigger: 'ui', handsfree_armed: !!cfg.handsfree.enabled })); break;
        default: break;
      }
    },
  };
}

let shared = null;
export function sharedMock() {
  if (!shared) {
    shared = createMock({ loop: params.get('mock') === 'demo' });
    window.__ochreMock = shared;
  }
  return shared;
}
