// Client-side mirror of the core's state for the settings and onboarding windows. Holds only what
// the core sent (config, secret *presence*, engines, ...); API keys are never stored here.

import { connect, send, applyTheme, platform } from './bridge.js';
import { deepMerge, patchFor, get } from './format.js';

export const store = {
  cfg: null,
  secrets: {},
  engines: { stt: [], refine: [] },
  connectors: [], // cloud connectors: one key, both stages (ochre::connectors)
  hello: null,
  platform: 'windows',
  state: { state: 'loading', detail: '', handsfree_armed: false },
  result: null,
  history: null,
  permissions: [],
  downloads: {}, // item -> {done, total}
  tests: {},     // `${stage}:${provider}` -> {pending} | test_result
  keyringError: null, // why the last set_secret failed (no keyring on Linux, ...)
  level: 0,
};

const subs = new Set();
export function subscribe(fn) {
  subs.add(fn);
  return () => subs.delete(fn);
}

function onEvent(ev) {
  switch (ev.event) {
    case 'hello': store.hello = ev; store.platform = platform(ev); break;
    case 'config': store.cfg = ev.config; store.secrets = ev.secrets || {}; applyTheme(get(ev.config, 'ui.theme', 'light')); break;
    case 'engines': store.engines = { stt: ev.stt || [], refine: ev.refine || [] }; break;
    case 'connectors': store.connectors = ev.items || []; break;
    case 'state': store.state = ev; break;
    case 'result': store.result = ev; break;
    case 'history': store.history = ev.items || []; break;
    case 'permissions': store.permissions = ev.missing || []; break;
    case 'download': store.downloads[ev.item] = { done: ev.done, total: ev.total }; break;
    case 'test_result': store.tests[`${ev.stage}:${ev.provider}`] = ev; break;
    case 'error': if (ev.code === 'keyring') store.keyringError = ev.message; break;
    case 'level': store.level = ev.rms; return; // too chatty to fan out
    default: break;
  }
  for (const f of subs) f(ev);
}

export function start() {
  return connect(onEvent);
}

/** Patch one setting (dotted path) through `set_config`; the core answers with a fresh config. */
export function setPath(path, value) {
  store.cfg = deepMerge(store.cfg || {}, patchFor(path, value)); // optimistic
  return send({ op: 'set_config', patch: patchFor(path, value) });
}

export function setPatch(patch) {
  store.cfg = deepMerge(store.cfg || {}, patch);
  return send({ op: 'set_config', patch });
}

export function cfg(path, dflt) {
  return get(store.cfg, path, dflt);
}

export function test(stage, provider) {
  store.tests[`${stage}:${provider}`] = { pending: true };
  for (const f of subs) f({ event: '_test_pending' });
  send({ op: 'test_provider', stage, provider });
}

/** Keys go straight to the core (OS keyring) and are not kept anywhere in the page. */
export function saveSecret(provider, key) {
  store.keyringError = null;
  return send({ op: 'set_secret', provider, key: key || null });
}
