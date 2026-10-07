// Renders docs/ui/*.png: every HUD state (light HUD over a light app, dark HUD over a dark app) and
// every settings / onboarding page in light and dark.
//
// Headless Microsoft Edge / Chrome driven over the DevTools protocol (Node 22+ built-in WebSocket),
// with the pages served from app/ui by a throwaway local server. Edge is the WebView2 engine, so the
// pixels match the Windows app. Nothing is shown on screen, no focus is taken and the desktop is never
// captured.
//
//   node app/scripts/render-shots.mjs                 # all shots
//   node app/scripts/render-shots.mjs --only=hud-recording,settings-general-light
//   node app/scripts/render-shots.mjs --browser="C:/path/to/chrome.exe"

import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, writeFileSync, existsSync, mkdirSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const UI = resolve(HERE, '..', 'ui');
const OUT = resolve(HERE, '..', '..', 'docs', 'ui');
const arg = (k, d) => {
  const a = process.argv.find((x) => x.startsWith(`--${k}=`));
  return a ? a.slice(k.length + 3) : d;
};
const ONLY = arg('only', '') ? new Set(arg('only', '').split(',')) : null;

const HUD_SHOTS = ['loading-download', 'loading', 'recording', 'locked', 'handsfree', 'transcribing', 'refining', 'inserted', 'not-inserted', 'error', 'notice', 'pip'];
// name -> [section, extra query]
const SECTIONS = {
  general: ['general', ''], 'general-app': ['general', '&scroll=bottom'],
  transcription: ['transcription', ''], 'transcription-cloud': ['transcription', '&variant=cloud&scroll=bottom'],
  refinement: ['refinement', ''], 'refinement-apps': ['refinement', '&scroll=bottom'],
  handsfree: ['handsfree', ''], dictionary: ['dictionary', ''], history: ['history', ''], about: ['about', ''],
};
const ONBOARDING = ['welcome', 'permissions', 'key', 'model', 'try'];

const shots = [];
for (const s of HUD_SHOTS) shots.push({ name: `hud-${s}`, url: `/dev/stage.html?shot=${s}`, w: 1520, h: 300, dpr: 2 });
for (const theme of ['light', 'dark']) {
  for (const [s, [sec, extra]] of Object.entries(SECTIONS)) {
    shots.push({ name: `settings-${s}-${theme}`, url: `/settings/index.html?mock&theme=${theme}${extra}#${sec}`, w: 980, h: 700, dpr: 2 });
  }
  for (const s of ONBOARDING) {
    const mac = s === 'permissions' ? '&platform=macos' : s === 'model' ? '&dl=312' : '';
    shots.push({ name: `onboarding-${s}-${theme}`, url: `/onboarding/index.html?mock&theme=${theme}&step=${s}${mac}`, w: 760, h: 640, dpr: 2 });
  }
}

// ---------------------------------------------------------------- static server

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.woff2': 'font/woff2', '.png': 'image/png', '.svg': 'image/svg+xml', '.json': 'application/json' };
const server = createServer((req, res) => {
  const u = new URL(req.url, 'http://localhost');
  const p = normalize(join(UI, decodeURIComponent(u.pathname)));
  if (!p.startsWith(UI) || !existsSync(p)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': MIME[extname(p)] || 'application/octet-stream', 'cache-control': 'no-store' });
  res.end(readFileSync(p));
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const ORIGIN = `http://127.0.0.1:${server.address().port}`;

// ---------------------------------------------------------------- browser

const candidates = [
  arg('browser', ''),
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Microsoft/Edge/Application/msedge.exe',
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
  '/usr/bin/google-chrome', '/usr/bin/chromium', '/usr/bin/microsoft-edge',
].filter(Boolean);
const BROWSER = candidates.find((c) => existsSync(c));
if (!BROWSER) throw new Error('No Edge / Chrome found; pass --browser=<path>');

const profile = mkdtempSync(join(tmpdir(), 'ochre-shots-'));
const browser = spawn(BROWSER, [
  '--headless=new', '--disable-gpu', '--hide-scrollbars', '--mute-audio', '--no-first-run', '--no-default-browser-check',
  '--disable-extensions', '--disable-sync', `--user-data-dir=${profile}`, '--remote-debugging-port=0', 'about:blank',
], { stdio: ['ignore', 'ignore', 'pipe'] });

const wsUrl = await new Promise((res, rej) => {
  let buf = '';
  const t = setTimeout(() => rej(new Error('browser did not start')), 20000);
  browser.stderr.on('data', (d) => {
    buf += d;
    const m = /DevTools listening on (ws:\/\/\S+)/.exec(buf);
    if (m) { clearTimeout(t); res(m[1]); }
  });
});

const ws = new WebSocket(wsUrl);
await new Promise((r) => ws.addEventListener('open', r, { once: true }));
let seq = 0;
const pending = new Map();
ws.addEventListener('message', (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) {
    const { res, rej } = pending.get(msg.id);
    pending.delete(msg.id);
    if (msg.error) rej(new Error(msg.error.message)); else res(msg.result);
  }
});
const cdp = (method, params = {}, sessionId) => new Promise((res, rej) => {
  const id = ++seq;
  pending.set(id, { res, rej });
  ws.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }));
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const { targetId } = await cdp('Target.createTarget', { url: 'about:blank' });
const { sessionId } = await cdp('Target.attachToTarget', { targetId, flatten: true });
const page = (m, p) => cdp(m, p, sessionId);
await page('Page.enable');
await page('Runtime.enable');

async function evaluate(expr) {
  const r = await page('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true });
  return r.result && r.result.value;
}

mkdirSync(OUT, { recursive: true });
let n = 0;
try {
  for (const s of shots) {
    if (ONLY && !ONLY.has(s.name)) continue;
    await page('Emulation.setDeviceMetricsOverride', { width: s.w, height: s.h, deviceScaleFactor: s.dpr, mobile: false });
    await page('Page.navigate', { url: 'about:blank' });
    await page('Page.navigate', { url: ORIGIN + s.url });
    const t0 = Date.now();
    while (!(await evaluate('document.readyState === "complete" && window.__ochreReady === true').catch(() => false))) {
      if (Date.now() - t0 > 10000) throw new Error(`${s.name}: page never became ready`);
      await sleep(50);
    }
    await sleep(250);
    if (arg('debug', '')) console.log(s.name, JSON.stringify(await evaluate(arg('debug', ''))));
    const { data } = await page('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
    writeFileSync(join(OUT, `${s.name}.png`), Buffer.from(data, 'base64'));
    n++;
    process.stdout.write(`${s.name}.png\n`);
  }
} finally {
  ws.close();
  browser.kill();
  server.close();
  await sleep(300);
  try { rmSync(profile, { recursive: true, force: true }); } catch { /* the browser may still hold it briefly */ }
}
console.log(`${n} shots -> ${OUT}`);
