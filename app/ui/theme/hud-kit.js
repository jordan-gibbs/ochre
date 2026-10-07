/* Ochre HUD kit: the three motion helpers that make the HUD feel smooth.
   Framework-free and dependency-free. Load as a classic <script src="../theme/hud-kit.js"> (or
   `import '../theme/hud-kit.js'` from a module) and use window.HudKit. The UI engineer owns when
   these are called.

     HudKit.setState(hudEl, state, apply)   morph the pill's width and cross-fade old -> new content
     HudKit.morph(hudEl, apply)             the same glide for a change within a state
     HudKit.oMeter(oEl)                     the speaking "o": -> { push(rms), start(), stop(), destroy() }
     HudKit.oSnapshot(oEl, scale)           a frozen o, for screenshots
     HudKit.fakeVoice(seed)                 a speech-like level generator for demos: t(ms) -> rms
     HudKit.levelMeter(barsEl)              legacy five-bar meter (kept for old markup)
     HudKit.fitLive(liveEl)                 live text line: pin the newest words to the right
     HudKit.fitBubble(bubbleEl)             toggle the top fade when text overflows

   Rules they follow: read layout once, write once, never read after a write in the same frame.
   Per frame the o meter writes `transform` on ONE element (the o), and stops writing once it has
   settled. 0 layouts while metering
   (python app/ui/theme/perf-check.py). */

(function (root) {
  'use strict';

  const reduced = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;
  const cssMs = (el, name, dflt) => {
    const v = getComputedStyle(el).getPropertyValue(name).trim();
    const n = parseFloat(v);
    return Number.isFinite(n) ? (v.endsWith('ms') ? n : v.endsWith('s') ? n * 1000 : n) : dflt;
  };

  /* Change the HUD state and let the pill glide to its new width (FLIP on width).
     `apply()` does the DOM writes for the new state (labels, hidden buttons, data-state). */
  function setState(hudEl, state, apply) {
    if (hudEl.dataset.state === 'idle') {
      hudEl.dataset.state = state;
      if (apply) apply();
      return;
    }
    morph(hudEl, () => {
      hudEl.dataset.state = state;
      if (apply) apply();
    });
  }

  /* The content block that is showing: the live line while words stream, else label + hint. */
  const shownContent = (hudEl) => hudEl.querySelector(hudEl.hasAttribute('data-live') ? '.hud-live' : '.hud-text');
  const contentSig = (hudEl, el) => (el ? (hudEl.hasAttribute('data-live') ? 'live' : 'text:' + el.textContent) : '');

  /* FLIP the pill's width around `apply()` (any DOM change that resizes it), and cross-fade the
     content: the old label (or live line) is cloned into an absolutely positioned ghost that
     fades out with a slight lift and blur while the new one rises in. Nothing snaps, and the
     ghost never takes part in layout. */
  function morph(hudEl, apply) {
    const pill = hudEl.querySelector('.hud-pill');
    if (!pill || pill.offsetParent === null) {
      if (apply) apply();
      return;
    }
    const pr = pill.getBoundingClientRect();                 // read
    const from = pr.width;
    const oldEl = shownContent(hudEl);
    const oldSig = contentSig(hudEl, oldEl);
    let ghost = null;
    if (oldEl && oldEl.offsetParent !== null && oldSig !== 'text:') {
      const r = oldEl.getBoundingClientRect();               // read (same layout as above)
      ghost = oldEl.cloneNode(true);
      ghost.classList.remove('is-swapping');
      ghost.classList.add('hud-ghost');
      ghost.setAttribute('aria-hidden', 'true');
      ghost.style.left = (r.left - pr.left) + 'px';
      ghost.style.width = r.width + 'px';
    }
    pill.style.width = '';
    if (apply) apply();
    const newEl = shownContent(hudEl);
    const changed = contentSig(hudEl, newEl) !== oldSig;
    if (ghost && changed) {
      pill.querySelectorAll(':scope > .hud-ghost').forEach((g) => g.remove());
      pill.appendChild(ghost);
      const drop = () => ghost.remove();
      ghost.addEventListener('animationend', drop, { once: true });
      setTimeout(drop, cssMs(hudEl, '--dur-3', 240) + 120);
    }
    if (newEl && changed) {
      newEl.classList.remove('is-swapping');
      void newEl.offsetWidth;                                // restart the enter animation
      newEl.classList.add('is-swapping');
    }
    const to = pill.getBoundingClientRect().width;          // read (one forced layout, on purpose)
    if (Math.abs(to - from) < 0.5 || reduced()) return;
    pill.style.width = from + 'px';                          // write
    pill.getBoundingClientRect();
    pill.style.width = to + 'px';
    clearTimeout(pill._ochreMorph);
    pill._ochreMorph = setTimeout(() => { pill.style.width = ''; }, cssMs(hudEl, '--dur-4', 380) + 60);
  }

  /* ------------------------------------------------------------------ the speaking "o"
     Level events (~30/s, 0..1 on a -60..-12 dB scale) scale the brand o: 1.0 in silence, up to
     1 + --o-grow (1.3) when you speak up. Nothing else moves: no rings, no breathing.
       floor     an adaptive noise floor, so room noise reads as silence (exactly 1.0)
       envelope  a one-pole follower with separate attack / release (--o-attack 80 ms,
                 --o-release 300 ms), then a second, faster one-pole (40 ms) so the motion is
                 eased (no corners) and never jitters between level events
       curve     ease-out on the envelope, so normal speech already reads, loud speech tops out
     The o's transform is the only per-frame style write, and the loop goes to sleep (no writes,
     no frames) once the o has settled back at 1.0. Reduced motion: the same, with less growth. */
  function oMeter(oEl) {
    const glyph = oEl.querySelector('.hud-o-glyph') || oEl;
    let raf = 0, last = 0, lastPush = 0, written = 1;
    let target = 0, env = 0, smooth = 0, floor = 0.3;
    let cfg = { attack: 80, release: 300, grow: 0.3 };

    function readCfg() {
      // style reads happen here, once per start, never inside the frame loop
      const cs = getComputedStyle(oEl);
      const num = (n, d) => {
        const v = String(cs.getPropertyValue(n) || '').trim();
        const x = parseFloat(v);
        return Number.isFinite(x) ? (v.endsWith('ms') ? x : v.endsWith('s') ? x * 1000 : x) : d;
      };
      const grow = Math.max(0, Math.min(0.5, num('--o-grow', 0.3)));
      cfg = { attack: num('--o-attack', 80), release: num('--o-release', 300), grow: reduced() ? grow * 0.4 : grow };
    }

    function write(sc) {
      if (Math.abs(sc - written) < 0.0005) return;
      written = sc;
      glyph.style.transform = sc === 1 ? '' : `scale(${sc.toFixed(4)})`;
    }

    function frame(t) {
      const dt = last ? Math.min(64, t - last) : 16;
      last = t;
      if (t - lastPush > 200) target *= Math.exp(-dt / 160);   // the core went quiet
      const tau = target > env ? cfg.attack : cfg.release;
      env += (target - env) * (1 - Math.exp(-dt / tau));
      smooth += (env - smooth) * (1 - Math.exp(-dt / 40));
      const x = Math.max(0, Math.min(1, smooth));
      const sc = 1 + cfg.grow * (1 - (1 - x) * (1 - x));
      if (target < 0.002 && smooth < 0.008) {               // settled (< 0.5% off): rest at 1.0, sleep
        env = smooth = target = 0;
        write(1);
        raf = 0;
        last = 0;
        return;
      }
      write(sc);
      raf = requestAnimationFrame(frame);
    }

    return {
      /* rms: 0..1 from the core's `level` event. */
      push(rms) {
        const now = performance.now();
        const v = Math.max(0, Math.min(1, Number(rms) || 0));
        const dt = lastPush ? Math.min(250, now - lastPush) : 33;
        lastPush = now;
        // adaptive noise floor: drops fast to quiet input, creeps up slowly under speech
        floor += (v - floor) * (1 - Math.exp(-dt / (v < floor ? 90 : 5000)));
        const x = Math.max(0, Math.min(1, (v - floor - 0.04) / Math.max(0.2, 0.9 - floor)));
        target = Math.pow(x, 0.8);
        if (!raf && target > 0.002) this.start();
      },
      start() {
        if (raf) return;
        readCfg();
        last = 0;
        raf = requestAnimationFrame(frame);
      },
      stop() {
        cancelAnimationFrame(raf); raf = 0; last = 0; target = 0; env = 0; smooth = 0;
        written = 1;
        glyph.style.transform = '';
      },
      /* the o's current scale (tests, previews) */
      get scale() { return written; },
      destroy() { this.stop(); },
    };
  }

  /* A frozen o for screenshots: the glyph at `scale` (1 = silent). */
  function oSnapshot(oEl, scale) {
    const g = oEl.querySelector('.hud-o-glyph');
    if (g) g.style.transform = scale && scale !== 1 ? `scale(${scale})` : '';
  }

  /* Speech-like level for demos (the same shape as the shell's --demo): syllables of 110-260 ms
     at varied loudness, short gaps between words, a breath every few words, over a room-noise
     floor. Returns t(ms) -> rms on the core's 0..1 (-60..-12 dB) scale. Deterministic per seed;
     call it with increasing t. */
  function fakeVoice(seed) {
    let x = (seed >>> 0) || 0x2545f491;
    const rnd = () => { x ^= x << 13; x >>>= 0; x ^= x >>> 17; x ^= x << 5; x >>>= 0; return (x % 10000) / 10000; };
    let segStart = 0, segEnd = 0, amp = 0, kind = 'gap', syllables = 0, every = 7;
    function next(t) {
      segStart = t;
      if (kind === 'syl') {
        syllables++;
        const pause = syllables % every === 0;
        if (pause) every = 5 + Math.floor(rnd() * 5);
        kind = 'gap';
        segEnd = t + (pause ? 260 + rnd() * 380 : rnd() < 0.55 ? 25 + rnd() * 50 : 70 + rnd() * 110);
      } else {
        kind = 'syl';
        amp = 0.5 + rnd() * 0.42;
        segEnd = t + 110 + rnd() * 150;
      }
    }
    return function level(t) {
      if (t < segStart) { segStart = segEnd = t; }
      while (t >= segEnd) next(segEnd);
      const noise = 0.2 + rnd() * 0.06;
      if (kind !== 'syl') return noise;
      const p = (t - segStart) / (segEnd - segStart);
      const shape = Math.pow(Math.sin(Math.PI * Math.min(1, p * 1.2)), 0.7);  // quick rise, softer fall
      return Math.min(1, Math.max(noise, noise + (amp - 0.15) * shape + (rnd() - 0.5) * 0.04));
    };
  }

  /* Level meter: core sends rms ~15/s; we animate at display rate with a critically-damped
     follow plus a little per-bar motion so it reads as a voice, not a VU needle. */
  const WEIGHTS = [0.55, 0.9, 1.15, 0.8, 0.5];
  const FLOOR = 0.17;

  function levelMeter(barsEl) {
    const bars = Array.from(barsEl.children);
    let target = 0, value = 0, raf = 0, t0 = 0, lastPush = 0, k = 0.22, calm = false;
    const phase = bars.map((_, i) => i * 1.7 + 0.4);

    function frame(t) {
      raf = requestAnimationFrame(frame);
      if (!t0) t0 = t;
      // decay towards silence if the core stops sending (e.g. a dropped event)
      if (t - lastPush > 260) target *= 0.9;
      value += (target - value) * k;
      const s = (t - t0) / 1000;
      for (let i = 0; i < bars.length; i++) {
        const wob = calm ? 1 : 0.82 + 0.18 * Math.sin(s * (5.2 + i * 0.9) + phase[i]);
        const y = Math.min(1, FLOOR + value * WEIGHTS[i % WEIGHTS.length] * wob);
        bars[i].style.transform = `scaleY(${y.toFixed(3)})`;
      }
    }
    return {
      /* rms: 0..1 from the core's `level` event. A gentle curve lifts quiet speech. */
      push(rms) {
        const v = Math.max(0, Math.min(1, Number(rms) || 0));
        target = Math.pow(v, 0.62);
        lastPush = performance.now();
        if (!raf) this.start();
      },
      start() {
        if (raf) return;
        // style reads happen here, once, never inside the frame loop
        k = parseFloat(getComputedStyle(barsEl).getPropertyValue('--level-smoothing')) || 0.22;
        calm = reduced();
        t0 = 0;
        raf = requestAnimationFrame(frame);
      },
      stop() {
        cancelAnimationFrame(raf); raf = 0; target = 0; value = 0;
        bars.forEach((b) => { b.style.transform = ''; });
      },
      destroy() { this.stop(); },
    };
  }

  /* Bubble: show the soft top fade only when older lines are clipped. */
  function fitBubble(bubbleEl) {
    const sc = bubbleEl.querySelector('.hud-bubble-scroll');
    if (!sc) return;
    bubbleEl.classList.toggle('is-overflowing', sc.scrollHeight > sc.clientHeight + 1);
  }

  /* Live text line: while it fits, words sit left-aligned; once it overflows, the line pins its
     end (newest words) to the right edge and the left fades out. One layout read per update. */
  function fitLive(liveEl) {
    const text = liveEl.querySelector('.hud-live-text');
    if (!text) return;
    const caret = liveEl.querySelector('.hud-live-caret');
    const need = text.getBoundingClientRect().width + (caret && caret.offsetParent !== null ? 6 : 0);
    liveEl.classList.toggle('is-overflowing', need > liveEl.clientWidth + 0.5);
  }

  const HudKit = { setState, morph, oMeter, oSnapshot, fakeVoice, levelMeter, fitLive, fitBubble };
  root.HudKit = HudKit;
  if (typeof module === 'object' && module.exports) module.exports = HudKit;
})(typeof self !== 'undefined' ? self : this);
