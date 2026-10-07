// Onboarding: which core events redraw what, and the "Try it" step's state. Pure (no DOM), so
// node --test covers it (app/test/ui.test.mjs).
//
// The try-it box is a real text field that Ochre types into like any other app. It must never be
// rebuilt while the step is open: rebuilding it on the `result` event (which arrives right after
// the text was typed) used to throw the dictated text away and bring the placeholder back.

/** What an event does to the open step: 'page' (rebuild it), 'patch' (update in place) or null. */
export function redrawFor(stepId, ev, { downloadDone = false } = {}) {
  switch (stepId) {
    case 'permissions': return ev.event === 'permissions' ? 'page' : null;
    case 'model':
      if (ev.event === 'state' || ev.event === 'engines') return 'page';
      if (ev.event === 'download' && downloadDone) return 'page';
      return null;
    // never 'page': the box holds what the user just dictated
    case 'try': return ev.event === 'result' ? 'patch' : null;
    default: return null;
  }
}

/** The try step's state: `baseline` is the result that was already there when onboarding opened
    (older sessions don't count). */
export function tryInitial(baseline = null) {
  return { baseline, result: null };
}

export function tryReduce(s, ev) {
  if (!ev || ev.event !== 'result' || ev === s.baseline) return s;
  if (s.baseline && ev.id && ev.id === s.baseline.id) return s;
  return { ...s, result: ev };
}

/** The line under the box. tone: 'tip' (waiting) | 'success' | 'notice' (dictated, not typed). */
export function tryStatus(s, keyLabel) {
  const r = s.result;
  if (!r) return { tone: 'tip', text: `Tip: tap ${keyLabel} twice to keep recording hands-off, then tap once to finish.` };
  const ms = r.timings && r.timings.release_to_insert_ms;
  if (r.inserted === false) return { tone: 'notice', text: 'Ochre heard you, but couldn’t type here. The text is in History.', ms: null };
  return { tone: 'success', text: 'That’s it.', ms: Number.isFinite(ms) ? ms : null };
}
