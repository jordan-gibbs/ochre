// Line icons (24 px grid, 1.6 stroke, round caps), drawn for this app. Use with `icon(name)`.

const P = {
  general: '<circle cx="12" cy="12" r="3.2"/><path d="M12 2.8v2.4M12 18.8v2.4M21.2 12h-2.4M5.2 12H2.8M18.5 5.5l-1.7 1.7M7.2 16.8l-1.7 1.7M18.5 18.5l-1.7-1.7M7.2 7.2 5.5 5.5"/>',
  transcription: '<rect x="9" y="2.8" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5v3.7"/>',
  refinement: '<path d="M12 3c.6 3.9 1.9 5.8 5.6 6.8.6.2.6 1.1 0 1.3-3.7 1-5 2.9-5.6 6.8-.1.6-1 .6-1.1 0-.6-3.9-1.9-5.8-5.6-6.8-.6-.2-.6-1.1 0-1.3 3.7-1 5-2.9 5.6-6.8.1-.6 1-.6 1.1 0z"/><path d="M19 15.5v4M17 17.5h4"/>',
  handsfree: '<path d="M4 12a8 8 0 0 1 16 0"/><path d="M7.5 12a4.5 4.5 0 0 1 9 0"/><circle cx="12" cy="12" r="1.3"/><path d="M12 15.5v5"/>',
  dictionary: '<path d="M5 4.5A1.5 1.5 0 0 1 6.5 3H19v15H6.5A1.5 1.5 0 0 0 5 19.5v-15z"/><path d="M5 19.5A1.5 1.5 0 0 0 6.5 21H19v-3"/><path d="M9 7.5h6M9 11h4"/>',
  history: '<path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1"/><path d="M3.5 4.5v3.6h3.6"/><path d="M12 7.5V12l3 2"/>',
  about: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5M12 7.6v.1"/>',
  check: '<path d="M5 12.5l4.3 4.3L19 7"/>',
  x: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
  copy: '<rect x="8.5" y="8.5" width="11" height="11" rx="2.2"/><path d="M15.5 8.5V6.2A1.7 1.7 0 0 0 13.8 4.5H6.2A1.7 1.7 0 0 0 4.5 6.2v7.6a1.7 1.7 0 0 0 1.7 1.7h2.3"/>',
  insert: '<path d="M4 12h11"/><path d="M11 7.5L15.5 12 11 16.5"/><path d="M19.5 5v14"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.3-4.3"/>',
  download: '<path d="M12 4v11"/><path d="M7.5 10.5L12 15l4.5-4.5"/><path d="M5 19.5h14"/>',
  key: '<circle cx="8" cy="15" r="4"/><path d="M10.8 12.2L19 4M16 7l2.5 2.5M14 9l1.8 1.8"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  trash: '<path d="M4.5 7h15M10 11v6M14 11v6M6.5 7l1 12.5h9l1-12.5M9.5 7V4.5h5V7"/>',
  mic: '<rect x="9" y="2.8" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5v3.7"/>',
  shield: '<path d="M12 3l7.5 3v5.5c0 4.6-3.2 8.2-7.5 9.5-4.3-1.3-7.5-4.9-7.5-9.5V6L12 3z"/><path d="M8.8 12.2l2.2 2.2 4.4-4.6"/>',
  keyboard: '<rect x="2.8" y="6" width="18.4" height="12" rx="2.4"/><path d="M6.5 9.5h.1M10 9.5h.1M13.5 9.5h.1M17 9.5h.1M7.5 14.5h9"/>',
  sparkle: '<path d="M12 3c.6 3.9 1.9 5.8 5.6 6.8.6.2.6 1.1 0 1.3-3.7 1-5 2.9-5.6 6.8-.1.6-1 .6-1.1 0-.6-3.9-1.9-5.8-5.6-6.8-.6-.2-.6-1.1 0-1.3 3.7-1 5-2.9 5.6-6.8.1-.6 1-.6 1.1 0z"/>',
  cloud: '<path d="M7 18.5h10.5a4 4 0 0 0 .6-8 6 6 0 0 0-11.6 1.3A3.4 3.4 0 0 0 7 18.5z"/>',
  chip: '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9.5 2.8V6M14.5 2.8V6M9.5 18v3.2M14.5 18v3.2M2.8 9.5H6M2.8 14.5H6M18 9.5h3.2M18 14.5h3.2"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  external: '<path d="M14 4.5h5.5V10M19.5 4.5L11 13M17 14v4.5a1 1 0 0 1-1 1H5.5a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1H10"/>',
  lock: '<rect x="5" y="10.5" width="14" height="10" rx="2.4"/><path d="M8.2 10.5V8a3.8 3.8 0 0 1 7.6 0v2.5"/>',
  bolt: '<path d="M13 2.8L5.5 13.5H12l-1 7.7 7.5-10.7H12l1-7.7z"/>',
};

export function icon(name, cls = '') {
  return `<svg class="ico ${cls}" viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${P[name] || ''}</svg>`;
}
