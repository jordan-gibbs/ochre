//! Live text for the HUD while the user talks. Display only: nothing here is ever injected; the
//! inserted text still comes from `finish()` -> strip_hesitations -> corrections -> refine.
//!
//! Hot-path contract (docs/latency.md): the decode thread (or a streaming engine's reader)
//! calls the session's `PartialFn`, which only copies the text into a one-slot mailbox and wakes
//! the pump. It never waits on the UI. The pump thread strips hesitations, coalesces to at most
//! one update per `MIN_INTERVAL` (~30 Hz; the newest text always wins, so the last partial is
//! never lost) and emits `Event::Partial` on the bus.
//!
//! Each session gets a generation number. Closing it (text released for insertion, cancel) bumps
//! the generation, so a late decode from an old session can never repaint the HUD.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use ochre_core::events::{Bus, Event};

use crate::seams::PartialFn;
use crate::text::strip_hesitations;

/// At most ~30 HUD updates a second.
pub const MIN_INTERVAL: Duration = Duration::from_millis(33);

struct Offer {
    epoch: u64,
    text: String,
    /// Byte offset into `text`.
    stable: usize,
}

#[derive(Default)]
struct Slot {
    next: Option<Offer>,
    quit: bool,
}

#[derive(Default)]
struct Inner {
    slot: Mutex<Slot>,
    wake: Condvar,
    /// The open session's generation; anything else is stale.
    epoch: AtomicU64,
}

impl Inner {
    fn offer(&self, epoch: u64, text: &str, stable: usize) {
        if self.epoch.load(Ordering::Acquire) != epoch {
            return;
        }
        let mut s = self.slot.lock();
        match s.next.as_mut() {
            Some(o) => {
                o.epoch = epoch;
                o.text.clear();
                o.text.push_str(text);
                o.stable = stable;
            }
            None => {
                s.next = Some(Offer {
                    epoch,
                    text: text.to_owned(),
                    stable,
                })
            }
        }
        drop(s);
        self.wake.notify_one();
    }
}

/// The pump: one per `App`. Dropping it stops the thread.
pub struct LiveText {
    inner: Arc<Inner>,
}

/// One session's handle: hand `partial_fn()` to the segmenter / stream, `close()` when done.
#[derive(Clone)]
pub struct LiveSession {
    epoch: u64,
    inner: Arc<Inner>,
}

impl LiveText {
    pub fn new(bus: Bus) -> Self {
        let inner = Arc::new(Inner::default());
        let pump = inner.clone();
        std::thread::Builder::new()
            .name("ochre-live".into())
            .spawn(move || run(&pump, &bus))
            .expect("spawn live-text pump");
        Self { inner }
    }

    /// Start a session; any previous session's partials become stale.
    pub fn open(&self) -> LiveSession {
        let epoch = self.inner.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.inner.slot.lock().next = None;
        LiveSession {
            epoch,
            inner: self.inner.clone(),
        }
    }
}

impl Drop for LiveText {
    fn drop(&mut self) {
        self.inner.epoch.fetch_add(1, Ordering::AcqRel);
        self.inner.slot.lock().quit = true;
        self.inner.wake.notify_all();
    }
}

impl LiveSession {
    /// `(text, stable byte offset)`, non-blocking; safe to call from the decode thread.
    pub fn partial_fn(&self) -> PartialFn {
        let (epoch, inner) = (self.epoch, self.inner.clone());
        Arc::new(move |text: &str, stable: usize| inner.offer(epoch, text, stable))
    }

    /// No more partials from this session (a no-op if a newer session is already open).
    pub fn close(&self) {
        if self
            .inner
            .epoch
            .compare_exchange(
                self.epoch,
                self.epoch + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
        {
            self.inner.slot.lock().next = None;
        }
    }
}

fn run(inner: &Inner, bus: &Bus) {
    let mut last: Option<(String, usize)> = None;
    loop {
        let offer = {
            let mut s = inner.slot.lock();
            loop {
                if s.quit {
                    return;
                }
                if let Some(o) = s.next.take() {
                    break o;
                }
                inner.wake.wait(&mut s);
            }
        };
        if inner.epoch.load(Ordering::Acquire) != offer.epoch {
            continue;
        }
        let shown = display(&offer.text, offer.stable);
        if last.as_ref() == Some(&shown) {
            continue;
        }
        if inner.epoch.load(Ordering::Acquire) != offer.epoch {
            continue; // closed while we were stripping
        }
        bus.emit(Event::Partial {
            text: shown.0.clone(),
            stable_chars: shown.1,
        });
        last = Some(shown);
        // Coalesce: offers that land meanwhile overwrite each other; the newest is sent next.
        std::thread::sleep(MIN_INTERVAL);
    }
}

/// What the HUD shows for a raw partial: hesitations stripped (so "um" never flashes), plus the
/// stable prefix length in UTF-16 code units (a JS string index), mapped through the stripping.
pub fn display(text: &str, stable: usize) -> (String, usize) {
    let mut k = stable.min(text.len());
    while !text.is_char_boundary(k) {
        k -= 1;
    }
    let full = strip_hesitations(text);
    if k == 0 {
        return (full, 0);
    }
    let head = strip_hesitations(&text[..k]);
    let common: usize = full
        .chars()
        .zip(head.chars())
        .take_while(|(a, b)| a == b)
        .map(|(a, _)| a.len_utf16())
        .sum();
    (full, common)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn partials(events: &Mutex<Vec<Event>>) -> Vec<(String, usize)> {
        events
            .lock()
            .iter()
            .filter_map(|e| match e {
                Event::Partial { text, stable_chars } => Some((text.clone(), *stable_chars)),
                _ => None,
            })
            .collect()
    }

    fn rig() -> (LiveText, Arc<Mutex<Vec<Event>>>) {
        let bus = Bus::new();
        let events = Arc::new(Mutex::new(Vec::new()));
        let ev = events.clone();
        bus.subscribe(move |e| ev.lock().push(e.clone()));
        (LiveText::new(bus), events)
    }

    fn wait(cond: impl Fn() -> bool) {
        let t = Instant::now();
        while !cond() {
            assert!(t.elapsed() < Duration::from_secs(5), "timed out");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn display_strips_hesitations_and_maps_stable() {
        assert_eq!(display("Um, so the", 0), ("So the".into(), 0));
        let (t, k) = display("so um the build uh", 6);
        assert_eq!(t, "so the build");
        assert_eq!(&t[..k], "so");
        // stable counts UTF-16 units, never splits a char
        let (t, k) = display("café ok", "café".len() - 1);
        assert_eq!(t, "café ok");
        assert_eq!(k, 3);
        let (_, k) = display("naïve 😀 done", "naïve 😀".len());
        assert_eq!(k, "naïve 😀".encode_utf16().count());
    }

    #[test]
    fn bursts_coalesce_in_order_and_keep_the_newest() {
        let (live, events) = rig();
        let s = live.open();
        let f = s.partial_fn();
        let words: Vec<String> = (0..200).map(|i| format!("w{i}")).collect();
        for i in 1..=words.len() {
            f(&words[..i].join(" "), 0);
        }
        let last = words.join(" ");
        wait(|| partials(&events).last().is_some_and(|p| p.0 == last));
        let got = partials(&events);
        assert!(got.len() < 50, "not coalesced: {} updates", got.len());
        for w in got.windows(2) {
            assert!(w[1].0.len() > w[0].0.len(), "out of order: {w:?}");
        }
    }

    #[test]
    fn offer_never_waits_on_a_slow_ui() {
        let bus = Bus::new();
        bus.subscribe(|_| std::thread::sleep(Duration::from_millis(80)));
        let live = LiveText::new(bus);
        let f = live.open().partial_fn();
        let mut worst = Duration::ZERO;
        for i in 0..50 {
            let t = Instant::now();
            f(&format!("hello {i}"), 0);
            worst = worst.max(t.elapsed());
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(worst < Duration::from_millis(20), "offer blocked {worst:?}");
    }

    #[test]
    fn closed_and_stale_sessions_emit_nothing() {
        let (live, events) = rig();
        let old = live.open();
        let old_f = old.partial_fn();
        old.close();
        old_f("late words", 0);
        let new = live.open();
        old_f("older session", 0);
        old.close(); // must not close the newer session
        new.partial_fn()("fresh", 0);
        wait(|| !partials(&events).is_empty());
        std::thread::sleep(MIN_INTERVAL * 3);
        assert_eq!(partials(&events), vec![("fresh".to_string(), 0)]);
    }
}
