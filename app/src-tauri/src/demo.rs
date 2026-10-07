//! Demo mode (`ochre --demo`): a stand-in core that drives every UI state from a script
//! and answers commands the way the real core would (in-memory config, fake downloads, canned
//! history). `--demo-hold=<label>` stops the script at a labelled step, which is how the window
//! behaviour (click-through, no focus steal, always on top) is checked by hand.
//!
//! One worker thread owns a timeline of (due, step). Commands arrive on a channel; the thread waits
//! on that channel with a timeout equal to the next due step, so nothing spins.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ochre_core::config::Config;
use ochre_core::events::{Bus, Command, Event, State, Timings, Trigger};
use ochre_core::history::Entry;

use crate::core_bridge::{self, Core};

const MB: u64 = 1024 * 1024;
/// The real core sends a level every 33 ms (ochre-audio capture.rs); the demo matches it so
/// the HUD's speaking "o" sees the same cadence.
const LEVEL_EVERY: Duration = Duration::from_millis(33);

/// Labels `--demo-hold` accepts, in script order.
pub const LABELS: &[&str] = &[
    "download",
    "loading",
    "recording",
    "transcribing",
    "refining",
    "inserted",
    "locked",
    "pip",
    "handsfree",
    "notice",
    "error",
];

pub const SENTENCE: &str = "Thanks for the notes, I will send the deck tonight and loop in Priya on the pricing questions.";
/// What the demo "hears" while the user talks; the live text strips the hesitations, as the real
/// core does, so "um" / "uh" never show in the pill.
pub const SPOKEN: &str = "Thanks for the notes, um I will send the deck tonight and uh loop in Priya on the pricing questions.";

/// The live-text event after `i` spoken words: hesitations stripped and the stable prefix
/// mapped exactly like the real core (`ochre::live::display`); the last three words unconfirmed.
fn partial_after(i: usize) -> Event {
    let words: Vec<&str> = SPOKEN.split(' ').collect();
    let i = i.min(words.len());
    let raw = words[..i].join(" ");
    let stable = words[..i.saturating_sub(3)].join(" ").len();
    let (text, stable_chars) = ochre::live::display(&raw, stable);
    Event::Partial { text, stable_chars }
}

/// A synthetic voice for the level meter, shaped like speech so the HUD's speaking "o" shows
/// what it does with a real one: syllables of 110-260 ms at varied loudness, short gaps between
/// words, a breath every 5-9 syllables, all over a room-noise floor. Values are on the core's
/// 0..1 (-60..-12 dB) scale. The same shape as `HudKit.fakeVoice` in app/ui/theme/hud-kit.js.
struct Voice {
    seed: u32,
    started: Instant,
    seg_start: f32,
    seg_end: f32,
    amp: f32,
    syllable: bool,
    syllables: u32,
    every: u32,
}

impl Voice {
    fn new(seed: u32) -> Self {
        Self {
            seed,
            started: Instant::now(),
            seg_start: 0.0,
            seg_end: 0.0,
            amp: 0.0,
            syllable: false,
            syllables: 0,
            every: 7,
        }
    }

    fn rnd(&mut self) -> f32 {
        self.seed ^= self.seed << 13;
        self.seed ^= self.seed >> 17;
        self.seed ^= self.seed << 5;
        (self.seed % 10_000) as f32 / 10_000.0
    }

    fn next(&mut self, t: f32) {
        self.seg_start = t;
        if self.syllable {
            self.syllables += 1;
            let pause = self.syllables.is_multiple_of(self.every);
            if pause {
                self.every = 5 + (self.rnd() * 5.0) as u32;
            }
            self.syllable = false;
            let r = self.rnd();
            self.seg_end = t + if pause {
                260.0 + r * 380.0
            } else if self.rnd() < 0.55 {
                25.0 + r * 50.0
            } else {
                70.0 + r * 110.0
            };
        } else {
            self.syllable = true;
            self.amp = 0.5 + self.rnd() * 0.42;
            self.seg_end = t + 110.0 + self.rnd() * 150.0;
        }
    }

    /// The level at `t` ms (call with increasing `t`).
    fn at(&mut self, t: f32) -> f32 {
        if t < self.seg_start {
            self.seg_start = t;
            self.seg_end = t;
        }
        while t >= self.seg_end {
            self.next(self.seg_end);
        }
        let noise = 0.2 + self.rnd() * 0.06;
        if !self.syllable {
            return noise;
        }
        let p = (t - self.seg_start) / (self.seg_end - self.seg_start);
        let shape = (std::f32::consts::PI * (p * 1.2).min(1.0))
            .sin()
            .max(0.0)
            .powf(0.7);
        let jitter = (self.rnd() - 0.5) * 0.04;
        (noise + (self.amp - 0.15) * shape + jitter).clamp(noise, 1.0)
    }
}

#[derive(Debug, Clone)]
enum Step {
    Emit(Event),
    Label(&'static str),
    /// End of the showcase: loop again (unless `--demo-once`).
    End,
}

enum Msg {
    Start,
    Cmd(Command),
    Stop,
}

pub struct DemoCore {
    tx: Mutex<Sender<Msg>>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl DemoCore {
    pub fn new(bus: Bus, hold: Option<String>, looping: bool) -> Self {
        let (tx, rx) = mpsc::channel::<Msg>();
        let worker = std::thread::Builder::new()
            .name("ochre-demo".into())
            .spawn(move || {
                let mut d = Driver::new(bus, hold, looping);
                loop {
                    let wait = d.next_wait();
                    match rx.recv_timeout(wait) {
                        Ok(Msg::Start) => d.start(),
                        Ok(Msg::Cmd(c)) => d.command(c),
                        Ok(Msg::Stop) | Err(RecvTimeoutError::Disconnected) => break,
                        Err(RecvTimeoutError::Timeout) => d.tick(),
                    }
                }
            })
            .expect("spawn demo thread");
        Self {
            tx: Mutex::new(tx),
            worker: Mutex::new(Some(worker)),
        }
    }

    fn send(&self, m: Msg) {
        let _ = self.tx.lock().unwrap().send(m);
    }
}

impl Core for DemoCore {
    fn start(&self) {
        self.send(Msg::Start);
    }
    fn command(&self, cmd: Command) {
        self.send(Msg::Cmd(cmd));
    }
    fn shutdown(&self) {
        self.send(Msg::Stop);
        if let Some(h) = self.worker.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

// --------------------------------------------------------------------------------------------

struct Driver {
    bus: Bus,
    cfg: Config,
    secrets: BTreeMap<String, bool>,
    hold: Option<String>,
    looping: bool,
    timeline: VecDeque<(Instant, Step)>,
    state: State,
    armed: bool,
    next_level: Option<Instant>,
    voice: Voice,
    /// Interactive session (tray / CLI start): words spoken so far.
    words: usize,
    session_started: Option<Instant>,
}

impl Driver {
    fn new(bus: Bus, hold: Option<String>, looping: bool) -> Self {
        let secrets = ["groq", "openai", "anthropic", "soniox"]
            .into_iter()
            .map(|p| (p.to_string(), p == "groq"))
            .collect();
        Self {
            bus,
            cfg: demo_config(),
            secrets,
            hold,
            looping,
            timeline: VecDeque::new(),
            state: State::Idle,
            armed: false,
            next_level: None,
            voice: Voice::new(0x2545_f491),
            words: 0,
            session_started: None,
        }
    }

    fn next_wait(&self) -> Duration {
        let now = Instant::now();
        let due = [self.timeline.front().map(|(t, _)| *t), self.next_level]
            .into_iter()
            .flatten()
            .min();
        due.map(|t| t.saturating_duration_since(now))
            .unwrap_or(Duration::from_secs(3600))
    }

    fn emit(&mut self, e: Event) {
        if let Event::State {
            state,
            handsfree_armed,
            ..
        } = &e
        {
            self.state = *state;
            self.armed = *handsfree_armed;
            let live = matches!(state, State::Recording | State::Locked | State::Handsfree);
            self.next_level = live.then(|| Instant::now() + LEVEL_EVERY);
        }
        self.bus.emit(e);
    }

    fn start(&mut self) {
        self.emit(Event::Hello {
            version: format!("{}-demo", core_bridge::version()),
            platform: core_bridge::platform(),
        });
        self.emit_config();
        self.emit(core_bridge::engines());
        self.emit(ochre::connectors::event());
        self.emit(Event::Permissions { missing: vec![] });
        self.schedule(showcase(), Duration::from_millis(300));
    }

    fn schedule(&mut self, steps: Vec<(u64, Step)>, after: Duration) {
        let mut t = Instant::now() + after;
        for (ms, step) in steps {
            t += Duration::from_millis(ms);
            self.timeline.push_back((t, step));
        }
    }

    fn tick(&mut self) {
        let now = Instant::now();
        if self.next_level.is_some_and(|t| t <= now) {
            let rms = self.level();
            self.bus.emit(Event::Level { rms });
            self.next_level = Some(now + LEVEL_EVERY);
            if let Some(started) = self.session_started {
                self.session_tick(started);
            }
        }
        while self.timeline.front().is_some_and(|(t, _)| *t <= now) {
            let (_, step) = self.timeline.pop_front().unwrap();
            match step {
                Step::Emit(e) => self.emit(e),
                Step::Label(l) => {
                    if self.hold.as_deref() == Some(l) {
                        self.timeline.clear();
                        // a held "inserted" would vanish with its toast; keep re-sending the result
                        if l == "inserted" {
                            self.schedule(
                                vec![
                                    (1800, Step::Emit(result_event(SENTENCE, true, true, 212))),
                                    (0, Step::Label("inserted")),
                                ],
                                Duration::ZERO,
                            );
                        }
                    }
                }
                Step::End => {
                    if self.looping {
                        self.schedule(showcase(), Duration::from_millis(1500));
                    }
                }
            }
        }
    }

    /// Speech-like level (see [`Voice`]).
    fn level(&mut self) -> f32 {
        self.voice
            .at(self.voice.started.elapsed().as_secs_f32() * 1000.0)
    }

    fn emit_config(&mut self) {
        let e = core_bridge::config_event(&self.cfg, self.secrets.clone());
        self.emit(e);
    }

    fn patch(&mut self, patch: serde_json::Value) {
        match self.cfg.patched(&patch) {
            Ok(c) => self.cfg = c,
            Err(e) => self.emit(Event::Error {
                message: e.to_string(),
                code: "config".into(),
            }),
        }
        self.emit_config();
    }

    fn state(&self, state: State, trigger: Option<Trigger>, detail: &str) -> Event {
        Event::State {
            state,
            trigger,
            handsfree_armed: self.armed,
            detail: detail.into(),
        }
    }

    // ---- interactive session (tray "Start dictation", `ochre toggle`)

    fn begin_session(&mut self, trigger: Trigger) {
        self.timeline.clear();
        self.words = 0;
        self.session_started = Some(Instant::now());
        let e = self.state(State::Recording, Some(trigger), "");
        self.emit(e);
    }

    fn session_tick(&mut self, started: Instant) {
        let spoken = SPOKEN.split(' ').count();
        let want = ((started.elapsed().as_millis() / 260) as usize).min(spoken);
        if want > self.words {
            self.words = want;
            if self.cfg.ui.show_partials {
                self.bus.emit(partial_after(want));
            }
        }
        if started.elapsed() > Duration::from_secs(12) {
            self.finish_session(Trigger::Ui);
        }
    }

    fn finish_session(&mut self, trigger: Trigger) {
        let Some(started) = self.session_started.take() else {
            return;
        };
        let words: Vec<&str> = SENTENCE.split(' ').collect();
        let text = words[..self.words.max(3).min(words.len())].join(" ");
        let refine = self.cfg.refine.provider != "off";
        let mut steps = vec![(
            0,
            Step::Emit(self.state(State::Transcribing, Some(trigger), "")),
        )];
        if refine {
            steps.push((
                420,
                Step::Emit(self.state(State::Refining, Some(trigger), "")),
            ));
        }
        steps.push((
            if refine { 520 } else { 380 },
            Step::Emit(self.state(State::Inserting, Some(trigger), "")),
        ));
        let mut r = result_event(&text, true, refine, if refine { 640 } else { 212 });
        if let Event::Result { timings, .. } = &mut r {
            timings.audio_ms = started.elapsed().as_millis() as u64;
        }
        steps.push((90, Step::Emit(r)));
        steps.push((40, Step::Emit(self.state(State::Idle, None, ""))));
        if self.looping && self.hold.is_none() {
            steps.push((5000, Step::End));
        }
        self.schedule(steps, Duration::ZERO);
    }

    fn command(&mut self, cmd: Command) {
        match cmd {
            Command::GetConfig => self.emit_config(),
            Command::SetConfig { patch } => self.patch(patch),
            Command::SetHandsfree { enabled } => {
                self.patch(serde_json::json!({ "handsfree": { "enabled": enabled } }));
                if self.state == State::Idle {
                    self.armed = enabled;
                    let e = self.state(State::Idle, None, "");
                    self.emit(e);
                }
            }
            Command::SetSecret { provider, key } => {
                let provider = ochre_core::secrets::canonical(&provider).to_string();
                self.secrets.insert(provider, key.is_some_and(|k| !k.is_empty()));
                self.emit_config();
            }
            Command::TestProvider { stage, provider } => {
                let local = matches!(provider.as_str(), "local" | "parakeet" | "whisper");
                let key = ochre_core::secrets::canonical(&provider);
                let ok = local || self.secrets.get(key).copied().unwrap_or(false);
                let message = match (ok, stage.as_str()) {
                    (true, "stt") => "Transcribed the test clip: “The quick brown fox jumps over the lazy dog.”".into(),
                    (true, _) => "Cleaned the test sentence: “Let’s meet Tuesday at 3 PM.”".into(),
                    (false, _) => format!("No API key saved for {provider}."),
                };
                let ms = if local { 184 } else { 412 };
                self.schedule(vec![(700, Step::Emit(Event::TestResult { stage, provider, ok, message, ms }))], Duration::ZERO);
            }
            Command::DownloadModel { name, .. } => {
                let total = if name.contains("whisper") || name.contains("large") { 1600 * MB } else { 505 * MB };
                let steps = (1..=25).map(|i| (140, Step::Emit(Event::Download { item: name.clone(), done: total * i / 25, total }))).collect();
                self.schedule(steps, Duration::ZERO);
            }
            Command::HistoryQuery { q, limit } => {
                let q = q.to_lowercase();
                let items = demo_history().into_iter().filter(|h| q.is_empty() || h.text.to_lowercase().contains(&q)).take(limit.max(1)).collect();
                self.emit(Event::History { items });
            }
            Command::Start if self.session_started.is_none() => self.begin_session(Trigger::Ui),
            Command::Toggle => {
                if self.session_started.is_some() {
                    self.finish_session(Trigger::Cli)
                } else {
                    self.begin_session(Trigger::Cli)
                }
            }
            Command::Stop => self.finish_session(Trigger::Ui),
            Command::Cancel => {
                let was_live = self.session_started.take().is_some() || self.state != State::Idle;
                self.timeline.retain(|(_, s)| matches!(s, Step::Emit(Event::Download { .. } | Event::TestResult { .. })));
                if was_live {
                    let e = self.state(State::Idle, Some(Trigger::Ui), "");
                    self.emit(e);
                }
                if self.looping && self.hold.is_none() {
                    self.schedule(vec![(5000, Step::End)], Duration::ZERO);
                }
            }
            Command::InsertText { text } => self.emit(Event::Notice {
                message: format!("Demo: {} characters would be typed into the focused app.", text.chars().count()),
            }),
            Command::TrainWake { word } => self.emit(Event::Notice {
                message: format!("Training “{word}” takes about 20 minutes on a GPU. Run `ochre train-wake --word {word}`."),
            }),
            Command::HistoryClear => self.emit(Event::History { items: Vec::new() }),
            Command::PasteLast => self.emit(Event::Notice {
                message: "Demo: the last transcript would be typed into the focused app.".into(),
            }),
            // Window, permission, device and download commands have nothing to script in the demo.
            _ => {}
        }
    }
}

// --------------------------------------------------------------------------------------------
// Fixtures

pub fn demo_config() -> Config {
    let mut c = Config::default();
    c.refine.provider = "local".into();
    c.dictionary.words = vec![
        "Parakeet".into(),
        "Ochre".into(),
        "Kubernetes".into(),
        "Priya Raman".into(),
    ];
    c.dictionary.replacements = [("oaker", "Ochre"), ("my email", "jordan@example.com")]
        .into_iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
    c.ui.onboarded = true;
    c
}

fn result_event(text: &str, inserted: bool, refined: bool, release_ms: u64) -> Event {
    Event::Result {
        id: format!("demo-{release_ms}"),
        raw: text.to_lowercase().replace([',', '.'], ""),
        text: text.into(),
        inserted,
        refined,
        timings: Timings {
            audio_ms: 5400,
            stt_tail_ms: 118,
            stt_total_ms: 690,
            refine_ms: if refined {
                release_ms.saturating_sub(200)
            } else {
                0
            },
            inject_ms: 14,
            release_to_insert_ms: release_ms,
        },
    }
}

pub fn demo_history() -> Vec<Entry> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let e = |id: i64, ago: f64, app: &str, text: &str, raw: &str, refiner: &str| Entry {
        id,
        created: now - ago,
        raw: raw.into(),
        text: text.into(),
        app: app.into(),
        stt: "parakeet".into(),
        refiner: refiner.into(),
        duration_ms: 4200,
        inserted: true,
        note: String::new(),
    };
    vec![
        e(
            41,
            90.0,
            "slack",
            "Sounds good, I can join the design review at two and bring the new HUD screenshots.",
            "uh sounds good I can join the design review at two and um bring the new HUD screenshots",
            "local",
        ),
        e(
            40,
            1300.0,
            "outlook",
            "Hi Sam, thanks for the notes. I'll send the deck tonight.",
            "hi sam thanks for the notes I'll send the deck tonight",
            "local",
        ),
        e(
            39,
            5400.0,
            "code",
            "TODO: coalesce level events for slow clients.",
            "todo coalesce level events for slow clients",
            "",
        ),
        e(
            38,
            86_400.0,
            "notion",
            "Parakeet is fast enough on CPU that release-to-text feels instant.",
            "parakeet is fast enough on cpu that release to text feels instant",
            "local",
        ),
        e(
            37,
            3.0 * 86_400.0,
            "chrome",
            "Can we move the sync to Thursday at 3 PM?",
            "can we move the sync to tuesday no wait thursday at three pm",
            "local",
        ),
    ]
}

/// The showcase: every HUD state once, ~45 s.
fn showcase() -> Vec<(u64, Step)> {
    let mut s: Vec<(u64, Step)> = Vec::new();
    let st = |state: State, trigger: Option<Trigger>, armed: bool, detail: &str| {
        Step::Emit(Event::State {
            state,
            trigger,
            handsfree_armed: armed,
            detail: detail.into(),
        })
    };
    // loading: download, then model load
    s.push((
        0,
        st(State::Loading, None, false, "Downloading speech model"),
    ));
    let total = 670 * MB;
    for i in 0..=20 {
        s.push((
            110,
            Step::Emit(Event::Download {
                item: "parakeet-tdt-0.6b-v3-int8".into(),
                done: total * i / 20,
                total,
            }),
        ));
        if i == 9 {
            s.push((0, Step::Label("download")));
        }
    }
    s.push((150, st(State::Loading, None, false, "Loading models…")));
    s.push((0, Step::Label("loading")));
    s.push((1300, st(State::Idle, None, false, "")));
    // hold to talk
    s.push((1200, st(State::Recording, Some(Trigger::Hotkey), false, "")));
    let (mut shown, mut wait) = (String::new(), 0);
    for i in 1..=SPOKEN.split(' ').count() {
        let e = partial_after(i);
        wait += 170;
        if let Event::Partial { text, .. } = &e {
            if *text == shown {
                continue; // a hesitation: nothing new to show
            }
            shown = text.clone();
        }
        s.push((std::mem::take(&mut wait), Step::Emit(e)));
        if i == 12 {
            s.push((0, Step::Label("recording")));
        }
    }
    s.push((
        400,
        st(State::Transcribing, Some(Trigger::Hotkey), false, ""),
    ));
    s.push((0, Step::Label("transcribing")));
    s.push((800, st(State::Refining, Some(Trigger::Hotkey), false, "")));
    s.push((0, Step::Label("refining")));
    s.push((1000, st(State::Inserting, Some(Trigger::Hotkey), false, "")));
    s.push((100, Step::Emit(result_event(SENTENCE, true, true, 212))));
    s.push((40, st(State::Idle, None, false, "")));
    s.push((0, Step::Label("inserted")));
    // double-tap: locked
    s.push((2600, st(State::Locked, Some(Trigger::Hotkey), false, "")));
    s.push((
        300,
        Step::Emit(Event::Partial {
            text: "Remind me to water the".into(),
            stable_chars: 15,
        }),
    ));
    s.push((0, Step::Label("locked")));
    s.push((
        900,
        Step::Emit(Event::Partial {
            text: "Remind me to water the plants on Friday".into(),
            stable_chars: 23,
        }),
    ));
    s.push((
        1400,
        st(State::Transcribing, Some(Trigger::Hotkey), false, ""),
    ));
    s.push((500, st(State::Inserting, Some(Trigger::Hotkey), false, "")));
    s.push((
        80,
        Step::Emit(result_event(
            "Remind me to water the plants on Friday.",
            true,
            false,
            188,
        )),
    ));
    s.push((40, st(State::Idle, None, false, "")));
    // hands-free: armed pip, a session, back to armed
    s.push((3200, st(State::Idle, None, true, "")));
    s.push((0, Step::Label("pip")));
    s.push((2600, st(State::Handsfree, Some(Trigger::Wake), true, "")));
    s.push((
        500,
        Step::Emit(Event::Partial {
            text: "Ship it after the tests pass".into(),
            stable_chars: 12,
        }),
    ));
    s.push((0, Step::Label("handsfree")));
    s.push((2200, st(State::Transcribing, Some(Trigger::Wake), true, "")));
    s.push((400, st(State::Inserting, Some(Trigger::Wake), true, "")));
    s.push((
        80,
        Step::Emit(result_event(
            "Ship it after the tests pass.",
            true,
            false,
            176,
        )),
    ));
    s.push((40, st(State::Idle, None, true, "")));
    // notice, then an error
    s.push((
        3000,
        Step::Emit(Event::Notice {
            message: "That window runs as administrator, so the text was saved to History instead."
                .into(),
        }),
    ));
    s.push((0, Step::Label("notice")));
    s.push((
        5500,
        Step::Emit(Event::Error {
            message: "Groq rejected the API key. Check it in Settings › Transcription.".into(),
            code: "auth".into(),
        }),
    ));
    s.push((
        0,
        st(State::Error, None, false, "Groq rejected the API key."),
    ));
    s.push((0, Step::Label("error")));
    s.push((6000, st(State::Idle, None, false, "")));
    s.push((0, Step::End));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_voice_sounds_like_speech() {
        let mut v = Voice::new(9);
        let xs: Vec<f32> = (0..180).map(|i| v.at(i as f32 * 33.0)).collect();
        assert!(xs.iter().all(|x| (0.0..=1.0).contains(x)));
        assert!(
            xs.iter().filter(|x| **x > 0.6).count() > 15,
            "loud syllables"
        );
        assert!(xs.iter().filter(|x| **x < 0.3).count() > 40, "quiet gaps");
    }

    #[test]
    fn every_hold_label_is_in_the_script() {
        let labels: Vec<&str> = showcase()
            .iter()
            .filter_map(|(_, s)| {
                if let Step::Label(l) = s {
                    Some(*l)
                } else {
                    None
                }
            })
            .collect();
        for l in LABELS {
            assert!(labels.contains(l), "missing label {l}");
        }
    }

    #[test]
    fn script_events_serialize() {
        for (_, s) in showcase() {
            if let Step::Emit(e) = s {
                let v = serde_json::to_value(&e).unwrap();
                assert!(v["event"].is_string());
            }
        }
    }

    #[test]
    fn demo_live_text_never_shows_hesitations() {
        let mut last = String::new();
        for i in 1..=SPOKEN.split(' ').count() {
            let Event::Partial { text, stable_chars } = partial_after(i) else {
                unreachable!()
            };
            assert!(!text.split(' ').any(|w| matches!(w, "um" | "uh")), "{text}");
            assert!(stable_chars <= text.encode_utf16().count());
            assert!(text.starts_with(last.as_str()), "{last:?} -> {text:?}");
            last = text;
        }
        assert_eq!(last, SENTENCE);
    }

    #[test]
    fn demo_answers_commands() {
        let bus = Bus::new();
        let seen = std::sync::Arc::new(Mutex::new(Vec::<Event>::new()));
        let s = seen.clone();
        bus.subscribe(move |e| s.lock().unwrap().push(e.clone()));
        let mut d = Driver::new(bus, None, false);
        d.command(Command::SetConfig {
            patch: serde_json::json!({ "refine": { "provider": "off" } }),
        });
        d.command(Command::HistoryQuery {
            q: "deck".into(),
            limit: 10,
        });
        d.command(Command::SetSecret {
            provider: "openai".into(),
            key: Some("sk-x".into()),
        });
        let seen = seen.lock().unwrap();
        assert!(seen.iter().any(
            |e| matches!(e, Event::Config { config, .. } if config["refine"]["provider"] == "off")
        ));
        assert!(
            seen.iter()
                .any(|e| matches!(e, Event::History { items } if items.len() == 1))
        );
        assert!(
            seen.iter()
                .any(|e| matches!(e, Event::Config { secrets, .. } if secrets["openai"]))
        );
    }
}
