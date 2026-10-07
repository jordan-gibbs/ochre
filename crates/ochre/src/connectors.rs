//! Cloud connectors: pick one provider, paste one key, and both stages (speech-to-text and
//! refinement) get fast defaults. The settings UI receives this table as `Event::Connectors` and
//! applies a choice with one `set_config` patch ([`patch`]); per-stage mix-and-match stays
//! available under "Advanced".
//!
//! Model choices and prices are from the providers' docs (Oct 2026); latencies are in
//! docs/refinement.md Â§6.1 / Â§7 and SPEC Â§4.3. Cost estimates assume ~400 dictations per hour of
//! dictated audio, each ~600 prompt and ~60 output tokens for refinement (docs/refinement.md Â§8).

use ochre_core::events::{ConnectorInfo, Event, StageChoice};

/// Dictations per hour of dictated audio, for the refinement share of the estimate.
const DICTATIONS_PER_HOUR: f64 = 400.0;
const PROMPT_TOKENS: f64 = 600.0;
const OUTPUT_TOKENS: f64 = 60.0;

/// USD per hour of refinement for a model priced `input` / `output` USD per 1M tokens.
fn refine_per_hour(input: f64, output: f64) -> f64 {
    DICTATIONS_PER_HOUR * (PROMPT_TOKENS * input + OUTPUT_TOKENS * output) / 1e6
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

struct Row {
    id: &'static str,
    label: &'static str,
    key: &'static str,
    extra: &'static [&'static str],
    stt: (&'static str, &'static str),
    refine: (&'static str, &'static str, &'static str),
    /// USD per hour of audio for speech-to-text.
    stt_per_hour: f64,
    /// USD per 1M tokens (input, output) for the refiner.
    refine_price: (f64, f64),
    notes: &'static str,
}

const ROWS: &[Row] = &[
    Row {
        id: "openai",
        label: "OpenAI",
        key: "openai",
        extra: &[],
        // gpt-live-transcribe streams while you talk: 0.46-0.68 s release -> final measured, vs
        // 0.66-1.94 s for batch gpt-transcribe on the same 7.3 s clip.
        stt: ("openai", "gpt-live-transcribe"),
        refine: ("openai", "gpt-6-luna", "clean"),
        stt_per_hour: 0.017 * 60.0,
        refine_price: (0.10, 0.50),
        notes: "Streams with gpt-live-transcribe and cleans up with GPT-6 Luna (no reasoning). \
                For about a third of the price, pick gpt-transcribe under Advanced (batch, \
                slower release).",
    },
    Row {
        id: "google",
        label: "Google",
        key: "gemini",
        extra: &[],
        stt: ("google", "gemini-3.5-transcribe"),
        // Gemini 3.5 Transcribe (SMART mode) already removes fillers, resolves self-corrections
        // and formats numbers and dates, so refinement stays on but light ("clean", never
        // "polish"): it mostly catches what the transcriber missed.
        refine: ("gemini", "gemini-3.5-flash-lite", "clean"),
        stt_per_hour: 0.005 * 60.0,
        refine_price: (0.30, 2.50),
        notes: "Gemini 3.5 Transcribe (preview) cleans up as it transcribes; Gemini 3.5 \
                Flash-Lite adds a light pass. One Gemini API key from Google AI Studio.",
    },
    Row {
        id: "groq",
        label: "Groq",
        key: "groq",
        extra: &[],
        stt: ("groq", "whisper-large-v3-turbo"),
        refine: ("groq", "openai/gpt-oss-20b", "clean"),
        // $0.04/h list, but every request bills at least 10 s and phrases are often shorter.
        stt_per_hour: 0.10,
        refine_price: (0.075, 0.30),
        notes: "Cheapest: Whisper large-v3-turbo and gpt-oss-20b on Groq.",
    },
    Row {
        id: "soniox+openai",
        label: "Soniox + OpenAI",
        key: "soniox",
        extra: &["openai"],
        // Soniox real-time: 110-166 ms release -> final measured.
        stt: ("soniox", ""),
        refine: ("openai", "gpt-6-luna", "clean"),
        stt_per_hour: 0.12,
        refine_price: (0.10, 0.50),
        notes: "Fastest release: Soniox streams while you talk; GPT-5.6 Luna cleans up. Needs a \
                Soniox key and an OpenAI key.",
    },
    Row {
        id: "local",
        label: "Local only",
        key: "",
        extra: &[],
        stt: ("parakeet", ""),
        refine: ("local", "", "clean"),
        stt_per_hour: 0.0,
        refine_price: (0.0, 0.0),
        notes: "Everything runs on this computer; audio never leaves it.",
    },
];

/// The connector catalog, in display order.
pub fn all() -> Vec<ConnectorInfo> {
    ROWS.iter()
        .map(|r| ConnectorInfo {
            id: r.id.into(),
            label: r.label.into(),
            key_provider: r.key.into(),
            extra_keys: r.extra.iter().map(|k| k.to_string()).collect(),
            stt: StageChoice {
                id: r.stt.0.into(),
                model: r.stt.1.into(),
                mode: String::new(),
            },
            refine: StageChoice {
                id: r.refine.0.into(),
                model: r.refine.1.into(),
                mode: r.refine.2.into(),
            },
            notes: r.notes.split_whitespace().collect::<Vec<_>>().join(" "),
            est_cost_per_hour: round2(
                r.stt_per_hour + refine_per_hour(r.refine_price.0, r.refine_price.1),
            ),
        })
        .collect()
}

pub fn get(id: &str) -> Option<ConnectorInfo> {
    all().into_iter().find(|c| c.id == id)
}

/// The `Connectors` event the shell emits at startup.
pub fn event() -> Event {
    Event::Connectors { items: all() }
}

/// The `set_config` patch that applies a connector to both stages.
pub fn patch(c: &ConnectorInfo) -> serde_json::Value {
    serde_json::json!({
        "stt": {"engine": c.stt.id, "model": c.stt.model},
        "refine": {"provider": c.refine.id, "model": c.refine.model, "mode": c.refine.mode},
    })
}

/// Which connector a config matches ("custom" when the stages were mixed by hand). Models match
/// when equal, or when either side is "" (the engine default). A connector with refinement
/// switched off is still that connector. Mirrored by `connectorId()` in app/ui/settings.
pub fn detect(cfg: &ochre_core::config::Config) -> String {
    let same = |a: &str, b: &str| a == b || a.is_empty() || b.is_empty();
    let refine_off = cfg.refine.provider == "off";
    all()
        .into_iter()
        .find(|c| {
            cfg.stt.engine == c.stt.id
                && same(&cfg.stt.model, &c.stt.model)
                && (refine_off
                    || (cfg.refine.provider == c.refine.id
                        && same(&cfg.refine.model, &c.refine.model)))
        })
        .map(|c| c.id)
        .unwrap_or_else(|| "custom".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ochre_core::config::Config;

    #[test]
    fn table() {
        let ids: Vec<String> = all().into_iter().map(|c| c.id).collect();
        assert_eq!(ids, ["openai", "google", "groq", "soniox+openai", "local"]);
        let openai = get("openai").unwrap();
        assert_eq!(
            (openai.stt.id.as_str(), openai.stt.model.as_str()),
            ("openai", "gpt-live-transcribe")
        );
        assert_eq!(openai.refine.model, "gpt-6-luna");
        // gpt-live-transcribe $1.02/h + GPT-6 Luna 400 x (600 x 0.1 + 60 x 0.5) / 1M = $0.036/h.
        assert_eq!(openai.est_cost_per_hour, 1.06);
        let google = get("google").unwrap();
        assert_eq!(google.key_provider, "gemini");
        assert_eq!(
            (google.refine.id.as_str(), google.refine.mode.as_str()),
            ("gemini", "clean")
        );
        assert_eq!(google.est_cost_per_hour, 0.43);
        assert_eq!(get("soniox+openai").unwrap().extra_keys, ["openai"]);
        assert_eq!(get("local").unwrap().est_cost_per_hour, 0.0);
        assert!(all().iter().all(|c| !c.notes.contains("  ")));
    }

    #[test]
    fn patch_round_trips_through_detect() {
        let base = Config::default();
        // parakeet + refinement off: everything local, which is the "Local only" connector.
        assert_eq!(detect(&base), "local");
        for c in all() {
            let cfg = base.patched(&patch(&c)).unwrap();
            assert_eq!(cfg.stt.engine, c.stt.id);
            assert_eq!(cfg.refine.mode, c.refine.mode);
            assert_eq!(detect(&cfg), c.id, "{}", c.id);
        }
        let mixed = base
            .patched(
                &serde_json::json!({"stt": {"engine": "openai"}, "refine": {"provider": "gemini"}}),
            )
            .unwrap();
        assert_eq!(detect(&mixed), "custom");
        let google_raw = base
            .patched(&patch(&get("google").unwrap()))
            .unwrap()
            .patched(&serde_json::json!({"refine": {"provider": "off"}}))
            .unwrap();
        assert_eq!(detect(&google_raw), "google");
    }

    #[test]
    fn wire_format() {
        let v = serde_json::to_value(event()).unwrap();
        assert_eq!(v["event"], "connectors");
        assert_eq!(v["items"][0]["stt"]["id"], "openai");
        assert_eq!(v["items"][1]["refine"]["mode"], "clean");
        let back: Event = serde_json::from_value(v).unwrap();
        assert_eq!(back, event());
    }
}
