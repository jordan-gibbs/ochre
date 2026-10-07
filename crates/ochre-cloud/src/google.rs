//! Google: Gemini 3.5 Transcribe through the Gemini API (public preview since 2026-08-26; it
//! replaces Chirp 3). Engine id `google`; the key is the Gemini API key (`secrets` alias
//! `google -> gemini`), so the refiner and the transcriber share it.
//!
//! - Batch, `gemini-3.5-transcribe` (default): `POST {v1beta}/models/gemini-3.5-transcribe:
//!   generateContent`, header `x-goog-api-key`, the phrase as `inlineData` WAV, and
//!   `generationConfig.audioTranscriptionConfig = {mode, languageCodes?, customVocabulary?}`.
//!   The transcript is the concatenated `candidates[0].content.parts[].text` (the SDK's
//!   `response.text`). ~$0.005/min (~$0.30/h).
//! - Streaming, `gemini-3.5-transcribe-live`: Live API WebSocket `BidiGenerateContent?key=`,
//!   `setup = {model, generationConfig.responseModalities: ["TEXT"], inputAudioTranscription,
//!   realtimeInputConfig.automaticActivityDetection.disabled: true}` (push-to-talk: we send
//!   `activityStart` on open and `activityEnd` on release), audio as base64
//!   `audio/pcm;rate=16000` in `realtimeInput.audio`. The server answers with
//!   `serverContent.interimInputTranscription` (partials) and `serverContent.inputTranscription`
//!   (finalized segments). ~$0.009/min (~$0.54/h). Sessions are capped at 10 minutes.
//!
//! `mode: "SMART"` (default here) removes fillers, stutters and false starts, resolves spoken
//! self-corrections, and formats numbers, dates and lists: most of what refinement does. VERBATIM
//! is the API default. Language codes are BCP-47 (`en-US`); without one the model auto-detects.
//! Docs: ai.google.dev/gemini-api/docs/generate-content/transcribe and
//! ai.google.dev/gemini-api/docs/live-api/live-transcribe.

use std::time::{Duration, Instant};

use ochre_core::config::SttConfig;
use ochre_core::events::EngineInfo;
use ochre_core::stt::{PartialSink, ProgressFn, SttEngine, SttOptions, SttResult, SttStream};
use ochre_core::{Error, Result, SAMPLE_RATE};
use serde_json::{Value, json};

use crate::common::{self, Core};
use crate::ws::{self, Read, Ws};

pub const API: &str = "https://generativelanguage.googleapis.com/v1beta";
pub const WS_URL: &str = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent";
pub const BATCH_MODEL: &str = "gemini-3.5-transcribe";
pub const LIVE_MODEL: &str = "gemini-3.5-transcribe-live";
/// "SMART" (cleanup + formatting) or "VERBATIM".
pub const DEFAULT_MODE: &str = "SMART";
const PROVIDER: &str = "google";
/// Docs: up to 1,000 terms; best results with up to ~100.
const MAX_TERMS: usize = 100;
/// ~100 ms of audio per Live frame (the docs recommend 100 ms chunks).
const FRAME: usize = SAMPLE_RATE as usize / 10;
/// After `activityEnd`, once a finalized segment has arrived, this much quiet ends the wait if the
/// server sends no `turnComplete`.
const SETTLE: Duration = Duration::from_millis(350);

pub fn info() -> EngineInfo {
    EngineInfo {
        id: PROVIDER.into(),
        label: "Google (Gemini 3.5 Transcribe)".into(),
        kind: "cloud".into(),
        models: vec![BATCH_MODEL.into(), LIVE_MODEL.into()],
        default_model: BATCH_MODEL.into(),
        needs_key: true,
        note: "~$0.30/h batch; ~$0.54/h streaming (preview); removes fillers itself".into(),
        languages: "85+ languages".into(),
    }
}

/// BCP-47 for the Gemini API: a code with a region passes through; a bare ISO-639-1 code gets its
/// most common supported locale; unknown codes, "auto" or none mean auto-detect.
pub fn bcp47(lang: Option<&str>) -> Option<String> {
    let l = lang?.trim();
    if l.is_empty() || l.eq_ignore_ascii_case("auto") {
        return None;
    }
    if l.contains(['-', '_']) {
        return Some(l.replace('_', "-"));
    }
    let tag = match l.to_lowercase().as_str() {
        "en" => "en-US",
        "es" => "es-US",
        "fr" => "fr-FR",
        "de" => "de-DE",
        "it" => "it-IT",
        "pt" => "pt-BR",
        "nl" => "nl-NL",
        "ja" => "ja-JP",
        "ko" => "ko-KR",
        "zh" => "cmn-Hans-CN",
        "hi" => "hi-IN",
        "ru" => "ru-RU",
        "pl" => "pl-PL",
        "sv" => "sv-SE",
        "da" => "da-DK",
        "fi" => "fi-FI",
        "nb" | "no" => "nb-NO",
        "tr" => "tr-TR",
        "uk" => "uk-UA",
        "cs" => "cs-CZ",
        "el" => "el-GR",
        "he" => "he-IL",
        "id" => "id-ID",
        "vi" => "vi-VN",
        "th" => "th-TH",
        "ro" => "ro-RO",
        "hu" => "hu-HU",
        _ => return None,
    };
    Some(tag.into())
}

/// `audioTranscriptionConfig` (batch) / `inputAudioTranscription` (Live): same fields.
pub fn transcription_config(mode: &str, lang: Option<&str>, vocab: &[String]) -> Value {
    let mut c = json!({"mode": mode});
    if let Some(l) = bcp47(lang) {
        c["languageCodes"] = json!([l]);
    }
    let t = common::terms(vocab, MAX_TERMS, 100);
    if !t.is_empty() {
        c["customVocabulary"] = json!(t);
    }
    c
}

pub struct Google {
    pub core: Core,
    base: String,
    ws_url: String,
    pub mode: String,
}

impl Google {
    pub fn new(cfg: &SttConfig) -> Self {
        Self {
            core: Core::new(PROVIDER, BATCH_MODEL, cfg),
            base: API.into(),
            ws_url: WS_URL.into(),
            mode: DEFAULT_MODE.into(),
        }
    }
    pub fn with_base(mut self, base: &str) -> Self {
        self.base = base.trim_end_matches('/').to_string();
        self
    }
    pub fn with_ws_url(mut self, url: &str) -> Self {
        self.ws_url = url.to_string();
        self
    }
    pub fn with_key(mut self, key: &str) -> Self {
        self.core.key_override = Some(key.into());
        self
    }

    fn is_live(&self) -> bool {
        self.core.model.ends_with("-live")
    }

    /// Per-call language wins; keeps the region (unlike `Core::lang`).
    fn language(&self, call: Option<&str>) -> Option<String> {
        call.map(str::to_string)
            .or_else(|| self.core.language.clone())
    }

    pub fn batch_body(&self, pcm: &[f32], opts: &SttOptions) -> Value {
        json!({
            "contents": [{"role": "user", "parts": [
                {"inlineData": {"mimeType": "audio/wav", "data": common::base64(&common::wav(pcm))}}
            ]}],
            "generationConfig": {"audioTranscriptionConfig": transcription_config(
                &self.mode, self.language(opts.language.as_deref()).as_deref(), &opts.vocabulary)},
        })
    }

    pub fn setup(&self, opts: &SttOptions) -> Value {
        json!({"setup": {
            "model": format!("models/{}", self.core.model),
            "generationConfig": {"responseModalities": ["TEXT"]},
            "realtimeInputConfig": {"automaticActivityDetection": {"disabled": true}},
            "inputAudioTranscription": transcription_config(
                &self.mode, self.language(opts.language.as_deref()).as_deref(), &opts.vocabulary),
        }})
    }

    fn open_live(&self, opts: &SttOptions, on_partial: Option<PartialSink>) -> Result<GeminiLive> {
        let key = self.core.key()?;
        let sep = if self.ws_url.contains('?') { '&' } else { '?' };
        GeminiLive::open(
            &format!("{}{sep}key={key}", self.ws_url),
            &self.setup(opts),
            self.core.timeout,
            on_partial,
        )
    }
}

/// Plain text of a generateContent response: every non-thought text part, in order. If the model
/// returned only word annotations, the words joined by spaces.
pub fn response_text(body: &Value) -> Result<String> {
    let Some(cand) = body["candidates"].get(0) else {
        let reason = body["promptFeedback"]["blockReason"]
            .as_str()
            .unwrap_or("empty response");
        return Err(Error::Other(format!("google: no candidate ({reason})")));
    };
    let parts = cand["content"]["parts"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let text: String = parts
        .iter()
        .filter(|p| p["thought"] != true)
        .filter_map(|p| p["text"].as_str())
        .collect();
    if !text.trim().is_empty() {
        return Ok(text.trim().to_string());
    }
    let words: Vec<&str> = parts
        .iter()
        .flat_map(|p| {
            p["audioTranscription"]["words"]
                .as_array()
                .into_iter()
                .flatten()
        })
        .filter_map(|w| w["word"].as_str())
        .collect();
    Ok(words.join(" "))
}

impl SttEngine for Google {
    fn info(&self) -> EngineInfo {
        let mut i = info();
        i.default_model = self.core.model.clone();
        i
    }

    fn load(&mut self, _progress: ProgressFn) -> Result<()> {
        let key = self.core.key()?;
        self.core.prewarm(
            &format!("{}/models/{}", self.base, self.core.model),
            &[("x-goog-api-key", key)],
        )
    }

    fn transcribe(&self, pcm: &[f32], opts: &SttOptions) -> Result<SttResult> {
        let t0 = Instant::now();
        if self.is_live() {
            let mut s = Box::new(self.open_live(opts, None)?);
            for chunk in pcm.chunks(FRAME) {
                s.send(chunk)?;
            }
            let mut r = s.finish()?;
            r.processing_ms = t0.elapsed().as_millis() as u64;
            return Ok(r);
        }
        let key = self.core.key()?;
        let req = self
            .core
            .client
            .post(format!(
                "{}/models/{}:generateContent",
                self.base, self.core.model
            ))
            .header("x-goog-api-key", key)
            .json(&self.batch_body(pcm, opts));
        let body = self.core.send(req, self.core.timeout)?;
        Ok(SttResult {
            text: response_text(&body)?,
            duration_ms: common::duration_ms(pcm),
            processing_ms: t0.elapsed().as_millis() as u64,
            language: self.language(opts.language.as_deref()),
        })
    }

    fn streaming(&self) -> bool {
        self.is_live()
    }

    fn stream(
        &self,
        opts: &SttOptions,
        on_partial: Option<PartialSink>,
    ) -> Option<Result<Box<dyn SttStream>>> {
        self.is_live().then(|| {
            self.open_live(opts, on_partial)
                .map(|s| Box::new(s) as Box<dyn SttStream>)
        })
    }

    fn prewarm(&self) {
        if let Ok(key) = self.core.key() {
            self.core.prewarm_async(
                format!("{}/models/{}", self.base, self.core.model),
                vec![("x-goog-api-key", key)],
            );
        }
    }
}

// ---------------------------------------------------------------- Live API stream

#[derive(Default)]
struct Heard {
    setup_done: bool,
    finals: Vec<String>,
    interim: String,
    turn_complete: bool,
    ended: bool,
    last_final: Option<Instant>,
}

impl Heard {
    fn text(&self) -> String {
        self.finals
            .iter()
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn handle(&mut self, r: Read, on_partial: &mut Option<PartialSink>) -> Result<bool> {
        let text = match r {
            Read::Json(t) => t,
            Read::Closed { code, reason } => {
                if self.ended && (self.last_final.is_some() || reason.is_empty()) {
                    self.turn_complete = true;
                    return Ok(true);
                }
                return Err(close_error(code, &reason));
            }
        };
        let msg: Value = serde_json::from_str(text).map_err(|_| common::bad_response(PROVIDER))?;
        if msg.get("setupComplete").is_some() {
            self.setup_done = true;
            return Ok(true);
        }
        if let Some(err) = msg.get("error") {
            let status = err["code"].as_u64().unwrap_or(500) as u16;
            return Err(common::status_error(PROVIDER, status, &err.to_string()));
        }
        let sc = &msg["serverContent"];
        let mut changed = false;
        if let Some(t) = sc["interimInputTranscription"]["text"].as_str() {
            self.interim = t.to_string();
            changed = true;
        }
        if let Some(t) = sc["inputTranscription"]["text"].as_str() {
            self.finals.push(t.to_string());
            self.interim.clear();
            self.last_final = Some(Instant::now());
            changed = true;
        }
        if changed && let Some(cb) = on_partial.as_mut() {
            let stable = self.text();
            let full = if self.interim.trim().is_empty() {
                stable.clone()
            } else if stable.is_empty() {
                self.interim.trim().to_string()
            } else {
                format!("{stable} {}", self.interim.trim())
            };
            cb(&full, stable.len());
        }
        if sc["turnComplete"] == true && self.ended {
            self.turn_complete = true;
            return Ok(true);
        }
        // After release, hand control back on every finalized segment so `finish` can switch
        // to the short settle wait.
        Ok(self.ended && sc["inputTranscription"]["text"].is_string())
    }
}

fn close_error(code: u16, reason: &str) -> Error {
    let reason = common::redact(reason);
    let lower = reason.to_lowercase();
    if lower.contains("api key") || lower.contains("permission") || lower.contains("unauthen") {
        Error::Auth {
            provider: PROVIDER.into(),
        }
    } else if lower.contains("quota") || lower.contains("resource exhausted") {
        Error::Quota {
            provider: PROVIDER.into(),
        }
    } else if lower.contains("not found") || lower.contains("not supported") {
        Error::Config(format!("{PROVIDER}: {reason}"))
    } else {
        Error::Network {
            provider: PROVIDER.into(),
            message: format!("closed ({code}) {reason}"),
        }
    }
}

/// One Live API transcription session for one dictation (manual activity detection).
pub struct GeminiLive {
    ws: Ws,
    timeout: Duration,
    pending: Vec<f32>,
    samples: u64,
    heard: Heard,
    on_partial: Option<PartialSink>,
}

impl GeminiLive {
    pub fn open(
        url: &str,
        setup: &Value,
        timeout: Duration,
        on_partial: Option<PartialSink>,
    ) -> Result<Self> {
        let req = ws::request(url, &[])?;
        let mut ws = ws::connect(PROVIDER, req, timeout)?;
        ws::send_text(&mut ws, PROVIDER, timeout, setup.to_string())?;
        let mut s = Self {
            ws,
            timeout,
            pending: Vec::with_capacity(FRAME * 2),
            samples: 0,
            heard: Heard::default(),
            on_partial,
        };
        // The session takes input only after `setupComplete` (one round trip, paid at key-down).
        let deadline = Instant::now() + timeout;
        while !s.heard.setup_done {
            let left = ws::left(deadline, timeout)?;
            s.pump(Some(left))?;
        }
        s.text(json!({"realtimeInput": {"activityStart": {}}}))?;
        Ok(s)
    }

    fn text(&mut self, v: Value) -> Result<()> {
        ws::send_text(&mut self.ws, PROVIDER, self.timeout, v.to_string())
    }

    fn audio(&mut self, pcm: &[f32]) -> Result<()> {
        if pcm.is_empty() {
            return Ok(());
        }
        self.text(json!({"realtimeInput": {"audio": {
            "data": common::base64(&common::pcm16(pcm)), "mimeType": "audio/pcm;rate=16000"}}}))
    }

    fn pump(&mut self, wait: Option<Duration>) -> Result<bool> {
        let (heard, cb) = (&mut self.heard, &mut self.on_partial);
        ws::drain(&mut self.ws, PROVIDER, self.timeout, wait, |m| {
            heard.handle(m, cb)
        })
    }
}

impl SttStream for GeminiLive {
    fn send(&mut self, pcm: &[f32]) -> Result<()> {
        self.samples += pcm.len() as u64;
        self.pending.extend_from_slice(pcm);
        if self.pending.len() >= FRAME {
            let chunk = std::mem::take(&mut self.pending);
            self.audio(&chunk)?;
        }
        self.pump(None).map(|_| ())
    }

    fn finish(mut self: Box<Self>) -> Result<SttResult> {
        let t_release = Instant::now();
        let chunk = std::mem::take(&mut self.pending);
        self.audio(&chunk)?;
        self.text(json!({"realtimeInput": {"activityEnd": {}}}))?;
        self.heard.ended = true;
        let deadline = t_release + self.timeout;
        while !self.heard.turn_complete {
            // Without a turnComplete, a finalized segment followed by SETTLE of quiet ends it.
            let wait = match self.heard.last_final {
                Some(t) if t >= t_release => {
                    let quiet_until = t + SETTLE;
                    match quiet_until.checked_duration_since(Instant::now()) {
                        Some(d) if !d.is_zero() => d,
                        _ => break,
                    }
                }
                _ => match ws::left(deadline, self.timeout) {
                    Ok(d) => d,
                    // Timed out: keep whatever was finalized during capture, if anything.
                    Err(e) if self.heard.finals.is_empty() => return Err(e),
                    Err(_) => break,
                },
            };
            match self.pump(Some(wait)) {
                Ok(_) => {}
                Err(Error::Timeout(_)) => {} // read timeout: re-check the clocks above
                Err(e) => return Err(e),
            }
        }
        ws::close(&mut self.ws);
        Ok(SttResult {
            text: self.heard.text(),
            duration_ms: self.samples * 1000 / SAMPLE_RATE as u64,
            processing_ms: t_release.elapsed().as_millis() as u64,
            language: None,
        })
    }

    fn cancel(mut self: Box<Self>) {
        ws::close(&mut self.ws);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{MockServer, Reply};
    use crate::ws::mock::MockWs;
    use std::sync::{Arc, Mutex};

    fn cfg(model: &str) -> SttConfig {
        SttConfig {
            engine: "google".into(),
            model: model.into(),
            language: Some("en".into()),
            ..Default::default()
        }
    }

    #[test]
    fn languages_and_config() {
        assert_eq!(bcp47(Some("en")).as_deref(), Some("en-US"));
        assert_eq!(bcp47(Some("en_GB")).as_deref(), Some("en-GB"));
        assert_eq!(bcp47(Some("auto")), None);
        assert_eq!(bcp47(Some("xx")), None);
        assert_eq!(
            transcription_config("SMART", Some("de"), &["Kubernetes, BigQuery".into()]),
            json!({"mode": "SMART", "languageCodes": ["de-DE"], "customVocabulary": ["Kubernetes", "BigQuery"]})
        );
        assert_eq!(
            transcription_config("VERBATIM", None, &[]),
            json!({"mode": "VERBATIM"})
        );
    }

    #[test]
    fn batch_request_shape() {
        let srv = MockServer::start(vec![Reply::json(
            200,
            json!({"candidates": [{"content": {"role": "model", "parts": [{"text": "Let's meet on "}, {"text": "Wednesday at 2:00 PM."}]}, "finishReason": "STOP"}]}),
        )]);
        let e = Google::new(&cfg(""))
            .with_base(&srv.url("/v1beta"))
            .with_key("test-key");
        assert!(!e.streaming());
        let r = e
            .transcribe(
                &vec![0.0; 16_000],
                &SttOptions {
                    language: None,
                    vocabulary: vec!["Kubernetes".into()],
                },
            )
            .unwrap();
        assert_eq!(r.text, "Let's meet on Wednesday at 2:00 PM.");
        assert_eq!(r.duration_ms, 1000);
        let req = &srv.requests()[0];
        assert_eq!(
            req.path,
            "/v1beta/models/gemini-3.5-transcribe:generateContent"
        );
        assert_eq!(req.header("x-goog-api-key"), Some("test-key"));
        let b: Value = serde_json::from_slice(&req.body).unwrap();
        let part = &b["contents"][0]["parts"][0]["inlineData"];
        assert_eq!(part["mimeType"], "audio/wav");
        assert!(part["data"].as_str().unwrap().starts_with("UklGR")); // "RIFF"
        assert_eq!(
            b["generationConfig"]["audioTranscriptionConfig"],
            json!({"mode": "SMART", "languageCodes": ["en-US"], "customVocabulary": ["Kubernetes"]})
        );
    }

    #[test]
    fn batch_errors_and_word_fallback() {
        for (status, auth) in [(403, true), (429, false)] {
            let srv = MockServer::start(vec![Reply::json(
                status,
                json!({"error": {"code": status, "message": "API key not valid. AIzaSyAbcdefghijklmnopqrstuvwx", "status": "PERMISSION_DENIED"}}),
            )]);
            let e = Google::new(&cfg(""))
                .with_base(&srv.url("/v1beta"))
                .with_key("k");
            let err = e
                .transcribe(&[0.0; 160], &SttOptions::default())
                .unwrap_err();
            assert_eq!(matches!(err, Error::Auth { .. }), auth, "{err:?}");
            assert!(err.is_fallback_worthy());
        }
        let words = json!({"candidates": [{"content": {"parts": [{"audioTranscription": {"words": [{"word": "Hello"}, {"word": "world"}]}}]}}]});
        assert_eq!(response_text(&words).unwrap(), "Hello world");
        assert!(response_text(&json!({"promptFeedback": {"blockReason": "OTHER"}})).is_err());
    }

    fn live_server(turn_complete: bool) -> MockWs {
        MockWs::start(
            vec![],
            Box::new(move |f: &str| {
                let v: Value = serde_json::from_str(f).unwrap();
                if v.get("setup").is_some() {
                    return vec![json!({"setupComplete": {}}).to_string()];
                }
                let ri = &v["realtimeInput"];
                if ri.get("audio").is_some() {
                    return vec![
                        json!({"serverContent": {"interimInputTranscription": {"text": "so um"}}})
                            .to_string(),
                    ];
                }
                if ri.get("activityEnd").is_some() {
                    let mut out = vec![
                        json!({"serverContent": {"inputTranscription": {"text": "So, let's meet on"}}}).to_string(),
                        json!({"serverContent": {"inputTranscription": {"text": "Wednesday."}}}).to_string(),
                    ];
                    if turn_complete {
                        out.push(json!({"serverContent": {"turnComplete": true}}).to_string());
                    }
                    return out;
                }
                vec![]
            }),
        )
    }

    #[test]
    fn live_stream_shape() {
        for turn_complete in [true, false] {
            let srv = live_server(turn_complete);
            let e = Google::new(&cfg(LIVE_MODEL))
                .with_ws_url(&srv.url("/ws/BidiGenerateContent"))
                .with_key("test-key");
            assert!(e.streaming());
            let seen = Arc::new(Mutex::new(Vec::<String>::new()));
            let s2 = seen.clone();
            let mut s = e
                .stream(
                    &SttOptions::default(),
                    Some(Box::new(move |t: &str, _| {
                        s2.lock().unwrap().push(t.into())
                    })),
                )
                .unwrap()
                .unwrap();
            for _ in 0..6 {
                s.send(&[0.0; 800]).unwrap();
                std::thread::sleep(Duration::from_millis(5));
            }
            let t0 = Instant::now();
            let r = s.finish().unwrap();
            assert_eq!(r.text, "So, let's meet on Wednesday.");
            assert_eq!(r.duration_ms, 300);
            let bound = if turn_complete { 300 } else { 2000 };
            assert!(
                t0.elapsed() < Duration::from_millis(bound),
                "{:?}",
                t0.elapsed()
            );
            assert!(srv.path.lock().unwrap().ends_with("?key=test-key"));
            let frames: Vec<Value> = srv
                .frames()
                .iter()
                .map(|f| serde_json::from_str(f).unwrap())
                .collect();
            assert_eq!(
                frames[0],
                json!({"setup": {
                    "model": "models/gemini-3.5-transcribe-live",
                    "generationConfig": {"responseModalities": ["TEXT"]},
                    "realtimeInputConfig": {"automaticActivityDetection": {"disabled": true}},
                    "inputAudioTranscription": {"mode": "SMART", "languageCodes": ["en-US"]}}})
            );
            assert_eq!(frames[1], json!({"realtimeInput": {"activityStart": {}}}));
            assert_eq!(
                frames.last().unwrap(),
                &json!({"realtimeInput": {"activityEnd": {}}})
            );
            let audio: Vec<&Value> = frames
                .iter()
                .filter(|f| f["realtimeInput"].get("audio").is_some())
                .collect();
            assert!(!audio.is_empty());
            assert_eq!(
                audio[0]["realtimeInput"]["audio"]["mimeType"],
                "audio/pcm;rate=16000"
            );
            assert!(seen.lock().unwrap().iter().any(|t| t == "so um"));
        }
    }

    #[test]
    fn close_reasons_map() {
        assert!(matches!(
            close_error(1008, "API key not valid. Please pass a valid API key."),
            Error::Auth { .. }
        ));
        assert!(close_error(1011, "Internal error").is_fallback_worthy());
    }
}
