//! OpenAI Realtime transcription session (`gpt-live-transcribe`) over a WebSocket: audio streams
//! while the user talks, and release waits only for the final transcript of the committed turn.
//!
//! Protocol (developers.openai.com/api/docs/guides/realtime-transcription, Oct 2026):
//! - connect `wss://api.openai.com/v1/realtime?intent=transcription`, `Authorization: Bearer`;
//! - `session.update` with `session.type = "transcription"`, `audio.input.format =
//!   {"type": "audio/pcm", "rate": 24000}` (the only PCM rate a transcription session takes, so we
//!   upsample from 16 kHz), `audio.input.transcription = {model, languages?, keywords?, delay}`, and
//!   `turn_detection: null` (gpt-live-transcribe supports neither server_vad nor semantic_vad);
//! - `input_audio_buffer.append` with base64 PCM16 (we batch ~100 ms per frame);
//! - on release, `input_audio_buffer.commit`, answered by
//!   `conversation.item.input_audio_transcription.delta` events and one `...completed` with the
//!   final `transcript`;
//! - failures arrive as `{"type": "error", "error": {"type", "code", "message"}}`.
//!
//! `gpt-live-transcribe` takes `languages` (a list), never the singular `language`. Keywords may
//! not contain `<`, `>`, CR or LF. Billing: $0.017 per minute of audio, while the session is open.

use std::time::{Duration, Instant};

use ochre_core::stt::{PartialSink, SttResult, SttStream};
use ochre_core::{Error, Result, SAMPLE_RATE};
use serde_json::{Value, json};

use crate::common::{self, Upsampler};
use crate::ws::{self, Read, Ws};

pub const LIVE_MODEL: &str = "gpt-live-transcribe";
/// `delay` trades partial-text latency for accuracy: minimal | low | medium | high | xhigh.
/// On release we commit and wait for the final either way; see docs/refinement.md §6.1 for the
/// measured release -> final latency per level.
pub const DEFAULT_DELAY: &str = "low";
const PROVIDER: &str = "openai";
/// ~100 ms of 16 kHz audio per `append` frame.
const FRAME: usize = SAMPLE_RATE as usize / 10;
/// The server rejects a commit of less than 100 ms of audio.
const MIN_COMMIT: usize = SAMPLE_RATE as usize / 10;

/// The Realtime transcription URL for an OpenAI-style REST base (`https://api.openai.com/v1`).
pub fn ws_url(base: &str) -> String {
    let b = base.trim_end_matches('/');
    let b = b
        .strip_prefix("https://")
        .map(|r| format!("wss://{r}"))
        .or_else(|| b.strip_prefix("http://").map(|r| format!("ws://{r}")))
        .unwrap_or_else(|| b.to_string());
    format!("{b}/realtime?intent=transcription")
}

/// The `session.update` that configures one dictation.
pub fn session_update(model: &str, lang: Option<&str>, vocab: &[String], delay: &str) -> Value {
    let mut tr = json!({"model": model});
    if model == LIVE_MODEL {
        tr["delay"] = json!(delay);
        if let Some(l) = lang {
            tr["languages"] = json!([l]);
        }
    } else if let Some(l) = lang {
        tr["language"] = json!(l);
    }
    let kw: Vec<String> = common::terms(vocab, 100, 100)
        .into_iter()
        .filter(|t| !t.contains(['<', '>', '\r', '\n']))
        .collect();
    if !kw.is_empty() {
        tr["keywords"] = json!(kw);
    }
    json!({
        "type": "session.update",
        "session": {
            "type": "transcription",
            "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": tr,
                "turn_detection": null,
            }},
        },
    })
}

fn api_error(err: &Value) -> Error {
    let code = err["code"].as_str().unwrap_or("");
    let kind = err["type"].as_str().unwrap_or("");
    let message = common::redact(err["message"].as_str().unwrap_or("error"));
    tracing::warn!("openai realtime error {kind}/{code}: {message}");
    if code.contains("api_key") || kind == "authentication_error" {
        Error::Auth {
            provider: PROVIDER.into(),
        }
    } else if code.contains("rate_limit") || code.contains("quota") {
        Error::Quota {
            provider: PROVIDER.into(),
        }
    } else if code == "model_not_found" {
        Error::Config(format!("{PROVIDER}: {message}"))
    } else {
        Error::Other(format!("{PROVIDER}: {message}"))
    }
}

/// What the server has told us so far (kept apart from the socket so `drain` can borrow both).
#[derive(Default)]
struct Heard {
    partial: String,
    item: Option<String>,
    transcript: Option<String>,
    committed: bool,
}

impl Heard {
    fn handle(&mut self, r: Read, on_partial: &mut Option<PartialSink>) -> Result<bool> {
        let text = match r {
            Read::Json(t) => t,
            Read::Closed { code, reason } => {
                return if self.transcript.is_some() {
                    Ok(true)
                } else {
                    Err(Error::Network {
                        provider: PROVIDER.into(),
                        message: common::redact(&format!("closed ({code}) {reason}")),
                    })
                };
            }
        };
        let ev: Value = serde_json::from_str(text).map_err(|_| common::bad_response(PROVIDER))?;
        match ev["type"].as_str().unwrap_or("") {
            "error" => return Err(api_error(&ev["error"])),
            "input_audio_buffer.committed" => {
                self.item = ev["item_id"].as_str().map(str::to_string);
            }
            "conversation.item.input_audio_transcription.delta" => {
                let stable = self.partial.trim().len();
                self.partial.push_str(ev["delta"].as_str().unwrap_or(""));
                if let Some(cb) = on_partial.as_mut() {
                    cb(self.partial.trim(), stable);
                }
            }
            "conversation.item.input_audio_transcription.completed" => {
                let ours = match (&self.item, ev["item_id"].as_str()) {
                    (Some(want), Some(got)) => want == got,
                    _ => true, // one committed turn per session
                };
                if ours && self.committed {
                    self.transcript = Some(ev["transcript"].as_str().unwrap_or("").to_string());
                    return Ok(true);
                }
            }
            "conversation.item.input_audio_transcription.failed" => {
                return Err(api_error(&ev["error"]));
            }
            _ => {}
        }
        Ok(false)
    }
}

/// One Realtime transcription session for one dictation.
pub struct OpenAiLive {
    ws: Ws,
    timeout: Duration,
    up: Upsampler,
    /// 16 kHz audio not yet sent (batched to ~100 ms frames).
    pending: Vec<f32>,
    samples: u64,
    heard: Heard,
    on_partial: Option<PartialSink>,
}

impl OpenAiLive {
    pub fn open(
        key: &str,
        url: &str,
        update: &Value,
        timeout: Duration,
        on_partial: Option<PartialSink>,
    ) -> Result<Self> {
        let req = ws::request(url, &[("Authorization", format!("Bearer {key}"))])?;
        let mut ws = ws::connect(PROVIDER, req, timeout)?;
        ws::send_text(&mut ws, PROVIDER, timeout, update.to_string())?;
        Ok(Self {
            ws,
            timeout,
            up: Upsampler::default(),
            pending: Vec::with_capacity(FRAME * 2),
            samples: 0,
            heard: Heard::default(),
            on_partial,
        })
    }

    fn append(&mut self, pcm16k: &[f32]) -> Result<()> {
        let mut hi = Vec::with_capacity(pcm16k.len() * 3 / 2 + 2);
        self.up.push(pcm16k, &mut hi);
        self.append_24k(&hi)
    }

    fn append_24k(&mut self, pcm24k: &[f32]) -> Result<()> {
        if pcm24k.is_empty() {
            return Ok(());
        }
        let msg = json!({"type": "input_audio_buffer.append", "audio": common::base64(&common::pcm16(pcm24k))});
        ws::send_text(&mut self.ws, PROVIDER, self.timeout, msg.to_string())
    }

    fn pump(&mut self, wait: Option<Duration>) -> Result<bool> {
        let (heard, cb) = (&mut self.heard, &mut self.on_partial);
        ws::drain(&mut self.ws, PROVIDER, self.timeout, wait, |m| {
            heard.handle(m, cb)
        })
    }
}

impl SttStream for OpenAiLive {
    fn send(&mut self, pcm: &[f32]) -> Result<()> {
        self.samples += pcm.len() as u64;
        self.pending.extend_from_slice(pcm);
        if self.pending.len() >= FRAME {
            let chunk = std::mem::take(&mut self.pending);
            self.append(&chunk)?;
        }
        self.pump(None).map(|_| ())
    }

    fn finish(mut self: Box<Self>) -> Result<SttResult> {
        let t_release = Instant::now();
        let chunk = std::mem::take(&mut self.pending);
        self.append(&chunk)?;
        let mut tail = Vec::new();
        self.up.flush(&mut tail);
        if (self.samples as usize) < MIN_COMMIT {
            if self.samples == 0 {
                ws::close(&mut self.ws);
                return Ok(SttResult::default());
            }
            tail.resize(
                tail.len() + (MIN_COMMIT - self.samples as usize) * 3 / 2,
                0.0,
            );
        }
        self.append_24k(&tail)?;
        ws::send_text(
            &mut self.ws,
            PROVIDER,
            self.timeout,
            json!({"type": "input_audio_buffer.commit"}).to_string(),
        )?;
        self.heard.committed = true;
        let deadline = t_release + self.timeout;
        while self.heard.transcript.is_none() {
            let left = ws::left(deadline, self.timeout)?;
            if self.pump(Some(left))? && self.heard.transcript.is_none() {
                return Err(common::bad_response(PROVIDER));
            }
        }
        ws::close(&mut self.ws);
        Ok(SttResult {
            text: self
                .heard
                .transcript
                .take()
                .unwrap_or_default()
                .trim()
                .to_string(),
            duration_ms: self.samples * 1000 / SAMPLE_RATE as u64,
            processing_ms: t_release.elapsed().as_millis() as u64,
            language: None, // gpt-live-transcribe returns no language prediction
        })
    }

    fn cancel(mut self: Box<Self>) {
        ws::close(&mut self.ws);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::OpenAiStt;
    use crate::ws::mock::MockWs;
    use ochre_core::config::SttConfig;
    use ochre_core::stt::{SttEngine, SttOptions};
    use std::sync::{Arc, Mutex};

    fn b64_len(s: &str) -> usize {
        s.len() / 4 * 3 - s.chars().rev().take_while(|c| *c == '=').count()
    }

    fn live(srv: &MockWs) -> OpenAiStt {
        OpenAiStt::openai(&SttConfig {
            model: LIVE_MODEL.into(),
            language: Some("en-US".into()),
            ..Default::default()
        })
        .with_base(&srv.url("/v1").replace("ws://", "http://"))
        .with_key("test-key")
    }

    #[test]
    fn urls_and_session_shape() {
        assert_eq!(
            ws_url("https://api.openai.com/v1/"),
            "wss://api.openai.com/v1/realtime?intent=transcription"
        );
        let u = session_update(LIVE_MODEL, Some("en"), &["K8s".into(), "a<b".into()], "low");
        assert_eq!(
            u,
            json!({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "gpt-live-transcribe", "delay": "low", "languages": ["en"], "keywords": ["K8s"]},
                "turn_detection": null}}}})
        );
        // gpt-transcribe in a Realtime session takes the singular `language` and no `delay`.
        let t = session_update("gpt-transcribe", Some("de"), &[], "low");
        assert_eq!(
            t["session"]["audio"]["input"]["transcription"],
            json!({"model": "gpt-transcribe", "language": "de"})
        );
    }

    #[test]
    fn streams_appends_then_commits() {
        let srv = MockWs::start(
            vec![json!({"type": "session.created"}).to_string()],
            Box::new(|f: &str| {
                let v: Value = serde_json::from_str(f).unwrap();
                match v["type"].as_str().unwrap() {
                    "session.update" => vec![json!({"type": "session.updated"}).to_string()],
                    "input_audio_buffer.append" => vec![
                        json!({"type": "conversation.item.input_audio_transcription.delta", "item_id": "item_1", "delta": "Hello"}).to_string(),
                    ],
                    "input_audio_buffer.commit" => vec![
                        json!({"type": "input_audio_buffer.committed", "item_id": "item_1"}).to_string(),
                        json!({"type": "conversation.item.input_audio_transcription.delta", "item_id": "item_1", "delta": " world."}).to_string(),
                        json!({"type": "conversation.item.input_audio_transcription.completed", "item_id": "item_1", "transcript": " Hello world. "}).to_string(),
                    ],
                    _ => vec![],
                }
            }),
        );
        let e = live(&srv);
        assert!(e.streaming());
        let seen = Arc::new(Mutex::new(Vec::<(String, usize)>::new()));
        let s2 = seen.clone();
        let opts = SttOptions {
            language: None,
            vocabulary: vec!["Kubernetes".into()],
        };
        let mut s = e
            .stream(
                &opts,
                Some(Box::new(move |t: &str, n: usize| {
                    s2.lock().unwrap().push((t.into(), n))
                })),
            )
            .unwrap()
            .unwrap();
        for _ in 0..10 {
            s.send(&[0.1; 800]).unwrap(); // 0.5 s in 50 ms chunks
            std::thread::sleep(Duration::from_millis(5));
        }
        let r = s.finish().unwrap();
        assert_eq!((r.text.as_str(), r.duration_ms), ("Hello world.", 500));
        assert_eq!(
            srv.header("authorization").as_deref(),
            Some("Bearer test-key")
        );
        assert_eq!(
            *srv.path.lock().unwrap(),
            "/v1/realtime?intent=transcription"
        );
        let frames: Vec<Value> = srv
            .frames()
            .iter()
            .map(|f| serde_json::from_str(f).unwrap())
            .collect();
        let tr = &frames[0]["session"]["audio"]["input"]["transcription"];
        assert_eq!(
            tr,
            &json!({"model": "gpt-live-transcribe", "delay": DEFAULT_DELAY, "languages": ["en"], "keywords": ["Kubernetes"]})
        );
        assert_eq!(frames.last().unwrap()["type"], "input_audio_buffer.commit");
        let appends: Vec<&Value> = frames
            .iter()
            .filter(|f| f["type"] == "input_audio_buffer.append")
            .collect();
        // 8000 samples at 16 kHz -> 12000 at 24 kHz -> 24000 PCM16 bytes, in ~100 ms frames.
        let bytes: usize = appends
            .iter()
            .map(|a| b64_len(a["audio"].as_str().unwrap()))
            .sum();
        assert_eq!(bytes, 24_000);
        assert!(appends.len() >= 5, "{}", appends.len());
        let p = seen.lock().unwrap();
        assert!(p.iter().any(|(t, _)| t.starts_with("Hello")), "{p:?}");
    }

    #[test]
    fn auth_error_event_maps_to_auth() {
        let srv = MockWs::start(
            vec![],
            Box::new(|_f: &str| {
                vec![json!({"type": "error", "error": {"type": "invalid_request_error", "code": "invalid_api_key", "message": "Incorrect API key provided: sk-abcdefghijklmnopqrstuv"}}).to_string()]
            }),
        );
        let e = live(&srv);
        let mut s = e.stream(&SttOptions::default(), None).unwrap().unwrap();
        let mut err = None;
        for _ in 0..50 {
            if let Err(x) = s.send(&[0.0; 1600]) {
                err = Some(x);
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let err = match err {
            Some(e) => e,
            None => s.finish().unwrap_err(),
        };
        assert!(matches!(err, Error::Auth { .. }), "{err:?}");
    }

    #[test]
    fn batch_transcribe_with_live_model_uses_the_socket() {
        let srv = MockWs::start(
            vec![],
            Box::new(|f: &str| {
                if f.contains("input_audio_buffer.commit") {
                    vec![json!({"type": "conversation.item.input_audio_transcription.completed", "item_id": "i", "transcript": "ok"}).to_string()]
                } else {
                    vec![]
                }
            }),
        );
        let r = live(&srv)
            .transcribe(&[0.0; 16_000], &SttOptions::default())
            .unwrap();
        assert_eq!(r.text, "ok");
    }

    #[test]
    fn short_audio_is_padded_to_the_commit_minimum() {
        let srv = MockWs::start(
            vec![],
            Box::new(|f: &str| {
                if f.contains("input_audio_buffer.commit") {
                    vec![json!({"type": "conversation.item.input_audio_transcription.completed", "transcript": ""}).to_string()]
                } else {
                    vec![]
                }
            }),
        );
        let mut s = live(&srv)
            .stream(&SttOptions::default(), None)
            .unwrap()
            .unwrap();
        s.send(&[0.0; 160]).unwrap(); // 10 ms
        assert_eq!(s.finish().unwrap().text, "");
        let bytes: usize = srv
            .frames()
            .iter()
            .filter_map(|f| serde_json::from_str::<Value>(f).ok())
            .filter(|v| v["type"] == "input_audio_buffer.append")
            .map(|v| b64_len(v["audio"].as_str().unwrap()))
            .sum();
        assert!(bytes >= 2400 * 2, "{bytes}"); // >= 100 ms at 24 kHz
    }
}
