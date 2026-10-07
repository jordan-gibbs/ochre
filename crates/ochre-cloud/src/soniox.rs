//! Soniox: async REST for batch phrases (`transcribe`), real-time WebSocket for streaming while
//! the user speaks (`stream`).
//!
//! Why both (measured on an 8.5 s clip, Oct 2026): pushing a *finished* segment through the
//! real-time socket takes ~6 s because the RT model consumes audio at only ~1.5x real time, while
//! the async flow (upload -> create -> poll -> fetch) takes ~1.4-2.2 s. Streaming *during* capture,
//! release latency is one `finalize` round trip (~1.1 s measured in Python). The engine streams
//! through `SttEngine::stream` (`streaming()` is true), so the orchestrator skips the segmenter.
//!
//! Real-time protocol: JSON config frame -> binary `pcm_s16le` frames ->
//! `{"type": "finalize"}`, answered by a final `<fin>` token; we close after `<fin>` (an empty
//! end-of-audio frame never produced `finished` in testing). Dictionary words go in
//! `context.terms` on both paths. Uploaded files and transcripts are deleted after reading, off the
//! latency path. Cost: async `stt-async-v5` $0.10/h; real-time `stt-rt-v5` $0.12/h (billed while
//! the socket is open).

use std::io::ErrorKind;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use ochre_core::config::SttConfig;
use ochre_core::events::EngineInfo;
use ochre_core::stt::{PartialSink, ProgressFn, SttEngine, SttOptions, SttResult, SttStream};
use ochre_core::{Error, Result, SAMPLE_RATE};
use reqwest::blocking::multipart::{Form, Part};
use serde_json::{Value, json};
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

use crate::common::{self, Core};

pub const API: &str = "https://api.soniox.com/v1";
pub const WS_URL: &str = "wss://stt-rt.soniox.com/transcribe-websocket";
pub const ASYNC_MODEL: &str = "stt-async-v5";
pub const RT_MODEL: &str = "stt-rt-v5";
const SPECIAL: [&str; 2] = ["<end>", "<fin>"];
/// Soniox caps context at ~8k tokens / ~10k chars.
const MAX_TERMS: usize = 200;

pub fn info() -> EngineInfo {
    EngineInfo {
        id: "soniox".into(),
        label: "Soniox".into(),
        kind: "cloud".into(),
        models: vec![ASYNC_MODEL.into()],
        default_model: ASYNC_MODEL.into(),
        needs_key: true,
        note: "~$0.10/h batch; ~$0.12/h streaming".into(),
        languages: "multilingual".into(),
    }
}

fn context(vocab: &[String]) -> Option<Value> {
    let t = common::terms(vocab, MAX_TERMS, 100);
    (!t.is_empty()).then(|| json!({"terms": t}))
}

pub struct Soniox {
    pub core: Core,
    base: String,
    pub poll: Duration,
}

impl Soniox {
    pub fn new(cfg: &SttConfig) -> Self {
        let mut core = Core::new("soniox", ASYNC_MODEL, cfg);
        if core.model.starts_with("stt-rt") {
            core.model = ASYNC_MODEL.into(); // an RT model name in config still means batch via REST
        }
        Self {
            core,
            base: API.into(),
            poll: Duration::from_millis(100),
        }
    }
    pub fn with_base(mut self, base: &str) -> Self {
        self.base = base.trim_end_matches('/').to_string();
        self
    }
    pub fn with_key(mut self, key: &str) -> Self {
        self.core.key_override = Some(key.into());
        self
    }

    pub fn job(&self, file_id: &str, lang: Option<&str>, vocab: &[String]) -> Value {
        let mut job =
            json!({"model": self.core.model, "file_id": file_id, "client_reference_id": "ochre"});
        if let Some(l) = lang {
            job["language_hints"] = json!([l]);
        }
        if let Some(c) = context(vocab) {
            job["context"] = c;
        }
        job
    }

    /// Open a real-time session for one dictation. Feed audio with [`SonioxStream::send`] as it is
    /// captured, then [`SonioxStream::finish`] on release.
    pub fn stream(&self, opts: &SttOptions, on_partial: Option<PartialFn>) -> Result<SonioxStream> {
        SonioxStream::open(
            &self.core.key()?,
            WS_URL,
            RT_MODEL,
            self.core.lang(opts.language.as_deref()).as_deref(),
            &opts.vocabulary,
            self.core.timeout,
            on_partial,
        )
    }
}

impl SttEngine for Soniox {
    fn info(&self) -> EngineInfo {
        let mut i = info();
        i.default_model = self.core.model.clone();
        i
    }

    fn load(&mut self, _progress: ProgressFn) -> Result<()> {
        let key = self.core.key()?;
        self.core.prewarm(
            &format!("{}/models", self.base),
            &[("Authorization", format!("Bearer {key}"))],
        )
    }

    fn transcribe(&self, pcm: &[f32], opts: &SttOptions) -> Result<SttResult> {
        let key = self.core.key()?;
        let c = &self.core;
        let t0 = Instant::now();
        let deadline = t0 + c.timeout;
        let lang = c.lang(opts.language.as_deref());
        let auth = format!("Bearer {key}");
        let mut cleanup: Vec<String> = Vec::new();
        let result = (|| {
            let form = Form::new().part(
                "file",
                Part::bytes(common::wav(pcm))
                    .file_name("audio.wav")
                    .mime_str("audio/wav")
                    .expect("mime"),
            );
            let file = c.send(
                c.client
                    .post(format!("{}/files", self.base))
                    .header("Authorization", &auth)
                    .multipart(form),
                c.left(deadline)?,
            )?;
            let file_id = file["id"]
                .as_str()
                .ok_or_else(|| common::bad_response("soniox"))?
                .to_string();
            cleanup.push(format!("{}/files/{file_id}", self.base));
            let job_body = self.job(&file_id, lang.as_deref(), &opts.vocabulary);
            let mut job = c.send(
                c.client
                    .post(format!("{}/transcriptions", self.base))
                    .header("Authorization", &auth)
                    .json(&job_body),
                c.left(deadline)?,
            )?;
            let id = job["id"]
                .as_str()
                .ok_or_else(|| common::bad_response("soniox"))?
                .to_string();
            cleanup.insert(0, format!("{}/transcriptions/{id}", self.base));
            // No webhook for a desktop app, so poll; the async wait itself is ~1.5 s.
            while !matches!(job["status"].as_str(), Some("completed") | Some("error")) {
                if Instant::now() + self.poll > deadline {
                    return Err(Error::Timeout(c.timeout));
                }
                std::thread::sleep(self.poll);
                job = c.send(
                    c.client
                        .get(format!("{}/transcriptions/{id}", self.base))
                        .header("Authorization", &auth),
                    c.left(deadline)?,
                )?;
            }
            if job["status"] == "error" {
                return Err(Error::Other(format!(
                    "soniox: {}",
                    common::redact(
                        job["error_message"]
                            .as_str()
                            .unwrap_or("transcription failed")
                    )
                )));
            }
            c.send(
                c.client
                    .get(format!("{}/transcriptions/{id}/transcript", self.base))
                    .header("Authorization", &auth),
                c.left(deadline)?,
            )
        })();
        c.delete_later(cleanup, ("Authorization", auth));
        let body = result?;
        let tokens = body["tokens"].as_array().cloned().unwrap_or_default();
        let text = match body["text"].as_str() {
            Some(t) if !t.is_empty() => t.to_string(),
            _ => tokens
                .iter()
                .filter_map(|t| t["text"].as_str())
                .filter(|t| !SPECIAL.contains(t))
                .collect(),
        };
        Ok(SttResult {
            text: text.trim().to_string(),
            duration_ms: common::duration_ms(pcm),
            processing_ms: t0.elapsed().as_millis() as u64,
            language: tokens
                .iter()
                .find_map(|t| t["language"].as_str())
                .map(str::to_string)
                .or(lang),
        })
    }

    fn streaming(&self) -> bool {
        true
    }

    fn stream(
        &self,
        opts: &SttOptions,
        on_partial: Option<PartialSink>,
    ) -> Option<Result<Box<dyn SttStream>>> {
        Some(Soniox::stream(self, opts, on_partial).map(|s| Box::new(s) as Box<dyn SttStream>))
    }

    fn prewarm(&self) {
        if let Ok(key) = self.core.key() {
            self.core.prewarm_async(
                format!("{}/models", self.base),
                vec![("Authorization", format!("Bearer {key}"))],
            );
        }
    }
}

/// `on_partial(text, stable_chars)`: final text plus the current non-final tail; the first
/// `stable_chars` will not change.
pub type PartialFn = Box<dyn FnMut(&str, usize) + Send>;

/// One real-time socket for one dictation. Single-threaded: `send` writes a frame and drains any
/// responses without blocking (so the server never stalls on a full receive buffer); `finish`
/// finalizes and waits for `<fin>`.
pub struct SonioxStream {
    ws: WebSocket<MaybeTlsStream<TcpStream>>,
    finals: String,
    pending: String,
    language: Option<String>,
    samples: u64,
    fin: bool,
    timeout: Duration,
    on_partial: Option<PartialFn>,
}

fn ws_error(e: tungstenite::Error, timeout: Duration) -> Error {
    match e {
        tungstenite::Error::Http(resp) => {
            common::status_error("soniox", resp.status().as_u16(), "")
        }
        tungstenite::Error::Io(io)
            if matches!(io.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) =>
        {
            Error::Timeout(timeout)
        }
        other => Error::Network {
            provider: "soniox".into(),
            message: common::redact(&other.to_string()),
        },
    }
}

impl SonioxStream {
    pub fn open(
        key: &str,
        url: &str,
        model: &str,
        lang: Option<&str>,
        vocab: &[String],
        timeout: Duration,
        on_partial: Option<PartialFn>,
    ) -> Result<Self> {
        common::install_crypto();
        let u: tungstenite::http::Uri = url
            .parse()
            .map_err(|_| Error::Config(format!("bad url {url}")))?;
        let host = u.host().unwrap_or("").to_string();
        let port = u.port_u16().unwrap_or(if u.scheme_str() == Some("ws") {
            80
        } else {
            443
        });
        let net = |e: std::io::Error| Error::Network {
            provider: "soniox".into(),
            message: e.to_string(),
        };
        let addr = (host.as_str(), port)
            .to_socket_addrs()
            .map_err(net)?
            .next()
            .ok_or_else(|| Error::Network {
                provider: "soniox".into(),
                message: "dns".into(),
            })?;
        let connect_timeout = timeout.min(Duration::from_secs(5));
        let tcp = TcpStream::connect_timeout(&addr, connect_timeout).map_err(|e| {
            if e.kind() == ErrorKind::TimedOut {
                Error::Timeout(timeout)
            } else {
                net(e)
            }
        })?;
        tcp.set_nodelay(true).ok();
        tcp.set_read_timeout(Some(connect_timeout)).ok();
        tcp.set_write_timeout(Some(timeout)).ok();
        let (mut ws, _) = tungstenite::client_tls(url, tcp).map_err(|e| match e {
            tungstenite::HandshakeError::Failure(e) => ws_error(e, timeout),
            tungstenite::HandshakeError::Interrupted(_) => Error::Timeout(timeout),
        })?;
        let mut conf = json!({
            "api_key": key, "model": model, "audio_format": "pcm_s16le", "sample_rate": SAMPLE_RATE,
            "num_channels": 1, "enable_endpoint_detection": false, "client_reference_id": "ochre",
        });
        if let Some(l) = lang {
            conf["language_hints"] = json!([l]);
        }
        if let Some(c) = context(vocab) {
            conf["context"] = c;
        }
        ws.send(Message::text(conf.to_string()))
            .map_err(|e| ws_error(e, timeout))?;
        Ok(SonioxStream {
            ws,
            finals: String::new(),
            pending: String::new(),
            language: None,
            samples: 0,
            fin: false,
            timeout,
            on_partial,
        })
    }

    /// Send captured audio (any chunk size; ~100 ms is typical) and pick up any responses.
    pub fn send(&mut self, pcm: &[f32]) -> Result<()> {
        self.samples += pcm.len() as u64;
        self.ws
            .send(Message::binary(common::pcm16(pcm)))
            .map_err(|e| ws_error(e, self.timeout))?;
        self.drain(None)
    }

    /// The TCP socket under TLS. Timeouts must be set on this handle: on Windows a `try_clone`d
    /// handle does not share SO_RCVTIMEO with the original.
    fn tcp(&self) -> Option<&TcpStream> {
        match self.ws.get_ref() {
            MaybeTlsStream::Plain(s) => Some(s),
            MaybeTlsStream::Rustls(s) => Some(&s.sock),
            _ => None,
        }
    }

    /// Read responses. `None` polls: non-blocking, returns as soon as nothing is buffered (a
    /// read timeout would round up to the ~15 ms Windows timer tick on every `send`). `Some(d)`
    /// blocks up to `d` per read, until `<fin>`.
    fn drain(&mut self, wait: Option<Duration>) -> Result<()> {
        let poll = wait.is_none();
        if let Some(t) = self.tcp() {
            t.set_nonblocking(poll).ok();
            if !poll {
                t.set_read_timeout(wait).ok();
            }
        }
        let result = loop {
            match self.ws.read() {
                Ok(Message::Text(t)) => {
                    if let Err(e) = self.handle(&t) {
                        break Err(e);
                    }
                    if self.fin {
                        break Ok(());
                    }
                }
                Ok(Message::Close(_)) => {
                    self.fin = true;
                    break Ok(());
                }
                Ok(_) => {}
                Err(tungstenite::Error::Io(e)) if poll && e.kind() == ErrorKind::WouldBlock => {
                    break Ok(());
                }
                Err(e) => break Err(ws_error(e, self.timeout)),
            }
        };
        if poll && let Some(t) = self.tcp() {
            t.set_nonblocking(false).ok(); // writes stay blocking (bounded by the write timeout)
        }
        result
    }

    fn handle(&mut self, text: &str) -> Result<()> {
        let msg: Value = serde_json::from_str(text).map_err(|_| common::bad_response("soniox"))?;
        if let Some(code) = msg.get("error_code").filter(|c| !c.is_null()) {
            let status = code.as_u64().unwrap_or(500) as u16;
            let detail = common::redact(msg["error_message"].as_str().unwrap_or(""));
            tracing::warn!("soniox error {status}: {detail}");
            return Err(common::status_error(
                "soniox",
                status,
                &json!({"message": detail}).to_string(),
            ));
        }
        let mut pending = String::new();
        for tok in msg["tokens"].as_array().into_iter().flatten() {
            let t = tok["text"].as_str().unwrap_or("");
            if tok["is_final"] == true {
                if t == "<fin>" {
                    self.fin = true;
                } else if !SPECIAL.contains(&t) {
                    self.finals.push_str(t);
                    if self.language.is_none() {
                        self.language = tok["language"].as_str().map(str::to_string);
                    }
                }
            } else if !SPECIAL.contains(&t) {
                pending.push_str(t);
            }
        }
        self.pending = pending;
        if msg["finished"] == true {
            self.fin = true;
        }
        if let Some(cb) = self.on_partial.as_mut() {
            let stable = self.finals.trim_start();
            let full = format!("{stable}{}", self.pending);
            cb(full.trim_end(), stable.trim_end().len());
        }
        Ok(())
    }

    /// Finalize everything sent so far and return the whole transcript. `processing_ms` is
    /// release-to-`<fin>`, the latency the user feels.
    pub fn finish(mut self) -> Result<SttResult> {
        let t_release = Instant::now();
        self.ws
            .send(Message::text(r#"{"type":"finalize"}"#))
            .map_err(|e| ws_error(e, self.timeout))?;
        let deadline = t_release + self.timeout;
        while !self.fin {
            let left = deadline
                .checked_duration_since(Instant::now())
                .filter(|d| !d.is_zero())
                .ok_or(Error::Timeout(self.timeout))?;
            self.drain(Some(left))?;
        }
        let _ = self.ws.close(None);
        let _ = self.ws.flush();
        Ok(SttResult {
            text: self.finals.trim().to_string(),
            duration_ms: self.samples * 1000 / SAMPLE_RATE as u64,
            processing_ms: t_release.elapsed().as_millis() as u64,
            language: self.language.take(),
        })
    }

    /// Discard the session (closing stops billing).
    pub fn cancel(mut self) {
        let _ = self.ws.close(None);
        let _ = self.ws.flush();
    }
}

impl SttStream for SonioxStream {
    fn send(&mut self, pcm: &[f32]) -> Result<()> {
        SonioxStream::send(self, pcm)
    }
    fn finish(self: Box<Self>) -> Result<SttResult> {
        SonioxStream::finish(*self)
    }
    fn cancel(self: Box<Self>) {
        SonioxStream::cancel(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{MockServer, Reply};

    #[test]
    fn async_flow() {
        let srv = MockServer::start(vec![
            Reply::json(200, json!({"id": "f1"})),
            Reply::json(200, json!({"id": "t1", "status": "queued"})),
            Reply::json(200, json!({"id": "t1", "status": "completed"})),
            Reply::json(
                200,
                json!({"text": "", "tokens": [{"text": "Hel", "language": "en"}, {"text": "lo"}, {"text": " world"}, {"text": "<end>"}]}),
            ),
            Reply::json(200, json!({})),
        ]);
        let mut e = Soniox::new(&SttConfig {
            language: Some("en".into()),
            ..Default::default()
        })
        .with_base(&srv.url("/v1"))
        .with_key("test-key");
        e.poll = Duration::from_millis(1);
        let r = e
            .transcribe(
                &vec![0.0; 1600],
                &SttOptions {
                    language: None,
                    vocabulary: vec!["Kubernetes".into()],
                },
            )
            .unwrap();
        assert_eq!(
            (r.text.as_str(), r.language.as_deref()),
            ("Hello world", Some("en"))
        );
        // Both DELETEs run on a background thread: wait for them (bounded), not a fixed sleep.
        for _ in 0..200 {
            if srv
                .requests()
                .iter()
                .filter(|r| r.method == "DELETE")
                .count()
                >= 2
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let reqs = srv.requests();
        assert_eq!(
            (reqs[0].path.as_str(), reqs[0].header("authorization")),
            ("/v1/files", Some("Bearer test-key"))
        );
        assert!(reqs[0].body_str().contains("filename=\"audio.wav\""));
        let job: Value = serde_json::from_slice(&reqs[1].body).unwrap();
        assert_eq!(
            job,
            json!({"model": "stt-async-v5", "file_id": "f1", "client_reference_id": "ochre", "language_hints": ["en"], "context": {"terms": ["Kubernetes"]}})
        );
        assert_eq!(reqs[3].path, "/v1/transcriptions/t1/transcript");
        let deletes: Vec<&str> = reqs
            .iter()
            .filter(|r| r.method == "DELETE")
            .map(|r| r.path.as_str())
            .collect();
        assert_eq!(deletes, ["/v1/transcriptions/t1", "/v1/files/f1"]);
    }

    #[test]
    fn async_error_status() {
        let srv = MockServer::start(vec![Reply::json(
            401,
            json!({"message": "Invalid API key"}),
        )]);
        let e = Soniox::new(&SttConfig::default())
            .with_base(&srv.url("/v1"))
            .with_key("bad");
        assert!(matches!(
            e.transcribe(&[0.0; 160], &SttOptions::default()),
            Err(Error::Auth { .. })
        ));
    }

    #[test]
    fn rt_model_in_config_means_async() {
        assert_eq!(
            Soniox::new(&SttConfig {
                model: "stt-rt-v5".into(),
                ..Default::default()
            })
            .core
            .model,
            ASYNC_MODEL
        );
    }
}
