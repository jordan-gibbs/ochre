//! Shared plumbing for the cloud engines: audio encoding, one pooled HTTPS client per engine (the
//! TLS session stays warm between phrases; a cold handshake cost ~1 s in testing), key lookup,
//! vocabulary splitting, and error mapping onto `ochre_core::Error` so the orchestrator can fall back
//! to the local engine (`Error::is_fallback_worthy`).

use std::sync::{LazyLock, Mutex, Once};
use std::time::{Duration, Instant};

use ochre_core::config::SttConfig;
use ochre_core::{Error, Result, SAMPLE_RATE};
use regex::Regex;
use serde_json::Value;

pub const USER_AGENT: &str = concat!("ochre/", env!("CARGO_PKG_VERSION"));

static CRYPTO: Once = Once::new();

/// reqwest/tungstenite are built against rustls without a bundled provider (ring, no aws-lc
/// build dependency), so the process-wide provider is installed before the first client.
pub fn install_crypto() {
    CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

pub fn client() -> reqwest::blocking::Client {
    install_crypto();
    reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(5))
        .pool_idle_timeout(Duration::from_secs(300))
        .tcp_keepalive(Duration::from_secs(30))
        .tcp_nodelay(true)
        .build()
        .expect("reqwest client")
}

// ---------------------------------------------------------------- audio

/// f32 [-1, 1] mono -> little-endian i16 bytes.
pub fn pcm16(pcm: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Mono 16-bit WAV in memory (~32 KB/s at 16 kHz, so a 30 s phrase is under 1 MB).
pub fn wav(pcm: &[f32]) -> Vec<u8> {
    let data = pcm16(pcm);
    let mut out = Vec::with_capacity(44 + data.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&(SAMPLE_RATE * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&data);
    out
}

pub fn duration_ms(pcm: &[f32]) -> u64 {
    (pcm.len() as u64 * 1000) / SAMPLE_RATE as u64
}

/// Standard base64 with padding (JSON audio frames for the OpenAI and Gemini real-time APIs).
pub fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 {
            A[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if c.len() > 2 {
            A[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

/// Streaming 16 kHz -> 24 kHz linear upsampler (exactly 3 outputs per 2 inputs). The OpenAI
/// Realtime API takes `audio/pcm` at 24 kHz only. Upsampling adds no aliasing, and linear
/// interpolation is plenty for speech recognition.
#[derive(Debug, Default)]
pub struct Upsampler {
    prev: Option<f32>,
    odd: bool,
}

impl Upsampler {
    /// Output k sits at input position 2k/3; each new input sample completes the outputs that lie
    /// between it and the previous one.
    pub fn push(&mut self, pcm: &[f32], out: &mut Vec<f32>) {
        for &x in pcm {
            if let Some(p) = self.prev {
                if self.odd {
                    out.push(p + (x - p) / 3.0);
                } else {
                    out.push(p);
                    out.push(p + (x - p) * 2.0 / 3.0);
                }
                self.odd = !self.odd;
            }
            self.prev = Some(x);
        }
    }

    /// End of audio: the one output still owed (at or 1/3 past the last sample) holds it, so
    /// n inputs always give round(1.5 n) outputs.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        if let Some(p) = self.prev.take() {
            out.push(p);
        }
        self.odd = false;
    }
}

// ---------------------------------------------------------------- vocabulary / language

/// Dictionary words as key terms: trimmed, deduplicated (case-insensitive), length-capped.
pub fn terms(vocab: &[String], max_terms: usize, max_len: usize) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    vocab
        .iter()
        .flat_map(|v| v.split([',', ';', '\n']))
        .map(str::trim)
        .filter(|t| !t.is_empty() && t.chars().count() <= max_len && seen.insert(t.to_lowercase()))
        .take(max_terms)
        .map(str::to_string)
        .collect()
}

/// Per-call language wins over the configured one; "auto"/"" mean let the provider detect.
/// "en-US" -> "en" (every provider here takes ISO-639-1).
pub fn lang(call: Option<&str>, configured: Option<&str>) -> Option<String> {
    let l = call.or(configured)?.trim();
    if l.is_empty() || l.eq_ignore_ascii_case("auto") {
        return None;
    }
    Some(l.split(['-', '_']).next().unwrap_or(l).to_lowercase())
}

// ---------------------------------------------------------------- errors

static KEYISH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(sk|gsk|xi|key|sk-ant|AIza)[-_A-Za-z0-9]{12,}\b").unwrap());

pub fn redact(text: &str) -> String {
    KEYISH.replace_all(text, "[redacted]").into_owned()
}

pub fn error_detail(body: &str) -> String {
    let text = match serde_json::from_str::<Value>(body) {
        Ok(v) => {
            let err = v
                .get("error")
                .or_else(|| v.get("detail"))
                .or_else(|| v.get("message"))
                .or_else(|| v.get("error_message"))
                .or_else(|| v.get("err_msg"))
                .unwrap_or(&v);
            match err {
                Value::String(s) => s.clone(),
                Value::Object(o) => o
                    .get("message")
                    .or_else(|| o.get("msg"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
                    .unwrap_or_else(|| err.to_string()),
                other => other.to_string(),
            }
        }
        Err(_) => body.to_string(),
    };
    redact(&text).chars().take(300).collect()
}

/// 401/403 auth, 402/429 quota, 408/5xx transient (network), 404 config, rest other.
pub fn status_error(provider: &str, status: u16, body: &str) -> Error {
    let provider = provider.to_string();
    match status {
        401 | 403 => Error::Auth { provider },
        402 | 429 => Error::Quota { provider },
        408 | 500..=599 => Error::Network {
            message: format!("HTTP {status}: {}", error_detail(body)),
            provider,
        },
        404 => Error::Config(format!(
            "{provider}: HTTP 404 (unknown model or endpoint): {}",
            error_detail(body)
        )),
        _ => Error::Other(format!("{provider}: HTTP {status}: {}", error_detail(body))),
    }
}

pub fn transport_error(provider: &str, e: &reqwest::Error, timeout: Duration) -> Error {
    if e.is_timeout() {
        return Error::Timeout(timeout);
    }
    let mut src: Option<&dyn std::error::Error> = Some(e);
    while let Some(s) = src {
        if let Some(io) = s.downcast_ref::<std::io::Error>()
            && io.kind() == std::io::ErrorKind::TimedOut
        {
            return Error::Timeout(timeout);
        }
        src = s.source();
    }
    Error::Network {
        provider: provider.to_string(),
        message: redact(&e.to_string()),
    }
}

pub fn bad_response(provider: &str) -> Error {
    Error::Other(format!("{provider}: unexpected response"))
}

// ---------------------------------------------------------------- engine core

/// What every cloud engine holds: provider id, model, language, timeout, key and a pooled client.
pub struct Core {
    pub provider: &'static str,
    pub model: String,
    pub language: Option<String>,
    pub timeout: Duration,
    pub key_override: Option<String>,
    pub client: reqwest::blocking::Client,
    /// Last key-down prewarm, so a burst of dictations doesn't fire a request per key press.
    pub warmed: Mutex<Option<Instant>>,
}

impl Core {
    pub fn new(provider: &'static str, default_model: &str, cfg: &SttConfig) -> Self {
        Core {
            provider,
            model: if cfg.model.is_empty() {
                default_model.to_string()
            } else {
                cfg.model.clone()
            },
            language: cfg.language.clone(),
            timeout: Duration::from_millis(cfg.cloud_timeout_ms.max(500)),
            key_override: None,
            client: client(),
            warmed: Mutex::new(None),
        }
    }

    pub fn key(&self) -> Result<String> {
        match &self.key_override {
            Some(k) => Ok(k.clone()),
            None => ochre_core::secrets::get(self.provider)
                .ok_or_else(|| Error::MissingKey(self.provider.to_string())),
        }
    }

    pub fn lang(&self, call: Option<&str>) -> Option<String> {
        lang(call, self.language.as_deref())
    }

    /// Time left before `deadline`, or a Timeout error.
    pub fn left(&self, deadline: Instant) -> Result<Duration> {
        deadline
            .checked_duration_since(Instant::now())
            .filter(|d| !d.is_zero())
            .ok_or(Error::Timeout(self.timeout))
    }

    /// Send a prepared request with the remaining time budget; map failures; parse JSON.
    pub fn send(&self, req: reqwest::blocking::RequestBuilder, timeout: Duration) -> Result<Value> {
        let resp = req
            .timeout(timeout)
            .send()
            .map_err(|e| transport_error(self.provider, &e, self.timeout))?;
        let status = resp.status().as_u16();
        let text = resp
            .text()
            .map_err(|e| transport_error(self.provider, &e, self.timeout))?;
        if !(200..300).contains(&status) {
            return Err(status_error(self.provider, status, &text));
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|_| bad_response(self.provider))
    }

    /// Warm DNS + TCP + TLS with a free authenticated GET, and fail early on a rejected key. Network
    /// failures are only logged: the network may be back by the first dictation.
    pub fn prewarm(&self, url: &str, headers: &[(&str, String)]) -> Result<()> {
        let mut req = self.client.get(url).timeout(Duration::from_secs(5));
        for (k, v) in headers {
            req = req.header(*k, v);
        }
        match req.send() {
            Ok(r) if matches!(r.status().as_u16(), 401 | 403) => Err(Error::Auth {
                provider: self.provider.to_string(),
            }),
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::warn!(
                    "{}: prewarm failed: {}",
                    self.provider,
                    redact(&e.to_string())
                );
                Ok(())
            }
        }
    }

    /// Key-down prewarm (`SttEngine::prewarm`): a free GET on a background thread re-opens the
    /// pooled TLS / HTTP/2 connection if it went idle, so the first request after release is warm
    /// (OpenAI measured 1.7 s cold vs ~0.8 s warm). Never blocks; at most once per 20 s, since the
    /// pool keeps an idle connection for 5 min.
    pub fn prewarm_async(&self, url: String, headers: Vec<(&'static str, String)>) {
        {
            let mut last = self.warmed.lock().unwrap_or_else(|e| e.into_inner());
            if last.is_some_and(|t| t.elapsed() < Duration::from_secs(20)) {
                return;
            }
            *last = Some(Instant::now());
        }
        let client = self.client.clone();
        let provider = self.provider;
        std::thread::Builder::new()
            .name(format!("{provider}-prewarm"))
            .spawn(move || {
                let mut req = client.get(&url).timeout(Duration::from_secs(5));
                for (k, v) in &headers {
                    req = req.header(*k, v);
                }
                if let Err(e) = req.send() {
                    tracing::debug!("{provider}: prewarm failed: {}", redact(&e.to_string()));
                }
            })
            .ok();
    }

    /// Fire-and-forget DELETEs (privacy cleanup) off the latency path.
    pub fn delete_later(&self, urls: Vec<String>, auth: (&'static str, String)) {
        if urls.is_empty() {
            return;
        }
        let client = self.client.clone();
        let provider = self.provider;
        std::thread::Builder::new()
            .name(format!("{provider}-cleanup"))
            .spawn(move || {
                for u in urls {
                    if let Err(e) = client
                        .delete(&u)
                        .header(auth.0, &auth.1)
                        .timeout(Duration::from_secs(10))
                        .send()
                    {
                        tracing::debug!("{provider} cleanup failed: {}", redact(&e.to_string()));
                    }
                }
            })
            .ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header() {
        let w = wav(&[0.0, 1.0, -1.0, 2.0]);
        assert_eq!(&w[..4], b"RIFF");
        assert_eq!(&w[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(w[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u32::from_le_bytes(w[40..44].try_into().unwrap()), 8);
        assert_eq!(w.len(), 52);
        assert_eq!(i16::from_le_bytes([w[46], w[47]]), 32767);
        assert_eq!(i16::from_le_bytes([w[48], w[49]]), -32767);
        assert_eq!(i16::from_le_bytes([w[50], w[51]]), 32767); // clipped
        assert_eq!(duration_ms(&vec![0.0; 16_000]), 1000);
    }

    #[test]
    fn base64_matches_rfc4648() {
        for (i, o) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(i.as_bytes()), o);
        }
        assert_eq!(base64(&[0xff, 0xfe, 0x00]), "//4A");
    }

    #[test]
    fn upsampler_is_3_for_2_and_chunk_invariant() {
        let x: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let mut a = Upsampler::default();
        let mut whole = Vec::new();
        a.push(&x, &mut whole);
        a.flush(&mut whole);
        // Positions 0, 2/3, 4/3, 2, ... of a ramp are the positions themselves.
        let want: Vec<f32> = (0..13).map(|k| k as f32 * 2.0 / 3.0).collect();
        assert_eq!(whole.len(), want.len());
        for (g, w) in whole.iter().zip(&want) {
            assert!((g - w).abs() < 1e-5, "{whole:?}");
        }
        let mut b = Upsampler::default();
        let mut parts = Vec::new();
        for c in x.chunks(2) {
            b.push(c, &mut parts);
        }
        b.flush(&mut parts);
        assert_eq!(parts, whole);
        let mut c = Upsampler::default();
        let mut n = Vec::new();
        c.push(&vec![0.0; 1600], &mut n);
        c.flush(&mut n);
        assert_eq!(n.len(), 2400);
    }

    #[test]
    fn terms_and_lang() {
        let v = vec![
            "Kubernetes, Soniox".to_string(),
            "kubernetes".into(),
            " ".into(),
            "x".repeat(60),
        ];
        assert_eq!(terms(&v, 10, 50), ["Kubernetes", "Soniox"]);
        assert_eq!(lang(Some("en-US"), None).as_deref(), Some("en"));
        assert_eq!(lang(None, Some("auto")), None);
        assert_eq!(lang(None, Some("de")).as_deref(), Some("de"));
    }

    #[test]
    fn statuses() {
        assert!(status_error("x", 401, "").is_fallback_worthy());
        assert!(status_error("x", 429, "").is_fallback_worthy());
        assert!(status_error("x", 502, "").is_fallback_worthy());
        assert!(!status_error("x", 400, "").is_fallback_worthy());
        assert_eq!(
            redact("bad key gsk_abcdefghijklmnopqrstu"),
            "bad key [redacted]"
        );
    }
}
