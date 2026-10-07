//! HTTP plumbing shared by the cloud refiners: one pooled client per refiner (TLS handshake paid
//! once, not per dictation), key lookup, and error mapping onto `ochre_core::Error` so the core can
//! fall back to raw text.

use std::sync::{LazyLock, Once};
use std::time::Duration;

use ochre_core::{Error, Result};
use regex::Regex;
use serde_json::Value;

pub const USER_AGENT: &str = concat!("ochre/", env!("CARGO_PKG_VERSION"));

static CRYPTO: Once = Once::new();

/// reqwest is built with `rustls-no-provider` (ring, no aws-lc/cmake/nasm build dependency), so
/// the process-wide rustls provider is installed here before the first client is built.
pub fn install_crypto() {
    CRYPTO.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
}

/// A pooled blocking client. Per-request timeouts are set on each call.
pub fn client() -> reqwest::blocking::Client {
    install_crypto();
    reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(3))
        .pool_idle_timeout(Duration::from_secs(300))
        .tcp_keepalive(Duration::from_secs(30))
        .tcp_nodelay(true)
        .build()
        .expect("reqwest client")
}

/// A local (loopback) client: no proxy, no TLS.
pub fn local_client() -> reqwest::blocking::Client {
    install_crypto();
    reqwest::blocking::Client::builder()
        .user_agent(USER_AGENT)
        .no_proxy()
        .tcp_nodelay(true)
        .connect_timeout(Duration::from_secs(2))
        .build()
        .expect("reqwest client")
}

static KEYISH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b(sk|gsk|xi|key|sk-ant|AIza|csk)[-_A-Za-z0-9]{12,}\b").unwrap());

/// Never let something that looks like an API key reach a log line or the HUD.
pub fn redact(text: &str) -> String {
    KEYISH.replace_all(text, "[redacted]").into_owned()
}

/// A short, key-free description of an error body.
pub fn error_detail(body: &str) -> String {
    let text = match serde_json::from_str::<Value>(body) {
        Ok(v) => {
            let err = v
                .get("error")
                .or_else(|| v.get("detail"))
                .or_else(|| v.get("message"))
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

/// HTTP status -> ochre error. 401/403 = fix the key, 402/429 = quota/rate limit, 5xx = transient
/// network trouble (fallback-worthy), anything else is a request problem.
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
    // Walk the source chain: a reqwest "error sending request" often wraps an io timeout.
    let mut src: Option<&dyn std::error::Error> = Some(e);
    while let Some(s) = src {
        if let Some(io) = s.downcast_ref::<std::io::Error>()
            && io.kind() == std::io::ErrorKind::TimedOut
        {
            return Error::Timeout(timeout);
        }
        src = s.source();
    }
    let kind = if e.is_connect() {
        "connect"
    } else if e.is_request() {
        "request"
    } else {
        "transport"
    };
    Error::Network {
        provider: provider.to_string(),
        message: redact(&format!("{kind}: {e}")),
    }
}

/// An HTTP error with its status kept, so callers can retry on specific 400s.
#[derive(Debug)]
pub struct HttpError {
    pub status: Option<u16>,
    pub body: String,
    pub error: Error,
}

impl From<HttpError> for Error {
    fn from(e: HttpError) -> Self {
        e.error
    }
}

/// POST JSON, return parsed JSON, mapping every failure.
pub fn post_json(
    client: &reqwest::blocking::Client,
    provider: &str,
    url: &str,
    headers: &[(&str, &str)],
    body: &Value,
    timeout: Duration,
) -> std::result::Result<Value, HttpError> {
    let mut req = client.post(url).timeout(timeout).json(body);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = req.send().map_err(|e| HttpError {
        status: None,
        body: String::new(),
        error: transport_error(provider, &e, timeout),
    })?;
    let status = resp.status().as_u16();
    let text = resp.text().map_err(|e| HttpError {
        status: Some(status),
        body: String::new(),
        error: transport_error(provider, &e, timeout),
    })?;
    if !(200..300).contains(&status) {
        let error = status_error(provider, status, &text);
        return Err(HttpError {
            status: Some(status),
            body: text,
            error,
        });
    }
    serde_json::from_str(&text).map_err(|_| HttpError {
        status: Some(status),
        body: String::new(),
        error: Error::Other(format!("{provider}: non-JSON response")),
    })
}

/// Warm the connection pool (DNS + TCP + TLS) and validate the key with a free GET. Auth errors are
/// returned; network errors are only logged, because the network may come back before the first
/// dictation and the core falls back to raw text anyway.
pub fn prewarm(
    client: &reqwest::blocking::Client,
    provider: &str,
    url: &str,
    headers: &[(&str, &str)],
) -> Result<()> {
    let mut req = client.get(url).timeout(Duration::from_secs(5));
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    match req.send() {
        Ok(resp) => match resp.status().as_u16() {
            401 | 403 => Err(Error::Auth {
                provider: provider.to_string(),
            }),
            _ => Ok(()),
        },
        Err(e) => {
            tracing::warn!("{provider}: prewarm failed: {}", redact(&e.to_string()));
            Ok(())
        }
    }
}

pub fn require_key(provider: &str) -> Result<String> {
    ochre_core::secrets::get(provider).ok_or_else(|| Error::MissingKey(provider.to_string()))
}

/// Output cap: the guard rejects anything over 2x the input anyway, so don't pay for (or wait on)
/// a runaway generation. ~4 chars/token for English, plus room for punctuation.
pub fn max_tokens_for(text: &str) -> u32 {
    ((text.chars().count() / 4 * 2) as u32 + 64).clamp(64, 4096)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_statuses() {
        assert!(matches!(status_error("x", 401, ""), Error::Auth { .. }));
        assert!(matches!(status_error("x", 403, ""), Error::Auth { .. }));
        assert!(matches!(status_error("x", 429, ""), Error::Quota { .. }));
        assert!(matches!(status_error("x", 402, ""), Error::Quota { .. }));
        assert!(matches!(status_error("x", 503, ""), Error::Network { .. }));
        assert!(status_error("x", 503, "").is_fallback_worthy());
        assert!(
            matches!(status_error("x", 400, "{\"error\":{\"message\":\"bad\"}}"), Error::Other(m) if m.ends_with("bad"))
        );
    }

    #[test]
    fn redacts_keys() {
        let s = redact("invalid key sk-proj-abcdefghijklmnop1234 given");
        assert!(!s.contains("abcdefghijklmnop") && s.contains("[redacted]"));
        assert_eq!(
            error_detail(
                "{\"error\":{\"message\":\"Incorrect API key: sk-abcdefghijklmnopqrst\"}}"
            ),
            "Incorrect API key: [redacted]"
        );
    }
}
