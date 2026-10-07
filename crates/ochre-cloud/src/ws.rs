//! WebSocket plumbing shared by the streaming engines (Soniox real-time, OpenAI Realtime
//! transcription, Gemini Live): connect with a bounded handshake, map errors onto
//! `ochre_core::Error`, and read responses either without blocking (while audio is being sent) or
//! with a deadline (after release).
//!
//! Single-threaded by design: `send` writes a frame and then drains whatever arrived, so the
//! server never stalls on a full receive buffer and no reader thread is needed.

use std::io::ErrorKind;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use ochre_core::{Error, Result};
use tungstenite::client::IntoClientRequest;
use tungstenite::handshake::client::Request;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

use crate::common;

pub type Ws = WebSocket<MaybeTlsStream<TcpStream>>;

pub fn error(provider: &str, e: tungstenite::Error, timeout: Duration) -> Error {
    match e {
        tungstenite::Error::Http(resp) => {
            let body = resp
                .body()
                .as_deref()
                .map(String::from_utf8_lossy)
                .unwrap_or_default()
                .into_owned();
            common::status_error(provider, resp.status().as_u16(), &body)
        }
        tungstenite::Error::Io(io)
            if matches!(io.kind(), ErrorKind::TimedOut | ErrorKind::WouldBlock) =>
        {
            Error::Timeout(timeout)
        }
        other => Error::Network {
            provider: provider.into(),
            message: common::redact(&other.to_string()),
        },
    }
}

/// A client request for `url` with extra headers (e.g. `Authorization`).
pub fn request(url: &str, headers: &[(&str, String)]) -> Result<Request> {
    let mut req = url
        .into_client_request()
        .map_err(|_| Error::Config("bad websocket url".into()))?;
    for (k, v) in headers {
        let name = tungstenite::http::HeaderName::from_bytes(k.as_bytes())
            .map_err(|_| Error::Config(format!("bad header {k}")))?;
        let value = tungstenite::http::HeaderValue::from_str(v)
            .map_err(|_| Error::Config(format!("bad header value for {k}")))?;
        req.headers_mut().insert(name, value);
    }
    Ok(req)
}

/// TCP connect (bounded by `min(timeout, 5 s)`), then the TLS + WebSocket handshake. `ws://`
/// URLs connect in plain text (tests).
pub fn connect(provider: &str, req: Request, timeout: Duration) -> Result<Ws> {
    common::install_crypto();
    let uri = req.uri().clone();
    let host = uri.host().unwrap_or("").to_string();
    let port = uri.port_u16().unwrap_or(if uri.scheme_str() == Some("ws") {
        80
    } else {
        443
    });
    let net = |e: std::io::Error| Error::Network {
        provider: provider.into(),
        message: e.to_string(),
    };
    let addr = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(net)?
        .next()
        .ok_or_else(|| Error::Network {
            provider: provider.into(),
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
    let (ws, _) = tungstenite::client_tls(req, tcp).map_err(|e| match e {
        tungstenite::HandshakeError::Failure(e) => error(provider, e, timeout),
        tungstenite::HandshakeError::Interrupted(_) => Error::Timeout(timeout),
    })?;
    Ok(ws)
}

/// The TCP socket under TLS. Timeouts must be set on this handle: on Windows a `try_clone`d
/// handle does not share SO_RCVTIMEO with the original.
pub fn tcp(ws: &Ws) -> Option<&TcpStream> {
    match ws.get_ref() {
        MaybeTlsStream::Plain(s) => Some(s),
        MaybeTlsStream::Rustls(s) => Some(&s.sock),
        _ => None,
    }
}

/// What a message handler saw.
pub enum Read<'a> {
    /// A text frame, or a binary frame holding UTF-8 JSON (Gemini Live sends JSON as binary).
    Json(&'a str),
    /// The server closed; `reason` may carry an error (Gemini puts API errors there).
    Closed { code: u16, reason: String },
}

/// Read responses until `on` returns `Ok(true)` (done) or nothing more is available.
/// `wait = None` polls without blocking (a read timeout would round up to the ~15 ms Windows
/// timer tick on every `send`); `Some(d)` blocks up to `d` per read. Returns whether `on` said done.
pub fn drain(
    ws: &mut Ws,
    provider: &str,
    timeout: Duration,
    wait: Option<Duration>,
    mut on: impl FnMut(Read) -> Result<bool>,
) -> Result<bool> {
    let poll = wait.is_none();
    if let Some(t) = tcp(ws) {
        t.set_nonblocking(poll).ok();
        if !poll {
            t.set_read_timeout(wait.map(|d| d.max(Duration::from_millis(1))))
                .ok();
        }
    }
    let result = loop {
        let msg = match ws.read() {
            Ok(m) => m,
            Err(tungstenite::Error::Io(e)) if poll && e.kind() == ErrorKind::WouldBlock => {
                break Ok(false);
            }
            Err(tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed) => {
                break on(Read::Closed {
                    code: 1000,
                    reason: String::new(),
                })
                .map(|_| true);
            }
            Err(e) => break Err(error(provider, e, timeout)),
        };
        let r = match &msg {
            Message::Text(t) => on(Read::Json(t.as_str())),
            Message::Binary(b) => match std::str::from_utf8(b) {
                Ok(s) => on(Read::Json(s)),
                Err(_) => Ok(false),
            },
            Message::Close(frame) => {
                let (code, reason) = frame
                    .as_ref()
                    .map(|f| (u16::from(f.code), f.reason.to_string()))
                    .unwrap_or((1000, String::new()));
                break on(Read::Closed { code, reason }).map(|_| true);
            }
            _ => Ok(false),
        };
        match r {
            Ok(true) => break Ok(true),
            Ok(false) => {}
            Err(e) => break Err(e),
        }
    };
    if poll && let Some(t) = tcp(ws) {
        t.set_nonblocking(false).ok(); // writes stay blocking (bounded by the write timeout)
    }
    result
}

/// Time left before `deadline`, or a Timeout error.
pub fn left(deadline: Instant, timeout: Duration) -> Result<Duration> {
    deadline
        .checked_duration_since(Instant::now())
        .filter(|d| !d.is_zero())
        .ok_or(Error::Timeout(timeout))
}

pub fn send_text(ws: &mut Ws, provider: &str, timeout: Duration, text: String) -> Result<()> {
    ws.send(Message::text(text))
        .map_err(|e| error(provider, e, timeout))
}

/// Close without waiting for the server's close frame (closing also stops billing).
pub fn close(ws: &mut Ws) {
    let _ = ws.close(None);
    let _ = ws.flush();
}

#[cfg(test)]
pub mod mock {
    //! A loopback WebSocket server for request-shape tests: records every text/binary frame the
    //! client sends and answers with a scripted reply function.

    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    use tungstenite::Message;

    pub type Script = Box<dyn FnMut(&str) -> Vec<String> + Send>;

    pub struct MockWs {
        pub port: u16,
        pub frames: Arc<Mutex<Vec<String>>>,
        pub headers: Arc<Mutex<Vec<(String, String)>>>,
        pub path: Arc<Mutex<String>>,
    }

    impl MockWs {
        /// `greeting` frames are sent right after the handshake; `script(frame)` returns the
        /// replies to each client frame. Binary frames are recorded as `<binary N>`.
        #[allow(clippy::result_large_err)] // tungstenite's handshake callback signature
        pub fn start(greeting: Vec<String>, mut script: Script) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let port = listener.local_addr().unwrap().port();
            let frames = Arc::new(Mutex::new(Vec::new()));
            let headers = Arc::new(Mutex::new(Vec::new()));
            let path = Arc::new(Mutex::new(String::new()));
            let (f, h, p) = (frames.clone(), headers.clone(), path.clone());
            std::thread::spawn(move || {
                let Ok((stream, _)) = listener.accept() else {
                    return;
                };
                let cb = |req: &tungstenite::handshake::server::Request,
                          resp: tungstenite::handshake::server::Response| {
                    *p.lock().unwrap() = req.uri().to_string();
                    let mut hs = h.lock().unwrap();
                    for (k, v) in req.headers() {
                        hs.push((k.to_string(), v.to_str().unwrap_or("").to_string()));
                    }
                    Ok(resp)
                };
                let Ok(mut ws) = tungstenite::accept_hdr(stream, cb) else {
                    return;
                };
                for g in greeting {
                    let _ = ws.send(Message::text(g));
                }
                while let Ok(msg) = ws.read() {
                    let text = match msg {
                        Message::Text(t) => t.to_string(),
                        Message::Binary(b) => format!("<binary {}>", b.len()),
                        Message::Close(_) => break,
                        _ => continue,
                    };
                    f.lock().unwrap().push(text.clone());
                    for r in script(&text) {
                        if ws.send(Message::text(r)).is_err() {
                            return;
                        }
                    }
                }
            });
            MockWs {
                port,
                frames,
                headers,
                path,
            }
        }

        pub fn url(&self, path: &str) -> String {
            format!("ws://127.0.0.1:{}{path}", self.port)
        }

        pub fn frames(&self) -> Vec<String> {
            self.frames.lock().unwrap().clone()
        }

        pub fn header(&self, name: &str) -> Option<String> {
            self.headers
                .lock()
                .unwrap()
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.clone())
        }
    }
}
