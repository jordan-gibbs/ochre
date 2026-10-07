#![allow(dead_code)]
//! Minimal HTTP/1.1 mock server for request-shape tests (std only, loopback, no network).

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Recorded {
    pub method: String,
    pub path: String,
    pub query: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
    pub fn body_str(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

#[derive(Debug, Clone)]
pub struct Reply {
    pub status: u16,
    pub body: Vec<u8>,
    pub delay: Duration,
}

impl Reply {
    pub fn json(status: u16, v: serde_json::Value) -> Self {
        Reply {
            status,
            body: v.to_string().into_bytes(),
            delay: Duration::ZERO,
        }
    }
    pub fn delay(d: Duration) -> Self {
        Reply {
            status: 200,
            body: b"{}".to_vec(),
            delay: d,
        }
    }
}

pub struct MockServer {
    port: u16,
    seen: Arc<Mutex<Vec<Recorded>>>,
}

impl MockServer {
    /// Replies are served in order; once exhausted, the last one repeats.
    pub fn start(replies: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen = Arc::new(Mutex::new(Vec::new()));
        let queue = Arc::new(Mutex::new(replies.into_iter().collect::<VecDeque<_>>()));
        let s = seen.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                let (s, q) = (s.clone(), queue.clone());
                std::thread::spawn(move || serve(conn, s, q));
            }
        });
        MockServer { port, seen }
    }

    pub fn url(&self, path: &str) -> String {
        format!("http://127.0.0.1:{}{path}", self.port)
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.seen.lock().unwrap().clone()
    }
}

fn serve(conn: TcpStream, seen: Arc<Mutex<Vec<Recorded>>>, queue: Arc<Mutex<VecDeque<Reply>>>) {
    let mut w = conn.try_clone().unwrap();
    let mut r = BufReader::new(conn);
    loop {
        let mut line = String::new();
        if r.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let mut parts = line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let target = parts.next().unwrap_or("").to_string();
        let (path, query) = target
            .split_once('?')
            .map(|(p, q)| (p.to_string(), q.to_string()))
            .unwrap_or((target, String::new()));
        let mut headers = Vec::new();
        loop {
            let mut h = String::new();
            if r.read_line(&mut h).unwrap_or(0) == 0 {
                return;
            }
            let h = h.trim_end();
            if h.is_empty() {
                break;
            }
            if let Some((k, v)) = h.split_once(':') {
                headers.push((k.trim().to_string(), v.trim().to_string()));
            }
        }
        let get = |n: &str| {
            headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(n))
                .map(|(_, v)| v.clone())
        };
        let mut body = Vec::new();
        if let Some(n) = get("content-length").and_then(|v| v.parse::<usize>().ok()) {
            body.resize(n, 0);
            if r.read_exact(&mut body).is_err() {
                return;
            }
        } else if get("transfer-encoding").is_some_and(|v| v.contains("chunked")) {
            loop {
                let mut sz = String::new();
                r.read_line(&mut sz).ok();
                let n = usize::from_str_radix(sz.trim(), 16).unwrap_or(0);
                let mut chunk = vec![0; n + 2];
                if r.read_exact(&mut chunk).is_err() {
                    return;
                }
                if n == 0 {
                    break;
                }
                body.extend_from_slice(&chunk[..n]);
            }
        }
        seen.lock().unwrap().push(Recorded {
            method,
            path,
            query,
            headers,
            body,
        });
        let reply = {
            let mut q = queue.lock().unwrap();
            if q.len() > 1 {
                q.pop_front().unwrap()
            } else {
                q.front()
                    .cloned()
                    .unwrap_or(Reply::json(500, serde_json::json!({})))
            }
        };
        std::thread::sleep(reply.delay);
        let head = format!(
            "HTTP/1.1 {} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            reply.status,
            reply.body.len()
        );
        if w.write_all(head.as_bytes())
            .and_then(|_| w.write_all(&reply.body))
            .is_err()
        {
            return;
        }
    }
}
