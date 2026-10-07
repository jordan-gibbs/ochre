//! Model and binary downloads: resumable, atomic, SHA-256 verified.
//!
//! The invariant: a partially downloaded file never sits at its final path, so a loader never
//! sees a truncated model. Bytes stream into `<dest>.partial`; an interrupted or canceled
//! download resumes from there with an HTTP `Range` request, and the finished file is renamed
//! into place only after the size and hash checks pass.
//!
//! Everything here is blocking; call it from a loader thread, never from the UI or audio thread.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use ochre_core::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const CHUNK: usize = 1 << 20;
const USER_AGENT: &str = concat!(
    "ochre/",
    env!("CARGO_PKG_VERSION"),
    " (+https://github.com/jordan-gibbs/ochre)"
);

/// Direct-download URL for a file in a Hugging Face model repo.
pub fn hf_url(repo: &str, file: &str, rev: &str) -> String {
    // Path segments of `file` stay as separators; everything else is percent-encoded.
    let file = file
        .split('/')
        .map(encode_segment)
        .collect::<Vec<_>>()
        .join("/");
    format!(
        "https://huggingface.co/{repo}/resolve/{}/{file}",
        encode_segment(rev)
    )
}

fn encode_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

const HF_PREFIX: &str = "https://huggingface.co/";

/// A Hugging Face access token for private repos: `HF_TOKEN`, else the token file the `hf` CLI
/// writes (`$HF_HOME/token`, default `~/.cache/huggingface/token`). Never logged.
fn hf_token() -> Option<String> {
    let file = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            directories::BaseDirs::new().map(|b| b.home_dir().join(".cache").join("huggingface"))
        })
        .map(|d| d.join("token"));
    token_from(std::env::var("HF_TOKEN").ok(), file.as_deref())
}

/// Whether a Hugging Face token is configured (private models may be downloadable). The token
/// itself never leaves this crate.
pub fn has_hf_token() -> bool {
    hf_token().is_some()
}

fn token_from(env: Option<String>, file: Option<&Path>) -> Option<String> {
    let clean = |t: String| Some(t.trim().to_string()).filter(|t| !t.is_empty());
    env.and_then(clean).or_else(|| {
        file.and_then(|f| fs::read_to_string(f).ok())
            .and_then(clean)
    })
}

/// The `Authorization` value for `url`: only for `https://huggingface.co/...`, so the token never
/// goes to another host (the signed CDN redirect drops it too, see `ensure_file`).
fn auth_for(url: &str, token: Option<&str>) -> Option<String> {
    let t = token?;
    url.starts_with(HF_PREFIX).then(|| format!("Bearer {t}"))
}

/// The error for a non-success status. A Hugging Face 401/403 means a private or gated repo.
fn status_error(name: &str, url: &str, status: u16, had_token: bool) -> Error {
    if url.starts_with(HF_PREFIX) && matches!(status, 401 | 403) {
        let msg = if had_token {
            format!(
                "Hugging Face refused the token (HTTP {status}); check that it can read this \
                 private model ({url})"
            )
        } else {
            format!(
                "this model is private; set HF_TOKEN (or run `hf auth login`) to download it \
                 (HTTP {status} from {url})"
            )
        };
        return dl_err(name, msg);
    }
    dl_err(name, format!("HTTP {status} from {url}"))
}

/// Hex SHA-256 of a file.
pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn partial_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".partial");
    dest.with_file_name(name)
}

/// `<file>.verified`: written after a successful SHA-256 check, holding the file's size, mtime
/// and hash. While size and mtime are unchanged the file is trusted without re-hashing, so a
/// 650 MB model costs a `stat`, not ~1 s of hashing, on every launch.
fn sidecar_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".verified");
    path.with_file_name(name)
}

fn stamp(meta: &fs::Metadata, sha256: &str) -> String {
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    format!("{} {} {}\n", meta.len(), mtime, sha256.to_ascii_lowercase())
}

/// Record that `path` was verified against `sha256` (best effort).
fn remember_verified(path: &Path, sha256: &str) {
    if let Ok(meta) = fs::metadata(path) {
        let _ = fs::write(sidecar_path(path), stamp(&meta, sha256));
    }
}

/// `remember`: write/consult the `.verified` sidecar (final paths only, never partials).
fn matches(
    path: &Path,
    sha256: Option<&str>,
    size: Option<u64>,
    remember: bool,
) -> io::Result<bool> {
    let Ok(meta) = fs::metadata(path) else {
        return Ok(false);
    };
    if !meta.is_file() {
        return Ok(false);
    }
    if size.is_some_and(|s| s != meta.len()) {
        return Ok(false);
    }
    let Some(want) = sha256 else { return Ok(true) };
    if remember && fs::read_to_string(sidecar_path(path)).is_ok_and(|s| s == stamp(&meta, want)) {
        return Ok(true);
    }
    let ok = sha256_file(path)?.eq_ignore_ascii_case(want);
    if ok && remember {
        remember_verified(path, want);
    }
    Ok(ok)
}

fn dl_err(name: &str, msg: impl std::fmt::Display) -> Error {
    Error::Download(format!("{name}: {msg}"))
}

/// Make sure `dest` exists and is complete; download it from `url` if not. Returns `dest`.
///
/// - `sha256`/`size`, when given, are checked on the existing file and on the download. Without
///   them an existing `dest` is trusted (it only ever got there through the atomic rename).
/// - `progress(done, total)` is called per chunk; `total` is 0 when unknown.
/// - Setting `cancel` stops at the next chunk with `Error::Canceled`; the partial file is kept, so
///   the next call resumes.
pub fn ensure_file(
    url: &str,
    dest: &Path,
    sha256: Option<&str>,
    size: Option<u64>,
    progress: &dyn Fn(u64, u64),
    cancel: &AtomicBool,
) -> Result<PathBuf> {
    let name = dest
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if matches(dest, sha256, size, true)? {
        let n = fs::metadata(dest)?.len();
        progress(n, n);
        return Ok(dest.to_path_buf());
    }
    if let Some(dir) = dest.parent() {
        fs::create_dir_all(dir)?;
    }
    let partial = partial_path(dest);
    let mut offset = fs::metadata(&partial).map(|m| m.len()).unwrap_or(0);
    if let Some(size) = size
        && offset >= size
    {
        // Complete-but-unrenamed, or garbage: verify rather than refetch blindly.
        if offset == size && matches(&partial, sha256, Some(size), false)? {
            fs::rename(&partial, dest)?;
            if let Some(want) = sha256 {
                remember_verified(dest, want);
            }
            progress(size, size);
            return Ok(dest.to_path_buf());
        }
        fs::remove_file(&partial)?;
        offset = 0;
    }

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(20)))
        .timeout_recv_response(Some(Duration::from_secs(45)))
        .user_agent(USER_AGENT)
        // Keep the token across huggingface.co's own (https, same-host) redirects; drop it on the
        // hop to the signed CDN URL.
        .redirect_auth_headers(ureq::config::RedirectAuthHeaders::SameHost)
        .build()
        .into();
    let mut req = agent.get(url);
    let auth = auth_for(url, hf_token().as_deref());
    let had_token = auth.is_some();
    if let Some(a) = auth {
        req = req.header("Authorization", a);
    }
    if offset > 0 {
        req = req.header("Range", format!("bytes={offset}-"));
    }
    let resp = req.call().map_err(|e| dl_err(&name, e))?;
    let status = resp.status().as_u16();
    if status == 416 && offset > 0 {
        // The server says our partial already covers (or overruns) the file: start clean next time.
        fs::remove_file(&partial).ok();
        return Err(dl_err(
            &name,
            "resume rejected (HTTP 416); retry the download",
        ));
    }
    if status != 200 && status != 206 {
        return Err(status_error(&name, url, status, had_token));
    }
    if status == 206 {
        let range = resp
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        if !range.starts_with(&format!("bytes {offset}-")) {
            return Err(dl_err(
                &name,
                format!("invalid Content-Range on resume: {range:?}"),
            ));
        }
    } else {
        offset = 0; // the server ignored Range: start over
    }
    let length: u64 = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let total = size.unwrap_or(if length > 0 { offset + length } else { 0 });

    let mut file = if offset > 0 {
        OpenOptions::new().append(true).open(&partial)?
    } else {
        File::create(&partial)?
    };
    let mut body = resp.into_body();
    let mut reader = body.as_reader();
    let mut buf = vec![0u8; CHUNK];
    let mut received = offset;
    progress(received, total);
    loop {
        if cancel.load(Ordering::Relaxed) {
            file.flush()?;
            return Err(Error::Canceled);
        }
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => {
                file.flush().ok();
                return Err(dl_err(&name, format!("{e}; retry to resume")));
            }
        };
        received += n as u64;
        if size.is_some_and(|s| received > s) {
            drop(file);
            fs::remove_file(&partial).ok();
            return Err(dl_err(&name, "download exceeds the expected size"));
        }
        file.write_all(&buf[..n])?;
        progress(received, total);
    }
    file.flush()?;
    file.sync_all()?;
    drop(file);

    let got = fs::metadata(&partial)?.len();
    if total > 0 && got != total {
        return Err(dl_err(
            &name,
            format!("incomplete download ({got} of {total} bytes); retry to resume"),
        ));
    }
    if let Some(want) = sha256 {
        let have = sha256_file(&partial)?;
        if !have.eq_ignore_ascii_case(want) {
            fs::remove_file(&partial).ok();
            return Err(dl_err(
                &name,
                format!("checksum mismatch (got {have}); retry the download"),
            ));
        }
    }
    fs::rename(&partial, dest)?;
    if let Some(want) = sha256 {
        remember_verified(dest, want);
    }
    tracing::info!(file = %name, bytes = got, "downloaded");
    Ok(dest.to_path_buf())
}

/// One downloadable file of a model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    /// Path relative to the model's directory (may contain `/`).
    pub name: String,
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

impl ModelFile {
    /// A file from a Hugging Face repo, pinned to `rev` (use a commit hash for reproducibility).
    pub fn hf(repo: &str, rev: &str, file: &str, sha256: Option<&str>, size: Option<u64>) -> Self {
        Self {
            name: file.to_string(),
            url: hf_url(repo, file, rev),
            sha256: sha256.map(str::to_string),
            size,
        }
    }
}

/// A model: an id plus the files that make it up. Files land in `models_dir()/<id>/<name>`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    pub id: String,
    pub files: Vec<ModelFile>,
}

impl ModelManifest {
    pub fn dir(&self) -> PathBuf {
        self.dir_in(&ochre_core::paths::models_dir())
    }

    pub fn dir_in(&self, root: &Path) -> PathBuf {
        root.join(&self.id)
    }

    /// Sum of declared sizes (0 when any file's size is unknown).
    pub fn total_size(&self) -> u64 {
        self.files
            .iter()
            .map(|f| f.size)
            .sum::<Option<u64>>()
            .unwrap_or(0)
    }

    /// True when every file is at its final path with the declared size. Cheap (no hashing), for
    /// "is it installed?" checks in the UI.
    pub fn is_present_in(&self, root: &Path) -> bool {
        let dir = self.dir_in(root);
        self.files.iter().all(|f| {
            fs::metadata(dir.join(&f.name))
                .is_ok_and(|m| m.is_file() && f.size.is_none_or(|s| s == m.len()))
        })
    }

    pub fn is_present(&self) -> bool {
        self.is_present_in(&ochre_core::paths::models_dir())
    }

    /// Download (or verify) every file under `models_dir()/<id>`. `progress(item, done, total)`
    /// reports cumulative bytes over the whole model when all sizes are known, else per file.
    pub fn ensure(
        &self,
        progress: &dyn Fn(&str, u64, u64),
        cancel: &AtomicBool,
    ) -> Result<PathBuf> {
        self.ensure_in(&ochre_core::paths::models_dir(), progress, cancel)
    }

    pub fn ensure_in(
        &self,
        root: &Path,
        progress: &dyn Fn(&str, u64, u64),
        cancel: &AtomicBool,
    ) -> Result<PathBuf> {
        let dir = self.dir_in(root);
        let grand = self.total_size();
        let mut before = 0u64;
        for f in &self.files {
            if cancel.load(Ordering::Relaxed) {
                return Err(Error::Canceled);
            }
            let dest = dir.join(&f.name);
            let cb = |done: u64, total: u64| {
                if grand > 0 {
                    progress(&self.id, before + done, grand)
                } else {
                    progress(&f.name, done, total)
                }
            };
            // A dropped connection keeps the partial file, so retrying resumes. Keep retrying
            // while each attempt makes progress; give up after 3 attempts in a row that don't.
            let mut stuck = 0;
            loop {
                let had = fs::metadata(partial_path(&dest))
                    .map(|m| m.len())
                    .unwrap_or(0);
                match ensure_file(&f.url, &dest, f.sha256.as_deref(), f.size, &cb, cancel) {
                    Ok(_) => break,
                    Err(Error::Download(msg)) => {
                        let has = fs::metadata(partial_path(&dest))
                            .map(|m| m.len())
                            .unwrap_or(0);
                        stuck = if has > had { 0 } else { stuck + 1 };
                        if stuck >= 3 || cancel.load(Ordering::Relaxed) {
                            return Err(Error::Download(msg));
                        }
                        tracing::warn!(file = %f.name, error = %msg, "download interrupted; resuming");
                        std::thread::sleep(Duration::from_millis(500 << stuck));
                    }
                    Err(e) => return Err(e),
                }
            }
            before += f.size.unwrap_or(0);
        }
        Ok(dir)
    }

    /// The local path of one file (whether or not it exists yet).
    pub fn path_of(&self, name: &str) -> PathBuf {
        self.dir().join(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ochre-models-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// A tiny HTTP/1.1 server that honours `Range: bytes=N-` and serves `body`. When
    /// `cut_after` is set, the first full (non-range) response is truncated after that many
    /// bytes, simulating a dropped connection.
    fn serve(
        body: Vec<u8>,
        honour_range: bool,
        cut_after: Option<usize>,
    ) -> (String, Arc<AtomicUsize>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let hits = Arc::new(AtomicUsize::new(0));
        let h = hits.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let n = h.fetch_add(1, Ordering::SeqCst);
                let mut reader = BufReader::new(s.try_clone().unwrap());
                let mut range: Option<u64> = None;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    let lower = line.to_ascii_lowercase();
                    if let Some(r) = lower.strip_prefix("range: bytes=") {
                        range = r.trim().trim_end_matches('-').parse().ok();
                    }
                }
                let (status, start) = match range {
                    Some(r) if honour_range => ("206 Partial Content", r as usize),
                    _ => ("200 OK", 0),
                };
                let chunk = &body[start..];
                let mut head = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n",
                    chunk.len()
                );
                if start > 0 {
                    head += &format!(
                        "Content-Range: bytes {start}-{}/{}\r\n",
                        body.len() - 1,
                        body.len()
                    );
                }
                head += "\r\n";
                let _ = s.write_all(head.as_bytes());
                let send = match cut_after {
                    Some(c) if n == 0 => &chunk[..c],
                    _ => chunk,
                };
                let _ = s.write_all(send);
            }
        });
        (format!("http://{addr}/file.bin"), hits)
    }

    fn body(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i * 31 % 251) as u8).collect()
    }

    fn sha(b: &[u8]) -> String {
        hex(&Sha256::digest(b))
    }

    #[test]
    fn hf_url_encodes() {
        assert_eq!(
            hf_url(
                "istupakov/parakeet-tdt-0.6b-v3-onnx",
                "encoder-model.int8.onnx",
                "main"
            ),
            "https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.int8.onnx"
        );
        assert_eq!(
            hf_url("a/b", "dir/x y.bin", "refs/pr/1"),
            "https://huggingface.co/a/b/resolve/refs%2Fpr%2F1/dir/x%20y.bin"
        );
    }

    #[test]
    fn token_only_goes_to_huggingface() {
        let hf = hf_url("polonuim210/ochre-refine-2b", "x.gguf", "main");
        assert_eq!(
            auth_for(&hf, Some("hf_abc")).as_deref(),
            Some("Bearer hf_abc")
        );
        assert_eq!(auth_for(&hf, None), None);
        for other in [
            "http://huggingface.co/a/b/resolve/main/x",
            "https://huggingface.co.evil.com/a",
            "https://huggingface.co@evil.com/a",
            "https://evil.com/https://huggingface.co/a",
            "https://cas-bridge.xethub.hf.co/x",
            "https://github.com/ggml-org/llama.cpp/releases/download/b1/x.zip",
        ] {
            assert_eq!(auth_for(other, Some("hf_abc")), None, "{other}");
        }
    }

    #[test]
    fn token_from_env_then_file() {
        let dir = tmpdir("token");
        let f = dir.join("token");
        fs::write(&f, "hf_file\n").unwrap();
        assert_eq!(
            token_from(Some(" hf_env ".into()), Some(&f)).as_deref(),
            Some("hf_env")
        );
        assert_eq!(
            token_from(Some("".into()), Some(&f)).as_deref(),
            Some("hf_file")
        );
        assert_eq!(token_from(None, Some(&f)).as_deref(), Some("hf_file"));
        assert_eq!(token_from(None, Some(&dir.join("missing"))), None);
        assert_eq!(token_from(None, None), None);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn private_repo_errors_are_explicit() {
        let hf = hf_url("polonuim210/ochre-refine-2b", "x.gguf", "main");
        let no_token = status_error("x.gguf", &hf, 401, false).to_string();
        assert!(
            no_token.contains("this model is private; set HF_TOKEN"),
            "{no_token}"
        );
        let bad_token = status_error("x.gguf", &hf, 403, true).to_string();
        assert!(bad_token.contains("refused the token"), "{bad_token}");
        let other = status_error("x.bin", "https://example.com/x.bin", 401, false).to_string();
        assert!(
            !other.contains("HF_TOKEN") && other.contains("HTTP 401"),
            "{other}"
        );
        let missing = status_error("x.gguf", &hf, 404, false).to_string();
        assert!(!missing.contains("HF_TOKEN"), "{missing}");
    }

    #[test]
    fn other_hosts_get_no_token_and_a_plain_error() {
        // The server answers 401 and echoes whether an Authorization header arrived.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            if let Ok((mut s, _)) = listener.accept() {
                let mut reader = BufReader::new(s.try_clone().unwrap());
                let mut saw_auth = false;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    saw_auth |= line.to_ascii_lowercase().starts_with("authorization:");
                }
                let status = if saw_auth {
                    "400 Leaked"
                } else {
                    "401 Unauthorized"
                };
                let _ = write!(
                    s,
                    "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
            }
        });
        let dir = tmpdir("401");
        let r = ensure_file(
            &format!("http://{addr}/x.bin"),
            &dir.join("x.bin"),
            None,
            None,
            &|_, _| {},
            &AtomicBool::new(false),
        );
        let e = r.unwrap_err().to_string();
        assert!(e.contains("HTTP 401") && !e.contains("HF_TOKEN"), "{e}");
        fs::remove_dir_all(dir).ok();
    }

    /// Cheap smoke test of the fine-tuned model with the real token: HEAD through the
    /// redirect chain, checking status and size. Downloads nothing.
    #[test]
    #[ignore = "network: HEAD on the ochre-refine-2b GGUF, with and without a token"]
    fn real_hf_head() {
        let url = hf_url(
            "polonuim210/ochre-refine-2b",
            "ochre-refine-2b-v2-Q4_K_M.gguf",
            "main",
        );
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .redirect_auth_headers(ureq::config::RedirectAuthHeaders::SameHost)
            .user_agent(USER_AGENT)
            .build()
            .into();
        let auth = auth_for(&url, hf_token().as_deref()).expect("no HF token found");
        let resp = agent
            .head(&url)
            .header("Authorization", auth)
            .call()
            .unwrap();
        let status = resp.status().as_u16();
        let len: u64 = resp
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        assert_eq!((status, len), (200, 1_274_396_640));
        // The repos are public: no token needed either.
        let anon = agent.head(&url).call().unwrap().status().as_u16();
        assert_eq!(anon, 200, "anonymous HEAD got {anon}");
    }

    #[test]
    fn downloads_verifies_and_skips_when_present() {
        let data = body(3 * CHUNK + 123);
        let (url, hits) = serve(data.clone(), true, None);
        let dir = tmpdir("basic");
        let dest = dir.join("m.bin");
        let never = AtomicBool::new(false);
        let last = std::sync::Mutex::new((0, 0));
        let cb = |d, t| *last.lock().unwrap() = (d, t);
        ensure_file(
            &url,
            &dest,
            Some(&sha(&data)),
            Some(data.len() as u64),
            &cb,
            &never,
        )
        .unwrap();
        assert_eq!(fs::read(&dest).unwrap(), data);
        assert!(!partial_path(&dest).exists());
        assert_eq!(
            *last.lock().unwrap(),
            (data.len() as u64, data.len() as u64)
        );
        // Second call: no network.
        ensure_file(
            &url,
            &dest,
            Some(&sha(&data)),
            Some(data.len() as u64),
            &|_, _| {},
            &never,
        )
        .unwrap();
        assert_eq!(hits.load(Ordering::SeqCst), 1);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn resumes_after_a_dropped_connection() {
        let data = body(2 * CHUNK + 777);
        let (url, hits) = serve(data.clone(), true, Some(CHUNK + 5));
        let dir = tmpdir("resume");
        let dest = dir.join("m.bin");
        let never = AtomicBool::new(false);
        let first = ensure_file(
            &url,
            &dest,
            Some(&sha(&data)),
            Some(data.len() as u64),
            &|_, _| {},
            &never,
        );
        assert!(first.is_err(), "truncated body must not be accepted");
        assert!(!dest.exists());
        assert_eq!(
            fs::metadata(partial_path(&dest)).unwrap().len(),
            (CHUNK + 5) as u64
        );
        ensure_file(
            &url,
            &dest,
            Some(&sha(&data)),
            Some(data.len() as u64),
            &|_, _| {},
            &never,
        )
        .unwrap();
        assert_eq!(fs::read(&dest).unwrap(), data);
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn server_ignoring_range_restarts_cleanly() {
        let data = body(CHUNK + 10);
        let (url, _) = serve(data.clone(), false, None);
        let dir = tmpdir("norange");
        let dest = dir.join("m.bin");
        fs::write(partial_path(&dest), b"garbage-prefix").unwrap();
        ensure_file(
            &url,
            &dest,
            Some(&sha(&data)),
            None,
            &|_, _| {},
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(fs::read(&dest).unwrap(), data);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn bad_checksum_is_rejected_and_partial_removed() {
        let data = body(1000);
        let (url, _) = serve(data.clone(), true, None);
        let dir = tmpdir("badsha");
        let dest = dir.join("m.bin");
        let r = ensure_file(
            &url,
            &dest,
            Some(&"0".repeat(64)),
            None,
            &|_, _| {},
            &AtomicBool::new(false),
        );
        assert!(matches!(r, Err(Error::Download(_))));
        assert!(!dest.exists() && !partial_path(&dest).exists());
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn cancel_keeps_partial() {
        let data = body(4 * CHUNK);
        let (url, _) = serve(data.clone(), true, None);
        let dir = tmpdir("cancel");
        let dest = dir.join("m.bin");
        let cancel = AtomicBool::new(false);
        let cb = |d: u64, _| {
            if d >= CHUNK as u64 {
                cancel.store(true, Ordering::Relaxed)
            }
        };
        let r = ensure_file(&url, &dest, None, Some(data.len() as u64), &cb, &cancel);
        assert!(matches!(r, Err(Error::Canceled)));
        assert!(!dest.exists());
        assert!(fs::metadata(partial_path(&dest)).unwrap().len() >= CHUNK as u64);
        cancel.store(false, Ordering::Relaxed);
        ensure_file(
            &url,
            &dest,
            Some(&sha(&data)),
            Some(data.len() as u64),
            &|_, _| {},
            &cancel,
        )
        .unwrap();
        assert_eq!(fs::read(&dest).unwrap(), data);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn manifest_reports_cumulative_progress() {
        let a = body(5000);
        let (url, _) = serve(a.clone(), true, None);
        let m = ModelManifest {
            id: "toy".into(),
            files: vec![
                ModelFile {
                    name: "a.bin".into(),
                    url: url.clone(),
                    sha256: Some(sha(&a)),
                    size: Some(5000),
                },
                ModelFile {
                    name: "sub/b.bin".into(),
                    url,
                    sha256: None,
                    size: Some(5000),
                },
            ],
        };
        let root = tmpdir("manifest");
        assert!(!m.is_present_in(&root));
        let max = std::sync::Mutex::new((0, 0));
        m.ensure_in(
            &root,
            &|_, d, t| *max.lock().unwrap() = (d, t),
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(*max.lock().unwrap(), (10_000, 10_000));
        assert!(m.is_present_in(&root));
        let json = serde_json::to_string(&m).unwrap();
        assert_eq!(serde_json::from_str::<ModelManifest>(&json).unwrap(), m);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn verified_sidecar_skips_rehash_until_the_file_changes() {
        let data = body(1000);
        let (url, hits) = serve(data.clone(), true, None);
        let dir = tmpdir("sidecar");
        let dest = dir.join("m.bin");
        let never = AtomicBool::new(false);
        let want = sha(&data);
        ensure_file(&url, &dest, Some(&want), Some(1000), &|_, _| {}, &never).unwrap();
        assert!(sidecar_path(&dest).exists());
        // Trusted from the sidecar (no hashing, no network).
        assert!(matches(&dest, Some(&want), Some(1000), true).unwrap());
        // A same-size corruption with a new mtime is caught by re-hashing and re-downloaded.
        std::thread::sleep(std::time::Duration::from_millis(20));
        let mut bad = data.clone();
        bad[0] ^= 0xff;
        fs::write(&dest, &bad).unwrap();
        assert!(!matches(&dest, Some(&want), Some(1000), true).unwrap());
        ensure_file(&url, &dest, Some(&want), Some(1000), &|_, _| {}, &never).unwrap();
        assert_eq!(fs::read(&dest).unwrap(), data);
        assert_eq!(hits.load(Ordering::SeqCst), 2);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn manifest_resumes_automatically_after_a_drop() {
        let data = body(2 * CHUNK + 99);
        let (url, hits) = serve(data.clone(), true, Some(CHUNK / 2));
        let m = ModelManifest {
            id: "retry".into(),
            files: vec![ModelFile {
                name: "a.bin".into(),
                url,
                sha256: Some(sha(&data)),
                size: Some(data.len() as u64),
            }],
        };
        let root = tmpdir("retry");
        m.ensure_in(&root, &|_, _, _| {}, &AtomicBool::new(false))
            .unwrap();
        assert_eq!(fs::read(m.dir_in(&root).join("a.bin")).unwrap(), data);
        assert_eq!(
            hits.load(Ordering::SeqCst),
            2,
            "one dropped attempt, one resumed"
        );
        fs::remove_dir_all(root).ok();
    }

    #[test]
    #[ignore = "network: downloads a small file from huggingface.co (exercises HF redirects + Range)"]
    fn real_hf_download_with_resume() {
        let url = hf_url("istupakov/parakeet-tdt-0.6b-v3-onnx", "vocab.txt", "main");
        let dir = tmpdir("hf");
        let dest = dir.join("vocab.txt");
        ensure_file(&url, &dest, None, None, &|_, _| {}, &AtomicBool::new(false)).unwrap();
        let full = fs::read(&dest).unwrap();
        let want = sha(&full);
        // Simulate an interrupted download, then resume through the redirect chain.
        fs::remove_file(&dest).unwrap();
        fs::write(partial_path(&dest), &full[..full.len() / 2]).unwrap();
        ensure_file(
            &url,
            &dest,
            Some(&want),
            Some(full.len() as u64),
            &|_, _| {},
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(fs::read(&dest).unwrap(), full);
        fs::remove_dir_all(dir).ok();
    }
}
