//! One managed `llama-server` process serving one GGUF on 127.0.0.1 (SPEC §5.2).
//!
//! Started once and kept resident (model loaded, prompt cache warm), restarted when it dies, and
//! killed with us: on Windows the child goes into a Job object with KILL_ON_JOB_CLOSE, so it dies
//! even if this process is killed; on Linux the child gets PR_SET_PDEATHSIG (through
//! `ochre_core::linux_child`, which forks from a thread that lives as long as we do). macOS has
//! neither, so there the server runs under a small `/bin/sh` wrapper ([`MAC_WATCHDOG`]) that holds
//! the read end of a pipe from us: when we exit for any reason (even `kill -9`) the pipe hits EOF
//! and the wrapper stops the server. No polling.
//!
//! Nothing here sends text anywhere but 127.0.0.1, and the server log never contains prompts (no
//! `--verbose`).

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ochre_core::{Error, Result};
use serde_json::{Value, json};

use crate::http;

#[derive(Debug, Clone, PartialEq)]
pub struct ServerOptions {
    pub ctx: u32,
    pub threads: u32,
    /// None = all layers on GPU builds, 0 on CPU builds.
    pub gpu_layers: Option<u32>,
    pub batch: u32,
    /// `--prio/--prio-batch 2`: the CUDA launch thread starves under CPU load without it.
    pub high_priority: bool,
    /// n-gram prompt-lookup speculative decoding. ~2x CPU generation; slower on GPU.
    pub ngram_spec: bool,
    pub extra: Vec<String>,
}

impl ServerOptions {
    /// The tuned defaults from docs/refinement.md §3.2 for an accelerator.
    pub fn tuned(accel: &str, threads: u32) -> Self {
        let cpu = accel == "cpu";
        let threads = if threads > 0 {
            threads
        } else if cpu {
            // Measured: all logical cores beat physical cores for this 0.5 GB model.
            std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4)
        } else {
            // GPU builds only use CPU threads for glue work.
            (std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(4)
                / 2)
            .clamp(2, 8)
        };
        ServerOptions {
            ctx: 2048,
            threads,
            gpu_layers: None,
            batch: 1024,
            high_priority: true,
            ngram_spec: cpu,
            extra: vec![],
        }
    }
}

pub fn free_port() -> Result<u16> {
    Ok(TcpListener::bind("127.0.0.1:0")?.local_addr()?.port())
}

struct Proc {
    child: Child,
    #[cfg(windows)]
    _job: job::Job,
}

impl Drop for Proc {
    fn drop(&mut self) {
        // macOS: closing the watchdog pipe makes the wrapper stop the server and exit; give it a
        // moment so the server is gone when we return (the app may exit right after).
        #[cfg(target_os = "macos")]
        if self.child.stdin.take().is_some() {
            let deadline = Instant::now() + Duration::from_secs(3);
            while Instant::now() < deadline {
                if !matches!(self.child.try_wait(), Ok(None)) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// macOS: runs `"$@"` (the server) and stops it when stdin (a pipe from us) reaches EOF, i.e. when
/// we close it or die. Exits with the server's status, so `alive()` and restarts work as without
/// it. The pipe is duplicated to fd 3 first because a background job's stdin is /dev/null.
#[cfg(any(target_os = "macos", test))]
pub const MAC_WATCHDOG: &str = r#"exec 3<&0 </dev/null
"$@" 3<&- &
pid=$!
(cat <&3 >/dev/null; kill -TERM "$pid" 2>/dev/null) &
watch=$!
exec 3<&-
wait "$pid"
status=$?
kill "$watch" 2>/dev/null
exit "$status""#;

pub struct LlamaServer {
    pub exe: PathBuf,
    pub model: PathBuf,
    pub accel: String,
    pub options: ServerOptions,
    pub log_path: PathBuf,
    port: AtomicU16,
    proc: Mutex<Option<Proc>>,
    restarting: AtomicBool,
    stopping: AtomicBool,
    pub restarts: AtomicU32,
    client: reqwest::blocking::Client,
}

impl LlamaServer {
    pub fn new(exe: &Path, model: &Path, accel: &str, options: ServerOptions) -> Self {
        LlamaServer {
            exe: exe.to_path_buf(),
            model: model.to_path_buf(),
            accel: accel.to_string(),
            options,
            log_path: ochre_core::paths::data_dir()
                .join("logs")
                .join("llama-server.log"),
            port: AtomicU16::new(0),
            proc: Mutex::new(None),
            restarting: AtomicBool::new(false),
            stopping: AtomicBool::new(false),
            restarts: AtomicU32::new(0),
            client: http::local_client(),
        }
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port.load(Ordering::Acquire))
    }

    pub fn args(&self, port: u16) -> Vec<String> {
        let o = &self.options;
        let ngl = o
            .gpu_layers
            .unwrap_or(if self.accel == "cpu" { 0 } else { 999 });
        let mut a: Vec<String> = [
            "-m",
            &self.model.to_string_lossy(),
            "--host",
            "127.0.0.1",
            "--port",
            &port.to_string(),
            "-c",
            &o.ctx.to_string(),
            "-np",
            "1",
            "-t",
            &o.threads.to_string(),
            "-ngl",
            &ngl.to_string(),
            "-fa",
            "auto",
            "-b",
            &o.batch.to_string(),
            // --no-jinja is explicit: this build enables jinja by default, and jinja makes Quill leak
            // its chain of thought. --cache-ram 0: the RAM prompt cache only helps exact repeats.
            "--no-webui",
            "--no-jinja",
            "--cache-prompt",
            "--no-context-shift",
            "--fit",
            "off",
            "--cache-ram",
            "0",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        if o.high_priority {
            a.extend(["--prio", "2", "--prio-batch", "2"].map(String::from));
        }
        if o.ngram_spec {
            a.extend(
                [
                    "--spec-type",
                    "ngram-simple",
                    "--spec-ngram-simple-size-n",
                    "3",
                    "--spec-ngram-simple-size-m",
                    "16",
                ]
                .map(String::from),
            );
        }
        a.extend(o.extra.iter().cloned());
        a
    }

    pub fn alive(&self) -> bool {
        let mut g = self.proc.lock().unwrap();
        match g.as_mut() {
            Some(p) => matches!(p.child.try_wait(), Ok(None)),
            None => false,
        }
    }

    /// Start (or restart) the process and block until `/health` reports the model loaded.
    pub fn start(&self, timeout: Duration) -> Result<()> {
        self.stopping.store(false, Ordering::Release);
        if self.alive() {
            return Ok(());
        }
        let port = free_port()?;
        if let Some(dir) = self.log_path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;
        #[cfg(target_os = "macos")]
        let mut cmd = {
            let mut c = Command::new("/bin/sh");
            c.arg("-c")
                .arg(MAC_WATCHDOG)
                .arg("llama-server")
                .arg(&self.exe)
                .stdin(Stdio::piped());
            c
        };
        #[cfg(not(target_os = "macos"))]
        let mut cmd = {
            let mut c = Command::new(&self.exe);
            c.stdin(Stdio::null());
            c
        };
        cmd.args(self.args(port))
            .current_dir(self.exe.parent().unwrap_or(Path::new(".")))
            .stdout(log.try_clone()?)
            .stderr(log);
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
        }
        tracing::info!(
            "starting llama-server ({}, {}) on port {port}",
            self.accel,
            self.model.display()
        );
        // Linux: PR_SET_PDEATHSIG through the process-lifetime spawner thread. Set here directly,
        // it fired as soon as the engine-loading thread finished and killed the fresh server.
        #[cfg(target_os = "linux")]
        let child = ochre_core::linux_child::spawn_tied(cmd);
        #[cfg(not(target_os = "linux"))]
        let child = cmd.spawn();
        let child =
            child.map_err(|e| Error::Model(format!("llama-server failed to start: {e}")))?;
        #[cfg(windows)]
        let proc = Proc {
            _job: job::Job::kill_on_close(&child),
            child,
        };
        #[cfg(not(windows))]
        let proc = Proc { child };
        *self.proc.lock().unwrap() = Some(proc);
        self.port.store(port, Ordering::Release);
        self.wait_healthy(timeout)
    }

    fn wait_healthy(&self, timeout: Duration) -> Result<()> {
        // Startup only (seconds, once), never on the dictation path.
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            {
                let mut g = self.proc.lock().unwrap();
                match g.as_mut().map(|p| p.child.try_wait()) {
                    Some(Ok(None)) => {}
                    Some(Ok(Some(code))) => {
                        *g = None;
                        return Err(Error::Model(format!(
                            "llama-server exited during startup ({code}); see {}",
                            self.log_path.display()
                        )));
                    }
                    _ => return Err(Error::Model("llama-server is not running".into())),
                }
            }
            if self.healthy() {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        self.stop();
        Err(Error::Model(format!(
            "llama-server did not become healthy in {timeout:?}; see {}",
            self.log_path.display()
        )))
    }

    pub fn healthy(&self) -> bool {
        self.client
            .get(format!("{}/health", self.url()))
            .timeout(Duration::from_secs(1))
            .send()
            .is_ok_and(|r| r.status().as_u16() == 200)
    }

    /// Restart a dead server in the background and return false, so a refine call falls back to raw
    /// text instead of blocking for seconds. True if it is running.
    pub fn ensure_running(self: &Arc<Self>) -> bool {
        if self.alive() {
            return true;
        }
        if self.stopping.load(Ordering::Acquire) || self.restarting.swap(true, Ordering::AcqRel) {
            return false;
        }
        let n = self.restarts.fetch_add(1, Ordering::Relaxed) + 1;
        tracing::warn!("llama-server not running; restart #{n}");
        let me = self.clone();
        std::thread::Builder::new()
            .name("llama-restart".into())
            .spawn(move || {
                match me.start(Duration::from_secs(90)) {
                    // Re-prime the default system prompt so the next dictation is fast again.
                    Ok(()) => {
                        let ctx = ochre_core::refine::RefineContext::default();
                        let _ = me.prime(&crate::local::priming_prefix(
                            &crate::prompts::system_prompt(&ctx),
                        ));
                    }
                    Err(e) => tracing::error!("llama-server restart failed: {e}"),
                }
                me.restarting.store(false, Ordering::Release);
            })
            .ok();
        false
    }

    pub fn stop(&self) {
        self.stopping.store(true, Ordering::Release);
        self.proc.lock().unwrap().take(); // Drop kills and reaps
    }

    /// Raw `/completion`, greedy and prompt-cached. Returns the server JSON (`content`, `timings`).
    pub fn completion(&self, prompt: &str, n_predict: u32, timeout: Duration) -> Result<Value> {
        let body = json!({
            "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "cache_prompt": true,
            "stop": crate::prompts::STOP, "stream": false,
        });
        http::post_json(
            &self.client,
            "local",
            &format!("{}/completion", self.url()),
            &[],
            &body,
            timeout,
        )
        .map_err(|e| e.error)
    }

    /// Process `prefix` into the slot without generating, so the next request only evaluates its
    /// own tokens (docs/refinement.md §3.4: Qwen3.5's recurrent state cannot roll back to a prefix).
    pub fn prime(&self, prefix: &str) -> Result<()> {
        let body = json!({"prompt": prefix, "n_predict": 0, "cache_prompt": true, "stream": false});
        http::post_json(
            &self.client,
            "local",
            &format!("{}/completion", self.url()),
            &[],
            &body,
            Duration::from_secs(30),
        )
        .map(|_| ())
        .map_err(|e| e.error)
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(windows)]
mod job {
    use std::os::windows::io::AsRawHandle;

    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    /// A Job object that kills its processes when the last handle closes, i.e. when this process
    /// exits however it exits (crash, kill, Task Manager).
    pub struct Job(HANDLE);

    // SAFETY: a job handle is a kernel handle, usable from any thread.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        pub fn kill_on_close(child: &std::process::Child) -> Job {
            unsafe {
                let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if job.is_null() {
                    tracing::warn!("CreateJobObjectW failed; llama-server may outlive a crash");
                    return Job(std::ptr::null_mut());
                }
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    job,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const _,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                );
                if ok == 0 || AssignProcessToJobObject(job, child.as_raw_handle() as HANDLE) == 0 {
                    tracing::warn!("job object setup failed; llama-server may outlive a crash");
                }
                Job(job)
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe { CloseHandle(self.0) };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    fn watchdog(args: &[&str]) -> Child {
        Command::new("/bin/sh")
            .arg("-c")
            .arg(MAC_WATCHDOG)
            .arg("llama-server")
            .args(args)
            .stdin(Stdio::piped())
            .spawn()
            .unwrap()
    }

    #[cfg(unix)]
    fn running(pattern: &str) -> bool {
        Command::new("pgrep")
            .args(["-f", pattern])
            .output()
            .is_ok_and(|o| !o.stdout.is_empty())
    }

    #[cfg(unix)]
    #[test]
    fn mac_watchdog_passes_status_and_stops_child_on_eof() {
        // the server's exit status comes through, so alive() / restarts see a dead server
        // (polled: `wait()` would close stdin first, which is the stop signal)
        let mut c = watchdog(&["/bin/sh", "-c", "exit 3"]);
        let t0 = Instant::now();
        let status = loop {
            if let Some(st) = c.try_wait().unwrap() {
                break st;
            }
            assert!(t0.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(status.code(), Some(3));

        // our end of the pipe closing (drop, or this process dying) stops the server
        let mut c = watchdog(&["/bin/sleep", "31.4159"]);
        let t0 = Instant::now();
        while !running("sleep 31.4159") {
            assert!(t0.elapsed() < Duration::from_secs(5), "child never started");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(matches!(c.try_wait(), Ok(None)));
        drop(c.stdin.take());
        let t0 = Instant::now();
        while matches!(c.try_wait(), Ok(None)) {
            assert!(t0.elapsed() < Duration::from_secs(5), "wrapper didn't exit");
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(!running("sleep 31.4159"), "server left behind");
    }

    #[test]
    fn tuned_flags() {
        let s = LlamaServer::new(
            Path::new("llama-server"),
            Path::new("m.gguf"),
            "cpu",
            ServerOptions::tuned("cpu", 6),
        );
        let a = s.args(1234).join(" ");
        assert!(
            a.contains("-t 6")
                && a.contains("-ngl 0")
                && a.contains("--no-jinja")
                && a.contains("--cache-ram 0")
        );
        assert!(a.contains("--prio 2 --prio-batch 2") && a.contains("--spec-type ngram-simple"));
        let g = LlamaServer::new(
            Path::new("llama-server"),
            Path::new("m.gguf"),
            "cuda",
            ServerOptions::tuned("cuda", 0),
        );
        let a = g.args(1).join(" ");
        assert!(a.contains("-ngl 999") && !a.contains("--spec-type") && !a.contains("--jinja "));
    }
}
