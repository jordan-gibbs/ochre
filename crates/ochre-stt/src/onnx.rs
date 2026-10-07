//! Shared ONNX Runtime plumbing for local engines: execution-provider selection with graceful
//! CPU fallback, and session construction tuned for low-latency single-stream inference.

use std::path::Path;

use ochre_core::{Error, Result};
#[allow(unused_imports)]
use ort::ep::{self, ExecutionProviderDispatch};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

/// Where to run. Parsed from `SttConfig::device` ("auto" | "cpu" | "cuda" | "directml" | "coreml").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Auto,
    Cpu,
    Cuda,
    DirectMl,
    CoreMl,
}

impl Device {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Device::Cpu,
            "cuda" | "gpu" => Device::Cuda,
            "directml" | "dml" => Device::DirectMl,
            "coreml" => Device::CoreMl,
            _ => Device::Auto,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Device::Auto => "auto",
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
            Device::DirectMl => "directml",
            Device::CoreMl => "coreml",
        }
    }

    /// Accelerators compiled into this build, in "auto" preference order.
    pub fn compiled() -> Vec<Device> {
        let mut v = Vec::new();
        if cfg!(feature = "cuda") {
            v.push(Device::Cuda);
        }
        if cfg!(feature = "coreml") {
            v.push(Device::CoreMl);
        }
        if cfg!(feature = "directml") {
            v.push(Device::DirectMl);
        }
        v
    }

    /// The concrete devices to try, in order; CPU is always last.
    pub fn candidates(self) -> Vec<Device> {
        let mut v = match self {
            Device::Auto => Device::compiled(),
            Device::Cpu => Vec::new(),
            d if Device::compiled().contains(&d) => vec![d],
            d => {
                tracing::warn!(
                    device = d.as_str(),
                    "requested device is not compiled into this build; using CPU"
                );
                Vec::new()
            }
        };
        v.push(Device::Cpu);
        v
    }

    fn provider(self) -> Option<ExecutionProviderDispatch> {
        match self {
            #[cfg(feature = "cuda")]
            Device::Cuda => Some(ep::CUDA::default().build().error_on_failure()),
            #[cfg(feature = "directml")]
            Device::DirectMl => Some(ep::DirectML::default().build().error_on_failure()),
            #[cfg(feature = "coreml")]
            Device::CoreMl => Some(ep::CoreML::default().build().error_on_failure()),
            _ => None,
        }
    }
}

/// How a session will be used; picks threading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Workload {
    /// One big graph run per utterance (an encoder): use the intra-op thread pool.
    Heavy,
    /// Thousands of tiny runs (a transducer decoder step): one thread, no pool hand-offs.
    Tiny,
}

/// Intra-op threads for heavy graphs: physical cores minus one (leave a core for the capture
/// pump, the UI and whatever else the user runs), overridable with `OCHRE_ORT_THREADS`.
pub fn heavy_threads() -> usize {
    std::env::var("OCHRE_ORT_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| crate::priority::physical_cores().saturating_sub(1).max(1))
}

/// Graph optimization level (`OCHRE_ORT_OPT` = 0..3 for experiments; default Level3).
fn opt_level() -> GraphOptimizationLevel {
    match std::env::var("OCHRE_ORT_OPT").ok().as_deref() {
        Some("0") => GraphOptimizationLevel::Disable,
        Some("1") => GraphOptimizationLevel::Level1,
        Some("2") => GraphOptimizationLevel::Level2,
        _ => GraphOptimizationLevel::Level3,
    }
}

/// ORT thread-pool workers as std threads at raised priority (new threads do not inherit the
/// creator's priority on Windows).
struct BoostedThreads;

impl ort::environment::ThreadManager for BoostedThreads {
    type Thread = std::thread::JoinHandle<()>;

    fn create(&self, work: impl FnOnce() + Send + 'static) -> ort::Result<Self::Thread> {
        std::thread::Builder::new()
            .name("ochre-ort".into())
            .spawn(move || {
                crate::priority::boost_current_thread();
                work();
            })
            .map_err(|e| ort::Error::new(e.to_string()))
    }

    fn join(thread: Self::Thread) -> ort::Result<()> {
        let _ = thread.join();
        Ok(())
    }
}

pub fn model_err(what: &str, e: impl std::fmt::Display) -> Error {
    Error::Model(format!("{what}: {e}"))
}

fn build(path: &Path, device: Device, workload: Workload) -> std::result::Result<Session, String> {
    let mut b = Session::builder()
        .map_err(|e| e.to_string())?
        .with_optimization_level(opt_level())
        .map_err(|e| e.to_string())?;
    // No spinning anywhere: spinning pools burn idle CPU and, under background load, spin-waiting
    // workers fight the threads doing real work (docs/latency.md rule 1).
    let spin = std::env::var("OCHRE_ORT_SPIN").is_ok_and(|v| v == "1");
    b = b
        .with_intra_op_spinning(spin)
        .map_err(|e| e.to_string())?
        .with_inter_op_spinning(spin)
        .map_err(|e| e.to_string())?
        .with_inter_threads(1)
        .map_err(|e| e.to_string())?;
    b = match workload {
        Workload::Heavy => b
            .with_intra_threads(heavy_threads())
            .map_err(|e| e.to_string())?
            // Pool workers are created by our manager so they run at raised priority too.
            .with_thread_manager(BoostedThreads)
            .map_err(|e| e.to_string())?,
        // One thread: the work runs on the (already boosted) calling decode thread.
        Workload::Tiny => b.with_intra_threads(1).map_err(|e| e.to_string())?,
    };
    if device == Device::DirectMl {
        // DirectML requires sequential execution and no memory pattern.
        b = b
            .with_parallel_execution(false)
            .map_err(|e| e.to_string())?
            .with_memory_pattern(false)
            .map_err(|e| e.to_string())?;
    }
    if let Some(ep) = device.provider() {
        b = b
            .with_execution_providers([ep])
            .map_err(|e| e.to_string())?;
    }
    b.commit_from_file(path).map_err(|e| e.to_string())
}

/// Build a session on the first device in `device.candidates()` that works. Returns the session
/// and the device actually used. A broken GPU runtime (missing CUDA/cuDNN DLLs, unsupported
/// driver) logs a warning and falls back to the next candidate, ending at CPU.
pub fn session(path: &Path, device: Device, workload: Workload) -> Result<(Session, Device)> {
    let mut last = None;
    for d in device.candidates() {
        match build(path, d, workload) {
            Ok(s) => return Ok((s, d)),
            Err(e) => {
                tracing::warn!(device = d.as_str(), model = %path.display(), error = %e, "session failed; trying next device");
                last = Some(e);
            }
        }
    }
    Err(model_err(
        &format!("could not load {}", path.display()),
        last.unwrap_or_default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_parsing_and_candidates() {
        assert_eq!(Device::parse("CUDA"), Device::Cuda);
        assert_eq!(Device::parse("whatever"), Device::Auto);
        assert_eq!(Device::Cpu.candidates(), vec![Device::Cpu]);
        assert_eq!(*Device::Auto.candidates().last().unwrap(), Device::Cpu);
        if !cfg!(feature = "coreml") {
            assert_eq!(Device::CoreMl.candidates(), vec![Device::Cpu]);
        }
    }
}
