//! `refine.model = "auto"` (the default): pick the local cleanup model for this machine.
//!
//! Rules (docs/refine-finetune-v4.md: 4B 80%, 2B 71%, 0.8B 64% on 773 rows, 3 blind judges):
//! - a usable GPU with >= 6 GB of VRAM, or Apple Silicon with >= 16 GB unified memory -> 4B
//! - a usable GPU with less memory, or memory we can't read -> 2B
//! - no usable GPU (llama-server would run on the CPU) -> 0.8B
//!
//! "Usable" follows the accelerator llama-server will actually run on (`install::pick_accel`):
//! CUDA with an NVIDIA driver, Metal on Apple Silicon, or Vulkan when `refine.local_accel` asks
//! for it (AMD / Intel). Memory comes from `nvidia-smi` (CUDA), DXGI (Windows, any vendor), the
//! DRM sysfs (Linux: amdgpu `mem_info_vram_total`, or Intel `lmem_total_bytes` where exposed)
//! or `sysctl hw.memsize` (Apple). Detection runs once per accelerator and is cached.
//!
//! Auto always picks an Ochre Refine model. If it can't be downloaded (offline, Hugging Face
//! down), [`ensure`] falls back to the Quill model of the same size, so the default still works.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{LazyLock, Mutex};

use ochre_core::Result;
use ochre_core::stt::ProgressFn;

use super::install::{self, LocalModel};

pub const AUTO: &str = "auto";

/// `refine.model` values that mean "pick for me" ("" is the default, which is auto).
pub fn is_auto(name: &str) -> bool {
    let n = name.trim();
    n.is_empty() || n.eq_ignore_ascii_case(AUTO)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gpu {
    /// llama-server runs on the CPU.
    None,
    Nvidia,
    Amd,
    Apple,
    /// Intel or unknown vendor.
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hardware {
    pub gpu: Gpu,
    /// Dedicated VRAM, or unified memory on Apple Silicon, in MiB. None = unknown.
    pub memory_mib: Option<u64>,
}

impl Hardware {
    pub const CPU: Hardware = Hardware {
        gpu: Gpu::None,
        memory_mib: None,
    };

    /// e.g. "NVIDIA GPU, 16 GB" / "no GPU".
    pub fn describe(&self) -> String {
        let kind = match self.gpu {
            Gpu::None => return "no usable GPU".into(),
            Gpu::Nvidia => "NVIDIA GPU",
            Gpu::Amd => "AMD GPU",
            Gpu::Apple => "Apple Silicon",
            Gpu::Other => "GPU",
        };
        match self.memory_mib {
            Some(m) => format!("{kind}, {} GB", (m + 512) / 1024),
            None => kind.into(),
        }
    }
}

/// Model size class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// 4B
    Large,
    /// 2B
    Medium,
    /// 0.8B
    Small,
}

/// "6 GB" cards report a little under 6 GiB on some APIs; 5.5 GiB keeps them in.
pub const GPU_LARGE_MIB: u64 = 5632;
/// 16 GB Macs (16384 MiB) and up.
pub const APPLE_LARGE_MIB: u64 = 15 * 1024;

pub fn tier(hw: &Hardware) -> Tier {
    let min = match hw.gpu {
        Gpu::None => return Tier::Small,
        Gpu::Apple => APPLE_LARGE_MIB,
        Gpu::Nvidia | Gpu::Amd | Gpu::Other => GPU_LARGE_MIB,
    };
    match hw.memory_mib {
        Some(m) if m >= min => Tier::Large,
        _ => Tier::Medium,
    }
}

/// The model id for a tier: Ochre Refine, or (`ochre = false`) the Quill model of that size.
pub fn model_for(tier: Tier, ochre: bool) -> &'static str {
    match (tier, ochre) {
        (Tier::Large, true) => "ochre-refine-4b",
        (Tier::Medium, true) => "ochre-refine-2b",
        (Tier::Small, true) => "ochre-refine-0.8b",
        (Tier::Large, false) => "quill-4b",
        (Tier::Medium, false) => "quill-2b",
        (Tier::Small, false) => "quill-0.8b",
    }
}

/// The Quill model of the same size as an Ochre Refine one (the download fallback).
pub fn public_equivalent(name: &str) -> Option<&'static str> {
    match name {
        "ochre-refine-4b" => Some("quill-4b"),
        "ochre-refine-2b" => Some("quill-2b"),
        "ochre-refine-0.8b" => Some("quill-0.8b"),
        _ => None,
    }
}

/// Pure resolution from the hardware.
pub fn choose(hw: &Hardware) -> &'static str {
    model_for(tier(hw), true)
}

/// What auto resolves to on this machine for the given `refine.local_accel`.
pub fn resolve(accel: &str) -> (&'static LocalModel, Hardware) {
    let hw = detect(accel);
    let name = choose(&hw);
    (install::model(name).expect("listed"), hw)
}

/// Resolve auto and make sure its file is on disk. If the Ochre Refine pick can't be fetched
/// (offline, Hugging Face unreachable) the same-size Quill model is used instead; if that fails
/// too, the original error is returned.
pub fn ensure(accel: &str, progress: ProgressFn) -> Result<(PathBuf, &'static LocalModel)> {
    let (m, hw) = resolve(accel);
    tracing::info!("refine auto: {} ({})", m.name, hw.describe());
    match install::ensure_model(m.name, progress) {
        Ok(p) => Ok((p, m)),
        Err(e) => {
            let Some(q) = public_equivalent(m.name).and_then(install::model) else {
                return Err(e);
            };
            tracing::warn!(
                "refine auto: {} unavailable ({e}); using {}",
                m.name,
                q.name
            );
            install::ensure_model(q.name, progress)
                .map(|p| (p, q))
                .map_err(|_| e)
        }
    }
}

static CACHE: LazyLock<Mutex<HashMap<String, Hardware>>> = LazyLock::new(Default::default);

/// This machine's GPU as llama-server will see it (cached per accelerator setting).
pub fn detect(accel: &str) -> Hardware {
    let accel = install::pick_accel(accel);
    if let Some(hw) = CACHE.lock().unwrap().get(&accel) {
        return *hw;
    }
    let hw = probe(&accel);
    tracing::info!("refine auto: accel {accel}, {}", hw.describe());
    CACHE.lock().unwrap().insert(accel, hw);
    hw
}

fn probe(accel: &str) -> Hardware {
    match accel {
        "cpu" => Hardware::CPU,
        "metal" => Hardware {
            gpu: Gpu::Apple,
            memory_mib: apple_memory_mib(),
        },
        "cuda" => Hardware {
            gpu: Gpu::Nvidia,
            memory_mib: nvidia_vram_mib().or_else(|| dxgi_best().map(|(_, m)| m)),
        },
        // vulkan or anything else GPU-ish
        _ => match dxgi_best().or_else(drm_best) {
            Some((gpu, m)) => Hardware {
                gpu,
                memory_mib: Some(m),
            },
            None => Hardware {
                gpu: Gpu::Other,
                memory_mib: None,
            },
        },
    }
}

/// Largest `memory.total` across NVIDIA GPUs, MiB.
fn nvidia_vram_mib() -> Option<u64> {
    let mut cmd = std::process::Command::new("nvidia-smi");
    cmd.args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_nvidia_smi(&String::from_utf8_lossy(&out.stdout))
}

pub fn parse_nvidia_smi(text: &str) -> Option<u64> {
    text.lines()
        .filter_map(|l| l.trim().parse::<u64>().ok())
        .max()
}

fn apple_memory_mib() -> Option<u64> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    let bytes: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
    Some(bytes / (1024 * 1024))
}

/// The hardware adapter with the most dedicated VRAM (DXGI; software adapters skipped).
#[cfg(windows)]
fn dxgi_best() -> Option<(Gpu, u64)> {
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1,
    };
    // SAFETY: plain COM calls on interfaces we own; DXGI needs no COM apartment.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let mut best: Option<(Gpu, u64)> = None;
        let mut i = 0;
        while let Ok(adapter) = factory.EnumAdapters1(i) {
            i += 1;
            let Ok(desc) = adapter.GetDesc1() else {
                continue;
            };
            if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                continue;
            }
            let mib = desc.DedicatedVideoMemory as u64 / (1024 * 1024);
            let gpu = match desc.VendorId {
                0x10DE => Gpu::Nvidia,
                0x1002 | 0x1022 => Gpu::Amd,
                _ => Gpu::Other,
            };
            if best.is_none_or(|(_, m)| mib > m) {
                best = Some((gpu, mib));
            }
        }
        best
    }
}

#[cfg(not(windows))]
fn dxgi_best() -> Option<(Gpu, u64)> {
    None
}

/// Linux: the DRM device with the most dedicated VRAM, from sysfs. amdgpu reports
/// `device/mem_info_vram_total`, and Intel discrete cards `lmem_total_bytes` where the driver
/// exposes it (both in bytes).
/// NVIDIA is covered by `nvidia-smi`, and integrated GPUs without a report are skipped.
#[cfg(target_os = "linux")]
fn drm_best() -> Option<(Gpu, u64)> {
    let read = |p: std::path::PathBuf| std::fs::read_to_string(p).ok();
    let cards: Vec<_> = std::fs::read_dir("/sys/class/drm")
        .ok()?
        .flatten()
        .filter(|e| {
            let n = e.file_name();
            let n = n.to_string_lossy();
            n.starts_with("card") && n[4..].chars().all(|c| c.is_ascii_digit())
        })
        .map(|e| {
            let card = e.path();
            let vram = read(card.join("device/mem_info_vram_total"))
                .or_else(|| read(card.join("lmem_total_bytes")))
                .or_else(|| read(card.join("device/lmem_total_bytes")));
            (read(card.join("device/vendor")).unwrap_or_default(), vram)
        })
        .collect();
    best_drm(&cards)
}

#[cfg(not(target_os = "linux"))]
fn drm_best() -> Option<(Gpu, u64)> {
    None
}

/// Pure part of [`drm_best`]: `(vendor id like "0x1002", vram bytes)` per card.
pub fn best_drm(cards: &[(String, Option<String>)]) -> Option<(Gpu, u64)> {
    cards
        .iter()
        .filter_map(|(vendor, vram)| {
            let bytes: u64 = vram.as_deref()?.trim().parse().ok()?;
            let gpu = match vendor.trim() {
                "0x10de" => Gpu::Nvidia,
                "0x1002" | "0x1022" => Gpu::Amd,
                _ => Gpu::Other,
            };
            Some((gpu, bytes / (1024 * 1024)))
        })
        .filter(|(_, mib)| *mib > 0)
        .max_by_key(|(_, mib)| *mib)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hw(gpu: Gpu, gb: Option<u64>) -> Hardware {
        Hardware {
            gpu,
            memory_mib: gb.map(|g| g * 1024),
        }
    }

    #[test]
    fn picks_by_gpu_memory() {
        let cases = [
            (hw(Gpu::Nvidia, Some(16)), "ochre-refine-4b"), // RTX 5070 Ti
            (hw(Gpu::Nvidia, Some(6)), "ochre-refine-4b"),
            (hw(Gpu::Amd, Some(8)), "ochre-refine-4b"),
            (hw(Gpu::Nvidia, Some(4)), "ochre-refine-2b"),
            (hw(Gpu::Other, Some(2)), "ochre-refine-2b"),
            (hw(Gpu::Nvidia, None), "ochre-refine-2b"), // unknown memory: safe middle
            (hw(Gpu::Other, None), "ochre-refine-2b"),
            (hw(Gpu::Apple, Some(16)), "ochre-refine-4b"),
            (hw(Gpu::Apple, Some(32)), "ochre-refine-4b"),
            (hw(Gpu::Apple, Some(8)), "ochre-refine-2b"),
            (hw(Gpu::Apple, None), "ochre-refine-2b"),
            (Hardware::CPU, "ochre-refine-0.8b"),
            (hw(Gpu::None, Some(64)), "ochre-refine-0.8b"),
        ];
        for (h, want) in cases {
            assert_eq!(choose(&h), want, "{h:?}");
        }
        // A "6 GB" card that reports slightly less still counts.
        let six = Hardware {
            gpu: Gpu::Nvidia,
            memory_mib: Some(6 * 1024 - 100),
        };
        assert_eq!(tier(&six), Tier::Large);
        assert_eq!(
            tier(&Hardware {
                gpu: Gpu::Nvidia,
                memory_mib: Some(5 * 1024)
            }),
            Tier::Medium
        );
    }

    #[test]
    fn quill_is_the_same_size_fallback() {
        for t in [Tier::Large, Tier::Medium, Tier::Small] {
            let o = model_for(t, true);
            let q = model_for(t, false);
            assert_eq!(public_equivalent(o), Some(q));
            assert!(install::model(o).is_some() && install::model(q).is_some());
            assert_eq!(install::model(q).unwrap().repo, install::QUILL_REPO);
        }
        assert_eq!(public_equivalent("quill-2b"), None);
    }

    #[test]
    fn auto_names_and_parsing() {
        assert!(is_auto("") && is_auto("auto") && is_auto(" AUTO ") && !is_auto("quill-2b"));
        assert_eq!(parse_nvidia_smi("16303\n"), Some(16303));
        assert_eq!(parse_nvidia_smi("8192\r\n24564\r\n"), Some(24564));
        assert_eq!(parse_nvidia_smi("[N/A]\n"), None);
        assert_eq!(parse_nvidia_smi(""), None);
        assert_eq!(hw(Gpu::Nvidia, Some(16)).describe(), "NVIDIA GPU, 16 GB");
        let gib = |g: u64| {
            Some(
                (g << 30).to_string()
                    + "
",
            )
        };
        let cards = [
            (
                "0x8086
"
                .to_string(),
                None,
            ), // integrated Intel: no report
            (
                "0x1002
"
                .to_string(),
                gib(16),
            ), // RX 7800 XT
            (
                "0x1002
"
                .to_string(),
                Some(
                    "536870912
"
                    .into(),
                ),
            ), // APU carve-out
            (
                "0x8086
"
                .to_string(),
                Some("garbage".into()),
            ),
        ];
        assert_eq!(best_drm(&cards), Some((Gpu::Amd, 16 * 1024)));
        assert_eq!(best_drm(&cards[..1]), None);
        assert_eq!(
            best_drm(&[("0x8086".into(), gib(12))]),
            Some((Gpu::Other, 12 * 1024))
        );
        assert_eq!(Hardware::CPU.describe(), "no usable GPU");
        assert_eq!(
            detect("cpu"),
            Hardware::CPU,
            "a forced CPU build means the 0.8B"
        );
    }

    #[test]
    #[ignore = "probes this machine's GPU (nvidia-smi / DXGI / DRM sysfs / sysctl)"]
    fn this_machine() {
        let (m, hw) = resolve("auto");
        println!(
            "auto -> {} ({hw:?}, {}), dxgi {:?}, drm {:?}",
            m.name,
            hw.describe(),
            dxgi_best(),
            drm_best()
        );
    }
}
