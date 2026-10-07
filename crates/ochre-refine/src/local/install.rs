//! Provisioning for the local refiner: the prebuilt llama.cpp release for this OS / arch /
//! accelerator, and the refinement GGUF (Quill, or our fine-tuned Ochre Refine). Both download
//! on first use through `ochre_models::ensure_file` (resumable, size + sha256 verified, atomic) and
//! are reused from disk afterwards with no network.
//!
//! Layout (shared with the dropped Python prototype, so its downloads are reused):
//! `models_dir()/llama.cpp/<tag>/<variant>/` with a `.complete` marker whose first line is the
//! accelerator, and `models_dir()/refine/<file>.gguf`.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use ochre_core::stt::{Progress, ProgressFn};
use ochre_core::{Error, Result};
use serde_json::Value;

/// Pinned llama.cpp build (docs/refinement.md §3.1). Bump deliberately and re-run the bench.
pub const LLAMA_TAG: &str = "b11398";
const RELEASE_API: &str = "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/";
const DOWNLOAD: &str = "https://github.com/ggml-org/llama.cpp/releases/download/";

/// Minimum NVIDIA driver major for each CUDA major the release ships builds for.
const CUDA_MIN_DRIVER: &[(u32, u32)] = &[(13, 580), (12, 528)];

static NO_CANCEL: AtomicBool = AtomicBool::new(false);

/// (os, arch) in llama.cpp release naming: os in win|macos|ubuntu, arch in x64|arm64.
pub fn host() -> (&'static str, &'static str) {
    let os = if cfg!(windows) {
        "win"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "ubuntu"
    };
    let arch = if cfg!(target_arch = "aarch64") {
        "arm64"
    } else {
        "x64"
    };
    (os, arch)
}

#[cfg(windows)]
const NO_WINDOW: u32 = 0x0800_0000;

/// Major version of the NVIDIA driver, or None if there is no NVIDIA GPU/driver.
pub fn nvidia_driver_major() -> Option<u32> {
    let mut cmd = std::process::Command::new("nvidia-smi");
    cmd.args(["--query-gpu=driver_version", "--format=csv,noheader"]);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(NO_WINDOW);
    }
    let out = cmd.output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().next()?.trim().split('.').next()?.parse().ok()
}

/// "auto" -> metal on Apple Silicon, cuda with an NVIDIA driver (x64), else cpu.
pub fn pick_accel(accel: &str) -> String {
    if !accel.is_empty() && accel != "auto" {
        return accel.to_string();
    }
    match host() {
        ("macos", "arm64") => "metal".into(),
        ("macos", _) => "cpu".into(),
        (_, "x64") if nvidia_driver_major().is_some() => "cuda".into(),
        _ => "cpu".into(),
    }
}

/// One llama.cpp binary variant: the archives to fetch and where they unpack.
#[derive(Debug, Clone, PartialEq)]
pub struct Build {
    /// e.g. "win-cuda-13.4-x64"
    pub variant: String,
    /// cpu | cuda | vulkan | metal
    pub accel: String,
    /// (asset name, size, sha256)
    pub assets: Vec<(String, u64, Option<String>)>,
}

/// Choose this host's archives from a release's asset list (`[(name, size, sha256)]`). Names are
/// matched by pattern, not hard-coded, so a tag bump picks up new CUDA versions.
pub fn select_build(
    assets: &[(String, u64, Option<String>)],
    tag: &str,
    accel: &str,
    os: &str,
    arch: &str,
    driver_major: u32,
) -> Result<Build> {
    let need = |name: String| -> Result<(String, u64, Option<String>)> {
        assets
            .iter()
            .find(|a| a.0 == name)
            .cloned()
            .ok_or_else(|| Error::Download(format!("llama.cpp {tag} has no {name}")))
    };
    if os == "macos" {
        let acc = if arch == "arm64" { "metal" } else { "cpu" };
        return Ok(Build {
            variant: format!("macos-{arch}"),
            accel: acc.into(),
            assets: vec![need(format!("llama-{tag}-bin-macos-{arch}.tar.gz"))?],
        });
    }
    let ext = if os == "win" { "zip" } else { "tar.gz" };
    match accel {
        "cuda" => {
            let prefix = format!("llama-{tag}-bin-{os}-cuda-");
            let suffix = format!("-{arch}.{ext}");
            let mut options: Vec<(u32, u32, String)> = assets
                .iter()
                .filter_map(|(n, _, _)| {
                    let ver = n.strip_prefix(&prefix)?.strip_suffix(&suffix)?;
                    let (ma, mi) = ver.split_once('.')?;
                    Some((ma.parse().ok()?, mi.parse().ok()?, n.clone()))
                })
                .collect();
            options.sort_by_key(|o| std::cmp::Reverse((o.0, o.1)));
            for (major, minor, name) in options {
                let min = CUDA_MIN_DRIVER
                    .iter()
                    .find(|(m, _)| *m == major)
                    .map(|(_, d)| *d)
                    .unwrap_or(u32::MAX);
                if driver_major >= min {
                    let mut list = vec![need(name)?];
                    // The CUDA runtime DLLs ship separately (Windows: cudart-llama-bin-win-...,
                    // Linux: cudart-llama-<tag>-bin-ubuntu-...); unpack them beside the server.
                    let cudart = assets.iter().find(|(n, _, _)| {
                        n.starts_with("cudart-")
                            && n.contains(&format!("-{os}-cuda-{major}.{minor}-{arch}."))
                            && n.ends_with(ext)
                    });
                    if let Some(c) = cudart {
                        list.push(c.clone());
                    } else if os == "win" {
                        return Err(Error::Download(format!(
                            "llama.cpp {tag} has no CUDA {major}.{minor} runtime for {os}-{arch}"
                        )));
                    }
                    return Ok(Build {
                        variant: format!("{os}-cuda-{major}.{minor}-{arch}"),
                        accel: "cuda".into(),
                        assets: list,
                    });
                }
            }
            Err(Error::Download(format!(
                "no CUDA build of llama.cpp {tag} supports NVIDIA driver {driver_major}"
            )))
        }
        "vulkan" => Ok(Build {
            variant: format!("{os}-vulkan-{arch}"),
            accel: "vulkan".into(),
            assets: vec![need(format!("llama-{tag}-bin-{os}-vulkan-{arch}.{ext}"))?],
        }),
        _ => {
            let name = if os == "win" {
                format!("llama-{tag}-bin-win-cpu-{arch}.zip")
            } else {
                format!("llama-{tag}-bin-{os}-{arch}.{ext}")
            };
            Ok(Build {
                variant: format!("{os}-cpu-{arch}"),
                accel: "cpu".into(),
                assets: vec![need(name)?],
            })
        }
    }
}

pub fn server_exe_name() -> &'static str {
    if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    }
}

/// `llama-server` inside an unpacked variant dir (archives unpack flat, but search one level down
/// in case a future tag nests them).
pub fn find_server(dir: &Path) -> Option<PathBuf> {
    let direct = dir.join(server_exe_name());
    if direct.is_file() {
        return Some(direct);
    }
    std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path().join(server_exe_name()))
        .find(|p| p.is_file())
}

fn release_assets(tag: &str) -> Result<Vec<(String, u64, Option<String>)>> {
    let client = crate::http::client();
    let resp = client
        .get(format!("{RELEASE_API}{tag}"))
        .header("Accept", "application/vnd.github+json")
        .timeout(Duration::from_secs(20))
        .send()
        .map_err(|e| crate::http::transport_error("github", &e, Duration::from_secs(20)))?;
    let status = resp.status().as_u16();
    let text = resp.text().unwrap_or_default();
    if status != 200 {
        return Err(Error::Download(format!(
            "github release {tag}: HTTP {status}"
        )));
    }
    let v: Value = serde_json::from_str(&text)
        .map_err(|e| Error::Download(format!("github release {tag}: {e}")))?;
    Ok(v["assets"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    let name = x["name"].as_str()?.to_string();
                    let size = x["size"].as_u64().unwrap_or(0);
                    let sha = x["digest"]
                        .as_str()
                        .and_then(|d| d.strip_prefix("sha256:"))
                        .map(str::to_string);
                    Some((name, size, sha))
                })
                .collect()
        })
        .unwrap_or_default())
}

pub fn llama_root() -> PathBuf {
    ochre_core::paths::models_dir()
        .join("llama.cpp")
        .join(LLAMA_TAG)
}

/// An already-installed variant for this accel (no network).
pub fn installed_binary(accel: &str) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(llama_root())
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter().find_map(|d| {
        let marker = std::fs::read_to_string(d.join(".complete")).ok()?;
        (marker.lines().next() == Some(accel))
            .then(|| find_server(&d))
            .flatten()
    })
}

fn bridge<'a>(item: &'a str, progress: ProgressFn<'a>) -> impl Fn(u64, u64) + 'a {
    move |done, total| {
        progress(Progress {
            item: item.to_string(),
            done,
            total,
        })
    }
}

/// Path to `llama-server` and the accel actually installed, downloading on first use.
pub fn ensure_binary(accel: &str, progress: ProgressFn) -> Result<(PathBuf, String)> {
    let accel = pick_accel(accel);
    if let Some(exe) = installed_binary(&accel) {
        return Ok((exe, accel));
    }
    let (os, arch) = host();
    let drv = if accel == "cuda" {
        nvidia_driver_major().unwrap_or(0)
    } else {
        0
    };
    let build = select_build(
        &release_assets(LLAMA_TAG)?,
        LLAMA_TAG,
        &accel,
        os,
        arch,
        drv,
    )?;
    let base = llama_root();
    let dest = base.join(&build.variant);
    std::fs::create_dir_all(&dest)?;
    for (name, size, sha) in &build.assets {
        let url = format!("{DOWNLOAD}{LLAMA_TAG}/{name}");
        let label = format!("llama.cpp ({})", build.accel);
        let archive = ochre_models::ensure_file(
            &url,
            &base.join(name),
            sha.as_deref(),
            Some(*size).filter(|s| *s > 0),
            &bridge(&label, progress),
            &NO_CANCEL,
        )?;
        extract(&archive, &dest)?;
    }
    let exe = find_server(&dest)
        .ok_or_else(|| Error::Download(format!("llama-server not found in {}", build.variant)))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Some(dir) = exe.parent() {
            for f in std::fs::read_dir(dir)?.flatten() {
                let p = f.path();
                let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if p.is_file()
                    && (n.starts_with("llama-") || n.contains(".so") || n.ends_with(".dylib"))
                {
                    let mut perm = std::fs::metadata(&p)?.permissions();
                    perm.set_mode(perm.mode() | 0o755);
                    std::fs::set_permissions(&p, perm)?;
                }
            }
        }
    }
    let names: Vec<&str> = build.assets.iter().map(|a| a.0.as_str()).collect();
    std::fs::write(
        dest.join(".complete"),
        format!("{}\n{}", build.accel, names.join("\n")),
    )?;
    for name in names {
        let _ = std::fs::remove_file(base.join(name)); // archives are only needed until extracted
    }
    Ok((exe, build.accel))
}

fn safe_join(dest: &Path, member: &str) -> Result<PathBuf> {
    let p = Path::new(member);
    if p.is_absolute()
        || member.starts_with(['/', '\\'])
        || p.components().any(|c| {
            matches!(
                c,
                std::path::Component::ParentDir | std::path::Component::Prefix(_)
            )
        })
    {
        return Err(Error::Download(format!("unsafe path in archive: {member}")));
    }
    Ok(dest.join(p))
}

/// Unpack a .zip or .tar.gz, refusing path traversal.
pub fn extract(archive: &Path, dest: &Path) -> Result<()> {
    let name = archive.to_string_lossy();
    let file = std::fs::File::open(archive)?;
    if name.ends_with(".zip") {
        let mut z =
            zip::ZipArchive::new(file).map_err(|e| Error::Download(format!("{name}: {e}")))?;
        for i in 0..z.len() {
            let mut entry = z
                .by_index(i)
                .map_err(|e| Error::Download(format!("{name}: {e}")))?;
            let out = safe_join(dest, entry.name())?;
            if entry.is_dir() {
                std::fs::create_dir_all(&out)?;
                continue;
            }
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let mut f = std::fs::File::create(&out)?;
            std::io::copy(&mut entry, &mut f)?;
        }
    } else {
        let mut t = tar::Archive::new(flate2::read::GzDecoder::new(file));
        for entry in t.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.to_string_lossy().into_owned();
            safe_join(dest, &path)?;
            entry.unpack_in(dest)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------- models

pub const QUILL_REPO: &str = "Quobi/Quill";
/// Our fine-tunes of Qwen3.5-4B / 2B / 0.8B (round 4, docs/refine-finetune-v4.md; the v3 files
/// stay in the repos). Public, Apache-2.0.
pub const OCHRE_REFINE_4B_REPO: &str = "polonuim210/ochre-refine-4b";
pub const OCHRE_REFINE_2B_REPO: &str = "polonuim210/ochre-refine-2b";
pub const OCHRE_REFINE_08B_REPO: &str = "polonuim210/ochre-refine-0.8b";
/// "auto" picks an Ochre Refine size by hardware, falling back to the same-size Quill model if
/// the download fails (`auto.rs`).
pub const DEFAULT_MODEL: &str = super::auto::AUTO;

/// One downloadable local refinement model, verified on download (docs/refinement.md §3.1).
/// Every entry uses the same prompt (ChatML, `<dictation>` tags, pre-seeded empty think block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalModel {
    /// Config / settings id, e.g. "quill-2b".
    pub name: &'static str,
    /// Human label.
    pub label: &'static str,
    /// Hugging Face repo.
    pub repo: &'static str,
    pub file: &'static str,
    pub size: u64,
    pub sha256: &'static str,
}

pub const MODELS: &[LocalModel] = &[
    LocalModel {
        name: "ochre-refine-4b",
        label: "Ochre Refine 4B",
        repo: OCHRE_REFINE_4B_REPO,
        file: "ochre-refine-4b-v4-Q4_K_M.gguf",
        size: 2_708_804_480,
        sha256: "baed73b8c2db25121abcc36a38efdade18020243d8250455b0621b8af5d0c485",
    },
    LocalModel {
        name: "ochre-refine-2b",
        label: "Ochre Refine 2B",
        repo: OCHRE_REFINE_2B_REPO,
        file: "ochre-refine-2b-v4-Q4_K_M.gguf",
        size: 1_274_396_640,
        sha256: "096fd42fedcfff134ea751e61e060a76cf198bb0a36989c16fb0e81f112e5683",
    },
    LocalModel {
        name: "ochre-refine-0.8b",
        label: "Ochre Refine 0.8B (CPU)",
        repo: OCHRE_REFINE_08B_REPO,
        file: "ochre-refine-0.8b-v4-Q4_K_M.gguf",
        size: 529_297_376,
        sha256: "169e5c938a8ddb55aa41080106b4b2fe5b2727b96db34adeab4131c03871a1bc",
    },
    LocalModel {
        name: "quill-0.8b",
        label: "Quill 0.8B",
        repo: QUILL_REPO,
        file: "quill-0.8b-Q4_K_M.gguf",
        size: 529_296_832,
        sha256: "aa54d6f6108d66e4b60a57bdc04ecca6e84e073504918a64b41ac4a0f816f16d",
    },
    LocalModel {
        name: "quill-2b",
        label: "Quill 2B",
        repo: QUILL_REPO,
        file: "quill-2b-Q4_K_M.gguf",
        size: 1_274_396_096,
        sha256: "b877a22b773d2aac40b3c642c24f1cbbb0b3f1d42cbd3c6eb936533719317196",
    },
    LocalModel {
        name: "quill-4b",
        label: "Quill 4B",
        repo: QUILL_REPO,
        file: "quill-4b-Q4_K_M.gguf",
        size: 2_708_803_936,
        sha256: "e5e6bd7e92690c6f954399c473e740561d9deff0862e1bfe42c1f6055535b987",
    },
];

/// A known model by id (case-insensitive, an optional `-q4_k_m` suffix ignored).
pub fn model(name: &str) -> Option<&'static LocalModel> {
    let key = name.to_lowercase();
    let key = key.trim_end_matches("-q4_k_m");
    MODELS.iter().find(|m| m.name == key)
}

/// Download URL of a known model.
pub fn model_url(m: &LocalModel) -> String {
    ochre_models::hf_url(m.repo, m.file, "main")
}

/// Where a known model lives on disk.
pub fn model_path(m: &LocalModel) -> PathBuf {
    ochre_core::paths::models_dir().join("refine").join(m.file)
}

/// Already on disk (size check only, no hashing).
pub fn is_downloaded(m: &LocalModel) -> bool {
    std::fs::metadata(model_path(m)).is_ok_and(|md| md.len() == m.size)
}

/// Path to a local GGUF: a known model (downloaded on first use), "auto" (resolved for this
/// machine with the default accelerator) or an explicit file path.
pub fn ensure_model(name: &str, progress: ProgressFn) -> Result<PathBuf> {
    if super::auto::is_auto(name) {
        return super::auto::ensure("auto", progress).map(|(p, _)| p);
    }
    if name.to_lowercase().ends_with(".gguf") {
        let p = PathBuf::from(name);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(Error::Model(format!(
                "model file not found: {}",
                p.display()
            )))
        };
    }
    let Some(m) = model(name) else {
        let known: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
        return Err(Error::Config(format!(
            "unknown local refinement model {name:?}; choose auto, one of {} or a .gguf path",
            known.join(", ")
        )));
    };
    let dest = model_path(m);
    // Hashing 0.5-2.7 GB on every launch is too slow; size + the atomic rename suffice.
    if is_downloaded(m) {
        return Ok(dest);
    }
    ochre_models::ensure_file(
        &model_url(m),
        &dest,
        Some(m.sha256),
        Some(m.size),
        &bridge(m.file, progress),
        &NO_CANCEL,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assets(names: &[&str]) -> Vec<(String, u64, Option<String>)> {
        names.iter().map(|n| (n.to_string(), 1, None)).collect()
    }

    const T: &str = "b11398";

    #[test]
    fn selects_windows_cuda_13_with_runtime() {
        let a = assets(&[
            "llama-b11398-bin-win-cuda-12.4-x64.zip",
            "llama-b11398-bin-win-cuda-13.4-x64.zip",
            "cudart-llama-bin-win-cuda-12.4-x64.zip",
            "cudart-llama-bin-win-cuda-13.4-x64.zip",
            "llama-b11398-bin-win-cpu-x64.zip",
        ]);
        let b = select_build(&a, T, "cuda", "win", "x64", 591).unwrap();
        assert_eq!(b.variant, "win-cuda-13.4-x64");
        assert_eq!(
            b.assets.iter().map(|x| x.0.as_str()).collect::<Vec<_>>(),
            [
                "llama-b11398-bin-win-cuda-13.4-x64.zip",
                "cudart-llama-bin-win-cuda-13.4-x64.zip"
            ]
        );
        // An older driver gets CUDA 12.
        assert_eq!(
            select_build(&a, T, "cuda", "win", "x64", 560)
                .unwrap()
                .variant,
            "win-cuda-12.4-x64"
        );
        assert!(select_build(&a, T, "cuda", "win", "x64", 400).is_err());
        assert_eq!(
            select_build(&a, T, "cpu", "win", "x64", 0).unwrap().assets[0].0,
            "llama-b11398-bin-win-cpu-x64.zip"
        );
    }

    #[test]
    fn selects_linux_and_macos() {
        let a = assets(&[
            "llama-b11398-bin-ubuntu-x64.tar.gz",
            "llama-b11398-bin-ubuntu-vulkan-x64.tar.gz",
            "llama-b11398-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "cudart-llama-b11398-bin-ubuntu-cuda-13.4-x64.tar.gz",
            "llama-b11398-bin-macos-arm64.tar.gz",
        ]);
        let b = select_build(&a, T, "cuda", "ubuntu", "x64", 590).unwrap();
        assert_eq!(b.assets.len(), 2);
        assert_eq!(
            select_build(&a, T, "cpu", "ubuntu", "x64", 0)
                .unwrap()
                .assets[0]
                .0,
            "llama-b11398-bin-ubuntu-x64.tar.gz"
        );
        assert_eq!(
            select_build(&a, T, "vulkan", "ubuntu", "x64", 0)
                .unwrap()
                .variant,
            "ubuntu-vulkan-x64"
        );
        let m = select_build(&a, T, "auto", "macos", "arm64", 0).unwrap();
        assert_eq!(
            (m.variant.as_str(), m.accel.as_str()),
            ("macos-arm64", "metal")
        );
    }

    #[test]
    fn refuses_traversal() {
        let d = Path::new("x");
        assert!(
            safe_join(d, "../evil").is_err()
                && safe_join(d, "/abs").is_err()
                && safe_join(d, "ok/file.dll").is_ok()
        );
    }

    #[test]
    fn unknown_model_is_config_error() {
        let p = |_: Progress| {};
        assert!(matches!(
            ensure_model("quill-9b", &p),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            ensure_model("C:/nope/x.gguf", &p),
            Err(Error::Model(_))
        ));
    }

    #[test]
    fn fine_tuned_entries() {
        let cases = [
            (
                "ochre-refine-4b",
                "Ochre Refine 4B",
                "ochre-refine-4b-v4-Q4_K_M.gguf",
                2_708_804_480,
                "baed73b8c2db25121abcc36a38efdade18020243d8250455b0621b8af5d0c485",
            ),
            (
                "ochre-refine-2b",
                "Ochre Refine 2B",
                "ochre-refine-2b-v4-Q4_K_M.gguf",
                1_274_396_640,
                "096fd42fedcfff134ea751e61e060a76cf198bb0a36989c16fb0e81f112e5683",
            ),
            (
                "ochre-refine-0.8b",
                "Ochre Refine 0.8B (CPU)",
                "ochre-refine-0.8b-v4-Q4_K_M.gguf",
                529_297_376,
                "169e5c938a8ddb55aa41080106b4b2fe5b2727b96db34adeab4131c03871a1bc",
            ),
        ];
        for (name, label, file, size, sha) in cases {
            let m = model(name).expect("listed");
            assert_eq!(m.repo, format!("polonuim210/{name}"));
            assert_eq!(
                (m.label, m.file, m.size, m.sha256),
                (label, file, size, sha)
            );
            assert_eq!(
                model_url(m),
                format!("https://huggingface.co/polonuim210/{name}/resolve/main/{file}")
            );
        }
        assert_eq!(
            model("OCHRE-REFINE-2B-Q4_K_M").map(|m| m.file),
            Some("ochre-refine-2b-v4-Q4_K_M.gguf")
        );
        // The default is auto, not a fixed (possibly private) model.
        assert_eq!(DEFAULT_MODEL, "auto");
        assert!(model(DEFAULT_MODEL).is_none());
        let mut names: Vec<&str> = MODELS.iter().map(|m| m.name).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), MODELS.len(), "model ids are unique");
        assert!(MODELS.iter().all(|m| m.sha256.len() == 64));
    }
}
