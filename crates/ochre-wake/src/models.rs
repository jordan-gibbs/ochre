//! The two openWakeWord front-end models and a tiny verified downloader.
//!
//! TODO(ochre-models): switch to `ochre_models::ensure_file` once that crate lands; the semantics here
//! are the same (download to `<name>.partial`, check size and sha256, then rename).

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ochre_core::{Error, Result};
use sha2::{Digest, Sha256};

/// One downloadable model file with its pinned size and hash.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFile {
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// openWakeWord v0.5.1 release assets (Apache-2.0), `docs/wakeword.md` §1.
pub const MEL_MODEL: ModelFile = ModelFile {
    name: "melspectrogram.onnx",
    url: "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    sha256: "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    size: 1_087_958,
};

pub const EMBED_MODEL: ModelFile = ModelFile {
    name: "embedding_model.onnx",
    url: "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
    sha256: "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    size: 1_326_578,
};

/// Where the front-end models live: `<models_dir>/openwakeword` (shared with the Python reference).
pub fn frontend_dir() -> PathBuf {
    ochre_core::paths::models_dir().join("openwakeword")
}

/// Download progress: `(file name, bytes done, bytes total)`.
pub type Progress<'a> = &'a dyn Fn(&str, u64, u64);

/// Make sure both front-end models are in `dir` (downloading once); returns `(mel, embedding)`.
pub fn ensure_frontend_models(
    dir: &Path,
    progress: Option<Progress<'_>>,
) -> Result<(PathBuf, PathBuf)> {
    Ok((
        ensure_file(&MEL_MODEL, dir, progress)?,
        ensure_file(&EMBED_MODEL, dir, progress)?,
    ))
}

/// `dir/<name>`, downloaded and verified if it is missing or wrong.
pub fn ensure_file(f: &ModelFile, dir: &Path, progress: Option<Progress<'_>>) -> Result<PathBuf> {
    let dest = dir.join(f.name);
    if dest.metadata().map(|m| m.len() == f.size).unwrap_or(false)
        && sha256_file(&dest)? == f.sha256
    {
        return Ok(dest);
    }
    fs::create_dir_all(dir)?;
    let partial = dir.join(format!("{}.partial", f.name));
    tracing::info!(url = f.url, dest = %dest.display(), "downloading wake front-end model");
    let resp = ureq::get(f.url)
        .call()
        .map_err(|e| Error::Download(format!("{}: {e}", f.url)))?;
    let mut reader = resp.into_body().into_reader();
    let mut out = fs::File::create(&partial)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    let mut done = 0u64;
    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| Error::Download(format!("{}: {e}", f.url)))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
        done += n as u64;
        if let Some(p) = progress {
            p(f.name, done, f.size);
        }
    }
    out.flush()?;
    drop(out);
    let got = hex(&hasher.finalize());
    if done != f.size || got != f.sha256 {
        let _ = fs::remove_file(&partial);
        return Err(Error::Download(format!(
            "{}: got {done} bytes sha256 {got}, expected {} bytes sha256 {}",
            f.name, f.size, f.sha256
        )));
    }
    fs::rename(&partial, &dest)?;
    Ok(dest)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
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

/// The shipped "transcribe" head and its metadata, compiled in so the app needs no resource
/// bundling (about 1 MB).
pub const BUNDLED: &[(&str, &[u8])] = &[
    (
        "transcribe.onnx",
        include_bytes!("../../../assets/wake/transcribe.onnx"),
    ),
    (
        "transcribe.json",
        include_bytes!("../../../assets/wake/transcribe.json"),
    ),
];

/// Where [`install_bundled`] puts the shipped heads: `<models_dir>/wake`.
pub fn bundled_dir() -> PathBuf {
    ochre_core::paths::models_dir().join("wake")
}

/// Write the compiled-in wake heads into `dir` (only files that are missing or differ, via a
/// `.partial` + rename) and return `dir`, ready for [`crate::resolve_model`].
pub fn install_bundled(dir: &Path) -> Result<PathBuf> {
    fs::create_dir_all(dir)?;
    for (name, bytes) in BUNDLED {
        let dest = dir.join(name);
        if fs::read(&dest).is_ok_and(|have| have == *bytes) {
            continue;
        }
        let partial = dir.join(format!("{name}.partial"));
        fs::write(&partial, bytes)?;
        fs::rename(&partial, &dest)?;
    }
    Ok(dir.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_heads_install_and_resolve() {
        let dir = std::env::temp_dir().join(format!("ochre-wake-bundled-{}", std::process::id()));
        let got = install_bundled(&dir).unwrap();
        let model = crate::resolve_model("transcribe", "", &got).unwrap();
        assert!(crate::recommended_threshold(&model).is_some());
        install_bundled(&dir).unwrap(); // idempotent
        let _ = fs::remove_dir_all(&dir);
    }
}
