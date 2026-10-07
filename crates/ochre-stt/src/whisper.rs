//! Whisper large-v3-turbo through whisper.cpp (`whisper-rs`), behind the `whisper` feature
//! (SPEC §4.2). 99 languages; the personal dictionary is passed as the initial prompt, which
//! biases spelling toward those words.
//!
//! Build notes: whisper.cpp is compiled with CMake. Set `WHISPER_DONT_GENERATE_BINDINGS=1` when
//! libclang is not installed (whisper-rs then uses its bundled bindings). GPU: the
//! `whisper-cuda` / `whisper-metal` / `whisper-vulkan` features.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use ochre_core::events::EngineInfo;
use ochre_core::stt::{Progress, ProgressFn, SttEngine, SttOptions, SttResult};
use ochre_core::{Error, Result, SAMPLE_RATE};
use ochre_models::{ModelFile, ModelManifest};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::onnx::Device;
use crate::parakeet::is_speech_like;

pub const ENGINE_ID: &str = "whisper";
pub const DEFAULT_MODEL: &str = "large-v3-turbo-q8_0";
const REPO: &str = "ggerganov/whisper.cpp";
const REV: &str = "5359861c739e955e79d9a303bcbc70fb988958b1";

/// (model id, file, size, sha256, label)
const MODELS: [(&str, &str, u64, &str, &str); 3] = [
    (
        "large-v3-turbo-q8_0",
        "ggml-large-v3-turbo-q8_0.bin",
        874_188_075,
        "317eb69c11673c9de1e1f0d459b253999804ec71ac4c23c17ecf5fbe24e259a1",
        "Whisper large-v3-turbo (q8_0)",
    ),
    (
        "large-v3-turbo-q5_0",
        "ggml-large-v3-turbo-q5_0.bin",
        574_041_195,
        "394221709cd5ad1f40c46e6031ca61bce88931e6e088c188294c6d5a55ffa7e2",
        "Whisper large-v3-turbo (q5_0)",
    ),
    (
        "large-v3-turbo",
        "ggml-large-v3-turbo.bin",
        1_624_555_275,
        "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
        "Whisper large-v3-turbo (f16)",
    ),
];

fn model(
    name: &str,
) -> Result<&'static (&'static str, &'static str, u64, &'static str, &'static str)> {
    let name = if name.is_empty() { DEFAULT_MODEL } else { name };
    MODELS.iter().find(|m| m.0 == name).ok_or_else(|| {
        Error::Config(format!(
            "unknown Whisper model {name:?}; choose one of {}",
            MODELS.iter().map(|m| m.0).collect::<Vec<_>>().join(", ")
        ))
    })
}

pub fn manifest(name: &str) -> Result<ModelManifest> {
    let (id, file, size, sha, _) = *model(name)?;
    Ok(ModelManifest {
        id: format!("whisper-{id}"),
        files: vec![ModelFile::hf(REPO, REV, file, Some(sha), Some(size))],
    })
}

pub fn info() -> EngineInfo {
    EngineInfo {
        id: ENGINE_ID.into(),
        label: "Whisper large-v3-turbo (local)".into(),
        kind: "local".into(),
        models: MODELS.iter().map(|m| m.0.to_string()).collect(),
        default_model: DEFAULT_MODEL.into(),
        needs_key: false,
        note: format!(
            "{} MB download; best with a GPU build",
            model(DEFAULT_MODEL).unwrap().2 / 1_000_000
        ),
        languages: "99 languages (auto)".into(),
    }
}

pub struct Whisper {
    model: &'static (&'static str, &'static str, u64, &'static str, &'static str),
    device: Device,
    models_root: Option<PathBuf>,
    cancel: AtomicBool,
    state: Option<Mutex<WhisperState>>,
    threads: i32,
    pub load_ms: u64,
    pub warmup_ms: u64,
}

impl Whisper {
    pub fn new(model_name: &str, device: &str) -> Result<Self> {
        // ggml busy-waits at its barriers, so never ask for more threads than free physical
        // cores (docs/latency.md rule 1). Even so, CPU whisper.cpp degrades badly under
        // background load; prefer a GPU build (`whisper-cuda` / `whisper-metal` / `whisper-vulkan`).
        let threads = crate::onnx::heavy_threads() as i32;
        Ok(Self {
            model: model(model_name)?,
            device: Device::parse(device),
            models_root: None,
            cancel: AtomicBool::new(false),
            state: None,
            threads,
            load_ms: 0,
            warmup_ms: 0,
        })
    }

    pub fn with_models_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.models_root = Some(root.into());
        self
    }

    pub fn load_from_file(&mut self, path: &Path) -> Result<()> {
        whisper_rs::install_logging_hooks(); // route whisper.cpp's chatter into `log`, not stderr
        let t0 = Instant::now();
        let mut cp = WhisperContextParameters::default();
        cp.use_gpu(self.device != Device::Cpu);
        let ctx = WhisperContext::new_with_params(path, cp)
            .map_err(|e| Error::Model(format!("whisper: {e}")))?;
        let state = ctx
            .create_state()
            .map_err(|e| Error::Model(format!("whisper: {e}")))?;
        self.load_ms = t0.elapsed().as_millis() as u64;
        self.state = Some(Mutex::new(state));
        let t1 = Instant::now();
        let noise: Vec<f32> = (0..SAMPLE_RATE as usize)
            .map(|i| ((i as f32) * 0.37).sin() * 0.003)
            .collect();
        self.run(&noise, &SttOptions::default())?;
        self.warmup_ms = t1.elapsed().as_millis() as u64;
        tracing::info!(
            model = self.model.0,
            load_ms = self.load_ms,
            warmup_ms = self.warmup_ms,
            "whisper ready"
        );
        Ok(())
    }

    fn run(&self, pcm: &[f32], opts: &SttOptions) -> Result<(String, Option<String>)> {
        let state = self
            .state
            .as_ref()
            .ok_or_else(|| Error::Model("whisper: load() was not called".into()))?;
        let mut state = state.lock().unwrap_or_else(|p| p.into_inner());
        let mut p = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        p.set_n_threads(self.threads);
        p.set_language(Some(opts.language.as_deref().unwrap_or("auto")));
        let prompt = if opts.vocabulary.is_empty() {
            String::new()
        } else {
            format!("{}.", opts.vocabulary.join(", "))
        };
        if !prompt.is_empty() {
            p.set_initial_prompt(&prompt);
        }
        p.set_no_context(true);
        p.set_no_timestamps(true);
        p.set_suppress_blank(true);
        p.set_print_special(false);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_timestamps(false);
        // whisper.cpp wants at least 1 s; pad short phrases with silence.
        let padded;
        let pcm = if pcm.len() < SAMPLE_RATE as usize + SAMPLE_RATE as usize / 10 {
            padded = [
                pcm,
                &vec![0.0; SAMPLE_RATE as usize + SAMPLE_RATE as usize / 10 - pcm.len()],
            ]
            .concat();
            &padded[..]
        } else {
            pcm
        };
        state
            .full(p, pcm)
            .map_err(|e| Error::Model(format!("whisper: {e}")))?;
        let mut text = String::new();
        for seg in state.as_iter() {
            if let Ok(s) = seg.to_str_lossy() {
                text.push_str(&s);
            }
        }
        let lang = whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(str::to_string);
        Ok((text.trim().to_string(), lang))
    }
}

impl SttEngine for Whisper {
    fn info(&self) -> EngineInfo {
        info()
    }

    fn load(&mut self, progress: ProgressFn) -> Result<()> {
        if self.state.is_some() {
            return Ok(());
        }
        let m = manifest(self.model.0)?;
        let root = self
            .models_root
            .clone()
            .unwrap_or_else(ochre_core::paths::models_dir);
        let report = |item: &str, done: u64, total: u64| {
            progress(Progress {
                item: item.to_string(),
                done,
                total,
            })
        };
        let dir = m.ensure_in(&root, &report, &self.cancel)?;
        self.load_from_file(&dir.join(self.model.1))
    }

    fn transcribe(&self, pcm: &[f32], opts: &SttOptions) -> Result<SttResult> {
        let t0 = Instant::now();
        let duration_ms = (pcm.len() as u64 * 1000) / SAMPLE_RATE as u64;
        let (text, language) = if is_speech_like(pcm) {
            let mut text = String::new();
            let mut language = None;
            for chunk in
                crate::chunk::split_at_pauses(pcm, SAMPLE_RATE as usize, 30 * SAMPLE_RATE as usize)
            {
                let (t, l) = self.run(chunk, opts)?;
                if !t.is_empty() {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(&t);
                }
                language = language.or(l);
            }
            (text, language)
        } else {
            (String::new(), None)
        };
        Ok(SttResult {
            text,
            duration_ms,
            processing_ms: t0.elapsed().as_millis() as u64,
            language,
        })
    }
}
