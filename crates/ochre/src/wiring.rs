//! The real engine loader: config -> live engines. The GUI and the headless CLI both use
//! [`default_loader`]; tests use fakes instead.
//!
//! The warm mic is opened once and reused across reloads (re-opened only when the device or the
//! warm setting changes), so changing a refinement setting never re-opens the microphone.

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tracing::{info, warn};

use ochre_audio::{CaptureOptions, DecodeThread, Earcon, Earcons, PhraseSegmenter, WarmMic};
use ochre_core::config::Config;
use ochre_core::events::EngineInfo;
use ochre_core::stt::{Progress, SttEngine, SttResult};
use ochre_core::{Error, Result};

use crate::app::{Cue, Engines, Loader};
use crate::seams::{
    LevelFn, Mic, PartialFn, PhraseFn, SegmentSession, SegmenterFactory, Sink, Tap, TranscribeFn,
};

// ---------------------------------------------------------------------------- seam adapters

impl Mic for WarmMic {
    fn begin(&self, preroll_ms: u32, sink: Sink, level: LevelFn) -> Result<()> {
        WarmMic::begin(self, preroll_ms, sink, level)
    }
    fn end(&self) -> Result<u64> {
        WarmMic::end(self)
    }
    fn wait_closed(&self, timeout: Duration) {
        WarmMic::wait_closed(self, timeout)
    }
    fn cancel(&self) {
        WarmMic::cancel(self)
    }
    fn set_tap(&self, tap: Option<Tap>) -> Result<()> {
        WarmMic::set_tap(self, tap)
    }
    fn begin_from(&self, from: u64, sink: Sink, level: LevelFn) -> Result<()> {
        WarmMic::begin_from(self, from, sink, level)
    }
    fn reopen(&self) {
        WarmMic::reopen(self)
    }
    fn prime(&self) {
        WarmMic::prime(self)
    }
}

impl SegmenterFactory for PhraseSegmenter {
    fn start(
        &self,
        transcribe: TranscribeFn,
        on_partial: Option<PartialFn>,
    ) -> Box<dyn SegmentSession> {
        Box::new(PhraseSegmenter::start(self, transcribe, on_partial))
    }

    fn start_with_phrases(
        &self,
        transcribe: TranscribeFn,
        on_partial: Option<PartialFn>,
        on_phrase: PhraseFn,
    ) -> Box<dyn SegmentSession> {
        // The segment hook fires for cut phrases only; previews of the open phrase skip it.
        let on_segment = Arc::new(move |_: usize, r: &SttResult| on_phrase(r));
        Box::new(PhraseSegmenter::start_with_segments(
            self, transcribe, on_partial, on_segment,
        ))
    }
}

impl SegmentSession for ochre_audio::Segmenter {
    fn feed(&mut self, pcm: &[f32]) {
        ochre_audio::Segmenter::feed(self, pcm)
    }
    fn release(&mut self) {
        ochre_audio::Segmenter::release(self)
    }
    fn finish(self: Box<Self>) -> Result<Vec<SttResult>> {
        Ok((*self).finish()?.phrases)
    }
    fn cancel(self: Box<Self>) {
        (*self).cancel()
    }
}

// ---------------------------------------------------------------------------- registry

/// Every speech engine this build offers: local first, then cloud.
pub fn stt_engines() -> Vec<EngineInfo> {
    let mut v = ochre_stt::local_engines();
    v.extend(ochre_cloud::engines());
    v
}

pub fn refine_engines() -> Vec<EngineInfo> {
    ochre_refine::engines()
}

/// A local engine compiled into this build.
fn is_local(engine: &str) -> bool {
    ochre_stt::local_engines().iter().any(|e| e.id == engine)
}

/// Whether this build can run the speech engine `id` (what Settings may offer).
pub fn stt_available(id: &str) -> bool {
    stt_engines().iter().any(|e| e.id == id)
}

/// Route a speech engine id to its factory. A local engine that is not compiled into this build
/// (Whisper without the `whisper` feature) gets a clear error; it is never handed to the cloud
/// factory, which would report it as an unknown cloud provider.
fn create_stt(cfg: &Config) -> Result<Box<dyn SttEngine>> {
    let id = cfg.stt.engine.as_str();
    if is_local(id) {
        return ochre_stt::create(id, &cfg.stt.model, &cfg.stt.device);
    }
    if let Some(name) = ochre_stt::known_local_label(id) {
        return Err(Error::Config(format!(
            "The {name} speech engine is not included in this build of Ochre. Choose another speech model in Settings › Transcription."
        )));
    }
    if ochre_cloud::engines().iter().any(|e| e.id == id) {
        return ochre_cloud::create(id, &cfg.stt);
    }
    Err(Error::Config(format!(
        "Unknown speech engine {id:?}. Choose a speech model in Settings › Transcription."
    )))
}

// ---------------------------------------------------------------------------- loader

#[derive(Default)]
struct MicCache {
    key: Option<(Option<String>, bool, bool, bool, u64)>,
    mic: Option<Arc<WarmMic>>,
}

/// Builds `Engines` from config. Models download (with progress) and warm up here, on the
/// orchestrator's loader thread.
pub fn default_loader() -> Loader {
    let mics = Arc::new(Mutex::new(MicCache::default()));
    let decoder = DecodeThread::spawn(None);
    Arc::new(
        move |cfg: &Config, progress: &(dyn Fn(Progress) + Send + Sync)| -> Result<Engines> {
            // Speech-to-text.
            let mut stt = create_stt(cfg)?;
            stt.load(progress)?;
            let stt: Arc<dyn SttEngine> = Arc::from(stt);
            info!(engine = %stt.info().id, "speech engine ready");

            // Local fallback for cloud engines, only when its model is already on disk: a cloud
            // user never gets a surprise 670 MB download.
            let fallback_stt = if is_local(&cfg.stt.engine) || !cfg.stt.fallback_local {
                None
            } else {
                local_fallback(cfg)
            };

            // Refinement.
            let refiner = if cfg.refine.provider == "off" {
                None
            } else {
                let mut r = ochre_refine::create(&cfg.refine)?;
                match r.load(progress) {
                    Ok(()) => Some(Arc::from(r)),
                    Err(e) => {
                        // Dictation must still work when the refiner can't start.
                        warn!("refinement unavailable: {e}");
                        None
                    }
                }
            };

            // Platform.
            let injector: Arc<dyn ochre_core::platform::Injector> =
                Arc::from(ochre_platform::injector()?);

            // Microphone, reused across reloads.
            let mic = {
                let mut cache = mics.lock();
                let key = (
                    cfg.audio.device.clone(),
                    cfg.audio.warm_mic,
                    cfg.audio.voice_processing,
                    cfg.audio.avoid_bluetooth_mic,
                    cfg.audio.warm_idle_release_s,
                );
                if cache.key.as_ref() != Some(&key) || cache.mic.is_none() {
                    cache.mic = None; // close the old stream before opening a new one
                    let mut opts = CaptureOptions::from_config(&cfg.audio);
                    // Room for a hands-free pre-roll (the wake word can start ~3 s before the hit).
                    opts.history_ms = opts.history_ms.max(4000);
                    cache.mic = Some(Arc::new(WarmMic::start(opts)?));
                    cache.key = Some(key);
                }
                cache.mic.clone().expect("mic just set")
            };

            let earcons = Arc::new(Earcons::new(cfg.audio.earcons));
            let cue = {
                let earcons = earcons.clone();
                Arc::new(move |c: Cue| {
                    earcons.play(match c {
                        Cue::Start => Earcon::Start,
                        Cue::Stop => Earcon::Stop,
                        Cue::Cancel => Earcon::Cancel,
                        Cue::Error => Earcon::Error,
                    })
                }) as Arc<dyn Fn(Cue) + Send + Sync>
            };

            Ok(Engines {
                stt,
                fallback_stt,
                refiner,
                injector,
                mic,
                segmenter: Arc::new(PhraseSegmenter::new(decoder.clone())),
                guard: ochre_refine::guard::apply,
                cue: Some(cue),
            })
        },
    )
}

fn local_fallback(cfg: &Config) -> Option<Arc<dyn SttEngine>> {
    let parakeet = ochre_stt::Parakeet::new("", &cfg.stt.device).ok()?;
    if !parakeet.is_downloaded() {
        return None;
    }
    let mut engine: Box<dyn SttEngine> = Box::new(parakeet);
    match engine.load(&|_| {}) {
        Ok(()) => Some(Arc::from(engine)),
        Err(e) => {
            warn!("local fallback unavailable: {e}");
            None
        }
    }
}

/// Download (or verify) a model on demand, from the settings UI.
pub fn download_stt(cfg: &Config, progress: &(dyn Fn(Progress) + Send + Sync)) -> Result<()> {
    if !is_local(&cfg.stt.engine) {
        return Err(Error::Config(
            "cloud engines have nothing to download".into(),
        ));
    }
    let mut stt = create_stt(cfg)?;
    stt.load(progress)
}

/// Input device names for the microphone picker, plus the system default's name.
pub fn input_devices() -> (Vec<String>, Option<String>) {
    let devices = ochre_audio::list_devices();
    let default = devices
        .iter()
        .find(|d| d.is_default)
        .map(|d| d.name.clone());
    (devices.into_iter().map(|d| d.name).collect(), default)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_with(engine: &str) -> Config {
        let mut c = Config::default();
        c.stt.engine = engine.into();
        c
    }

    fn err(engine: &str) -> String {
        match create_stt(&cfg_with(engine)) {
            Ok(_) => panic!("{engine} should not be creatable here"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn every_offered_engine_routes_to_a_factory() {
        for e in stt_engines() {
            assert!(create_stt(&cfg_with(&e.id)).is_ok(), "{}", e.id);
            assert!(stt_available(&e.id));
        }
    }

    #[test]
    fn a_local_engine_left_out_of_the_build_is_not_sent_to_the_cloud() {
        if is_local("whisper") {
            return; // compiled in: nothing to check
        }
        assert!(!stt_available("whisper"));
        let msg = err("whisper");
        assert!(msg.contains("not included in this build"), "{msg}");
        assert!(!msg.contains("cloud"), "{msg}");
    }

    #[test]
    fn unknown_engine_is_a_clear_config_error() {
        let msg = err("nope");
        assert!(msg.contains("Unknown speech engine"), "{msg}");
    }
}
