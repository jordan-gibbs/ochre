//! Settings checks use the real configured engines without opening the mic,
//! typing text, touching history, or falling back to a different provider.
use std::io::Cursor;
use std::time::{Duration, Instant};

use ochre_core::config::Config;
use ochre_core::events::Event;
use ochre_core::refine::{RefineContext, Refiner};
use ochre_core::stt::{ProgressFn, SttEngine, SttOptions};
use ochre_core::{Error, Result};

const TEST_TEXT: &str = "um please send me the file tomorrow morning";
// Reuse the existing speech fixture; no new audio asset or user recording.
const TEST_WAV: &[u8] = include_bytes!("../../../tests/fixtures/wake/transcribe_sample.wav");

fn speech_clip() -> Result<Vec<f32>> {
    let bad_clip = |e| Error::Other(format!("Built-in speech clip: {e}"));
    let mut reader = hound::WavReader::new(Cursor::new(TEST_WAV)).map_err(bad_clip)?;
    let spec = reader.spec();
    if spec.channels != 1
        || spec.sample_rate != ochre_core::SAMPLE_RATE
        || spec.bits_per_sample != 16
        || spec.sample_format != hound::SampleFormat::Int
    {
        return Err(Error::Other(
            "Built-in speech clip must be mono PCM16 at 16 kHz.".into(),
        ));
    }
    reader
        .samples::<i16>()
        .map(|s| s.map(|s| s as f32 / 32768.0).map_err(bad_clip))
        .collect()
}

fn check_stt(engine: &mut dyn SttEngine, options: &SttOptions, progress: ProgressFn) -> Result<()> {
    engine.load(progress)?;
    let result = engine.transcribe(&speech_clip()?, options)?;
    if result.text.trim().is_empty() {
        return Err(Error::Other(
            "Connected, but no speech was recognized in the built-in test clip.".into(),
        ));
    }
    Ok(())
}

fn check_refiner(
    refiner: &mut dyn Refiner,
    mode: &str,
    timeout: Duration,
    progress: ProgressFn,
) -> Result<()> {
    refiner.load(progress)?;
    let context = RefineContext {
        mode: mode.into(),
        ..Default::default()
    };
    let output = refiner.refine(TEST_TEXT, &context, timeout)?;
    if let Some(reason) = ochre_refine::guard::check(TEST_TEXT, &output, mode) {
        return Err(Error::Other(format!(
            "The cleanup response was rejected: {}.",
            reason.as_str()
        )));
    }
    Ok(())
}

fn check(cfg: &Config, stage: &str, provider: &str, progress: ProgressFn) -> Result<String> {
    let mut cfg = cfg.clone();
    match stage {
        "stt" => {
            if cfg.stt.engine != provider {
                cfg.stt.engine = provider.into();
                cfg.stt.model.clear();
            }
            let mut engine = crate::wiring::create_stt(&cfg)?;
            let options = SttOptions {
                language: cfg.stt.language.clone(),
                ..Default::default()
            };
            check_stt(engine.as_mut(), &options, progress)?;
            Ok("Works — transcribed the built-in speech clip.".into())
        }
        "refine" => {
            if cfg.refine.provider != provider {
                cfg.refine.provider = provider.into();
                cfg.refine.model.clear();
            }
            if cfg.refine.mode == "off" {
                cfg.refine.mode = "clean".into();
            }
            let mut refiner = ochre_refine::create(&cfg.refine)?;
            let timeout = ochre_refine::timeout_for(&cfg.refine, refiner.as_ref())
                .max(Duration::from_secs(10));
            check_refiner(refiner.as_mut(), &cfg.refine.mode, timeout, progress)?;
            Ok("Works — cleaned up the built-in test sentence.".into())
        }
        _ => Err(Error::Config(format!("Unknown test stage {stage:?}."))),
    }
}

/// Blocking provider check. Call from a worker, never from the UI/event thread.
pub fn test_provider(cfg: &Config, stage: String, provider: String, progress: ProgressFn) -> Event {
    let started = Instant::now();
    let result = check(cfg, &stage, &provider, progress);
    Event::TestResult {
        stage,
        provider,
        ok: result.is_ok(),
        message: match result {
            Ok(message) => message,
            Err(error) => error.to_string(),
        },
        ms: started.elapsed().as_millis() as u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ochre_core::events::EngineInfo;
    use ochre_core::stt::SttResult;

    struct FakeStt {
        loaded: bool,
        fail_load: bool,
        text: Option<&'static str>,
    }
    impl SttEngine for FakeStt {
        fn info(&self) -> EngineInfo {
            panic!("not needed by check")
        }
        fn load(&mut self, _: ProgressFn) -> Result<()> {
            if self.fail_load {
                return Err(Error::Auth {
                    provider: "test".into(),
                });
            }
            self.loaded = true;
            Ok(())
        }
        fn transcribe(&self, pcm: &[f32], opts: &SttOptions) -> Result<SttResult> {
            assert!(self.loaded);
            assert!(pcm.len() > 16000);
            assert!(pcm.iter().any(|x| x.abs() > 0.05));
            assert_eq!(opts.language.as_deref(), Some("en"));
            self.text
                .map(|text| SttResult {
                    text: text.into(),
                    ..Default::default()
                })
                .ok_or(Error::Quota {
                    provider: "test".into(),
                })
        }
    }

    #[test]
    fn transcription_checks_real_decode_and_surfaces_failures() {
        let opts = SttOptions {
            language: Some("en".into()),
            ..Default::default()
        };
        for (fail_load, text, expected) in [
            (false, Some("Hello"), "ok"),
            (true, Some("Hello"), "auth"),
            (false, None, "quota"),
            (false, Some(""), "other"),
        ] {
            let mut engine = FakeStt {
                loaded: false,
                fail_load,
                text,
            };
            let result = check_stt(&mut engine, &opts, &|_| {});
            assert_eq!(result.err().map(|e| e.code()).unwrap_or("ok"), expected);
        }
    }

    struct FakeRefiner {
        output: &'static str,
    }
    impl Refiner for FakeRefiner {
        fn info(&self) -> EngineInfo {
            panic!("not needed by check")
        }
        fn load(&mut self, _: ProgressFn) -> Result<()> {
            Ok(())
        }
        fn refine(&self, text: &str, ctx: &RefineContext, timeout: Duration) -> Result<String> {
            assert_eq!(text, TEST_TEXT);
            assert_eq!(ctx.mode, "clean");
            assert_eq!(timeout, Duration::from_secs(10));
            Ok(self.output.into())
        }
    }

    #[test]
    fn cleanup_validates_output_instead_of_silently_falling_back() {
        for (output, ok) in [
            ("Please send me the file tomorrow morning.", true),
            ("", false),
            ("As an AI, I cannot send files.", false),
        ] {
            let mut refiner = FakeRefiner { output };
            assert_eq!(
                check_refiner(&mut refiner, "clean", Duration::from_secs(10), &|_| {}).is_ok(),
                ok
            );
        }
    }

    #[test]
    fn invalid_stage_reports_failure_without_reconfiguring_or_loading() {
        let cfg = Config::default();
        let original = cfg.clone();
        let event = test_provider(&cfg, "invalid".into(), "openai".into(), &|_| {
            panic!("must not load")
        });
        assert!(matches!(event, Event::TestResult { ok: false, .. }));
        assert_eq!(cfg, original);
    }

    /// Uses the app's configured OpenAI model and OS-keyring/environment key.
    /// Sends only the repository fixture and TEST_TEXT; never captures the mic.
    #[test]
    #[ignore = "network + configured OpenAI key; incurs a small API charge"]
    fn live_openai_provider_checks() {
        let cfg = Config::load().expect("app configuration");
        assert_eq!(cfg.stt.engine, "openai", "select OpenAI in Settings first");
        assert_eq!(cfg.refine.provider, "openai", "select OpenAI cleanup first");
        assert!(
            ochre_core::secrets::has("openai"),
            "save an OpenAI key first"
        );
        for stage in ["stt", "refine"] {
            let event = test_provider(&cfg, stage.into(), "openai".into(), &|_| {});
            let Event::TestResult {
                ok, message, ms, ..
            } = event
            else {
                unreachable!()
            };
            assert!(ok, "{stage}: {message}");
            eprintln!("{stage}: {message} ({ms} ms)");
        }
    }
}
