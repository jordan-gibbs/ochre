//! A [`SegmentSession`] over a streaming engine (`SttEngine::streaming()`): audio goes to the
//! provider while the user talks, so release waits only for the provider's final (Soniox
//! real-time 110-166 ms, OpenAI gpt-live-transcribe 0.46-0.68 s measured), instead of decoding a
//! tail phrase.
//!
//! The stream is opened on its own thread at key-down (a TLS + WebSocket handshake, ~0.4-0.9 s,
//! must never block the hotkey thread); audio captured meanwhile queues in the channel and is
//! forwarded once the socket is up. Every sample is also kept, so if opening, sending or the final
//! fails, `finish` falls back to one batch transcription of the whole recording through the
//! normal path (`TranscribeFn`, which includes the local-engine fallback).

use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{Sender, unbounded};
use tracing::warn;

use ochre_core::Result;
use ochre_core::stt::{PartialSink, SttEngine, SttOptions, SttResult, SttStream};

use crate::seams::{PartialFn, SegmentSession, TranscribeFn};

enum Msg {
    Audio(Vec<f32>),
    Finish,
    Cancel,
}

pub struct StreamSession {
    tx: Sender<Msg>,
    worker: Option<JoinHandle<Result<Vec<SttResult>>>>,
}

impl StreamSession {
    pub fn start(
        engine: Arc<dyn SttEngine>,
        opts: SttOptions,
        on_partial: Option<PartialFn>,
        transcribe: TranscribeFn,
    ) -> Self {
        let (tx, rx) = unbounded::<Msg>();
        let worker = std::thread::Builder::new()
            .name("ochre-stream".into())
            .spawn(move || {
                let sink: Option<PartialSink> = on_partial
                    .map(|p| Box::new(move |t: &str, n: usize| p(t, n)) as PartialSink);
                let mut stream: Option<Box<dyn SttStream>> = match engine.stream(&opts, sink) {
                    Some(Ok(s)) => Some(s),
                    Some(Err(e)) => {
                        warn!(code = e.code(), "stream open failed; will transcribe in one batch");
                        None
                    }
                    None => None,
                };
                let mut all: Vec<f32> = Vec::new();
                for msg in rx {
                    match msg {
                        Msg::Audio(pcm) => {
                            if let Some(s) = stream.as_mut()
                                && let Err(e) = s.send(&pcm)
                            {
                                warn!(code = e.code(), "stream send failed; will transcribe in one batch");
                                if let Some(s) = stream.take() {
                                    s.cancel();
                                }
                            }
                            all.extend_from_slice(&pcm);
                        }
                        Msg::Finish => {
                            if let Some(s) = stream.take() {
                                match s.finish() {
                                    Ok(r) => return Ok(vec![r]),
                                    Err(e) => warn!(
                                        code = e.code(),
                                        "stream final failed; transcribing the recording in one batch"
                                    ),
                                }
                            }
                            if all.is_empty() {
                                return Ok(vec![]);
                            }
                            return transcribe(&all).map(|r| vec![r]);
                        }
                        Msg::Cancel => break,
                    }
                }
                if let Some(s) = stream.take() {
                    s.cancel();
                }
                Ok(vec![])
            })
            .expect("spawn stream worker");
        Self {
            tx,
            worker: Some(worker),
        }
    }
}

impl SegmentSession for StreamSession {
    fn feed(&mut self, pcm: &[f32]) {
        let _ = self.tx.send(Msg::Audio(pcm.to_vec()));
    }

    fn finish(mut self: Box<Self>) -> Result<Vec<SttResult>> {
        let _ = self.tx.send(Msg::Finish);
        match self.worker.take().map(JoinHandle::join) {
            Some(Ok(r)) => r,
            _ => Err(ochre_core::Error::Other("stream worker panicked".into())),
        }
    }

    fn cancel(mut self: Box<Self>) {
        let _ = self.tx.send(Msg::Cancel);
        drop(self.worker.take()); // detached: closing the socket must not block the caller
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ochre_core::events::EngineInfo;
    use ochre_core::stt::ProgressFn;
    use parking_lot::Mutex;

    #[derive(Default)]
    struct Log {
        sent: usize,
        finished: bool,
        canceled: bool,
        batch: Vec<usize>,
    }

    struct FakeStream {
        log: Arc<Mutex<Log>>,
        fail_finish: bool,
        partial: Option<PartialSink>,
    }

    impl SttStream for FakeStream {
        fn send(&mut self, pcm: &[f32]) -> Result<()> {
            self.log.lock().sent += pcm.len();
            if let Some(p) = self.partial.as_mut() {
                p("hel", 0);
            }
            Ok(())
        }
        fn finish(self: Box<Self>) -> Result<SttResult> {
            self.log.lock().finished = true;
            if self.fail_finish {
                return Err(ochre_core::Error::Timeout(std::time::Duration::from_secs(
                    1,
                )));
            }
            Ok(SttResult {
                text: "streamed".into(),
                ..Default::default()
            })
        }
        fn cancel(self: Box<Self>) {
            self.log.lock().canceled = true;
        }
    }

    struct FakeEngine {
        log: Arc<Mutex<Log>>,
        open: Option<bool>, // None = not streaming, Some(false) = open fails
        fail_finish: bool,
    }

    impl SttEngine for FakeEngine {
        fn info(&self) -> EngineInfo {
            EngineInfo {
                id: "fake".into(),
                label: "Fake".into(),
                kind: "cloud".into(),
                models: vec![],
                default_model: String::new(),
                needs_key: false,
                note: String::new(),
                languages: String::new(),
            }
        }
        fn load(&mut self, _: ProgressFn) -> Result<()> {
            Ok(())
        }
        fn transcribe(&self, _: &[f32], _: &SttOptions) -> Result<SttResult> {
            unreachable!("the session uses TranscribeFn for batch")
        }
        fn streaming(&self) -> bool {
            self.open.is_some()
        }
        fn stream(
            &self,
            _: &SttOptions,
            partial: Option<PartialSink>,
        ) -> Option<Result<Box<dyn SttStream>>> {
            std::thread::sleep(std::time::Duration::from_millis(30)); // handshake
            self.open.map(|ok| {
                if ok {
                    Ok(Box::new(FakeStream {
                        log: self.log.clone(),
                        fail_finish: self.fail_finish,
                        partial,
                    }) as Box<dyn SttStream>)
                } else {
                    Err(ochre_core::Error::Network {
                        provider: "fake".into(),
                        message: "down".into(),
                    })
                }
            })
        }
    }

    fn run(
        open: Option<bool>,
        fail_finish: bool,
    ) -> (Result<Vec<SttResult>>, Arc<Mutex<Log>>, usize) {
        let log = Arc::new(Mutex::new(Log::default()));
        let engine = Arc::new(FakeEngine {
            log: log.clone(),
            open,
            fail_finish,
        });
        let l2 = log.clone();
        let transcribe: TranscribeFn = Arc::new(move |pcm: &[f32]| {
            l2.lock().batch.push(pcm.len());
            Ok(SttResult {
                text: "batch".into(),
                ..Default::default()
            })
        });
        let partials = Arc::new(Mutex::new(0usize));
        let p2 = partials.clone();
        let partial: PartialFn = Arc::new(move |_t: &str, _n: usize| *p2.lock() += 1);
        let mut s: Box<dyn SegmentSession> = Box::new(StreamSession::start(
            engine,
            SttOptions::default(),
            Some(partial),
            transcribe,
        ));
        for _ in 0..10 {
            s.feed(&[0.0; 160]); // fed while the stream is still opening
        }
        s.release();
        let r = s.finish();
        let n = *partials.lock();
        (r, log, n)
    }

    #[test]
    fn streams_audio_queued_during_open_and_returns_the_final() {
        let (r, log, partials) = run(Some(true), false);
        assert_eq!(r.unwrap()[0].text, "streamed");
        let l = log.lock();
        assert_eq!(l.sent, 1600);
        assert!(l.finished && l.batch.is_empty());
        assert_eq!(partials, 10);
    }

    #[test]
    fn open_failure_falls_back_to_one_batch() {
        let (r, log, _) = run(Some(false), false);
        assert_eq!(r.unwrap()[0].text, "batch");
        assert_eq!(log.lock().batch, [1600]);
    }

    #[test]
    fn final_failure_falls_back_to_one_batch() {
        let (r, log, _) = run(Some(true), true);
        assert_eq!(r.unwrap()[0].text, "batch");
        let l = log.lock();
        assert!(l.finished);
        assert_eq!(l.batch, [1600]);
    }

    #[test]
    fn cancel_closes_the_stream() {
        let log = Arc::new(Mutex::new(Log::default()));
        let engine = Arc::new(FakeEngine {
            log: log.clone(),
            open: Some(true),
            fail_finish: false,
        });
        let s: Box<dyn SegmentSession> = Box::new(StreamSession::start(
            engine,
            SttOptions::default(),
            None,
            Arc::new(|_: &[f32]| unreachable!()),
        ));
        s.cancel();
        for _ in 0..100 {
            if log.lock().canceled {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        panic!("stream was not canceled");
    }
}
