//! macOS: the microphone through Apple's voice processing (`AVAudioEngine` input with
//! `setVoiceProcessingEnabled`), the path FaceTime and Zoom use. It cancels echo of whatever the
//! Mac plays, suppresses noise, levels the voice (AGC), and makes Ochre eligible for the system
//! **Mic Mode** menu: Voice Isolation there removes other people's voices too. [`show_mic_modes`]
//! opens that menu.
//!
//! Same contract as the cpal path in `capture.rs`: the tap block (an AVAudioEngine thread, not the
//! IO thread) copies samples into the SPSC ring, stamps the time and wakes the pump. The processed
//! input arrives as N identical channels (9 on a MacBook Pro); only channel 0 is copied.
//!
//! * Voice processing ducks other apps' audio while it runs. With the warm mic that would be all
//!   the time, so ducking is set to its minimum (macOS 14+; older systems duck at the default level).
//! * It follows the system default input, or a device given by id (the built-in mic standing in
//!   for a Bluetooth headset, see `devices_mac.rs`), set on the input element of the voice unit
//!   and read back; if it doesn't stick, `open` fails and capture uses the plain cpal path.
//! * A device change stops the engine; the pump's stall watchdog reopens it.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::Ordering;
use std::thread::Thread;

use block2::RcBlock;
use objc2::rc::Retained;
use objc2::runtime::Bool;
use objc2::{class, msg_send, sel};
use objc2_avf_audio::{
    AVAudioEngine, AVAudioPCMBuffer, AVAudioTime,
    AVAudioVoiceProcessingOtherAudioDuckingConfiguration,
    AVAudioVoiceProcessingOtherAudioDuckingLevel,
};
use objc2_foundation::NSObjectProtocol;
use ochre_core::{Error, Result};

use crate::capture::CallbackShared;

/// A running voice-processed input. Stops the engine on drop.
pub(crate) struct VoiceStream {
    engine: Retained<AVAudioEngine>,
}

// SAFETY: the engine is created on one thread and then only stopped (on Drop) from the pump;
// AVAudioEngine allows start/stop from any thread.
unsafe impl Send for VoiceStream {}

impl Drop for VoiceStream {
    fn drop(&mut self) {
        // SAFETY: the engine was started by `open` on this thread.
        unsafe {
            self.engine.inputNode().removeTapOnBus(0);
            self.engine.stop();
        }
    }
}

/// Start voice-processed capture of the default input. Returns the stream and its sample rate;
/// samples (mono) go to `producer`.
pub(crate) fn open(
    device: Option<u32>,
    producer: rtrb::Producer<f32>,
    cb: Arc<CallbackShared>,
    pump: Thread,
) -> Result<(VoiceStream, u32)> {
    let err = |what: &str, e: &dyn std::fmt::Display| {
        Error::Audio(format!("voice processing: {what}: {e}"))
    };
    // SAFETY: plain AVAudioEngine setup on the pump thread, which owns the engine from here on.
    unsafe {
        let engine = AVAudioEngine::new();
        let input = engine.inputNode();
        input
            .setVoiceProcessingEnabled_error(true)
            .map_err(|e| err("enable", &e.localizedDescription()))?;
        if let Some(id) = device {
            pin_input_device(&input, id)?;
        }
        let setter = sel!(setVoiceProcessingOtherAudioDuckingConfiguration:);
        if input.respondsToSelector(setter) {
            input.setVoiceProcessingOtherAudioDuckingConfiguration(
                AVAudioVoiceProcessingOtherAudioDuckingConfiguration {
                    enableAdvancedDucking: Bool::NO,
                    duckingLevel: AVAudioVoiceProcessingOtherAudioDuckingLevel::Min,
                },
            );
        }
        let format = input.outputFormatForBus(0);
        let rate = format.sampleRate() as u32;
        if rate == 0 || format.channelCount() == 0 {
            return Err(Error::Audio("voice processing: no input format".into()));
        }
        let producer = Mutex::new(producer);
        let block = RcBlock::new(
            move |buf: NonNull<AVAudioPCMBuffer>, _: NonNull<AVAudioTime>| {
                let buf = buf.as_ref();
                let frames = buf.frameLength() as usize;
                let data = buf.floatChannelData();
                if data.is_null() || frames == 0 {
                    return;
                }
                // Channel 0 holds the processed voice (the others repeat it).
                let ch0 = std::slice::from_raw_parts((*data).as_ptr(), frames);
                if let Ok(mut p) = producer.try_lock() {
                    let n = frames.min(p.slots());
                    if n < frames {
                        cb.overruns.fetch_add(1, Ordering::Relaxed);
                    }
                    if let Ok(chunk) = p.write_chunk_uninit(n) {
                        chunk.fill_from_iter(ch0[..n].iter().copied());
                    }
                }
                cb.last_cb_us
                    .store(cb.epoch.elapsed().as_micros() as u64, Ordering::Relaxed);
                pump.unpark();
            },
        );
        // ~10 ms blocks at 48 kHz (the engine may round it).
        input.installTapOnBus_bufferSize_format_block(
            0,
            (rate / 100).max(256),
            Some(&format),
            &*block as *const _ as *mut _,
        );
        if let Err(e) = engine.startAndReturnError() {
            input.removeTapOnBus(0);
            return Err(err("start", &e.localizedDescription()));
        }
        Ok((VoiceStream { engine }, rate))
    }
}

/// Open the system Mic Mode menu (Standard / Voice Isolation / Wide Spectrum) for Ochre. macOS
/// only lists it while Ochre's voice-processed microphone is running.
pub fn show_mic_modes() {
    // SAFETY: class method with an enum argument (AVCaptureSystemUserInterfaceMicrophoneModes = 2).
    unsafe {
        let cls = class!(AVCaptureDevice);
        let _: () = msg_send![cls, showSystemUserInterface: 2isize];
    }
}

/// The mic mode macOS applies to Ochre right now: "standard", "voice_isolation" or
/// "wide_spectrum".
pub fn mic_mode() -> &'static str {
    // SAFETY: class property returning AVCaptureMicrophoneMode (NSInteger).
    let mode: isize = unsafe { msg_send![class!(AVCaptureDevice), activeMicrophoneMode] };
    match mode {
        1 => "wide_spectrum",
        2 => "voice_isolation",
        _ => "standard",
    }
}

#[link(name = "AVFoundation", kind = "framework")]
unsafe extern "C" {}

#[link(name = "AudioToolbox", kind = "framework")]
unsafe extern "C" {
    fn AudioUnitSetProperty(
        unit: *mut std::ffi::c_void,
        id: u32,
        scope: u32,
        element: u32,
        data: *const std::ffi::c_void,
        size: u32,
    ) -> i32;
    fn AudioUnitGetProperty(
        unit: *mut std::ffi::c_void,
        id: u32,
        scope: u32,
        element: u32,
        data: *mut std::ffi::c_void,
        size: *mut u32,
    ) -> i32;
}

/// Make the voice unit record from `id` (element 1 is the input side; element 0 is the output
/// device, which must stay as it is). Fails unless the unit reports the device back.
fn pin_input_device(input: &objc2_avf_audio::AVAudioInputNode, id: u32) -> Result<()> {
    const CURRENT_DEVICE: u32 = 2000; // kAudioOutputUnitProperty_CurrentDevice
    const INPUT_ELEMENT: u32 = 1;
    // SAFETY: the node's AudioUnit lives as long as the node; properties are 4-byte device ids.
    unsafe {
        // (msg_send: the typed getter needs the AudioToolbox bindings for one pointer)
        let unit: *mut std::ffi::c_void = msg_send![input, audioUnit];
        if unit.is_null() {
            return Err(Error::Audio("voice processing: no audio unit".into()));
        }
        let st = AudioUnitSetProperty(
            unit,
            CURRENT_DEVICE,
            0,
            INPUT_ELEMENT,
            &id as *const u32 as *const _,
            4,
        );
        let (mut now, mut size) = (0u32, 4u32);
        AudioUnitGetProperty(
            unit,
            CURRENT_DEVICE,
            0,
            INPUT_ELEMENT,
            &mut now as *mut u32 as *mut _,
            &mut size,
        );
        if st != 0 || now != id {
            return Err(Error::Audio(format!(
                "voice processing: couldn't select input device {id} (status {st}, now {now})"
            )));
        }
    }
    Ok(())
}
