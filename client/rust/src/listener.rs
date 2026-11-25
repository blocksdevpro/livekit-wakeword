// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Microphone listener (the `listener` feature).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SizedSample};

use crate::{Detection, DetectionGate, Error, MODEL_SAMPLE_RATE, Resampler, Scores, WakeWordModel};

/// Buffers of microphone audio the worker may fall behind by before audio is dropped.
const QUEUE: usize = 64;

/// Called with every model's score for each chunk of microphone audio.
pub type ScoresCallback = Box<dyn FnMut(&Scores) + Send>;

/// How a [`WakeWordListener`] turns scores into detections.
pub struct ListenerOptions {
    /// Score at or above which a model fires (default 0.5, calibrated to about 0.5 false alarms
    /// per hour).
    pub threshold: f32,
    /// Minimum time between two detections of the same wake word (default 2 s).
    pub debounce: Duration,
    /// Called on the listener's thread with every model's score for each chunk of microphone
    /// audio (for meters and debugging).
    pub on_scores: Option<ScoresCallback>,
}

impl Default for ListenerOptions {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            debounce: Duration::from_secs(2),
            on_scores: None,
        }
    }
}

enum Event {
    Audio(Vec<f32>),
    Failed(String),
}

/// Streams the microphone through a [`WakeWordModel`] on a background thread.
///
/// After a detection the listener pauses (and the model is reset) until
/// [`wait_for_detection`](Self::wait_for_detection) is called again, so nothing fires while the
/// caller handles it:
///
/// ```no_run
/// use livekit_wakeword::{ListenerOptions, WakeWordListener, WakeWordModel};
///
/// # fn main() -> Result<(), livekit_wakeword::Error> {
/// let model = WakeWordModel::with_models(["hey_livekit.onnx"], 16_000)?;
/// let listener = WakeWordListener::start(model, ListenerOptions::default())?;
/// loop {
///     let detection = listener.wait_for_detection()?;
///     println!("Detected {} ({:.2})", detection.name, detection.confidence);
/// }
/// # }
/// ```
///
/// The microphone is opened at its own rate and resampled to 16 kHz here, whatever the model's
/// [`sample_rate`](WakeWordModel::sample_rate). Dropping the listener (or
/// [`stop`](Self::stop)) closes the microphone.
pub struct WakeWordListener {
    stream: Option<cpal::Stream>,
    worker: Option<JoinHandle<WakeWordModel>>,
    detections: Receiver<Result<Detection, Error>>,
    listening: Arc<AtomicBool>,
    sample_rate: u32,
}

impl WakeWordListener {
    /// Opens the default input device and starts listening.
    pub fn start(model: WakeWordModel, options: ListenerOptions) -> Result<Self, Error> {
        let device = cpal::default_host()
            .default_input_device()
            .ok_or_else(|| Error::Audio("no input device".into()))?;
        Self::start_with_device(model, options, &device)
    }

    /// Opens `device` (at its default input configuration) and starts listening.
    pub fn start_with_device(
        model: WakeWordModel,
        options: ListenerOptions,
        device: &cpal::Device,
    ) -> Result<Self, Error> {
        let audio_error = |e: cpal::Error| Error::Audio(e.to_string());
        let config = device.default_input_config().map_err(audio_error)?;
        let sample_rate = config.sample_rate();
        let resampler = match sample_rate {
            MODEL_SAMPLE_RATE => None,
            rate => Some(Resampler::new(rate)?),
        };
        let (audio_tx, audio_rx) = mpsc::sync_channel(QUEUE);
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => open::<f32>(device, config.config(), audio_tx),
            cpal::SampleFormat::F64 => open::<f64>(device, config.config(), audio_tx),
            cpal::SampleFormat::I16 => open::<i16>(device, config.config(), audio_tx),
            cpal::SampleFormat::I32 => open::<i32>(device, config.config(), audio_tx),
            cpal::SampleFormat::I8 => open::<i8>(device, config.config(), audio_tx),
            cpal::SampleFormat::U16 => open::<u16>(device, config.config(), audio_tx),
            cpal::SampleFormat::U8 => open::<u8>(device, config.config(), audio_tx),
            format => return Err(Error::Audio(format!("unsupported sample format {format}"))),
        }
        .map_err(audio_error)?;

        let listening = Arc::new(AtomicBool::new(true));
        let (detection_tx, detections) = mpsc::channel();
        let worker = {
            let listening = listening.clone();
            std::thread::Builder::new()
                .name("livekit-wakeword-listener".into())
                .spawn(move || run(model, options, resampler, audio_rx, detection_tx, listening))?
        };
        stream.play().map_err(audio_error)?;
        Ok(Self {
            stream: Some(stream),
            worker: Some(worker),
            detections,
            listening,
            sample_rate,
        })
    }

    /// The microphone's rate, in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Resumes listening and blocks until the next detection. Fails if the microphone or the
    /// model failed.
    pub fn wait_for_detection(&self) -> Result<Detection, Error> {
        self.listening.store(true, Ordering::SeqCst);
        self.detections
            .recv()
            .unwrap_or(Err(Error::ListenerStopped))
    }

    /// Like [`wait_for_detection`](Self::wait_for_detection), giving up after `timeout`
    /// (`Ok(None)`; listening stays on).
    pub fn wait_for_detection_timeout(
        &self,
        timeout: Duration,
    ) -> Result<Option<Detection>, Error> {
        self.listening.store(true, Ordering::SeqCst);
        match self.detections.recv_timeout(timeout) {
            Ok(result) => result.map(Some),
            Err(RecvTimeoutError::Timeout) => Ok(None),
            Err(RecvTimeoutError::Disconnected) => Err(Error::ListenerStopped),
        }
    }

    /// Closes the microphone and gives the model back.
    pub fn stop(mut self) -> WakeWordModel {
        self.shutdown().expect("the listener thread panicked")
    }

    fn shutdown(&mut self) -> Option<WakeWordModel> {
        drop(self.stream.take()); // drops the callbacks' senders, which ends the worker
        self.worker.take().and_then(|w| w.join().ok())
    }
}

impl Drop for WakeWordListener {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl std::fmt::Debug for WakeWordListener {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WakeWordListener")
            .field("sample_rate", &self.sample_rate)
            .finish()
    }
}

/// Opens an input stream that sends mono f32 buffers to the worker.
fn open<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    audio: SyncSender<Event>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = usize::from(config.channels).max(1);
    let errors = audio.clone();
    device.build_input_stream::<T, _, _>(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            // never block the audio thread: if the worker is that far behind, drop this buffer
            let _ = audio.try_send(Event::Audio(downmix(data, channels)));
        },
        move |e| {
            let _ = errors.try_send(Event::Failed(e.to_string()));
        },
        None,
    )
}

/// Interleaved samples of `channels` channels -> mono f32, the average of each frame.
fn downmix<T: Copy>(data: &[T], channels: usize) -> Vec<f32>
where
    f32: FromSample<T>,
{
    data.chunks(channels.max(1))
        .map(|frame| frame.iter().map(|&s| f32::from_sample_(s)).sum::<f32>() / frame.len() as f32)
        .collect()
}

/// The listener thread: resample, score, gate. Returns the model when the stream closes.
fn run(
    mut model: WakeWordModel,
    mut options: ListenerOptions,
    mut resampler: Option<Resampler>,
    audio: Receiver<Event>,
    detections: mpsc::Sender<Result<Detection, Error>>,
    listening: Arc<AtomicBool>,
) -> WakeWordModel {
    let mut gate = DetectionGate::new(options.threshold, options.debounce);
    let mut x = Vec::new();
    for event in audio {
        let mono = match event {
            Event::Audio(mono) => mono,
            Event::Failed(e) => {
                let _ = detections.send(Err(Error::Audio(e)));
                break;
            }
        };
        if !listening.load(Ordering::SeqCst) {
            continue; // paused after a detection
        }
        x.clear();
        match &mut resampler {
            Some(r) => r.process_into(&mono, &mut x),
            None => x.extend_from_slice(&mono),
        }
        let scores = match model.predict_16k(&x) {
            Ok(scores) => scores,
            Err(e) => {
                let _ = detections.send(Err(e));
                break;
            }
        };
        if let Some(on_scores) = &mut options.on_scores {
            on_scores(&scores);
        }
        if let Some(detection) = gate.check(&scores, Instant::now()) {
            listening.store(false, Ordering::SeqCst);
            model.reset();
            if let Some(r) = &mut resampler {
                r.reset();
            }
            if detections.send(Ok(detection)).is_err() {
                break;
            }
        }
    }
    model
}

#[cfg(test)]
mod tests {
    use super::downmix;

    #[test]
    fn downmix_averages_channels() {
        assert_eq!(downmix(&[0.5f32, -0.5, 1.0, 0.0], 2), [0.0, 0.5]);
        assert_eq!(downmix(&[0.25f32, 0.75], 1), [0.25, 0.75]);
        // a trailing partial frame is averaged over the samples it has
        assert_eq!(downmix(&[1.0f32, 1.0, 0.5], 2), [1.0, 0.5]);
    }

    #[test]
    fn downmix_converts_integer_samples() {
        assert_eq!(downmix(&[i16::MIN, 0, 16384], 1), [-1.0, 0.0, 0.5]);
        assert_eq!(downmix(&[i16::MIN, i16::MIN, 16384, 16384], 2), [-1.0, 0.5]);
    }
}
