// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Streaming wake word detection for [LiveKit WakeWord](https://github.com/livekit/livekit-wakeword)
//! models (`.onnx` from `livekit-wakeword train`), run with [tract](https://github.com/sonos/tract):
//! pure Rust, no native libraries.
//!
//! ```no_run
//! use livekit_wakeword::WakeWordModel;
//!
//! # fn main() -> Result<(), livekit_wakeword::Error> {
//! // the rate of the audio you feed; models run at 16 kHz and other rates are resampled
//! let mut model = WakeWordModel::with_models(["hey_livekit.onnx"], 48_000)?;
//! # let frame = vec![0i16; 3840];
//! // stream audio as it arrives: mono, i16 or f32, any chunk size
//! let scores = model.predict(&frame)?;
//! if scores["hey_livekit"] >= 0.5 {
//!     println!("Wake word detected!");
//!     model.reset(); // start fresh after a detection
//! }
//! # Ok(())
//! # }
//! ```
//!
//! Each model is one stateful network: every 80 ms of audio (1280 samples at 16 kHz) is one step,
//! and its state carries over between calls. [`WakeWordModel::predict`] returns, for each model,
//! the highest score among the steps that call completed (0.0 if none).
//!
//! With the `listener` feature, [`WakeWordListener`] streams the microphone through a model.

mod error;
mod gate;
mod model;
mod resample;
mod stream;

#[cfg(feature = "listener")]
mod listener;

pub use error::Error;
pub use gate::{Detection, DetectionGate};
#[cfg(feature = "listener")]
pub use listener::{ListenerOptions, ScoresCallback, WakeWordListener};
pub use model::{Sample, Scores, WakeWordModel};
pub use resample::Resampler;

/// The rate models run at, in Hz.
pub const MODEL_SAMPLE_RATE: u32 = 16_000;
/// Samples per model step: 80 ms at 16 kHz.
pub const FRAME_SAMPLES: usize = 1280;
/// Steps of silence every stream starts from (2 s), as in training.
pub const WARMUP_FRAMES: usize = 25;
