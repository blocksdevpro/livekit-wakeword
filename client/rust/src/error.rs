// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::path::PathBuf;

/// Everything that can go wrong loading or running wake word models.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    #[error("wake word model not found: {}", .0.display())]
    ModelNotFound(PathBuf),

    #[error(
        "{} is a micro model (int8 TFLite for ESP32 / TensorFlow Lite Micro); load the standard \
         .onnx model here",
        .0.display()
    )]
    MicroModel(PathBuf),

    /// The file is not an ONNX model tract can run.
    #[error("could not load {model} (is it an .onnx model?): {reason}")]
    InvalidModel { model: String, reason: String },

    /// An ONNX model without the streaming interface (`chunk` + state in, `score` + state out).
    #[error(
        "{model} is not a streaming wake word model ({reason}); v1 classifier models are no \
         longer supported, retrain with `livekit-wakeword train`"
    )]
    NotStreaming { model: String, reason: String },

    #[error("sample rate must be positive, got {0}")]
    InvalidSampleRate(u32),

    #[error("model '{model}' failed: {reason}")]
    Inference { model: String, reason: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// The microphone could not be opened, or its stream failed.
    #[error("audio input: {0}")]
    Audio(String),

    /// The listener's audio loop has ended (after an error, or once stopped).
    #[error("the listener has stopped")]
    ListenerStopped,
}
