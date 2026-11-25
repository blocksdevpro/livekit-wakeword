// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use crate::stream::Stream;
use crate::{Error, MODEL_SAMPLE_RATE, Resampler};

/// Each loaded model's score, keyed by model name.
pub type Scores = HashMap<String, f32>;

/// Mono PCM samples [`WakeWordModel::predict`] accepts: `i16`, or `f32` in [-1, 1].
pub trait Sample: Copy + sealed::Sealed {
    /// The samples as f32 in [-1, 1].
    fn to_float(audio: &[Self]) -> Cow<'_, [f32]>;
}

impl Sample for i16 {
    fn to_float(audio: &[Self]) -> Cow<'_, [f32]> {
        Cow::Owned(audio.iter().map(|&s| f32::from(s) / 32768.0).collect())
    }
}

impl Sample for f32 {
    fn to_float(audio: &[Self]) -> Cow<'_, [f32]> {
        Cow::Borrowed(audio)
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for i16 {}
    impl Sealed for f32 {}
}

/// Wake word models (`.onnx` from `livekit-wakeword train`), streamed over mono audio.
///
/// ```no_run
/// # fn main() -> Result<(), livekit_wakeword::Error> {
/// use livekit_wakeword::WakeWordModel;
///
/// let mut model = WakeWordModel::with_models(["hey_livekit.onnx"], 16_000)?;
/// # let frame = vec![0i16; 1280];
/// let scores = model.predict(&frame)?; // {"hey_livekit": 0.97}; frames of any length
/// # Ok(())
/// # }
/// ```
///
/// `sample_rate` is the rate of the audio passed to [`predict`](Self::predict); models run at
/// 16 kHz, and audio at any other rate is resampled internally.
pub struct WakeWordModel {
    sample_rate: u32,
    resampler: Option<Resampler>,
    models: Vec<(String, Stream)>, // in load order
}

impl WakeWordModel {
    /// An empty model set for audio at `sample_rate` Hz; add models with
    /// [`load_model`](Self::load_model).
    pub fn new(sample_rate: u32) -> Result<Self, Error> {
        let resampler = match sample_rate {
            0 => return Err(Error::InvalidSampleRate(0)),
            MODEL_SAMPLE_RATE => None,
            rate => Some(Resampler::new(rate)?),
        };
        Ok(Self {
            sample_rate,
            resampler,
            models: Vec::new(),
        })
    }

    /// Loads `models` (each named after its file), for audio at `sample_rate` Hz.
    pub fn with_models<P: AsRef<Path>>(
        models: impl IntoIterator<Item = P>,
        sample_rate: u32,
    ) -> Result<Self, Error> {
        let mut this = Self::new(sample_rate)?;
        for path in models {
            this.load_model(path, None)?;
        }
        Ok(this)
    }

    /// The rate of the audio [`predict`](Self::predict) takes, in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Adds a model (or replaces the one with that name), named after the file (`hey_livekit`
    /// for `hey_livekit.onnx`) unless `name` is given.
    pub fn load_model(&mut self, path: impl AsRef<Path>, name: Option<&str>) -> Result<(), Error> {
        let path = path.as_ref();
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("tflite"))
        {
            return Err(Error::MicroModel(path.to_path_buf()));
        }
        let bytes = match std::fs::read(path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::ModelNotFound(path.to_path_buf()));
            }
            result => result?,
        };
        let name = match name {
            Some(name) => name.to_string(),
            None => path
                .file_stem()
                .map_or("model".into(), |s| s.to_string_lossy().into_owned()),
        };
        let stream = Stream::load(&bytes, &path.display().to_string())?;
        self.insert(name, stream);
        Ok(())
    }

    /// Adds a model from the bytes of an `.onnx` file (or replaces the one named `name`).
    pub fn load_model_from_bytes(&mut self, bytes: &[u8], name: &str) -> Result<(), Error> {
        let stream = Stream::load(bytes, &format!("'{name}'"))?;
        self.insert(name.to_string(), stream);
        Ok(())
    }

    fn insert(&mut self, name: String, stream: Stream) {
        match self.models.iter_mut().find(|(n, _)| *n == name) {
            Some((_, s)) => *s = stream,
            None => self.models.push((name, stream)),
        }
    }

    /// Removes a model; returns whether it was loaded.
    pub fn unload_model(&mut self, name: &str) -> bool {
        let before = self.models.len();
        self.models.retain(|(n, _)| n != name);
        self.models.len() != before
    }

    /// The loaded models' names, in load order.
    pub fn model_names(&self) -> Vec<&str> {
        self.models.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Forgets all audio heard so far (e.g. after a detection).
    pub fn reset(&mut self) {
        if let Some(r) = &mut self.resampler {
            r.reset();
        }
        for (_, m) in &mut self.models {
            m.reset();
        }
    }

    /// Feeds the next audio: mono at [`sample_rate`](Self::sample_rate), `i16` or `f32` in
    /// [-1, 1], any length. Returns each model's highest score among the 80 ms steps this call
    /// completed (0.0 if none).
    pub fn predict<S: Sample>(&mut self, audio: &[S]) -> Result<Scores, Error> {
        if self.models.is_empty() {
            return Ok(Scores::new());
        }
        let audio = S::to_float(audio);
        match &mut self.resampler {
            Some(r) => {
                let x = r.process(&audio);
                self.predict_16k(&x)
            }
            None => self.predict_16k(&audio),
        }
    }

    /// Each model's highest score over a whole clip, starting fresh.
    pub fn predict_clip<S: Sample>(&mut self, audio: &[S]) -> Result<Scores, Error> {
        self.reset();
        self.predict(audio)
    }

    /// Feeds 16 kHz audio, bypassing `sample_rate` (for listeners that resample themselves).
    pub(crate) fn predict_16k(&mut self, audio: &[f32]) -> Result<Scores, Error> {
        self.models
            .iter_mut()
            .map(|(name, m)| Ok((name.clone(), m.feed(audio)?)))
            .collect()
    }
}

impl std::fmt::Debug for WakeWordModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WakeWordModel")
            .field("sample_rate", &self.sample_rate)
            .field("models", &self.model_names())
            .finish()
    }
}
