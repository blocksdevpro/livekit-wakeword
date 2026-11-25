// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

//! One streaming model, run with tract.

use tract_onnx::prelude::*;

use crate::{Error, FRAME_SAMPLES, WARMUP_FRAMES};

/// One model's state: inputs `chunk` (1, 1280) + state tensors, outputs `score` +
/// `<state>_out`. The state starts at zero and is warmed up on 2 s of silence, as in training.
pub(crate) struct Stream {
    label: String,
    run: TypedSimpleState,     // the plan, with buffers reused between steps
    chunk: usize,              // input index of `chunk`
    score: usize,              // output index of `score`
    feeds: Vec<Option<usize>>, // for each output, the input index it feeds next step
    state: TVec<TValue>,       // one per input (the `chunk` slot is replaced each step)
    warm: TVec<TValue>,
    pending: Vec<f32>,
}

impl Stream {
    /// Loads an `.onnx` file's bytes; `label` names it in errors.
    pub(crate) fn load(bytes: &[u8], label: &str) -> Result<Self, Error> {
        let invalid = |e: TractError| Error::InvalidModel {
            model: label.to_string(),
            reason: format!("{e:#}"),
        };
        let not_streaming = |reason: String| Error::NotStreaming {
            model: label.to_string(),
            reason,
        };

        let model = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(bytes))
            .map_err(invalid)?;
        let inputs: Vec<String> = model
            .input_outlets()
            .map_err(invalid)?
            .iter()
            .map(|o| model.node(o.node).name.clone())
            .collect();
        let chunk = inputs
            .iter()
            .position(|n| n == "chunk")
            .ok_or_else(|| not_streaming("no `chunk` input".into()))?;

        let model = model.into_typed().map_err(invalid)?;
        let outputs: Vec<String> = model
            .output_outlets()
            .map_err(invalid)?
            .iter()
            .map(|&o| match model.outlet_label(o) {
                Some(label) => label.to_string(),
                None => model.node(o.node).name.clone(),
            })
            .collect();
        let score = outputs
            .iter()
            .position(|n| n == "score")
            .ok_or_else(|| not_streaming("no `score` output".into()))?;

        // state inputs start at zero; their shapes must be fixed
        let mut state = TVec::with_capacity(inputs.len());
        for (i, name) in inputs.iter().enumerate() {
            let fact = model.input_fact(i).map_err(invalid)?;
            let shape = fact
                .shape
                .as_concrete()
                .ok_or_else(|| not_streaming(format!("input `{name}` has no fixed shape")))?;
            if i == chunk && shape != [1, FRAME_SAMPLES] {
                return Err(not_streaming(format!("`chunk` has shape {shape:?}")));
            }
            state.push(
                Tensor::zero_dt(fact.datum_type, shape)
                    .map_err(invalid)?
                    .into_tvalue(),
            );
        }
        let mut feeds = vec![None; outputs.len()];
        for (o, name) in outputs.iter().enumerate() {
            if o == score {
                continue;
            }
            let input = name
                .strip_suffix("_out")
                .and_then(|n| inputs.iter().position(|i| i == n))
                .ok_or_else(|| not_streaming(format!("output `{name}` feeds no input")))?;
            feeds[o] = Some(input);
        }
        for (i, name) in inputs.iter().enumerate() {
            if i != chunk && !feeds.contains(&Some(i)) {
                return Err(not_streaming(format!("no `{name}_out` output")));
            }
        }

        let plan = model
            .into_optimized()
            .map_err(invalid)?
            .into_runnable()
            .and_then(|plan| plan.spawn())
            .map_err(invalid)?;
        let mut stream = Self {
            label: label.to_string(),
            run: plan,
            chunk,
            score,
            feeds,
            warm: TVec::new(),
            state,
            pending: Vec::with_capacity(FRAME_SAMPLES),
        };
        let silence = [0.0f32; FRAME_SAMPLES];
        for _ in 0..WARMUP_FRAMES {
            stream.step(&silence)?;
        }
        stream.warm = stream.state.clone();
        Ok(stream)
    }

    /// Forgets all audio heard so far.
    pub(crate) fn reset(&mut self) {
        self.state.clone_from(&self.warm);
        self.pending.clear();
    }

    /// Feeds 16 kHz audio; returns the highest score among the 80 ms steps it completes (0.0 if
    /// none).
    pub(crate) fn feed(&mut self, mut audio: &[f32]) -> Result<f32, Error> {
        let mut best = 0.0f32;
        while !audio.is_empty() {
            let n = (FRAME_SAMPLES - self.pending.len()).min(audio.len());
            self.pending.extend_from_slice(&audio[..n]);
            audio = &audio[n..];
            if self.pending.len() == FRAME_SAMPLES {
                let chunk = std::mem::replace(&mut self.pending, Vec::with_capacity(FRAME_SAMPLES));
                best = best.max(self.step(&chunk)?);
            }
        }
        Ok(best)
    }

    fn step(&mut self, chunk: &[f32]) -> Result<f32, Error> {
        let failed = |e: TractError| Error::Inference {
            model: self.label.clone(),
            reason: format!("{e:#}"),
        };
        let mut inputs = self.state.clone();
        inputs[self.chunk] = Tensor::from_shape(&[1, FRAME_SAMPLES], chunk)
            .map_err(failed)?
            .into_tvalue();
        let outputs = self.run.run(inputs).map_err(failed)?;
        let score = outputs[self.score]
            .cast_to_scalar::<f32>()
            .map_err(failed)?;
        for (o, value) in outputs.into_iter().enumerate() {
            if let Some(i) = self.feeds[o] {
                self.state[i] = value;
            }
        }
        Ok(score)
    }
}
