// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Scores a WAV clip step by step, and times the model.
//!
//!     cargo run --release --example inference -- [model.onnx] [clip.wav]
//!
//! Defaults to the hey livekit model and the positive test clip in this repo.

use std::time::Instant;

use livekit_wakeword::{FRAME_SAMPLES, MODEL_SAMPLE_RATE, WakeWordModel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .unwrap_or(format!("{root}/models/hey_livekit.onnx"));
    let clip = args
        .next()
        .unwrap_or(format!("{root}/client/python/tests/fixtures/positive.wav"));

    let mut reader = hound::WavReader::open(&clip)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err(format!("{clip}: expected 16-bit mono PCM").into());
    }
    let pcm: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>()?;

    let start = Instant::now();
    let mut model = WakeWordModel::with_models([&model_path], spec.sample_rate)?;
    println!(
        "loaded {model_path} in {:.0} ms",
        start.elapsed().as_secs_f64() * 1e3
    );

    // feed 80 ms at a time, as a microphone would
    let frame = (spec.sample_rate as usize * FRAME_SAMPLES).div_ceil(MODEL_SAMPLE_RATE as usize);
    let (mut best, mut steps) = (0.0f32, 0usize);
    let start = Instant::now();
    for chunk in pcm.chunks(frame) {
        for (name, score) in model.predict(chunk)? {
            if score >= 0.5 {
                println!("{:6.2} s  {name}: {score:.3}", (steps + 1) as f64 * 0.08);
            }
            best = best.max(score);
        }
        steps += 1;
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "highest score {best:.3} ({})",
        if best >= 0.5 {
            "detected"
        } else {
            "not detected"
        }
    );
    println!(
        "{steps} steps: {:.2} ms per 80 ms step, {:.1} ms of CPU per second of audio",
        elapsed * 1e3 / steps as f64,
        elapsed * 1e3 / (steps as f64 * 0.08)
    );
    Ok(())
}
