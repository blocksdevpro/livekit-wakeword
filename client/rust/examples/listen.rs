// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Listens to the default microphone and prints detections.
//!
//!     cargo run --release --features listener --example listen -- [model.onnx] [threshold]
//!
//! Defaults to the hey livekit model in this repo and a threshold of 0.5.

use livekit_wakeword::{ListenerOptions, WakeWordListener, WakeWordModel};

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/hey_livekit.onnx").into());
    let threshold: f32 = match args.next() {
        Some(t) => t
            .parse()
            .map_err(|_| format!("threshold must be a number, got {t:?}"))?,
        None => 0.5,
    };

    let model = WakeWordModel::with_models([&model_path], 16_000)?;
    let names = model.model_names().join(", ");
    let listener = WakeWordListener::start(
        model,
        ListenerOptions {
            threshold,
            ..Default::default()
        },
    )?;
    println!(
        "listening for {names} at {} Hz (threshold {threshold}); Ctrl+C to stop",
        listener.sample_rate()
    );
    loop {
        let detection = listener.wait_for_detection()?;
        println!("Detected {} ({:.2})", detection.name, detection.confidence);
    }
}
