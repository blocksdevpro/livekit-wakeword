// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

mod common;

use common::*;
use livekit_wakeword::{Error, FRAME_SAMPLES, WakeWordModel};

fn steps(
    model: &mut WakeWordModel,
    audio: &[impl livekit_wakeword::Sample],
    name: &str,
) -> Vec<f32> {
    audio
        .chunks(FRAME_SAMPLES)
        .map(|c| model.predict(c).unwrap()[name])
        .collect()
}

#[test]
fn matches_the_python_client_step_by_step() {
    let parity = parity();
    for clip in ["positive", "negative"] {
        let (_, audio) = read_wav(&format!("{clip}.wav"));
        let mut model = WakeWordModel::with_models([hey_livekit()], 16_000).unwrap();
        let got = steps(&mut model, &audio, "hey_livekit");
        assert_close(&got, &floats(&parity["scores"][clip]), 1e-4);
    }
}

#[test]
fn matches_the_python_client_on_the_tiny_model() {
    // streaming_tiny_expected.json: per-80 ms scores of a 3 s chirp (make_streaming_tiny.py)
    let expected: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(fixture("streaming_tiny_expected.json")).unwrap(),
    )
    .unwrap();
    let expected = floats(&expected["scores"]);
    let pcm: Vec<i16> = (0..16_000 * 3)
        .map(|i| {
            let t = i as f64 / 16_000.0;
            ((2.0 * std::f64::consts::PI * (150.0 + 400.0 * t) * t).sin() * 9000.0).trunc() as i16
        })
        .collect();
    let mut model = WakeWordModel::with_models([tiny()], 16_000).unwrap();
    let got = steps(
        &mut model,
        &pcm[..pcm.len() / FRAME_SAMPLES * FRAME_SAMPLES],
        "streaming_tiny",
    );
    assert_close(&got, &expected, 1e-4);
}

#[test]
fn detects_the_positive_clip_and_not_the_negative_one() {
    let mut model = WakeWordModel::with_models([hey_livekit()], 16_000).unwrap();
    let (rate, positive) = read_wav("positive.wav");
    assert_eq!(rate, 16_000);
    assert!(model.predict_clip(&positive).unwrap()["hey_livekit"] >= 0.5);
    assert!(model.predict_clip(&read_wav("negative.wav").1).unwrap()["hey_livekit"] < 0.5);
    // less than one 80 ms step
    assert_eq!(
        model.predict_clip(&positive[..800]).unwrap()["hey_livekit"],
        0.0
    );
}

#[test]
fn gives_the_same_steps_whatever_the_chunk_sizes() {
    let audio = noise(16_000 * 3, 0.05, 1);
    let mut a = WakeWordModel::with_models([tiny()], 16_000).unwrap();
    let frames = steps(&mut a, &audio, "streaming_tiny");
    let mut b = WakeWordModel::with_models([tiny()], 16_000).unwrap();
    let (mut best, mut pos) = (0.0f32, 0);
    for size in [700, 1860, 1280, 333, 2227].into_iter().cycle() {
        if pos >= audio.len() {
            break;
        }
        let end = audio.len().min(pos + size);
        best = best.max(b.predict(&audio[pos..end]).unwrap()["streaming_tiny"]);
        pos = end;
    }
    let max = frames.iter().copied().fold(0.0, f32::max);
    assert!((best - max).abs() < 1e-6, "{best} vs {max}");
    assert!(frames.iter().all(|s| (0.0..=1.0).contains(s)));
}

#[test]
fn carries_state_between_calls_until_reset() {
    let audio = noise(FRAME_SAMPLES * 20, 0.05, 1);
    let mut model = WakeWordModel::with_models([tiny()], 16_000).unwrap();
    let first = steps(&mut model, &audio, "streaming_tiny");
    let second = steps(&mut model, &audio, "streaming_tiny");
    assert_ne!(first, second); // the model remembers what it heard
    model.reset();
    assert_close(&steps(&mut model, &audio, "streaming_tiny"), &first, 1e-6);
}

#[test]
fn takes_int16_and_float_input_alike() {
    let (_, pcm) = read_wav("positive.wav");
    let float: Vec<f32> = pcm.iter().map(|&s| f32::from(s) / 32768.0).collect();
    let mut model = WakeWordModel::with_models([hey_livekit()], 16_000).unwrap();
    let a = model.predict_clip(&pcm).unwrap()["hey_livekit"];
    let b = model.predict_clip(&float).unwrap()["hey_livekit"];
    assert!((a - b).abs() < 1e-6);
}

#[test]
fn resamples_other_rates() {
    let mut model = WakeWordModel::with_models([hey_livekit()], 48_000).unwrap();
    assert_eq!(model.sample_rate(), 48_000);
    let (_, pcm) = read_wav("positive.wav");
    // 16 kHz -> 48 kHz by linear interpolation; the score survives the round trip
    let at = |j: usize| pcm.get(j).map_or(0.0, |&s| f32::from(s) / 32768.0);
    let up: Vec<f32> = (0..pcm.len() * 3)
        .map(|i| {
            let (j, f) = (i / 3, (i % 3) as f32 / 3.0);
            at(j) * (1.0 - f) + at(j + 1) * f
        })
        .collect();
    assert!(model.predict_clip(&up).unwrap()["hey_livekit"] >= 0.5);
}

#[test]
fn loads_names_and_unloads_models() {
    let mut model = WakeWordModel::new(16_000).unwrap();
    assert!(model.predict(&[0.0f32; FRAME_SAMPLES]).unwrap().is_empty());
    model
        .load_model_from_bytes(&std::fs::read(tiny()).unwrap(), "custom")
        .unwrap();
    model.load_model(tiny(), None).unwrap();
    assert_eq!(model.model_names(), ["custom", "streaming_tiny"]);
    model.load_model(tiny(), Some("custom")).unwrap(); // replaces, keeps its place
    assert_eq!(model.model_names(), ["custom", "streaming_tiny"]);
    let scores = model.predict(&[0.0f32; FRAME_SAMPLES]).unwrap();
    assert_eq!(scores.len(), 2);
    assert!(model.unload_model("custom"));
    assert!(!model.unload_model("custom"));
    assert_eq!(model.model_names(), ["streaming_tiny"]);
}

#[test]
fn explains_files_it_cannot_load() {
    let mut model = WakeWordModel::new(16_000).unwrap();
    let micro = root().join("models/hey_livekit_micro.tflite");
    assert!(matches!(
        model.load_model(&micro, None),
        Err(Error::MicroModel(_))
    ));
    assert!(matches!(
        model.load_model("missing.onnx", None),
        Err(Error::ModelNotFound(_))
    ));
    let bad = std::env::temp_dir().join("lkww-bad.onnx");
    std::fs::write(&bad, "not a model").unwrap();
    let err = model.load_model(&bad, None).unwrap_err();
    assert!(matches!(err, Error::InvalidModel { .. }), "{err}");
    assert!(err.to_string().contains("lkww-bad.onnx"), "{err}");
    assert!(model.model_names().is_empty());
    assert!(matches!(
        WakeWordModel::new(0),
        Err(Error::InvalidSampleRate(0))
    ));
}

#[test]
fn models_move_across_threads() {
    fn assert_send<T: Send>() {}
    assert_send::<WakeWordModel>();
    let mut model = WakeWordModel::with_models([tiny()], 16_000).unwrap();
    let handle = std::thread::spawn(move || model.predict(&[0i16; FRAME_SAMPLES]).unwrap());
    assert!(handle.join().unwrap().contains_key("streaming_tiny"));
}
