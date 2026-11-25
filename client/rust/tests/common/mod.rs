// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Fixtures shared with the Python, Swift and JavaScript tests.

#![allow(dead_code)]

use std::path::PathBuf;

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The shipped hey livekit model.
pub fn hey_livekit() -> PathBuf {
    root().join("models/hey_livekit.onnx")
}

/// The tiny streaming model shared with the Python tests.
pub fn tiny() -> PathBuf {
    fixture("streaming_tiny.onnx")
}

/// client/python/tests/fixtures/<name>
pub fn fixture(name: &str) -> PathBuf {
    root().join("client/python/tests/fixtures").join(name)
}

/// Reference outputs from the Python client (client/js/scripts/make_parity.py).
pub fn parity() -> serde_json::Value {
    let path = root().join("client/js/test/fixtures/parity.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

pub fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

/// A 16-bit mono PCM WAV as (sample rate, samples).
pub fn read_wav(name: &str) -> (u32, Vec<i16>) {
    let mut reader = hound::WavReader::open(fixture(name)).unwrap();
    let rate = reader.spec().sample_rate;
    (rate, reader.samples::<i16>().map(Result::unwrap).collect())
}

/// Seeded uniform noise in [-amp, amp] (the same as `noise()` in the JS tests).
pub fn noise(n: usize, amp: f32, seed: u32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s as f64 / 2f64.powi(32) - 0.5) * 2.0 * amp as f64) as f32
        })
        .collect()
}

pub fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "lengths differ");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "step {i}: {a} vs {b} (tolerance {tol})"
        );
    }
}
