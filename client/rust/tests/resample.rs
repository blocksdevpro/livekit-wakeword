// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

mod common;

use common::*;
use livekit_wakeword::Resampler;

/// The same signal as `chirp()` in client/js/scripts/make_parity.py.
fn chirp(rate: u32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            (0.3 * (2.0 * std::f64::consts::PI * (200.0 + 1500.0 * t) * t).sin()) as f32
        })
        .collect()
}

#[test]
fn matches_the_python_resampler() {
    let parity = parity();
    for rate in [44_100u32, 48_000] {
        let want = floats(&parity["resample"][rate.to_string()]);
        let got = Resampler::new(rate)
            .unwrap()
            .process(&chirp(rate, rate as usize / 4));
        assert_close(&got, &want, 1e-5);
    }
}

#[test]
fn chunking_does_not_change_the_output() {
    for rate in [8_000u32, 22_050, 44_100, 48_000] {
        let x = chirp(rate, rate as usize / 2);
        let whole = Resampler::new(rate).unwrap().process(&x);
        let mut r = Resampler::new(rate).unwrap();
        let mut pieces = Vec::new();
        let mut pos = 0;
        for size in [1usize, 7, 480, 1023, 3, 4410].into_iter().cycle() {
            if pos >= x.len() {
                break;
            }
            let end = x.len().min(pos + size);
            r.process_into(&x[pos..end], &mut pieces);
            pos = end;
        }
        assert_eq!(whole, pieces, "{rate} Hz");
        // output m covers input time m * rate / 16000
        assert_eq!(
            whole.len(),
            (x.len() * 16_000).div_ceil(rate as usize),
            "{rate} Hz"
        );
    }
}

#[test]
fn reset_starts_over() {
    let x = chirp(48_000, 4800);
    let mut r = Resampler::new(48_000).unwrap();
    let a = r.process(&x);
    r.reset();
    assert_eq!(r.process(&x), a);
}
