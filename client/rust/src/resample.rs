// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

use crate::{Error, MODEL_SAMPLE_RATE};

const ATTENUATION_DB: f64 = 90.0;
const TRANSITION: f64 = 0.1; // transition band, as a fraction of the lower Nyquist rate

/// Streaming resampler from any rate to 16 kHz: a polyphase windowed-sinc FIR (Kaiser window,
/// about 90 dB stopband), the same filter as the Python, Swift and JavaScript clients, with state
/// carried between calls.
///
/// Output sample `m` is at input time `m * down / up`; it is computed as soon as the input
/// reaching that time has arrived, from the last [`taps`](Self::taps) input samples (zeros before
/// the stream starts).
#[derive(Debug, Clone)]
pub struct Resampler {
    rate: u32,
    up: u64,
    down: u64,
    taps: usize,
    bank: Vec<f32>, // bank[p * taps + k]: weight of input i - k at upsampled phase p
    history: Vec<f32>, // the last taps - 1 inputs, oldest first
    seen: u64,      // input samples consumed
    made: u64,      // output samples produced
}

impl Resampler {
    /// A resampler from `rate` Hz to 16 kHz.
    pub fn new(rate: u32) -> Result<Self, Error> {
        if rate == 0 {
            return Err(Error::InvalidSampleRate(rate));
        }
        let g = gcd(rate, MODEL_SAMPLE_RATE);
        let up = u64::from(MODEL_SAMPLE_RATE / g);
        let down = u64::from(rate / g);
        // low-pass in the upsampled domain: cut below the lower of the two Nyquist rates
        let nyq = 0.5 / up.max(down) as f64;
        let width = TRANSITION * nyq;
        let beta = 0.1102 * (ATTENUATION_DB - 8.7);
        let n = ((ATTENUATION_DB - 7.95) / (2.285 * 2.0 * std::f64::consts::PI * width)).ceil()
            as usize
            | 1;
        let up_n = up as usize;
        let taps = n.div_ceil(up_n);
        let n = taps * up_n;
        let cutoff = nyq - width / 2.0;
        let i0_beta = bessel_i0(beta);
        let h: Vec<f64> = (0..n)
            .map(|j| {
                let t = j as f64 - (n - 1) as f64 / 2.0;
                let w = if n > 1 {
                    let r = 2.0 * j as f64 / (n - 1) as f64 - 1.0;
                    bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta
                } else {
                    1.0
                };
                2.0 * cutoff * sinc(2.0 * cutoff * t) * w
            })
            .collect();
        let gain = up as f64 / h.iter().sum::<f64>(); // unity gain after zero-stuffing
        let mut bank = vec![0.0f32; up_n * taps];
        for p in 0..up_n {
            for k in 0..taps {
                bank[p * taps + k] = (h[k * up_n + p] * gain) as f32;
            }
        }
        Ok(Self {
            rate,
            up,
            down,
            taps,
            bank,
            history: vec![0.0; taps - 1],
            seen: 0,
            made: 0,
        })
    }

    /// The input rate, in Hz.
    pub fn rate(&self) -> u32 {
        self.rate
    }

    /// Input samples each output sample is computed from.
    pub fn taps(&self) -> usize {
        self.taps
    }

    /// Forgets all audio heard so far.
    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|x| *x = 0.0);
        self.seen = 0;
        self.made = 0;
    }

    /// Resamples the next mono audio; returns the 16 kHz samples it completes.
    pub fn process(&mut self, audio: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        self.process_into(audio, &mut out);
        out
    }

    /// Like [`process`](Self::process), appending to `out`.
    pub fn process_into(&mut self, audio: &[f32], out: &mut Vec<f32>) {
        let hist = self.taps - 1;
        let mut x = Vec::with_capacity(hist + audio.len());
        x.extend_from_slice(&self.history);
        x.extend_from_slice(audio);
        let base = self.seen as i64 - hist as i64; // absolute index of x[0]
        self.seen += audio.len() as u64;
        // outputs whose newest input index i = floor(m * down / up) has arrived
        if self.seen > 0 {
            let last = (self.seen * self.up - 1) / self.down; // largest m with i < seen
            out.reserve((last + 1).saturating_sub(self.made) as usize);
            for m in self.made..=last {
                let i = (m * self.down / self.up) as i64;
                let p = ((m * self.down) % self.up) as usize;
                let w = &self.bank[p * self.taps..(p + 1) * self.taps];
                let at = (i - base) as usize; // x[at] is input i
                let window = &x[at + 1 - self.taps..=at];
                // window is oldest first; w[k] weighs input i - k
                let acc: f32 = window.iter().rev().zip(w).map(|(a, b)| a * b).sum();
                out.push(acc);
            }
            self.made = last + 1;
        }
        self.history.copy_from_slice(&x[x.len() - hist..]);
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let y = std::f64::consts::PI * x;
    y.sin() / y
}

/// Modified Bessel function of the first kind, order 0 (power series).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..500 {
        let k = k as f64;
        term *= q / (k * k);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design() {
        let r = Resampler::new(48_000).unwrap();
        assert_eq!((r.up, r.down), (1, 3));
        let r = Resampler::new(44_100).unwrap();
        assert_eq!((r.up, r.down), (160, 441));
        assert!(Resampler::new(0).is_err());
    }

    #[test]
    fn identity_rate_passes_through() {
        let mut r = Resampler::new(16_000).unwrap();
        let x: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let y = r.process(&x);
        assert_eq!(y.len(), x.len());
        // a long linear-phase filter delays by (taps - 1) / 2; past the delay it is near identity
        let d = (r.taps() - 1) / 2;
        for (a, b) in y[d..].iter().zip(&x) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }
}
