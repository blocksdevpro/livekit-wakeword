// Copyright 2026 LiveKit, Inc.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::Scores;

/// A wake word heard.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// The model that fired.
    pub name: String,
    /// Its score, at or above the threshold.
    pub confidence: f32,
    /// When it fired.
    pub timestamp: Instant,
}

/// Turns scores into detections: a model fires when its score reaches `threshold`, at most once
/// per `debounce`. The same rule as the Python, Swift and JavaScript listeners; when several
/// models fire on one step, the highest score wins.
///
/// ```
/// use std::time::{Duration, Instant};
/// use livekit_wakeword::{DetectionGate, Scores};
///
/// let mut gate = DetectionGate::new(0.5, Duration::from_secs(2));
/// let scores = Scores::from([("hey_livekit".to_string(), 0.9)]);
/// let now = Instant::now();
/// assert!(gate.check(&scores, now).is_some());
/// assert!(gate.check(&scores, now + Duration::from_secs(1)).is_none()); // debounced
/// ```
#[derive(Debug, Clone)]
pub struct DetectionGate {
    threshold: f32,
    debounce: Duration,
    last: HashMap<String, Instant>,
}

impl DetectionGate {
    pub fn new(threshold: f32, debounce: Duration) -> Self {
        Self {
            threshold,
            debounce,
            last: HashMap::new(),
        }
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn debounce(&self) -> Duration {
        self.debounce
    }

    /// The model in `scores` that fires at time `now`, if any.
    pub fn check(&mut self, scores: &Scores, now: Instant) -> Option<Detection> {
        let (name, &confidence) = scores
            .iter()
            .filter(|&(name, &score)| {
                score >= self.threshold
                    && self
                        .last
                        .get(name)
                        .is_none_or(|&t| now.saturating_duration_since(t) >= self.debounce)
            })
            .max_by(|a, b| a.1.total_cmp(b.1))?;
        self.last.insert(name.clone(), now);
        Some(Detection {
            name: name.clone(),
            confidence,
            timestamp: now,
        })
    }

    /// Forgets past detections, so the next score over the threshold fires.
    pub fn reset(&mut self) {
        self.last.clear();
    }
}

impl Default for DetectionGate {
    /// Threshold 0.5 (calibrated to about 0.5 false alarms per hour), debounce 2 s.
    fn default() -> Self {
        Self::new(0.5, Duration::from_secs(2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(pairs: &[(&str, f32)]) -> Scores {
        pairs.iter().map(|&(n, s)| (n.to_string(), s)).collect()
    }

    #[test]
    fn fires_over_threshold_and_debounces_per_model() {
        let mut gate = DetectionGate::default();
        let t0 = Instant::now();
        assert!(gate.check(&scores(&[("a", 0.49)]), t0).is_none());
        let d = gate.check(&scores(&[("a", 0.5)]), t0).unwrap();
        assert_eq!((d.name.as_str(), d.confidence), ("a", 0.5));
        assert!(
            gate.check(&scores(&[("a", 0.9)]), t0 + Duration::from_millis(1999))
                .is_none()
        );
        // another model is not debounced by the first
        assert_eq!(
            gate.check(&scores(&[("a", 0.9), ("b", 0.6)]), t0)
                .unwrap()
                .name,
            "b"
        );
        assert!(
            gate.check(&scores(&[("a", 0.9)]), t0 + Duration::from_secs(2))
                .is_some()
        );
        gate.reset();
        assert!(
            gate.check(&scores(&[("a", 0.9)]), t0 + Duration::from_secs(2))
                .is_some()
        );
    }

    #[test]
    fn highest_score_wins() {
        let mut gate = DetectionGate::default();
        let d = gate.check(
            &scores(&[("a", 0.6), ("b", 0.8), ("c", 0.7)]),
            Instant::now(),
        );
        assert_eq!(d.unwrap().name, "b");
    }
}
