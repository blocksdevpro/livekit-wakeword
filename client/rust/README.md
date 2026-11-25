# Rust

Streaming wake word detection in pure Rust: models run on [tract](https://github.com/sonos/tract),
so there are no native libraries to install. Models come from
[training](../python/TRAINING.md) or the [release](../../README.md#models).

```bash
cargo add livekit-wakeword                       # detection only
cargo add livekit-wakeword --features listener   # plus the microphone listener (cpal)
```

The crate is not published from this repo yet; until then, depend on it from git:

```toml
[dependencies]
livekit-wakeword = { git = "https://github.com/livekit/livekit-wakeword" }
```

```rust
use livekit_wakeword::WakeWordModel;

// sample rate of the audio you feed; other rates than 16 kHz are resampled internally
let mut model = WakeWordModel::with_models(["hey_livekit.onnx"], 48_000)?;

// stream audio as it arrives: mono, i16 or f32 in [-1, 1], any chunk size
let scores = model.predict(&audio_chunk)?;
if scores["hey_livekit"] >= 0.5 {  // 0.5 is calibrated to about 0.5 false alarms per hour
    println!("Wake word detected!");
    model.reset();  // start fresh after a detection
}

model.predict_clip(&clip)?;  // highest score in a whole clip, from a fresh state

// add or replace models at runtime
let mut model = WakeWordModel::new(16_000)?;
model.load_model("hey_livekit.onnx", None)?;                  // named "hey_livekit"
model.load_model("another.onnx", Some("custom"))?;
model.load_model_from_bytes(include_bytes!("my.onnx"), "mine")?;
println!("{:?}", model.model_names());                       // in load order
model.unload_model("custom");
```

`predict()` returns one score per loaded model, keyed by file name stem. Each 80 ms of audio is one
model step and the state carries over between calls; a call returns each model's highest score
among the steps it completed (0.0 if none).

## Microphone listener

With the `listener` feature, `WakeWordListener` streams the default input device (or one you pass
to `start_with_device`) through a model on a background thread. It opens the microphone at its own
rate and resamples to 16 kHz itself.

```rust
use std::time::Duration;
use livekit_wakeword::{ListenerOptions, WakeWordListener, WakeWordModel};

let model = WakeWordModel::with_models(["hey_livekit.onnx"], 16_000)?;
let listener = WakeWordListener::start(model, ListenerOptions {
    threshold: 0.5,
    debounce: Duration::from_secs(2),     // minimum time between two detections of one wake word
    on_scores: None,                      // Some(Box::new(|scores| ...)) for meters and debugging
})?;
loop {
    let detection = listener.wait_for_detection()?;
    println!("Detected {} ({:.2})", detection.name, detection.confidence);
}
```

After a detection the listener pauses (and resets the model) until `wait_for_detection()` is
called again, so nothing fires while you handle it. `wait_for_detection_timeout(d)` returns
`Ok(None)` if nothing was heard in time. `stop()` closes the microphone and gives the model back;
dropping the listener closes it too. Calls block; from async code, run them on a blocking thread
(e.g. `tokio::task::spawn_blocking`).

On Linux the listener needs ALSA headers: `sudo apt install libasound2-dev`.

For your own audio source (a LiveKit track, a file, a socket), feed `predict()` and use
`DetectionGate`: the same threshold and debounce rule. When several models fire on one step, the
highest score wins.

```rust
use std::time::{Duration, Instant};
use livekit_wakeword::DetectionGate;

let mut gate = DetectionGate::new(0.5, Duration::from_secs(2));
if let Some(detection) = gate.check(&model.predict(&frame)?, Instant::now()) {
    model.reset();
}
```

## Performance

tract runs a step of the standard model in about 1 ms on one CPU thread (measured on x86-64 in a
release build), about 12 ms of CPU per second of audio. Build with `--release`: debug builds are
many times slower.

## Examples

From a clone of this repo, in `client/rust`:

```bash
cargo run --release --example inference -- [model.onnx] [clip.wav]   # scores a clip, times the model
cargo run --release --features listener --example listen -- ../../models/hey_livekit.onnx [threshold]
```

## Tests

```bash
cargo test --all-features
```

The tests share fixtures with the Python and JavaScript clients and check scores step by step
against the Python client (`client/js/test/fixtures/parity.json`,
`client/python/tests/fixtures/streaming_tiny_expected.json`).
