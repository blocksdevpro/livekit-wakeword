"""Record the TTS requests `setup` would send to LiveKit Inference instead of sending them.

`livekit-wakeword setup --record-tts DIR` (or `LIVEKIT_WAKEWORD_RECORD_TTS=DIR`) plans each
dataset exactly as livekit-wakeword-bench would, then writes every request to a JSONL manifest
and stops: no credentials, no network, no audio. Another TTS engine can read the manifest and
produce the audio later.

Manifest lines (one JSON object each, in this order):

- `header`: format version, library versions, the full bench config (seed, voices' language,
  pronunciations, hard negatives, splits...), how audio must be delivered (`audio`) and what the
  bench does with it afterwards (`downstream`).
- `request`: one per unique TTS call: the text actually sent (phonemes included), voice, speed,
  language, the exact `inference.TTS(...)` keyword arguments, and `cache_path`, where the bench
  looks for the audio. Write a FLAC (mono, `audio.sample_rate`) there and the unchanged bench
  uses it instead of calling TTS.
- `clip`: one per planned clip: split, category, text segments, which segment is the key phrase,
  the voice, the speed and the `request_keys` whose audio it is built from.
"""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from .config import WakeWordConfig

ENV_VAR = "LIVEKIT_WAKEWORD_RECORD_TTS"
FORMAT_VERSION = 1


def record_dir() -> Path | None:
    """The directory manifests go to, when recording is switched on."""
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value) if value else None


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


@contextmanager
def _no_noise_sources() -> Iterator[None]:
    """`Generator.__init__` downloads the noise, room and speech sources (about 4 GB). Planning
    TTS requests never touches them, so skip that one call and keep the rest of the real
    constructor (voice selection and the held-out split)."""
    # the package re-exports the `generate` function under the module's name: import the module
    bench_generate: Any = importlib.import_module("livekit_wakeword_bench.generate")

    original = bench_generate.load_pools
    bench_generate.load_pools = lambda *_a, **_k: None
    try:
        yield
    finally:
        bench_generate.load_pools = original


def _voice(voice: Any, held_out: bool) -> dict[str, Any]:
    return {
        "model": voice.model,
        "id": voice.id,
        "name": voice.name,
        "gender": voice.gender,
        "accent": voice.accent,
        "accent_detail": voice.accent_detail,
        "languages": sorted(voice.languages),
        "held_out": held_out,
    }


def record_requests(
    config: WakeWordConfig,
    out: Path,
    splits: list[str],
    record_to: Path,
    **knobs: Any,
) -> Path:
    """Plan the dataset `_generate` would build in `out` and write its TTS requests to a
    manifest under `record_to`; returns the manifest path. `knobs` are the bench knobs
    `_generate` passes to `livekit_wakeword_bench.generate` (dotted paths with `__`)."""
    from livekit_wakeword_bench.config import make_config
    from livekit_wakeword_bench.generate import Generator
    from livekit_wakeword_bench.phonemes import arpabet_string
    from livekit_wakeword_bench.tts import speed_kwargs

    kv = {k.replace("__", "."): v for k, v in knobs.items()}
    cfg = make_config(config.wakeword, None, "small", config.plan, kv)
    with _no_noise_sources():
        gen = Generator(cfg, out)

    requests: dict[str, dict[str, Any]] = {}
    clips: list[dict[str, Any]] = []
    for split in splits:
        for plan in gen.plan(split, cfg.splits[split]):
            clip: dict[str, Any] = {
                "type": "clip",
                "id": plan.id,
                "split": split,
                "category": plan.category,
                "request_keys": [],
            }
            if plan.utt is not None and plan.voice is not None:
                held_out = plan.voice in gen.test_voices
                reqs = gen._requests(plan)
                clip.update(
                    text=plan.utt.text,
                    text_form=plan.utt.form,
                    phrase=plan.utt.phrase,
                    adversarial_distance=plan.utt.distance,
                    segments=list(plan.utt.segments),
                    key_segment=plan.utt.key,
                    voice=f"{plan.voice.model}:{plan.voice.id}",
                    speed=round(plan.speed, 3),
                    request_keys=[r.key() for r in reqs],
                )
                for j, r in enumerate(reqs):
                    key = r.key()
                    if key in requests:
                        continue
                    is_key = j == plan.utt.key
                    requests[key] = {
                        "type": "request",
                        "key": key,
                        "cache_path": f"tts/{key[:2]}/{key}.flac",
                        # `text` is what LiveKit Inference receives (provider phoneme markup for
                        # a steered key phrase); `source_text` is the same segment as plain text
                        "text": r.text,
                        "source_text": plan.utt.segments[j],
                        "is_key_phrase": is_key,
                        "arpabet": arpabet_string(plan.utt.phrase, cfg.pronunciations)
                        if is_key and cfg.tts.phonemes
                        else "",
                        "model": r.model,
                        "provider": r.provider,
                        "voice": _voice(plan.voice, held_out),
                        "speed": r.speed,
                        "language": r.language,
                        "extra": r.extra,
                        "inference_kwargs": {
                            "model": r.model,
                            "voice": r.voice,
                            "language": r.language,
                            "sample_rate": cfg.tts.sample_rate,
                            "extra_kwargs": {**speed_kwargs(r.model, r.speed), **r.extra},
                        },
                    }
            clips.append(clip)

    header = {
        "type": "header",
        "format_version": FORMAT_VERSION,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "wakeword": config.wakeword,
        "dataset": out.name,
        "splits": splits,
        "versions": {
            "livekit-wakeword": _version("livekit-wakeword"),
            "livekit-wakeword-bench": _version("livekit-wakeword-bench"),
        },
        "n_requests": len(requests),
        "n_clips": len(clips),
        "audio": {
            "cache_dir": str(cfg.cache_path),
            "sample_rate": cfg.tts.sample_rate,
            "channels": 1,
            "format": "FLAC (any soundfile-readable mono float32 audio)",
            "min_duration_s": 0.15,
            "note": "write each request's audio to <cache_dir>/<cache_path>; the bench skips TTS "
            "for requests whose file exists",
        },
        "downstream": {
            "note": "what the bench does with each clip's audio (not part of the request)",
            "trim_silence": True,
            "resample_to": cfg.sample_rate,
            "segment_gap_s": [0.1, 0.4],
            "peak_normalize": 0.5,
            "then": "reverb, noise scenes and device effects (config.augment)",
        },
        "wakeword_config": config.model_dump(mode="json"),
        "bench_config": cfg.model_dump(mode="json"),
    }
    record_to.mkdir(parents=True, exist_ok=True)
    path = record_to / f"tts_requests_{out.name}_{'-'.join(splits)}.jsonl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in (header, *requests.values(), *clips):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path
