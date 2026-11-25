"""`setup --record-tts`: the TTS requests are written to a manifest, none are sent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("livekit_wakeword_bench")  # Python 3.12+, the train extra

from livekit.wakeword_train.config import WakeWordConfig  # noqa: E402
from livekit.wakeword_train.record import record_requests  # noqa: E402


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_manifest_lists_every_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(var, raising=False)  # recording needs no credentials
    monkeypatch.chdir(tmp_path)  # the bench's cache dir is relative
    config = WakeWordConfig(wakeword="hey robot")
    path = record_requests(
        config,
        tmp_path / "steered",
        ["train"],
        tmp_path / "rec",
        splits__train__n_positive=6,
        splits__train__n_adversarial=6,
        splits__train__n_speech=3,
        language="en",
    )
    rows = _rows(path)
    header = rows[0]
    requests = {r["key"]: r for r in rows if r["type"] == "request"}
    clips = [r for r in rows if r["type"] == "clip"]

    assert header["type"] == "header" and header["wakeword"] == "hey robot"
    assert header["n_requests"] == len(requests) and header["n_clips"] == len(clips) == 15
    assert header["audio"]["sample_rate"] == 24000
    for clip in clips:  # every clip's audio can be found among the requests
        assert set(clip["request_keys"]) <= set(requests)
    for key, req in requests.items():
        assert req["cache_path"] == f"tts/{key[:2]}/{key}.flac"
        assert req["inference_kwargs"]["voice"] == req["voice"]["id"]
        assert req["text"] and req["source_text"]
    keyed = [r for r in requests.values() if r["is_key_phrase"]]
    assert keyed and all(r["arpabet"] for r in keyed)  # phoneme-steered set
    assert not (tmp_path / ".cache").exists()  # nothing downloaded or synthesized
