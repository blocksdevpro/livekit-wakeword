"""ElevenLabs TTS: the bench plans the clips as usual, ElevenLabs speaks them (no network here)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("livekit_wakeword_bench")  # Python 3.12+, the train extra

from livekit.wakeword_train import elevenlabs, pipeline  # noqa: E402
from livekit.wakeword_train.config import WakeWordConfig  # noqa: E402
from livekit.wakeword_train.record import ENV_VAR as RECORD_ENV  # noqa: E402

CATALOG = [
    {"id": f"voice{i:02d}", "name": f"V{i}", "gender": g, "accent": "american", "category": "x"}
    for i, g in enumerate(["female", "male"] * 5)
]


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class _Response:
    def __init__(self, status: int, body: bytes) -> None:
        self.status, self._body = status, body

    async def read(self) -> bytes:
        return self._body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> None:
        pass


class _Http:
    """An aiohttp session answering from a list of (status, body), recording each request."""

    def __init__(self, *answers: tuple[int, bytes]) -> None:
        self.answers = list(answers)
        self.sent: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any], headers: dict[str, str], **_: Any) -> _Response:
        self.sent.append({"url": url, "json": json, "headers": headers})
        return _Response(*self.answers.pop(0))

    async def close(self) -> None:
        pass


def test_phrase_becomes_arpabet_phoneme_tags() -> None:
    out = elevenlabs.render_phrase("Hey, LiveKit!", {"livekit": "L AY1 V | K IH2 T"})
    assert out == (
        '<phoneme alphabet="cmu-arpabet" ph="HH EY1">Hey</phoneme>, '
        '<phoneme alphabet="cmu-arpabet" ph="L AY1 V K IH2 T">LiveKit</phoneme>!'
    )
    assert elevenlabs.render_phrase("hey zxqv") is None  # no pronunciation


def test_held_out_voices_are_unseen_on_every_model() -> None:
    voices = elevenlabs.bench_voices(CATALOG)
    assert {v.model for v in voices} == set(elevenlabs.MODELS)
    train, test = elevenlabs.split_by_speaker(voices, 0.3, seed=1234)
    assert {v.id for v in train}.isdisjoint({v.id for v in test})
    assert len({v.id for v in test}) == 3  # ceil(0.3 * 10 speakers)


def test_setup_plans_elevenlabs_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole setup path, recorded instead of sent: ElevenLabs voices and models, the key
    phrase in phoneme tags on Flash v2, speeds within ElevenLabs' range."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(elevenlabs.ENV_KEY, raising=False)  # recording sends nothing
    monkeypatch.setenv(RECORD_ENV, str(tmp_path / "rec"))
    config = WakeWordConfig(wakeword="hey robot", data_dir=tmp_path / "data")
    assert config.tts == "elevenlabs"
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / elevenlabs.VOICES_FILE).write_text(json.dumps(CATALOG))
    knobs = {"splits__train__n_positive": 8, "splits__train__n_adversarial": 4}
    pipeline._generate(config, tmp_path / "data" / "steered", ["train"], **knobs)
    rows = _rows(tmp_path / "rec" / "tts_requests_steered_train.jsonl")
    requests = [r for r in rows if r["type"] == "request"]
    assert requests and {r["model"] for r in requests} == {elevenlabs.PHONEME_MODEL}
    assert all(r["voice"]["id"].startswith("voice") for r in requests)
    keyed = [r for r in requests if r["is_key_phrase"]]
    assert keyed and all('<phoneme alphabet="cmu-arpabet"' in r["text"] for r in keyed)
    assert all(0.7 <= r["speed"] <= 1.2 for r in requests)
    assert rows[0]["bench_config"]["tts"]["concurrency"] == elevenlabs.DEFAULT_CONCURRENCY

    # plain text: every model that speaks the language, no phoneme tags
    pipeline._generate(
        config, tmp_path / "data" / "natural", ["train"], tts__phonemes=False, **knobs
    )
    rows = _rows(tmp_path / "rec" / "tts_requests_natural_train.jsonl")
    requests = [r for r in rows if r["type"] == "request"]
    assert {r["model"] for r in requests} <= set(elevenlabs.MODELS)
    assert not any("<phoneme" in r["text"] for r in requests)


def test_synthesize_request_and_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_wait(_: float) -> None:
        pass

    monkeypatch.setattr(elevenlabs.asyncio, "sleep", no_wait)
    pcm = np.arange(100, dtype=np.int16).tobytes()
    busy = json.dumps({"detail": {"status": "too_many_concurrent_requests"}}).encode()
    http = _Http((429, busy), (200, pcm + b"\x00"))
    model = "elevenlabs/eleven_flash_v2_5"
    out = asyncio.run(elevenlabs.synthesize(http, "k", model, "v1", "hola", 1.5, "es-MX"))
    assert out == pcm  # the odd trailing byte is dropped
    assert len(http.sent) == 2  # retried after the 429
    sent = http.sent[-1]
    assert sent["url"].endswith("/v1/text-to-speech/v1?output_format=pcm_24000")
    assert sent["headers"]["xi-api-key"] == "k"
    assert sent["json"] == {
        "text": "hola",
        "model_id": "eleven_flash_v2_5",
        "voice_settings": {"speed": 1.2},  # clipped to ElevenLabs' range
        "language_code": "es",
    }
    http = _Http((200, pcm))
    asyncio.run(elevenlabs.synthesize(http, "k", elevenlabs.PHONEME_MODEL, "v1", "hi"))
    assert http.sent[0]["json"] == {"text": "hi", "model_id": "eleven_flash_v2"}


def test_refusals_are_not_retried() -> None:
    quota = json.dumps({"detail": {"status": "quota_exceeded", "message": "no credits"}})
    http = _Http((401, quota.encode()))
    with pytest.raises(elevenlabs.ElevenLabsError, match="BAD_REQUEST.*401.*quota_exceeded"):
        asyncio.run(elevenlabs.synthesize(http, "k", elevenlabs.PHONEME_MODEL, "v", "hi"))


def test_bench_synthesizer_caches_elevenlabs_audio(tmp_path: Path) -> None:
    from livekit_wakeword_bench.tts import Synthesizer, TTSRequest

    tone = (np.sin(np.arange(24000) / 10) * 8000).astype(np.int16)
    http = _Http((200, tone.tobytes()))

    async def run() -> np.ndarray | None:
        cls = elevenlabs._synthesizer(Synthesizer, "k")
        async with cls(tmp_path) as synth:
            synth._http = http
            req = TTSRequest(elevenlabs.PHONEME_MODEL, "voice00", "hey robot", 1.1)
            audio = await synth.synthesize(req)
            assert synth.cache_path(req).exists()
            return audio

    audio = asyncio.run(run())
    assert audio is not None and len(audio) == 24000
    assert http.sent[0]["json"]["voice_settings"] == {"speed": 1.1}


def test_environment_asks_for_the_elevenlabs_key(
    sample_config: WakeWordConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    for k in (elevenlabs.ENV_KEY, "LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        monkeypatch.delenv(k, raising=False)
    problems = pipeline.check_environment(sample_config)
    assert any(elevenlabs.ENV_KEY in p for p in problems)
    assert not any("LIVEKIT_URL" in p for p in problems)  # LiveKit is not used
    monkeypatch.setenv(elevenlabs.ENV_KEY, "k")
    assert not any(elevenlabs.ENV_KEY in p for p in pipeline.check_environment(sample_config))
    tlh = sample_config.model_copy(update={"language": "tlh"})
    assert any("speaks 'tlh'" in p for p in pipeline.check_environment(tlh))


def test_voice_list_is_saved_with_the_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fetched once, so the held-out voices stay put between setup and eval."""
    fetched: list[str] = []
    monkeypatch.setattr(elevenlabs, "fetch_voices", lambda key: fetched.append(key) or CATALOG)
    assert elevenlabs.voice_catalog(tmp_path, "k") == CATALOG
    assert elevenlabs.voice_catalog(tmp_path, None) == CATALOG  # from the file, no key needed
    assert fetched == ["k"]


def test_setup_stops_fast_when_elevenlabs_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import livekit_wakeword_bench

    def generate(*a: object, **k: object) -> None:
        raise AssertionError("generated after a refused test request")

    monkeypatch.setenv(elevenlabs.ENV_KEY, "bad")
    monkeypatch.setattr(pipeline, "_TTS_OK", set())
    monkeypatch.setattr(livekit_wakeword_bench, "generate", generate)
    monkeypatch.setattr(elevenlabs, "fetch_voices", lambda key: CATALOG)
    monkeypatch.setattr(elevenlabs, "probe", lambda *a: "BAD_REQUEST ElevenLabs 401: invalid")
    cfg = WakeWordConfig(wakeword="hey robot", data_dir=tmp_path)
    with pytest.raises(pipeline.SetupError, match=r"ElevenLabs refused.*401.*ELEVENLABS_API_KEY"):
        pipeline._generate(cfg, tmp_path / "steered", ["train"])
