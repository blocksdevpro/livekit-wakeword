"""Generate the speech with ElevenLabs instead of LiveKit Inference.

livekit-wakeword-bench still plans every clip (texts, voices, speeds, the held-out voice split)
and builds the datasets (noise, rooms, augmentation); only the TTS call changes. `patched()`
swaps, for the duration of a block:

- the bench's voice catalog for the voices of your ElevenLabs account (`GET /v1/voices`: the
  default voices plus any you added from the Voice Library), each offered on every model in
  `MODELS` that speaks the phrase's language. The list is saved next to the dataset
  (`elevenlabs_voices.json`) the first time, so the held-out test voices stay the same between
  `setup` and `eval`; delete it to pick up voices added later.
- the held-out split: by speaker, so a test voice is unseen on every model.
- phoneme steering: the key phrase is sent as CMU ARPAbet `<phoneme>` tags, which ElevenLabs
  reads on Eleven Flash v2 (English) only.
- speed: `voice_settings.speed` (0.7 to 1.2).
- the synthesizer: `POST /v1/text-to-speech/{voice_id}` returning 16-bit PCM, written into the
  bench's cache like any other clip (the bench's retries, checks and cache are unchanged).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

API = "https://api.elevenlabs.io"
ENV_KEY = "ELEVENLABS_API_KEY"
PROVIDER = "elevenlabs"
VOICES_FILE = "elevenlabs_voices.json"
# concurrent requests when `concurrency` is not set: the free plan's limit (paid plans allow
# more: Starter 3, Creator 5, Pro 10, Scale/Business 15; `setup --concurrency N`)
DEFAULT_CONCURRENCY = 2
SPEED = (0.7, 1.2)
PCM_RATES = (8000, 16000, 22050, 24000, 44100, 48000)

# phoneme tags (CMU ARPAbet) are read by Eleven Flash v2 only, which speaks English only
PHONEME_MODEL = f"{PROVIDER}/eleven_flash_v2"
_MULTILINGUAL_V2 = (
    "en ja zh de hi fr ko pt it es id nl tr fil pl sv bg ro ar cs el fi hr ms sk da ta uk ru"
)
# models voices are offered on, with the languages each speaks (codes without a region)
MODELS: dict[str, frozenset[str]] = {
    PHONEME_MODEL: frozenset({"en"}),
    f"{PROVIDER}/eleven_multilingual_v2": frozenset(_MULTILINGUAL_V2.split()),
    f"{PROVIDER}/eleven_flash_v2_5": frozenset(f"{_MULTILINGUAL_V2} hu no vi".split()),
}
LANGUAGE_CODE_MODELS = {f"{PROVIDER}/eleven_flash_v2_5"}  # the others reject `language_code`


def api_key() -> str | None:
    return os.getenv(ENV_KEY, "").strip() or None


def is_elevenlabs(model: str) -> bool:
    return model.split("/")[0] == PROVIDER


class ElevenLabsError(RuntimeError):
    """An error response. `BAD_REQUEST` in the message makes the bench stop retrying."""

    def __init__(self, status: int, detail: str) -> None:
        retry = status == 429 or status >= 500
        super().__init__(f"{'' if retry else 'BAD_REQUEST '}ElevenLabs {status}: {detail}")
        self.status = status


def _detail(body: bytes) -> str:
    try:
        d = json.loads(body).get("detail", body.decode(errors="replace"))
    except (ValueError, AttributeError):
        return body.decode(errors="replace")[:200]
    if isinstance(d, dict):
        return f"{d.get('status', '')} {d.get('message', '')}".strip()
    return str(d)[:200]


# voices


def fetch_voices(key: str) -> list[dict[str, Any]]:
    """The account's voices, as {id, name, gender, accent, category}."""
    req = urllib.request.Request(f"{API}/v1/voices", headers={"xi-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        raise ElevenLabsError(e.code, _detail(e.read())) from None
    out = []
    for v in data.get("voices", []):
        labels = v.get("labels") or {}
        out.append(
            {
                "id": v["voice_id"],
                "name": v.get("name") or v["voice_id"],
                "gender": labels.get("gender"),
                "accent": labels.get("accent"),
                "category": v.get("category"),
            }
        )
    return sorted(out, key=lambda v: v["id"])


def voice_catalog(dataset_dir: Path, key: str | None) -> list[dict[str, Any]]:
    """The voices a phrase's datasets are made with: saved in `dataset_dir` on first use."""
    path = Path(dataset_dir) / VOICES_FILE
    if path.exists():
        voices: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
        return voices
    if key is None:
        raise RuntimeError(f"set {ENV_KEY} to list your ElevenLabs voices")
    voices = fetch_voices(key)
    if not voices:
        raise RuntimeError("your ElevenLabs account has no voices")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(voices, indent=1, ensure_ascii=False), encoding="utf-8")
    return voices


def bench_voices(catalog: list[dict[str, Any]], models: list[str] | None = None) -> list[Any]:
    """Every voice on every model in `MODELS` (or `models`), as bench `Voice`s."""
    from livekit_wakeword_bench.voices import Voice, normalize_accent

    return [
        Voice(
            model=m,
            id=v["id"],
            name=v["name"],
            gender=v.get("gender") or "unknown",
            accent=normalize_accent(v.get("accent")),
            accent_detail=v.get("accent") or "",
            languages=langs,
        )
        for m, langs in MODELS.items()
        if not models or m in models
        for v in catalog
    ]


def split_by_speaker(
    voices: list[Any], test_fraction: float, seed: int = 0
) -> tuple[list[Any], list[Any]]:
    """The bench's held-out split, by speaker: the bench splits per model, which with one voice
    list on several models would put a test speaker in training on another model."""
    import hashlib

    def rank(i: str) -> int:
        return int(hashlib.sha1(f"{seed}:{PROVIDER}:{i}".encode()).hexdigest(), 16)

    ids = sorted({v.id for v in voices}, key=rank)
    n_test = max(1, math.ceil(test_fraction * len(ids))) if len(ids) > 1 else 0
    test_ids = set(ids[:n_test])
    return [v for v in voices if v.id not in test_ids], [v for v in voices if v.id in test_ids]


# phonemes


def render_phrase(text: str, overrides: dict[str, str] | None = None) -> str | None:
    """Each word of `text` as a CMU ARPAbet phoneme tag, punctuation kept (it shapes prosody);
    None when a word has no pronunciation."""
    from livekit_wakeword_bench.phonemes import _ensure_primary, phrase_parts

    pp = phrase_parts(text, {k.lower(): v for k, v in (overrides or {}).items()})
    if pp is None:
        return None
    it = iter(pp)
    out = []
    for tok in re.split(r"([A-Za-z']+)", text):
        if re.fullmatch(r"[A-Za-z']+", tok):
            arpa = _ensure_primary([ph for part in next(it) for ph in part])
            out.append(f'<phoneme alphabet="cmu-arpabet" ph="{" ".join(arpa)}">{tok}</phoneme>')
        else:
            out.append(tok)
    return "".join(out)


def speed_kwargs(speed: float) -> dict[str, Any]:
    if abs(speed - 1.0) < 1e-3:
        return {}
    return {"speed": round(min(max(speed, SPEED[0]), SPEED[1]), 3)}


# synthesis


async def synthesize(
    http: Any,
    key: str,
    model: str,
    voice: str,
    text: str,
    speed: float = 1.0,
    language: str = "en",
    sample_rate: int = 24000,
    timeout_s: float = 25.0,
) -> bytes:
    """16-bit mono PCM for `text`. Waits and retries while over the plan's concurrency."""
    import aiohttp
    from livekit_wakeword_bench.voices import base_language

    if sample_rate not in PCM_RATES:
        raise ValueError(f"ElevenLabs PCM comes at {PCM_RATES} Hz, not {sample_rate}")
    body: dict[str, Any] = {"text": text, "model_id": model.split("/", 1)[1]}
    if s := speed_kwargs(speed):
        body["voice_settings"] = s
    if model in LANGUAGE_CODE_MODELS:
        body["language_code"] = base_language(language)
    url = f"{API}/v1/text-to-speech/{voice}?output_format=pcm_{sample_rate}"
    headers = {"xi-api-key": key, "accept": "audio/pcm"}
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    for wait in (1.0, 2.0, 4.0, 8.0, None):
        async with http.post(url, json=body, headers=headers, timeout=timeout) as r:
            data = await r.read()
            if r.status == 200:
                return bytes(data[: len(data) // 2 * 2])
            err = ElevenLabsError(r.status, _detail(data))
        if r.status != 429 or wait is None:
            raise err
        await asyncio.sleep(wait)
    raise AssertionError("unreachable")


class _Speech:
    """What the bench's `Synthesizer` expects from `inference.TTS`: `synthesize(text)` yielding
    events with `.frame.data` (int16 bytes), and `aclose()`."""

    def __init__(self, http: Any, key: str, req: Any, sample_rate: int) -> None:
        self._http, self._key, self._req, self._sr = http, key, req, sample_rate

    async def synthesize(self, text: str) -> AsyncIterator[Any]:
        r = self._req
        pcm = await synthesize(
            self._http, self._key, r.model, r.voice, text, r.speed, r.language, self._sr
        )
        yield SimpleNamespace(frame=SimpleNamespace(data=pcm))

    async def aclose(self) -> None:
        pass


def _synthesizer(base: type, key: str) -> type:
    """The bench's `Synthesizer` (cache, concurrency, retries) on ElevenLabs for its models."""

    class Synthesizer(base):  # type: ignore[misc]
        def _make_tts(self, req: Any) -> Any:
            if not is_elevenlabs(req.model):
                return super()._make_tts(req)
            return _Speech(self._http, key, req, self.sample_rate)

    return Synthesizer


def probe(key: str, voice: str, text: str, language: str) -> str | None:
    """One short request; None when it made audio, else the error."""
    import aiohttp

    model = next(m for m, langs in MODELS.items() if _speaks(langs, language))

    async def run() -> str | None:
        async with aiohttp.ClientSession() as http:
            try:
                pcm = await synthesize(http, key, model, voice, text, language=language)
            except Exception as e:  # noqa: BLE001
                return str(e) or type(e).__name__
        return None if pcm else "no audio"

    return asyncio.run(run())


def _speaks(langs: frozenset[str], language: str) -> bool:
    from livekit_wakeword_bench.voices import _ALIASES, base_language

    code = base_language(language)
    return bool({code, _ALIASES.get(code, code)} & langs)


def speaks(language: str) -> bool:
    """Whether any model in `MODELS` speaks `language`."""
    return any(_speaks(langs, language) for langs in MODELS.values())


@contextmanager
def patched(catalog: list[dict[str, Any]], key: str | None) -> Iterator[None]:
    """Inside the block, the bench generates speech with ElevenLabs (see the module doc)."""
    gen: Any = importlib.import_module("livekit_wakeword_bench.generate")
    tts: Any = importlib.import_module("livekit_wakeword_bench.tts")
    orig_render, orig_supports = gen.render_phrase, gen.supports_phonemes
    orig_options, orig_speed = gen.provider_options, tts.speed_kwargs

    def render(text: str, model: str, overrides: dict[str, str] | None = None) -> str | None:
        if is_elevenlabs(model):
            return render_phrase(text, overrides)
        return orig_render(text, model, overrides)  # type: ignore[no-any-return]

    def supports(model: str) -> bool:
        return model == PHONEME_MODEL if is_elevenlabs(model) else bool(orig_supports(model))

    def options(model: str) -> dict[str, Any]:
        return {} if is_elevenlabs(model) else orig_options(model)

    def speed(model: str, s: float) -> dict[str, Any]:
        return speed_kwargs(s) if is_elevenlabs(model) else orig_speed(model, s)

    patches: list[tuple[Any, str, Any]] = [
        (gen, "load_voices", lambda path=None, models=None: bench_voices(catalog, models)),
        (gen, "split_voices", split_by_speaker),
        (gen, "render_phrase", render),
        (gen, "supports_phonemes", supports),
        (gen, "provider_options", options),
        (gen, "speed_kwargs", speed),
        (tts, "speed_kwargs", speed),
    ]
    if key is not None:  # recording plans requests without a key
        patches.append((gen, "Synthesizer", _synthesizer(tts.Synthesizer, key)))
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    for mod, name, value in patches:
        setattr(mod, name, value)
    try:
        yield
    finally:
        for mod, name, value in saved:
            setattr(mod, name, value)
