"""Capture the real TTS traffic `setup` exchanges with LiveKit Inference.

`livekit-wakeword setup --capture-tts DIR` (or `LIVEKIT_WAKEWORD_CAPTURE_TTS=DIR`) runs the
normal pipeline, sending requests as usual, and writes everything that crosses the wire to a
JSONL file: what is sent, what comes back, how it is typed and how long it took. Unlike
`record.py` (which plans requests and sends none) this needs LiveKit credentials and spends TTS
quota; use `--clips 0.1` for a small capture.

Lines (one JSON object each; `seq` orders them, `t` is seconds since the capture started):

- `header`: versions, the gateway URL and a description of the wire protocol.
- `request_start`: a bench request (`TTSRequest` fields, its cache key and cache path).
- `tts_instance`: the resolved options of the `inference.TTS` built for one attempt (every
  default filled in; the API key and secret are never written).
- `ws_connect` / `ws_close` / `ws_error`: the websocket to `<gateway>/tts?model=...`, its
  request headers (Authorization and anything secret-looking redacted) and response status.
- `message`: one websocket message, `dir` = `send` or `recv`, `msg_type` its `type` field and
  `payload` the JSON, with audio replaced by its size and sha256 (the audio itself lives in the
  bench's cache, see the `cache_file` of `request_result`).
- `request_result`: per request, whether it worked, attempts, timing, the PCM audio returned
  (sample rate, channels, dtype, bytes, samples, seconds, sha256), message counts and the file
  the bench wrote.
- `schema`: for each message type seen, how often and with which fields and value types.
- `footer`: totals.

Messages sent while no bench request is running (the access probe `setup` makes first) have
`request_key: null`.
"""

from __future__ import annotations

import base64
import contextvars
import dataclasses
import hashlib
import importlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import IO, Any

ENV_VAR = "LIVEKIT_WAKEWORD_CAPTURE_TTS"
FORMAT_VERSION = 1
_SECRET_HEADER = re.compile(r"auth|token|key|secret|cookie|credential", re.I)

PROTOCOL = {
    "transport": "websocket to <gateway>/tts?model=<model> (gateway: LIVEKIT_INFERENCE_URL, "
    "or LiveKit's production gateway); one connection can serve several requests",
    "auth": "Authorization: Bearer <JWT signed with LIVEKIT_API_KEY / LIVEKIT_API_SECRET> "
    "(redacted in this file)",
    "client_to_server": {
        "session.create": "sent once per connection: sample_rate, encoding, extra, voice, "
        "model, language, optional fallback and connection {timeout, retries}",
        "input_transcript": "one per sentence: transcript, generation_config {voice, model, "
        "language}, extra (provider options such as speed)",
        "session.flush": "no more input; the server finishes and sends `done`",
    },
    "server_to_client": {
        "session.created": "carries session_id",
        "output_audio": "audio: base64 of raw PCM, int16 little-endian, mono, at the "
        "requested sample_rate (the client declares mime_type audio/pcm)",
        "output_alignment": "optional word or character timings: words [{word, start, end}] "
        "or chars [{char, start, end}]",
        "done": "synthesis finished",
        "error": "synthesis failed; the whole message is the error",
    },
    "returned_to_caller": "livekit.agents tts.ChunkedStream: async iterator of SynthesizedAudio "
    "events, each with a frame (int16 PCM, sample_rate, num_channels, samples_per_channel); "
    "the bench concatenates the frames into float32 mono and stores a FLAC",
    "source": "documented from livekit-agents inference/tts.py; message types actually "
    "observed are listed in the `schema` line",
}


def capture_dir() -> Path | None:
    """The directory captures go to, when capturing is switched on."""
    value = os.environ.get(ENV_VAR, "").strip()
    return Path(value) if value else None


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _jsonable(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    text = str(value)
    return None if text == "NOT_GIVEN" else text  # livekit's "option not set" sentinel


def _redact_headers(headers: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in dict(headers or {}).items():
        out[str(k)] = "<redacted>" if _SECRET_HEADER.search(str(k)) else str(v)
    return out


def _type_name(value: Any) -> str:
    return type(value).__name__


class Capture:
    """Writes capture lines and keeps the counters behind `request_result` and `schema`."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._f: IO[str] = path.open("w", encoding="utf-8")
        self._t0 = time.monotonic()
        self._seq = 0
        self._ws_ids = 0
        self.current: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
            "lk_capture_request", default=None
        )
        self.schema: dict[tuple[str, str], dict[str, Any]] = {}
        self.n_requests = 0
        self.n_failed = 0
        self.n_cache_hits = 0

    def now(self) -> float:
        return round(time.monotonic() - self._t0, 4)

    def write(self, kind: str, **fields: Any) -> None:
        self._seq += 1
        row = {"type": kind, "seq": self._seq, "t": self.now(), **fields}
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()

    def next_ws_id(self) -> int:
        self._ws_ids += 1
        return self._ws_ids

    # -- websocket messages -------------------------------------------------------------

    def message(self, direction: str, ws_id: int, raw: Any, msg_type: str) -> None:
        """Record one websocket message. `raw` is the text, or the aiohttp message for non-text."""
        req = self.current.get()
        payload: Any = None
        kind = "non-json"
        size = len(raw) if isinstance(raw, str | bytes) else 0
        if isinstance(raw, str):
            try:
                payload = json.loads(raw)
                kind = str(payload.get("type", "untyped")) if isinstance(payload, dict) else "json"
            except ValueError:
                payload = raw[:500]
        else:
            payload = raw
        extra: dict[str, Any] = {}
        if isinstance(payload, dict) and kind == "output_audio" and "audio" in payload:
            pcm = base64.b64decode(payload["audio"])
            extra = {"pcm_bytes": len(pcm), "pcm_sha256": hashlib.sha256(pcm).hexdigest()}
            payload = {**payload, "audio": f"<{len(pcm)} bytes of PCM, see pcm_sha256>"}
            if req is not None:
                req["pcm_bytes"] += len(pcm)
                req["hasher"].update(pcm)
                req["first_audio_t"] = req["first_audio_t"] or self.now()
        if req is not None:
            req["sent" if direction == "send" else "recv"][kind] += 1
            if direction == "send" and kind == "input_transcript":
                req["first_send_t"] = req["first_send_t"] or self.now()
        entry = self.schema.setdefault(
            (direction, kind), {"count": 0, "fields": defaultdict(Counter), "example": payload}
        )
        entry["count"] += 1
        if isinstance(payload, dict):
            for k, v in payload.items():
                entry["fields"][k][_type_name(v)] += 1
        self.write(
            "message",
            dir=direction,
            ws=ws_id,
            request_key=req["key"] if req else None,
            msg_type=kind if kind != "non-json" else msg_type,
            bytes=size,
            payload=payload if kind != "non-json" else str(payload)[:500],
            **extra,
        )

    def schema_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "dir": d,
                "msg_type": t,
                "count": e["count"],
                "fields": {k: dict(c) for k, c in e["fields"].items()},
                "example": e["example"],
            }
            for (d, t), e in sorted(self.schema.items())
        ]

    def close(self) -> None:
        self.write("schema", messages=self.schema_rows())
        self.write(
            "footer",
            n_requests=self.n_requests,
            n_failed=self.n_failed,
            n_cache_hits=self.n_cache_hits,
            seconds=self.now(),
        )
        self._f.close()


class _CapturedWebSocket:
    """Proxy around aiohttp's ClientWebSocketResponse that records what is sent and received."""

    def __init__(self, ws: Any, cap: Capture, ws_id: int) -> None:
        self._ws = ws
        self._cap = cap
        self._id = ws_id

    async def send_str(self, data: str, *args: Any, **kwargs: Any) -> None:
        self._cap.message("send", self._id, data, "text")
        await self._ws.send_str(data, *args, **kwargs)

    async def receive(self, *args: Any, **kwargs: Any) -> Any:
        try:
            msg = await self._ws.receive(*args, **kwargs)
        except (TimeoutError, Exception) as e:
            self._cap.write(
                "ws_error", ws=self._id, where="receive", error=type(e).__name__, detail=str(e)
            )
            raise
        if msg.type.name == "TEXT":
            self._cap.message("recv", self._id, msg.data, "text")
        else:
            self._cap.message("recv", self._id, f"{msg.type.name} {msg.data!r}", msg.type.name)
        return msg

    async def close(self, *args: Any, **kwargs: Any) -> Any:
        self._cap.write("ws_close", ws=self._id, close_code=getattr(self._ws, "close_code", None))
        return await self._ws.close(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ws, name)


class _RequestLogHandler(logging.Handler):
    """The bench logs why an attempt failed and swallows the exception; keep that text."""

    def __init__(self, cap: Capture) -> None:
        super().__init__(logging.WARNING)
        self.cap = cap

    def emit(self, record: logging.LogRecord) -> None:
        req = self.cap.current.get()
        if req is not None:
            req["failures"].append(record.getMessage())


def _tts_options(tts: Any) -> dict[str, Any]:
    """The resolved options of an `inference.TTS` (defaults filled in), without credentials."""
    opts = getattr(tts, "_opts", None)
    if opts is None or not dataclasses.is_dataclass(opts):
        return {}
    skip = {"api_key", "api_secret"}
    return {
        f.name: _jsonable(getattr(opts, f.name))
        for f in dataclasses.fields(opts)
        if f.name not in skip
    }


@contextmanager
def capturing(path: Path) -> Iterator[Capture]:
    """Record all LiveKit Inference TTS traffic made inside the block to `path`."""
    import aiohttp

    bench_tts: Any = importlib.import_module("livekit_wakeword_bench.tts")
    synthesizer: Any = bench_tts.Synthesizer
    cap = Capture(path)
    try:
        from livekit.agents.inference._utils import get_default_inference_url

        gateway: str | None = get_default_inference_url()
    except Exception:  # noqa: BLE001
        gateway = None
    cap.write(
        "header",
        format_version=FORMAT_VERSION,
        recorded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        gateway=gateway,
        versions={
            "livekit-wakeword": _version("livekit-wakeword"),
            "livekit-wakeword-bench": _version("livekit-wakeword-bench"),
            "livekit-agents": _version("livekit-agents"),
            "aiohttp": _version("aiohttp"),
            "python": sys.version.split()[0],
        },
        protocol=PROTOCOL,
    )

    orig_ws_connect = aiohttp.ClientSession.ws_connect
    orig_make_tts = synthesizer._make_tts
    orig_synthesize = synthesizer.synthesize

    async def ws_connect(self: Any, url: Any, *args: Any, **kwargs: Any) -> Any:
        ws_id = cap.next_ws_id()
        req = cap.current.get()
        base = {
            "ws": ws_id,
            "url": str(url),
            "request_key": req["key"] if req else None,
            "request_headers": _redact_headers(kwargs.get("headers")),
        }
        try:
            ws = await orig_ws_connect(self, url, *args, **kwargs)
        except Exception as e:
            cap.write(
                "ws_error",
                where="connect",
                error=type(e).__name__,
                detail=str(e),
                status=getattr(e, "status", None),
                **base,
            )
            raise
        response = getattr(ws, "_response", None)
        cap.write(
            "ws_connect",
            status=getattr(response, "status", None),
            response_headers=_redact_headers(getattr(response, "headers", None)),
            **base,
        )
        return _CapturedWebSocket(ws, cap, ws_id)

    def make_tts(self: Any, req: Any) -> Any:
        tts = orig_make_tts(self, req)
        ctx = cap.current.get()
        if ctx is not None:
            ctx["attempts"] += 1
        cap.write(
            "tts_instance",
            request_key=ctx["key"] if ctx else None,
            attempt=ctx["attempts"] if ctx else None,
            options=_tts_options(tts),
        )
        return tts

    async def synthesize(self: Any, req: Any, load: bool = True) -> Any:
        key = req.key()
        cache_path = self.cache_path(req)
        ctx: dict[str, Any] = {
            "key": key,
            "attempts": 0,
            "pcm_bytes": 0,
            "hasher": hashlib.sha256(),
            "first_send_t": None,
            "first_audio_t": None,
            "sent": Counter(),
            "recv": Counter(),
            "failures": [],
        }
        cap.n_requests += 1
        cap.write(
            "request_start",
            key=key,
            cache_path=str(cache_path),
            request=_jsonable(dataclasses.asdict(req)),
            bench={
                "sample_rate": self.sample_rate,
                "retries": self.retries,
                "timeout_s": self.timeout_s,
                "cache_hit": cache_path.exists(),
            },
        )
        token = cap.current.set(ctx)
        started = time.monotonic()
        try:
            result = await orig_synthesize(self, req, load)
        finally:
            cap.current.reset(token)
        ok = result is not None
        cache_hit = ctx["attempts"] == 0 and ok
        cap.n_failed += not ok
        cap.n_cache_hits += cache_hit
        pcm = ctx["pcm_bytes"]
        cache_file: dict[str, Any] | None = None
        if ok and cache_path.exists():
            import soundfile as sf

            info = sf.info(cache_path)
            cache_file = {
                "path": str(cache_path),
                "bytes": cache_path.stat().st_size,
                "format": info.format,
                "subtype": info.subtype,
                "samplerate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "seconds": round(info.duration, 4),
            }
        sample_rate = self.sample_rate
        # what the caller got back: the client appends a 10 ms silent `is_final` marker frame to
        # the wire audio of every segment (livekit-agents tts.AudioEmitter)
        returned = len(result) if (ok and not cache_hit and hasattr(result, "__len__")) else None
        cap.write(
            "request_result",
            key=key,
            ok=ok,
            cache_hit=cache_hit,
            attempts=ctx["attempts"],
            elapsed_s=round(time.monotonic() - started, 4),
            time_to_first_audio_s=(
                round(ctx["first_audio_t"] - ctx["first_send_t"], 4)
                if ctx["first_audio_t"] and ctx["first_send_t"]
                else None
            ),
            audio=None
            if not pcm
            else {
                "encoding": "PCM int16 little-endian (mime audio/pcm)",
                "sample_rate": sample_rate,
                "channels": 1,
                "bytes": pcm,
                "samples": pcm // 2,
                "seconds": round(pcm / 2 / sample_rate, 4),
                "sha256": ctx["hasher"].hexdigest(),
            },
            returned_samples=returned,
            client_added_samples=returned - pcm // 2 if returned is not None and pcm else None,
            messages={"sent": dict(ctx["sent"]), "received": dict(ctx["recv"])},
            failures=ctx["failures"],
            returned="float32 mono numpy array at the TTS sample rate"
            if ok and load
            else ("True/False through ensure(); the audio is in cache_file" if ok else "None"),
            cache_file=cache_file,
        )
        return result

    handler = _RequestLogHandler(cap)
    bench_logger = logging.getLogger("livekit_wakeword_bench.tts")
    aiohttp.ClientSession.ws_connect = ws_connect  # type: ignore[method-assign,assignment]
    synthesizer._make_tts = make_tts
    synthesizer.synthesize = synthesize
    bench_logger.addHandler(handler)
    try:
        yield cap
    finally:
        bench_logger.removeHandler(handler)
        synthesizer.synthesize = orig_synthesize
        synthesizer._make_tts = orig_make_tts
        aiohttp.ClientSession.ws_connect = orig_ws_connect  # type: ignore[method-assign]
        cap.close()
