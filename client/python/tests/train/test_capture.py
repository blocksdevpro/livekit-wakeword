"""`setup --capture-tts`: the real inference client talks to a local stand-in gateway (the
protocol of livekit-agents' inference/tts.py) while every request and response is captured.

The server here is a test double: it checks the capture machinery, not LiveKit's real responses.
"""

from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("livekit_wakeword_bench")  # Python 3.12+, the train extra
pytest.importorskip("livekit.agents")
from aiohttp import web  # noqa: E402

from livekit.wakeword_train.capture import capturing  # noqa: E402

SECRET = "test-secret-0123456789-0123456789-0123456789"
SAMPLE_RATE = 24000


def _pcm(seconds: float) -> bytes:
    n = int(SAMPLE_RATE * seconds)
    return b"".join(
        struct.pack("<h", int(9000 * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE)))
        for i in range(n)
    )


async def _gateway(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    pcm = _pcm(0.5)
    async for msg in ws:
        data = json.loads(msg.data)
        if data["type"] == "session.create":
            await ws.send_json({"type": "session.created", "session_id": "sess-1"})
        elif data["type"] == "input_transcript":
            half = len(pcm) // 2
            for chunk in (pcm[:half], pcm[half:]):
                await ws.send_json(
                    {
                        "type": "output_audio",
                        "session_id": "sess-1",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
            await ws.send_json(
                {
                    "type": "output_alignment",
                    "session_id": "sess-1",
                    "words": [{"word": "hello", "start": 0.0, "end": 0.4}],
                }
            )
        elif data["type"] == "session.flush":
            await ws.send_json({"type": "done", "session_id": "sess-1"})
    return ws


async def test_capture_records_requests_and_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from livekit_wakeword_bench.tts import Synthesizer, TTSRequest

    app = web.Application()
    app.router.add_get("/tts", _gateway)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    monkeypatch.setenv("LIVEKIT_INFERENCE_URL", f"ws://127.0.0.1:{port}")
    monkeypatch.setenv("LIVEKIT_API_KEY", "devkey")
    monkeypatch.setenv("LIVEKIT_API_SECRET", SECRET)

    req = TTSRequest("cartesia/sonic-3", "voice-1", "hey robot", 0.95, "en")
    out = tmp_path / "capture.jsonl"
    try:
        with capturing(out):
            async with Synthesizer(tmp_path / "cache", concurrency=2) as synth:
                audio = await synth.synthesize(req)
                again = await synth.synthesize(req)  # served from the bench's cache: no traffic
    finally:
        await runner.cleanup()

    wire_samples = SAMPLE_RATE // 2  # what the gateway sent
    # the client adds a 10 ms silent marker frame at the end of the segment
    assert audio is not None and again is not None and len(audio) == wire_samples + 240
    text = out.read_text(encoding="utf-8")
    assert SECRET not in text and "Bearer ey" not in text  # credentials never reach the file
    rows: list[dict[str, Any]] = [json.loads(line) for line in text.splitlines()]
    kinds = [r["type"] for r in rows]
    assert kinds[0] == "header" and kinds[-1] == "footer"
    assert rows[0]["protocol"]["client_to_server"]["session.create"]

    connect = next(r for r in rows if r["type"] == "ws_connect")
    assert connect["url"].endswith("/tts?model=cartesia%2Fsonic-3")
    assert connect["status"] == 101
    assert connect["request_headers"]["Authorization"] == "<redacted>"

    instance = next(r for r in rows if r["type"] == "tts_instance")["options"]
    assert instance["voice"] == "voice-1" and instance["sample_rate"] == SAMPLE_RATE
    assert instance["extra_kwargs"] == {"speed": 0.95} and "api_secret" not in instance

    messages = [r for r in rows if r["type"] == "message"]
    sent = [m["msg_type"] for m in messages if m["dir"] == "send"]
    received = [m["msg_type"] for m in messages if m["dir"] == "recv"]
    assert sent == ["session.create", "input_transcript", "session.flush"]
    assert received == [
        "session.created",
        "output_audio",
        "output_audio",
        "output_alignment",
        "done",
    ]
    create = messages[0]["payload"]
    assert create["voice"] == "voice-1" and create["extra"] == {"speed": 0.95}
    transcript = next(m for m in messages if m["msg_type"] == "input_transcript")
    assert transcript["payload"]["transcript"].strip() == "hey robot"
    assert all(m["request_key"] == req.key() for m in messages)

    first, second = [r for r in rows if r["type"] == "request_result"]
    assert first["ok"] and not first["cache_hit"] and first["attempts"] == 1
    assert first["audio"]["sample_rate"] == SAMPLE_RATE and first["audio"]["channels"] == 1
    assert first["audio"]["bytes"] == len(_pcm(0.5)) and first["audio"]["seconds"] == 0.5
    assert first["messages"]["received"]["output_audio"] == 2
    assert first["audio"]["samples"] == wire_samples
    assert first["returned_samples"] == wire_samples + 240 and first["client_added_samples"] == 240
    assert first["cache_file"]["format"] == "FLAC" and first["cache_file"]["frames"] == 12240
    assert second["ok"] and second["cache_hit"] and second["attempts"] == 0

    schema = next(r for r in rows if r["type"] == "schema")["messages"]
    audio_schema = next(s for s in schema if s["msg_type"] == "output_audio")
    assert audio_schema["fields"]["audio"] == {"str": 2}
    assert rows[-1]["n_requests"] == 2 and rows[-1]["n_cache_hits"] == 1
