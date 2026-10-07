"""Request construction and error mapping for the cloud STT engines and refiners, against
httpx.MockTransport (no network). Deepgram, ElevenLabs, AssemblyAI, Groq, Anthropic and Gemini
are not live-verified; these tests pin the request shapes we implemented from their docs."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from openwhisprflow.config import RefineConfig, SttConfig
from openwhisprflow.refine import anthropic, gemini, openai_compat
from openwhisprflow.refine.base import RefineContext, RefineError
from openwhisprflow.stt import assemblyai, deepgram, elevenlabs, groq
from openwhisprflow.stt.base import SttError

AUDIO = np.zeros(16000, dtype=np.float32)
KEY = lambda: "test-key"  # noqa: E731


def transport(handler):
    seen: list[httpx.Request] = []

    def wrapped(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)
    return httpx.MockTransport(wrapped), seen


def test_deepgram_request() -> None:
    t, seen = transport(lambda r: httpx.Response(200, json={"results": {"channels": [{"alternatives": [
        {"transcript": "hello world", "words": [{"word": "hello", "punctuated_word": "Hello", "start": 0.1,
                                                 "end": 0.4, "confidence": 0.9}]}]}]}}))
    eng = deepgram.create(SttConfig(engine="deepgram", language="en"), transport=t, key_getter=KEY)
    res = eng.transcribe(AUDIO, prompt="Kubernetes, Soniox")
    req = seen[0]
    assert req.url.host == "api.deepgram.com" and req.headers["authorization"] == "Token test-key"
    assert req.url.params.get_list("keyterm") == ["Kubernetes", "Soniox"]
    assert req.url.params["model"] == "nova-3" and req.content[:4] == b"RIFF"
    assert res.text == "hello world" and res.words and res.words[0].text == "Hello"


def test_elevenlabs_request() -> None:
    t, seen = transport(lambda r: httpx.Response(200, json={"text": "hi", "language_code": "en", "words": []}))
    eng = elevenlabs.create(SttConfig(engine="elevenlabs"), transport=t, key_getter=KEY)
    assert eng.transcribe(AUDIO, prompt="Acme").text == "hi"
    body = seen[0].content.decode("latin-1")
    assert seen[0].headers["xi-api-key"] == "test-key"
    for field in ('name="model_id"\r\n\r\nscribe_v2', 'name="file_format"\r\n\r\npcm_s16le_16',
                  'name="keyterms"\r\n\r\nAcme', 'name="tag_audio_events"\r\n\r\nfalse'):
        assert field in body


def test_assemblyai_flow() -> None:
    def handler(r: httpx.Request) -> httpx.Response:
        if r.url.path == "/v2/upload":
            return httpx.Response(200, json={"upload_url": "https://cdn/x"})
        if r.method == "POST":
            job = json.loads(r.content)
            assert job["speech_models"] == ["universal-3-5-pro"] and job["keyterms_prompt"] == ["Acme"]
            return httpx.Response(200, json={"id": "t1", "status": "queued"})
        if r.method == "GET":
            return httpx.Response(200, json={"id": "t1", "status": "completed", "text": "hi there",
                                             "language_code": "en", "words": []})
        return httpx.Response(200, json={})
    t, seen = transport(handler)
    eng = assemblyai.create(SttConfig(engine="assemblyai"), transport=t, key_getter=KEY)
    eng.poll_s = 0
    assert eng.transcribe(AUDIO, prompt="Acme").text == "hi there"
    assert seen[0].headers["authorization"] == "test-key"


@pytest.mark.parametrize("status,code,retryable", [(401, "auth", False), (402, "quota", False),
                                                   (429, "rate_limit", True), (503, "server", True)])
def test_stt_error_mapping(status: int, code: str, retryable: bool) -> None:
    t, _ = transport(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
    eng = groq.create(SttConfig(engine="groq"), transport=t, key_getter=KEY)
    with pytest.raises(SttError) as ei:
        eng.transcribe(AUDIO)
    assert (ei.value.code, ei.value.retryable) == (code, retryable)


def test_stt_timeout_is_retryable() -> None:
    def boom(r: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=r)
    t, _ = transport(boom)
    eng = groq.create(SttConfig(engine="groq"), transport=t, key_getter=KEY)
    with pytest.raises(SttError) as ei:
        eng.transcribe(AUDIO)
    assert ei.value.code == "timeout" and ei.value.retryable


def test_openai_compat_reasoning_params() -> None:
    assert openai_compat.model_params("gpt-4.1-nano") == {"temperature": 0}
    assert openai_compat.model_params("gpt-5.4-nano") == {"reasoning_effort": "none"}
    assert openai_compat.model_params("gpt-5-nano") == {"reasoning_effort": "minimal"}
    assert openai_compat.model_params("openai/gpt-oss-20b")["reasoning_effort"] == "low"


def test_openai_compat_retries_without_rejected_param() -> None:
    calls: list[dict] = []

    def handler(r: httpx.Request) -> httpx.Response:
        body = json.loads(r.content)
        calls.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'temperature'"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "Hi."}}]})
    t, seen = transport(handler)
    r = openai_compat.create(RefineConfig(provider="groq", model="some-model"), transport=t, key_getter=KEY)
    assert r.refine("hi", RefineContext(), timeout_s=5) == "Hi."
    assert len(calls) == 2 and seen[0].url.host == "api.groq.com" and "max_tokens" in calls[0]
    assert calls[0]["messages"][1]["content"] == "<dictation>\nhi\n</dictation>"


def test_anthropic_request() -> None:
    t, seen = transport(lambda r: httpx.Response(200, json={"content": [{"type": "text", "text": "Hi."}],
                                                             "stop_reason": "end_turn"}))
    r = anthropic.create(RefineConfig(provider="anthropic"), transport=t, key_getter=KEY)
    assert r.refine("hi", RefineContext(), timeout_s=5) == "Hi."
    body = json.loads(seen[0].content)
    assert seen[0].headers["x-api-key"] == "test-key" and seen[0].headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-haiku-4-5" and body["temperature"] == 0 and "system" in body


def test_gemini_request_skips_thoughts() -> None:
    t, seen = transport(lambda r: httpx.Response(200, json={"candidates": [{"content": {"parts": [
        {"text": "thinking...", "thought": True}, {"text": "Hi."}]}}]}))
    r = gemini.create(RefineConfig(provider="gemini"), transport=t, key_getter=KEY)
    assert r.refine("hi", RefineContext(), timeout_s=5) == "Hi."
    body = json.loads(seen[0].content)
    assert seen[0].url.path.endswith("/gemini-3.5-flash-lite:generateContent")
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}


def test_refine_auth_error() -> None:
    t, _ = transport(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
    r = anthropic.create(RefineConfig(provider="anthropic"), transport=t, key_getter=KEY)
    with pytest.raises(RefineError) as ei:
        r.refine("hi", RefineContext(), timeout_s=5)
    assert ei.value.code == "auth"
