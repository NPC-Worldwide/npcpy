import asyncio
import base64
import json
import sys
import types

from npcpy.gen import audio_gen

# gemini-2.0-flash-exp is no longer served by the Gemini API.
RETIRED = {"gemini-2.0-flash-exp"}


def test_gemini_live_connect_default_model_is_not_retired(monkeypatch):
    sent = []

    class FakeWebSocket:
        async def send(self, message):
            sent.append(json.loads(message))

        async def recv(self):
            return json.dumps({"setupComplete": {}})

    async def connect(url):
        return FakeWebSocket()

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=connect))

    asyncio.run(audio_gen.gemini_live_connect(api_key="test-key"))

    model = sent[0]["setup"]["model"].removeprefix("models/")
    assert model not in RETIRED


def test_openai_realtime_tts_uses_ga_api(monkeypatch):
    sent = []
    seen = {}
    events = [
        {
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(b"\x01\x02").decode(),
        },
        {"type": "response.done"},
    ]

    class FakeWebSocket:
        async def send(self, message):
            sent.append(json.loads(message))

        async def recv(self):
            return json.dumps({"type": "session.created"})

        async def __aiter__(self):
            for event in events:
                yield json.dumps(event)

        async def close(self):
            pass

    async def connect(url, additional_headers=None):
        seen["url"] = url
        seen["headers"] = additional_headers
        return FakeWebSocket()

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=connect))

    audio = asyncio.run(audio_gen.tts_openai_realtime("hi", api_key="test-key"))

    assert audio == b"\x01\x02"
    assert "OpenAI-Beta" not in seen["headers"]
    assert "gpt-4o-realtime-preview" not in seen["url"]
    session = sent[0]["session"]
    assert session["type"] == "realtime"
    assert session["audio"]["output"]["voice"] == "alloy"


def test_kokoro_forwards_explicit_lang_code(monkeypatch):
    def fake_tts(text, *, voice, lang_code, speed):
        assert (text, voice, lang_code, speed) == ("Bonjour", "ff_siwis", "f", 1.2)
        return b"wav"

    monkeypatch.setattr(audio_gen, "tts_kokoro", fake_tts)

    assert audio_gen.text_to_speech(
        "Bonjour", engine="kokoro", voice="ff_siwis", lang_code="f", speed=1.2
    ) == b"wav"


def test_kokoro_infers_lang_code_from_voice(monkeypatch):
    def fake_tts(text, *, voice, lang_code):
        assert (text, voice, lang_code) == ("Hello", "bf_emma", "b")
        return b"wav"

    monkeypatch.setattr(audio_gen, "tts_kokoro", fake_tts)

    assert audio_gen.text_to_speech("Hello", engine="kokoro", voice="bf_emma") == b"wav"
