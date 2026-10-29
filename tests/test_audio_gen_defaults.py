import asyncio
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
