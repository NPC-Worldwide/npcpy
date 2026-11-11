import pytest

from npcpy import streaming
from npcpy.streaming import StreamConfig


@pytest.mark.parametrize("kind", ["chat", "tool"])
@pytest.mark.parametrize("temperature", [0.0, 0.3])
def test_stream_forwards_temperature(monkeypatch, kind, temperature):
    seen = []

    def fake_get_llm_response(*args, **kwargs):
        seen.append(kwargs)
        return iter([{"content": "hello"}])

    monkeypatch.setattr(streaming, "get_llm_response", fake_get_llm_response)
    cfg = StreamConfig(
        model="local-model",
        provider="ollama",
        temperature=temperature,
        max_tool_iterations=1,
    )
    if kind == "chat":
        list(streaming.create_chat_stream(cfg))
    else:
        list(streaming.create_tool_agent_stream(cfg, [], {}))

    assert seen[0].get("temperature") == temperature
