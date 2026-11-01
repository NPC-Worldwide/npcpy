"""Regression test for issue #332.

get_ollama_response must not call client.chat twice when stream=True,
tools/tool_map are set, and the model returns a plain reply (no tool calls).
Before the fix the function made a non-streaming call to check for tool calls
and then, finding none, issued a second streaming call for the same prompt.
"""
import pytest

from npcpy.gen import response as response_module


class _FakeClient:
    calls = []

    def __init__(self, **kwargs):
        pass

    def chat(self, **kwargs):
        type(self).calls.append(kwargs)
        return {"message": {"role": "assistant", "content": "hello", "tool_calls": None}}


@pytest.fixture
def fake_ollama(monkeypatch):
    _FakeClient.calls = []

    class _FakeOllama:
        Client = _FakeClient

    monkeypatch.setattr(response_module, "ollama", _FakeOllama)
    monkeypatch.setattr(response_module, "HAS_OLLAMA", True)
    return _FakeClient


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dummy_tool",
            "description": "A dummy tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]
_TOOL_MAP = {"dummy_tool": lambda: "result"}


def test_stream_with_tools_calls_chat_once(fake_ollama):
    """Only one client.chat call should be made when the reply has no tool calls."""
    result = response_module.get_ollama_response(
        prompt="hi",
        model="llama3.2:latest",
        messages=[],
        stream=True,
        tools=_TOOLS,
        tool_map=_TOOL_MAP,
        auto_process_tool_calls=True,
    )

    assert len(fake_ollama.calls) == 1, (
        f"Expected 1 client.chat call, got {len(fake_ollama.calls)}"
    )


def test_stream_with_tools_response_is_iterable(fake_ollama):
    """result['response'] must be iterable (a stream) when stream=True with tools."""
    result = response_module.get_ollama_response(
        prompt="hi",
        model="llama3.2:latest",
        messages=[],
        stream=True,
        tools=_TOOLS,
        tool_map=_TOOL_MAP,
        auto_process_tool_calls=True,
    )

    chunks = list(result["response"])
    assert len(chunks) == 1
    assert chunks[0]["message"]["content"] == "hello"
