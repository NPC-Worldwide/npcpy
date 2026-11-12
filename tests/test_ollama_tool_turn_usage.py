from unittest.mock import MagicMock

from npcpy.gen import response as response_module

import pytest

from npcpy.gen.response import get_ollama_response


TOOLS = [{"type": "function", "function": {
    "name": "add", "description": "Add numbers",
    "parameters": {"type": "object", "properties": {}},
}}]


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    monkeypatch.setattr(response_module, "HAS_OLLAMA", True)
    monkeypatch.setattr(response_module, "ollama", MagicMock())


@pytest.mark.parametrize("stream", [False, True])
def test_plain_reply_with_tools_preserves_usage(stream):
    reply = {"message": {"content": "hello"}, "prompt_eval_count": 12, "eval_count": 3}
    client = response_module.ollama.Client
    client.return_value.chat.return_value = reply
    result = get_ollama_response(
        "hi", model="llama3.2", auto_process_tool_calls=True, tools=TOOLS, tool_map={"add": lambda: 3},
        stream=stream,
    )
    assert client.return_value.chat.call_count == 1
    assert result["usage"] == {"input_tokens": 12, "output_tokens": 3}
    if stream:
        assert list(result["response"]) == [reply]


@pytest.mark.parametrize("summary_usage, expected", [
    ({"prompt_eval_count": 50, "eval_count": 10}, (150, 30)),
    ({}, (100, 20)),
    ({"prompt_eval_count": None, "eval_count": None}, (100, 20)),
])
def test_tool_turn_usage_includes_both_calls(summary_usage, expected):
    first = {"message": {"content": "", "tool_calls": [
        {"function": {"name": "add", "arguments": {}}},
    ]}, "prompt_eval_count": 100, "eval_count": 20}
    summary = {"message": {"content": "3"}, **summary_usage}
    client = response_module.ollama.Client
    client.return_value.chat.side_effect = [first, summary]
    result = get_ollama_response(
        "add", model="llama3.2", auto_process_tool_calls=True, tools=TOOLS, tool_map={"add": lambda: 3},
    )
    assert client.return_value.chat.call_count == 2
    assert result["response"] == "3"
    assert result["tool_results"][0]["result"] == 3
    assert result["usage"] == dict(zip(("input_tokens", "output_tokens"), expected))
