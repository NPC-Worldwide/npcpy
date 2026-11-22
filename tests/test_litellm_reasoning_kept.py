from types import SimpleNamespace
from unittest.mock import patch

from npcpy.gen.response import get_litellm_response


def _response(with_tool_call=True, **message_fields):
    tool_calls = None
    if with_tool_call:
        tool_calls = [SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="add", arguments="{}"),
        )]
    message = SimpleNamespace(
        content=None if with_tool_call else "4", tool_calls=tool_calls, **message_fields
    )
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _run(resp):
    with patch("npcpy.gen.response.completion", return_value=resp):
        return get_litellm_response(
            prompt="add",
            model="deepseek-v4-pro",
            provider="deepseek",
            api_key="sk-test",
            messages=[],
        )


def test_reasoning_content_is_kept_on_assistant_turn():
    result = _run(_response(reasoning_content="thinking..."))
    assistant = result["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["reasoning_content"] == "thinking..."
    assert assistant["tool_calls"][0]["id"] == "call_1"


def test_thinking_blocks_are_kept_on_assistant_turn():
    blocks = [{"type": "thinking", "thinking": "hmm", "signature": "sig"}]
    result = _run(_response(thinking_blocks=blocks))
    assert result["messages"][-1]["thinking_blocks"] == blocks


def test_no_reasoning_keys_when_absent():
    result = _run(_response())
    assert "reasoning_content" not in result["messages"][-1]
    assert "thinking_blocks" not in result["messages"][-1]


def test_reasoning_not_stored_on_plain_answer():
    result = _run(_response(with_tool_call=False, reasoning_content="thinking..."))
    assistant = result["messages"][-1]
    assert assistant["content"] == "4"
    assert "reasoning_content" not in assistant
