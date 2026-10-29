import json
from types import SimpleNamespace
from unittest.mock import patch

from npcpy.gen.response import get_litellm_response


def _usage(prompt_tokens, completion_tokens):
    return SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def _tool_call_response(usage):
    tool_call = SimpleNamespace(
        id="call_1",
        type="function",
        function=SimpleNamespace(name="add", arguments=json.dumps({"a": 1, "b": 2})),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _text_response(text, usage):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        },
    },
}]


def _run(second_usage):
    responses = [_tool_call_response(_usage(100, 20)), _text_response("3", second_usage)]
    with patch("npcpy.gen.response.completion", side_effect=responses) as mocked:
        result = get_litellm_response(
            prompt="add 1 and 2",
            model="gpt-test",
            provider="openai",
            tools=TOOLS,
            tool_map={"add": lambda a, b: a + b},
            api_key="sk-test",
            auto_process_tool_calls=True,
        )
    assert mocked.call_count == 2
    return result


def test_tool_turn_usage_includes_the_summary_completion():
    result = _run(_usage(50, 10))

    assert result["response"] == "3"
    assert result["usage"] == {"input_tokens": 150, "output_tokens": 30}


def test_tool_turn_usage_tolerates_a_summary_completion_without_usage():
    result = _run(None)

    assert result["usage"] == {"input_tokens": 100, "output_tokens": 20}
