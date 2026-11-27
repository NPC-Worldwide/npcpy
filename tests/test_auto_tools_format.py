from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from npcpy.gen.response import get_litellm_response, get_ollama_response


def add(a, b):
    return a + b


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

CASES = [
    ("json", '```json\n{"answer": 3}\n```'),
    ("yaml", "```yaml\nanswer: 3\n```"),
]


@pytest.mark.parametrize("fmt,text", CASES)
def test_ollama_auto_tools_parses_format_when_no_tool_is_called(fmt, text):
    pytest.importorskip("ollama")
    client = MagicMock()
    client.chat.return_value = {"message": {"role": "assistant", "content": text}}
    with patch("ollama.Client", return_value=client), patch("ollama.chat", client.chat):
        result = get_ollama_response(
            prompt="what is 1 + 2?",
            model="llama3.2",
            tools=TOOLS,
            tool_map={"add": add},
            format=fmt,
            auto_process_tool_calls=True,
        )

    assert client.chat.call_count == 1
    assert result["response"] == {"answer": 3}


@pytest.mark.parametrize("fmt,text", CASES)
def test_litellm_auto_tools_parses_format_when_no_tool_is_called(fmt, text):
    message = SimpleNamespace(content=text, tool_calls=None)
    response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
    with patch("npcpy.gen.response.completion", return_value=response):
        result = get_litellm_response(
            prompt="what is 1 + 2?",
            model="gpt-test",
            provider="openai",
            tools=TOOLS,
            tool_map={"add": add},
            format=fmt,
            api_key="sk-test",
            auto_process_tool_calls=True,
        )

    assert result["response"] == {"answer": 3}
    assert result["messages"][-1] == {"role": "assistant", "content": text}


class Answer(BaseModel):
    answer: int


@pytest.mark.parametrize("fmt,text", [
    ("json", 'Here you go:\n```json\n{"answer": 3}\n```'),
    ("yaml", "```\nanswer: 3\n```"),
    (Answer, '```json\n{"answer": 3}\n```'),
])
def test_ollama_parses_format_without_auto_tools(fmt, text):
    pytest.importorskip("ollama")
    client = MagicMock()
    client.chat.return_value = {"message": {"role": "assistant", "content": text}}
    with patch("ollama.Client", return_value=client), patch("ollama.chat", client.chat):
        result = get_ollama_response(prompt="what is 1 + 2?", model="llama3.2", format=fmt)

    assert result["response"] == {"answer": 3}
    assert "error" not in result
