"""Exercise llama.cpp request construction without loading a local model."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from pydantic import BaseModel
import pytest


@pytest.fixture
def response_module():
    # Load this provider without the package's unrelated optional integrations.
    path = Path(__file__).parents[1] / "npcpy" / "gen" / "response.py"
    spec = importlib.util.spec_from_file_location("llamacpp_response_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "npcpy.data.image": SimpleNamespace(compress_image=Mock()),
        "npcpy.npc_sysenv": SimpleNamespace(
            get_system_message=Mock(), lookup_provider=Mock(),
            render_markdown=Mock(), MINIMAX_MODELS=(),
        ),
        "torch": None,
    }):
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def completion(monkeypatch):
    completion = Mock(return_value={"choices": []})
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(
        Llama=Mock(return_value=SimpleNamespace(create_chat_completion=completion)),
    ))
    return completion


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("choice", ["none", {"type": "function", "function": {"name": "weather"}}])
def test_tools_and_choice_reach_llamacpp(response_module, completion, stream, choice):
    tools = [{"type": "function", "function": {
        "name": "weather", "parameters": {"type": "object", "properties": {}},
    }}]
    result = response_module.get_llamacpp_response(
        prompt="Weather?", model="test.gguf", tools=tools,
        tool_choice=choice, stream=stream,
    )
    assert "error" not in result
    assert completion.call_args.kwargs.get("tools") == tools
    assert completion.call_args.kwargs.get("tool_choice") == choice


class Answer(BaseModel):
    answer: str


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("output_format,expected", [
    ("json", {"type": "json_object"}),
    (Answer, {"type": "json_object", "schema": Answer.model_json_schema()}),
])
def test_format_reaches_llamacpp(response_module, completion, stream, output_format, expected):
    result = response_module.get_llamacpp_response(
        prompt="Answer", model="test.gguf", format=output_format, stream=stream,
    )
    assert "error" not in result
    assert completion.call_args.kwargs.get("response_format") == expected


def test_default_request_omits_optional_params(response_module, completion):
    response_module.get_llamacpp_response(prompt="Hello", model="test.gguf")
    assert completion.call_args.kwargs == {
        "messages": [{"role": "user", "content": "Hello"}], "stream": False,
    }
