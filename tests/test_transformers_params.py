"""Check generation parameters reach transformers' generate() without a real model."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture
def response_module():
    path = Path(__file__).parents[1] / "npcpy" / "gen" / "response.py"
    spec = importlib.util.spec_from_file_location("transformers_response_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "npcpy.data.image": SimpleNamespace(compress_image=Mock()),
        "npcpy.npc_sysenv": SimpleNamespace(
            get_system_message=Mock(), lookup_provider=Mock(),
            render_markdown=Mock(), MINIMAX_MODELS=(),
        ),
    }):
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def generate(monkeypatch):
    torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=Mock(), AutoModelForCausalLM=Mock(),
    ))
    ids = MagicMock()
    ids.shape = (1, 1)
    outputs = MagicMock()
    outputs.__getitem__.return_value = []
    generate = Mock(return_value=outputs)
    model = SimpleNamespace(
        generate=generate,
        parameters=lambda: iter([SimpleNamespace(device="cpu")]),
    )
    tokenizer = MagicMock(eos_token_id=0)
    tokenizer.return_value = {"input_ids": ids}
    tokenizer.decode.return_value = "ok"
    generate.model, generate.tokenizer = model, tokenizer
    return generate


def test_defaults_unchanged(response_module, generate):
    response_module.get_transformers_response(
        prompt="hi", model=generate.model, tokenizer=generate.tokenizer,
    )
    kw = generate.call_args.kwargs
    assert kw["max_new_tokens"] == 256 and kw["temperature"] == 0.7
    assert "top_p" not in kw
    assert "top_k" not in kw


def test_user_params_reach_generate(response_module, generate):
    response_module.get_transformers_response(
        prompt="hi", model=generate.model, tokenizer=generate.tokenizer,
        max_tokens=17, temperature=0.2, top_p=0.5, top_k=7,
    )
    kw = generate.call_args.kwargs
    assert kw["max_new_tokens"] == 17
    assert kw["temperature"] == 0.2
    assert kw["top_p"] == 0.5
    assert kw["top_k"] == 7


def test_zero_temperature_is_greedy(response_module, generate):
    response_module.get_transformers_response(
        prompt="hi", model=generate.model, tokenizer=generate.tokenizer,
        temperature=0, top_p=0.5, top_k=7,
    )
    kw = generate.call_args.kwargs
    assert kw["do_sample"] is False
    assert "temperature" not in kw and "top_p" not in kw and "top_k" not in kw
