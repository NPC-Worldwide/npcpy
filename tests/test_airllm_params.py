"""AirLLM (CUDA branch) must forward sampling parameters to generate()."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def response_module():
    path = Path(__file__).parents[1] / "npcpy" / "gen" / "response.py"
    spec = importlib.util.spec_from_file_location("airllm_response_test", path)
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
    module._AIRLLM_MODEL_CACHE.clear()
    return module


class _Ids:
    def cuda(self):
        return self

    def __getitem__(self, i):
        return [1, 2]


class _FakeModel:
    def __init__(self):
        self.calls = []
        self.tokenizer = Mock(side_effect=lambda *a, **k: {"input_ids": _Ids()})
        self.tokenizer.apply_chat_template = Mock(return_value="hi")
        self.tokenizer.decode = Mock(return_value="ok")

    def generate(self, ids, **kwargs):
        self.calls.append(kwargs)
        return [[1, 2, 3]]


def _run(module, **kwargs):
    model = _FakeModel()
    airllm = SimpleNamespace(
        AutoModel=SimpleNamespace(from_pretrained=Mock(return_value=model))
    )
    with patch.dict(sys.modules, {"airllm": airllm}), \
            patch("platform.system", return_value="Linux"):
        module.get_airllm_response("hello", model="fake/model", **kwargs)
    assert len(model.calls) == 1
    return model.calls[0]


def test_sampling_params_reach_generate(response_module):
    call = _run(response_module, temperature=0.2, top_p=0.9, max_tokens=7)
    assert call["max_new_tokens"] == 7
    assert call["do_sample"] is True
    assert call["temperature"] == 0.2
    assert call["top_p"] == 0.9


def test_zero_temperature_is_greedy(response_module):
    call = _run(response_module, temperature=0, top_p=0.9)
    assert call["do_sample"] is False
    assert "temperature" not in call
    assert "top_p" not in call
