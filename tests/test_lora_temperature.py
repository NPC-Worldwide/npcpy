"""get_lora_response must not pass temperature=0 together with do_sample=True."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest


@pytest.fixture
def response_module():
    path = Path(__file__).parents[1] / "npcpy" / "gen" / "response.py"
    spec = importlib.util.spec_from_file_location("lora_response_test", path)
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


def _run(response_module, tmp_path, **kwargs):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "base/model"})
    )
    torch = MagicMock()
    torch.cuda.is_available.return_value = False
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.return_value = {"input_ids": MagicMock()}
    tokenizer.decode.return_value = "ok"
    model = MagicMock()
    model.generate.return_value = [MagicMock()]
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=MagicMock())),
    )
    peft = SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=Mock(return_value=model)))
    with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "peft": peft}):
        out = response_module.get_lora_response(
            prompt="hi", model=str(tmp_path), messages=[], **kwargs
        )
    return out, model.generate.call_args.kwargs


def test_temperature_zero_uses_greedy_decoding(response_module, tmp_path):
    out, gen_kwargs = _run(response_module, tmp_path, temperature=0)
    assert gen_kwargs["do_sample"] is False
    assert "temperature" not in gen_kwargs


def test_positive_temperature_still_samples(response_module, tmp_path):
    out, gen_kwargs = _run(response_module, tmp_path, temperature=0.4)
    assert gen_kwargs["do_sample"] is True
    assert gen_kwargs["temperature"] == 0.4


def test_top_p_and_top_k_reach_generate(response_module, tmp_path):
    out, gen_kwargs = _run(response_module, tmp_path, top_p=0.5, top_k=20)
    assert gen_kwargs["top_p"] == 0.5
    assert gen_kwargs["top_k"] == 20


def test_greedy_decoding_omits_top_p_and_top_k(response_module, tmp_path):
    out, gen_kwargs = _run(
        response_module, tmp_path, temperature=0, top_p=0.5, top_k=20
    )
    assert "top_p" not in gen_kwargs
    assert "top_k" not in gen_kwargs
