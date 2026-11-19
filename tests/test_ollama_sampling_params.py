"""seed / frequency_penalty / presence_penalty must reach Ollama as options."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def response_module():
    # Load this provider without the package's unrelated optional integrations.
    path = Path(__file__).parents[1] / "npcpy" / "gen" / "response.py"
    spec = importlib.util.spec_from_file_location("ollama_sampling_test", path)
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
def options(response_module, monkeypatch):
    calls = []

    class _Client:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            calls.append(kwargs)
            return {"message": {"role": "assistant", "content": "ok"}}

    monkeypatch.setattr(response_module, "ollama", SimpleNamespace(Client=_Client))
    monkeypatch.setattr(response_module, "HAS_OLLAMA", True)

    def run(**kwargs):
        response_module.get_ollama_response(
            prompt="hi", model="llama3.2:latest", messages=[], **kwargs
        )
        return calls[-1]["options"]

    return run


@pytest.mark.parametrize("key,value", [
    ("seed", 42),
    ("frequency_penalty", 0.5),
    ("presence_penalty", 0.3),
])
def test_sampling_param_is_forwarded(options, key, value):
    assert options(**{key: value})[key] == value


def test_seed_zero_is_forwarded(options):
    assert options(seed=0)["seed"] == 0


def test_unset_params_are_not_added(options):
    opts = options(temperature=0.2)
    for key in ("seed", "frequency_penalty", "presence_penalty"):
        assert key not in opts
