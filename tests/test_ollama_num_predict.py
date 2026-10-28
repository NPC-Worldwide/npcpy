"""max_tokens / max_completion_tokens must reach Ollama as options["num_predict"]."""
import pytest

from npcpy.gen import response as response_module


class _FakeClient:
    calls = []

    def __init__(self, **kwargs):
        pass

    def chat(self, **kwargs):
        type(self).calls.append(kwargs)
        return {"message": {"role": "assistant", "content": "ok"}}


@pytest.fixture
def fake_ollama(monkeypatch):
    _FakeClient.calls = []

    class _FakeOllama:
        Client = _FakeClient

    monkeypatch.setattr(response_module, "ollama", _FakeOllama)
    monkeypatch.setattr(response_module, "HAS_OLLAMA", True)
    return _FakeClient


def _options(fake, **kwargs):
    response_module.get_ollama_response(
        prompt="hi", model="llama3.2:latest", messages=[], **kwargs
    )
    return fake.calls[-1]["options"]


@pytest.mark.parametrize("key", ["max_tokens", "max_completion_tokens"])
def test_token_limit_is_sent_as_num_predict(fake_ollama, key):
    options = _options(fake_ollama, **{key: 64})

    assert options["num_predict"] == 64
    assert key not in options


def test_other_options_are_unchanged(fake_ollama):
    options = _options(fake_ollama, temperature=0.2, top_k=5)

    assert options["temperature"] == 0.2
    assert options["top_k"] == 5
    assert "num_predict" not in options


def test_unset_token_limit_adds_no_num_predict(fake_ollama):
    assert "num_predict" not in _options(fake_ollama, max_tokens=None)
