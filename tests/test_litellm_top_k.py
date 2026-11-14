from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import npcpy.gen.response as resp_mod


def _call(monkeypatch, provider, model, **kwargs):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello", tool_calls=None))],
        usage=None,
    )
    completion = Mock(return_value=response)
    monkeypatch.setattr(resp_mod, "completion", completion)
    resp_mod.get_litellm_response(
        provider=provider, model=model, api_key="k",
        messages=[{"role": "user", "content": "hi"}], **kwargs,
    )
    completion.assert_called_once()
    return completion.call_args.kwargs


@pytest.mark.parametrize("provider,model", [
    ("anthropic", "claude-sonnet-4-5"),
    ("gemini", "gemini-2.5-flash"),
])
def test_top_k_reaches_completion(monkeypatch, provider, model):
    assert _call(monkeypatch, provider, model, top_k=7).get("top_k") == 7


def test_top_k_not_sent_to_openai(monkeypatch):
    assert "top_k" not in _call(monkeypatch, "openai", "gpt-4o-mini", top_k=7)
