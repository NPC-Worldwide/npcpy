from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import npcpy.gen.response as resp_mod


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "params",
    [
        {"seed": 42, "frequency_penalty": 0.5, "presence_penalty": -0.5,
         "logit_bias": {"123": -100}, "n": 2},
        {"seed": 0, "frequency_penalty": 0, "presence_penalty": 0,
         "logit_bias": {}, "n": 1},
    ],
)
def test_generation_params_reach_completion(monkeypatch, stream, params):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello", tool_calls=None))],
        usage=None,
    )
    completion = Mock(return_value=iter(()) if stream else response)
    monkeypatch.setattr(resp_mod, "completion", completion)

    resp_mod.get_litellm_response(
        provider="openai", model="gpt-4o-mini", api_key="sk-test",
        messages=[{"role": "user", "content": "hi"}], stream=stream, **params,
    )

    completion.assert_called_once()
    for key, value in params.items():
        assert completion.call_args.kwargs.get(key) == value
