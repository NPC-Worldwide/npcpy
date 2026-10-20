"""Offline request-boundary coverage using the real LiteLLM transport."""

import copy
import json
from types import SimpleNamespace

import httpx
import litellm
import pytest

from npcpy.llm_funcs import get_llm_response, resolve_model_provider
from npcpy.gen.response import get_litellm_response
from npcpy import npc_sysenv


@pytest.fixture
def requests_sent(monkeypatch):
    captured = []

    def send(client, request, **kwargs):
        assert request.method == "POST"
        body = json.loads(request.content)
        captured.append((str(request.url), body))
        if body.get("stream"):
            chunks = [
                {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0,
                 "model": body["model"], "choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]},
                {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 0,
                 "model": body["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
            ]
            content = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, content=content.encode(), headers={"content-type": "text/event-stream"}, request=request)
        if request.url.path.endswith("/messages"):
            response = {
                "id": "msg_test", "type": "message", "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        else:
            response = {
                "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
                "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        return httpx.Response(200, json=response, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(litellm, "model_cost", copy.deepcopy(litellm.model_cost))
    monkeypatch.delenv("MINIMAX_API_URL", raising=False)
    return captured


@pytest.mark.parametrize("host", ["api.minimax.io", "api.minimaxi.com"])
@pytest.mark.parametrize("protocol", ["v1", "anthropic"])
@pytest.mark.parametrize("model", [None, "MiniMax-M3", "MiniMax-M2.7", "saved-custom-model"])
def test_direct_serialized_requests(requests_sent, monkeypatch, host, protocol, model):
    monkeypatch.setenv("MINIMAX_API_URL", f"https://{host}/{protocol}")
    result = get_llm_response("Hello", model=model, provider="minimax", max_tokens=32)
    assert result["response"] == "ok"
    url, body = requests_sent.pop()
    suffix = "/v1/messages" if protocol == "anthropic" else "/chat/completions"
    assert url == f"https://{host}/{protocol}{suffix}"
    assert body["model"] == (model or "MiniMax-M3.1-Flash")
    assert body["messages"][-1]["content"] in ("Hello", [{"type": "text", "text": "Hello"}])
    assert "service_tier" not in body
    assert "reasoning_split" not in body


@pytest.mark.parametrize("protocol", ["v1", "anthropic"])
@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_flash_reasoning_at_http_boundary(requests_sent, protocol, effort):
    get_llm_response(
        "Hello", provider="minimax", api_url=f"https://api.minimax.io/{protocol}",
        thinking={"type": "adaptive"}, reasoning_effort=effort, max_tokens=32,
    )
    _, body = requests_sent.pop()
    assert body["thinking"] == {"type": "adaptive"}
    if protocol == "anthropic":
        assert body["output_config"] == {"effort": effort}
        assert "reasoning_effort" not in body
    else:
        assert body["reasoning_effort"] == effort


@pytest.mark.parametrize("kwargs", [
    {"thinking": {"type": "disabled"}},
    {"thinking": {"type": "enabled", "budget_tokens": 1024}},
    {"reasoning_effort": "none"},
    {"reasoning_effort": "minimal"},
])
def test_flash_rejects_unsupported_reasoning(requests_sent, kwargs):
    with pytest.raises(ValueError, match="MiniMax-M3.1-Flash"):
        get_llm_response("Hello", provider="minimax", **kwargs)
    assert requests_sent == []


def test_anthropic_output_config(requests_sent):
    get_llm_response("Hello", provider="minimax", api_url="https://api.minimax.io/anthropic",
                     output_config={"effort": "xhigh"}, max_tokens=32)
    assert requests_sent[-1][1]["output_config"] == {"effort": "xhigh"}


@pytest.mark.parametrize("protocol", ["v1", "anthropic"])
def test_m3_disabled_thinking(requests_sent, protocol):
    get_llm_response("Hello", model="MiniMax-M3", provider="minimax",
                     api_url=f"https://api.minimax.io/{protocol}",
                     thinking={"type": "disabled"}, max_tokens=32)
    assert requests_sent[-1][1]["thinking"] == {"type": "disabled"}


@pytest.mark.parametrize("protocol", ["v1", "anthropic"])
def test_low_level_entrypoint_default(requests_sent, protocol):
    result = get_litellm_response("Hello", provider="minimax",
                                  api_url=f"https://api.minimax.io/{protocol}", max_tokens=32)
    assert result["response"] == "ok"
    assert requests_sent[-1][1]["model"] == "MiniMax-M3.1-Flash"


def test_saved_model_and_custom_endpoint(requests_sent):
    npc = SimpleNamespace(model="MiniMax-M2.7", provider="minimax",
                          api_url="https://custom.example/v1", api_key="saved-key")
    assert resolve_model_provider(npc=npc) == (
        "MiniMax-M2.7", "openai", "https://custom.example/v1", "saved-key"
    )
    get_llm_response("Hello", model=npc.model, provider=npc.provider,
                     api_url=npc.api_url, api_key=npc.api_key)
    assert requests_sent[-1][0] == "https://custom.example/v1/chat/completions"
    assert requests_sent[-1][1]["model"] == npc.model


@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "MiniMax-M2.5", "MiniMax-Text-01"])
def test_model_only_inference(requests_sent, model):
    get_llm_response("Hello", model=model)
    assert requests_sent[-1][0] == "https://api.minimax.io/v1/chat/completions"
    assert requests_sent[-1][1]["model"] == model


def test_flash_streaming(requests_sent):
    result = get_llm_response("Hello", provider="minimax", stream=True)
    chunks = list(result["response"])
    assert chunks[0].choices[0].delta.content == "ok"
    assert requests_sent[-1][1]["stream"] is True
    assert requests_sent[-1][1]["model"] == "MiniMax-M3.1-Flash"


def test_multiple_samples_keep_resolved_endpoint(requests_sent, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_URL", "https://api.minimaxi.com/anthropic")
    result = get_llm_response("Hello", provider="minimax", n_samples=2, max_tokens=32)
    assert len(result["runs"]) == 2
    assert all(url == "https://api.minimaxi.com/anthropic/v1/messages" for url, _ in requests_sent)


@pytest.mark.parametrize("model", ["minimax/minimax-m3.1-flash:free", "minimax/minimax-m3", "minimax/minimax-m2.7"])
def test_gateway_preserves_namespaced_model(requests_sent, model):
    get_llm_response("Hello", model=model, provider="openrouter", api_key="test-key")
    url, body = requests_sent.pop()
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert body["model"] == model


def test_non_minimax_and_multimodal_history_unchanged(requests_sent):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]}]
    original = copy.deepcopy(messages)
    get_llm_response("", model="gpt-4o", provider="openai", api_key="test-key", messages=messages)
    assert requests_sent[-1][1]["model"] == "gpt-4o"
    assert requests_sent[-1][1]["messages"] == original
    assert messages == original


def test_real_pinned_catalog(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(npc_sysenv, "check_internet_connection", lambda: False)
    monkeypatch.setattr(npc_sysenv.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    models = npc_sysenv.get_locally_available_models(str(tmp_path))
    direct = [model for model, provider in models.items() if provider == "minimax"]
    assert direct[:2] == ["MiniMax-M3.1-Flash", "MiniMax-M3"]
    assert not any(model.startswith(("MiniMax-M2", "MiniMax-Text")) for model in direct)
    assert models["minimax/minimax-m3.1-flash:free"] == "openrouter"
    assert models["minimax/minimax-m3"] == "openrouter"


def test_catalog_filters_upstream_legacy_models(monkeypatch, tmp_path):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    monkeypatch.setattr(litellm, "together_ai_models", {
        "MiniMaxAI/MiniMax-M2.7", "MiniMaxAI/MiniMax-M3", "other/keep-together",
    })
    monkeypatch.setattr(npc_sysenv, "check_internet_connection", lambda: False)
    monkeypatch.setattr(litellm.utils, "refresh_model_cost_map", lambda: None, raising=False)
    monkeypatch.setattr(litellm, "minimax_models", {"MiniMax-M2.7", "MiniMax-Text-01"}, raising=False)
    monkeypatch.setattr(litellm, "openrouter_models", {
        "minimax/minimax-m2.7", "openrouter/minimax/minimax-01", "other/keep-model",
    })
    monkeypatch.setattr(litellm, "model_cost", {
        "minimax/MiniMax-M2.5": {"litellm_provider": "minimax"},
        "openrouter/minimax/minimax-m2": {"litellm_provider": "openrouter"},
        "speech-02-turbo": {"litellm_provider": "minimax", "mode": "audio_speech"},
    })
    # Model discovery also probes optional local HTTP servers.
    monkeypatch.setattr(npc_sysenv.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    models = npc_sysenv.get_locally_available_models(str(tmp_path))
    assert [model for model, provider in models.items() if provider == "minimax"] == [
        "MiniMax-M3.1-Flash", "MiniMax-M3", "speech-02-turbo",
    ]
    assert models["MiniMaxAI/MiniMax-M3"] == "together"
    assert "MiniMaxAI/MiniMax-M2.7" not in models
    assert models["keep-together"] == "together"
    assert [model for model in models if model.startswith("minimax/")] == [
        "minimax/minimax-m3.1-flash:free", "minimax/minimax-m3",
    ]
    assert models["keep-model"] == "openrouter"
    assert npc_sysenv.lookup_provider("MiniMax-M2.7") == "minimax"
