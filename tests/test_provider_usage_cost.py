import json
from types import SimpleNamespace

import npcpy.gen.response as resp_mod
from npcpy.gen.response import (
    _attach_raw_usage_snoop,
    _extract_raw_sse_usage,
    get_litellm_response,
)


RAW_USAGE_LINE = 'data: {"id":"gen-1","usage":{"prompt_tokens":9485,"completion_tokens":98,"total_tokens":9583,"cost":0.00123}}'


def test_extract_usage_from_str_and_bytes_lines():
    assert _extract_raw_sse_usage(RAW_USAGE_LINE) == {
        "prompt_tokens": 9485,
        "completion_tokens": 98,
        "total_tokens": 9583,
        "cost": 0.00123,
    }
    assert _extract_raw_sse_usage(RAW_USAGE_LINE.encode())["cost"] == 0.00123


def test_extract_ignores_non_usage_lines():
    assert _extract_raw_sse_usage('data: {"id":"gen-1","choices":[]}') is None
    assert _extract_raw_sse_usage("data: [DONE]") is None
    assert _extract_raw_sse_usage("") is None
    assert _extract_raw_sse_usage("not json at all") is None
    assert _extract_raw_sse_usage('data: {"usage": "oops"}') is None
    assert _extract_raw_sse_usage(None) is None
    assert _extract_raw_sse_usage({"usage": {}}) is None


def test_snoop_nested_chain_captures_usage_and_passes_through():
    lines = ['data: {"id":"gen-1","choices":[]}', RAW_USAGE_LINE, "data: [DONE]"]
    raw = iter(lines)
    inner = SimpleNamespace(response_iterator=raw, streaming_response=raw)
    outer = SimpleNamespace(completion_stream=inner)
    sink = {}
    _attach_raw_usage_snoop(outer, sink)
    assert list(outer.completion_stream.response_iterator) == lines
    assert sink == {
        "prompt_tokens": 9485,
        "completion_tokens": 98,
        "total_tokens": 9583,
        "cost": 0.00123,
    }


def test_snoop_direct_chain_and_missing_attrs_safe():
    lines = [RAW_USAGE_LINE]
    direct = SimpleNamespace(completion_stream=iter(lines))
    sink = {}
    _attach_raw_usage_snoop(direct, sink)
    assert list(direct.completion_stream) == lines
    assert sink["cost"] == 0.00123
    _attach_raw_usage_snoop(SimpleNamespace(), {})
    _attach_raw_usage_snoop(SimpleNamespace(completion_stream="data: complete"), {})
    _attach_raw_usage_snoop(None, {})


class _FakeStream:
    def __init__(self, raw_lines, chunks):
        raw = iter(raw_lines)
        self.completion_stream = SimpleNamespace(response_iterator=raw, streaming_response=raw)
        self._chunks = chunks

    def __iter__(self):
        inner = self.completion_stream.response_iterator
        for chunk in self._chunks:
            try:
                next(inner)
            except StopIteration:
                pass
            yield chunk
        for _ in inner:
            pass


def _parsed_chunks():
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    return [
        SimpleNamespace(id="gen-e2e", usage=None, choices=[]),
        SimpleNamespace(id="gen-e2e", usage=usage, choices=[]),
    ]


def _stream_kwargs():
    return dict(
        provider="openrouter",
        model="meta/muse-spark",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        include_usage=True,
        api_key="sk-test",
    )


def test_stream_yields_provider_cost_from_raw_usage(monkeypatch):
    raw = ['data: {"id":"gen-e2e","choices":[]}', RAW_USAGE_LINE]
    monkeypatch.setattr(resp_mod, "completion", lambda **kw: _FakeStream(raw, _parsed_chunks()))
    result = get_litellm_response(**_stream_kwargs())
    items = list(result["response"])
    assert items[-1] == {"type": "usage", "input_tokens": 9485, "output_tokens": 98, "cost": 0.00123}


def test_stream_without_provider_cost_falls_through(monkeypatch):
    raw = ['data: {"id":"gen-e2e","choices":[]}', "data: [DONE]"]
    monkeypatch.setattr(resp_mod, "completion", lambda **kw: _FakeStream(raw, _parsed_chunks()))
    result = get_litellm_response(**_stream_kwargs())
    items = list(result["response"])
    assert len(items) == 2
    assert all(not isinstance(i, dict) for i in items)


def test_stream_close_skips_trailing_usage(monkeypatch):
    raw = [RAW_USAGE_LINE]
    monkeypatch.setattr(resp_mod, "completion", lambda **kw: _FakeStream(raw, _parsed_chunks()))
    result = get_litellm_response(**_stream_kwargs())
    stream = result["response"]
    next(stream)
    stream.close()
