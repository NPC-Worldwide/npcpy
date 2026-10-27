from types import SimpleNamespace

from npcpy.streaming import _accumulate_tool_call_deltas


def _delta(index=None, id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=function)


def test_indexed_deltas_are_merged_by_index():
    collected = []
    _accumulate_tool_call_deltas(collected, [_delta(0, "call_a", "search", '{"q":')])
    _accumulate_tool_call_deltas(collected, [_delta(0, None, None, '"x"}')])
    _accumulate_tool_call_deltas(collected, [_delta(1, "call_b", "open", "{}")])

    assert [tc["id"] for tc in collected] == ["call_a", "call_b"]
    assert collected[0]["function"] == {"name": "search", "arguments": '{"q":"x"}'}
    assert collected[1]["function"] == {"name": "open", "arguments": "{}"}


def test_parallel_calls_without_index_stay_separate():
    # Some OpenAI-compatible endpoints send index=None on every tool call delta.
    collected = []
    _accumulate_tool_call_deltas(collected, [
        _delta(None, "call_a", "search", '{"q": "a"}'),
        _delta(None, "call_b", "open", '{"url": "b"}'),
    ])

    assert [tc["id"] for tc in collected] == ["call_a", "call_b"]
    assert collected[0]["function"] == {"name": "search", "arguments": '{"q": "a"}'}
    assert collected[1]["function"] == {"name": "open", "arguments": '{"url": "b"}'}


def test_chunked_call_without_index_continues_the_last_call():
    collected = []
    _accumulate_tool_call_deltas(collected, [_delta(None, "call_a", "search", '{"q":')])
    _accumulate_tool_call_deltas(collected, [_delta(None, None, None, ' "x"}')])
    _accumulate_tool_call_deltas(collected, [_delta(None, "call_b", "open", "{}")])

    assert [tc["id"] for tc in collected] == ["call_a", "call_b"]
    assert collected[0]["function"]["arguments"] == '{"q": "x"}'
    assert collected[1]["function"]["arguments"] == "{}"
