from ollama import ChatResponse

from npcpy.streaming import accumulate_tool_call_deltas, parse_stream_chunk


def chunk(*calls):
    return ChatResponse(
        model="qwen3",
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls],
        },
    )


def collect(chunks):
    collected = []
    for c in chunks:
        _, _, deltas = parse_stream_chunk(c, "qwen3", "ollama")
        accumulate_tool_call_deltas(collected, deltas)
    return collected


def test_ollama_tool_call_keeps_name_and_arguments():
    collected = collect([chunk(("get_weather", {"city": "Paris"}))])

    assert [tc["function"] for tc in collected] == [
        {"name": "get_weather", "arguments": '{"city": "Paris"}'}
    ]


def test_ollama_tool_calls_from_separate_chunks_stay_separate():
    collected = collect([
        chunk(("get_weather", {"city": "Paris"})),
        chunk(("get_weather", {"city": "Rome"})),
    ])

    assert [tc["function"]["arguments"] for tc in collected] == [
        '{"city": "Paris"}',
        '{"city": "Rome"}',
    ]
