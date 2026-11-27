from npcpy.gen.response import process_tool_calls


def test_unknown_tool_call_gets_error_result():
    response = {
        "messages": [{"role": "user", "content": "add 1 and 2"}],
        "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}},
            {"id": "call_2", "type": "function",
             "function": {"name": "sum_numbers", "arguments": "{}"}},
        ],
    }

    result = process_tool_calls(
        response, {"add": lambda a, b: a + b}, "model", "openai", response["messages"]
    )

    tool_messages = [m for m in result["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_1", "call_2"]
    assert "Tool 'sum_numbers' not found" in tool_messages[1]["content"]
    assert [r["tool_name"] for r in result["tool_results"]] == ["add", "sum_numbers"]
    assert "not found" in result["tool_results"][1]["result"]
