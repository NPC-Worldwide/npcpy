"""Streaming usage must be summed across every tool-loop turn, not overwritten."""
from types import SimpleNamespace as NS

import pytest

import npcpy.llm_funcs as llm_funcs
import npcpy.streaming as streaming
from npcpy.streaming import StreamConfig, create_tool_agent_stream

TOOLS = [{"type": "function", "function": {"name": "noop", "description": "n",
                                           "parameters": {"type": "object", "properties": {}}}}]
EXECUTORS = {"noop": {"type": "python", "func": lambda **kw: "ok"}}


def _openai_turns():
    tool_chunk = NS(choices=[NS(delta=NS(content=None, tool_calls=[
        NS(index=0, id="t1", type="function", function=NS(name="noop", arguments="{}"))]))],
        usage=None)
    usage1 = NS(choices=[], usage=NS(prompt_tokens=100, completion_tokens=10))
    text = NS(choices=[NS(delta=NS(content="done", tool_calls=None))], usage=None)
    usage2 = NS(choices=[], usage=NS(prompt_tokens=150, completion_tokens=20))
    return "openai", [[tool_chunk, usage1], [text, usage2]]


def _ollama_turns():
    tool_chunk = NS(message={"content": "", "tool_calls": [
        {"id": "t1", "function": {"name": "noop", "arguments": {}}}]},
        prompt_eval_count=100, eval_count=10)
    text = NS(message={"content": "done"}, prompt_eval_count=150, eval_count=20)
    return "ollama", [[tool_chunk], [text]]


@pytest.mark.parametrize("make_turns", [_openai_turns, _ollama_turns])
def test_create_tool_agent_stream_sums_usage_over_turns(monkeypatch, make_turns):
    provider, turns = make_turns()
    calls = iter(turns)
    monkeypatch.setattr(streaming, "get_llm_response",
                        lambda *a, **k: {"response": iter(next(calls)), "usage": {}})
    cfg = StreamConfig(model="m", provider=provider, messages=[], commandstr="hi")
    events = list(create_tool_agent_stream(cfg, TOOLS, EXECUTORS))
    usage = [e for e in events if e.type == "usage"][-1].data
    assert (usage["input_tokens"], usage["output_tokens"]) == (250, 30)


@pytest.mark.parametrize("make_turns", [_openai_turns, _ollama_turns])
def test_agent_run_stream_sums_usage_over_turns(monkeypatch, make_turns):
    from npcpy.npc_compiler import Agent
    provider, turns = make_turns()
    calls = iter(turns)
    tail = [NS(choices=[NS(delta=NS(content="end", tool_calls=None))], usage=None)]
    monkeypatch.setattr(llm_funcs, "get_llm_response",
                        lambda *a, **k: {"response": iter(next(calls, tail)), "usage": {}})
    monkeypatch.setattr(streaming, "resolve_npc_tools", lambda *a, **k: (TOOLS, EXECUTORS))
    agent = Agent(name="a", primary_directive="d", model="m", provider=provider)
    events = list(agent.run_stream("hi"))
    usage = [e for e in events if isinstance(e, dict) and e.get("type") == "usage"][-1]
    assert (usage["input_tokens"], usage["output_tokens"]) == (250, 30)


def test_create_tool_agent_stream_counts_usage_of_non_stream_response(monkeypatch):
    monkeypatch.setattr(streaming, "get_llm_response",
                        lambda *a, **k: {"response": "hello",
                                         "usage": {"input_tokens": 7, "output_tokens": 3}})
    cfg = StreamConfig(model="m", provider="openai", messages=[], commandstr="hi")
    events = list(create_tool_agent_stream(cfg, TOOLS, EXECUTORS))
    usage = [e for e in events if e.type == "usage"][-1].data
    assert (usage["input_tokens"], usage["output_tokens"]) == (7, 3)
