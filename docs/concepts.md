# Core Concepts

This page explains the key abstractions in `npcpy`.

## NPC (Agent)

An **NPC** is an AI agent defined by a persona, a model, and optional tools. NPCs wrap LLM calls with consistent behavior driven by a `primary_directive`.

```python
from npcpy.npc_compiler import NPC

agent = NPC(
    name='Analyst',
    primary_directive='You analyze data and provide insights.',
    model='llama3.2',
    provider='ollama',
    tools=[my_function],  # optional
)
```

NPCs can also be defined in `.npc` files (YAML) and loaded from an `npc_team/` directory.

## Team

A **Team** groups multiple NPCs under a coordinator (`forenpc`) that routes tasks between them. The coordinator decides which NPC should handle each part of a request.

```python
from npcpy.npc_compiler import Team

team = Team(npcs=[npc_a, npc_b], forenpc=coordinator_npc)
result = team.orchestrate("Analyze this dataset and write a report.")
```

The orchestration result contains:

- `result['output']` - the final text output
- `result['result']` - full response dict with messages and usage

## Jinx (Jinja Execution Template)

A **Jinx** is a multi-step workflow defined as a sequence of steps. Each step can use a `python` engine (runs code) or a `natural` engine (sends a prompt to the LLM). Steps can reference outputs from previous steps via Jinja templating.

```yaml
jinx_name: summarizer
inputs: [text]
steps:
  - name: extract_key_points
    engine: natural
    prompt: "Extract 5 key points from: {{ text }}"
  - name: write_summary
    engine: natural
    prompt: "Write a summary from these points: {{ extract_key_points }}"
```

Jinxs are stored as `.jinx` files or created programmatically via the `Jinx` class.

## Providers and Models

`npcpy` supports multiple inference backends through LiteLLM:

| Provider | Example Models | Notes |
|----------|---------------|-------|
| `ollama` | `llama3.2`, `gemma3:4b`, `qwen3:latest` | Local, free |
| `openai` | `gpt-4o`, `gpt-4o-mini` | API key required |
| `anthropic` | `claude-3-5-sonnet-latest` | API key required |
| `gemini` | `gemini-2.5-flash` | API key required |
| `deepseek` | `deepseek-chat`, `deepseek-reasoner` | API key required |
| `airllm` | `Qwen/Qwen2.5-7B-Instruct` | 70B+ on consumer hardware |
| `openai-like` | Any OpenAI-compatible endpoint | Custom servers |

Set the `model` and `provider` on any NPC or pass them directly to `get_llm_response`.

## The Response Dictionary

All LLM calls return a standard dictionary:

```python
{
    'response': str,           # The text output
    'raw_response': object,    # Raw provider response
    'messages': list,          # Conversation history
    'tool_calls': list,        # Tool calls made (if any)
    'tool_results': list,      # Tool call results (if any)
}
```

When `format='json'` is specified, `response` is a parsed Python dict/list instead of a string.

## NPCArray (Vectorized AI)

**NPCArray** provides NumPy-like operations over populations of models. Operations are **lazy** — they build a computation DAG that executes only when `.collect()` is called.

```python
from npcpy.npc_array import NPCArray

models = NPCArray.from_llms(['llama3.2', 'gemma3:1b'])
result = models.infer("What is 2+2?").vote(axis=0).collect()
```

Key operations: `infer`, `predict`, `map`, `filter`, `reduce`, `vote`, `consensus`, `chain`, `evolve`.

See the [NPCArray guide](guides/npc-array.md) for full details.

## Knowledge Graphs

`npcpy` can build and evolve **knowledge graphs** from text using LLM-driven entity and relation extraction. Graphs support incremental updates, sleep/dream evolution cycles, and hybrid search.

See the [Knowledge Graphs guide](guides/knowledge-graphs.md) for full details.

## Lazy Evaluation

Both NPCArray and Jinx workflows use lazy evaluation patterns. NPCArray builds a DAG of operations that executes on `.collect()`. This enables:

- **Optimization**: The engine can batch and parallelize operations
- **Inspection**: Call `.explain()` to see the computation plan before running
- **Composition**: Chain operations without intermediate execution overhead
