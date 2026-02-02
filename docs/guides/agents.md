# Building Agents with NPCs

NPCs are the core agent abstraction in npcpy. Each NPC wraps a language model with a persona, directive, and optional tools, giving you a reusable agent that can reason, call functions, and participate in teams. This guide covers creating NPCs programmatically, using tools, and the different ways to invoke LLM responses through agents.

## Creating an NPC

An NPC requires a `name`, `primary_directive`, `model`, and `provider`. No files or databases are needed for basic usage.

```python
from npcpy.npc_compiler import NPC

simon = NPC(
    name='Simon Bolivar',
    primary_directive='Liberate South America from the Spanish Royalists.',
    model='gemma3:4b',
    provider='ollama'
)
```

The `primary_directive` becomes the system message that shapes how the NPC responds. The `model` and `provider` determine which LLM backs the agent.

## Getting a Response from an NPC

Call `get_llm_response()` on the NPC instance. It returns a dict with the full response structure.

```python
response = simon.get_llm_response(
    "What is the most important territory to retain in the Andes mountains?"
)
print(response['response'])
```

```
The most important territory to retain in the Andes mountains is **Cuzco**.
It's the heart of the Inca Empire, a crucial logistical hub, and holds immense
symbolic value for our liberation efforts. Control of Cuzco is paramount.
```

The returned dict has these keys:

```python
print(response.keys())
# dict_keys(['response', 'raw_response', 'messages', 'tool_calls', 'tool_results'])
```

- `response` -- the text reply from the LLM
- `raw_response` -- the unprocessed provider response object
- `messages` -- the full conversation history including system, user, and assistant messages
- `tool_calls` -- list of tool calls the LLM made (empty if no tools)
- `tool_results` -- list of tool execution results (empty if no tools)

## Assigning Tools to NPCs

Pass a list of Python functions to the `tools` parameter. npcpy automatically generates the tool schema from type hints and docstrings.

```python
import os
import json
from npcpy.npc_compiler import NPC

def list_files(directory: str = ".") -> list:
    """List all files in a directory."""
    return os.listdir(directory)

def read_file(filepath: str) -> str:
    """Read and return the contents of a file."""
    with open(filepath, 'r') as f:
        return f.read()

assistant = NPC(
    name='File Assistant',
    primary_directive='You are a helpful assistant who can list and read files.',
    model='llama3.2',
    provider='ollama',
    tools=[list_files, read_file],
)

response = assistant.get_llm_response(
    "List the files in the current directory.",
    auto_process_tool_calls=True,  # this is the default for NPCs
)
print(response.keys())
```

```
dict_keys(['response', 'raw_response', 'messages', 'tool_calls', 'tool_results'])
```

## Tool Results Structure

Each entry in `tool_results` is a dict with `tool_call_id`, `arguments`, and `result`.

```python
from npcpy.npc_sysenv import render_markdown

for tool_call in response['tool_results']:
    render_markdown(tool_call['tool_call_id'])
    for arg in tool_call['arguments']:
        render_markdown('- ' + arg + ': ' + str(tool_call['arguments'][arg]))
    render_markdown('- Results:' + str(tool_call['result']))
```

```
 - directory: .
 - Results:['research_pipeline.jinx', 'mkdocs.yml', 'LICENSE', 'npcpy',
   'Makefile', 'README.md', 'tests', 'docs', 'setup.py', '.gitignore',
   '.env', 'examples', 'npcpy.egg-info', '.github', '.git']
```

When tools are invoked, `response['response']` may be `None` because the LLM chose to call a tool instead of generating text. The structured data in `tool_calls` and `tool_results` gives you everything needed to decide what happens next.

## Passing an NPC to get_llm_response

You can also pass an NPC to the standalone `get_llm_response` function instead of calling the NPC method directly. This is useful for testing or when you want to control the call site independently from the agent definition.

```python
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response

simon = NPC(
    name='Simon Bolivar',
    primary_directive='Liberate South America from the Spanish Royalists.',
    model='gemma3:4b',
    provider='ollama'
)

response = get_llm_response(
    "Who was the mythological Chilean bird that guides lucky visitors to gold?",
    npc=simon
)
print(response['response'])
```

When you pass `npc=simon`, the function uses the NPC's model, provider, and system message automatically.

## Tool Calling Without NPCs

You can use tools directly with `get_llm_response` by generating schemas with `auto_tools` and passing them in.

```python
from npcpy.llm_funcs import get_llm_response
from npcpy.tools import auto_tools
import os

def create_file(filename: str, content: str) -> str:
    """Create a new file with content."""
    with open(filename, 'w') as f:
        f.write(content)
    return f"Created {filename}"

def count_files(directory: str = ".") -> int:
    """Count files in a directory."""
    return len([f for f in os.listdir(directory) if os.path.isfile(f)])

def get_file_size(filename: str) -> str:
    """Get the size of a file."""
    size = os.path.getsize(filename)
    return f"{filename} is {size} bytes"

# auto_tools returns (tools_schema, tool_map)
tools_schema, tool_map = auto_tools([create_file, count_files, get_file_size])

response = get_llm_response(
    "Create a file called 'hello.txt' with the content 'Hello World!', "
    "then tell me how many files are in the current directory.",
    model='qwen3:latest',
    provider='ollama',
    tools=tools_schema,
    tool_map=tool_map
)
```

## Response Structure When Tools Are Used

When the LLM decides to call tools, the response structure changes:

```python
# response['response'] is None -- the LLM called a tool instead of replying
# response['tool_calls'] contains the calls the LLM made
# response['tool_results'] contains the execution results

for i, (call, result) in enumerate(
    zip(response.get('tool_calls', []), response.get('tool_results', []))
):
    func_name = call['function']['name']
    args = call['function']['arguments']
    print(f"{i+1}. {func_name}({args}) -> {result}")
```

```
1. create_file({"filename": "hello.txt", "content": "Hello World!"}) -> Created hello.txt
2. count_files({"directory": "."}) -> 42
```

This design lets you build pipelines where tool results feed into subsequent reasoning steps, or where you decide programmatically how to handle each tool invocation.

## Skills as Agent Knowledge

Skills are jinxs that serve knowledge content instead of executing code. They use the same jinx pipeline and are assigned to agents the same way — through the `jinxs:` list in `.npc` files or via the team's `jinxs/` directory.

```yaml
# reviewer.npc
name: reviewer
primary_directive: |
  You review code and provide actionable feedback.
model: llama3.2
provider: ollama
jinxs:
  - lib/core/sh
  - lib/core/python
  - skills/code-review
  - skills/debugging
```

The agent sees `code-review` and `debugging` in its tool catalog alongside `sh` and `python`. When it encounters a review task, it calls `code-review(section=correctness)` to get methodology, then uses `sh` or `python` to inspect the actual code. No separate configuration is needed — skills flow through the existing jinx assignment mechanism.

See the [Skills guide](skills.md) for details on authoring skills in both SKILL.md and .jinx formats.
