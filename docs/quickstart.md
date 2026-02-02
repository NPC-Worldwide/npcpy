# Quickstart

Get up and running with `npcpy` in five steps.

## 1. Install npcpy

```bash
pip install npcpy
```

For local model support, install [Ollama](https://ollama.com) and pull a model:

```bash
ollama pull llama3.2
```

For API providers (OpenAI, Anthropic, Gemini, etc.), create a `.env` file with your keys:

```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
```

## 2. Get your first LLM response

```python
from npcpy.llm_funcs import get_llm_response

response = get_llm_response(
    "What is the capital of France?",
    model='llama3.2',
    provider='ollama'
)
print(response['response'])
```

```
The capital of France is Paris.
```

Every response is a dictionary with at minimum a `response` key containing the text output.

## 3. Create your first NPC agent

An NPC is an agent with a persona, model assignment, and optional tools:

```python
from npcpy.npc_compiler import NPC

historian = NPC(
    name='Historian',
    primary_directive='You are a knowledgeable historian who gives concise answers.',
    model='llama3.2',
    provider='ollama'
)

response = historian.get_llm_response("When did the Roman Empire fall?")
print(response['response'])
```

## 4. Give your agent tools

Pass Python functions as tools and the agent will call them when relevant:

```python
import os
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
    primary_directive='You help users explore files.',
    model='llama3.2',
    provider='ollama',
    tools=[list_files, read_file],
)

response = assistant.get_llm_response("What files are in the current directory?")
print(response['response'])
```

Tool calls and results are available in `response['tool_calls']` and `response['tool_results']`.

## 5. Next steps

Now that you have the basics, explore the guides:

- **[Working with LLMs](guides/llm-responses.md)** - Streaming, JSON output, messages, attachments
- **[Building Agents](guides/agents.md)** - NPC files, directives, tool assignment
- **[Multi-Agent Teams](guides/teams.md)** - Team orchestration with a coordinator
- **[Jinx Workflows](guides/jinx-workflows.md)** - Jinja Execution templates for prompt pipelines
- **[NPCArray](guides/npc-array.md)** - Vectorized operations over model populations
- **[Image, Audio & Video](guides/image-audio-video.md)** - Generation across providers
- **[AirLLM](guides/airllm.md)** - Run 70B+ models on consumer hardware
- **[Knowledge Graphs](guides/knowledge-graphs.md)** - Build and evolve knowledge graphs from text
- **[Fine-Tuning & Evolution](guides/fine-tuning.md)** - SFT, RL, genetic algorithms
- **[Serving & Deployment](guides/serving.md)** - Flask server for production teams
- **[ML Functions](guides/ml-funcs.md)** - Scikit-learn grid search, ensemble prediction
