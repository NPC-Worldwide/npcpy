<p align="center">
  <a href="https://npcpy.readthedocs.io/">
  <img src="https://raw.githubusercontent.com/cagostino/npcpy/main/npcpy/npc-python.png" alt="npc-python logo" width=250></a>
</p>

# npcpy

`npcpy` is a flexible agent framework for building AI applications and conducting research with LLMs. It supports local and cloud providers, multi-agent teams, tool calling, image/audio/video generation, knowledge graphs, fine-tuning, and more.

```bash
pip install npcpy
```

## Quick Examples

### Create and use personas

```python
from npcpy import NPC

simon = NPC(
    name='Simon Bolivar',
    primary_directive='Liberate South America from the Spanish Royalists.',
    model='gemma3:4b',
    provider='ollama'
)
response = simon.get_llm_response("What is the most important territory to retain in the Andes?")
print(response['response'])

```

### Direct LLM call

```python
from npcpy import get_llm_response

response = get_llm_response("Who was the celtic messenger god?", model='qwen3:4b', provider='ollama')
print(response['response'])
# or use ollama's cloud models

test = get_llm_response('who is john wick', model='minimax-m2.7:cloud', provider='ollama',)

print(test['response'])
```

### Agent with tools

```python
from npcpy import Agent, ToolAgent, CodingAgent

# Agent — comes with default tools (sh, python, edit_file, web_search, etc.)
agent = Agent(name='ops', model='qwen3.5:2b', provider='ollama')
print(agent.run("Find all Python files over 500 lines in this repo and list them"))

# ToolAgent — add your own tools alongside defaults
import subprocess

def run_tests(test_path: str = "tests/") -> str:
    """Run pytest on the given path and return results."""
    result = subprocess.run(["python3", "-m", "pytest", test_path, "-v", "--tb=short"],
                            capture_output=True, text=True, timeout=120)
    return result.stdout + result.stderr

def git_diff(branch: str = "main") -> str:
    """Show the git diff against a branch."""
    result = subprocess.run(["git", "diff", branch, "--stat"], capture_output=True, text=True)
    return result.stdout

reviewer = ToolAgent(
    name='code_reviewer',
    primary_directive='You review code changes, run tests, and report issues.',
    tools=[run_tests, git_diff],
    model='qwen3.5:2b', provider='ollama'
)
print(reviewer.run("Run the tests and summarize any failures"))

# CodingAgent — auto-executes code blocks from LLM responses
coder = CodingAgent(name='coder', language='python', model='qwen3.5:2b', provider='ollama')
print(coder.run("Write a script that finds duplicate files by hash in the current directory"))
```

### Streaming

```python
from npcpy import get_llm_response

response = get_llm_response("Explain quantum entanglement.", model='qwen3.5:2b', provider='ollama', stream=True)
for chunk in response['response']:
    print(chunk.get('message', {}).get('content', ''), end='', flush=True)
```

### JSON output

Include the expected JSON structure in your prompt. With `format='json'`, the response is auto-parsed — `response['response']` is already a dict or list.

```python
from npcpy import get_llm_response

response = get_llm_response(
    '''List 3 planets from the sun.
    Return JSON: {"planets": [{"name": "planet name", "distance_au": 0.0, "num_moons": 0}]}''',
    model='qwen3.5:2b', provider='ollama',
    format='json'
)
for planet in response['response']['planets']:
    print(f"{planet['name']}: {planet['distance_au']} AU, {planet['num_moons']} moons")

response = get_llm_response(
    '''Analyze this review: 'The battery life is amazing but the screen is too dim.'
    Return JSON: {"tone": "positive/negative/mixed", "key_phrases": ["phrase1", "phrase2"], "confidence": 0.0}''',
    model='qwen3.5:2b', provider='ollama',
    format='json'
)
result = response['response']
print(result['tone'], result['key_phrases'])
```

### Pydantic structured output

Pass a Pydantic model and the JSON schema is sent to the LLM directly.

```python
from npcpy import get_llm_response
from pydantic import BaseModel
from typing import List

class Planet(BaseModel):
    name: str
    distance_au: float
    num_moons: int

class SolarSystem(BaseModel):
    planets: List[Planet]

response = get_llm_response(
    "List the first 4 planets from the sun.",
    model='qwen3.5:2b', provider='ollama',
    format=SolarSystem
)
for p in response['response']['planets']:
    print(f"{p['name']}: {p['distance_au']} AU, {p['num_moons']} moons")
```

### Image, audio, and video generation

```python
from npcpy.llm_funcs import gen_image, gen_video
from npcpy.gen.audio_gen import text_to_speech

# Image — OpenAI, Gemini, Ollama, or diffusers
images = gen_image("A sunset over the mountains", model='gpt-image-1', provider='openai')
images[0].save("sunset.png")

# Audio — OpenAI, Gemini, ElevenLabs, Kokoro, gTTS
audio_bytes = text_to_speech("Hello from npcpy!", engine="openai", voice="alloy")
with open("hello.wav", "wb") as f:
    f.write(audio_bytes)

# Video — Gemini Veo
result = gen_video("A cat riding a skateboard", model='veo-3.1-fast-generate-preview', provider='gemini')
print(result['output'])
```

### Multi-agent team

```python
from npcpy import NPC, Team

team = Team(team_path='./npc_team')
result = team.orchestrate("Analyze the latest sales data and draft a report")
print(result['output'])
```

Or define a team in code:

```python
from npcpy import NPC, Team

coordinator = NPC(name='lead', primary_directive='Coordinate the team. Delegate to @analyst and @writer.')
analyst = NPC(name='analyst', primary_directive='Analyze data. Provide numbers and trends.', model='gemini-2.5-flash', provider='gemini')
writer = NPC(name='writer', primary_directive='Write clear reports from analysis.', model='qwen3:8b', provider='ollama')

team = Team(npcs=[coordinator, analyst, writer], forenpc='lead')
result = team.orchestrate("What are the trends in renewable energy adoption?")
print(result['output'])
```

### Team from files

**team.ctx:**
```yaml
context: |
  Research team for analyzing scientific literature.
  The lead delegates to specialists as needed.
forenpc: lead
model: qwen3.5:2b
provider: ollama
mcp_servers:
  - path: ~/.npcsh/mcp_server.py
```

**lead.npc:**
```yaml
#!/usr/bin/env npc
name: lead
primary_directive: |
  You lead the research team. Delegate literature searches to @searcher,
  data analysis to @analyst. Synthesize their findings into a coherent summary.
jinxes:
  - sh
  - python
  - delegate
  - web_search
```

**searcher.npc:**
```yaml
#!/usr/bin/env npc
name: searcher
primary_directive: |
  You search for scientific papers and extract key findings.
  Use web_search and load_file to find and read papers.
model: gemini-2.5-flash
provider: gemini
jinxes:
  - web_search
  - load_file
  - sh
```

```
my_project/
├── npc_team/
│   ├── team.ctx
│   ├── lead.npc
│   ├── searcher.npc
│   ├── analyst.npc
│   ├── jinxes/
│   │   └── skills/
│   └── models/
├── agents.md             # Optional: define agents in markdown
└── agents/               # Optional: one .md file per agent
    └── translator.md
```

`.npc` and `.jinx` files are directly executable:
```bash
./npc_team/lead.npc "summarize the latest arxiv papers on transformers"
./npc_team/jinxes/lib/sh.jinx bash_command="echo hello"
```

### MCP server integration

Add MCP servers to your team for external tool access:

**team.ctx:**
```yaml
forenpc: assistant
mcp_servers:
  - path: ./tools/db_server.py
  - path: ./tools/api_server.py
```

**db_server.py:**
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Database Tools")

@mcp.tool()
def query_orders(customer_id: str, limit: int = 10) -> str:
    """Query recent orders for a customer."""
    # Your database logic here
    return f"Found {limit} orders for customer {customer_id}"

@mcp.tool()
def search_products(query: str) -> str:
    """Search the product catalog."""
    return f"Products matching: {query}"

if __name__ == "__main__":
    mcp.run()
```

The team's NPCs automatically get access to MCP tools alongside their jinxes.

### Agent definitions in markdown

**agents.md** — multiple agents in one file:
```markdown
## summarizer
You summarize long documents into concise bullet points.
Focus on key findings, methodology, and conclusions.

## fact_checker
You verify claims against reliable sources and flag inaccuracies.
Always cite your sources.
```

**agents/translator.md** — one file per agent with optional frontmatter:
```markdown
---
model: gemini-2.5-flash
provider: gemini
---
You translate content between languages while preserving tone and idiom.
```

### Skills

Skills are knowledge-content jinxes that provide instructional sections to agents on demand.

**npc_team/jinxes/skills/code-review/SKILL.md:**
```markdown
---
name: code-review
description: Use when reviewing code for quality, security, and best practices.
---
# Code Review Skill

## checklist
- Check for security vulnerabilities (SQL injection, XSS, etc.)
- Verify error handling and edge cases
- Review naming conventions and code clarity

## security
Focus on OWASP top 10 vulnerabilities...
```

Reference in your NPC:
```yaml
jinxes:
  - skills/code-review
```

### CLI tools

```bash
# The NPC shell — the recommended way to use NPC teams
npcsh                        # Interactive shell with agents, tools, and jinxes

# Scaffold a new team
npc-init

# Launch AI coding tools as an NPC from your team
npc-claude --npc corca       # Claude Code
npc-codex --npc analyst      # Codex
npc-gemini                   # Gemini CLI (interactive picker)
npc-opencode / npc-aider / npc-amp

# Register MCP server + hooks for deeper integration
npc-plugin claude
```

## Features

- **[Agents (NPCs)](https://npcpy.readthedocs.io/en/latest/guides/agents/)** — Agents with personas, directives, and tool calling. Subclasses: `Agent` (default tools), `ToolAgent` (custom tools + MCP), `CodingAgent` (auto-execute code blocks)
- **[Multi-Agent Teams](https://npcpy.readthedocs.io/en/latest/guides/teams/)** — Team orchestration with a coordinator (forenpc)
- **[Jinx Workflows](https://npcpy.readthedocs.io/en/latest/guides/jinx-workflows/)** — Jinja Execution templates for multi-step prompt pipelines
- **[Skills](https://npcpy.readthedocs.io/en/latest/guides/skills/)** — Knowledge-content jinxes that serve instructional sections to agents on demand
- **[NPCArray](https://npcpy.readthedocs.io/en/latest/guides/npc-array/)** — NumPy-like vectorized operations over model populations
- **[Image, Audio & Video](https://npcpy.readthedocs.io/en/latest/guides/image-audio-video/)** — Generation via Ollama, diffusers, OpenAI, Gemini, ElevenLabs
- **[Knowledge Graphs](https://npcpy.readthedocs.io/en/latest/guides/knowledge-graphs/)** — Build and evolve knowledge graphs from text
- **[Fine-Tuning & Evolution](https://npcpy.readthedocs.io/en/latest/guides/fine-tuning/)** — SFT, RL, diffusion, genetic algorithms
- **[Serving](https://npcpy.readthedocs.io/en/latest/guides/serving/)** — Flask server for deploying teams via REST API
- **[ML Functions](https://npcpy.readthedocs.io/en/latest/guides/ml-funcs/)** — Scikit-learn grid search, ensemble prediction, PyTorch training
- **[Streaming & JSON](https://npcpy.readthedocs.io/en/latest/guides/llm-responses/)** — Streaming responses, structured JSON output, message history

## Providers

Works with all major LLM providers through LiteLLM: `ollama`, `openai`, `anthropic`, `gemini`, `deepseek`, `airllm`, `openai-like`, and more.

## Installation

```bash
pip install npcpy              # base
pip install npcpy[lite]        # + API provider libraries
pip install npcpy[local]       # + ollama, diffusers, transformers, airllm
pip install npcpy[yap]         # + TTS/STT
pip install npcpy[all]         # everything
```

<details><summary>System dependencies</summary>

**Linux:**
```bash
sudo apt-get install espeak portaudio19-dev python3-pyaudio ffmpeg libcairo2-dev libgirepository1.0-dev
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.5:2b
```

**macOS:**
```bash
brew install portaudio ffmpeg pygobject3 ollama
brew services start ollama
ollama pull qwen3.5:2b
```

**Windows:** Install [Ollama](https://ollama.com) and [ffmpeg](https://ffmpeg.org), then `ollama pull qwen3.5:2b`.

</details>

API keys go in a `.env` file:
```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
```

## Read the Docs

Full documentation, guides, and API reference at [npcpy.readthedocs.io](https://npcpy.readthedocs.io/en/latest/).

## Links

- **[Incognide](https://github.com/npc-worldwide/incognide)** — Desktop environment with AI chat, browser, file viewers, code editor, terminal, knowledge graphs, team management, and more ([download](https://enpisi.com/incognide))
- **[NPC Shell](https://github.com/npc-worldwide/npcsh)** — Command-line shell for interacting with NPCs
- **[Newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)** — Stay in the loop

## Research

- A Quantum Semantic Framework for natural language processing: [arxiv](https://arxiv.org/abs/2506.10077), accepted at [QNLP 2025](https://qnlp.ai)
- Simulating hormonal cycles for AI: [arxiv](https://arxiv.org/abs/2508.11829)
- TinyTim: A Family of Language Models for Divergent Generation [arxiv](https://arxiv.org/abs/2508.11607)
- The production of meaning in the processing of natural language: [arxiv](https://arxiv.org/abs/2603.20381)
- ALARA for Agents: Least-Privilege Context Engineering Through Portable Composable Multi-Agent Teams: [arxiv](https://arxiv.org/abs/2603.20381)

Has your research benefited from npcpy? Let us know!

## Support

[Monthly donation](https://buymeacoffee.com/npcworldwide) | [Merch](https://enpisi.com/shop) | Consulting: info@npcworldwi.de

## Contributing

Contributions welcome! Submit issues and pull requests on the [GitHub repository](https://github.com/NPC-Worldwide/npcpy).

## License

MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cagostino/npcpy&type=Date)](https://star-history.com/#cagostino/npcpy&Date)
