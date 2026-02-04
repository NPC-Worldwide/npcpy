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

### Agent with persona

```python
from npcpy.npc_compiler import NPC

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
from npcpy.llm_funcs import get_llm_response

response = get_llm_response("Who was the celtic messenger god?", model='qwen3:4b', provider='ollama')
print(response['response'])
```

### Agent with tools

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
response = assistant.get_llm_response("List the files in the current directory.")
print(response['response'])
```

### Multi-agent team orchestration

```python
from npcpy.npc_compiler import NPC, Team

# Create specialist agents
coordinator = NPC(
    name='coordinator',
    primary_directive='''You coordinate a team of specialists.
    Delegate tasks by mentioning @analyst for data questions or @writer for content.
    Synthesize their responses into a final answer.''',
    model='llama3.2',
    provider='ollama'
)

analyst = NPC(
    name='analyst',
    primary_directive='You analyze data and provide insights with specific numbers.',
    model='~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf',  # local GGUF file
    provider='llamacpp'
)

writer = NPC(
    name='writer',
    primary_directive='You write clear, engaging summaries and reports.',
    model='gemini-2.5-flash',
    provider='gemini'
)

# Create team - coordinator (forenpc) automatically delegates via @mentions
team = Team(npcs=[coordinator, analyst, writer], forenpc='coordinator')

# Orchestrate a request - coordinator decides who to involve
result = team.orchestrate("What are the trends in renewable energy adoption?")
print(result['output'])
```

### Team from directory

```python
from npcpy.npc_compiler import Team

# Load team from directory with .npc files and team.ctx
team = Team(team_path='./npc_team')

# Orchestrate through the forenpc (set in team.ctx)
result = team.orchestrate("Analyze the sales data and write a summary")
print(result['output'])
```

### Team directory structure

```
npc_team/
├── team.ctx           # Team configuration
├── coordinator.npc    # Coordinator agent
├── analyst.npc        # Specialist agent
├── writer.npc         # Specialist agent
└── jinxs/             # Optional workflows
    └── research.jinx
```

**team.ctx** - Team configuration:
```yaml
context: |
  A research team that analyzes topics and produces reports.
  The coordinator delegates to specialists as needed.
forenpc: coordinator
model: llama3.2
provider: ollama
mcp_servers:
  - ~/.npcsh/mcp_server.py
```

**coordinator.npc** - Agent definition:
```yaml
name: coordinator
primary_directive: |
  You coordinate research tasks. Delegate to @analyst for data
  analysis and @writer for content creation. Synthesize results.
model: llama3.2
provider: ollama
```

**analyst.npc** - Specialist agent:
```yaml
name: analyst
primary_directive: |
  You analyze data and provide insights with specific numbers and trends.
model: qwen3:8b
provider: ollama
```

### Initialize a team

Installing `npcpy` also installs two command-line tools:
- **`npc`** — CLI for project management and one-off commands
- **`npcsh`** — Interactive shell for chatting with agents and running jinxs

```bash
# Using npc CLI
npc init ./my_project

# Using npcsh (interactive)
npcsh
📁 ~/projects
🤖 npcsh | llama3.2
> /init directory=./my_project
> what files are in the current directory?
```

This creates:
```
my_project/
├── npc_team/
│   ├── forenpc.npc      # Default coordinator
│   ├── jinxs/           # Workflows
│   │   └── skills/      # Knowledge skills
│   ├── tools/           # Custom tools
│   └── triggers/        # Event triggers
├── images/
├── models/
└── mcp_servers/
```

Then add your agents:
```bash
# Add team context
cat > my_project/npc_team/team.ctx << 'EOF'
context: Research and analysis team
forenpc: lead
model: llama3.2
provider: ollama
EOF

# Add agents
cat > my_project/npc_team/lead.npc << 'EOF'
name: lead
primary_directive: |
  You lead the team. Delegate to @researcher for data
  and @writer for content. Synthesize their output.
EOF

cat > my_project/npc_team/researcher.npc << 'EOF'
name: researcher
primary_directive: You research topics and provide detailed findings.
model: gemini-2.5-flash
provider: gemini
EOF

cat > my_project/npc_team/writer.npc << 'EOF'
name: writer
primary_directive: You write clear, engaging content.
model: qwen3:8b
provider: ollama
EOF
```

## Features

- **[Agents (NPCs)](https://npcpy.readthedocs.io/en/latest/guides/agents/)** — Agents with personas, directives, and tool calling
- **[Multi-Agent Teams](https://npcpy.readthedocs.io/en/latest/guides/teams/)** — Team orchestration with a coordinator (forenpc)
- **[Jinx Workflows](https://npcpy.readthedocs.io/en/latest/guides/jinx-workflows/)** — Jinja Execution templates for multi-step prompt pipelines
- **[Skills](https://npcpy.readthedocs.io/en/latest/guides/skills/)** — Knowledge-content jinxs that serve instructional sections to agents on demand
- **[NPCArray](https://npcpy.readthedocs.io/en/latest/guides/npc-array/)** — NumPy-like vectorized operations over model populations
- **[Image, Audio & Video](https://npcpy.readthedocs.io/en/latest/guides/image-audio-video/)** — Generation via Ollama, diffusers, OpenAI, Gemini
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
ollama pull llama3.2
```

**macOS:**
```bash
brew install portaudio ffmpeg pygobject3 ollama
brew services start ollama
ollama pull llama3.2
```

**Windows:** Install [Ollama](https://ollama.com) and [ffmpeg](https://ffmpeg.org), then `ollama pull llama3.2`.

</details>

API keys go in a `.env` file:
```bash
export OPENAI_API_KEY="your_key"
export ANTHROPIC_API_KEY="your_key"
export GEMINI_API_KEY="your_key"
```

## Read the Docs

Full documentation, guides, and API reference at [npcpy.readthedocs.io](https://npcpy.readthedocs.io/en/latest/).

## Inference Capabilities

- Local and cloud LLM providers through LiteLLM (Ollama, OpenAI, Anthropic, Gemini, Deepseek, and more)
- AirLLM for 70B+ parameter models on consumer hardware (MLX on macOS, CUDA with 4-bit compression on Linux)
- Image generation through Ollama (`x/z-image-turbo`, `x/flux2-klein`), Huggingface diffusers, OpenAI (DALL-E, GPT Image), and Gemini

## Links

- **[Incognide](https://github.com/cagostino/incognide)** — GUI for the NPC Toolkit ([download](https://enpisi.com/incognide))
- **[NPC Shell](https://github.com/npc-worldwide/npcsh)** — Command-line shell for interacting with NPCs
- **[Newsletter](https://forms.gle/n1NzQmwjsV4xv1B2A)** — Stay in the loop

## Research

- Quantum-like nature of natural language interpretation: [arxiv](https://arxiv.org/abs/2506.10077), accepted at [QNLP 2025](https://qnlp.ai)
- Simulating hormonal cycles for AI: [arxiv](https://arxiv.org/abs/2508.11829)

Has your research benefited from npcpy? Let us know!

## Support

[Monthly donation](https://buymeacoffee.com/npcworldwide) | [Merch](https://enpisi.com/shop) | Consulting: info@npcworldwi.de

## Contributing

Contributions welcome! Submit issues and pull requests on the [GitHub repository](https://github.com/NPC-Worldwide/npcpy).

## License

MIT License.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=cagostino/npcpy&type=Date)](https://star-history.com/#cagostino/npcpy&Date)
