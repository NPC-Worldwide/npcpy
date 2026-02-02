# Serving & Deployment

npcpy ships with a built-in Flask server that exposes your NPC teams as REST API endpoints. You can register NPCs and teams at startup, then interact with them over HTTP from any language or framework. The server handles streaming responses, conversation history, jinx execution, model management, and an OpenAI-compatible completions endpoint.

## Basic Server Setup

The simplest way to start the server is to call `start_flask_server` with your NPCs and teams.

```python
from npcpy.serve import start_flask_server
from npcpy.npc_compiler import NPC, Team

# Create NPCs
researcher = NPC(
    name="researcher",
    primary_directive="You are a research assistant. Find and summarize information.",
    model="gemma3:4b",
    provider="ollama",
)

writer = NPC(
    name="writer",
    primary_directive="You are a technical writer. Produce clear documentation.",
    model="gemma3:4b",
    provider="ollama",
)

# Build a team
team = Team(
    npcs=[researcher, writer],
    team_name="docs_team",
)

# Register and start
start_flask_server(
    port=5337,
    cors_origins=["http://localhost:3000"],
    debug=True,
    teams={"docs_team": team},
    npcs={"researcher": researcher, "writer": writer},
)
```

The server binds to `0.0.0.0` on the given port and accepts requests immediately.

### start_flask_server Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `port` | `5337` | HTTP port to listen on |
| `cors_origins` | `None` | List of allowed CORS origins |
| `debug` | `False` | Enable Flask debug mode |
| `teams` | `None` | Dict mapping team names to `Team` objects |
| `npcs` | `None` | Dict mapping NPC names to `NPC` objects |
| `db_path` | `""` | Path to SQLite database for conversation history |
| `user_npc_directory` | `None` | Directory containing `.npc` and `.jinx` files |

## NPC Loading Priority

When a request specifies an NPC name, the server resolves it in this order:

1. **Registered team NPCs** -- checks all teams passed via the `teams` parameter for an NPC with a matching name.
2. **Globally registered NPCs** -- checks the `npcs` dict passed at startup.
3. **Database / file fallback** -- loads the NPC from disk using the `npc_source` field (`"global"` checks `~/.npcsh/npc_team/`, `"project"` checks the request's `currentPath`).

This means programmatically registered NPCs always take precedence over file-based definitions.

## REST API Endpoints

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/stream` | Stream a response from an NPC (SSE) |
| `POST` | `/api/jinx/execute` | Execute a jinx with arguments |
| `GET` | `/api/models` | List available models |
| `GET` | `/api/npc_team_global` | List globally registered NPCs |
| `GET` | `/api/jinxs/global` | List globally available jinxs |
| `GET` | `/api/conversations` | List conversations |
| `GET` | `/api/conversation/<id>/messages` | Get messages for a conversation |
| `GET` | `/api/health` | Health check (returns `{"status": "ok"}`) |
| `POST` | `/v1/chat/completions` | OpenAI-compatible completions endpoint |

### ML and Fine-Tuning Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/ml/train` | Train a sklearn model |
| `POST` | `/api/ml/predict` | Run prediction on a trained model |
| `POST` | `/api/finetune_instruction` | Start an SFT fine-tuning job |
| `POST` | `/api/finetune_diffusers` | Start a diffusion fine-tuning job |
| `GET` | `/api/finetuned_models` | List fine-tuned models |
| `POST` | `/api/genetic/create_population` | Create a genetic evolution population |
| `POST` | `/api/genetic/evolve` | Run one generation of evolution |

### Model Management Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/ollama/models` | List Ollama models |
| `POST` | `/api/ollama/pull` | Pull an Ollama model |
| `POST` | `/api/ollama/delete` | Delete an Ollama model |
| `GET` | `/api/models/hf/search` | Search HuggingFace models |
| `POST` | `/api/models/hf/download` | Download a HuggingFace model |

## curl Examples

### Health Check

```bash
curl http://localhost:5337/api/health
```

```json
{"status": "ok", "error": null}
```

### Stream a Response

```bash
curl -X POST http://localhost:5337/api/stream \
  -H "Content-Type: application/json" \
  -d '{
    "commandstr": "Explain quantum computing in two sentences.",
    "conversationId": "conv-001",
    "model": "gemma3:4b",
    "provider": "ollama"
  }'
```

### Stream with a Specific NPC

```bash
curl -X POST http://localhost:5337/api/stream \
  -H "Content-Type: application/json" \
  -d '{
    "commandstr": "Write a summary of the latest research on fusion energy.",
    "conversationId": "conv-002",
    "npc": "researcher",
    "model": "gemma3:4b",
    "provider": "ollama"
  }'
```

### Execute a Jinx

```bash
curl -X POST http://localhost:5337/api/jinx/execute \
  -H "Content-Type: application/json" \
  -d '{
    "jinxName": "summarize",
    "jinxArgs": [{"name": "topic", "value": "machine learning"}],
    "conversationId": "conv-003",
    "model": "gemma3:4b",
    "provider": "ollama"
  }'
```

### List Available Models

```bash
curl http://localhost:5337/api/models?currentPath=/path/to/project
```

### List Global NPCs

```bash
curl http://localhost:5337/api/npc_team_global
```

### List Global Jinxs

```bash
curl http://localhost:5337/api/jinxs/global
```

### OpenAI-Compatible Completions

The server exposes an OpenAI-compatible endpoint so you can use standard OpenAI clients:

```bash
curl -X POST http://localhost:5337/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3:4b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "stream": false
  }'
```

## JavaScript / React Integration

Use the standard `fetch` API to connect a frontend to the NPC server. The stream endpoint returns server-sent events.

```javascript
// Stream a response from an NPC
async function streamNPCResponse(prompt, npcName, conversationId) {
  const response = await fetch("http://localhost:5337/api/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      commandstr: prompt,
      conversationId: conversationId,
      npc: npcName,
      model: "gemma3:4b",
      provider: "ollama",
    }),
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullText = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value, { stream: true });
    fullText += chunk;
    // Update your UI with the streamed text
    console.log("Chunk:", chunk);
  }

  return fullText;
}

// List available models
async function getModels() {
  const response = await fetch("http://localhost:5337/api/models");
  const data = await response.json();
  return data.models;
}

// List NPCs
async function getNPCs() {
  const response = await fetch("http://localhost:5337/api/npc_team_global");
  const data = await response.json();
  return data.npcs;
}
```

### React Hook Example

```javascript
import { useState, useCallback } from "react";

function useNPCChat(serverUrl = "http://localhost:5337") {
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);

  const sendMessage = useCallback(async (prompt, npcName, conversationId) => {
    setIsStreaming(true);
    setMessages((prev) => [...prev, { role: "user", content: prompt }]);

    try {
      const response = await fetch(`${serverUrl}/api/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          commandstr: prompt,
          conversationId,
          npc: npcName,
        }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMessage = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        assistantMessage += decoder.decode(value, { stream: true });
        setMessages((prev) => {
          const updated = [...prev];
          const lastIdx = updated.length - 1;
          if (updated[lastIdx]?.role === "assistant") {
            updated[lastIdx] = { role: "assistant", content: assistantMessage };
          } else {
            updated.push({ role: "assistant", content: assistantMessage });
          }
          return updated;
        });
      }
    } finally {
      setIsStreaming(false);
    }
  }, [serverUrl]);

  return { messages, sendMessage, isStreaming };
}
```

## Python Client Example

```python
import requests

SERVER = "http://localhost:5337"

# Health check
health = requests.get(f"{SERVER}/api/health").json()
print(health)  # {"status": "ok", "error": null}

# List models
models = requests.get(f"{SERVER}/api/models").json()
for m in models["models"]:
    print(f"  {m['display_name']}")

# Send a message and get a streamed response
response = requests.post(
    f"{SERVER}/api/stream",
    json={
        "commandstr": "Explain gradient descent in one paragraph.",
        "conversationId": "py-conv-001",
        "model": "gemma3:4b",
        "provider": "ollama",
    },
    stream=True,
)

for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    print(chunk, end="", flush=True)
print()

# Execute a jinx
result = requests.post(
    f"{SERVER}/api/jinx/execute",
    json={
        "jinxName": "analyze_code",
        "jinxArgs": [{"name": "language", "value": "python"}],
        "conversationId": "py-conv-002",
    },
).json()
print(result)

# Use the OpenAI-compatible endpoint
completion = requests.post(
    f"{SERVER}/v1/chat/completions",
    json={
        "model": "gemma3:4b",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
    },
).json()
print(completion["choices"][0]["message"]["content"])
```

## Team with Tools Example

A common deployment pattern is a multimedia team where each NPC has specialized tools. The server exposes all team capabilities through the same streaming endpoint.

```python
import os
from npcpy.npc_compiler import NPC, Team
from npcpy.serve import start_flask_server
from npcpy.data.web import search_web
from npcpy.data.load import load_file_contents
from npcpy.data.image import capture_screenshot

def web_search(query: str) -> str:
    """Search the web for information."""
    results = search_web(query)
    return str(results)

def read_file(filepath: str) -> str:
    """Read a local file and return its contents."""
    return load_file_contents(filepath)

def screenshot() -> str:
    """Capture a screenshot of the current desktop."""
    path = capture_screenshot()
    return f"Screenshot saved to {path}"

# Research NPC with web access
researcher = NPC(
    name="researcher",
    primary_directive="Search the web and summarize findings.",
    model="gemma3:4b",
    provider="ollama",
    tools=[web_search],
)

# File analyst with local file access
analyst = NPC(
    name="analyst",
    primary_directive="Analyze files and data.",
    model="gemma3:4b",
    provider="ollama",
    tools=[read_file],
)

# Desktop automation NPC
automator = NPC(
    name="automator",
    primary_directive="Automate desktop tasks and capture screenshots.",
    model="gemma3:4b",
    provider="ollama",
    tools=[screenshot],
)

team = Team(
    npcs=[researcher, analyst, automator],
    team_name="multimedia_team",
)

start_flask_server(
    port=5337,
    cors_origins=["http://localhost:3000"],
    teams={"multimedia_team": team},
    npcs={
        "researcher": researcher,
        "analyst": analyst,
        "automator": automator,
    },
)
```

When you send a request to `/api/stream` with `"npc": "researcher"`, the researcher NPC can invoke its `web_search` tool during the response. Tool calls and results are tracked and stored alongside the conversation.

## Complex Workflows with Jinxs

Jinxs are multi-step prompt pipelines defined in YAML. When served through the API, they can chain LLM calls, tool invocations, and template rendering.

First, define a jinx file at `~/.npcsh/npc_team/jinxs/research_report.jinx`:

```yaml
jinx_name: research_report
description: Research a topic and write a structured report.
inputs:
  - name: topic
    description: The topic to research
    default: "artificial intelligence"
  - name: depth
    description: How detailed the report should be
    default: "medium"
steps:
  - name: research
    prompt: |
      Research the following topic thoroughly: {{ topic }}
      Depth level: {{ depth }}
      Provide key findings, statistics, and recent developments.

  - name: outline
    prompt: |
      Based on this research: {{ research }}
      Create a structured outline for a report on {{ topic }}.

  - name: report
    prompt: |
      Using this outline: {{ outline }}
      Write a complete report. Include an introduction, body sections,
      and conclusion.
```

Then execute it through the API:

```bash
curl -X POST http://localhost:5337/api/jinx/execute \
  -H "Content-Type: application/json" \
  -d '{
    "jinxName": "research_report",
    "jinxArgs": [
      {"name": "topic", "value": "quantum computing"},
      {"name": "depth", "value": "detailed"}
    ],
    "conversationId": "jinx-conv-001",
    "model": "gemma3:4b",
    "provider": "ollama"
  }'
```

Each step in the jinx executes sequentially. Output variables from earlier steps (like `{{ research }}` and `{{ outline }}`) are automatically available in later step prompts via Jinja2 templating. The final response includes the output of the last step along with all intermediate results.
