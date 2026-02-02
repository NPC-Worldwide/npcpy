# Multi-Agent Teams

Teams let you combine multiple NPCs under a coordinator (the `forenpc`) that routes requests, delegates tasks, and synthesizes results. This guide covers creating teams, orchestrating requests, and understanding the result structure.

## Team Concept

A Team is a collection of NPCs with a designated `forenpc` (short for "foreman NPC") that acts as the coordinator. When you send a request to a team via `orchestrate()`, the forenpc decides how to handle it -- answering directly, delegating to a team member, or passing work between agents.

## Creating a Team

Provide a list of NPCs and a forenpc. The forenpc can be one of the team members or a separate NPC.

```python
from npcpy.npc_compiler import NPC, Team

ggm = NPC(
    name='Gabriel Garcia Marquez',
    primary_directive='You are Gabriel Garcia Marquez, master of magical realism. '
                      'Research, analyze, and write with poetic flair.',
    model='gemma3:4b',
    provider='ollama',
)

isabel = NPC(
    name='Isabel Allende',
    primary_directive='You are Isabel Allende, weaving stories with emotion '
                      'and history. Analyze texts and provide insight.',
    model='llama3.2',
    provider='ollama',
)

borges = NPC(
    name='Jorge Luis Borges',
    primary_directive='You are Borges, philosopher of labyrinths and libraries. '
                      'Synthesize findings and create literary puzzles.',
    model='qwen3:latest',
    provider='ollama',
)

lit_team = Team(
    npcs=[ggm, isabel],
    forenpc=borges
)
```

When the Team is initialized:

- Each NPC's `.team` attribute is set to the team instance
- The forenpc is determined (either the provided NPC or resolved by name)
- Team-level jinxs are rendered and distributed to NPCs
- A jinx tool catalog is built for the team

## Orchestration

Call `team.orchestrate(prompt)` to send a request through the team. The forenpc receives the request first and decides how to handle it.

```python
result = lit_team.orchestrate(
    "Which book are your team members most proud of? Ask them please."
)
```

### Result Structure

`orchestrate()` returns a dict with two top-level keys: `output` and `result`.

```python
print(result.keys())
# dict_keys(['output', 'result'])
```

- `result['output']` -- the final text output from the orchestration
- `result['result']` -- the full result dict from the underlying `check_llm_command` call, containing `messages`, `output`, and `usage` data

```python
# The text answer
print(result['output'])

# The detailed result from the coordinating NPC
print(result['result'].keys())
# Typical keys: 'messages', 'output', 'usage'
```

### Full Example

```python
from npcpy.npc_compiler import NPC, Team

ggm = NPC(
    name='Gabriel Garcia Marquez',
    primary_directive='You are Gabriel Garcia Marquez, master of magical realism.',
    model='gemma3:4b',
    provider='ollama',
)

isabel = NPC(
    name='Isabel Allende',
    primary_directive='You are Isabel Allende, weaving stories with emotion and history.',
    model='llama3.2',
    provider='ollama',
)

borges = NPC(
    name='Jorge Luis Borges',
    primary_directive='You are Borges, philosopher of labyrinths and libraries. '
                      'Synthesize findings and create literary puzzles.',
    model='qwen3:latest',
    provider='ollama',
)

lit_team = Team(npcs=[ggm, isabel], forenpc=borges)

result = lit_team.orchestrate(
    "Research the topic of magical realism and summarize the findings."
)

# Access the text output
print(result['output'])

# Access the underlying result dict
inner = result['result']
print(inner.get('output', '')[:200])
```

## Agent Passing

During orchestration, the forenpc can delegate work to other team members. This happens automatically when the forenpc mentions another NPC by name (or with `@name` syntax) in its response.

The orchestration logic:

1. The forenpc receives the request and generates a response
2. If the response mentions a team member's name, the request is delegated to that NPC
3. The delegated NPC processes the request with its own model and directive
4. The result is returned through the orchestration pipeline

```python
# The forenpc (Borges) might respond: "Let me ask Gabriel Garcia Marquez about this..."
# This triggers automatic delegation to the ggm NPC
result = lit_team.orchestrate(
    "Tell me about the use of time in One Hundred Years of Solitude."
)
print(result['output'])
```

Agent passing also works explicitly through the `handle_agent_pass` mechanism when NPCs are used with the `check_llm_command` pipeline, where the forenpc can choose a `pass_to_npc` action with a target agent.

## Team-Based Jinx Usage

Teams can have jinxs (workflow templates) that are shared across all member NPCs. Pass jinx objects when creating the team.

```python
from npcpy.npc_compiler import NPC, Team, Jinx

literary_research_jinx = Jinx(jinx_data={
    "jinx_name": "literary_research",
    "description": "Research a literary topic and summarize findings",
    "inputs": ["topic"],
    "steps": [
        {
            "name": "gather_info",
            "engine": "natural",
            "code": "Research the topic: {{ topic }}. "
                    "Summarize the main themes and historical context."
        },
        {
            "name": "final_summary",
            "engine": "natural",
            "code": "Based on the research in {{ gather_info }}, "
                    "write a concise, creative summary."
        }
    ]
})

lit_team = Team(
    npcs=[ggm, isabel],
    forenpc=borges,
    jinxs=[literary_research_jinx]
)

result = lit_team.orchestrate(
    "Research the topic of magical realism, analyze it, and summarize the findings."
)
print(result['output'])
```

Team-level jinxs are distributed to all NPCs during initialization, so any team member can execute them when the forenpc delegates work.

## Creating Teams from Directories

For production setups, you can define teams as a directory structure with `.npc` files and a `team.ctx` context file:

```python
team = Team(team_path='~/.npcsh/npc_team')
```

The directory should contain:

- `.npc` files for each agent
- A `jinxs/` subdirectory with `.jinx` workflow files and `skills/` subfolder
- An optional `team.ctx` YAML file defining the forenpc and team context

The `team.ctx` file can also specify `SKILLS_DIRECTORY` to load skills from an external directory:

```yaml
model: llama3.2
provider: ollama
forenpc: lead-dev
SKILLS_DIRECTORY: ~/shared-skills
```

Skills loaded this way are merged into the team's `jinxs_dict` and distributed to all NPCs, just like jinxs in the `jinxs/` directory. See the [Skills guide](skills.md) for details.

## Team Parameters

The `Team` constructor accepts:

| Parameter | Type | Description |
|-----------|------|-------------|
| `team_path` | `str` | Path to team directory |
| `npcs` | `List[NPC]` | List of NPC objects |
| `forenpc` | `NPC` or `str` | Coordinator NPC or name |
| `jinxs` | `List[Jinx]` | Team-level workflow templates |
| `db_conn` | connection | Database connection |
| `model` | `str` | Default model for the team |
| `provider` | `str` | Default provider for the team |
