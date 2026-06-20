---
name: knowledge_sememolution_skill
description: "Skill for population-based Knowledge Graph evolution via Sememolution.\
  \ Use this when the user wants creative cross-domain synthesis, speculative reasoning,\
  \ or when a single KG search might be too narrow.\nSememolution maintains a population\
  \ of KG \"individuals\". Each individual has its own graph (different facts, concepts,\
  \ links) and its own genome controlling how it searches and evolves.\nCore genome\
  \ parameters: - `lambda_depth` \u2014 Poisson rate for search traversal depth -\
  \ `lambda_breadth` \u2014 Poisson rate for search breadth per step - `sleep_ops`\
  \ \u2014 which refinement ops to apply during sleep - `dream_probability` \u2014\
  \ chance of speculative synthesis per cycle\nWorkflow: 1. Create a population: `SememolutionPopulation(model,\
  \ provider, population_size=100, sample_size=10)` 2. Initialize: `pop.initialize()`\
  \ 3. Assimilate text: `pop.assimilate_text(chunk)` \u2014 each individual absorbs\
  \ it differently 4. Sleep cycle: `pop.sleep_cycle()` \u2014 each individual prunes/deepens\
  \ independently 5. Query and rank: `pop.query_and_rank(question)` \u2014 sample\
  \ individuals, each searches\n   its own graph with Poisson-sampled depth/breadth,\
  \ generates a response,\n   and responses are ranked. Winners get fitness bumps.\n\
  6. Evolve: `pop.evolve_generation()` \u2014 tournament selection, crossover, mutation.\n\
  When to use this: - The user asks open-ended \"what if\" or \"how might X relate\
  \ to Y\" questions - You need diverse perspectives on the same knowledge corpus\
  \ - You want to discover non-obvious connections across domains - Standard KG search\
  \ returns shallow or overly literal results\nImportant: this is computationally\
  \ expensive. Only invoke after checking whether standard keyword/embedding/hybrid\
  \ search is sufficient."
---

# knowledge_sememolution_skill

Skill for population-based Knowledge Graph evolution via Sememolution. Use this when the user wants creative cross-domain synthesis, speculative reasoning, or when a single KG search might be too narrow.
Sememolution maintains a population of KG "individuals". Each individual has its own graph (different facts, concepts, links) and its own genome controlling how it searches and evolves.
Core genome parameters: - `lambda_depth` — Poisson rate for search traversal depth - `lambda_breadth` — Poisson rate for search breadth per step - `sleep_ops` — which refinement ops to apply during sleep - `dream_probability` — chance of speculative synthesis per cycle
Workflow: 1. Create a population: `SememolutionPopulation(model, provider, population_size=100, sample_size=10)` 2. Initialize: `pop.initialize()` 3. Assimilate text: `pop.assimilate_text(chunk)` — each individual absorbs it differently 4. Sleep cycle: `pop.sleep_cycle()` — each individual prunes/deepens independently 5. Query and rank: `pop.query_and_rank(question)` — sample individuals, each searches
   its own graph with Poisson-sampled depth/breadth, generates a response,
   and responses are ranked. Winners get fitness bumps.
6. Evolve: `pop.evolve_generation()` — tournament selection, crossover, mutation.
When to use this: - The user asks open-ended "what if" or "how might X relate to Y" questions - You need diverse perspectives on the same knowledge corpus - You want to discover non-obvious connections across domains - Standard KG search returns shallow or overly literal results
Important: this is computationally expensive. Only invoke after checking whether standard keyword/embedding/hybrid search is sufficient.

## Inputs

- `name` (default: `'task'`)
- `description` (default: `'initialize | assimilate | query_rank | evolve | sleep'`)
- `name` (default: `'population_size'`)
- `description` (default: `'Number of individuals (default 100)'`)
- `name` (default: `'query_text'`)
- `description` (default: `'Question to ask the population (for query_rank)'`)

## Steps

- `instruct` → [`instruct.py`](./instruct.py)

## Usage

```
/run_jinx jinx_ref=knowledge_sememolution_skill input_values={"name": "query_text", "description": "Question to ask the population (for query_rank)"}
```
