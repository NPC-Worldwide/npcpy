# Knowledge Graphs

npcpy includes an LLM-driven knowledge graph system that extracts facts from text, organizes them into concepts, links everything together, and supports multi-modal search. The KG evolves over time through incremental ingestion, a sleep process (pruning and deepening), and a dream process (creative speculation).

## Data Structure

A knowledge graph is a plain Python dictionary with these keys:

| Key                     | Type                          | Description                                          |
|------------------------|-------------------------------|------------------------------------------------------|
| `generation`           | `int`                         | Current generation counter (increments on each evolution) |
| `facts`               | `List[dict]`                  | Each dict has `statement`, `generation`, optional `origin` |
| `concepts`            | `List[dict]`                  | Each dict has `name`, `generation`, optional `description` |
| `fact_to_concept_links`| `Dict[str, List[str]]`       | Maps fact statement to list of concept names          |
| `fact_to_fact_links`  | `List[Tuple[str, str]]`       | Pairs of related fact statements                      |
| `concept_links`       | `List[Tuple[str, str]]`       | Pairs of related concept names                        |

Every fact and concept carries a `generation` number so you can track when each piece of knowledge was added.

## Creating a KG from Text

`kg_initial` takes raw text, extracts facts via an LLM, infers implied facts with `zoom_in`, generates concept groups, and links everything.

```python
from npcpy.memory.knowledge_graph import kg_initial

kg = kg_initial(
    content="Python was created by Guido van Rossum and released in 1991. "
            "It emphasizes code readability and supports multiple paradigms "
            "including procedural, object-oriented, and functional programming.",
    model="qwen3:4b",
    provider="ollama",
)

print(f"Facts:    {len(kg['facts'])}")
print(f"Concepts: {len(kg['concepts'])}")
print(f"Generation: {kg['generation']}")  # 0
```

What happens inside `kg_initial`:

1. **Fact extraction** -- calls `get_facts()` to pull structured statements from the content.
2. **Zoom in** -- calls `zoom_in()` on the extracted facts to infer implied facts.
3. **Concept generation** -- calls `generate_groups()` to cluster facts into named concepts.
4. **Linking** -- links facts to concepts (`get_related_concepts_multi`), and facts to other facts (`get_related_facts_llm`).

For large content (over 10,000 characters), the text is sampled in 10,000-character windows to keep LLM calls manageable.

## Evolving Incrementally

Add new content to an existing KG without rebuilding from scratch.

```python
from npcpy.memory.knowledge_graph import kg_evolve_incremental

evolved_kg, _ = kg_evolve_incremental(
    existing_kg=kg,
    new_content_text="Python 3.12 introduced performance improvements and better error messages.",
    model="qwen3:4b",
    provider="ollama",
    get_concepts=True,           # generate new concepts from new facts
    link_concepts_facts=True,    # link new facts to concepts
    link_concepts_concepts=True, # link new concepts to existing ones
    link_facts_facts=True,       # link new facts to existing facts
)

print(f"Generation: {evolved_kg['generation']}")  # 1
print(f"Total facts: {len(evolved_kg['facts'])}")
```

New facts receive the incremented generation number. Existing facts and concepts are preserved as-is. Duplicate concepts are skipped.

## Sleep Process

The sleep process refines and consolidates the KG. It finds orphaned facts, prunes redundancy, and deepens understanding.

```python
from npcpy.memory.knowledge_graph import kg_sleep_process

cleaned_kg, _ = kg_sleep_process(
    existing_kg=evolved_kg,
    model="qwen3:4b",
    provider="ollama",
)
```

What happens during sleep:

1. **Phase 1: Structure orphans** -- Facts not linked to any concept are identified. If more than 20 orphans exist, `kg_initial` is called on them to generate structure.
2. **Phase 2: Refinement operations** -- One or two operations are randomly selected:
   - `prune` -- picks a random fact and calls `consolidate_facts_llm` to check if it is redundant. Redundant facts are removed.
   - `deepen` -- picks a random fact and calls `zoom_in` to infer new implied facts.
   - `abstract_link` -- creates higher-order links between concepts.

You can control which operations run:

```python
cleaned_kg, _ = kg_sleep_process(
    existing_kg=evolved_kg,
    model="qwen3:4b",
    provider="ollama",
    operations_config=['prune', 'deepen'],  # explicit list
)
```

## Dream Process

The dream process generates speculative knowledge by seeding random concepts and asking the LLM to create a narrative connecting them.

```python
from npcpy.memory.knowledge_graph import kg_dream_process

dreamed_kg, _ = kg_dream_process(
    existing_kg=evolved_kg,
    model="qwen3:4b",
    provider="ollama",
    num_seeds=3,   # pick 3 random concepts as seeds
)
```

What happens during dreaming:

1. Random concepts are sampled as seeds.
2. The LLM writes a short speculative paragraph connecting those concepts.
3. The dream text is fed into `kg_evolve_incremental` as new content.
4. Facts originating from the dream are tagged with `origin: 'dream'`.

## Searching

npcpy provides four search methods over the KG, each suited to different use cases.

### Keyword Search

```python
from npcpy.memory.knowledge_graph import kg_search_facts

results = kg_search_facts(engine, "Python")
# Returns: List[str] -- matching fact statements
```

### Embedding Search (Semantic)

Uses vector embeddings for cosine-similarity matching.

```python
from npcpy.memory.knowledge_graph import kg_embedding_search

results = kg_embedding_search(
    engine,
    query="programming language design",
    embedding_model="nomic-embed-text",
    embedding_provider="ollama",
    similarity_threshold=0.6,
    max_results=20,
    include_concepts=True,
)
# Returns: List[dict] with 'content', 'type' ('fact'|'concept'), 'score'
```

### Link Search (Graph Traversal)

Starts from keyword-matched seeds and traverses links using BFS or DFS.

```python
from npcpy.memory.knowledge_graph import kg_link_search

results = kg_link_search(
    engine,
    query="Python",
    max_depth=2,
    breadth_per_step=5,
    max_results=20,
    strategy='bfs',   # or 'dfs'
)
# Returns: List[dict] with 'content', 'type', 'depth', 'path', 'score'
```

### Hybrid Search

Combines keyword, embedding, and link search. Results found by multiple methods get boosted scores.

```python
from npcpy.memory.knowledge_graph import kg_hybrid_search

results = kg_hybrid_search(
    engine,
    query="Python paradigms",
    mode='all',              # 'keyword', 'embedding', 'link', 'keyword+link', 'keyword+embedding', 'all'
    max_depth=2,
    similarity_threshold=0.6,
    max_results=20,
)
# Returns: List[dict] with 'content', 'type', 'score', 'source'
```

## Visualization

The `npcpy.memory.kg_vis` module provides several visualization functions.

### Interactive Graph (PyVis)

Generates an HTML file with an interactive node-link diagram.

```python
from npcpy.memory.kg_vis import visualize_knowledge_graph_final_interactive

visualize_knowledge_graph_final_interactive(kg, filename="my_kg.html")
# Open my_kg.html in a browser. Facts are blue, concepts are green.
```

### Growth Chart

Track facts and concepts over generations.

```python
from npcpy.memory.kg_vis import visualize_growth

# Pass a list of KG snapshots from successive generations
visualize_growth([kg_gen0, kg_gen1, kg_gen2], filename="growth.png")
```

### Concept Trajectories

Track how concept centrality (importance) changes over generations.

```python
from npcpy.memory.kg_vis import visualize_concept_trajectories

visualize_concept_trajectories(
    [kg_gen0, kg_gen1, kg_gen2],
    n_pillars=2,   # stable backbone concepts (dashed lines)
    n_risers=3,    # fast-growing newcomers (solid lines)
    filename="trajectories.png"
)
```

## Corpus-to-KG Pipeline

A complete pipeline that loads files, creates a KG, and evolves it:

```python
from pathlib import Path
from npcpy.data.load import load_file_contents
from npcpy.memory.knowledge_graph import kg_initial, kg_evolve_incremental

MODEL   = "qwen3:4b"
PROVIDER = "ollama"

# Step 1: Load files into text chunks
files = [Path("report.pdf"), Path("notes.txt"), Path("data.csv")]
corpus = []
for f in files:
    chunks = load_file_contents(str(f), chunk_size=600)
    corpus.extend(chunks[:4])   # limit chunks per file

# Step 2: Build initial KG
content = "\n".join(corpus)
kg = kg_initial(content=content, model=MODEL, provider=PROVIDER)
print(f"Initial: {len(kg['facts'])} facts, {len(kg['concepts'])} concepts")

# Step 3: Evolve with new content
new_text = "Additional information from a meeting transcript..."
kg, _ = kg_evolve_incremental(
    kg, new_content_text=new_text,
    model=MODEL, provider=PROVIDER, get_concepts=True
)
print(f"Evolved: {len(kg['facts'])} facts, {len(kg['concepts'])} concepts")
```

## Scoping

KGs stored via `save_kg_to_db` / `load_kg_from_db` are scoped by three keys:

- **team_name** -- the team the KG belongs to
- **npc_name** -- the individual NPC
- **directory_path** -- the working directory

This allows different NPCs and teams to maintain separate knowledge bases, or share a global one when scoped to `'global_team'` / `'default_npc'`.

## Generational Tracking

Every fact and concept carries a `generation` field. This enables:

- Filtering knowledge by age (e.g., only facts from the last 3 generations).
- Visualizing how the KG grew over time with `visualize_growth`.
- Identifying dream-originated vs. observation-originated knowledge via the `origin` field.

## Memory Extraction and Lifecycle

npcpy includes a memory extraction pipeline that pulls structured memories from conversation history, stores them as pending for human approval, and feeds approved memories into the KG through a backfill process.

### Extracting Memories from Conversations

```python
from npcpy.llm_funcs import get_facts

conversation = """user: We need to use Clerk for all auth, no Stripe.
assistant: I'll configure Clerk for auth, billing, and subscriptions.
user: Make sure CSP headers allow Clerk domains."""

facts = get_facts(
    conversation,
    model="qwen3:4b",
    provider="ollama",
    context="Extract precise technical decisions and their rationale.",
)

for f in facts:
    print(f['statement'])
```

### Memory Approval Pipeline

Memories go through a lifecycle: `pending_approval` → `human-approved` / `human-rejected` / `human-edited`. Approved and rejected memories are fed back as positive and negative examples to future extraction calls, creating a self-improving quality loop.

```python
from npcpy.memory.command_history import CommandHistory

ch = CommandHistory("~/npcsh_history.db")

# Get pending memories
pending = ch.get_pending_memories(limit=20)

# Approve or reject
ch.update_memory_status(memory_id=42, new_status="human-approved")
ch.update_memory_status(memory_id=43, new_status="human-rejected")

# Get quality examples for future extraction
examples = ch.get_memory_examples_for_context(
    npc="sibiji", team="npc_team", directory_path="/my/project"
)
# Returns approved, rejected, and edited memories as few-shot examples
```

### Backfilling Approved Memories into the KG

```python
from npcpy.memory.knowledge_graph import kg_backfill_from_memories

result = kg_backfill_from_memories(
    engine=ch.engine,
    model="qwen3:4b",
    provider="ollama",
    get_concepts=True,
)
print(f"Added {result['facts_after'] - result['facts_before']} new facts")
```

## Sememolution: Population-Based KG Evolution

The Sememolution framework maintains a population of KG individuals that evolve independently. Each individual has its own graph (different facts, concepts, links) and its own genome controlling how it searches and evolves. Stochastic LLM responses and randomized sleep/dream configurations produce structural diversity — this is the mutation. Selection happens through response ranking.

### Core Concepts

- **Individual**: a full KG with its own genome of parameters
- **Genome**: lambda_depth, lambda_breadth (Poisson rate params for search traversal), sleep_ops, dream_probability, linking behavior
- **Lifecycle**: waking (assimilate text), sleeping (prune/deepen/link), dreaming (speculative synthesis)
- **Poisson sampling**: each search samples `depth ~ Poisson(lambda_depth)` and `breadth ~ Poisson(lambda_breadth)`, so the same individual produces different traversals each time
- **Fitness**: based on how well an individual's search results produce good LLM responses when ranked against other individuals

### Creating a Population

```python
from npcpy.memory.kg_population import SememolutionPopulation

pop = SememolutionPopulation(
    model="gemma3:4b",
    provider="ollama",
    population_size=100,
    sample_size=10,
)
pop.initialize()
print(f"Population: {len(pop.ga.population)} individuals")
```

### Assimilating Text

Every individual absorbs the text according to its own genome — different linking configs produce different graph structures from the same input.

```python
pop.assimilate_text("TypeORM is used for MySQL in the Celeria project...")
```

### Sleep/Dream Cycle

Each individual sleeps with its own ops config. Dreaming happens probabilistically based on the genome's dream_probability.

```python
pop.sleep_cycle()
```

### Query and Rank

Sample 10 individuals, each searches its own graph with Poisson-sampled depth/breadth, each generates a response, responses are ranked. Winners get fitness bumps.

```python
rankings = pop.query_and_rank("How should I implement the database connection?")
for r in rankings[:3]:
    print(f"  Rank {r['rank']}: {r['response'][:100]}...")
    print(f"    Facts used: {r['n_facts']}, Individual: {r['individual'].individual_id[:20]}")
```

### GA Evolution

Uses `GeneticEvolver` from `npcpy.ft.ge` for tournament selection, elitism, crossover, and mutation.

```python
stats = pop.evolve_generation()
print(f"Best fitness: {stats['best_fitness']:.3f}")
print(f"Avg fitness: {stats['avg_fitness']:.3f}")
```

### Structural Diversity

After several generations, individuals specialize. Some develop dense, shallow graphs good for keyword-heavy queries. Others build deep hierarchies that find cross-domain connections. The Poisson sampling ensures each individual's quality reveals itself over many queries despite per-query stochasticity.

```python
stats = pop.get_stats()
print(f"Lambda depth:  mean={stats['lambda_depth']['mean']:.2f}")
print(f"Lambda breadth: mean={stats['lambda_breadth']['mean']:.2f}")
print(f"Unique sleep configs: {stats['unique_sleep_configs']}")
```
