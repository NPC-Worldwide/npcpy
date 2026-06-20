---
name: knowledge_graph_skill
description: "Skill for searching and evolving the SQLite-backed Knowledge Graph.\
  \ Use this when you need structured fact/concept/link search across one or more\
  \ teams, NPCs, or directory scopes.\nThe Knowledge Graph (KG) is stored in the application's\
  \ database (not YAML). It is scoped by (team_name, npc_name, directory_path). Facts\
  \ and concepts carry generation numbers and origin tags.\nSearch methods (choose\
  \ the right one):\n1. Keyword search \u2014 fast substring match over fact statements.\n\
  \   `kg_search_facts(engine_or_kg, \"keyword\")` \u2192 List[str]\n\n2. Embedding\
  \ search \u2014 semantic cosine similarity via vector embeddings.\n   `kg_embedding_search(engine_or_kg,\
  \ query=\"...\", embedding_model=\"nomic-embed-text\",\n    embedding_provider=\"\
  ollama\", similarity_threshold=0.6, max_results=20)`\n   \u2192 List[dict] with\
  \ 'content', 'type', 'score'\n\n3. Link search \u2014 graph traversal (BFS/DFS)\
  \ starting from keyword-matched seeds.\n   `kg_link_search(engine_or_kg, query=\"\
  ...\", max_depth=2, breadth_per_step=5,\n    strategy=\"bfs\", max_results=20)`\n\
  \   \u2192 List[dict] with 'content', 'type', 'depth', 'path', 'score'\n\n4. Hybrid\
  \ search \u2014 combines keyword + embedding + link, boosting results\n   found\
  \ by multiple methods.\n   `kg_hybrid_search(engine_or_kg, query=\"...\", mode=\"\
  all\", max_depth=2,\n    similarity_threshold=0.6, max_results=20)`\n   \u2192 List[dict]\
  \ with 'content', 'type', 'score', 'source'\n\nGraph evolution (use sparingly, usually\
  \ in background): - `kg_initial(content, model, provider)` \u2014 build a new KG\
  \ from text - `kg_evolve_incremental(existing_kg, new_content_text, ...)` \u2014\
  \ add content - `kg_sleep_process(existing_kg, model, provider)` \u2014 prune/deepen/consolidate\
  \ - `kg_dream_process(existing_kg, model, provider, num_seeds)` \u2014 speculative\
  \ synthesis\nWhen a user asks a question that spans facts, concepts, and their relationships,\
  \ prefer hybrid search. For pure semantic similarity without graph structure, use\
  \ embedding search. For exploring connected neighborhoods, use link search with\
  \ BFS."
---

# knowledge_graph_skill

Skill for searching and evolving the SQLite-backed Knowledge Graph. Use this when you need structured fact/concept/link search across one or more teams, NPCs, or directory scopes.
The Knowledge Graph (KG) is stored in the application's database (not YAML). It is scoped by (team_name, npc_name, directory_path). Facts and concepts carry generation numbers and origin tags.
Search methods (choose the right one):
1. Keyword search — fast substring match over fact statements.
   `kg_search_facts(engine_or_kg, "keyword")` → List[str]

2. Embedding search — semantic cosine similarity via vector embeddings.
   `kg_embedding_search(engine_or_kg, query="...", embedding_model="nomic-embed-text",
    embedding_provider="ollama", similarity_threshold=0.6, max_results=20)`
   → List[dict] with 'content', 'type', 'score'

3. Link search — graph traversal (BFS/DFS) starting from keyword-matched seeds.
   `kg_link_search(engine_or_kg, query="...", max_depth=2, breadth_per_step=5,
    strategy="bfs", max_results=20)`
   → List[dict] with 'content', 'type', 'depth', 'path', 'score'

4. Hybrid search — combines keyword + embedding + link, boosting results
   found by multiple methods.
   `kg_hybrid_search(engine_or_kg, query="...", mode="all", max_depth=2,
    similarity_threshold=0.6, max_results=20)`
   → List[dict] with 'content', 'type', 'score', 'source'

Graph evolution (use sparingly, usually in background): - `kg_initial(content, model, provider)` — build a new KG from text - `kg_evolve_incremental(existing_kg, new_content_text, ...)` — add content - `kg_sleep_process(existing_kg, model, provider)` — prune/deepen/consolidate - `kg_dream_process(existing_kg, model, provider, num_seeds)` — speculative synthesis
When a user asks a question that spans facts, concepts, and their relationships, prefer hybrid search. For pure semantic similarity without graph structure, use embedding search. For exploring connected neighborhoods, use link search with BFS.

## Inputs

- `name` (default: `'search_method'`)
- `description` (default: `'keyword | embedding | link | hybrid'`)
- `name` (default: `'query'`)
- `description` (default: `"The user's query or topic to search"`)
- `name` (default: `'scope_team'`)
- `description` (default: `'Team name to scope the search (optional)'`)
- `name` (default: `'scope_npc'`)
- `description` (default: `'NPC name to scope the search (optional)'`)
- `name` (default: `'scope_directory'`)
- `description` (default: `'Directory path to scope the search (optional)'`)

## Steps

- `instruct` → [`instruct.py`](./instruct.py)

## Usage

```
/run_jinx jinx_ref=knowledge_graph_skill input_values={"name": "scope_directory", "description": "Directory path to scope the search (optional)"}
```
