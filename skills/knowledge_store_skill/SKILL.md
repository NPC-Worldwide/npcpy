---
name: knowledge_store_skill
description: "Skill for working with local .knowledge.yaml files via KnowledgeStore.\
  \ Use this when you need to recall, search, or manage directory-local memories and\
  \ knowledge links stored in plain YAML alongside the user's project files.\nKnowledgeStore\
  \ is directory-scoped. Each directory that contains a `.knowledge.yaml` file maintains\
  \ its own append-only memory graph. There is no global database \u2014 the YAML\
  \ IS the source of truth.\nKey operations: - Load a directory's store: `npcpy.memory.knowledge_store.get_store_for_path(path)`\
  \ - Append a memory: `store.append_memory(initial_memory=\"...\", status=\"pending_approval\"\
  , ...)` - Update a memory (approve/reject/edit): `store.update_memory(mem_id, status,\
  \ final_memory)` - Search memories (keyword substring): `store.search_memories(\"\
  query\", limit=20)` - Get approved context for LLM prompts: `store.build_context(max_memories=10)`\
  \ - Get links for a memory: `store.get_links_for_memory(mem_id)` - Create a link\
  \ between memories: `store.append_link(from_mem, to_mem, relation=\"refines\", agent=\"\
  your_name\")` - Aggregate across a tree: `KnowledgeStore.aggregate(root_directory,\
  \ max_depth=3)`\nMemory statuses: - `pending_approval` \u2014 raw extraction, needs\
  \ human review - `human-approved` \u2014 confirmed and available for context injection\
  \ - `human-rejected` \u2014 discard, can be used as negative examples - `human-edited`\
  \ \u2014 corrected version supersedes initial_memory\nWhen answering questions,\
  \ prefer `build_context()` for recently approved local knowledge, and `search_memories()`\
  \ for targeted recall. Always respect `human-rejected` memories \u2014 do not repeat\
  \ them."
---

# knowledge_store_skill

Skill for working with local .knowledge.yaml files via KnowledgeStore. Use this when you need to recall, search, or manage directory-local memories and knowledge links stored in plain YAML alongside the user's project files.
KnowledgeStore is directory-scoped. Each directory that contains a `.knowledge.yaml` file maintains its own append-only memory graph. There is no global database — the YAML IS the source of truth.
Key operations: - Load a directory's store: `npcpy.memory.knowledge_store.get_store_for_path(path)` - Append a memory: `store.append_memory(initial_memory="...", status="pending_approval", ...)` - Update a memory (approve/reject/edit): `store.update_memory(mem_id, status, final_memory)` - Search memories (keyword substring): `store.search_memories("query", limit=20)` - Get approved context for LLM prompts: `store.build_context(max_memories=10)` - Get links for a memory: `store.get_links_for_memory(mem_id)` - Create a link between memories: `store.append_link(from_mem, to_mem, relation="refines", agent="your_name")` - Aggregate across a tree: `KnowledgeStore.aggregate(root_directory, max_depth=3)`
Memory statuses: - `pending_approval` — raw extraction, needs human review - `human-approved` — confirmed and available for context injection - `human-rejected` — discard, can be used as negative examples - `human-edited` — corrected version supersedes initial_memory
When answering questions, prefer `build_context()` for recently approved local knowledge, and `search_memories()` for targeted recall. Always respect `human-rejected` memories — do not repeat them.

## Inputs

- `name` (default: `'action'`)
- `description` (default: `'load | search | append | update | link | context | aggregate'`)
- `name` (default: `'directory_path'`)
- `description` (default: `'Absolute path of the directory containing .knowledge.yaml'`)
- `name` (default: `'query_or_memory'`)
- `description` (default: `'Search query, memory text, or JSON params depending on action'`)

## Steps

- `instruct` → [`instruct.py`](./instruct.py)

## Usage

```
/run_jinx jinx_ref=knowledge_store_skill input_values={"name": "query_or_memory", "description": "Search query, memory text, or JSON params depending on action"}
```
