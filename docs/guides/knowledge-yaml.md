# Knowledge YAML

npcpy supports a directory-local knowledge system based on plain `.knowledge.yaml` files. Each working directory can maintain its own append-only memory and knowledge graph, independent of any global database or team configuration.

## Architecture

| Component | Responsibility | Writes to |
|-----------|--------------|-----------|
| `KnowledgeStore` | Read/write `.knowledge.yaml` in a single directory | YAML only |
| `KnowledgeIndex` | Registry of *which* directories have a `.knowledge.yaml` | Caller-provided SQLite DB |

`KnowledgeStore` never touches a database. `KnowledgeIndex` is a lightweight SQLite cache that maps `directory → file metadata` so cross-directory searches don't need to walk the filesystem on every call. The database path is always passed by the caller; npcpy does not hardcode a default.

## Data Structure

A `.knowledge.yaml` file has this top-level structure:

```yaml
version: "1.0"
directory: /abs/path/to/project
created_at: "2024-06-11T12:00:00+00:00"
updated_at: "2024-06-11T12:00:00+00:00"
memories:
  - id: abc123...
    message_id: msg-456
    conversation_id: conv-789
    npc: levi
    team: npc_team
    directory_path: /abs/path/to/project
    timestamp: "2024-06-11T12:00:00+00:00"
    initial_memory: "User wants to migrate from Stripe to Clerk."
    final_memory: "Migrated auth from Stripe to Clerk."
    status: human-approved
    model: qwen3:4b
    provider: ollama
    created_at: "2024-06-11T12:00:00+00:00"
knowledge:
  - id: def456...
    from: abc123...
    to: abc124...
    relation: refines
    created_at: "2024-06-11T12:00:00+00:00"
    agent: levi
```

### Fields

| Section | Field | Description |
|---------|-------|-------------|
| Top-level | `version` | Schema version (currently `"1.0"`) |
| Top-level | `directory` | Absolute path of the directory this file belongs to |
| Top-level | `created_at` / `updated_at` | ISO 8601 timestamps |
| `memories` | `id` | UUID (hex) |
| `memories` | `message_id` | Optional reference to a message in conversation history |
| `memories` | `conversation_id` | Optional reference to a conversation |
| `memories` | `npc` / `team` / `directory_path` | Context where the memory was captured |
| `memories` | `initial_memory` | Raw extracted text |
| `memories` | `final_memory` | Human-edited or approved version |
| `memories` | `status` | `pending_approval`, `human-approved`, `human-rejected`, or `human-edited` |
| `memories` | `model` / `provider` | LLM that performed the extraction |
| `knowledge` | `id` | UUID for the link |
| `knowledge` | `from` / `to` | Memory IDs that this link connects |
| `knowledge` | `relation` | Free-text relation label (e.g. `refines`, `contradicts`, `enables`) |
| `knowledge` | `agent` | NPC or tool that created the link |

## Using KnowledgeStore

### Open or initialize a directory

```python
from npcpy.memory.knowledge_store import KnowledgeStore, get_store_for_path

store = get_store_for_path("/home/user/myproject")
# If .knowledge.yaml doesn't exist, writes an empty template.
```

### Append a memory

```python
mem_id = store.append_memory(
    message_id="msg-456",
    conversation_id="conv-789",
    npc="levi",
    team="npc_team",
    directory_path="/home/user/myproject",
    initial_memory="User wants to migrate from Stripe to Clerk.",
    status="pending_approval",
    model="qwen3:4b",
    provider="ollama",
)
```

### Update a memory (approve / reject / edit)

```python
store.update_memory(
    mem_id=mem_id,
    status="human-approved",
    final_memory="Migrated auth from Stripe to Clerk.",
)
```

### Query memories

```python
# All memories in this directory
all_mems = store.get_memories()

# Only pending approval
pending = store.get_pending_memories()

# Approved memories, most recent 10
approved = store.get_memories(status="human-approved", limit=10)

# Keyword search (case-insensitive substring)
results = store.search_memories("Stripe", limit=5)
```

### Build context for LLM prompts

```python
context = store.build_context(max_memories=10)
# Returns a formatted string of approved memories, or empty string if none.
```

### Create links between memories

```python
link_id = store.append_link(
    from_mem=mem_id_a,
    to_mem=mem_id_b,
    relation="refines",
    agent="levi",
)
```

### Read links

```python
all_links = store.get_links()
related = store.get_links_for_memory(mem_id)
# Returns {"outgoing": [...], "incoming": [...]}
```

### Low-level I/O

```python
# Load raw dict
data = store.load()

# Save a modified dict directly
store.save(data)
```

### Aggregate across a tree

```python
# Find every .knowledge.yaml under a root
stores = KnowledgeStore.find_all("/home/user/projects")

# Merge them into a single in-memory view (no files are modified)
aggregate = KnowledgeStore.aggregate("/home/user/projects", max_depth=3)
# Returns a dict with keys: version, root, sources, memories, knowledge
```

## Using KnowledgeIndex

`KnowledgeIndex` is a SQLite registry that tracks `.knowledge.yaml` files across a directory tree. It is purely a cache — the source of truth is still the YAML files themselves.

### Register a directory after writing to its YAML

```python
from npcpy.memory.knowledge_index import upsert_directory

upsert_directory(
    db_path="/home/user/.incognide/knowledge_index.db",
    directory="/home/user/myproject",
    memory_count=len(store.get_memories()),
    link_count=len(store.get_links()),
)
```

### List known directories

```python
from npcpy.memory.knowledge_index import get_known_directories

rows = get_known_directories("/home/user/.incognide/knowledge_index.db")
# Each row: {"directory": str, "mtime": float, "memory_count": int, "link_count": int}
```

### Full rescan of a tree

```python
from npcpy.memory.knowledge_index import scan_root

found = scan_root(
    db_path="/home/user/.incognide/knowledge_index.db",
    root="/home/user/projects",
    max_depth=5,
)
# Returns list of directory paths that contain .knowledge.yaml
```

### Remove a directory from the index

```python
from npcpy.memory.knowledge_index import remove_directory

remove_directory(
    db_path="/home/user/.incognide/knowledge_index.db",
    directory="/home/user/old_project",
)
```

## Thread Safety

`KnowledgeStore` uses an `RLock` so concurrent reads and writes to the same `.knowledge.yaml` file are serialized. `KnowledgeIndex` uses a module-level `threading.Lock` for the same purpose. Safe to call from multi-threaded servers.

## Relationship to Knowledge Graphs

The `.knowledge.yaml` system and the SQLite-backed knowledge graph (`npcpy.memory.knowledge_graph`) are complementary:

- **Knowledge YAML** — directory-local, human-editable, append-only, no database required. Good for project-specific memories that follow the code.
- **Knowledge Graph** — global or team-scoped, LLM-evolved, structured facts/concepts/links. Good for cross-project reasoning.

Approved memories from `.knowledge.yaml` can be fed into `kg_evolve_incremental` or `kg_backfill_from_memories` to enrich the graph. Conversely, facts extracted by the KG pipeline can be written back to `.knowledge.yaml` via `append_memory` for local persistence.
