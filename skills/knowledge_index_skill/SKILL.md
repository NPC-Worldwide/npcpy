---
name: knowledge_index_skill
description: "Skill for using the KnowledgeIndex registry to discover which directories\
  \ contain `.knowledge.yaml` files without walking the filesystem.\nKnowledgeIndex\
  \ is a lightweight SQLite cache. It maps directory paths to file metadata (mtime,\
  \ memory_count, link_count). The database path is caller-provided \u2014 npcpy does\
  \ not hardcode a default.\nKey operations: - Upsert a directory after writing to\
  \ its YAML:\n  `upsert_directory(db_path, directory, memory_count, link_count)`\n\
  - List known directories: `get_known_directories(db_path, min_mtime=None)` - Full\
  \ rescan of a tree: `scan_root(db_path, root, max_depth=5)` - Remove a stale directory:\
  \ `remove_directory(db_path, directory)`\nTypical flow: 1. Call `scan_root(db_path,\
  \ root=\"/home/user/projects\", max_depth=5)`\n   to populate the index with every\
  \ `.knowledge.yaml` found.\n2. Query `get_known_directories(db_path)` to get a list\
  \ of directories\n   with memory/link counts.\n3. For each directory of interest,\
  \ instantiate `KnowledgeStore(directory)`\n   and call `load()` or `build_context()`.\n\
  \nThe index is a cache, not the source of truth. If a `.knowledge.yaml` is deleted\
  \ or modified outside the app, re-run `scan_root` to refresh."
---

# knowledge_index_skill

Skill for using the KnowledgeIndex registry to discover which directories contain `.knowledge.yaml` files without walking the filesystem.
KnowledgeIndex is a lightweight SQLite cache. It maps directory paths to file metadata (mtime, memory_count, link_count). The database path is caller-provided — npcpy does not hardcode a default.
Key operations: - Upsert a directory after writing to its YAML:
  `upsert_directory(db_path, directory, memory_count, link_count)`
- List known directories: `get_known_directories(db_path, min_mtime=None)` - Full rescan of a tree: `scan_root(db_path, root, max_depth=5)` - Remove a stale directory: `remove_directory(db_path, directory)`
Typical flow: 1. Call `scan_root(db_path, root="/home/user/projects", max_depth=5)`
   to populate the index with every `.knowledge.yaml` found.
2. Query `get_known_directories(db_path)` to get a list of directories
   with memory/link counts.
3. For each directory of interest, instantiate `KnowledgeStore(directory)`
   and call `load()` or `build_context()`.

The index is a cache, not the source of truth. If a `.knowledge.yaml` is deleted or modified outside the app, re-run `scan_root` to refresh.

## Inputs

- `name` (default: `'action'`)
- `description` (default: `'scan | list | upsert | remove'`)
- `name` (default: `'db_path'`)
- `description` (default: `'Path to the knowledge index SQLite file'`)
- `name` (default: `'root_or_directory'`)
- `description` (default: `'Root to scan or specific directory to upsert/remove'`)

## Steps

- `instruct` → [`instruct.py`](./instruct.py)

## Usage

```
/run_jinx jinx_ref=knowledge_index_skill input_values={"name": "root_or_directory", "description": "Root to scan or specific directory to upsert/remove"}
```
