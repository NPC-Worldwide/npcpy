# Conversation History

npcpy persists conversations, attachments, jinx executions, NPC executions, labels, and autocomplete training data directly in SQLite via SQLAlchemy. There is no intermediate wrapper class — callers use `npcpy.db` helpers or direct SQLAlchemy calls.

## Database Path

The database path is always provided by the caller (typically the application that hosts npcpy, such as incognide). npcpy does not hardcode a default path.

```python
from sqlalchemy import create_engine
from npcpy.db import ensure_engine, generate_message_id, get_db_connection

# Explicit engine
db_path = "/home/user/.incognide/history.db"
engine = create_engine(f"sqlite:///{db_path}")

# Or use the npcpy.db helper
engine = ensure_engine(db_path)
```

## Conversations

### Starting a Conversation

```python
import datetime

conversation_id = f"my_app_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"
```

### Saving Messages

```python
from npcpy.db import generate_message_id, get_db_connection
import json

message_id = generate_message_id()

with get_db_connection(db_path) as conn:
    conn.execute(
        """
        INSERT INTO conversation_history
        (message_id, timestamp, role, content, conversation_id, directory_path,
         model, provider, npc, team, reasoning_content, tool_calls, tool_results,
         parent_message_id, device_id, device_name, params,
         input_tokens, output_tokens, cost, execution_mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "assistant",
            "The sky is blue because of Rayleigh scattering.",
            conversation_id,
            "/home/user/projects",
            "qwen3:4b",
            "ollama",
            "levi",
            "npc_team",
            None,
            json.dumps([{"id": "call_123", "function": {"name": "studio.read_pane", "arguments": "{}"}}]),
            None,
            None,
            None,
            None,
            None,
            150,
            42,
            "0.003",
            "chat",
        ),
    )
    conn.commit()
```

### Retrieving Conversations

```python
with get_db_connection(db_path) as conn:
    rows = conn.execute(
        "SELECT * FROM conversation_history WHERE conversation_id = ? ORDER BY timestamp",
        (conversation_id,),
    ).fetchall()
    messages = [dict(row) for row in rows]
```

### Deleting Messages

```python
with get_db_connection(db_path) as conn:
    conn.execute("DELETE FROM conversation_history WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM message_attachments WHERE conversation_id = ?", (conversation_id,))
    conn.commit()
```

## Attachments

```python
# Add a binary attachment to a message
with get_db_connection(db_path) as conn:
    conn.execute(
        """
        INSERT INTO message_attachments
        (message_id, conversation_id, name, attachment_type, data, size, file_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (message_id, conversation_id, "screenshot.png", "image/png", b"...", 45231, "/home/user/screenshot.png"),
    )
    conn.commit()

# Retrieve attachments
with get_db_connection(db_path) as conn:
    rows = conn.execute(
        "SELECT * FROM message_attachments WHERE message_id = ?",
        (message_id,),
    ).fetchall()
    attachments = [dict(row) for row in rows]
```

## Jinx Executions

```python
with get_db_connection(db_path) as conn:
    conn.execute(
        """
        INSERT INTO jinx_executions
        (message_id, jinx_name, input, timestamp, npc, team, conversation_id, output, status, error_message, duration_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message_id,
            "propose_fix",
            json.dumps({"file": "main.py", "issue": "off-by-one error"}),
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "levi",
            "npc_team",
            conversation_id,
            json.dumps({"diff": "...", "success": True}),
            "complete",
            None,
            1250,
        ),
    )
    conn.commit()
```

## Labels

```python
# Add a label
with get_db_connection(db_path) as conn:
    conn.execute(
        "INSERT INTO labels (entity_type, entity_id, label, metadata) VALUES (?, ?, ?, ?)",
        ("message", message_id, "training", json.dumps({"quality": "high"})),
    )
    conn.commit()

# Query labels
with get_db_connection(db_path) as conn:
    rows = conn.execute(
        "SELECT * FROM labels WHERE entity_type = ?",
        ("message",),
    ).fetchall()
    labels = [dict(row) for row in rows]
```

## Activity Logging

```python
with get_db_connection(db_path) as conn:
    conn.execute(
        "INSERT INTO activity_log (activity_type, activity_data, directory_path, npc, session_id, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
        ("file_open", "main.py", "/home/user/projects", "levi", "session_abc123", datetime.datetime.now(datetime.timezone.utc).isoformat()),
    )
    conn.commit()
```

## Autocomplete Training

```python
with get_db_connection(db_path) as conn:
    conn.execute(
        """
        INSERT INTO autocomplete_suggestions
        (suggestion_type, input_context, suggestion, accepted, npc, model, provider, directory_path, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "shell_command",
            "git sta",
            "git status",
            True,
            "levi",
            "qwen3:4b",
            "ollama",
            "/home/user/projects",
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
        ),
    )
    conn.commit()
```

## Schema

The database creates these tables on first use:

| Table | Purpose |
|---|---|
| `command_history` | Shell commands with output and location |
| `conversation_history` | All chat messages with metadata (role, model, NPC, tokens, cost, reasoning, tool calls) |
| `message_attachments` | Binary attachments linked to messages |
| `labels` | User-defined tags on messages and executions |
| `jinx_executions` | Workflow execution records |
| `npc_executions` | NPC invocation records |
| `activity_log` | General activity events |
| `autocomplete_suggestions` | Suggestion acceptance tracking |
| `autocomplete_training` | Curated training pairs |
| `kg_facts`, `kg_concepts`, `kg_links` | Knowledge graph tables |

SQLite ALTER TABLE migrations are applied automatically for backward compatibility with older databases.

## Memory and Knowledge YAML

For directory-local memory that lives alongside your code, see [Knowledge YAML](knowledge-yaml.md). This is the preferred system for project-specific memories, while the database tables above handle conversation history and execution tracking.
