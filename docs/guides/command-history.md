# Command History

`npcpy.memory.command_history` provides persistent storage for conversations, commands, jinx executions, NPC interactions, attachments, memory lifecycle, and autocomplete training data. It uses SQLAlchemy with SQLite (or PostgreSQL) and automatically manages schema migration.

## Quick Start

```python
from npcpy.memory.command_history import CommandHistory, start_new_conversation, save_conversation_message

# Initialize with default SQLite path (~/.npcsh_history.db)
ch = CommandHistory()

# Or with a custom path
ch = CommandHistory("~/my_project_history.db")

# Or with an existing SQLAlchemy engine
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@localhost/db")
ch = CommandHistory(engine)
```

## Conversations

### Starting a Conversation

```python
conversation_id = start_new_conversation(prepend="my_app")
# -> "my_app_20260115143022"
```

### Saving Messages

```python
save_conversation_message(
    command_history=ch,
    conversation_id=conversation_id,
    role="assistant",
    content="The sky is blue because of Rayleigh scattering.",
    wd="/home/user/projects",
    model="claude-sonnet-4-6",
    provider="anthropic",
    npc="levi",
    team="npc_team",
    tool_calls=[{"id": "call_123", "function": {"name": "studio.read_pane", "arguments": "{}"}}],
    reasoning_content="I need to check the current pane content first...",
    input_tokens=150,
    output_tokens=42,
    cost=0.003,
)
```

### Retrieving Conversations

```python
# Get all messages in a conversation
messages = ch.get_conversations_by_id(conversation_id)

# Get the most recent message ID
last_id = ch.get_last_message_id(conversation_id)

# Get messages by NPC or team
npc_msgs = ch.get_messages_by_npc("levi", n_last=50)
team_msgs = ch.get_messages_by_team("npc_team", n_last=50)

# Search across all conversations
results = ch.search_conversations("Rayleigh scattering")
```

### Deleting Messages

```python
ch.delete_message(conversation_id, message_id)
```

## Commands

```python
# Log a shell command
ch.add_command(
    command="ls -la",
    subcommands=["-la"],
    output="total 128\ndrwxr-xr-x ...",
    location="/home/user/projects"
)

# Get recent commands
recent = ch.get_all_commands(limit=10)

# Search command history
found = ch.search_commands("git commit")

# Analyze command patterns over time
df = ch.get_command_patterns(timeframe='day')  # 'hour', 'day', 'week', 'month'
```

## Jinx Executions

Jinxes (workflow scripts) are automatically tracked via SQLite triggers when a user message starts with `/`. You can also save them explicitly:

```python
ch.save_jinx_execution(
    triggering_message_id="msg_123",
    conversation_id=conversation_id,
    npc_name="levi",
    jinx_name="propose_fix",
    jinx_inputs={"file": "main.py", "issue": "off-by-one error"},
    jinx_output={"diff": "...", "success": True},
    status="complete",
    team_name="npc_team",
    duration_ms=1250,
)

# Query executions
executions = ch.get_jinx_executions(jinx_name="propose_fix", limit=100)
```

## NPC Executions

NPC interactions are automatically tracked when a message has an `npc` field. Query them with:

```python
npc_runs = ch.get_npc_executions(npc_name="levi", limit=50)
```

## Labels

Tag any entity (messages, jinx executions, etc.) for later filtering or training data extraction:

```python
ch.add_label(entity_type="message", entity_id="msg_123", label="training", metadata={"quality": "high"})
ch.add_label(entity_type="message", entity_id="msg_123", label="bug_fix")

# Query labels
labels = ch.get_labels(entity_type="message")
training_data = ch.get_training_data_by_label(label="training")
```

## Attachments

```python
# Add a binary attachment to a message
ch.add_attachment(
    message_id="msg_123",
    name="screenshot.png",
    attachment_type="image/png",
    data=b"...",
    size=45231,
    file_path="/home/user/screenshot.png"
)

# Retrieve attachment metadata
attachments = ch.get_message_attachments("msg_123")

# Get binary data
data, name, mimetype = ch.get_attachment_data(attachment_id=1)

# Delete attachment
ch.delete_attachment(attachment_id=1)
```

## Memory Lifecycle

Track memory generation, approval, and feedback for iterative memory refinement:

```python
# Store a memory pending approval
memory_id = ch.add_memory_to_database(
    message_id="msg_123",
    conversation_id=conversation_id,
    npc="levi",
    team="npc_team",
    directory_path="/home/user/projects",
    initial_memory="User prefers snake_case for Python variables.",
    status="pending_approval",
    model="claude-sonnet-4-6",
    provider="anthropic"
)

# After human review, update status
ch.update_memory_status(memory_id, new_status="human-approved")
# Or reject
ch.update_memory_status(memory_id, new_status="human-rejected")
# Or edit
ch.update_memory_status(memory_id, new_status="human-edited", final_memory="User prefers snake_case for Python and camelCase for JavaScript.")

# Get memories scoped to NPC/team/path
memories = ch.get_memories_for_scope(
    npc="levi",
    team="npc_team",
    directory_path="/home/user/projects"
)

# Get approved memories for knowledge graph backfill
approved = ch.get_approved_memories_by_scope()

# Search memories
results = ch.search_memory("snake_case", npc="levi", limit=10)

# Get examples for in-context learning
examples = ch.get_memory_examples_for_context(
    npc="levi",
    team="npc_team",
    directory_path="/home/user/projects",
    n_approved=10,
    n_rejected=10,
    n_edited=5,
)
# Returns {"approved": [...], "rejected": [...], "edited": [...]}
```

## Analytics

```python
import pandas as pd

# NPC conversation stats
df = ch.get_npc_conversation_stats(start_date="2026-01-01", end_date="2026-06-01")
# Columns: npc, total_messages, avg_message_length, total_conversations, models_used, providers_used, model_list, provider_list, first_conversation, last_conversation

# Autocomplete performance
df = ch.get_autocomplete_stats(suggestion_type="shell_command", npc="levi")
# Columns: suggestion_type, total, accepted, rejected

# Training data export
training = ch.get_training_data(suggestion_type="shell_command", accepted_only=True, limit=1000)
```

## Activity Logging

```python
ch.log_activity(
    activity_type="file_open",
    activity_data="main.py",
    directory_path="/home/user/projects",
    npc="levi",
    session_id="session_abc123"
)

activities = ch.get_activities(activity_type="file_open", limit=50)
```

## Autocomplete Training

Log autocomplete suggestions and whether they were accepted to build a training corpus:

```python
ch.log_autocomplete(
    suggestion_type="shell_command",
    input_context="git sta",
    suggestion="git status",
    accepted=True,
    npc="levi",
    model="claude-sonnet-4-6",
    provider="anthropic",
    directory_path="/home/user/projects"
)
```

## Schema

The database automatically creates these tables on first use:

| Table | Purpose |
|---|---|
| `command_history` | Shell commands with output and location |
| `conversation_history` | All chat messages with metadata (role, model, NPC, tokens, cost, reasoning, tool calls) |
| `message_attachments` | Binary attachments linked to messages |
| `labels` | User-defined tags on messages and executions |
| `jinx_executions` | Workflow execution records |
| `npc_executions` | NPC invocation records |
| `memory_lifecycle` | Memory generation → approval pipeline |
| `activity_log` | General activity events |
| `autocomplete_suggestions` | Suggestion acceptance tracking |
| `autocomplete_training` | Curated training pairs |
| `kg_facts`, `kg_concepts`, `kg_links` | Knowledge graph tables (initialized via `init_kg_schema`) |

SQLite ALTER TABLE migrations are applied automatically for backward compatibility with older databases.

## Closing

```python
ch.close()  # Dispose SQLAlchemy engine
```
