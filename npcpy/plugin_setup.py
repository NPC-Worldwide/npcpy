"""
Generate NPC plugin packages for AI coding tools.

Each plugin bundles:
  - MCP server config (npcpy.mcp_server)
  - Conversation logging hooks
  - /npc:npcs skill for NPC switching

Usage:
    python -m npcpy.plugin_setup claude [--dir OUTPUT_DIR]
    python -m npcpy.plugin_setup codex  [--dir OUTPUT_DIR]
    python -m npcpy.plugin_setup gemini [--dir OUTPUT_DIR]
"""

import os
import sys
import json
import stat
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Hook script — logs conversation events to npcsh_history.db
# ---------------------------------------------------------------------------

HOOK_SCRIPT = '''#!/usr/bin/env python3
"""NPC conversation logger hook for {tool_name}."""
import sys
import json
import os
from datetime import datetime

NPC_STATE_FILE = os.path.expanduser("~/.npcsh/.active_npc_state.json")


def _get_active_npc_directive():
    """Read the active NPC directive from the MCP server state file."""
    try:
        with open(NPC_STATE_FILE, "r") as f:
            state = json.load(f)
        return state.get("name", ""), state.get("directive", "")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return "", ""


def main():
    raw = sys.stdin.read()
    if not raw.strip():
        return

    try:
        event = json.loads(raw)
    except json.JSONDecodeError:
        return

    event_name = event.get("hook_event_name", "")
    session_id = event.get("session_id", "unknown")
    cwd = event.get("cwd", os.getcwd())
    conversation_id = "{conv_prefix}_" + session_id

    # On UserPromptSubmit, inject active NPC directive as systemMessage
    if event_name == "UserPromptSubmit":
        npc_name, directive = _get_active_npc_directive()
        if directive:
            print(json.dumps({{"systemMessage":
                f"You are {{npc_name}}. {{directive.strip()}}\\n"
                f"Only use the MCP tools available to you — these are your jinxes."
            }}))

    # Log to conversation history
    role = None
    content = None
    tool_calls = None
    tool_results = None

    if event_name == "UserPromptSubmit":
        role = "user"
        content = event.get("prompt", "")
    elif event_name == "Stop":
        role = "assistant"
        content = event.get("response_text", "")
    elif event_name == "PostToolUse":
        role = "tool"
        tool_name = event.get("tool_name", "")
        tool_input = event.get("tool_input", {{}})
        tool_output = event.get("tool_output", "")
        content = f"[{{tool_name}}]"
        tool_calls = json.dumps({{"tool_name": tool_name, "input": tool_input}})
        tool_results = json.dumps({{"output": str(tool_output)[:2000]}}) if tool_output else None
    else:
        return

    if not content:
        return

    try:
        from npcpy.memory.command_history import CommandHistory, generate_message_id

        db_path = os.environ.get("NPCSH_DB_PATH")
        if not db_path:
            try:
                from npcsh._state import NPCSH_DB_PATH
                db_path = NPCSH_DB_PATH
            except ImportError:
                db_path = os.path.expanduser("~/.npcsh/npcsh_history.db")

        ch = CommandHistory(db=db_path)
        ch.add_conversation(
            message_id=generate_message_id(),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            role=role,
            content=content,
            conversation_id=conversation_id,
            directory_path=cwd,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
    except Exception as e:
        print(f"[npc-hook] log error: {{e}}", file=sys.stderr)

if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------------------
# Skill definition for /npc:npcs
# ---------------------------------------------------------------------------

NPCS_SKILL = """---
name: npcs
description: This skill should be used when the user asks to "list NPCs", "switch NPC", "show team", "change agent", or discusses NPC team management. Lists available NPCs and switches the active NPC.
---

# NPC Team Management

List all NPCs in the team and switch to one.

Use the MCP prompts to see available NPCs and switch:

1. First, use the `npc_team` MCP prompt to see all available NPCs
2. Ask the user which NPC they want to switch to
3. Use the `npc_<name>` MCP prompt to switch to that NPC

When switching, the available tools will change to match the new NPC's jinxes.

If called with arguments: switch directly to the NPC named "$ARGUMENTS" by using the `npc_$ARGUMENTS` MCP prompt.
"""

# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def _write_file(path: Path, content: str, executable: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  wrote {path}")


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# Claude Code plugin
# ---------------------------------------------------------------------------

def setup_claude_code(output_dir: Optional[str] = None):
    """Generate NPC plugin for Claude Code."""
    plugin_dir = Path(output_dir or os.path.join(os.getcwd(), "npc-plugin"))
    print(f"Generating NPC plugin for Claude Code in {plugin_dir}")

    # plugin.json manifest — only name/description/author; dirs auto-discovered
    _write_json(plugin_dir / ".claude-plugin" / "plugin.json", {
        "name": "npc",
        "description": "NPC team integration — agents, tools, and conversation logging via npcpy MCP server",
        "author": {
            "name": "NPC Worldwide",
            "email": "support@npcworldwide.com",
        },
    })

    # MCP server config — flat format (no mcpServers wrapper)
    _write_json(plugin_dir / ".mcp.json", {
        "npc": {
            "command": sys.executable,
            "args": ["-m", "npcpy.mcp_server"],
        }
    })

    # Hooks config
    hook_cmd = f"python3 ${{CLAUDE_PLUGIN_ROOT}}/scripts/npc_log.py"
    hook_entry = [{"hooks": [{"type": "command", "command": hook_cmd, "timeout": 10}]}]
    _write_json(plugin_dir / "hooks" / "hooks.json", {
        "description": "NPC conversation logger — logs all messages to npcsh_history.db",
        "hooks": {
            "UserPromptSubmit": hook_entry,
            "Stop": hook_entry,
            "PostToolUse": hook_entry,
        }
    })

    # Hook script
    hook_script = HOOK_SCRIPT.format(
        tool_name="Claude Code",
        conv_prefix="claude_code",
    )
    _write_file(plugin_dir / "scripts" / "npc_log.py", hook_script, executable=True)

    # /npc:npcs skill
    _write_file(plugin_dir / "skills" / "npcs" / "SKILL.md", NPCS_SKILL)

    print(f"\nDone. Copy plugin to marketplace:")
    print(f"  cp -r {plugin_dir} ~/.claude/plugins/marketplaces/claude-plugins-official/plugins/npc")
    print(f"Then run /reload-plugins in Claude Code.")


# ---------------------------------------------------------------------------
# Codex plugin
# ---------------------------------------------------------------------------

def setup_codex(output_dir: Optional[str] = None):
    """Generate NPC plugin for Codex."""
    plugin_dir = Path(output_dir or os.path.join(os.getcwd(), "npc-plugin-codex"))
    print(f"Generating NPC plugin for Codex in {plugin_dir}")

    _write_json(plugin_dir / ".mcp.json", {
        "npc": {
            "command": sys.executable,
            "args": ["-m", "npcpy.mcp_server"],
        }
    })

    print("\nDone. Copy .mcp.json to your Codex project root.")
    print("Conversation logging requires Codex hook support (TBD).")


# ---------------------------------------------------------------------------
# Gemini plugin
# ---------------------------------------------------------------------------

def setup_gemini(output_dir: Optional[str] = None):
    """Generate NPC plugin for Gemini CLI."""
    plugin_dir = Path(output_dir or os.path.join(os.getcwd(), "npc-plugin-gemini"))
    print(f"Generating NPC plugin for Gemini CLI in {plugin_dir}")

    _write_json(plugin_dir / ".mcp.json", {
        "npc": {
            "command": sys.executable,
            "args": ["-m", "npcpy.mcp_server"],
        }
    })

    print("\nDone. Copy .mcp.json to your Gemini project root.")
    print("Conversation logging requires Gemini hook support (TBD).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

TARGETS = {
    "claude": setup_claude_code,
    "claude-code": setup_claude_code,
    "codex": setup_codex,
    "gemini": setup_gemini,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate NPC plugin for AI coding tools",
        usage="python -m npcpy.plugin_setup <target> [--dir OUTPUT_DIR]",
    )
    parser.add_argument("target", choices=list(TARGETS.keys()))
    parser.add_argument("--dir", type=str, default=None,
                        help="Output directory for plugin (default: ./npc-plugin)")
    args = parser.parse_args()
    TARGETS[args.target](output_dir=args.dir)


if __name__ == "__main__":
    main()
