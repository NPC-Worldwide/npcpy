"""
Register NPC integration with AI coding tools.

Writes config directly to the tool's config directory — no intermediate files.

Usage:
    python -m npcpy.plugin_setup claude
    python -m npcpy.plugin_setup codex
    python -m npcpy.plugin_setup gemini
    python -m npcpy.plugin_setup --uninstall claude
"""

import os
import sys
import json
import stat
import shutil
from pathlib import Path
from typing import Optional


NPCS_SKILL = """---
name: npcs
description: List available NPCs and switch the active NPC. Use when the user asks to "list NPCs", "switch NPC", "show team", or "change agent".
---

# NPC Team Management

Use the MCP prompts to see available NPCs and switch:

1. Use the `npc_team` MCP prompt to see all available NPCs
2. Use the `npc_<name>` MCP prompt to switch to that NPC

When switching, the available tools change to match the new NPC's jinxes.

If called with arguments: switch directly to the NPC named "$ARGUMENTS" by using the `npc_$ARGUMENTS` MCP prompt.
"""


# ---------------------------------------------------------------------------
# Direct registration
# ---------------------------------------------------------------------------

def _write(path: Path, content: str, executable: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def setup_claude(uninstall: bool = False):
    """Register/unregister NPC plugin directly with Claude Code."""
    plugin_dir = Path.home() / ".claude" / "plugins" / "marketplaces" / "claude-plugins-official" / "plugins" / "npc"

    if uninstall:
        if plugin_dir.exists():
            shutil.rmtree(plugin_dir)
            print(f"Removed {plugin_dir}")
        # Also remove from settings.json mcpServers
        settings_path = Path.home() / ".claude" / "settings.json"
        if settings_path.exists():
            try:
                data = json.loads(settings_path.read_text())
                if "mcpServers" in data and "npc" in data["mcpServers"]:
                    del data["mcpServers"]["npc"]
                    settings_path.write_text(json.dumps(data, indent=2) + "\n")
                    print("Removed NPC from Claude settings.json mcpServers.")
            except Exception:
                pass
        return

    print(f"Installing NPC plugin to {plugin_dir}")

    # Plugin manifest
    _write(plugin_dir / ".claude-plugin" / "plugin.json", json.dumps({
        "name": "npc",
        "description": "NPC team integration — agents, tools, and conversation logging via npcpy MCP server",
        "author": {"name": "NPC Worldwide", "email": "support@npcworldwide.com"},
    }, indent=2) + "\n")

    mcp_args = ["-m", "npcpy.mcp_server"]

    _write(plugin_dir / ".mcp.json", json.dumps({
        "npc": {
            "command": sys.executable,
            "args": mcp_args,
        }
    }, indent=2) + "\n")

    # Skill
    _write(plugin_dir / "skills" / "npcs" / "SKILL.md", NPCS_SKILL)

    # Also write to settings.json mcpServers for direct MCP access
    settings_path = Path.home() / ".claude" / "settings.json"
    try:
        data = {}
        if settings_path.exists():
            data = json.loads(settings_path.read_text())
        if "mcpServers" not in data:
            data["mcpServers"] = {}
        data["mcpServers"]["npc"] = {
            "command": sys.executable,
            "args": mcp_args,
        }
        settings_path.write_text(json.dumps(data, indent=2) + "\n")
        print(f"Added NPC to {settings_path} mcpServers.")
    except Exception as e:
        print(f"Warning: could not update settings.json: {e}")

    print("Done. Run /plugins in Claude Code to verify.")


def setup_codex(uninstall: bool = False):
    """Write MCP config for Codex."""
    mcp_path = Path.home() / ".codex" / ".mcp.json"

    if uninstall:
        if mcp_path.exists():
            # Remove just the npc entry, not the whole file
            data = json.loads(mcp_path.read_text())
            data.pop("npc", None)
            if data:
                mcp_path.write_text(json.dumps(data, indent=2) + "\n")
            else:
                mcp_path.unlink()
            print("Removed NPC from Codex MCP config.")
        return

    # Merge into existing .mcp.json if it exists
    data = {}
    if mcp_path.exists():
        try:
            data = json.loads(mcp_path.read_text())
        except json.JSONDecodeError:
            pass

    data["npc"] = {
        "command": sys.executable,
        "args": ["-m", "npcpy.mcp_server"],
    }
    _write(mcp_path, json.dumps(data, indent=2) + "\n")
    print(f"Wrote NPC MCP config to {mcp_path}")


def setup_gemini(uninstall: bool = False):
    """Write MCP config for Gemini CLI."""
    mcp_path = Path.home() / ".gemini" / ".mcp.json"

    if uninstall:
        if mcp_path.exists():
            data = json.loads(mcp_path.read_text())
            data.pop("npc", None)
            if data:
                mcp_path.write_text(json.dumps(data, indent=2) + "\n")
            else:
                mcp_path.unlink()
            print("Removed NPC from Gemini MCP config.")
        return

    data = {}
    if mcp_path.exists():
        try:
            data = json.loads(mcp_path.read_text())
        except json.JSONDecodeError:
            pass

    data["npc"] = {
        "command": sys.executable,
        "args": ["-m", "npcpy.mcp_server"],
    }
    _write(mcp_path, json.dumps(data, indent=2) + "\n")
    print(f"Wrote NPC MCP config to {mcp_path}")


TARGETS = {
    "claude": setup_claude,
    "claude-code": setup_claude,
    "codex": setup_codex,
    "gemini": setup_gemini,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Register NPC with AI coding tools",
        usage="python -m npcpy.plugin_setup <target> [--uninstall]",
    )
    parser.add_argument("target", choices=list(TARGETS.keys()))
    parser.add_argument("--uninstall", action="store_true", help="Remove NPC integration")
    args = parser.parse_args()
    TARGETS[args.target](uninstall=args.uninstall)


if __name__ == "__main__":
    main()
