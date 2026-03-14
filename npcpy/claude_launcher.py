"""
Launch Claude Code as an NPC from your team.

Shows an interactive picker, then launches Claude with the selected
NPC's system prompt and all team agents available for delegation.

Usage:
    python -m npcpy.claude_launcher                   # interactive picker
    python -m npcpy.claude_launcher --npc corca       # skip picker
    python -m npcpy.claude_launcher --team /path      # explicit team path
"""

import os
import sys
import json
import argparse
from typing import Optional


def _discover_team_path(explicit: Optional[str] = None) -> str:
    if explicit and os.path.isdir(explicit):
        return os.path.abspath(explicit)
    cwd_team = os.path.join(os.getcwd(), "npc_team")
    if os.path.isdir(cwd_team):
        return cwd_team
    global_team = os.path.expanduser("~/.npcsh/npc_team")
    if os.path.isdir(global_team):
        return global_team
    raise FileNotFoundError("No npc_team found. Checked: ./npc_team, ~/.npcsh/npc_team")


def _load_team(team_path: str):
    """Load team and return dict of {name: {directive, tools, first_line}}."""
    from npcpy.npc_compiler import Team
    team = Team(team_path=team_path)
    npcs = {}
    for name, npc_obj in team.npcs.items():
        directive = (npc_obj.primary_directive or "").strip()
        if not directive:
            continue
        npcs[name] = {
            "directive": directive,
            "tools": list(npc_obj.jinxes_dict.keys()),
            "first_line": directive.split("\n")[0].strip(),
        }
    return npcs


def _pick_npc(npcs: dict) -> str:
    """Interactive terminal picker for NPC selection."""
    names = list(npcs.keys())

    # Try fzf first
    try:
        import shutil
        if shutil.which("fzf"):
            from subprocess import run, PIPE
            preview = "\n".join(f"{n}  —  {npcs[n]['first_line']}" for n in names)
            result = run(
                ["fzf", "--height=~20", "--layout=reverse",
                 "--header=Select NPC (↑↓ to navigate, Enter to select)"],
                input=preview, capture_output=True, text=True,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("  —")[0].strip()
    except Exception:
        pass

    # Fallback: simple numbered menu
    print("\n  NPC Team\n")
    for i, name in enumerate(names, 1):
        info = npcs[name]
        print(f"  {i}. {name}  —  {info['first_line']}")
    print()

    while True:
        try:
            choice = input("  Select NPC [1-{}]: ".format(len(names))).strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(names):
                    return names[idx]
            elif choice in names:
                return choice
            print(f"  Invalid choice. Enter 1-{len(names)} or NPC name.")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)


def launch(team_path: Optional[str] = None, npc_name: Optional[str] = None,
           extra_args: Optional[list] = None):
    """Launch claude CLI as selected NPC with team agents."""
    tp = _discover_team_path(team_path)
    npcs = _load_team(tp)

    if not npcs:
        print("No NPCs found in team", file=sys.stderr)
        sys.exit(1)

    # Pick NPC if not specified
    if not npc_name:
        npc_name = _pick_npc(npcs)

    if npc_name not in npcs:
        print(f"NPC '{npc_name}' not found. Available: {list(npcs.keys())}", file=sys.stderr)
        sys.exit(1)

    selected = npcs[npc_name]
    print(f"\n  Starting as {npc_name}\n")

    # Write state file so MCP server starts with the right NPC
    state_dir = os.path.expanduser("~/.npcsh")
    os.makedirs(state_dir, exist_ok=True)
    state_file = os.path.join(state_dir, ".active_npc_state.json")
    try:
        with open(state_file, "w") as f:
            json.dump({
                "name": npc_name,
                "directive": selected["directive"],
                "tools": selected["tools"],
                "team_npcs": list(npcs.keys()),
            }, f)
    except Exception:
        pass

    # Build --agents JSON for delegation to other NPCs
    agents = {}
    for name, info in npcs.items():
        agents[name] = {
            "description": info["first_line"],
            "prompt": info["directive"],
        }

    cmd = [
        "claude",
        "--system-prompt", selected["directive"],
        "--agents", json.dumps(agents),
    ]

    if extra_args:
        cmd.extend(extra_args)

    os.execvp("claude", cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Launch Claude Code as an NPC",
    )
    parser.add_argument("--team", type=str, default=None,
                        help="Path to npc_team directory")
    parser.add_argument("--npc", type=str, default=None,
                        help="Start as specific NPC (skip picker)")
    args, extra = parser.parse_known_args()

    launch(
        team_path=args.team,
        npc_name=args.npc,
        extra_args=extra if extra else None,
    )


if __name__ == "__main__":
    main()
