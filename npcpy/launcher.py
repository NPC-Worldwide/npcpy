"""
Launch AI coding tools as an NPC from your team.

Supports: Claude Code, Codex, Gemini CLI, OpenCode, Aider, Amp

Usage:
    python -m npcpy.claude_launcher                        # Claude Code (default)
    python -m npcpy.claude_launcher --tool codex
    python -m npcpy.claude_launcher --tool gemini
    python -m npcpy.claude_launcher --tool opencode
    python -m npcpy.claude_launcher --tool aider
    python -m npcpy.claude_launcher --tool amp
    python -m npcpy.claude_launcher --npc corca             # skip picker
    python -m npcpy.claude_launcher --team /path            # explicit team path
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
    names = list(npcs.keys())

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

    print("\n  NPC Team\n")
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}  —  {npcs[name]['first_line']}")
    print()

    while True:
        try:
            choice = input(f"  Select NPC [1-{len(names)}]: ").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(names):
                    return names[idx]
            elif choice in names:
                return choice
            print(f"  Invalid choice.")
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)


def _write_npc_state(npc_name: str, selected: dict, npcs: dict):
    """Write state file so MCP server and hooks know the active NPC."""
    state_dir = os.path.expanduser("~/.npcsh")
    os.makedirs(state_dir, exist_ok=True)
    try:
        with open(os.path.join(state_dir, ".active_npc_state.json"), "w") as f:
            json.dump({
                "name": npc_name,
                "directive": selected["directive"],
                "tools": selected["tools"],
                "team_npcs": list(npcs.keys()),
            }, f)
    except Exception:
        pass


def _build_system_prompt(npc_name: str, selected: dict, npcs: dict) -> str:
    """Build the full system prompt for the NPC."""
    parts = [f"You are {npc_name}.\n\n{selected['directive']}"]
    parts.append(f"\nYour tools: {selected['tools']}")
    other = {n: info['first_line'] for n, info in npcs.items() if n != npc_name}
    if other:
        parts.append("\nOther NPCs on the team:")
        for n, desc in other.items():
            parts.append(f"  @{n}: {desc}")
    return "\n".join(parts)


# ── Launchers ──

def launch_claude(npc_name, selected, npcs, extra_args):
    agents = {}
    for name, info in npcs.items():
        agents[name] = {"description": info["first_line"], "prompt": info["directive"]}

    cmd = ["claude", "--system-prompt", selected["directive"], "--agents", json.dumps(agents)]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Starting Claude Code as {npc_name}")
    os.execvp("claude", cmd)


def launch_codex(npc_name, selected, npcs, extra_args):
    prompt = _build_system_prompt(npc_name, selected, npcs)
    cmd = ["codex", "--system-prompt", prompt]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Starting Codex as {npc_name}")
    os.execvp("codex", cmd)


def launch_gemini(npc_name, selected, npcs, extra_args):
    import tempfile
    prompt = _build_system_prompt(npc_name, selected, npcs)

    # Gemini CLI uses GEMINI_SYSTEM_MD env var pointing to a markdown file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, prefix=f'npc_{npc_name}_') as f:
        f.write(prompt)
        prompt_file = f.name

    os.environ["GEMINI_SYSTEM_MD"] = prompt_file
    cmd = ["gemini"]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Starting Gemini CLI as {npc_name}")
    print(f"  (system prompt: {prompt_file})")
    os.execvp("gemini", cmd)


def launch_opencode(npc_name, selected, npcs, extra_args):
    # OpenCode uses AGENT.md or config.json agents
    prompt = _build_system_prompt(npc_name, selected, npcs)
    agent_md = os.path.join(os.getcwd(), "AGENT.md")

    # Write/update AGENT.md in current directory
    with open(agent_md, "w") as f:
        f.write(prompt)

    cmd = ["opencode"]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Starting OpenCode as {npc_name}")
    print(f"  (wrote {agent_md})")
    os.execvp("opencode", cmd)


def launch_aider(npc_name, selected, npcs, extra_args):
    import tempfile
    prompt = _build_system_prompt(npc_name, selected, npcs)

    # Aider uses --read for static context files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, prefix=f'npc_{npc_name}_') as f:
        f.write(prompt)
        prompt_file = f.name

    cmd = ["aider", "--read", prompt_file]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Starting Aider as {npc_name}")
    os.execvp("aider", cmd)


def launch_amp(npc_name, selected, npcs, extra_args):
    # Amp uses AGENT.md in project root
    prompt = _build_system_prompt(npc_name, selected, npcs)
    agent_md = os.path.join(os.getcwd(), "AGENT.md")

    with open(agent_md, "w") as f:
        f.write(prompt)

    cmd = ["amp"]
    if extra_args:
        cmd.extend(extra_args)
    print(f"  Starting Amp as {npc_name}")
    print(f"  (wrote {agent_md})")
    os.execvp("amp", cmd)


TOOLS = {
    "claude": launch_claude,
    "claude-code": launch_claude,
    "codex": launch_codex,
    "gemini": launch_gemini,
    "opencode": launch_opencode,
    "aider": launch_aider,
    "amp": launch_amp,
}


def launch(tool: str = "claude", team_path: Optional[str] = None,
           npc_name: Optional[str] = None, extra_args: Optional[list] = None):
    tp = _discover_team_path(team_path)
    npcs = _load_team(tp)

    if not npcs:
        print("No NPCs found in team", file=sys.stderr)
        sys.exit(1)

    if not npc_name:
        npc_name = _pick_npc(npcs)

    if npc_name not in npcs:
        print(f"NPC '{npc_name}' not found. Available: {list(npcs.keys())}", file=sys.stderr)
        sys.exit(1)

    selected = npcs[npc_name]
    _write_npc_state(npc_name, selected, npcs)

    launcher = TOOLS.get(tool)
    if not launcher:
        print(f"Unknown tool: {tool}. Available: {list(TOOLS.keys())}", file=sys.stderr)
        sys.exit(1)

    launcher(npc_name, selected, npcs, extra_args)


def main():
    # Auto-detect tool from how we were invoked (npc-claude, npc-codex, etc.)
    invoked = os.path.basename(sys.argv[0])
    default_tool = invoked.replace("npc-", "") if invoked.startswith("npc-") else "claude"
    if default_tool not in TOOLS:
        default_tool = "claude"

    parser = argparse.ArgumentParser(description="Launch an AI coding tool as an NPC")
    parser.add_argument("--tool", "-t", type=str, default=default_tool,
                        choices=list(TOOLS.keys()), help="Which tool to launch")
    parser.add_argument("--team", type=str, default=None)
    parser.add_argument("--npc", type=str, default=None)
    args, extra = parser.parse_known_args()

    launch(tool=args.tool, team_path=args.team, npc_name=args.npc,
           extra_args=extra if extra else None)


if __name__ == "__main__":
    main()
