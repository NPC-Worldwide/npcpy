"""
NPC-governed MCP server.

Only jinxes are exposed as tools. NPC switching happens via MCP prompts,
which swap the tool list and notify the client.

Usage:
    python -m npcpy.mcp_server                        # auto-discover team
    python -m npcpy.mcp_server --npc ledbi             # start as specific NPC
    python -m npcpy.mcp_server --team /path/to/team    # explicit team path
"""

import os
import sys
import json
import time
import argparse
from typing import Optional


def discover_team_path(explicit_path: Optional[str] = None) -> str:
    """Find the team directory: explicit > cwd/npc_team > ~/.npcsh/npc_team."""
    if explicit_path and os.path.isdir(explicit_path):
        return os.path.abspath(explicit_path)

    cwd_team = os.path.join(os.getcwd(), "npc_team")
    if os.path.isdir(cwd_team):
        return cwd_team

    global_team = os.path.expanduser("~/.npcsh/npc_team")
    if os.path.isdir(global_team):
        return global_team

    raise FileNotFoundError(
        "No npc_team found. Checked: ./npc_team, ~/.npcsh/npc_team"
    )


class NPCServerState:
    """Holds the loaded team, all NPCs, and tracks the active NPC."""

    def __init__(self, team_path: str, npc_name: Optional[str] = None,
                 db_path: Optional[str] = None):
        from npcpy.npc_compiler import Team
        from npcpy.memory.command_history import CommandHistory

        self.team_path = team_path
        self.mcp = None  # set by build_server

        # Resolve DB path
        if db_path is None:
            try:
                from npcsh._state import NPCSH_DB_PATH
                db_path = NPCSH_DB_PATH
            except ImportError:
                db_path = os.path.expanduser("~/.npcsh/npcsh_history.db")

        self.command_history = CommandHistory(db=db_path)

        print(f"[npc-mcp] Loading team from {team_path}", file=sys.stderr)
        self.team = Team(
            team_path=team_path,
            db_conn=self.command_history.engine,
        )

        # Resolve active NPC: explicit arg > state file > forenpc > first
        if npc_name and npc_name in self.team.npcs:
            self.active_npc = self.team.npcs[npc_name]
        else:
            if self.team.forenpc:
                self.active_npc = self.team.forenpc
            elif self.team.npcs:
                self.active_npc = next(iter(self.team.npcs.values()))
            else:
                self.active_npc = None

        # Track which jinx tool names are currently registered
        self._registered_jinx_names = set()

        if self.active_npc:
            print(f"[npc-mcp] Active NPC: {self.active_npc.name}", file=sys.stderr)
        else:
            print("[npc-mcp] WARNING: No NPC loaded", file=sys.stderr)

        print(f"[npc-mcp] NPCs: {list(self.team.npcs.keys())}", file=sys.stderr)

    def get_system_prompt_text(self) -> str:
        """Build the personality prompt for the active NPC."""
        npc = self.active_npc
        if not npc:
            return "No active NPC"
        directive = npc.primary_directive or ""
        tools = list(npc.jinxes_dict.keys())
        other_npcs = {
            n: list(obj.jinxes_dict.keys())
            for n, obj in self.team.npcs.items()
            if n != npc.name
        }
        prompt = f"You are {npc.name}.\n\n{directive}\n\n"
        prompt += f"Your tools: {tools}\n"
        if "delegate" in npc.jinxes_dict:
            prompt += "Use 'delegate' for tools you don't have.\n"
        prompt += "\nOther NPCs:\n"
        for n, t in other_npcs.items():
            prompt += f"  @{n}: {t}\n"
        return prompt

    def is_authorized(self, tool_name: str) -> bool:
        if not self.active_npc:
            return False
        return tool_name in self.active_npc.jinxes_dict

    def get_rejection_message(self, tool_name: str) -> str:
        npc = self.active_npc
        name = npc.name if npc else "unknown"
        capable_npcs = [
            n for n, obj in self.team.npcs.items()
            if tool_name in obj.jinxes_dict
        ]
        msg = f"Tool '{tool_name}' not permitted for NPC '{name}'. "
        if capable_npcs:
            msg += f"NPCs with '{tool_name}': {capable_npcs}. "
            if npc and "delegate" in npc.jinxes_dict:
                msg += "Use 'delegate' to have one of them run it."
        return msg

    def execute_jinx(self, tool_name: str, args: dict) -> str:
        npc = self.active_npc
        if not self.is_authorized(tool_name):
            return self.get_rejection_message(tool_name)

        jinx = npc.jinxes_dict.get(tool_name) if npc else None
        if not jinx:
            # Check team-level jinxes as fallback
            jinx = self.team.jinxes_dict.get(tool_name)
        if not jinx:
            return f"Jinx '{tool_name}' not found."

        try:
            start = time.time()
            result = jinx.execute(
                input_values=args,
                npc=npc,
                jinja_env=getattr(npc, 'jinja_env', None),
            )
            duration_ms = int((time.time() - start) * 1000)

            if isinstance(result, dict):
                output = str(result.get("output", "")) or json.dumps(result, default=str)
            else:
                output = str(result)

            try:
                from npcpy.memory.command_history import generate_message_id
                self.command_history.save_jinx_execution(
                    triggering_message_id=generate_message_id(),
                    conversation_id="mcp",
                    npc_name=npc.name if npc else None,
                    jinx_name=tool_name,
                    jinx_inputs=args,
                    jinx_output=output,
                    status="success",
                    team_name=getattr(self.team, "name", None),
                    duration_ms=duration_ms,
                )
            except Exception as log_err:
                print(f"[npc-mcp] History log error: {log_err}", file=sys.stderr)

            return output
        except Exception as e:
            try:
                from npcpy.memory.command_history import generate_message_id
                self.command_history.save_jinx_execution(
                    triggering_message_id=generate_message_id(),
                    conversation_id="mcp",
                    npc_name=npc.name if npc else None,
                    jinx_name=tool_name,
                    jinx_inputs=args,
                    jinx_output=None,
                    status="error",
                    team_name=getattr(self.team, "name", None),
                    error_message=str(e),
                )
            except Exception:
                pass
            return f"Jinx execution error: {e}"

    async def switch_npc(self, npc_name: str, ctx=None) -> str:
        """Switch active NPC, swap tools, notify client."""
        if npc_name not in self.team.npcs:
            return f"NPC '{npc_name}' not found. Available: {list(self.team.npcs.keys())}"

        self.active_npc = self.team.npcs[npc_name]

        # Swap jinx tools
        if self.mcp:
            # Remove old jinx tools
            for name in list(self._registered_jinx_names):
                try:
                    self.mcp.remove_tool(name)
                except Exception:
                    pass
            self._registered_jinx_names.clear()

            # Register new NPC's jinxes
            _register_jinxes(self.mcp, self)

            # Notify client to re-fetch tool list
            if ctx:
                try:
                    session = ctx.request_context.session
                    await session.send_tool_list_changed()
                except Exception as e:
                    print(f"[npc-mcp] tool list notify error: {e}", file=sys.stderr)

        print(f"[npc-mcp] Switched to {npc_name}", file=sys.stderr)
        return self.get_system_prompt_text()

    def _resolve_npc(self, npc_name: Optional[str] = None):
        if npc_name and npc_name in self.team.npcs:
            return self.team.npcs[npc_name]
        return self.active_npc


def _make_jinx_handler(state, jname, params, desc):
    """Create an async handler function for a jinx."""
    if not params:
        async def handler(input: str = "") -> str:
            return state.execute_jinx(jname, {"input": input})
    elif len(params) == 1:
        pname = list(params.keys())[0]
        async def handler(input: str = "") -> str:
            return state.execute_jinx(jname, {pname: input})
        desc = f"{desc}\n\nArgs:\n    input: {params[pname]}"
    else:
        param_doc = "\n".join(f"    {k}: {v}" for k, v in params.items())
        async def handler(kwargs_json: str = "{}") -> str:
            try:
                args = json.loads(kwargs_json) if kwargs_json else {}
            except json.JSONDecodeError:
                args = {"input": kwargs_json}
            return state.execute_jinx(jname, args)
        desc = f"{desc}\n\nPass arguments as JSON:\n{param_doc}"
    handler.__name__ = jname
    handler.__doc__ = desc
    return handler


def _register_jinxes(mcp, state):
    """Register the active NPC's jinxes as MCP tools."""
    if not state.active_npc:
        return

    for jinx_name, jinx_obj in state.active_npc.jinxes_dict.items():
        _desc = jinx_obj.description or f"Jinx: {jinx_name}"

        params = {}
        for inp in (jinx_obj.inputs or []):
            if isinstance(inp, str):
                params[inp] = f"Parameter: {inp}"
            elif isinstance(inp, dict):
                pname = list(inp.keys())[0]
                default = inp.get(pname, "")
                params[pname] = f"Parameter: {pname} (default: {default})"

        handler = _make_jinx_handler(state, jinx_name, params, _desc)
        try:
            mcp.add_tool(handler)
            state._registered_jinx_names.add(jinx_name)
        except Exception as e:
            print(f"[npc-mcp] Failed to register {jinx_name}: {e}", file=sys.stderr)


def build_server(state: NPCServerState):
    """Build and return a FastMCP server from an NPCServerState."""
    from mcp.server.fastmcp import FastMCP, Context

    mcp = FastMCP("npcsh_mcp")
    state.mcp = mcp

    # Register active NPC's jinxes
    _register_jinxes(mcp, state)

    # -- MCP Prompts --

    # Per-NPC prompts — selecting one switches NPC and swaps tools
    for npc_name in state.team.npcs:
        def make_npc_prompt(name):
            @mcp.prompt(name=f"npc_{name}")
            async def npc_prompt(ctx: Context) -> str:
                result = await state.switch_npc(name, ctx=ctx)
                return result
            npc_prompt.__name__ = f"npc_{name}_prompt"
            npc_prompt.__doc__ = f"Switch to {name} and adopt their personality and tools."
            return npc_prompt
        make_npc_prompt(npc_name)

    @mcp.prompt(name="npc_team")
    async def team_prompt() -> str:
        """Overview of all NPCs in the team."""
        lines = []
        for name, npc_obj in state.team.npcs.items():
            active = " (active)" if npc_obj == state.active_npc else ""
            directive = (npc_obj.primary_directive or "").strip().split("\n")[0][:120]
            tools = list(npc_obj.jinxes_dict.keys())
            lines.append(f"@{name}{active}: {directive}\n  tools: {tools}")
        return "\n\n".join(lines)

    @mcp.prompt(name="npc_bootstrap")
    async def bootstrap_prompt() -> str:
        """Active NPC personality and team roster."""
        parts = [state.get_system_prompt_text()]
        parts.append("\n--- Team Roster ---")
        for name, npc_obj in state.team.npcs.items():
            active = " (active)" if npc_obj == state.active_npc else ""
            directive = (npc_obj.primary_directive or "").strip().split("\n")[0][:120]
            parts.append(f"  @{name}{active}: {directive}")
        return "\n".join(parts)

    # -- Resources --

    @mcp.resource("npc://system_prompt")
    async def get_system_prompt() -> str:
        return state.get_system_prompt_text()

    @mcp.resource("npc://team_context")
    async def get_team_context() -> str:
        if hasattr(state.team, "team_ctx") and state.team.team_ctx:
            return json.dumps(state.team.team_ctx, indent=2, default=str)
        ctx = getattr(state.team, "context", None)
        return ctx if ctx else "No team context loaded"

    return mcp


def main():
    parser = argparse.ArgumentParser(description="NPC-governed MCP server")
    parser.add_argument("--npc", type=str, default=None, help="NPC to start as")
    parser.add_argument("--team", type=str, default=None, help="Path to npc_team directory")
    args = parser.parse_args()

    team_path = discover_team_path(args.team)
    state = NPCServerState(team_path=team_path, npc_name=args.npc)
    mcp = build_server(state)

    npc_label = state.active_npc.name if state.active_npc else "none"
    print(f"[npc-mcp] Starting as '{npc_label}' from {team_path}", file=sys.stderr)
    print(f"[npc-mcp] Tools: {list(state._registered_jinx_names)}", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
