"""
NPC-governed MCP server.

Any NPC team can be served as an MCP server — it's just the data layer.
Point it at a team directory, pick an NPC, and go.

Usage:
    python -m npcpy.mcp_server                        # auto-discover team
    python -m npcpy.mcp_server --npc ledbi             # start as specific NPC
    python -m npcpy.mcp_server --team /path/to/team    # explicit team path
"""

import os
import sys
import json
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

        # Resolve DB path — try npcsh first, fall back to default
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

        # Resolve active NPC
        if npc_name and npc_name in self.team.npcs:
            self.active_npc = self.team.npcs[npc_name]
        elif self.team.forenpc:
            self.active_npc = self.team.forenpc
        elif self.team.npcs:
            self.active_npc = next(iter(self.team.npcs.values()))
        else:
            self.active_npc = None

        if self.active_npc:
            print(f"[npc-mcp] Active NPC: {self.active_npc.name}", file=sys.stderr)
            print(f"[npc-mcp] Jinxes: {list(self.active_npc.jinxes_dict.keys())}", file=sys.stderr)
        else:
            print("[npc-mcp] WARNING: No NPC loaded", file=sys.stderr)

        # Only register the active NPC's jinxes — not the union of all NPCs
        self.all_jinxes = {}
        if self.active_npc:
            self.all_jinxes = dict(self.active_npc.jinxes_dict)

        print(f"[npc-mcp] Total jinxes: {len(self.all_jinxes)}", file=sys.stderr)
        print(f"[npc-mcp] NPCs: {list(self.team.npcs.keys())}", file=sys.stderr)

    def is_authorized(self, tool_name: str, npc_name: Optional[str] = None) -> bool:
        npc = self._resolve_npc(npc_name)
        if not npc:
            return False
        if tool_name in ("set_npc", "list_npcs", "whoami",
                         "add_memory", "search_memory",
                         "query_npcsh_database"):
            return True
        return tool_name in npc.jinxes_dict

    def get_rejection_message(self, tool_name: str, npc_name: Optional[str] = None) -> str:
        npc = self._resolve_npc(npc_name)
        name = npc.name if npc else "unknown"
        available = list(npc.jinxes_dict.keys()) if npc else []
        capable_npcs = [
            n for n, obj in self.team.npcs.items()
            if tool_name in obj.jinxes_dict
        ]
        msg = f"Tool '{tool_name}' not permitted for NPC '{name}'. "
        msg += f"Available: {available}. "
        if capable_npcs:
            msg += f"NPCs with '{tool_name}': {capable_npcs}. "
            if "delegate" in (npc.jinxes_dict if npc else {}):
                msg += "Use 'delegate' to have one of them run it."
        return msg

    def execute_jinx(self, tool_name: str, args: dict, npc_name: Optional[str] = None) -> str:
        npc = self._resolve_npc(npc_name)
        if not self.is_authorized(tool_name, npc_name):
            return self.get_rejection_message(tool_name, npc_name)

        jinx = self.all_jinxes.get(tool_name)
        if not jinx:
            return f"Jinx '{tool_name}' not found in team."

        try:
            result = jinx.execute(
                input_values=args,
                npc=npc,
                jinja_env=getattr(npc, 'jinja_env', None),
            )
            if isinstance(result, dict):
                return str(result.get("output", "")) or json.dumps(result, default=str)
            return str(result)
        except Exception as e:
            return f"Jinx execution error: {e}"

    def switch_npc(self, npc_name: str) -> str:
        if npc_name not in self.team.npcs:
            return f"NPC '{npc_name}' not found. Available: {list(self.team.npcs.keys())}"
        self.active_npc = self.team.npcs[npc_name]
        return f"Switched to {npc_name}. Tools: {list(self.active_npc.jinxes_dict.keys())}"

    def _resolve_npc(self, npc_name: Optional[str] = None):
        if npc_name and npc_name in self.team.npcs:
            return self.team.npcs[npc_name]
        return self.active_npc


def build_server(state: NPCServerState):
    """Build and return a FastMCP server from an NPCServerState."""
    from mcp.server.fastmcp import FastMCP
    from sqlalchemy import text

    mcp = FastMCP("npcsh_mcp")

    # -- Core governance tools --

    @mcp.tool()
    async def set_npc(npc_name: str) -> str:
        """Switch the active NPC. Changes which tools are available."""
        return state.switch_npc(npc_name)

    @mcp.tool()
    async def list_npcs() -> str:
        """List all available NPCs and their permitted tools."""
        result = {}
        for name, npc_obj in state.team.npcs.items():
            jinxes = list(npc_obj.jinxes_dict.keys())
            result[name] = {
                "tools": jinxes,
                "active": (npc_obj == state.active_npc),
                "directive": (npc_obj.primary_directive or "")[:200],
            }
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def whoami() -> str:
        """Show the active NPC, its tools, and its directive."""
        npc = state.active_npc
        if not npc:
            return "No active NPC"
        return json.dumps({
            "name": npc.name,
            "tools": list(npc.jinxes_dict.keys()),
            "directive": npc.primary_directive or "",
        }, indent=2)

    # -- Data-layer tools --

    @mcp.tool()
    async def add_memory(
        content: str,
        memory_type: str = "observation",
        npc_name: str = None,
        directory_path: str = None,
    ) -> str:
        """Add a memory for the active NPC."""
        npc = state._resolve_npc(npc_name)
        npc_label = npc.name if npc else "unknown"
        team_label = getattr(state.team, "name", "default_team")
        if directory_path is None:
            directory_path = os.getcwd()
        try:
            from npcpy.memory.command_history import generate_message_id
            mid = generate_message_id()
            memory_id = state.command_history.add_memory_to_database(
                message_id=mid, conversation_id="mcp_direct",
                npc=npc_label, team=team_label,
                directory_path=directory_path,
                initial_memory=content, status="active",
                model=None, provider=None,
            )
            return f"Memory created: {memory_id}"
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def search_memory(
        query: str, npc_name: str = None,
        directory_path: str = None, limit: int = 10,
    ) -> str:
        """Search memories. Scoped to the active NPC by default."""
        npc = state._resolve_npc(npc_name)
        try:
            results = state.command_history.search_memory(
                query=query, npc=npc.name if npc else None,
                team=getattr(state.team, "name", None),
                directory_path=directory_path or os.getcwd(),
                limit=limit,
            )
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error: {e}"

    @mcp.tool()
    async def query_npcsh_database(sql_query: str) -> str:
        """Execute a read-only SQL query against npcsh_history.db."""
        if not sql_query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries allowed"
        try:
            with state.command_history.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                if not rows:
                    return "No results"
                columns = result.keys()
                return json.dumps(
                    [dict(zip(columns, row)) for row in rows],
                    indent=2, default=str,
                )
        except Exception as e:
            return f"Database error: {e}"

    # -- Register jinxes as tools --

    for jinx_name, jinx_obj in state.all_jinxes.items():
        _name = jinx_name
        _desc = jinx_obj.description or f"Jinx: {jinx_name}"

        params = {}
        for inp in (jinx_obj.inputs or []):
            if isinstance(inp, str):
                params[inp] = f"Parameter: {inp}"
            elif isinstance(inp, dict):
                pname = list(inp.keys())[0]
                default = inp.get(pname, "")
                params[pname] = f"Parameter: {pname} (default: {default})"

        def make_handler(jname, params, desc):
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

        handler = make_handler(_name, params, _desc)
        try:
            mcp.tool()(handler)
        except Exception as e:
            print(f"[npc-mcp] Failed to register {_name}: {e}", file=sys.stderr)

    # -- Resources --

    @mcp.resource("npc://system_prompt")
    async def get_system_prompt() -> str:
        npc = state.active_npc
        if not npc:
            return "No active NPC"
        directive = npc.primary_directive or ""
        tools = list(npc.jinxes_dict.keys())
        other_npcs = {
            n: list(obj.jinxes_dict.keys())
            for n, obj in state.team.npcs.items()
            if n != npc.name
        }
        prompt = f"You are {npc.name}.\n\n{directive}\n\n"
        prompt += f"Your tools: {tools}\n"
        prompt += "You can ONLY use these tools. "
        if "delegate" in npc.jinxes_dict:
            prompt += "Use 'delegate' for tools you don't have.\n"
        else:
            prompt += "Tell the user which NPC can handle tools you don't have.\n"
        prompt += "\nOther NPCs:\n"
        for n, t in other_npcs.items():
            prompt += f"  @{n}: {t}\n"
        prompt += "\nAlways available: set_npc, list_npcs, whoami, add_memory, search_memory, query_npcsh_database\n"
        return prompt

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
    print(f"[npc-mcp] Tools: {list(state.all_jinxes.keys())}", file=sys.stderr)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
