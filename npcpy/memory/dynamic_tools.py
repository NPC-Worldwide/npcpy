import json
import logging
import os
from typing import Any, Dict, List, Optional

from npcpy.npc_compiler import compile_npc_script

logger = logging.getLogger(__name__)


class DynamicTool:
    """A single dynamic tool created by an LLM during a session."""

    def __init__(
        self,
        tool_id: int,
        name: str,
        version: str,
        description: str,
        jinja_template: str,
        parameter_schema: dict,
        output_schema: dict,
        reasoning: str,
        session_id: Optional[str],
        workspace_id: Optional[str],
        created_at: str,
        updated_at: str,
    ):
        self.tool_id = tool_id
        self.name = name
        self.version = version
        self.description = description
        self.jinja_template = jinja_template
        self.parameter_schema = parameter_schema
        self.output_schema = output_schema
        self.reasoning = reasoning
        self.session_id = session_id
        self.workspace_id = workspace_id
        self.created_at = created_at
        self.updated_at = updated_at

        self._compiled = None

    @property
    def compiled_template(self):
        if self._compiled is None:
            self._compiled = compile_npc_script(self.jinja_template)
        return self._compiled

    def to_schema(self) -> dict:
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "parameter_schema": self.parameter_schema,
            "output_schema": self.output_schema,
            "reasoning": self.reasoning,
            "session_id": self.session_id,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def execute(self, env: dict, parameters: dict, memory=None) -> dict:
        """Render the template with the given parameters."""
        merged_env = dict(env)
        merged_env["input"] = parameters
        if memory is not None:
            merged_env["memory"] = memory
        return self.compiled_template(merged_env)


class DynamicToolManager:
    """
    Manages creation, retrieval, versioning, and execution of
    LLM-generated tools (dynamic jinx templates).

    Tools are stored in the command_history database and can be
    scoped to a specific session, a workspace, or promoted to
    shared npcsh jinx templates.
    """

    _TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS dynamic_tools (
            tool_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name             TEXT NOT NULL,
            version          TEXT NOT NULL DEFAULT '1.0',
            description      TEXT NOT NULL,
            jinja_template   TEXT NOT NULL,
            parameter_schema TEXT NOT NULL DEFAULT '{}',
            output_schema    TEXT NOT NULL DEFAULT '{}',
            reasoning        TEXT,
            session_id       TEXT,
            workspace_id     TEXT,
            created_at       TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at       TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(name, version, session_id, workspace_id)
        );
        CREATE INDEX IF NOT EXISTS idx_dynamic_tools_session
            ON dynamic_tools(session_id);
        CREATE INDEX IF NOT EXISTS idx_dynamic_tools_workspace
            ON dynamic_tools(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_dynamic_tools_name
            ON dynamic_tools(name);
    """

    def __init__(self, command_history_manager=None, db_path=None):
        self.ch = command_history_manager
        self._db_path = db_path
        self._init_table()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cursor(self):
        if self.ch is not None:
            return self.ch._cursor()
        import sqlite3

        return sqlite3.connect(self._db_path).cursor()

    def _init_table(self):
        for sql in self._TABLE_SQL.split(";"):
            if not sql.strip():
                continue
            cur = self._cursor()
            cur.execute(sql.strip())
            cur.connection.commit()

    @staticmethod
    def _serialize(value) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    @staticmethod
    def _deserialize(raw: str) -> dict:
        return json.loads(raw) if raw else {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_tool(
        self,
        name: str,
        jinja_template: str,
        description: str = "",
        parameter_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
        reasoning: str = "",
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> DynamicTool:
        """Persist a new dynamic tool, auto-bumping version on collision."""
        parameter_schema = parameter_schema or {}
        output_schema = output_schema or {}

        # --- version bump if (name, version, session_id, workspace_id) already exists ---
        cur = self._cursor()
        cur.execute(
            """
            SELECT version FROM dynamic_tools
            WHERE name = ? AND session_id IS ? AND workspace_id IS ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (name, session_id, workspace_id),
        )
        row = cur.fetchone()
        version = self._next_version(row[0] if row else None)

        cur.execute(
            """
            INSERT INTO dynamic_tools (
                name, version, description, jinja_template,
                parameter_schema, output_schema, reasoning,
                session_id, workspace_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                version,
                description,
                jinja_template,
                self._serialize(parameter_schema),
                self._serialize(output_schema),
                reasoning,
                session_id,
                workspace_id,
            ),
        )
        cur.connection.commit()
        tool_id = cur.lastrowid
        return self.get_tool(tool_id)

    def get_tool(self, tool_id: int) -> Optional[DynamicTool]:
        cur = self._cursor()
        cur.execute("SELECT * FROM dynamic_tools WHERE tool_id = ?", (tool_id,))
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_tool(row)

    def get_tool_by_name(
        self,
        name: str,
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Optional[DynamicTool]:
        """Get the newest version of a tool matching scope."""
        cur = self._cursor()
        cur.execute(
            """
            SELECT * FROM dynamic_tools
            WHERE name = ? AND session_id IS ? AND workspace_id IS ?
            ORDER BY updated_at DESC LIMIT 1
            """,
            (name, session_id, workspace_id),
        )
        row = cur.fetchone()
        if not row:
            return None
        return self._row_to_tool(row)

    def list_tools(
        self,
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> List[DynamicTool]:
        """List tools scoped to session/workspace, optionally filtered by name."""
        clauses: List[str] = []
        params: List[Any] = []
        if session_id is not None:
            clauses.append("session_id IS ?")
            params.append(session_id)
        if workspace_id is not None:
            clauses.append("workspace_id IS ?")
            params.append(workspace_id)
        if name is not None:
            clauses.append("name = ?")
            params.append(name)

        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"""
            SELECT * FROM dynamic_tools
            WHERE {where}
            ORDER BY updated_at DESC
        """
        cur = self._cursor()
        cur.execute(sql, params)
        return [self._row_to_tool(r) for r in cur.fetchall()]

    def update_tool(
        self,
        tool_id: int,
        jinja_template: Optional[str] = None,
        description: Optional[str] = None,
        parameter_schema: Optional[dict] = None,
        output_schema: Optional[dict] = None,
        reasoning: Optional[str] = None,
    ) -> Optional[DynamicTool]:
        """Bump version and update editable fields."""
        existing = self.get_tool(tool_id)
        if existing is None:
            return None

        new_version = self._next_version(existing.version)
        sets = ["version = ?", "updated_at = CURRENT_TIMESTAMP"]
        params: List[Any] = [new_version]

        if jinja_template is not None:
            sets.append("jinja_template = ?")
            params.append(jinja_template)
        if description is not None:
            sets.append("description = ?")
            params.append(description)
        if parameter_schema is not None:
            sets.append("parameter_schema = ?")
            params.append(self._serialize(parameter_schema))
        if output_schema is not None:
            sets.append("output_schema = ?")
            params.append(self._serialize(output_schema))
        if reasoning is not None:
            sets.append("reasoning = ?")
            params.append(reasoning)

        params.append(tool_id)
        sql = f"""UPDATE dynamic_tools SET {', '.join(sets)} WHERE tool_id = ?"""
        cur = self._cursor()
        cur.execute(sql, params)
        cur.connection.commit()
        return self.get_tool(tool_id)

    def delete_tool(self, tool_id: int) -> bool:
        cur = self._cursor()
        cur.execute("DELETE FROM dynamic_tools WHERE tool_id = ?", (tool_id,))
        cur.connection.commit()
        return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def call_tool(
        self,
        name: str,
        parameters: dict,
        env: Optional[dict] = None,
        memory=None,
        session_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> dict:
        """
        Find the best-matching tool and execute it.

        Lookup order:
            1. session scope (exact session match)
            2. workspace scope (exact workspace match)
            3. global scope (session_id=NULL, workspace_id=NULL)
        """
        tool = (
            self.get_tool_by_name(name, session_id=session_id, workspace_id=workspace_id)
            or self.get_tool_by_name(name, session_id=None, workspace_id=workspace_id)
            or self.get_tool_by_name(name, session_id=None, workspace_id=None)
        )
        if tool is None:
            return {
                "status": "error",
                "error": f"No dynamic tool named '{name}' found.",
            }
        env = env or {}
        try:
            result = tool.execute(env, parameters, memory=memory)
            return {"status": "ok", "tool": tool.to_schema(), "result": result}
        except Exception as exc:
            logger.exception("Dynamic tool execution failed")
            return {"status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Promotion to npcsh jinx
    # ------------------------------------------------------------------

    def promote_to_jinx(
        self,
        tool_id: int,
        jinx_dir: str,
        category: str = "llm_generated",
        overwrite: bool = False,
    ) -> dict:
        """
        Write a dynamic tool out as a ``.jinx`` file that npcsh can load.

        Parameters
        ----------
        tool_id : int
        jinx_dir : str
            Path to an npcsh ``npc_team/jinxes/`` directory.
        category : str
            Sub-directory under ``jinxes/`` (default ``llm_generated``).
        overwrite : bool
        """
        tool = self.get_tool(tool_id)
        if tool is None:
            return {"status": "error", "error": "Tool not found"}

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool.name)
        dest_dir = os.path.join(jinx_dir, "lib", category)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"{safe_name}.jinx")

        if os.path.exists(dest_path) and not overwrite:
            return {
                "status": "error",
                "error": f"Jinx file already exists at {dest_path}",
            }

        # ---- prepend metadata as jinja comments ----
        header = f"""{{# {{
    "name": "{tool.name}",
    "version": "{tool.version}",
    "description": "{tool.description}",
    "parameter_schema": {json.dumps(tool.parameter_schema)},
    "output_schema": {json.dumps(tool.output_schema)},
    "reasoning": {json.dumps(tool.reasoning)}
}} #}}

"""
        with open(dest_path, "w", encoding="utf-8") as fh:
            fh.write(header + tool.jinja_template)

        return {"status": "ok", "path": dest_path, "name": safe_name}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_version(self, current: Optional[str]) -> str:
        if current is None:
            return "1.0"
        try:
            major, minor = map(int, str(current).split("."))
            return f"{major}.{minor + 1}"
        except Exception:
            return str(current) + ".1"

    def _row_to_tool(self, row) -> DynamicTool:
        # row from sqlite3.Row or plain tuple
        keys = [
            "tool_id",
            "name",
            "version",
            "description",
            "jinja_template",
            "parameter_schema",
            "output_schema",
            "reasoning",
            "session_id",
            "workspace_id",
            "created_at",
            "updated_at",
        ]
        if hasattr(row, "keys"):
            d = dict(row)
        else:
            d = {k: row[i] for i, k in enumerate(keys)}
        return DynamicTool(
            tool_id=d["tool_id"],
            name=d["name"],
            version=d["version"],
            description=d["description"],
            jinja_template=d["jinja_template"],
            parameter_schema=self._deserialize(d.get("parameter_schema", "{}")),
            output_schema=self._deserialize(d.get("output_schema", "{}")),
            reasoning=d.get("reasoning", "") or "",
            session_id=d.get("session_id"),
            workspace_id=d.get("workspace_id"),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
        )
