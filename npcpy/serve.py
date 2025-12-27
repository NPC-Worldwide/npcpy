import datetime
from flask import Flask, request, jsonify, Response
from flask_sse import sse
import redis
import threading
import uuid
import sys 
import traceback
import glob
import re
import time
import asyncio
from typing import Optional, List, Dict, Callable, Any
from contextlib import AsyncExitStack

import io
from flask_cors import CORS
import os
import sqlite3
import json
from pathlib import Path
import yaml
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from PIL import Image
from PIL import ImageFile
from io import BytesIO
import networkx as nx
from collections import defaultdict
import numpy as np
import pandas as pd 
import subprocess
try:
    import ollama 
except:
    pass
from jinja2 import Environment, FileSystemLoader, Template, Undefined, DictLoader
class SilentUndefined(Undefined):
    def _fail_with_undefined_error(self, *args, **kwargs):
        return ""

# Import ShellState and helper functions from npcsh
from npcsh._state import ShellState


from npcpy.memory.knowledge_graph import load_kg_from_db
from npcpy.memory.search import execute_rag_command, execute_brainblast_command
from npcpy.data.load import load_file_contents
from npcpy.data.web import search_web


import base64
import shutil
import uuid

from npcpy.llm_funcs import gen_image, breathe                                                                                                                                                                

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from npcpy.npc_sysenv import get_locally_available_models
from npcpy.memory.command_history import (
    CommandHistory,
    save_conversation_message,
    generate_message_id,
)
from npcpy.npc_compiler import  Jinx, NPC, Team, load_jinxs_from_directory, build_jinx_tool_catalog, initialize_npc_project

from npcpy.llm_funcs import (
    get_llm_response, check_llm_command
)
from termcolor import cprint
from npcpy.tools import auto_tools

import json
import os
from pathlib import Path
from flask_cors import CORS






cancellation_flags = {}
cancellation_lock = threading.Lock()


def normalize_path_for_db(path_str):
    """
    Normalize a path for consistent database storage/querying.
    Converts backslashes to forward slashes for cross-platform compatibility.
    This ensures Windows paths match Unix paths in the database.
    """
    if not path_str:
        return path_str
    # Convert backslashes to forward slashes
    normalized = path_str.replace('\\', '/')
    # Remove trailing slashes for consistency
    normalized = normalized.rstrip('/')
    return normalized


# Minimal MCP client (inlined from npcsh corca to avoid corca import)
class MCPClientNPC:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.session: Optional[ClientSession] = None
        try:
            self._loop = asyncio.get_event_loop()
            if self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        self._exit_stack = self._loop.run_until_complete(AsyncExitStack().__aenter__())
        self.available_tools_llm: List[Dict[str, Any]] = []
        self.tool_map: Dict[str, Callable] = {}
        self.server_script_path: Optional[str] = None

    def _log(self, message: str, color: str = "cyan") -> None:
        if self.debug:
            cprint(f"[MCP Client] {message}", color, file=sys.stderr)

    async def _connect_async(self, server_script_path: str) -> None:
        self._log(f"Attempting to connect to MCP server: {server_script_path}")
        self.server_script_path = server_script_path
        abs_path = os.path.abspath(server_script_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"MCP server script not found: {abs_path}")

        if abs_path.endswith('.py'):
            cmd_parts = [sys.executable, abs_path]
        elif os.access(abs_path, os.X_OK):
            cmd_parts = [abs_path]
        else:
            raise ValueError(f"Unsupported MCP server script type or not executable: {abs_path}")

        server_params = StdioServerParameters(
            command=cmd_parts[0],
            args=[abs_path],
            env=os.environ.copy(),
            cwd=os.path.dirname(abs_path) or "."
        )
        if self.session:
            await self._exit_stack.aclose()

        self._exit_stack = AsyncExitStack()

        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(*stdio_transport))
        await self.session.initialize()

        response = await self.session.list_tools()
        self.available_tools_llm = []
        self.tool_map = {}

        if response.tools:
            for mcp_tool in response.tools:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                        "parameters": getattr(mcp_tool, "inputSchema", {"type": "object", "properties": {}})
                    }
                }
                self.available_tools_llm.append(tool_def)

                def make_tool_func(tool_name_closure):
                    async def tool_func(**kwargs):
                        if not self.session:
                            return {"error": "No MCP session"}
                        self._log(f"About to call MCP tool {tool_name_closure}")
                        try:
                            cleaned_kwargs = {k: (None if v == 'None' else v) for k, v in kwargs.items()}
                            result = await asyncio.wait_for(
                                self.session.call_tool(tool_name_closure, cleaned_kwargs),
                                timeout=30.0
                            )
                            self._log(f"MCP tool {tool_name_closure} returned: {type(result)}")
                            return result
                        except asyncio.TimeoutError:
                            self._log(f"Tool {tool_name_closure} timed out after 30 seconds", "red")
                            return {"error": f"Tool {tool_name_closure} timed out"}
                        except Exception as e:
                            self._log(f"Tool {tool_name_closure} error: {e}", "red")
                            return {"error": str(e)}

                    def sync_wrapper(**kwargs):
                        self._log(f"Sync wrapper called for {tool_name_closure}")
                        return self._loop.run_until_complete(tool_func(**kwargs))

                    return sync_wrapper

                self.tool_map[mcp_tool.name] = make_tool_func(mcp_tool.name)
        tool_names = list(self.tool_map.keys())
        self._log(f"Connection successful. Tools: {', '.join(tool_names) if tool_names else 'None'}")

    def connect_sync(self, server_script_path: str) -> bool:
        loop = self._loop
        if loop.is_closed():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            loop = self._loop
        try:
            loop.run_until_complete(self._connect_async(server_script_path))
            return True
        except Exception as e:
            cprint(f"MCP connection failed: {e}", "red", file=sys.stderr)
            return False

    def disconnect_sync(self):
        if self.session:
            self._log("Disconnecting MCP session.")
            loop = self._loop
            if not loop.is_closed():
                try:
                    async def close_session():
                        await self.session.close()
                        await self._exit_stack.aclose()
                    loop.run_until_complete(close_session())
                except RuntimeError:
                    pass
                except Exception as e:
                    print(f"Error during MCP client disconnect: {e}", file=sys.stderr)
            self.session = None
            self._exit_stack = None


def get_llm_response_with_handling(prompt, npc,model, provider, messages, tools, stream, team, context=None):
    """Unified LLM response with basic exception handling (inlined from corca to avoid that dependency)."""
    try:
        return get_llm_response(
            prompt=prompt,
            npc=npc,
            model=model, 
            provider=provider,
            messages=messages,
            tools=tools,
            auto_process_tool_calls=False,
            stream=stream,
            team=team,
            context=context
        )
    except Exception:
        # Fallback retry without context compression logic to keep it simple here.
        return get_llm_response(
            prompt=prompt,
            npc=npc,
            model=model, 
            provider=provider,
            messages=messages,
            tools=tools,
            auto_process_tool_calls=False,
            stream=stream,
            team=team,
            context=context
        )
    
class MCPServerManager:
    """
    Simple in-process tracker for launching/stopping MCP servers.
    Currently uses subprocess.Popen to start a Python stdio MCP server script.
    """

    def __init__(self):
        self._procs = {}
        self._lock = threading.Lock()

    def start(self, server_path: str):
        server_path = os.path.expanduser(server_path)
        abs_path = os.path.abspath(server_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"MCP server script not found at {abs_path}")

        with self._lock:
            existing = self._procs.get(abs_path)
            if existing and existing.poll() is None:
                return {"status": "running", "pid": existing.pid, "serverPath": abs_path}

            cmd = [sys.executable, abs_path]
            proc = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(abs_path) or ".",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._procs[abs_path] = proc
            return {"status": "started", "pid": proc.pid, "serverPath": abs_path}

    def stop(self, server_path: str):
        server_path = os.path.expanduser(server_path)
        abs_path = os.path.abspath(server_path)
        with self._lock:
            proc = self._procs.get(abs_path)
            if not proc:
                return {"status": "not_found", "serverPath": abs_path}
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            del self._procs[abs_path]
            return {"status": "stopped", "serverPath": abs_path}

    def status(self, server_path: str):
        server_path = os.path.expanduser(server_path)
        abs_path = os.path.abspath(server_path)
        with self._lock:
            proc = self._procs.get(abs_path)
            if not proc:
                return {"status": "not_started", "serverPath": abs_path}
            running = proc.poll() is None
            return {
                "status": "running" if running else "exited",
                "serverPath": abs_path,
                "pid": proc.pid,
                "returncode": None if running else proc.returncode,
            }

    def running(self):
        with self._lock:
            return {
                path: {
                    "pid": proc.pid,
                    "status": "running" if proc.poll() is None else "exited",
                    "returncode": None if proc.poll() is None else proc.returncode,
                }
                for path, proc in self._procs.items()
            }


mcp_server_manager = MCPServerManager()

def get_project_npc_directory(current_path=None):
    """
    Get the project NPC directory based on the current path
    
    Args:
        current_path: The current path where project NPCs should be looked for
        
    Returns:
        Path to the project's npc_team directory
    """
    if current_path:
        return os.path.join(current_path, "npc_team")
    else:
        
        return os.path.abspath("./npc_team")


def load_project_env(current_path):
    """
    Load environment variables from a project's .env file
    
    Args:
        current_path: The current project directory path
    
    Returns:
        Dictionary of environment variables that were loaded
    """
    if not current_path:
        return {}
    
    env_path = os.path.join(current_path, ".env")
    loaded_vars = {}
    
    if os.path.exists(env_path):
        print(f"Loading project environment from {env_path}")
        
        
        success = load_dotenv(env_path, override=True)
        
        if success:
            
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            loaded_vars[key.strip()] = value.strip().strip("\"'")
            
            print(f"Loaded {len(loaded_vars)} variables from project .env file")
        else:
            print(f"Failed to load environment variables from {env_path}")
    else:
        print(f"No .env file found at {env_path}")
    
    return loaded_vars




def load_kg_data(generation=None):
    """Helper function to load data up to a specific generation."""
    engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
    
    query_suffix = f" WHERE generation <= {generation}" if generation is not None else ""
    
    concepts_df = pd.read_sql_query(f"SELECT * FROM kg_concepts{query_suffix}", engine)
    facts_df = pd.read_sql_query(f"SELECT * FROM kg_facts{query_suffix}", engine)
    
    
    all_links_df = pd.read_sql_query("SELECT * FROM kg_links", engine)
    valid_nodes = set(concepts_df['name']).union(set(facts_df['statement']))
    links_df = all_links_df[all_links_df['source'].isin(valid_nodes) & all_links_df['target'].isin(valid_nodes)]
        
    return concepts_df, facts_df, links_df


app = Flask(__name__)
app.config["REDIS_URL"] = "redis://localhost:6379"
app.config['DB_PATH'] = ''
app.jinx_conversation_contexts ={}

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

available_models = {}
CORS(
    app,
    origins=["http://localhost:5173"],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    supports_credentials=True,
)

def get_db_connection():
    engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
    return engine

def get_db_session():
    engine = get_db_connection()
    Session = sessionmaker(bind=engine)
    return Session()


def resolve_mcp_server_path(current_path=None, explicit_path=None, force_global=False):
    """
    Resolve an MCP server path using npcsh.corca's helper when available.
    Falls back to ~/.npcsh/npc_team/mcp_server.py.
    """
    if explicit_path:
        abs_path = os.path.abspath(os.path.expanduser(explicit_path))
        if os.path.exists(abs_path):
            return abs_path
    try:
        from npcsh.corca import _resolve_and_copy_mcp_server_path
        resolved = _resolve_and_copy_mcp_server_path(
            explicit_path=explicit_path,
            current_path=current_path,
            team_ctx_mcp_servers=None,
            interactive=False,
            auto_copy_bypass=True,
            force_global=force_global,
        )
        if resolved:
            return os.path.abspath(resolved)
    except Exception as e:
        print(f"resolve_mcp_server_path: fallback path due to error: {e}")
    
    fallback = os.path.expanduser("~/.npcsh/npc_team/mcp_server.py")
    return fallback

extension_map = {
    "PNG": "images",
    "JPG": "images",
    "JPEG": "images",
    "GIF": "images",
    "SVG": "images",
    "MP4": "videos",
    "AVI": "videos",
    "MOV": "videos",
    "WMV": "videos",
    "MPG": "videos",
    "MPEG": "videos",
    "DOC": "documents",
    "DOCX": "documents",
    "PDF": "documents",
    "PPT": "documents",
    "PPTX": "documents",
    "XLS": "documents",
    "XLSX": "documents",
    "TXT": "documents",
    "CSV": "documents",
    "ZIP": "archives",
    "RAR": "archives",
    "7Z": "archives",
    "TAR": "archives",
    "GZ": "archives",
    "BZ2": "archives",
    "ISO": "archives",
}
def load_npc_by_name_and_source(name, source, db_conn=None, current_path=None):
    """
    Loads an NPC from either project or global directory based on source
    
    Args:
        name: The name of the NPC to load
        source: Either 'project' or 'global' indicating where to look for the NPC
        db_conn: Optional database connection
        current_path: The current path where project NPCs should be looked for
    
    Returns:
        NPC object or None if not found
    """
    if not db_conn:
        db_conn = get_db_connection()
    
    
    if source == 'project':
        npc_directory = get_project_npc_directory(current_path)
        print(f"Looking for project NPC in: {npc_directory}")
    else:  
        npc_directory = app.config['user_npc_directory']
        print(f"Looking for global NPC in: {npc_directory}")
    
    
    npc_path = os.path.join(npc_directory, f"{name}.npc")
    
    if os.path.exists(npc_path):
        try:
            npc = NPC(file=npc_path, db_conn=db_conn)
            return npc
        except Exception as e:
            print(f"Error loading NPC {name} from {source}: {str(e)}")
            return None
    else:
        print(f"NPC file not found: {npc_path}")
        
        

def get_conversation_history(conversation_id):
    """Fetch all messages for a conversation in chronological order."""
    if not conversation_id:
        return []

    engine = get_db_connection()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT role, content, timestamp
                FROM conversation_history
                WHERE conversation_id = :conversation_id
                ORDER BY timestamp ASC
            """)
            result = conn.execute(query, {"conversation_id": conversation_id})
            messages = result.fetchall()

            return [
                {
                    "role": msg[0],  
                    "content": msg[1],  
                    "timestamp": msg[2],  
                }
                for msg in messages
            ]
    except Exception as e:
        print(f"Error fetching conversation history: {e}")
        return []


def fetch_messages_for_conversation(conversation_id):
    """Fetch all messages for a conversation in chronological order."""
    engine = get_db_connection()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT role, content, timestamp
                FROM conversation_history
                WHERE conversation_id = :conversation_id
                ORDER BY timestamp ASC
            """)
            result = conn.execute(query, {"conversation_id": conversation_id})
            messages = result.fetchall()

            return [
                {
                    "role": message[0],  
                    "content": message[1],  
                    "timestamp": message[2],  
                }
                for message in messages
            ]
    except Exception as e:
        print(f"Error fetching messages for conversation: {e}")
        return []
    
    
        
            
@app.route('/api/kg/generations')
def list_generations():
    try:
        engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
        
        query = "SELECT DISTINCT generation FROM kg_concepts UNION SELECT DISTINCT generation FROM kg_facts"
        generations_df = pd.read_sql_query(query, engine)
        generations = generations_df.iloc[:, 0].tolist()
        return jsonify({"generations": sorted([g for g in generations if g is not None])})
    except Exception as e:
        
        print(f"Error listing generations (likely new DB): {e}")
        return jsonify({"generations": []})

@app.route('/api/kg/graph')
def get_graph_data():
    generation_str = request.args.get('generation')
    generation = int(generation_str) if generation_str and generation_str != 'null' else None
    
    concepts_df, facts_df, links_df = load_kg_data(generation)
    
    nodes = []
    nodes.extend([{'id': name, 'type': 'concept'} for name in concepts_df['name']])
    nodes.extend([{'id': statement, 'type': 'fact'} for statement in facts_df['statement']])
    
    links = [{'source': row['source'], 'target': row['target']} for _, row in links_df.iterrows()]
    
    return jsonify(graph={'nodes': nodes, 'links': links})

@app.route('/api/kg/network-stats')
def get_network_stats():
    generation = request.args.get('generation', type=int)
    _, _, links_df = load_kg_data(generation)
    G = nx.DiGraph()
    for _, link in links_df.iterrows():
        G.add_edge(link['source'], link['target'])
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return jsonify(stats={'nodes': 0, 'edges': 0, 'density': 0, 'avg_degree': 0, 'node_degrees': {}})
    degrees = dict(G.degree())
    stats = {
        'nodes': n_nodes, 'edges': G.number_of_edges(), 'density': nx.density(G),
        'avg_degree': np.mean(list(degrees.values())) if degrees else 0, 'node_degrees': degrees
    }
    return jsonify(stats=stats)

@app.route('/api/kg/cooccurrence')
def get_cooccurrence_network():
    generation = request.args.get('generation', type=int)
    min_cooccurrence = request.args.get('min_cooccurrence', 2, type=int)
    _, _, links_df = load_kg_data(generation)
    fact_to_concepts = defaultdict(set)
    for _, link in links_df.iterrows():
        if link['type'] == 'fact_to_concept':
            fact_to_concepts[link['source']].add(link['target'])
    cooccurrence = defaultdict(int)
    for concepts in fact_to_concepts.values():
        concepts_list = list(concepts)
        for i, c1 in enumerate(concepts_list):
            for c2 in concepts_list[i+1:]:
                pair = tuple(sorted((c1, c2)))
                cooccurrence[pair] += 1
    G_cooccur = nx.Graph()
    for (c1, c2), weight in cooccurrence.items():
        if weight >= min_cooccurrence:
            G_cooccur.add_edge(c1, c2, weight=weight)
    if G_cooccur.number_of_nodes() == 0:
        return jsonify(network={'nodes': [], 'links': []})
    components = list(nx.connected_components(G_cooccur))
    node_to_community = {node: i for i, component in enumerate(components) for node in component}
    nodes = [{'id': node, 'type': 'concept', 'community': node_to_community.get(node, 0)} for node in G_cooccur.nodes()]
    links = [{'source': u, 'target': v, 'weight': d['weight']} for u, v, d in G_cooccur.edges(data=True)]
    return jsonify(network={'nodes': nodes, 'links': links})

@app.route('/api/kg/centrality')
def get_centrality_data():
    generation = request.args.get('generation', type=int)
    concepts_df, _, links_df = load_kg_data(generation)
    G = nx.Graph()
    fact_concept_links = links_df[links_df['type'] == 'fact_to_concept']
    for _, link in fact_concept_links.iterrows():
        if link['target'] in concepts_df['name'].values:
            G.add_edge(link['source'], link['target'])
    concept_degree = {node: cent for node, cent in nx.degree_centrality(G).items() if node in concepts_df['name'].values}
    return jsonify(centrality={'degree': concept_degree})



@app.route("/api/attachments/<message_id>", methods=["GET"])
def get_message_attachments(message_id):
    """Get all attachments for a message"""
    try:
        command_history = CommandHistory(app.config.get('DB_PATH'))
        attachments = command_history.get_message_attachments(message_id)
        return jsonify({"attachments": attachments, "error": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/attachment/<attachment_id>", methods=["GET"])
def get_attachment(attachment_id):
    """Get specific attachment data"""
    try:
        command_history = CommandHistory(app.config.get('DB_PATH'))
        data, name, type = command_history.get_attachment_data(attachment_id)

        if data:
            
            base64_data = base64.b64encode(data).decode("utf-8")
            return jsonify(
                {"data": base64_data, "name": name, "type": type, "error": None}
            )
        return jsonify({"error": "Attachment not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/capture_screenshot", methods=["GET"])
def capture():
    
    screenshot = capture_screenshot(None, full=True)

    
    if not screenshot:
        print("Screenshot capture failed")
        return None

    return jsonify({"screenshot": screenshot})
@app.route("/api/settings/global", methods=["GET", "OPTIONS"])
def get_global_settings():
    if request.method == "OPTIONS":
        return "", 200

    try:
        npcshrc_path = os.path.expanduser("~/.npcshrc")

        global_settings = {
            "model": "llama3.2",
            "provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "embedding_provider": "ollama",
            "search_provider": "perplexity",
            "default_folder": os.path.expanduser("~/.npcsh/"),
            "is_predictive_text_enabled": False,
            "predictive_text_model": "llama3.2",
            "predictive_text_provider": "ollama",
            "backend_python_path": "",  # Empty means use bundled backend
        }
        global_vars = {}

        if os.path.exists(npcshrc_path):
            with open(npcshrc_path, "r") as f:
                for line in f:
                    line = line.split("#")[0].strip()
                    if not line:
                        continue

                    if "=" not in line:
                        continue

                    key, value = line.split("=", 1)
                    key = key.strip()
                    if key.startswith("export "):
                        key = key[7:]

                    value = value.strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    key_mapping = {
                        "NPCSH_MODEL": "model",
                        "NPCSH_PROVIDER": "provider",
                        "NPCSH_EMBEDDING_MODEL": "embedding_model",
                        "NPCSH_EMBEDDING_PROVIDER": "embedding_provider",
                        "NPCSH_SEARCH_PROVIDER": "search_provider",
                        "NPCSH_STREAM_OUTPUT": "NPCSH_STREAM_OUTPUT",
                        "NPC_STUDIO_DEFAULT_FOLDER": "default_folder",
                        "NPC_STUDIO_PREDICTIVE_TEXT_ENABLED": "is_predictive_text_enabled",
                        "NPC_STUDIO_PREDICTIVE_TEXT_MODEL": "predictive_text_model",
                        "NPC_STUDIO_PREDICTIVE_TEXT_PROVIDER": "predictive_text_provider",
                        "BACKEND_PYTHON_PATH": "backend_python_path",  # Custom Python for backend
                    }

                    if key in key_mapping:
                        if key == "NPC_STUDIO_PREDICTIVE_TEXT_ENABLED":
                            global_settings[key_mapping[key]] = value.lower() == 'true'
                        else:
                            global_settings[key_mapping[key]] = value
                    else:
                        global_vars[key] = value

        print("Global settings loaded from .npcshrc")
        print(global_settings)
        return jsonify(
            {
                "global_settings": global_settings,
                "global_vars": global_vars,
                "error": None,
            }
        )

    except Exception as e:
        print(f"Error in get_global_settings: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
def _get_jinx_files_recursively(directory):
    """Helper to recursively find all .jinx file paths."""
    jinx_paths = []
    if os.path.exists(directory):
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".jinx"):
                    jinx_paths.append(os.path.join(root, filename))
    return jinx_paths

@app.route("/api/jinxs/available", methods=["GET"])
def get_available_jinxs():
    try:
        current_path = request.args.get('currentPath')
        jinx_names = set()

        if current_path:
            team_jinxs_dir = os.path.join(current_path, 'npc_team', 'jinxs')
            jinx_paths = _get_jinx_files_recursively(team_jinxs_dir)
            for path in jinx_paths:
                jinx_names.add(os.path.basename(path)[:-5])

        global_jinxs_dir = os.path.expanduser('~/.npcsh/npc_team/jinxs')
        jinx_paths = _get_jinx_files_recursively(global_jinxs_dir)
        for path in jinx_paths:
            jinx_names.add(os.path.basename(path)[:-5])

        return jsonify({'jinxs': sorted(list(jinx_names)), 'error': None})
    except Exception as e:
        print(f"Error getting available jinxs: {str(e)}")
        traceback.print_exc()
        return jsonify({'jinxs': [], 'error': str(e)}), 500


@app.route("/api/jinx/execute", methods=["POST"])
def execute_jinx():
    """
    Execute a specific jinx with provided arguments.
    Returns the output as a JSON response.
    """
    data = request.json
    
    stream_id = data.get("streamId")
    if not stream_id:
        stream_id = str(uuid.uuid4())
    
    with cancellation_lock:
        cancellation_flags[stream_id] = False
    
    print(f"--- Jinx Execution Request for streamId: {stream_id} ---", file=sys.stderr)
    print(f"Request Data: {json.dumps(data, indent=2)}", file=sys.stderr)

    jinx_name = data.get("jinxName")
    jinx_args = data.get("jinxArgs", [])
    print(f"Jinx Name: {jinx_name}, Jinx Args: {jinx_args}", file=sys.stderr)
    conversation_id = data.get("conversationId")
    model = data.get("model")
    provider = data.get("provider")

    if not conversation_id:
        print("ERROR: conversationId is required for Jinx execution with persistent variables", file=sys.stderr)
        return jsonify({"error": "conversationId is required for Jinx execution with persistent variables"}), 400

    npc_name = data.get("npc")
    npc_source = data.get("npcSource", "global")
    current_path = data.get("currentPath")
    
    if not jinx_name:
        print("ERROR: jinxName is required", file=sys.stderr)
        return jsonify({"error": "jinxName is required"}), 400
    
    if current_path:
        load_project_env(current_path)
    
    jinx = None
    
    if npc_name:
        db_conn = get_db_connection()
        npc_object = load_npc_by_name_and_source(npc_name, npc_source, db_conn, current_path)
        if not npc_object and npc_source == 'project':
            npc_object = load_npc_by_name_and_source(npc_name, 'global', db_conn)
    else:
        npc_object = None
    
    if npc_object and hasattr(npc_object, 'jinxs_dict') and jinx_name in npc_object.jinxs_dict:
        jinx = npc_object.jinxs_dict[jinx_name]
        print(f"Found jinx in NPC's jinxs_dict", file=sys.stderr)
    
    if not jinx and current_path:
        project_jinxs_base = os.path.join(current_path, 'npc_team', 'jinxs')
        if os.path.exists(project_jinxs_base):
            for root, dirs, files in os.walk(project_jinxs_base):
                if f'{jinx_name}.jinx' in files:
                    project_jinx_path = os.path.join(root, f'{jinx_name}.jinx')
                    jinx = Jinx(jinx_path=project_jinx_path)
                    print(f"Found jinx at: {project_jinx_path}", file=sys.stderr)
                    break
        
    if not jinx:
        global_jinxs_base = os.path.expanduser('~/.npcsh/npc_team/jinxs')
        if os.path.exists(global_jinxs_base):
            for root, dirs, files in os.walk(global_jinxs_base):
                if f'{jinx_name}.jinx' in files:
                    global_jinx_path = os.path.join(root, f'{jinx_name}.jinx')
                    jinx = Jinx(jinx_path=global_jinx_path)
                    print(f"Found jinx at: {global_jinx_path}", file=sys.stderr)
                    
                    # Initialize jinx steps by calling render_first_pass
                    from jinja2 import Environment
                    temp_env = Environment()
                    jinx.render_first_pass(temp_env, {})
                    
                    break
    
    if not jinx:
        print(f"ERROR: Jinx '{jinx_name}' not found", file=sys.stderr)
        searched_paths = []
        if npc_object:
            searched_paths.append(f"NPC {npc_name} jinxs_dict")
        if current_path:
            searched_paths.append(f"Project jinxs at {os.path.join(current_path, 'npc_team', 'jinxs')}")
        searched_paths.append(f"Global jinxs at {os.path.expanduser('~/.npcsh/npc_team/jinxs')}")
        print(f"Searched in: {', '.join(searched_paths)}", file=sys.stderr)
        return jsonify({"error": f"Jinx '{jinx_name}' not found"}), 404
    
    from npcpy.npc_compiler import extract_jinx_inputs

    fixed_args = []
    i = 0
    
    # Filter out None values from jinx_args before processing
    cleaned_jinx_args = [arg for arg in jinx_args if arg is not None]

    while i < len(cleaned_jinx_args):
        arg = cleaned_jinx_args[i]
        if arg.startswith('-'):
            fixed_args.append(arg)
            value_parts = []
            i += 1
            while i < len(cleaned_jinx_args) and not cleaned_jinx_args[i].startswith('-'):
                value_parts.append(cleaned_jinx_args[i])
                i += 1
            
            if value_parts:
                full_value = " ".join(value_parts)
                if full_value.startswith("'") and full_value.endswith("'"):
                    full_value = full_value[1:-1]
                elif full_value.startswith('"') and full_value.endswith('"'):
                    full_value = full_value[1:-1]
                fixed_args.append(full_value)
        else:
            fixed_args.append(arg)
            i += 1

    input_values = extract_jinx_inputs(fixed_args, jinx)

    print(f'Executing jinx with input_values: {input_values}', file=sys.stderr)
    
    command_history = CommandHistory(app.config.get('DB_PATH'))
    messages = fetch_messages_for_conversation(conversation_id)
    
    all_jinxs = {}
    if npc_object and hasattr(npc_object, 'jinxs_dict'):
        all_jinxs.update(npc_object.jinxs_dict)
    
    if conversation_id not in app.jinx_conversation_contexts:
        app.jinx_conversation_contexts[conversation_id] = {}
    jinx_local_context = app.jinx_conversation_contexts[conversation_id]

    print(f"--- CONTEXT STATE (conversationId: {conversation_id}) ---", file=sys.stderr)
    print(f"jinx_local_context BEFORE Jinx execution: {jinx_local_context}", file=sys.stderr)

    
    # Create state object
    state = ShellState(
        npc=npc_object,
        team=None,
        conversation_id=conversation_id,
        chat_model=model or os.getenv('NPCSH_CHAT_MODEL', 'gemma3:4b'),
        chat_provider=provider or os.getenv('NPCSH_CHAT_PROVIDER', 'ollama'),
        current_path=current_path or os.getcwd(),
        search_provider=os.getenv('NPCSH_SEARCH_PROVIDER', 'duckduckgo'),
        embedding_model=os.getenv('NPCSH_EMBEDDING_MODEL', 'nomic-embed-text'),
        embedding_provider=os.getenv('NPCSH_EMBEDDING_PROVIDER', 'ollama'),
    )
    
    # Build extra_globals with state and all necessary functions
    extra_globals_for_jinx = {
        **jinx_local_context,
        'state': state,
        'CommandHistory': CommandHistory,
        'load_kg_from_db': load_kg_from_db,
        #'get_relevant_memories': get_relevant_memories,
        #'search_kg_facts': search_kg_facts,
    }

    jinx_execution_result = jinx.execute(
        input_values=input_values,
        jinja_env=npc_object.jinja_env if npc_object else None,
        npc=npc_object,
        messages=messages,
        extra_globals=extra_globals_for_jinx
    )

    output_from_jinx_result = jinx_execution_result.get('output')
    
    final_output_string = str(output_from_jinx_result) if output_from_jinx_result is not None else ""

    if isinstance(jinx_execution_result, dict):
        for key, value in jinx_execution_result.items():
            jinx_local_context[key] = value

    print(f"jinx_local_context AFTER Jinx execution (final state): {jinx_local_context}", file=sys.stderr)
    print(f"Jinx execution result output: {output_from_jinx_result}", file=sys.stderr)

    user_message_id = generate_message_id()
    
    # Use cleaned_jinx_args for logging the user message
    user_command_log = f"/{jinx_name} {' '.join(cleaned_jinx_args)}"
    save_conversation_message(
        command_history,
        conversation_id,
        "user",
        user_command_log,
        wd=current_path,
        model=model,
        provider=provider,
        npc=npc_name,
        message_id=user_message_id
    )
    
    assistant_message_id = generate_message_id()
    save_conversation_message(
        command_history,
        conversation_id,
        "assistant",
        final_output_string,
        wd=current_path,
        model=model,
        provider=provider,
        npc=npc_name,
        message_id=assistant_message_id
    )

    # Determine mimetype based on content
    is_html = bool(re.search(r'<[a-z][\s\S]*>', final_output_string, re.IGNORECASE))
    
    if is_html:
        return Response(final_output_string, mimetype="text/html")
    else:
        return Response(final_output_string, mimetype="text/plain")
@app.route("/api/settings/global", methods=["POST", "OPTIONS"])
def save_global_settings():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.json
        npcshrc_path = os.path.expanduser("~/.npcshrc")

        key_mapping = {
            "model": "NPCSH_CHAT_MODEL",
            "provider": "NPCSH_CHAT_PROVIDER",
            "embedding_model": "NPCSH_EMBEDDING_MODEL",
            "embedding_provider": "NPCSH_EMBEDDING_PROVIDER",
            "search_provider": "NPCSH_SEARCH_PROVIDER",
            "NPCSH_STREAM_OUTPUT": "NPCSH_STREAM_OUTPUT",
            "default_folder": "NPC_STUDIO_DEFAULT_FOLDER",
            "is_predictive_text_enabled": "NPC_STUDIO_PREDICTIVE_TEXT_ENABLED",
            "predictive_text_model": "NPC_STUDIO_PREDICTIVE_TEXT_MODEL",
            "predictive_text_provider": "NPC_STUDIO_PREDICTIVE_TEXT_PROVIDER",
            "backend_python_path": "BACKEND_PYTHON_PATH",  # Custom Python for backend (requires restart)
        }

        os.makedirs(os.path.dirname(npcshrc_path), exist_ok=True)
        print(data)
        with open(npcshrc_path, "w") as f:

            for key, value in data.get("global_settings", {}).items():
                if key in key_mapping and value is not None: # Check for None explicitly
                    # Handle boolean conversion for saving
                    if key == "is_predictive_text_enabled":
                        value_to_write = str(value).upper()
                    elif " " in str(value):
                        value_to_write = f'"{value}"'
                    else:
                        value_to_write = str(value)
                    f.write(f"export {key_mapping[key]}={value_to_write}\n")

            for key, value in data.get("global_vars", {}).items():
                if key and value is not None: # Check for None explicitly
                    if " " in str(value):
                        value_to_write = f'"{value}"'
                    else:
                        value_to_write = str(value)
                    f.write(f"export {key}={value_to_write}\n")

        return jsonify({"message": "Global settings saved successfully", "error": None})

    except Exception as e:
        print(f"Error in save_global_settings: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/settings/project", methods=["GET", "OPTIONS"])  
def get_project_settings():
    if request.method == "OPTIONS":
        return "", 200

    try:
        current_dir = request.args.get("path")
        if not current_dir:
            return jsonify({"error": "No path provided"}), 400

        env_path = os.path.join(current_dir, ".env")
        env_vars = {}

        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            env_vars[key.strip()] = value.strip().strip("\"'")

        return jsonify({"env_vars": env_vars, "error": None})

    except Exception as e:
        print(f"Error in get_project_settings: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings/project", methods=["POST", "OPTIONS"])  
def save_project_settings():
    if request.method == "OPTIONS":
        return "", 200

    try:
        current_dir = request.args.get("path")
        if not current_dir:
            return jsonify({"error": "No path provided"}), 400

        data = request.json
        env_path = os.path.join(current_dir, ".env")

        with open(env_path, "w") as f:
            for key, value in data.get("env_vars", {}).items():
                f.write(f"{key}={value}\n")

        return jsonify(
            {"message": "Project settings saved successfully", "error": None}
        )

    except Exception as e:
        print(f"Error in save_project_settings: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def get_models():
    """
    Endpoint to retrieve available models based on the current project path.
    Checks for local configurations (.env) and Ollama.
    """
    global available_models
    current_path = request.args.get("currentPath")
    if not current_path:
        
        
        current_path = os.path.expanduser("~/.npcsh")  
        print("Warning: No currentPath provided for /api/models, using default.")
        

    try:
        
        available_models = get_locally_available_models(current_path)

        
        
        formatted_models = []
        for m, p in available_models.items():
            
            text_only = (
                "(text only)"
                if p == "ollama"
                and m in ["llama3.2", "deepseek-v3", "phi4", "gemma3:1b"]
                else ""
            )
            
            display_model = m
            if m.endswith(('.gguf', '.ggml')):
                # For local GGUF/GGML files, show just the filename
                display_model = os.path.basename(m)

            display_name = f"{display_model} | {p} {text_only}".strip()

            formatted_models.append(
                {
                    "value": m,  
                    "provider": p,
                    "display_name": display_name,
                }
            )
            print(m, p)
        return jsonify({"models": formatted_models, "error": None})

    except Exception as e:
        print(f"Error getting available models: {str(e)}")

        traceback.print_exc()
        
        return jsonify({"models": [], "error": str(e)}), 500

@app.route('/api/<command>', methods=['POST'])
def api_command(command):
    data = request.json or {}
    
    
    handler = router.get_route(command)
    if not handler:
        return jsonify({"error": f"Unknown command: {command}"})
    
    
    if router.shell_only.get(command, False):
        return jsonify({"error": f"Command {command} is only available in shell mode"})
    
    
    try:
        
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        
        
        command_str = command
        if args:
            command_str += " " + " ".join(str(arg) for arg in args)
            
        result = handler(command_str, **kwargs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/jinxs/save", methods=["POST"])
def save_jinx():
    try:
        data = request.json
        jinx_data = data.get("jinx")
        is_global = data.get("isGlobal")
        current_path = data.get("currentPath")
        jinx_name = jinx_data.get("jinx_name")

        if not jinx_name:
            return jsonify({"error": "Jinx name is required"}), 400

        if is_global:
            jinxs_dir = os.path.join(
                os.path.expanduser("~"), ".npcsh", "npc_team", "jinxs"
            )
        else:
            if not current_path.endswith("npc_team"):
                current_path = os.path.join(current_path, "npc_team")
            jinxs_dir = os.path.join(current_path, "jinxs")

        os.makedirs(jinxs_dir, exist_ok=True)

        
        jinx_yaml = {
            "description": jinx_data.get("description", ""),
            "inputs": jinx_data.get("inputs", []),
            "steps": jinx_data.get("steps", []),
        }

        file_path = os.path.join(jinxs_dir, f"{jinx_name}.jinx")
        with open(file_path, "w") as f:
            yaml.safe_dump(jinx_yaml, f, sort_keys=False)

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
def serialize_jinx_inputs(inputs):
    result = []
    for inp in inputs:
        if isinstance(inp, str):
            result.append(inp)
        elif isinstance(inp, dict):
            key = list(inp.keys())[0]
            result.append(key)
        else:
            result.append(str(inp))
    return result

@app.route("/api/jinx/test", methods=["POST"])
def test_jinx():
    data = request.json
    jinx_data = data.get("jinx")
    test_inputs = data.get("inputs", {})
    current_path = data.get("currentPath")
    
    if current_path:
        load_project_env(current_path)
    
    jinx = Jinx(jinx_data=jinx_data)
    
    from jinja2 import Environment
    temp_env = Environment()
    jinx.render_first_pass(temp_env, {})
    
    conversation_id = f"jinx_test_{uuid.uuid4().hex[:8]}"
    command_history = CommandHistory(app.config.get('DB_PATH'))
    
    # 1. Save user's test command to conversation_history to get a message_id
    user_test_command = f"Testing jinx /{jinx.jinx_name} with inputs: {test_inputs}"
    user_message_id = generate_message_id()
    save_conversation_message(
        command_history,
        conversation_id,
        "user",
        user_test_command,
        wd=current_path,
        model=None, # Or appropriate model/provider for the test context
        provider=None,
        npc=None,
        message_id=user_message_id
    )

    # Jinx execution status and output are now part of the assistant's response
    jinx_execution_status = "success"
    jinx_error_message = None
    output = "Jinx execution did not complete." # Default output

    try:
        result = jinx.execute(
            input_values=test_inputs,
            npc=None,
            messages=[],
            extra_globals={},
            jinja_env=temp_env
        )
        output = result.get('output', str(result))
        if result.get('error'): # Assuming jinx.execute might return an 'error' key
            jinx_execution_status = "failed"
            jinx_error_message = str(result.get('error'))
    except Exception as e:
        jinx_execution_status = "failed"
        jinx_error_message = str(e)
        output = f"Jinx execution failed: {e}"

    # The jinx_executions table is populated by a trigger from conversation_history.
    # The details of the execution (inputs, output, status) are now expected to be
    # derived by analyzing the user's command and the subsequent assistant's response.
    # No explicit update to jinx_executions is needed here.

    # 2. Save assistant's response to conversation_history
    assistant_response_message_id = generate_message_id() # ID for the assistant's response
    save_conversation_message(
        command_history,
        conversation_id,
        "assistant",
        output, # The jinx output is the assistant's response for the test
        wd=current_path,
        model=None,
        provider=None,
        npc=None,
        message_id=assistant_response_message_id
    )

    return jsonify({
        "output": output,
        "conversation_id": conversation_id,
        "execution_id": user_message_id, # Return the user's message_id as the execution_id
        "error": jinx_error_message
    })
from npcpy.ft.diff import train_diffusion, DiffusionConfig
import threading

from npcpy.memory.knowledge_graph import (
    load_kg_from_db,
    save_kg_to_db # ADD THIS LINE to import the correct function
)

from collections import defaultdict # ADD THIS LINE for collecting links if not already present

finetune_jobs = {}

def extract_and_store_memories(
    conversation_text,
    conversation_id,
    command_history,
    npc_name,
    team_name,
    current_path,
    model,
    provider,
    npc_object=None
):
    from npcpy.llm_funcs import get_facts
    from npcpy.memory.command_history import format_memory_context
    # Your CommandHistory.get_memory_examples_for_context returns a dict with 'approved' and 'rejected'
    memory_examples_dict = command_history.get_memory_examples_for_context(
        npc=npc_name,
        team=team_name,
        directory_path=current_path
    )
    
    memory_context = format_memory_context(memory_examples_dict)
    
    facts = get_facts(
        conversation_text,
        model=npc_object.model if npc_object else model,
        provider=npc_object.provider if npc_object else provider,
        npc=npc_object,
        context=memory_context
    )
    
    memories_for_approval = []
    
    # Initialize structures to collect KG data for a single save_kg_to_db call
    kg_facts_to_save = []
    kg_concepts_to_save = []
    fact_to_concept_links_temp = defaultdict(list)
    
    
    if facts:
        for i, fact in enumerate(facts):
            # Store memory in memory_lifecycle table
            memory_id = command_history.add_memory_to_database(
                message_id=f"{conversation_id}_{datetime.datetime.now().strftime('%H%M%S')}_{i}",
                conversation_id=conversation_id,
                npc=npc_name or "default",
                team=team_name or "default",
                directory_path=current_path or "/",
                initial_memory=fact.get('statement', str(fact)),
                status="pending_approval",
                model=npc_object.model if npc_object else model,
                provider=npc_object.provider if npc_object else provider,
                final_memory=None # Explicitly None for pending memories
            )
            
            memories_for_approval.append({
                "memory_id": memory_id,
                "content": fact.get('statement', str(fact)),
                "type": fact.get('type', 'unknown'),
                "context": fact.get('source_text', ''),
                "npc": npc_name or "default"
            })
            
            # Collect facts and concepts for the Knowledge Graph
            #if fact.get('type') == 'concept':
            #    kg_concepts_to_save.append({
            #        "name": fact.get('statement'),
            #        "generation": current_kg_generation,
            #        "origin": "organic" # Assuming 'organic' for extracted facts
            #    })
            #else: # It's a fact (or unknown type, treat as fact for KG)
            #    kg_facts_to_save.append({
            #        "statement": fact.get('statement'),
            #        "source_text": fact.get('source_text', conversation_text), # Use source_text if available, else conversation_text
            #        "type": fact.get('type', 'fact'), # Default to 'fact' if type is unknown
            #        "generation": current_kg_generation,
            #        "origin": "organic"
            #    })
            #    if fact.get('concepts'): # If this fact has related concepts
            #        for concept_name in fact.get('concepts'):
            #            fact_to_concept_links_temp[fact.get('statement')].append(concept_name)
    
    # After processing all facts, save them to the KG database in one go
    if kg_facts_to_save or kg_concepts_to_save:
        temp_kg_data = {
            "facts": kg_facts_to_save,
            "concepts": kg_concepts_to_save,
            "generation": current_kg_generation,
            "fact_to_concept_links": fact_to_concept_links_temp,
            "concept_links": [], # Assuming no concept-to-concept links from direct extraction
            "fact_to_fact_links": [] # Assuming no fact-to-fact links from direct extraction
        }
        
        # Get the SQLAlchemy engine using your existing helper function
        db_engine = get_db_connection(app.config.get('DB_PATH'))
        
        # Call the existing save_kg_to_db function
        save_kg_to_db(
            engine=db_engine,
            kg_data=temp_kg_data,
            team_name=team_name or "default",
            npc_name=npc_name or "default",
            directory_path=current_path or "/"
        )
    
    return memories_for_approval
@app.route('/api/finetuned_models', methods=['GET'])
def get_finetuned_models():
    current_path = request.args.get("currentPath")
    
    # Define a list of potential root directories where fine-tuned models might be saved.
    # We'll be very generous here, including both 'models' and 'images' directories
    # at both global and project levels, as the user's logs indicate saving to 'images'.
    potential_root_paths = [
        os.path.expanduser('~/.npcsh/models'),  # Standard global models directory
        os.path.expanduser('~/.npcsh/images'),  # Global images directory (where user's model was saved)
    ]
    if current_path:
        # Add project-specific model directories if a current_path is provided
        project_models_path = os.path.join(current_path, 'models')
        project_images_path = os.path.join(current_path, 'images') # Also check project images directory
        potential_root_paths.extend([project_models_path, project_images_path])
            
    finetuned_models = []
    
    print(f"🌋 Searching for fine-tuned models in potential root paths: {set(potential_root_paths)}") # Use set for unique paths

    for root_path in set(potential_root_paths): # Iterate through unique potential root paths
        if not os.path.exists(root_path) or not os.path.isdir(root_path):
            print(f"🌋 Skipping non-existent or non-directory root path: {root_path}")
            continue

        print(f"🌋 Scanning root path: {root_path}")
        for model_dir_name in os.listdir(root_path):
            full_model_path = os.path.join(root_path, model_dir_name)
            
            if not os.path.isdir(full_model_path):
                print(f"🌋 Skipping {full_model_path}: Not a directory.")
                continue

            # NEW STRATEGY: Check for user's specific output files
            # Look for 'model_final.pt' or the 'checkpoints' directory
            has_model_final_pt = os.path.exists(os.path.join(full_model_path, 'model_final.pt'))
            has_checkpoints_dir = os.path.isdir(os.path.join(full_model_path, 'checkpoints'))

            if has_model_final_pt or has_checkpoints_dir:
                print(f"🌋 Identified fine-tuned model: {model_dir_name} at {full_model_path} (found model_final.pt or checkpoints dir)")
                finetuned_models.append({
                    "value": full_model_path, # This is the path to the directory containing the .pt files
                    "provider": "diffusers",   # Provider is still "diffusers"
                    "display_name": f"{model_dir_name} | Fine-tuned Diffuser"
                })
                continue # Move to the next model_dir_name found in this root_path

            print(f"🌋 Skipping {full_model_path}: No model_final.pt or checkpoints directory found at root.")
    
    print(f"🌋 Finished scanning. Found {len(finetuned_models)} fine-tuned models.")
    return jsonify({"models": finetuned_models, "error": None})

@app.route('/api/finetune_diffusers', methods=['POST'])
def finetune_diffusers():
    data = request.json
    images = data.get('images', [])
    captions = data.get('captions', [])
    output_name = data.get('outputName', 'my_diffusion_model')
    num_epochs = data.get('epochs', 100)
    batch_size = data.get('batchSize', 4)
    learning_rate = data.get('learningRate', 1e-4)
    output_path = data.get('outputPath', '~/.npcsh/models')
    
    print(f"🌋 Finetune Diffusers Request Received!")
    print(f"  Images: {len(images)} files")
    print(f"  Output Name: {output_name}")
    print(f"  Epochs: {num_epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    
    if not images:
        print("🌋 Error: No images provided for finetuning.")
        return jsonify({'error': 'No images provided'}), 400
    
    if not captions or len(captions) != len(images):
        print("🌋 Warning: Captions not provided or mismatching image count. Using empty captions.")
        captions = [''] * len(images)
    
    expanded_images = [os.path.expanduser(p) for p in images]
    output_dir = os.path.expanduser(
        os.path.join(output_path, output_name)
    )
    
    job_id = f"ft_{int(time.time())}"
    finetune_jobs[job_id] = {
        'status': 'running',
        'output_dir': output_dir,
        'epochs': num_epochs,
        'current_epoch': 0,
        'start_time': datetime.datetime.now().isoformat()
    }
    print(f"🌋 Finetuning job {job_id} initialized. Output directory: {output_dir}")
    
    def run_training_async():
        print(f"🌋 Finetuning job {job_id}: Starting asynchronous training thread...")
        try:
            config = DiffusionConfig(
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_model_path=output_dir
            )
            
            print(f"🌋 Finetuning job {job_id}: Calling train_diffusion with config: {config}")
            # Assuming train_diffusion might print its own progress or allow callbacks
            # For more granular logging, you'd need to modify train_diffusion itself
            model_path = train_diffusion(
                expanded_images,
                captions,
                config=config
            )
            
            finetune_jobs[job_id]['status'] = 'complete'
            finetune_jobs[job_id]['model_path'] = model_path
            finetune_jobs[job_id]['end_time'] = datetime.datetime.now().isoformat()
            print(f"🌋 Finetuning job {job_id}: Training complete! Model saved to: {model_path}")
        except Exception as e:
            finetune_jobs[job_id]['status'] = 'error'
            finetune_jobs[job_id]['error_msg'] = str(e)
            finetune_jobs[job_id]['end_time'] = datetime.datetime.now().isoformat()
            print(f"🌋 Finetuning job {job_id}: ERROR during training: {e}")
            traceback.print_exc()
        print(f"🌋 Finetuning job {job_id}: Asynchronous training thread finished.")

    # Start the training in a separate thread
    thread = threading.Thread(target=run_training_async)
    thread.daemon = True # Allow the main program to exit even if this thread is still running
    thread.start()
    
    print(f"🌋 Finetuning job {job_id} successfully launched in background. Returning initial status.")
    return jsonify({
        'status': 'started',
        'jobId': job_id,
        'message': f"Finetuning job '{job_id}' started. Check /api/finetune_status/{job_id} for updates."
    })


@app.route('/api/finetune_status/<job_id>', methods=['GET'])
def finetune_status(job_id):
    if job_id not in finetune_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = finetune_jobs[job_id]
    
    if job['status'] == 'complete':
        return jsonify({
            'complete': True,
            'outputPath': job.get('model_path', job['output_dir'])
        })
    elif job['status'] == 'error':
        return jsonify({'error': job.get('error_msg', 'Unknown error')})
    
    return jsonify({
        'step': job.get('current_epoch', 0),
        'total': job['epochs'],
        'status': 'running'
    })

@app.route("/api/ml/train", methods=["POST"])
def train_ml_model():
    import pickle
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    
    data = request.json
    model_name = data.get("name")
    model_type = data.get("type")
    target = data.get("target")
    features = data.get("features")
    training_data = data.get("data")
    hyperparams = data.get("hyperparameters", {})
    
    df = pd.DataFrame(training_data)
    X = df[features].values
    
    metrics = {}
    model = None
    
    if model_type == "linear_regression":
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    elif model_type == "logistic_regression":
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {"accuracy": accuracy_score(y_test, y_pred)}
    
    elif model_type == "random_forest":
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    elif model_type == "clustering":
        n_clusters = hyperparams.get("n_clusters", 3)
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        metrics = {"inertia": model.inertia_, "n_clusters": n_clusters}
    
    elif model_type == "gradient_boost":
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    model_id = f"{model_name}_{int(time.time())}"
    model_path = os.path.expanduser(f"~/.npcsh/models/{model_id}.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            "model": model,
            "features": features,
            "target": target,
            "type": model_type
        }, f)
    
    return jsonify({
        "model_id": model_id,
        "metrics": metrics,
        "error": None
    })


@app.route("/api/ml/predict", methods=["POST"])
def ml_predict():
    import pickle
    
    data = request.json
    model_name = data.get("model_name")
    input_data = data.get("input_data")
    
    model_dir = os.path.expanduser("~/.npcsh/models/")
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_name)]
    
    if not model_files:
        return jsonify({"error": f"Model {model_name} not found"})
    
    model_path = os.path.join(model_dir, model_files[0])
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data["model"]
    prediction = model.predict([input_data])
    
    return jsonify({
        "prediction": prediction.tolist(),
        "error": None
    })
@app.route("/api/jinx/executions/label", methods=["POST"])
def label_jinx_execution():
    data = request.json
    execution_id = data.get("executionId")
    label = data.get("label")
    
    command_history = CommandHistory(app.config.get('DB_PATH'))
    command_history.label_jinx_execution(execution_id, label)
    
    return jsonify({"success": True, "error": None})


@app.route("/api/npc/executions", methods=["GET"])
def get_npc_executions():
    npc_name = request.args.get("npcName")

    
    command_history = CommandHistory(app.config.get('DB_PATH'))
    executions = command_history.get_npc_executions(npc_name)
    
    return jsonify({"executions": executions, "error": None})


@app.route("/api/npc/executions/label", methods=["POST"])
def label_npc_execution():
    data = request.json
    execution_id = data.get("executionId")
    label = data.get("label")
    
    command_history = CommandHistory(app.config.get('DB_PATH'))
    command_history.label_npc_execution(execution_id, label)
    
    return jsonify({"success": True, "error": None})


@app.route("/api/training/dataset", methods=["POST"])
def build_training_dataset():
    data = request.json
    filters = data.get("filters", {})
    
    command_history = CommandHistory(app.config.get('DB_PATH'))
    dataset = command_history.get_training_dataset(
        include_jinxs=filters.get("jinxs", True),
        include_npcs=filters.get("npcs", True),
        npc_names=filters.get("npc_names")
    )
    
    return jsonify({
        "dataset": dataset,
        "count": len(dataset),
        "error": None
    })
@app.route("/api/save_npc", methods=["POST"])
def save_npc():
    try:
        data = request.json
        npc_data = data.get("npc")
        is_global = data.get("isGlobal")
        current_path = data.get("currentPath")

        if not npc_data or "name" not in npc_data:
            return jsonify({"error": "Invalid NPC data"}), 400

        
        if is_global:
            npc_directory = os.path.expanduser("~/.npcsh/npc_team")
        else:
            npc_directory = os.path.join(current_path, "npc_team")

        
        os.makedirs(npc_directory, exist_ok=True)

        
        yaml_content = f"""name: {npc_data['name']}
primary_directive: "{npc_data['primary_directive']}"
model: {npc_data['model']}
provider: {npc_data['provider']}
api_url: {npc_data.get('api_url', '')}
use_global_jinxs: {str(npc_data.get('use_global_jinxs', True)).lower()}
"""

        
        file_path = os.path.join(npc_directory, f"{npc_data['name']}.npc")
        with open(file_path, "w") as f:
            f.write(yaml_content)

        return jsonify({"message": "NPC saved successfully", "error": None})

    except Exception as e:
        print(f"Error saving NPC: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/jinxs/global")
def get_jinxs_global():
    global_jinx_directory = os.path.expanduser("~/.npcsh/npc_team/jinxs")
    jinx_data = []

    if not os.path.exists(global_jinx_directory):
        return jsonify({"jinxs": [], "error": None})

    for root, dirs, files in os.walk(global_jinx_directory):
        for file in files:
            if file.endswith(".jinx"):
                jinx_path = os.path.join(root, file)
                with open(jinx_path, 'r') as f:
                    raw_data = yaml.safe_load(f)
                
                inputs = []
                for inp in raw_data.get("inputs", []):
                    if isinstance(inp, str):
                        inputs.append(inp)
                    elif isinstance(inp, dict):
                        inputs.append(list(inp.keys())[0])
                    else:
                        inputs.append(str(inp))
                
                rel_path = os.path.relpath(jinx_path, global_jinx_directory)
                path_without_ext = rel_path[:-5]
                
                jinx_data.append({
                    "jinx_name": raw_data.get("jinx_name", file[:-5]),
                    "path": path_without_ext,
                    "description": raw_data.get("description", ""),
                    "inputs": inputs,
                    "steps": raw_data.get("steps", [])
                })

    return jsonify({"jinxs": jinx_data, "error": None})

@app.route("/api/jinxs/project", methods=["GET"])
def get_jinxs_project():
    project_dir = request.args.get("currentPath")
    if not project_dir:
        return jsonify({"jinxs": [], "error": "currentPath required"}), 400

    if not project_dir.endswith("jinxs"):
        project_dir = os.path.join(project_dir, "jinxs")

    jinx_data = []
    if not os.path.exists(project_dir):
        return jsonify({"jinxs": [], "error": None})

    for root, dirs, files in os.walk(project_dir):
        for file in files:
            if file.endswith(".jinx"):
                jinx_path = os.path.join(root, file)
                with open(jinx_path, 'r') as f:
                    raw_data = yaml.safe_load(f)
                
                inputs = []
                for inp in raw_data.get("inputs", []):
                    if isinstance(inp, str):
                        inputs.append(inp)
                    elif isinstance(inp, dict):
                        inputs.append(list(inp.keys())[0])
                    else:
                        inputs.append(str(inp))
                
                rel_path = os.path.relpath(jinx_path, project_dir)
                path_without_ext = rel_path[:-5]
                
                jinx_data.append({
                    "jinx_name": raw_data.get("jinx_name", file[:-5]),
                    "path": path_without_ext,
                    "description": raw_data.get("description", ""),
                    "inputs": inputs,
                    "steps": raw_data.get("steps", [])
                })
    print(jinx_data)
    return jsonify({"jinxs": jinx_data, "error": None})

# ============== SQL Models (npcsql) API Endpoints ==============
@app.route("/api/npcsql/run_model", methods=["POST"])
def run_npcsql_model():
    """Execute a single SQL model using ModelCompiler"""
    try:
        from npcpy.sql.npcsql import ModelCompiler

        data = request.json
        models_dir = data.get("modelsDir")
        model_name = data.get("modelName")
        npc_directory = data.get("npcDirectory", os.path.expanduser("~/.npcsh/npc_team"))
        target_db = data.get("targetDb", os.path.expanduser("~/npcsh_history.db"))

        if not models_dir or not model_name:
            return jsonify({"success": False, "error": "modelsDir and modelName are required"}), 400

        if not os.path.exists(models_dir):
            return jsonify({"success": False, "error": f"Models directory not found: {models_dir}"}), 404

        compiler = ModelCompiler(
            models_dir=models_dir,
            target_engine=target_db,
            npc_directory=npc_directory
        )

        compiler.discover_models()

        if model_name not in compiler.models:
            available = list(compiler.models.keys())
            return jsonify({
                "success": False,
                "error": f"Model '{model_name}' not found. Available: {available}"
            }), 404

        result_df = compiler.execute_model(model_name)
        row_count = len(result_df) if result_df is not None else 0

        return jsonify({
            "success": True,
            "rows": row_count,
            "message": f"Model '{model_name}' executed successfully. {row_count} rows materialized."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/npcsql/run_all", methods=["POST"])
def run_all_npcsql_models():
    """Execute all SQL models in dependency order using ModelCompiler"""
    try:
        from npcpy.sql.npcsql import ModelCompiler

        data = request.json
        models_dir = data.get("modelsDir")
        npc_directory = data.get("npcDirectory", os.path.expanduser("~/.npcsh/npc_team"))
        target_db = data.get("targetDb", os.path.expanduser("~/npcsh_history.db"))

        if not models_dir:
            return jsonify({"success": False, "error": "modelsDir is required"}), 400

        if not os.path.exists(models_dir):
            return jsonify({"success": False, "error": f"Models directory not found: {models_dir}"}), 404

        compiler = ModelCompiler(
            models_dir=models_dir,
            target_engine=target_db,
            npc_directory=npc_directory
        )

        results = compiler.run_all_models()

        summary = {
            name: len(df) if df is not None else 0
            for name, df in results.items()
        }

        return jsonify({
            "success": True,
            "models_executed": list(results.keys()),
            "row_counts": summary,
            "message": f"Executed {len(results)} models successfully."
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/npcsql/models", methods=["GET"])
def list_npcsql_models():
    """List available SQL models in a directory"""
    try:
        from npcpy.sql.npcsql import ModelCompiler

        models_dir = request.args.get("modelsDir")
        if not models_dir:
            return jsonify({"success": False, "error": "modelsDir query param required"}), 400

        if not os.path.exists(models_dir):
            return jsonify({"models": [], "error": None})

        compiler = ModelCompiler(
            models_dir=models_dir,
            target_engine=os.path.expanduser("~/npcsh_history.db"),
            npc_directory=os.path.expanduser("~/.npcsh/npc_team")
        )

        compiler.discover_models()

        models_info = []
        for name, model in compiler.models.items():
            models_info.append({
                "name": name,
                "path": model.path,
                "has_ai_function": model.has_ai_function,
                "dependencies": list(model.dependencies),
                "config": model.config
            })

        return jsonify({"models": models_info, "error": None})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"models": [], "error": str(e)}), 500

@app.route("/api/npc_team_global")
def get_npc_team_global():
    global_npc_directory = os.path.expanduser("~/.npcsh/npc_team")
    npc_data = []

    if not os.path.exists(global_npc_directory):
        return jsonify({"npcs": [], "error": None})

    for file in os.listdir(global_npc_directory):
        if file.endswith(".npc"):
            npc_path = os.path.join(global_npc_directory, file)
            with open(npc_path, 'r') as f:
                raw_data = yaml.safe_load(f)
            
            npc_data.append({
                "name": raw_data.get("name", file[:-4]),
                "primary_directive": raw_data.get("primary_directive", ""),
                "model": raw_data.get("model", ""),
                "provider": raw_data.get("provider", ""),
                "api_url": raw_data.get("api_url", ""),
                "use_global_jinxs": raw_data.get("use_global_jinxs", True),
                "jinxs": raw_data.get("jinxs", "*"),
            })

    return jsonify({"npcs": npc_data, "error": None})


@app.route("/api/npc_team_project", methods=["GET"])
def get_npc_team_project():
    project_npc_directory = request.args.get("currentPath")
    if not project_npc_directory:
        return jsonify({"npcs": [], "error": "currentPath required"}), 400

    if not project_npc_directory.endswith("npc_team"):
        project_npc_directory = os.path.join(
            project_npc_directory, 
            "npc_team"
        )

    npc_data = []

    if not os.path.exists(project_npc_directory):
        return jsonify({"npcs": [], "error": None})

    for file in os.listdir(project_npc_directory):
        if file.endswith(".npc"):
            npc_path = os.path.join(project_npc_directory, file)
            with open(npc_path, 'r') as f:
                raw_npc_data = yaml.safe_load(f)
            
            serialized_npc = {
                "name": raw_npc_data.get("name", file[:-4]),
                "primary_directive": raw_npc_data.get("primary_directive", ""),
                "model": raw_npc_data.get("model", ""),
                "provider": raw_npc_data.get("provider", ""),
                "api_url": raw_npc_data.get("api_url", ""),
                "use_global_jinxs": raw_npc_data.get("use_global_jinxs", True),
                "jinxs": raw_npc_data.get("jinxs", "*"),
            }
            npc_data.append(serialized_npc)

    return jsonify({"npcs": npc_data, "error": None})
        
def get_last_used_model_and_npc_in_directory(directory_path):
    """
    Fetches the model and NPC from the most recent message in any conversation
    within the given directory.
    """
    engine = get_db_connection()
    try:
        with engine.connect() as conn:
            # Normalize path for cross-platform compatibility
            query = text("""
                SELECT model, npc
                FROM conversation_history
                WHERE REPLACE(RTRIM(directory_path, '/\\'), '\\', '/') = :normalized_path
                AND model IS NOT NULL AND npc IS NOT NULL
                AND model != '' AND npc != ''
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
            """)
            normalized_path = normalize_path_for_db(directory_path)
            result = conn.execute(query, {"normalized_path": normalized_path}).fetchone()
            return {"model": result[0], "npc": result[1]} if result else {"model": None, "npc": None}
    except Exception as e:
        print(f"Error getting last used model/NPC for directory {directory_path}: {e}")
        return {"model": None, "npc": None, "error": str(e)}
def get_last_used_model_and_npc_in_conversation(conversation_id):
    """
    Fetches the model and NPC from the most recent message within a specific conversation.
    """
    engine = get_db_connection()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT model, npc
                FROM conversation_history
                WHERE conversation_id = :conversation_id 
                AND model IS NOT NULL AND npc IS NOT NULL 
                AND model != '' AND npc != ''
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
            """)
            result = conn.execute(query, {"conversation_id": conversation_id}).fetchone()
            return {"model": result[0], "npc": result[1]} if result else {"model": None, "npc": None}
    except Exception as e:
        print(f"Error getting last used model/NPC for conversation {conversation_id}: {e}")
        return {"model": None, "npc": None, "error": str(e)}



@app.route("/api/last_used_in_directory", methods=["GET"])
def api_get_last_used_in_directory():
    """API endpoint to get the last used model/NPC in a given directory."""
    current_path = request.args.get("path")
    if not current_path:
        return jsonify({"error": "Path parameter is required."}), 400
    
    result = get_last_used_model_and_npc_in_directory(current_path)
    return jsonify(result)

@app.route("/api/last_used_in_conversation", methods=["GET"])
def api_get_last_used_in_conversation():
    """API endpoint to get the last used model/NPC in a specific conversation."""
    conversation_id = request.args.get("conversationId")
    if not conversation_id:
        return jsonify({"error": "conversationId parameter is required."}), 400
    
    result = get_last_used_model_and_npc_in_conversation(conversation_id)
    return jsonify(result)

def get_ctx_path(is_global, current_path=None, create_default=False):
    """Determines the path to the .ctx file."""
    if is_global:
        ctx_dir = os.path.join(os.path.expanduser("~/.npcsh/npc_team/"))
        ctx_files = glob.glob(os.path.join(ctx_dir, "*.ctx"))
        if ctx_files:
            return ctx_files[0]
        elif create_default:
            return os.path.join(ctx_dir, "team.ctx")
        return None
    else:
        if not current_path:
            return None

        ctx_dir = os.path.join(current_path, "npc_team")
        ctx_files = glob.glob(os.path.join(ctx_dir, "*.ctx"))
        if ctx_files:
            return ctx_files[0]
        elif create_default:
            return os.path.join(ctx_dir, "team.ctx")
        return None


def read_ctx_file(file_path):
    """Reads and parses a YAML .ctx file, normalizing list of strings to list of objects."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = yaml.safe_load(f) or {}

                
                if 'databases' in data and isinstance(data['databases'], list):
                    data['databases'] = [{"value": item} for item in data['databases']]
                
                
                if 'mcp_servers' in data and isinstance(data['mcp_servers'], list):
                    data['mcp_servers'] = [{"value": item} for item in data['mcp_servers']]

                
                if 'preferences' in data and isinstance(data['preferences'], list):
                    data['preferences'] = [{"value": item} for item in data['preferences']]

                # Normalize websites list
                if 'websites' in data and isinstance(data['websites'], list):
                    data['websites'] = [{"value": item} for item in data['websites']]

                return data
            except yaml.YAMLError as e:
                print(f"YAML parsing error in {file_path}: {e}")
                return {"error": "Failed to parse YAML."}
    return {} 

def write_ctx_file(file_path, data):
    """Writes a dictionary to a YAML .ctx file, denormalizing list of objects back to strings."""
    if not file_path:
        return False
    
    
    data_to_save = json.loads(json.dumps(data)) 

    
    if 'databases' in data_to_save and isinstance(data_to_save['databases'], list):
        data_to_save['databases'] = [item.get("value", "") for item in data_to_save['databases'] if isinstance(item, dict)]
    
    
    if 'mcp_servers' in data_to_save and isinstance(data_to_save['mcp_servers'], list):
        data_to_save['mcp_servers'] = [item.get("value", "") for item in data_to_save['mcp_servers'] if isinstance(item, dict)]

    
    if 'preferences' in data_to_save and isinstance(data_to_save['preferences'], list):
        data_to_save['preferences'] = [item.get("value", "") for item in data_to_save['preferences'] if isinstance(item, dict)]

    # Denormalize websites list
    if 'websites' in data_to_save and isinstance(data_to_save['websites'], list):
        data_to_save['websites'] = [item.get("value", "") for item in data_to_save['websites'] if isinstance(item, dict)]

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data_to_save, f, default_flow_style=False, sort_keys=False)
    return True


@app.route("/api/context/global", methods=["GET"])
def get_global_context():
    """Gets the global team.ctx content."""
    try:
        ctx_path = get_ctx_path(is_global=True)
        data = read_ctx_file(ctx_path)
        return jsonify({"context": data, "path": ctx_path, "error": None})
    except Exception as e:
        print(f"Error getting global context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/context/global", methods=["POST"])
def save_global_context():
    """Saves the global team.ctx content."""
    try:
        data = request.json.get("context", {})
        ctx_path = get_ctx_path(is_global=True)
        if write_ctx_file(ctx_path, data):
            return jsonify({"message": "Global context saved.", "error": None})
        else:
            return jsonify({"error": "Failed to write global context file."}), 500
    except Exception as e:
        print(f"Error saving global context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/context/project", methods=["GET"])
def get_project_context():
    """Gets the project-specific team.ctx content."""
    try:
        current_path = request.args.get("path")
        if not current_path:
            return jsonify({"error": "Project path is required."}), 400
        
        ctx_path = get_ctx_path(is_global=False, current_path=current_path)
        data = read_ctx_file(ctx_path)
        return jsonify({"context": data, "path": ctx_path, "error": None})
    except Exception as e:
        print(f"Error getting project context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/context/project", methods=["POST"])
def save_project_context():
    """Saves the project-specific team.ctx content."""
    try:
        data = request.json
        current_path = data.get("path")
        context_data = data.get("context", {})

        if not current_path:
            return jsonify({"error": "Project path is required."}), 400

        ctx_path = get_ctx_path(is_global=False, current_path=current_path)
        if write_ctx_file(ctx_path, context_data):
            return jsonify({"message": "Project context saved.", "error": None})
        else:
            return jsonify({"error": "Failed to write project context file."}), 500
    except Exception as e:
        print(f"Error saving project context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/context/project/init", methods=["POST"])
def init_project_team():
    """Initialize a new npc_team folder in the project directory."""
    try:
        data = request.json
        project_path = data.get("path")

        if not project_path:
            return jsonify({"error": "Project path is required."}), 400

        # Use the existing initialize_npc_project function
        result = initialize_npc_project(directory=project_path)
        return jsonify({"message": "Project team initialized.", "path": result, "error": None})
    except Exception as e:
        print(f"Error initializing project team: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/npcsh/check", methods=["GET"])
def check_npcsh_folder():
    """Check if ~/.npcsh folder exists and has a valid npc_team."""
    try:
        npcsh_path = os.path.expanduser("~/.npcsh")
        npc_team_path = os.path.join(npcsh_path, "npc_team")

        exists = os.path.exists(npcsh_path)
        has_npc_team = os.path.exists(npc_team_path)
        has_forenpc = os.path.exists(os.path.join(npc_team_path, "forenpc.npc")) if has_npc_team else False

        return jsonify({
            "exists": exists,
            "has_npc_team": has_npc_team,
            "has_forenpc": has_forenpc,
            "path": npcsh_path,
            "error": None
        })
    except Exception as e:
        print(f"Error checking npcsh folder: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/npcsh/init", methods=["POST"])
def init_npcsh_folder():
    """Initialize the ~/.npcsh folder with a default npc_team."""
    try:
        npcsh_path = os.path.expanduser("~/.npcsh")
        result = initialize_npc_project(directory=npcsh_path)
        return jsonify({
            "message": result,
            "path": npcsh_path,
            "error": None
        })
    except Exception as e:
        print(f"Error initializing npcsh folder: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/context/websites", methods=["GET"])
def get_context_websites():
    """Gets the websites list from a .ctx file."""
    try:
        current_path = request.args.get("path")
        is_global = request.args.get("global", "false").lower() == "true"
        
        ctx_path = get_ctx_path(is_global=is_global, current_path=current_path)
        data = read_ctx_file(ctx_path)
        
        websites = data.get("websites", [])
        # Normalize to list of objects if needed
        if isinstance(websites, list):
            normalized = []
            for item in websites:
                if isinstance(item, str):
                    normalized.append({"value": item})
                elif isinstance(item, dict):
                    normalized.append(item)
            websites = normalized
        
        return jsonify({
            "websites": websites,
            "path": ctx_path,
            "error": None
        })
    except Exception as e:
        print(f"Error getting websites from context: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/context/websites", methods=["POST"])
def save_context_websites():
    """Saves the websites list to a .ctx file."""
    try:
        data = request.json
        websites = data.get("websites", [])
        current_path = data.get("path")
        is_global = data.get("global", False)
        
        ctx_path = get_ctx_path(is_global=is_global, current_path=current_path, create_default=True)
        
        if not ctx_path:
            return jsonify({"error": "Could not determine ctx file path. Provide a path or use global=true."}), 400
        
        # Read existing ctx data
        existing_data = read_ctx_file(ctx_path) or {}
        
        # Normalize websites to list of strings for YAML storage
        normalized_websites = []
        for item in websites:
            if isinstance(item, dict) and "value" in item:
                normalized_websites.append(item["value"])
            elif isinstance(item, str):
                normalized_websites.append(item)
        
        existing_data["websites"] = normalized_websites
        
        if write_ctx_file(ctx_path, existing_data):
            return jsonify({
                "message": "Websites saved to context.",
                "websites": [{"value": w} for w in normalized_websites],
                "path": ctx_path,
                "error": None
            })
        else:
            return jsonify({"error": "Failed to write context file."}), 500
            
    except Exception as e:
        print(f"Error saving websites to context: {e}")
        return jsonify({"error": str(e)}), 500







@app.route("/api/get_attachment_response", methods=["POST"])
def get_attachment_response():
    data = request.json
    attachments = data.get("attachments", [])
    messages = data.get("messages")
    conversation_id = data.get("conversationId")
    current_path = data.get("currentPath")
    command_history = CommandHistory(app.config.get('DB_PATH'))
    model = data.get("model")
    npc_name = data.get("npc")
    npc_source = data.get("npcSource", "global")
    team = data.get("team")
    provider = data.get("provider")
    message_id = data.get("messageId")
    
    
    if current_path:
        loaded_vars = load_project_env(current_path)
        print(f"Loaded project env variables for attachment response: {list(loaded_vars.keys())}")
    
    
    npc_object = None
    if npc_name:
        db_conn = get_db_connection()
        npc_object = load_npc_by_name_and_source(npc_name, npc_source, db_conn, current_path)
        
        if not npc_object and npc_source == 'project':
            print(f"NPC {npc_name} not found in project directory, trying global...")
            npc_object = load_npc_by_name_and_source(npc_name, 'global', db_conn)
            
        if npc_object:
            print(f"Successfully loaded NPC {npc_name} from {npc_source} directory")
        else:
            print(f"Warning: Could not load NPC {npc_name}")
    
    images = []
    attachments_loaded = []
    
    for attachment in attachments:
        extension = attachment["name"].split(".")[-1]
        extension_mapped = extension_map.get(extension.upper(), "others")
        file_path = os.path.expanduser("~/.npcsh/" + extension_mapped + "/" + attachment["name"])
        
        if extension_mapped == "images":
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(attachment["path"])
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            img.save(file_path, optimize=True, quality=50)
            images.append(file_path)
            attachments_loaded.append({
                "name": attachment["name"], "type": extension_mapped,
                "data": img_byte_arr.read(), "size": os.path.getsize(file_path)
            })

    message_to_send = messages[-1]["content"]
    if isinstance(message_to_send, list):
        message_to_send = message_to_send[0]

    response = get_llm_response(
        message_to_send,
        images=images,
        messages=messages,
        model=model,
        provider=provider,
        npc=npc_object,
    )
    
    messages = response["messages"]
    response = response["response"]

    
    save_conversation_message(
        command_history, 
        conversation_id, 
        "user", 
        message_to_send, 
        wd=current_path, 
        team=team, 
        model=model, 
        provider=provider, 
        npc=npc_name, 
        attachments=attachments_loaded
    )

    save_conversation_message(
        command_history, 
        conversation_id, 
        "assistant", 
        response,
        wd=current_path, 
        team=team, 
        model=model, 
        provider=provider,
        npc=npc_name, 
        attachments=attachments_loaded, 
        message_id=message_id
    )
    
    return jsonify({
        "status": "success",
        "message": response,
        "conversationId": conversation_id,
        "messages": messages,
    })

                                                                                                                                                                                                           
IMAGE_MODELS = {
    "openai": [
        {"value": "gpt-image-1.5", "display_name": "GPT-Image-1.5"},
        {"value": "gpt-image-1", "display_name": "GPT-Image-1"},
        {"value": "dall-e-3", "display_name": "DALL-E 3"},
        {"value": "dall-e-2", "display_name": "DALL-E 2"},
    ],
    "gemini": [
        {"value": "gemini-3-pro-image-preview", "display_name": "Gemini 3 Pro Image"},
        {"value": "gemini-2.5-flash-image-preview", "display_name": "Gemini 2.5 Flash Image"},
        {"value": "imagen-3.0-generate-002", "display_name": "Imagen 3.0 Generate (Preview)"},
    ],
    "diffusers": [
        {"value": "runwayml/stable-diffusion-v1-5", "display_name": "Stable Diffusion v1.5"},
    ],
}
# In npcpy/serve.py, find the @app.route('/api/finetuned_models', methods=['GET'])
# and replace the entire function with this:

# This is now an internal helper function, not a Flask route.
def _get_finetuned_models_internal(current_path=None): # Renamed to indicate internal use
    
    # Define a list of potential root directories where fine-tuned models might be saved.
    potential_root_paths = [
        os.path.expanduser('~/.npcsh/models'),  # Standard global models directory
        os.path.expanduser('~/.npcsh/images'),  # Global images directory (where user's model was saved)
    ]
    if current_path:
        # Add project-specific model directories if a current_path is provided
        project_models_path = os.path.join(current_path, 'models')
        project_images_path = os.path.join(current_path, 'images') # Also check project images directory
        potential_root_paths.extend([project_models_path, project_images_path])
            
    finetuned_models = []
    
    print(f"🌋 (Internal) Searching for fine-tuned models in potential root paths: {set(potential_root_paths)}")

    for root_path in set(potential_root_paths):
        if not os.path.exists(root_path) or not os.path.isdir(root_path):
            print(f"🌋 (Internal) Skipping non-existent or non-directory root path: {root_path}")
            continue

        print(f"🌋 (Internal) Scanning root path: {root_path}")
        for model_dir_name in os.listdir(root_path):
            full_model_path = os.path.join(root_path, model_dir_name)
            
            if not os.path.isdir(full_model_path):
                print(f"🌋 (Internal) Skipping {full_model_path}: Not a directory.")
                continue

            # Check for 'model_final.pt' or the 'checkpoints' directory
            has_model_final_pt = os.path.exists(os.path.join(full_model_path, 'model_final.pt'))
            has_checkpoints_dir = os.path.isdir(os.path.join(full_model_path, 'checkpoints'))

            if has_model_final_pt or has_checkpoints_dir:
                print(f"🌋 (Internal) Identified fine-tuned model: {model_dir_name} at {full_model_path} (found model_final.pt or checkpoints dir)")
                finetuned_models.append({
                    "value": full_model_path, # This is the path to the directory containing the .pt files
                    "provider": "diffusers",   # Provider is still "diffusers"
                    "display_name": f"{model_dir_name} | Fine-tuned Diffuser"
                })
                continue

            print(f"🌋 (Internal) Skipping {full_model_path}: No model_final.pt or checkpoints directory found at root.")
    
    print(f"🌋 (Internal) Finished scanning. Found {len(finetuned_models)} fine-tuned models.")
    # <--- CRITICAL FIX: Directly return the list of models, not a Flask Response
    return {"models": finetuned_models, "error": None} # Return a dict for consistency
def get_available_image_models(current_path=None):
    """
    Retrieves available image generation models based on environment variables
    and predefined configurations, including locally fine-tuned Diffusers models.
    """
    
    if current_path:
        load_project_env(current_path) 
    
    all_image_models = []

    # Add models configured via environment variables
    env_image_model = os.getenv("NPCSH_IMAGE_MODEL")
    env_image_provider = os.getenv("NPCSH_IMAGE_PROVIDER")

    if env_image_model and env_image_provider:
        all_image_models.append({
            "value": env_image_model,
            "provider": env_image_provider,
            "display_name": f"{env_image_model} | {env_image_provider} (Configured)"
        })

    # Add predefined models (OpenAI, Gemini, and standard Diffusers)
    for provider_key, models_list in IMAGE_MODELS.items():
        if provider_key == "openai":
            if os.environ.get("OPENAI_API_KEY"):
                all_image_models.extend([
                    {**model, "provider": provider_key, "display_name": f"{model['display_name']} | {provider_key}"}
                    for model in models_list
                ])
        elif provider_key == "gemini":
            if os.environ.get("GEMINI_API_KEY"): 
                all_image_models.extend([
                    {**model, "provider": provider_key, "display_name": f"{model['display_name']} | {provider_key}"}
                    for model in models_list
                ])
        elif provider_key == "diffusers": # This entry in IMAGE_MODELS is for standard diffusers
            all_image_models.extend([
                {**model, "provider": provider_key, "display_name": f"{model['display_name']} | {provider_key}"}
                for model in models_list
            ])
        
    # <--- CRITICAL FIX: Directly call the internal helper function for fine-tuned models
    try:
        finetuned_data_result = _get_finetuned_models_internal(current_path)
        if finetuned_data_result and finetuned_data_result.get("models"):
            all_image_models.extend(finetuned_data_result["models"])
        else:
            print(f"No fine-tuned models returned by internal helper or an error occurred internally.")
            if finetuned_data_result.get("error"):
                print(f"Internal error in _get_finetuned_models_internal: {finetuned_data_result['error']}")
    except Exception as e:
        print(f"Error calling _get_finetuned_models_internal: {e}")

    # Deduplicate models
    seen_models = set()
    unique_models = []
    for model_entry in all_image_models:
        key = (model_entry["value"], model_entry["provider"])
        if key not in seen_models:
            seen_models.add(key)
            unique_models.append(model_entry)

    # Return the combined, deduplicated list of models as a dictionary with a 'models' key
    return unique_models

@app.route('/api/generative_fill', methods=['POST'])
def generative_fill():
    data = request.get_json()
    image_path = data.get('imagePath')
    mask_data = data.get('mask')
    prompt = data.get('prompt')
    model = data.get('model')
    provider = data.get('provider')
    
    if not all([image_path, mask_data, prompt, model, provider]):
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        image_path = os.path.expanduser(image_path)
        
        mask_b64 = mask_data.split(',')[1] if ',' in mask_data else mask_data
        mask_bytes = base64.b64decode(mask_b64)
        mask_image = Image.open(BytesIO(mask_bytes))
        
        original_image = Image.open(image_path)
        
        if provider == 'openai':
            result = inpaint_openai(original_image, mask_image, prompt, model)
        elif provider == 'gemini':
            result = inpaint_gemini(original_image, mask_image, prompt, model)
        elif provider == 'diffusers':
            result = inpaint_diffusers(original_image, mask_image, prompt, model)
        else:
            return jsonify({"error": f"Provider {provider} not supported"}), 400
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inpaint_{timestamp}.png"
        save_dir = os.path.dirname(image_path)
        result_path = os.path.join(save_dir, filename)
        
        result.save(result_path)
        
        return jsonify({"resultPath": result_path, "error": None})
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def inpaint_openai(image, mask, prompt, model):
    import io
    from openai import OpenAI
    from PIL import Image
    import base64
    
    client = OpenAI()
    
    original_size = image.size
    
    if model == 'dall-e-2':
        valid_sizes = ['256x256', '512x512', '1024x1024']
        max_dim = max(image.width, image.height)
        
        if max_dim <= 256:
            target_size = (256, 256)
            size_str = '256x256'
        elif max_dim <= 512:
            target_size = (512, 512)
            size_str = '512x512'
        else:
            target_size = (1024, 1024)
            size_str = '1024x1024'
    else:
        valid_sizes = {
            (1024, 1024): "1024x1024",
            (1024, 1536): "1024x1536", 
            (1536, 1024): "1536x1024"
        }
        
        target_size = (1024, 1024)
        for size in valid_sizes.keys():
            if image.width > image.height and size == (1536, 1024):
                target_size = size
                break
            elif image.height > image.width and size == (1024, 1536):
                target_size = size
                break
        
        size_str = valid_sizes[target_size]
    
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    resized_mask = mask.resize(target_size, Image.Resampling.LANCZOS)
    
    img_bytes = io.BytesIO()
    resized_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_bytes.name = 'image.png'
    
    mask_bytes = io.BytesIO()
    resized_mask.save(mask_bytes, format='PNG')
    mask_bytes.seek(0)
    mask_bytes.name = 'mask.png'
    
    response = client.images.edit(
        model=model,
        image=img_bytes,
        mask=mask_bytes,
        prompt=prompt,
        n=1,
        size=size_str
    )
    
    if response.data[0].url:
        import requests
        img_data = requests.get(response.data[0].url).content
    elif hasattr(response.data[0], 'b64_json'):
        img_data = base64.b64decode(response.data[0].b64_json)
    else:
        raise Exception("No image data in response")
    
    result_image = Image.open(io.BytesIO(img_data))
    return result_image.resize(original_size, Image.Resampling.LANCZOS)

def inpaint_diffusers(image, mask, prompt, model):
    from diffusers import StableDiffusionInpaintPipeline
    import torch
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask
    ).images[0]
    
    return result
def inpaint_gemini(image, mask, prompt, model):
    from npcpy.gen.image_gen import generate_image
    import io
    import numpy as np
    
    mask_np = np.array(mask.convert('L'))
    ys, xs = np.where(mask_np > 128)
    
    if len(xs) == 0:
        return image
    
    x_center = int(np.mean(xs))
    y_center = int(np.mean(ys))
    width_pct = (xs.max() - xs.min()) / image.width * 100
    height_pct = (ys.max() - ys.min()) / image.height * 100
    
    position = "center"
    if y_center < image.height / 3:
        position = "top"
    elif y_center > 2 * image.height / 3:
        position = "bottom"
    
    if x_center < image.width / 3:
        position += " left"
    elif x_center > 2 * image.width / 3:
        position += " right"
    
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    full_prompt =  f"""Using the provided image, change only the region in the {position} 
        approximately {int(width_pct)}% wide by {int(height_pct)}% tall) to: {prompt}. 
        
        Keep everything else exactly the same, matching the original lighting and style.
        You are in-painting the image. You should not be changing anything other than what was requested in prompt: {prompt}
        """    
    results = generate_image(
        prompt=full_prompt,
        model=model,
        provider='gemini',
        attachments=[img_bytes],
        n_images=1
    )
    
    return results[0] if results else None

@app.route('/api/generate_images', methods=['POST'])
def generate_images():
    data = request.get_json()
    prompt = data.get('prompt')
    n = data.get('n', 1)
    model_name = data.get('model')
    provider_name = data.get('provider')
    attachments = data.get('attachments', [])
    base_filename = data.get('base_filename', 'vixynt_gen')  
    save_dir = data.get('currentPath', '~/.npcsh/images')     

    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400

    if not model_name or not provider_name:
        return jsonify({"error": "Image model and provider are required."}), 400

    
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename_with_time = f"{base_filename}_{timestamp}"

    generated_images_base64 = []
    generated_filenames = []
    command_history = CommandHistory(app.config.get('DB_PATH'))
    
    try:
        
        input_images = []
        attachments_loaded = []
        
        if attachments:
            for attachment in attachments:
                print(attachment)
                if isinstance(attachment, dict) and 'path' in attachment:
                    image_path = attachment['path']
                    if os.path.exists(image_path):
                        try:
                            pil_img = Image.open(image_path)
                            pil_img = pil_img.convert("RGB")
                            pil_img.thumbnail((1024, 1024))
                            input_images.append(pil_img)
                            
                            compressed_bytes = BytesIO()
                            pil_img.save(compressed_bytes, format="JPEG", quality=85, optimize=True)
                            img_data = compressed_bytes.getvalue()
                            attachments_loaded.append({
                                "name": os.path.basename(image_path),
                                "type": "images",
                                "data": img_data,
                                "size": len(img_data)
                            })
                        except Exception as e:
                            print(f"Warning: Could not load attachment image {image_path}: {e}")

        
        images_list = gen_image(
            prompt, 
            model=model_name, 
            provider=provider_name, 
            n_images=n,
            input_images=input_images if input_images else None
        )
        print(images_list)
        if not isinstance(images_list, list):
            images_list = [images_list] if images_list is not None else []

        generated_attachments = []
        for i, pil_image in enumerate(images_list):
            if isinstance(pil_image, Image.Image):
                
                filename = f"{base_filename_with_time}_{i+1:03d}.png" if n > 1 else f"{base_filename_with_time}.png"
                filepath = os.path.join(save_dir, filename)
                print(f'saved file to {filepath}')
                
                
                pil_image.save(filepath, format="PNG")
                generated_filenames.append(filepath)
                
                
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_data = buffered.getvalue()
                
                generated_attachments.append({
                    "name": filename,
                    "type": "images", 
                    "data": img_data,
                    "size": len(img_data)
                })
                
                
                img_str = base64.b64encode(img_data).decode("utf-8")
                generated_images_base64.append(f"data:image/png;base64,{img_str}")
            else:
                print(f"Warning: gen_image returned non-PIL object ({type(pil_image)}). Skipping image conversion.")

        
        generation_id = generate_message_id()
        
        
        save_conversation_message(
            command_history,
            generation_id,  
            "user",
            f"Generate {n} image(s): {prompt}",
            wd=save_dir,
            model=model_name,
            provider=provider_name,
            npc="vixynt",
            attachments=attachments_loaded,
            message_id=generation_id
        )
        
        
        response_message = f"Generated {len(generated_images_base64)} image(s) saved to {save_dir}"
        save_conversation_message(
            command_history,
            generation_id,  
            "assistant", 
            response_message,
            wd=save_dir,
            model=model_name,
            provider=provider_name,
            npc="vixynt",
            attachments=generated_attachments,
            message_id=generate_message_id()
        )
        
        return jsonify({
            "images": generated_images_base64, 
            "filenames": generated_filenames,
            "generation_id": generation_id,  
            "error": None
        })
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        traceback.print_exc()
        return jsonify({"images": [], "filenames": [], "error": str(e)}), 500



@app.route("/api/mcp_tools", methods=["GET"])
def get_mcp_tools():
    """
    API endpoint to retrieve the list of tools available from a given MCP server script.
    It will try to use an existing client from corca_states if available and matching,
    otherwise it creates a temporary client.
    """
    raw_server_path = request.args.get("mcpServerPath")
    current_path_arg = request.args.get("currentPath")
    conversation_id = request.args.get("conversationId")
    npc_name = request.args.get("npc")
    selected_filter = request.args.get("selected", "")
    selected_names = [s.strip() for s in selected_filter.split(",") if s.strip()]
    
    if not raw_server_path:
        return jsonify({"error": "mcpServerPath parameter is required."}), 400

    # Normalize/expand the provided path so cwd/tilde don't break imports
    resolved_path = resolve_mcp_server_path(
        current_path=current_path_arg,
        explicit_path=raw_server_path,
        force_global=False
    )
    server_path = os.path.abspath(os.path.expanduser(resolved_path))

    # MCPClientNPC is defined inline at the top of this file
    temp_mcp_client = None
    jinx_tools = []
    try:
        
        if conversation_id and npc_name and hasattr(app, 'corca_states'):
            state_key = f"{conversation_id}_{npc_name or 'default'}"
            if state_key in app.corca_states:
                existing_corca_state = app.corca_states[state_key]
                if hasattr(existing_corca_state, 'mcp_client') and existing_corca_state.mcp_client \
                   and existing_corca_state.mcp_client.server_script_path == server_path:
                    print(f"Using existing MCP client for {state_key} to fetch tools.")
                    temp_mcp_client = existing_corca_state.mcp_client
                    tools = temp_mcp_client.available_tools_llm
                    if selected_names:
                        tools = [t for t in tools if t.get("function", {}).get("name") in selected_names]
                    return jsonify({"tools": tools, "error": None})

        
        print(f"Creating a temporary MCP client to fetch tools for {server_path}.")
        temp_mcp_client = MCPClientNPC()
        if temp_mcp_client.connect_sync(server_path):
            tools = temp_mcp_client.available_tools_llm
            # Append Jinx-derived tools discovered from global/project jinxs
            try:
                jinx_dirs = []
                if current_path_arg:
                    proj_jinx_dir = os.path.join(os.path.abspath(current_path_arg), "npc_team", "jinxs")
                    if os.path.isdir(proj_jinx_dir):
                        jinx_dirs.append(proj_jinx_dir)
                global_jinx_dir = os.path.expanduser("~/.npcsh/npc_team/jinxs")
                if os.path.isdir(global_jinx_dir):
                    jinx_dirs.append(global_jinx_dir)
                all_jinxs = []
                for d in jinx_dirs:
                    all_jinxs.extend(load_jinxs_from_directory(d))
                if all_jinxs:
                    jinx_tools = list(build_jinx_tool_catalog({j.jinx_name: j for j in all_jinxs}).values())
                    print(f"[MCP] Discovered {len(jinx_tools)} Jinx tools for listing.")
                    tools = tools + jinx_tools
            except Exception as e:
                print(f"[MCP] Error discovering Jinx tools for listing: {e}")
            if selected_names:
                tools = [t for t in tools if t.get("function", {}).get("name") in selected_names]
            return jsonify({"tools": tools, "error": None})
        else:
            return jsonify({"error": f"Failed to connect to MCP server at {server_path}."}), 500
    except FileNotFoundError as e:
        return jsonify({"error": f"MCP Server script not found: {e}"}), 404
    except ValueError as e:
        return jsonify({"error": f"Invalid MCP Server script: {e}"}), 400
    except Exception as e:
        print(f"Error getting MCP tools for {server_path}: {traceback.format_exc()}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
    finally:
        
        if temp_mcp_client and temp_mcp_client.session and (
            not (conversation_id and npc_name and hasattr(app, 'corca_states') and state_key in app.corca_states and getattr(app.corca_states[state_key], 'mcp_client', None) == temp_mcp_client)
        ):
            print(f"Disconnecting temporary MCP client for {server_path}.")
            temp_mcp_client.disconnect_sync()


@app.route("/api/mcp/server/resolve", methods=["GET"])
def api_mcp_resolve():
    current_path = request.args.get("currentPath")
    explicit = request.args.get("serverPath")
    try:
        resolved = resolve_mcp_server_path(current_path=current_path, explicit_path=explicit)
        return jsonify({"serverPath": resolved, "error": None})
    except Exception as e:
        return jsonify({"serverPath": None, "error": str(e)}), 500


@app.route("/api/mcp/server/start", methods=["POST"])
def api_mcp_start():
    data = request.get_json() or {}
    current_path = data.get("currentPath")
    explicit = data.get("serverPath")
    try:
        server_path = resolve_mcp_server_path(current_path=current_path, explicit_path=explicit)
        result = mcp_server_manager.start(server_path)
        return jsonify({**result, "error": None})
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/mcp/server/stop", methods=["POST"])
def api_mcp_stop():
    data = request.get_json() or {}
    explicit = data.get("serverPath")
    if not explicit:
        return jsonify({"error": "serverPath is required to stop a server."}), 400
    try:
        result = mcp_server_manager.stop(explicit)
        return jsonify({**result, "error": None})
    except Exception as e:
        print(f"Error stopping MCP server: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/mcp/server/status", methods=["GET"])
def api_mcp_status():
    explicit = request.args.get("serverPath")
    current_path = request.args.get("currentPath")
    try:
        if explicit:
            result = mcp_server_manager.status(explicit)
        else:
            resolved = resolve_mcp_server_path(current_path=current_path, explicit_path=explicit)
            result = mcp_server_manager.status(resolved)
        return jsonify({**result, "running": result.get("status") == "running", "all": mcp_server_manager.running(), "error": None})
    except Exception as e:
        print(f"Error checking MCP server status: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/image_models", methods=["GET"]) 
def get_image_models_api():
    """
    API endpoint to retrieve available image generation models.
    """
    current_path = request.args.get("currentPath")
    try:
        image_models = get_available_image_models(current_path)
        print('image models', image_models)
        return jsonify({"models": image_models, "error": None})
    except Exception as e:
        print(f"Error getting available image models: {str(e)}")
        traceback.print_exc()
        return jsonify({"models": [], "error": str(e)}), 500






def _run_stream_post_processing(
    conversation_turn_text,
    conversation_id,
    command_history,
    npc_name,
    team_name,
    current_path,
    model,
    provider,
    npc_object,
    messages # For context compression
):
    """
    Runs memory extraction and context compression in a background thread.
    These operations will not block the main stream.
    """
    print(f"🌋 Background task started for conversation {conversation_id}!")

    # Memory extraction and KG fact insertion
    try:
        if len(conversation_turn_text) > 50: # Only extract memories if the turn is substantial
            memories_for_approval = extract_and_store_memories(
                conversation_turn_text,
                conversation_id,
                command_history,
                npc_name,
                team_name,
                current_path,
                model,
                provider,
                npc_object
            )
            if memories_for_approval:
                print(f"🔥 Background: Extracted {len(memories_for_approval)} memories for approval for conversation {conversation_id}. Stored as pending in the database (table: memory_lifecycle).")
        else:
            print(f"Background: Conversation turn too short ({len(conversation_turn_text)} chars) for memory extraction. Skipping.")
    except Exception as e:
        print(f"🌋 Background: Error during memory extraction and KG insertion for conversation {conversation_id}: {e}")
        traceback.print_exc()

    # Context compression using breathe from llm_funcs
    try:
        if len(messages) > 30: # Use the threshold specified in your request
            # Directly call breathe for summarization
            breathe_result = breathe(
                messages=messages,
                model=model,
                provider=provider,
                npc=npc_object # Pass npc for context if available
            )
            compressed_output = breathe_result.get('output', '')
            
            if compressed_output:
                # Save the compressed context as a new system message in conversation_history
                compressed_message_id = generate_message_id()
                save_conversation_message(
                    command_history,
                    conversation_id,
                    "system", # Role for compressed context
                    f"[AUTOMATIC CONTEXT COMPRESSION]: {compressed_output}",
                    wd=current_path,
                    model=model, # Use the same model/provider that generated the summary
                    provider=provider,
                    npc=npc_name, # Associate with the NPC
                    team=team_name, # Associate with the team
                    message_id=compressed_message_id
                )
                print(f"💨 Background: Compressed context for conversation {conversation_id} saved as new system message: {compressed_output[:100]}...")
            else:
                print(f"Background: Context compression returned no output for conversation {conversation_id}. Skipping saving.")
        else:
            print(f"Background: Conversation messages count ({len(messages)}) below threshold for context compression. Skipping.")
    except Exception as e:
        print(f"🌋 Background: Error during context compression with breathe for conversation {conversation_id}: {e}")
        traceback.print_exc()

    print(f"🌋 Background task finished for conversation {conversation_id}!")




@app.route("/api/text_predict", methods=["POST"])
def text_predict():
    data = request.json

    stream_id = data.get("streamId")
    if not stream_id:
        stream_id = str(uuid.uuid4())

    with cancellation_lock:
        cancellation_flags[stream_id] = False

    print(f"Starting text prediction stream with ID: {stream_id}")
    print('data')


    text_content = data.get("text_content", "")
    cursor_position = data.get("cursor_position", len(text_content))
    current_path = data.get("currentPath")
    model = data.get("model")
    provider = data.get("provider")
    context_type = data.get("context_type", "general") # e.g., 'code', 'chat', 'general'
    file_path = data.get("file_path") # Optional: for code context

    if current_path:
        load_project_env(current_path)

    text_before_cursor = text_content[:cursor_position]


    if context_type == 'code':
        prompt_for_llm = f"You are an AI code completion assistant. Your task is to complete the provided code snippet.\nYou MUST ONLY output the code that directly completes the snippet.\nDO NOT include any explanations, comments, or additional text.\nDO NOT wrap the completion in markdown code blocks.\n\nHere is the code context where the completion should occur (file: {file_path or 'unknown'}):\n\n{text_before_cursor}\n\nPlease provide the completion starting from the end of the last line shown.\n"
        system_prompt = "You are an AI code completion assistant. Only provide code. Do not add explanations or any other text."
    elif context_type == 'chat':
        prompt_for_llm = f"You are an AI chat assistant. Your task is to provide a natural and helpful completion to the user's ongoing message.\nYou MUST ONLY output the text that directly completes the message.\nDO NOT include any explanations or additional text.\n\nHere is the message context where the completion should occur:\n\n{text_before_cursor}\n\nPlease provide the completion starting from the end of the last line shown.\n"
        system_prompt = "You are an AI chat assistant. Only provide natural language completion. Do not add explanations or any other text."
    else: # general text prediction
        prompt_for_llm = f"You are an AI text completion assistant. Your task is to provide a natural and helpful completion to the user's ongoing text.\nYou MUST ONLY output the text that directly completes the snippet.\nDO NOT include any explanations or additional text.\n\nHere is the text context where the completion should occur:\n\n{text_before_cursor}\n\nPlease provide the completion starting from the end of the last line shown.\n"
        system_prompt = "You are an AI text completion assistant. Only provide natural language completion. Do not add explanations or any other text."


    npc_object = None # For prediction, we don't necessarily use a specific NPC

    messages_for_llm = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_for_llm}
    ]

    def event_stream_text_predict(current_stream_id):
        complete_prediction = []
        try:
            stream_response_generator = get_llm_response(
                prompt_for_llm,
                messages=messages_for_llm,
                model=model,
                provider=provider,
                npc=npc_object,
                stream=True,
            )

            # get_llm_response returns a dict with 'response' as a generator when stream=True
            if isinstance(stream_response_generator, dict) and 'response' in stream_response_generator:
                stream_generator = stream_response_generator['response']
            else:
                # Fallback for non-streaming LLM responses or errors
                output_content = ""
                if isinstance(stream_response_generator, dict) and 'output' in stream_response_generator:
                    output_content = stream_response_generator['output']
                elif isinstance(stream_response_generator, str):
                    output_content = stream_response_generator

                yield f"data: {json.dumps({'choices': [{'delta': {'content': output_content}}]})}\n\n"
                yield f"data: [DONE]\n\n"
                return


            for response_chunk in stream_generator:
                with cancellation_lock:
                    if cancellation_flags.get(current_stream_id, False):
                        print(f"Cancellation flag triggered for {current_stream_id}. Breaking loop.")
                        break

                chunk_content = ""
                # Handle different LLM API response formats
                if "hf.co" in model or (provider == 'ollama' and 'gpt-oss' not in model): # Heuristic for Ollama/HF models
                    chunk_content = response_chunk["message"]["content"] if "message" in response_chunk and "content" in response_chunk["message"] else ""
                else: # Assume OpenAI-like streaming format
                    chunk_content = "".join(choice.delta.content for choice in response_chunk.choices if choice.delta.content is not None)

                print(chunk_content, end='')

                if chunk_content:
                    complete_prediction.append(chunk_content)
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk_content}}]})}\n\n"

        except Exception as e:
            print(f"\nAn exception occurred during text prediction streaming for {current_stream_id}: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        finally:
            print(f"\nText prediction stream {current_stream_id} finished.")
            yield f"data: [DONE]\n\n" # Signal end of stream
            with cancellation_lock:
                if current_stream_id in cancellation_flags:
                    del cancellation_flags[current_stream_id]
                    print(f"Cleaned up cancellation flag for stream ID: {current_stream_id}")

    return Response(event_stream_text_predict(stream_id), mimetype="text/event-stream")

@app.route("/api/stream", methods=["POST"])
def stream():
    data = request.json
    
    stream_id = data.get("streamId")
    if not stream_id:
        import uuid
        stream_id = str(uuid.uuid4())

    with cancellation_lock:
        cancellation_flags[stream_id] = False
    print(f"Starting stream with ID: {stream_id}")
    
    commandstr = data.get("commandstr")
    conversation_id = data.get("conversationId")
    if not conversation_id:
        return jsonify({"error": "conversationId is required"}), 400
    model = data.get("model", None)
    provider = data.get("provider", None)
    if provider is None:
        provider = available_models.get(model)

    npc_name = data.get("npc", None)
    npc_source = data.get("npcSource", "global")
    current_path = data.get("currentPath")
    is_resend = data.get("isResend", False)  # ADD THIS LINE

    if current_path:
        loaded_vars = load_project_env(current_path)
        print(f"Loaded project env variables for stream request: {list(loaded_vars.keys())}")
    
    npc_object = None
    team_object = None
    team = None  
    tool_results_for_db = []
    if npc_name:
        if hasattr(app, 'registered_teams'):
            for team_name, team_object in app.registered_teams.items():
                if hasattr(team_object, 'npcs'):
                    team_npcs = team_object.npcs
                    if isinstance(team_npcs, dict):
                        if npc_name in team_npcs:
                            npc_object = team_npcs[npc_name]
                            team = team_name 
                            npc_object.team = team_object
                            print(f"Found NPC {npc_name} in registered team {team_name}")
                            break
                    elif isinstance(team_npcs, list):
                        for npc in team_npcs:
                            if hasattr(npc, 'name') and npc.name == npc_name:
                                npc_object = npc
                                team = team_name  
                                npc_object.team = team_object
                                print(f"Found NPC {npc_name} in registered team {team_name}")
                                break

                if not npc_object and hasattr(team_object, 'forenpc') and hasattr(team_object.forenpc, 'name'):
                    if team_object.forenpc.name == npc_name:
                        npc_object = team_object.forenpc
                        npc_object.team = team_object

                        team = team_name
                        print(f"Found NPC {npc_name} as forenpc in team {team_name}")
                        break
                

                if npc_object:
                    break
        

        if not npc_object and hasattr(app, 'registered_npcs') and npc_name in app.registered_npcs:
            npc_object = app.registered_npcs[npc_name]
            print(f"Found NPC {npc_name} in registered NPCs (no specific team)")
            team_object = Team(team_path=npc_object.npc_directory, db_conn=db_conn)
            npc_object.team = team_object
        if not npc_object:
            db_conn = get_db_connection()
            npc_object = load_npc_by_name_and_source(npc_name, 
                                                     npc_source, 
                                                     db_conn, 
                                                     current_path)
            if not npc_object and npc_source == 'project':
                print(f"NPC {npc_name} not found in project directory, trying global...")
                npc_object = load_npc_by_name_and_source(npc_name, 'global', db_conn)
            if npc_object and hasattr(npc_object, 'npc_directory') and npc_object.npc_directory:
                team_directory = npc_object.npc_directory
                
                if os.path.exists(team_directory):
                    team_object = Team(team_path=team_directory, db_conn=db_conn)
                    print('team', team_object)

                else:
                    team_object = Team(npcs=[npc_object], db_conn=db_conn)
                    team_object.name = os.path.basename(team_directory) if team_directory else f"{npc_name}_team"
                    npc_object.team = team_object
                    print('team', team_object)                    
                team_name = team_object.name
                
                if not hasattr(app, 'registered_teams'):
                    app.registered_teams = {}
                app.registered_teams[team_name] = team_object
                
                team = team_name
                
                print(f"Created and registered team '{team_name}' with NPC {npc_name}")
            
            if npc_object:
                npc_object.team = team_object

                print(f"Successfully loaded NPC {npc_name} from {npc_source} directory")
            else:
                print(f"Warning: Could not load NPC {npc_name}")
            if npc_object:
                print(f"Successfully loaded NPC {npc_name} from {npc_source} directory")
            else:
                print(f"Warning: Could not load NPC {npc_name}")




    attachments = data.get("attachments", [])
    command_history = CommandHistory(app.config.get('DB_PATH'))
    images = []     
    attachments_for_db = []
    attachment_paths_for_llm = []

    message_id = generate_message_id()
    if attachments:
        attachment_dir = os.path.expanduser(f"~/.npcsh/attachments/{conversation_id+message_id}/")
        os.makedirs(attachment_dir, exist_ok=True)

        for attachment in attachments:
            try:
                file_name = attachment["name"]
                
                extension = file_name.split(".")[-1].upper() if "." in file_name else ""
                extension_mapped = extension_map.get(extension, "others")
                
                save_path = os.path.join(attachment_dir, file_name)

                if "data" in attachment and attachment["data"]:
                    decoded_data = base64.b64decode(attachment["data"])
                    with open(save_path, "wb") as f:
                        f.write(decoded_data)
                
                elif "path" in attachment and attachment["path"]:
                    shutil.copy(attachment["path"], save_path)
                
                else:
                    continue

                attachment_paths_for_llm.append(save_path)

                if extension_mapped == "images":
                    images.append(save_path)

                with open(save_path, "rb") as f:
                    file_content_bytes = f.read()

                attachments_for_db.append({
                    "name": file_name,
                    "path": save_path,
                    "type": extension_mapped,
                    "data": file_content_bytes,
                    "size": os.path.getsize(save_path)
                })

            except Exception as e:
                print(f"Error processing attachment {attachment.get('name', 'N/A')}: {e}")
                traceback.print_exc()
    messages = fetch_messages_for_conversation(conversation_id)
    if len(messages) == 0 and npc_object is not None:
        messages = [{'role': 'system', 
                     'content': npc_object.get_system_prompt()}]
    elif len(messages) > 0 and messages[0]['role'] != 'system' and npc_object is not None:
        messages.insert(0, {'role': 'system', 
                            'content': npc_object.get_system_prompt()})
    elif len(messages) > 0 and npc_object is not None:
        messages[0]['content'] = npc_object.get_system_prompt()
    if npc_object is not None and messages and messages[0]['role'] == 'system':
        messages[0]['content'] = npc_object.get_system_prompt()
    tool_args = {}
    if npc_object is not None:
        if hasattr(npc_object, 'tools') and npc_object.tools:
            if isinstance(npc_object.tools, list) and callable(npc_object.tools[0]):
                tools_schema, tool_map = auto_tools(npc_object.tools)
                tool_args['tools'] = tools_schema
                tool_args['tool_map'] = tool_map
            else:
                tool_args['tools'] = npc_object.tools
                if hasattr(npc_object, 'tool_map') and npc_object.tool_map:
                    tool_args['tool_map'] = npc_object.tool_map
        elif hasattr(npc_object, 'tool_map') and npc_object.tool_map:
            tool_args['tool_map'] = npc_object.tool_map
        if 'tools' in tool_args and tool_args['tools']:
            tool_args['tool_choice'] = {"type": "auto"}
    
    # Default stream response so closures below always have a value
    stream_response = {"output": "", "messages": messages}

    exe_mode = data.get('executionMode','chat')

    # Initialize api_url with default before checking npc_object
    api_url = None
    if npc_object is not None:
        try:
            api_url = npc_object.api_url if npc_object.api_url else None
        except AttributeError:
            api_url = None

    if exe_mode == 'chat':
        stream_response = get_llm_response(
            commandstr, 
            messages=messages, 
            images=images, 
            model=model,
            provider=provider, 
            npc=npc_object, 
            api_url = api_url,
            team=team_object,
            stream=True, 
            attachments=attachment_paths_for_llm,
            auto_process_tool_calls=True,
            **tool_args
        )
        messages = stream_response.get('messages', messages)
    elif exe_mode == 'tool_agent':
        mcp_server_path_from_request = data.get("mcpServerPath")
        selected_mcp_tools_from_request = data.get("selectedMcpTools", [])

        # Resolve MCP server path (explicit -> team ctx -> default resolver)
        effective_mcp_server_path = mcp_server_path_from_request
        if not effective_mcp_server_path and team_object and hasattr(team_object, 'team_ctx') and team_object.team_ctx:
            mcp_servers_list = team_object.team_ctx.get('mcp_servers', [])
            if mcp_servers_list and isinstance(mcp_servers_list, list):
                first_server_obj = next((s for s in mcp_servers_list if isinstance(s, dict) and 'value' in s), None)
                if first_server_obj:
                    effective_mcp_server_path = first_server_obj['value']
            elif isinstance(team_object.team_ctx.get('mcp_server'), str):
                effective_mcp_server_path = team_object.team_ctx.get('mcp_server')

        effective_mcp_server_path = resolve_mcp_server_path(
            current_path=current_path,
            explicit_path=effective_mcp_server_path,
            force_global=False
        )
        print(f"[MCP] effective server path: {effective_mcp_server_path}")

        if not hasattr(app, 'mcp_clients'):
            app.mcp_clients = {}

        state_key = f"{conversation_id}_{npc_name or 'default'}"
        client_entry = app.mcp_clients.get(state_key)

        if not client_entry or not client_entry.get("client") or not client_entry["client"].session \
           or client_entry.get("server_path") != effective_mcp_server_path:
            mcp_client = MCPClientNPC()
            if effective_mcp_server_path and mcp_client.connect_sync(effective_mcp_server_path):
                print(f"[MCP] connected client for {state_key} to {effective_mcp_server_path}")
                app.mcp_clients[state_key] = {
                    "client": mcp_client,
                    "server_path": effective_mcp_server_path,
                    "messages": messages
                }
            else:
                print(f"[MCP] Failed to connect client for {state_key} to {effective_mcp_server_path}")
                app.mcp_clients[state_key] = {
                    "client": None,
                    "server_path": effective_mcp_server_path,
                    "messages": messages
                }

        mcp_client = app.mcp_clients[state_key]["client"]
        messages = app.mcp_clients[state_key].get("messages", messages)
        def stream_mcp_sse():
            nonlocal messages
            iteration = 0
            prompt = commandstr
            while iteration < 10:
                iteration += 1
                print(f"[MCP] iteration {iteration} prompt len={len(prompt)}")
                jinx_tool_catalog = {}
                if npc_object and hasattr(npc_object, "jinx_tool_catalog"):
                    jinx_tool_catalog = npc_object.jinx_tool_catalog or {}
                tools_for_llm = []
                if mcp_client:
                    tools_for_llm.extend(mcp_client.available_tools_llm)
                # append Jinx-derived tools
                tools_for_llm.extend(list(jinx_tool_catalog.values()))
                if selected_mcp_tools_from_request:
                    tools_for_llm = [t for t in tools_for_llm if t["function"]["name"] in selected_mcp_tools_from_request]
                print(f"[MCP] tools_for_llm: {[t['function']['name'] for t in tools_for_llm]}")

                llm_response = get_llm_response_with_handling(
                    prompt=prompt,
                    npc=npc_object,
                    model=model, 
                    provider=provider,
                    messages=messages,
                    tools=tools_for_llm,
                    stream=True,
                    team=team_object,
                    context=f' The users working directory is {current_path}'
                )
                print('RESPONSE', llm_response)

                stream = llm_response.get("response", [])
                messages = llm_response.get("messages", messages)
                collected_content = ""
                collected_tool_calls = []
                agent_tool_call_data = {"id": None, "function_name": None, "arguments": ""}

                for response_chunk in stream:
                    with cancellation_lock:
                        if cancellation_flags.get(stream_id, False):
                            yield {"type": "interrupt"}
                            return

                    if "hf.co" in model or provider == 'ollama' and 'gpt-oss' not in model:
                        # Ollama returns ChatResponse objects - support both attribute and dict access
                        msg = getattr(response_chunk, "message", None) or (response_chunk.get("message", {}) if hasattr(response_chunk, "get") else {})
                        chunk_content = getattr(msg, "content", None) or (msg.get("content") if hasattr(msg, "get") else "") or ""
                        # Extract Ollama thinking/reasoning tokens
                        reasoning_content = getattr(msg, "thinking", None) or (msg.get("thinking") if hasattr(msg, "get") else None)
                        # Handle tool calls with robust attribute/dict access
                        tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if hasattr(msg, "get") else None)
                        if tool_calls:
                            for tool_call in tool_calls:
                                tc_id = getattr(tool_call, "id", None) or (tool_call.get("id") if hasattr(tool_call, "get") else None)
                                tc_func = getattr(tool_call, "function", None) or (tool_call.get("function") if hasattr(tool_call, "get") else None)
                                if tc_func:
                                    tc_name = getattr(tc_func, "name", None) or (tc_func.get("name") if hasattr(tc_func, "get") else None)
                                    tc_args = getattr(tc_func, "arguments", None) or (tc_func.get("arguments") if hasattr(tc_func, "get") else None)
                                    if tc_name:
                                        arg_str = tc_args
                                        if isinstance(arg_str, dict):
                                            arg_str = json.dumps(arg_str)
                                        elif arg_str is None:
                                            arg_str = "{}"
                                        # Add to collected_tool_calls for Ollama
                                        collected_tool_calls.append({
                                            "id": tc_id or f"call_{len(collected_tool_calls)}",
                                            "type": "function",
                                            "function": {"name": tc_name, "arguments": arg_str}
                                        })
                        if chunk_content:
                            collected_content += chunk_content
                        # Extract other fields with robust access
                        created_at = getattr(response_chunk, "created_at", None) or (response_chunk.get("created_at") if hasattr(response_chunk, "get") else None)
                        model_name = getattr(response_chunk, "model", None) or (response_chunk.get("model") if hasattr(response_chunk, "get") else model)
                        msg_role = getattr(msg, "role", None) or (msg.get("role") if hasattr(msg, "get") else "assistant")
                        done_reason = getattr(response_chunk, "done_reason", None) or (response_chunk.get("done_reason") if hasattr(response_chunk, "get") else None)

                        # Build chunk_data with proper structure
                        chunk_data = {
                            "id": None,
                            "object": None,
                            "created": str(created_at) if created_at else datetime.datetime.now().isoformat(),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": chunk_content,
                                        "role": msg_role,
                                        "reasoning_content": reasoning_content
                                    },
                                    "finish_reason": done_reason
                                }
                            ]
                        }
                        yield chunk_data

                    elif hasattr(response_chunk, "choices") and response_chunk.choices:
                        delta = response_chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            collected_content += delta.content
                            chunk_data = {
                                "id": getattr(response_chunk, "id", None),
                                "object": getattr(response_chunk, "object", None),
                                "created": getattr(response_chunk, "created", datetime.datetime.now().strftime('YYYY-DD-MM-HHMMSS')),
                                "model": getattr(response_chunk, "model", model),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": delta.content,
                                            "role": "assistant"
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield chunk_data

                        if hasattr(delta, "tool_calls") and delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                idx = getattr(tool_call_delta, "index", 0)
                                while len(collected_tool_calls) <= idx:
                                    collected_tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                if getattr(tool_call_delta, "id", None):
                                    collected_tool_calls[idx]["id"] = tool_call_delta.id
                                if hasattr(tool_call_delta, "function"):
                                    fn = tool_call_delta.function
                                    if getattr(fn, "name", None):
                                        collected_tool_calls[idx]["function"]["name"] = fn.name
                                    if getattr(fn, "arguments", None):
                                        collected_tool_calls[idx]["function"]["arguments"] += fn.arguments

                if not collected_tool_calls:
                    print("[MCP] no tool calls, finishing streaming loop")
                    break

                print(f"[MCP] collected tool calls: {[tc['function']['name'] for tc in collected_tool_calls]}")
                yield {
                    "type": "tool_execution_start",
                    "tool_calls": [
                        {
                            "name": tc["function"]["name"],
                            "id": tc["id"],
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"].get("arguments", "")
                            }
                        } for tc in collected_tool_calls
                    ]
                }

                tool_results = []
                for tc in collected_tool_calls:
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"]["arguments"]
                    tool_id = tc["id"]

                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args) if tool_args.strip() else {}
                        except json.JSONDecodeError:
                            tool_args = {}

                    print(f"[MCP] tool_start {tool_name} args={tool_args}")
                    yield {"type": "tool_start", "name": tool_name, "id": tool_id, "args": tool_args}
                    try:
                        tool_content = ""
                        # First, try local Jinx execution
                        if npc_object and hasattr(npc_object, "jinxs_dict") and tool_name in npc_object.jinxs_dict:
                            jinx_obj = npc_object.jinxs_dict[tool_name]
                            try:
                                jinx_ctx = jinx_obj.execute(
                                    input_values=tool_args if isinstance(tool_args, dict) else {},
                                    npc=npc_object
                                )
                                tool_content = str(jinx_ctx)
                            except Exception as e:
                                tool_content = f"Jinx execution error: {str(e)}"
                        else:
                            # Execute via MCP client
                            if mcp_client and tool_name in mcp_client.tool_map:
                                try:
                                    tool_func = mcp_client.tool_map[tool_name]
                                    result = tool_func(**(tool_args if isinstance(tool_args, dict) else {}))
                                    # Handle MCP CallToolResult
                                    if hasattr(result, 'content'):
                                        tool_content = str(result.content[0].text) if result.content else str(result)
                                    else:
                                        tool_content = str(result)
                                except Exception as mcp_e:
                                    tool_content = f"MCP tool error: {str(mcp_e)}"
                            else:
                                tool_content = f"Tool '{tool_name}' not found in MCP server or Jinxs"
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": tool_content
                        })
                        
                        print(f"[MCP] tool_result {tool_name}: {tool_content}")
                        yield {"type": "tool_result", "name": tool_name, "id": tool_id, "result": tool_content}

                    except Exception as e:
                        error_msg = f"Tool execution error: {str(e)}"
                        print(f"[MCP] tool_error {tool_name}: {error_msg}")
                        yield {"type": "tool_error", "name": tool_name, "id": tool_id, "error": error_msg}

                serialized_tool_calls = []
                for tc in collected_tool_calls:
                    parsed_args = tc["function"]["arguments"]
                    # Gemini/LLM expects arguments as JSON string, not dict
                    if isinstance(parsed_args, dict):
                        args_for_message = json.dumps(parsed_args)
                    else:
                        args_for_message = str(parsed_args)
                    serialized_tool_calls.append({
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": args_for_message
                        }
                    })

                messages.append({
                    "role": "assistant",
                    "content": collected_content,
                    "tool_calls": serialized_tool_calls
                })
                tool_results_for_db = tool_results

                prompt = ""

            app.mcp_clients[state_key]["messages"] = messages
            return
        stream_response = stream_mcp_sse()

    else:
        stream_response = {"output": f"Unsupported execution mode: {exe_mode}", "messages": messages}

    user_message_filled = ''

    if isinstance(messages[-1].get('content'), list):
        for cont in messages[-1].get('content'):
            txt = cont.get('text')
            if txt is not None:
                user_message_filled += txt
    
    # Only save user message if it's NOT a resend
    if not is_resend:  # ADD THIS CONDITION
        save_conversation_message(
            command_history, 
            conversation_id, 
            "user", 
            user_message_filled if len(user_message_filled) > 0 else commandstr, 
            wd=current_path, 
            model=model, 
            provider=provider, 
            npc=npc_name,
            team=team, 
            attachments=attachments_for_db, 
            message_id=message_id,
        )




    message_id = generate_message_id()

    def event_stream(current_stream_id):
        complete_response = []
        complete_reasoning = []  # Accumulate reasoning content
        accumulated_tool_calls = []  # Accumulate all tool calls
        dot_count = 0
        interrupted = False
        tool_call_data = {"id": None, "function_name": None, "arguments": ""}

        try:
            # New: handle generators (tool_agent streaming)
            if hasattr(stream_response, "__iter__") and not isinstance(stream_response, (dict, str)):
                for chunk in stream_response:
                    with cancellation_lock:
                        if cancellation_flags.get(current_stream_id, False):
                            interrupted = True
                            break
                    if chunk is None:
                        continue
                    if isinstance(chunk, dict):
                        if chunk.get("type") == "interrupt":
                            interrupted = True
                            break
                        yield f"data: {json.dumps(chunk)}\n\n"
                        if chunk.get("choices"):
                            for choice in chunk["choices"]:
                                delta = choice.get("delta", {})
                                content_piece = delta.get("content")
                                if content_piece:
                                    complete_response.append(content_piece)
                                # Accumulate reasoning content from generator chunks
                                reasoning_piece = delta.get("reasoning_content")
                                if reasoning_piece:
                                    complete_reasoning.append(reasoning_piece)
                        # Accumulate tool calls from generator chunks
                        if chunk.get("type") == "tool_call":
                            tc = chunk.get("tool_call", {})
                            if tc.get("id") and tc.get("name"):
                                accumulated_tool_calls.append({
                                    "id": tc.get("id"),
                                    "function_name": tc.get("name"),
                                    "arguments": tc.get("arguments", "")
                                })
                        if chunk.get("type") == "tool_result":
                            tool_results_for_db.append({
                                "name": chunk.get("name"),
                                "tool_call_id": chunk.get("id"),
                                "content": chunk.get("result", "")
                            })
                        continue
                    yield f"data: {json.dumps({'choices':[{'delta':{'content': str(chunk), 'role': 'assistant'},'finish_reason':None}]})}\n\n"
                # Generator finished - skip the other stream handling paths

            elif isinstance(stream_response, str) :
                print('stream a str and not a gen')
                chunk_data = {
                        "id": None,
                        "object": None,
                        "created": datetime.datetime.now().strftime('YYYY-DD-MM-HHMMSS'),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta":
                                    {
                                        "content": stream_response,
                                        "role": "assistant"
                                  },
                                "finish_reason": 'done'
                            }
                        ]
                    }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            elif isinstance(stream_response, dict) and 'output' in stream_response and isinstance(stream_response.get('output'), str):
                print('stream a str and not a gen')
                chunk_data = {
                        "id": None,
                        "object": None,
                        "created": datetime.datetime.now().strftime('YYYY-DD-MM-HHMMSS'),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta":
                                    {
                                        "content": stream_response.get('output') ,
                                        "role": "assistant"
                                  },
                                "finish_reason": 'done'
                            }
                        ]
                    }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            elif isinstance(stream_response, dict):
                for response_chunk in stream_response.get('response', stream_response.get('output')):
                    with cancellation_lock:
                        if cancellation_flags.get(current_stream_id, False):
                            print(f"Cancellation flag triggered for {current_stream_id}. Breaking loop.")
                            interrupted = True
                            break

                    print('.', end="", flush=True)
                    dot_count += 1
                    if provider == 'llamacpp':
                        # llama-cpp-python returns OpenAI-format dicts
                        chunk_content = ""
                        reasoning_content = None
                        if isinstance(response_chunk, dict) and response_chunk.get("choices"):
                            delta = response_chunk["choices"][0].get("delta", {})
                            chunk_content = delta.get("content", "") or ""
                            reasoning_content = delta.get("reasoning_content")
                        if chunk_content:
                            complete_response.append(chunk_content)
                        if reasoning_content:
                            complete_reasoning.append(reasoning_content)
                        chunk_data = {
                            "id": response_chunk.get("id"),
                            "object": response_chunk.get("object"),
                            "created": response_chunk.get("created"),
                            "model": response_chunk.get("model", model),
                            "choices": [{"index": 0, "delta": {"content": chunk_content, "role": "assistant", "reasoning_content": reasoning_content}, "finish_reason": response_chunk.get("choices", [{}])[0].get("finish_reason")}]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    elif "hf.co" in model or provider == 'ollama' and 'gpt-oss' not in model:
                        # Ollama returns ChatResponse objects - support both attribute and dict access
                        msg = getattr(response_chunk, "message", None) or response_chunk.get("message", {}) if hasattr(response_chunk, "get") else {}
                        chunk_content = getattr(msg, "content", None) or (msg.get("content") if hasattr(msg, "get") else "") or ""
                        # Extract Ollama thinking/reasoning tokens
                        reasoning_content = getattr(msg, "thinking", None) or (msg.get("thinking") if hasattr(msg, "get") else None)
                        # Handle tool calls with robust attribute/dict access
                        tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if hasattr(msg, "get") else None)
                        if tool_calls:
                            for tool_call in tool_calls:
                                tc_id = getattr(tool_call, "id", None) or (tool_call.get("id") if hasattr(tool_call, "get") else None)
                                if tc_id:
                                    tool_call_data["id"] = tc_id
                                tc_func = getattr(tool_call, "function", None) or (tool_call.get("function") if hasattr(tool_call, "get") else None)
                                if tc_func:
                                    tc_name = getattr(tc_func, "name", None) or (tc_func.get("name") if hasattr(tc_func, "get") else None)
                                    if tc_name:
                                        tool_call_data["function_name"] = tc_name
                                    tc_args = getattr(tc_func, "arguments", None) or (tc_func.get("arguments") if hasattr(tc_func, "get") else None)
                                    if tc_args:
                                        arg_val = tc_args
                                        if isinstance(arg_val, dict):
                                            arg_val = json.dumps(arg_val)
                                        tool_call_data["arguments"] += arg_val
                                # Accumulate complete tool call info for DB storage (Ollama path)
                                if tc_id and tc_func and tc_name:
                                    accumulated_tool_calls.append({
                                        "id": tc_id,
                                        "function_name": tc_name,
                                        "arguments": arg_val if tc_args else ""
                                    })
                        # Accumulate reasoning content
                        if reasoning_content:
                            complete_reasoning.append(reasoning_content)
                        if chunk_content:
                            complete_response.append(chunk_content)
                        # Extract other fields with robust access
                        created_at = getattr(response_chunk, "created_at", None) or (response_chunk.get("created_at") if hasattr(response_chunk, "get") else None)
                        model_name = getattr(response_chunk, "model", None) or (response_chunk.get("model") if hasattr(response_chunk, "get") else model)
                        msg_role = getattr(msg, "role", None) or (msg.get("role") if hasattr(msg, "get") else "assistant")
                        done_reason = getattr(response_chunk, "done_reason", None) or (response_chunk.get("done_reason") if hasattr(response_chunk, "get") else None)
                        chunk_data = {
                            "id": None, "object": None,
                            "created": created_at or datetime.datetime.now(),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": chunk_content, "role": msg_role, "reasoning_content": reasoning_content}, "finish_reason": done_reason}]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    else:
                        chunk_content = ""
                        reasoning_content = ""
                        for choice in response_chunk.choices:
                            if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                                for tool_call in choice.delta.tool_calls:
                                    if tool_call.id:
                                        tool_call_data["id"] = tool_call.id
                                    if tool_call.function:
                                        if hasattr(tool_call.function, "name") and tool_call.function.name:
                                            tool_call_data["function_name"] = tool_call.function.name
                                        if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                                            tool_call_data["arguments"] += tool_call.function.arguments
                                    # Accumulate complete tool call info for DB storage
                                    if tool_call.id and tool_call.function and tool_call.function.name:
                                        accumulated_tool_calls.append({
                                            "id": tool_call.id,
                                            "function_name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments or ""
                                        })
                        for choice in response_chunk.choices:
                            if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                                reasoning_content += choice.delta.reasoning_content
                                complete_reasoning.append(choice.delta.reasoning_content)
                        chunk_content = "".join(choice.delta.content for choice in response_chunk.choices if choice.delta.content is not None)
                        if chunk_content:
                            complete_response.append(chunk_content)
                        chunk_data = {
                            "id": response_chunk.id, "object": response_chunk.object, "created": response_chunk.created, "model": response_chunk.model,
                            "choices": [{"index": choice.index, "delta": {"content": choice.delta.content, "role": choice.delta.role, "reasoning_content": reasoning_content if hasattr(choice.delta, "reasoning_content") else None}, "finish_reason": choice.finish_reason} for choice in response_chunk.choices]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"

        except Exception as e:
            print(f"\nAn exception occurred during streaming for {current_stream_id}: {e}")
            traceback.print_exc()
            interrupted = True

        finally:
            print(f"\nStream {current_stream_id} finished. Interrupted: {interrupted}")
            print('\r' + ' ' * dot_count*2 + '\r', end="", flush=True)

            final_response_text = ''.join(complete_response)

            # Yield message_stop immediately so the client's stream ends quickly
            yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"

            # Persist tool call metadata and results before final assistant content
            if tool_call_data.get("function_name") or tool_call_data.get("arguments"):
                save_conversation_message(
                    command_history,
                    conversation_id,
                    "assistant",
                    {"tool_call": tool_call_data},
                    wd=current_path,
                    model=model,
                    provider=provider,
                    npc=npc_name,
                    team=team,
                    message_id=generate_message_id(),
                )

            if tool_results_for_db:
                for tr in tool_results_for_db:
                    save_conversation_message(
                        command_history,
                        conversation_id,
                        "tool",
                        {"tool_name": tr.get("name"), "tool_call_id": tr.get("tool_call_id"), "content": tr.get("content")},
                        wd=current_path,
                        model=model,
                        provider=provider,
                        npc=npc_name,
                        team=team,
                        message_id=generate_message_id(),
                    )

            # Save assistant message to the database with reasoning content and tool calls
            npc_name_to_save = npc_object.name if npc_object else ''
            save_conversation_message(
                command_history,
                conversation_id,
                "assistant",
                final_response_text,
                wd=current_path,
                model=model,
                provider=provider,
                npc=npc_name_to_save,
                team=team,
                message_id=message_id,
                reasoning_content=''.join(complete_reasoning) if complete_reasoning else None,
                tool_calls=accumulated_tool_calls if accumulated_tool_calls else None,
                tool_results=tool_results_for_db if tool_results_for_db else None,
            )

            # Start background tasks for memory extraction and context compression
            # These will run without blocking the main response stream.
            conversation_turn_text = f"User: {commandstr}\nAssistant: {final_response_text}"
            background_thread = threading.Thread(
                target=_run_stream_post_processing,
                args=(
                    conversation_turn_text,
                    conversation_id,
                    command_history,
                    npc_name,
                    team, # Pass the team variable from the outer scope
                    current_path,
                    model,
                    provider,
                    npc_object,
                    messages # Pass messages for context compression
                )
            )
            background_thread.daemon = True # Allow the main program to exit even if this thread is still running
            background_thread.start()

            with cancellation_lock:
                if current_stream_id in cancellation_flags:
                    del cancellation_flags[current_stream_id]
                    print(f"Cleaned up cancellation flag for stream ID: {current_stream_id}")
    return Response(event_stream(stream_id), mimetype="text/event-stream")

@app.route('/api/delete_message', methods=['POST'])
def delete_message():
    data = request.json
    conversation_id = data.get('conversationId')
    message_id = data.get('messageId')
    
    if not conversation_id or not message_id:
        return jsonify({"error": "Missing conversationId or messageId"}), 400
    
    try:
        command_history = CommandHistory(app.config.get('DB_PATH'))
        
        # Delete the message from the database
        result = command_history.delete_message(conversation_id, message_id)
        
        print(f"[DELETE_MESSAGE] Deleted message {message_id} from conversation {conversation_id}. Rows affected: {result}")
        
        return jsonify({
            "success": True,
            "deletedMessageId": message_id,
            "rowsAffected": result
        }), 200
        
    except Exception as e:
        print(f"[DELETE_MESSAGE] Error: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/api/memory/approve", methods=["POST"])
def approve_memories():
    try:
        data = request.json
        approvals = data.get("approvals", [])
        
        command_history = CommandHistory(app.config.get('DB_PATH'))
        
        for approval in approvals:
            command_history.update_memory_status(
                approval['memory_id'],
                approval['decision'],
                approval.get('final_memory')
            )
        
        return jsonify({"success": True, "processed": len(approvals)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500





@app.route("/api/interrupt", methods=["POST"])
def interrupt_stream():
    data = request.json
    stream_id_to_cancel = data.get("streamId")

    if not stream_id_to_cancel:
        return jsonify({"error": "streamId is required"}), 400

    with cancellation_lock:
        print(f"Received interruption request for stream ID: {stream_id_to_cancel}")
        cancellation_flags[stream_id_to_cancel] = True

    return jsonify({"success": True, "message": f"Interruption for stream {stream_id_to_cancel} registered."})



@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    try:
        path = request.args.get("path")

        if not path:
            return jsonify({"error": "No path provided", "conversations": []}), 400

        engine = get_db_connection()
        try:
            with engine.connect() as conn:
                # Use REPLACE to normalize paths in the query for cross-platform compatibility
                # This handles both forward slashes and backslashes stored in the database
                query = text("""
                SELECT DISTINCT conversation_id,
                       MIN(timestamp) as start_time,
                       MAX(timestamp) as last_message_timestamp,
                       GROUP_CONCAT(content) as preview
                FROM conversation_history
                WHERE REPLACE(RTRIM(directory_path, '/\\'), '\\', '/') = :normalized_path
                GROUP BY conversation_id
                ORDER BY MAX(timestamp) DESC
                """)

                # Normalize the input path (convert backslashes to forward slashes, strip trailing slashes)
                normalized_path = normalize_path_for_db(path)

                result = conn.execute(query, {
                    "normalized_path": normalized_path
                })
                conversations = result.fetchall()

                return jsonify(
                    {
                        "conversations": [
                            {
                                "id": conv[0],  
                                "timestamp": conv[1],  
                                "last_message_timestamp": conv[2],  
                                "preview": (
                                    conv[3][:100] + "..."  
                                    if conv[3] and len(conv[3]) > 100
                                    else conv[3]
                                ),
                            }
                            for conv in conversations
                        ],
                        "error": None,
                    }
                )
        finally:
            engine.dispose()

    except Exception as e:
        print(f"Error getting conversations: {str(e)}")
        return jsonify({"error": str(e), "conversations": []}), 500



@app.route("/api/conversation/<conversation_id>/messages", methods=["GET"])
def get_conversation_messages(conversation_id):
    try:
        engine = get_db_connection()
        with engine.connect() as conn:

            query = text("""
                WITH ranked_messages AS (
                    SELECT
                        ch.id,
                        ch.message_id,
                        ch.timestamp,
                        ch.role,
                        ch.content,
                        ch.conversation_id,
                        ch.directory_path,
                        ch.model,
                        ch.provider,
                        ch.npc,
                        ch.team,
                        ch.reasoning_content,
                        ch.tool_calls,
                        ch.tool_results,
                        GROUP_CONCAT(ma.id) as attachment_ids,
                        ROW_NUMBER() OVER (
                            PARTITION BY ch.role, strftime('%s', ch.timestamp)
                            ORDER BY ch.id DESC
                        ) as rn
                    FROM conversation_history ch
                    LEFT JOIN message_attachments ma
                        ON ch.message_id = ma.message_id
                    WHERE ch.conversation_id = :conversation_id
                    GROUP BY ch.id, ch.timestamp
                )
                SELECT *
                FROM ranked_messages
                WHERE rn = 1
                ORDER BY timestamp ASC, id ASC
            """)

            result = conn.execute(query, {"conversation_id": conversation_id})
            messages = result.fetchall()

            def parse_json_field(value):
                """Parse a JSON string field, returning None if empty or invalid."""
                if not value:
                    return None
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return None

            return jsonify(
                {
                    "messages": [
                        {
                            "message_id": msg[1] if len(msg) > 1 else None,
                            "role": msg[3] if len(msg) > 3 else None,
                            "content": msg[4] if len(msg) > 4 else None,
                            "timestamp": msg[2] if len(msg) > 2 else None,
                            "model": msg[7] if len(msg) > 7 else None,
                            "provider": msg[8] if len(msg) > 8 else None,
                            "npc": msg[9] if len(msg) > 9 else None,
                            "reasoningContent": msg[11] if len(msg) > 11 else None,
                            "toolCalls": parse_json_field(msg[12]) if len(msg) > 12 else None,
                            "toolResults": parse_json_field(msg[13]) if len(msg) > 13 else None,
                            "attachments": (
                                get_message_attachments(msg[1])
                                if len(msg) > 1 and msg[14]  # attachment_ids is at index 14
                                else []
                            ),
                        }
                        for msg in messages
                    ],
                    "error": None,
                }
            )

    except Exception as e:
        print(f"Error getting conversation messages: {str(e)}")
        return jsonify({"error": str(e), "messages": []}), 500



@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response



@app.route('/api/ollama/status', methods=['GET'])
def ollama_status():
    try:
        
        
        ollama.list()
        return jsonify({"status": "running"})
    except ollama.RequestError as e:
        
        print(f"Ollama status check failed: {e}")
        return jsonify({"status": "not_found"})
    except Exception as e:
        print(f"An unexpected error occurred during Ollama status check: {e}")
        return jsonify({"status": "not_found"})


@app.route("/api/ollama/tool_models", methods=["GET"])
def get_ollama_tool_models():
    """
    Returns all Ollama models. Tool capability detection is unreliable,
    so we don't filter - let the user try and the backend will handle failures.
    """
    try:
        detected = []
        listing = ollama.list()
        for model in listing.get("models", []):
            name = getattr(model, "model", None) or model.get("name") if isinstance(model, dict) else None
            if name:
                detected.append(name)
        return jsonify({"models": detected, "error": None})
    except Exception as e:
        print(f"Error listing Ollama models: {e}")
        return jsonify({"models": [], "error": str(e)}), 500


@app.route('/api/ollama/models', methods=['GET'])
def get_ollama_models():
    response = ollama.list()
    models_list = []
    
    
    for model_obj in response['models']:
        models_list.append({
            "name": model_obj.model,
            "size": model_obj.details.parameter_size, 
            
        })
            
    return jsonify(models_list)



@app.route('/api/ollama/delete', methods=['POST'])
def delete_ollama_model():
    data = request.get_json()
    model_name = data.get('name')
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400
    try:
        ollama.delete(model_name)
        return jsonify({"success": True, "message": f"Model {model_name} deleted."})
    except ollama.ResponseError as e:
        
        return jsonify({"error": e.error}), e.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/ollama/pull', methods=['POST'])
def pull_ollama_model():
    data = request.get_json()
    model_name = data.get('name')
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400

    def generate_progress():
        try:
            stream = ollama.pull(model_name, stream=True)
            for progress_obj in stream:
                
                
                yield json.dumps({
                    'status': getattr(progress_obj, 'status', None),
                    'digest': getattr(progress_obj, 'digest', None),
                    'total': getattr(progress_obj, 'total', None),
                    'completed': getattr(progress_obj, 'completed', None)
                }) + '\n'
        except ollama.ResponseError as e:
            error_message = {"status": "Error", "details": e.error}
            yield json.dumps(error_message) + '\n'
        except Exception as e:
            error_message = {"status": "Error", "details": str(e)}
            yield json.dumps(error_message) + '\n'

    return Response(generate_progress(), content_type='application/x-ndjson')
@app.route('/api/ollama/install', methods=['POST'])
def install_ollama():
    try:
        install_command = "curl -fsSL https://ollama.com/install.sh | sh"
        result = subprocess.run(install_command, shell=True, check=True, capture_output=True, text=True)
        return jsonify({"success": True, "output": result.stdout})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

extension_map = {
    "PNG": "images",
    "JPG": "images",
    "JPEG": "images",
    "GIF": "images",
    "SVG": "images",
    "MP4": "videos",
    "AVI": "videos",
    "MOV": "videos",
    "WMV": "videos",
    "MPG": "videos",
    "MPEG": "videos",
    "DOC": "documents",
    "DOCX": "documents",
    "PDF": "documents",
    "PPT": "documents",
    "PPTX": "documents",
    "XLS": "documents",
    "XLSX": "documents",
    "TXT": "documents",
    "CSV": "documents",
    "ZIP": "archives",
    "RAR": "archives",
    "7Z": "archives",
    "TAR": "archives",
    "GZ": "archives",
    "BZ2": "archives",
    "ISO": "archives",
}


    


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "error": None})


# OpenAI-compatible completions API
@app.route("/v1/chat/completions", methods=["POST"])
def openai_chat_completions():
    """
    OpenAI-compatible chat completions endpoint.
    Allows using NPC team as a drop-in replacement for OpenAI API.

    Extra parameter:
      - agent: NPC name to use (optional, uses team's forenpc if not specified)
    """
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        model = data.get("model", "gpt-4o-mini")
        stream = data.get("stream", False)
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 4096)

        # Extra: agent/npc selection
        agent_name = data.get("agent") or data.get("npc")

        current_path = request.headers.get("X-Current-Path", os.getcwd())

        # Load team and NPC
        db_path = app.config.get('DB_PATH') or os.path.expanduser("~/.npcsh/npcsh_history.db")
        db_conn = create_engine(f'sqlite:///{db_path}')

        npc = None
        team = None

        # Try to load from project or global
        project_team_path = os.path.join(current_path, "npc_team")
        global_team_path = os.path.expanduser("~/.npcsh/npc_team")

        team_path = project_team_path if os.path.exists(project_team_path) else global_team_path

        if os.path.exists(team_path):
            try:
                team = Team(team_path, db_conn=db_conn)
                if agent_name and agent_name in team.npcs:
                    npc = team.npcs[agent_name]
                elif team.forenpc:
                    npc = team.forenpc
            except Exception as e:
                print(f"Error loading team: {e}")

        # Extract the prompt from messages
        prompt = ""
        conversation_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Handle multimodal content
                content = " ".join([c.get("text", "") for c in content if c.get("type") == "text"])
            conversation_messages.append({"role": role, "content": content})
            if role == "user":
                prompt = content

        # Determine provider from model name
        provider = data.get("provider")
        if not provider:
            if "gpt" in model or "o1" in model or model.startswith("o3"):
                provider = "openai"
            elif "claude" in model:
                provider = "anthropic"
            elif "gemini" in model:
                provider = "gemini"
            else:
                provider = "openai"  # default

        if stream:
            def generate_stream():
                request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())

                try:
                    response = get_llm_response(
                        prompt,
                        model=model,
                        provider=provider,
                        npc=npc,
                        team=team,
                        messages=conversation_messages[:-1],  # exclude last user message (it's the prompt)
                        stream=True,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                    for chunk in response:
                        if isinstance(chunk, str):
                            delta_content = chunk
                        elif hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            delta_content = getattr(delta, 'content', '') or ''
                        else:
                            delta_content = str(chunk)

                        if delta_content:
                            chunk_data = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": delta_content},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"

                    # Final chunk
                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                except Exception as e:
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "server_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return Response(
                generate_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            # Non-streaming response
            response = get_llm_response(
                prompt,
                model=model,
                provider=provider,
                npc=npc,
                team=team,
                messages=conversation_messages[:-1],
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = ""
            if isinstance(response, str):
                content = response
            elif hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content or ""
            elif isinstance(response, dict):
                content = response.get("response") or response.get("output") or str(response)
            else:
                content = str(response)

            return jsonify({
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1
                }
            })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }), 500


@app.route("/v1/models", methods=["GET"])
def openai_list_models():
    """OpenAI-compatible models listing - returns available NPCs as models."""
    current_path = request.headers.get("X-Current-Path", os.getcwd())

    models = []

    # Add NPCs as available "models"
    project_team_path = os.path.join(current_path, "npc_team")
    global_team_path = os.path.expanduser("~/.npcsh/npc_team")

    for team_path in [project_team_path, global_team_path]:
        if os.path.exists(team_path):
            for npc_file in Path(team_path).glob("*.npc"):
                models.append({
                    "id": npc_file.stem,
                    "object": "model",
                    "created": int(os.path.getmtime(npc_file)),
                    "owned_by": "npc-team"
                })

    return jsonify({
        "object": "list",
        "data": models
    })


# ============== GGUF/GGML Model Scanning ==============
@app.route('/api/models/gguf/scan', methods=['GET'])
def scan_gguf_models():
    """Scan for GGUF/GGML model files in specified or default directories."""
    directory = request.args.get('directory')

    # Default directories to scan
    default_dirs = [
        os.path.expanduser('~/.npcsh/models/gguf'),
        os.path.expanduser('~/.npcsh/models'),
        os.path.expanduser('~/models'),
        os.path.expanduser('~/.cache/huggingface/hub'),
    ]

    # Add env var directory if set
    env_dir = os.environ.get('NPCSH_GGUF_DIR')
    if env_dir:
        default_dirs.insert(0, os.path.expanduser(env_dir))

    dirs_to_scan = [os.path.expanduser(directory)] if directory else default_dirs

    models = []
    seen_paths = set()

    for scan_dir in dirs_to_scan:
        if not os.path.isdir(scan_dir):
            continue

        for root, dirs, files in os.walk(scan_dir):
            for f in files:
                if f.endswith(('.gguf', '.ggml', '.bin')) and not f.startswith('.'):
                    full_path = os.path.join(root, f)
                    if full_path not in seen_paths:
                        seen_paths.add(full_path)
                        try:
                            size = os.path.getsize(full_path)
                            models.append({
                                'name': f,
                                'path': full_path,
                                'size': size,
                                'size_gb': round(size / (1024**3), 2)
                            })
                        except OSError:
                            pass

    return jsonify({'models': models, 'error': None})


@app.route('/api/models/hf/download', methods=['POST'])
def download_hf_model():
    """Download a GGUF model from HuggingFace."""
    data = request.json
    url = data.get('url', '')
    target_dir = data.get('target_dir', '~/.npcsh/models/gguf')

    target_dir = os.path.expanduser(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    try:
        # Parse HuggingFace URL or model ID
        # Formats:
        # - TheBloke/Llama-2-7B-GGUF
        # - https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

        if url.startswith('http'):
            # Direct URL - download the file
            import requests
            filename = url.split('/')[-1].split('?')[0]
            target_path = os.path.join(target_dir, filename)

            print(f"Downloading {url} to {target_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return jsonify({'path': target_path, 'error': None})
        else:
            # Model ID - use huggingface_hub to download
            try:
                from huggingface_hub import hf_hub_download, list_repo_files

                # List files in repo to find GGUF files
                files = list_repo_files(url)
                gguf_files = [f for f in files if f.endswith('.gguf')]

                if not gguf_files:
                    return jsonify({'error': 'No GGUF files found in repository'}), 400

                # Download the first/smallest Q4 quantized version or first available
                q4_files = [f for f in gguf_files if 'Q4' in f or 'q4' in f]
                file_to_download = q4_files[0] if q4_files else gguf_files[0]

                print(f"Downloading {file_to_download} from {url}")
                path = hf_hub_download(
                    repo_id=url,
                    filename=file_to_download,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False
                )

                return jsonify({'path': path, 'error': None})
            except ImportError:
                return jsonify({'error': 'huggingface_hub not installed. Run: pip install huggingface_hub'}), 500

    except Exception as e:
        print(f"Error downloading HF model: {e}")
        return jsonify({'error': str(e)}), 500


# ============== Local Model Provider Status ==============
@app.route('/api/models/local/scan', methods=['GET'])
def scan_local_models():
    """Scan for models from local providers (LM Studio, llama.cpp)."""
    provider = request.args.get('provider', '')

    if provider == 'lmstudio':
        # LM Studio typically runs on port 1234
        try:
            import requests
            response = requests.get('http://127.0.0.1:1234/v1/models', timeout=2)
            if response.ok:
                data = response.json()
                models = [{'name': m.get('id', m.get('name', 'unknown'))} for m in data.get('data', [])]
                return jsonify({'models': models, 'error': None})
        except:
            pass
        return jsonify({'models': [], 'error': 'LM Studio not running or not accessible'})

    elif provider == 'llamacpp':
        # llama.cpp server typically runs on port 8080
        try:
            import requests
            response = requests.get('http://127.0.0.1:8080/v1/models', timeout=2)
            if response.ok:
                data = response.json()
                models = [{'name': m.get('id', m.get('name', 'unknown'))} for m in data.get('data', [])]
                return jsonify({'models': models, 'error': None})
        except:
            pass
        return jsonify({'models': [], 'error': 'llama.cpp server not running or not accessible'})

    return jsonify({'models': [], 'error': f'Unknown provider: {provider}'})


@app.route('/api/models/local/status', methods=['GET'])
def get_local_model_status():
    """Check if a local model provider is running."""
    provider = request.args.get('provider', '')

    if provider == 'lmstudio':
        try:
            import requests
            response = requests.get('http://127.0.0.1:1234/v1/models', timeout=2)
            if response.ok:
                return jsonify({'status': 'running', 'running': True})
        except:
            pass
        return jsonify({'status': 'not_running', 'running': False})

    elif provider == 'llamacpp':
        try:
            import requests
            response = requests.get('http://127.0.0.1:8080/v1/models', timeout=2)
            if response.ok:
                return jsonify({'status': 'running', 'running': True})
        except:
            pass
        return jsonify({'status': 'not_running', 'running': False})

    return jsonify({'status': 'unknown', 'running': False, 'error': f'Unknown provider: {provider}'})


# ============== Activity Tracking ==============
@app.route('/api/activity/track', methods=['POST'])
def track_activity():
    """Track user activity for predictive features."""
    try:
        data = request.json or {}
        # For now, just acknowledge the activity - can be expanded later
        # to store in database for RNN-based predictions
        activity_type = data.get('type', 'unknown')
        return jsonify({'success': True, 'tracked': activity_type})
    except Exception as e:
        print(f"Error tracking activity: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


def start_flask_server(
    port=5337,
    cors_origins=None,
    static_files=None, 
    debug=False,
    teams=None,
    npcs=None,
    db_path: str ='',
    user_npc_directory = None
):
    try:
        
        if teams:
            app.registered_teams = teams
            print(f"Registered {len(teams)} teams: {list(teams.keys())}")
        else:
            app.registered_teams = {}
            
        if npcs:
            app.registered_npcs = npcs
            print(f"Registered {len(npcs)} NPCs: {list(npcs.keys())}")
        else:
            app.registered_npcs = {}
        
        app.config['DB_PATH'] = db_path
        app.config['user_npc_directory'] = user_npc_directory

        command_history = CommandHistory(db_path)
        app.command_history = command_history

        
        if cors_origins:

            CORS(
                app,
                origins=cors_origins,
                allow_headers=["Content-Type", "Authorization"],
                methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                supports_credentials=True,
                
            )

        
        print(f"Starting Flask server on http://0.0.0.0:{port}")
        app.run(host="0.0.0.0", port=port, debug=debug,  threaded=True)
    except Exception as e:
        print(f"Error starting server: {str(e)}")


if __name__ == "__main__":

    SETTINGS_FILE = Path(os.path.expanduser("~/.npcshrc"))


    db_path = os.path.expanduser("~/npcsh_history.db")
    user_npc_directory = os.path.expanduser("~/.npcsh/npc_team")

    start_flask_server(db_path=db_path, user_npc_directory=user_npc_directory)
