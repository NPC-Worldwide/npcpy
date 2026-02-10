import os
import shutil
from pyexpat.errors import messages
import yaml
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 20,
    'axes.grid.axis': 'both',
    'axes.grid.which': 'major',
    'axes.prop_cycle': mpl.cycler('color', ['k', 'b', 'r', 'g', 'c', 'm', 'y', 'k']),
    'xtick.top': True,
    'xtick.direction': 'in',
    'xtick.major.size': 10,
    'xtick.minor.size': 5,
    'xtick.labelsize': 20,
    'xtick.minor.visible': True,
    'xtick.major.top': True,
    'xtick.major.bottom': True,
    'xtick.minor.top': True,
    'xtick.minor.bottom': True,
    'ytick.left': True,
    'ytick.right': True,
    'ytick.direction': 'in',
    'ytick.major.size': 10,
    'ytick.minor.size': 5,
    'ytick.labelsize': 20,
    'ytick.minor.visible': True,
    'ytick.major.left': True,
    'ytick.major.right': True,
    'ytick.minor.left': True,
    'ytick.minor.right': True,
    'legend.frameon': False,
    'legend.fontsize': 12,
    'image.cmap': 'plasma',
    'errorbar.capsize': 1,
})
import re
import random
from datetime import datetime
import hashlib
import pathlib
import sys 
import fnmatch
import subprocess
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from jinja2 import Environment, FileSystemLoader, Template, Undefined, DictLoader
from jinja2.sandbox import SandboxedEnvironment
from sqlalchemy import create_engine, text
import npcpy as npy 
from npcpy.tools import auto_tools
import math
import random
import base64
from npcpy.npc_sysenv import (
    ensure_dirs_exist, 
    init_db_tables,
    get_system_message, 

    )
from npcpy.memory.command_history import CommandHistory, generate_message_id

class SilentUndefined(Undefined):
    """Undefined that silently returns empty string instead of raising errors"""
    def _fail_with_undefined_error(self, *args, **kwargs):
        return ""

    def __str__(self):
        return ""

    def __repr__(self):
        return ""

    def __bool__(self):
        return False

    def __eq__(self, other):
        return other == "" or other is None or isinstance(other, Undefined)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

import math
from PIL import Image
from jinja2 import Environment, ChainableUndefined

class PreserveUndefined(ChainableUndefined):
    """Undefined that preserves the original {{ variable }} syntax"""
    def __str__(self):
        return f"{{{{ {self._undefined_name} }}}}"

def agent_pass_handler(command, extracted_data, **kwargs):
    """Handler for agent pass action"""
    npc = kwargs.get('npc')
    team = kwargs.get('team')    
    if not team and npc and hasattr(npc, '_current_team'):
        team = npc._current_team
    
    
    if not npc or not team:
        return {"messages": kwargs.get('messages', []), "output": f"Error: No NPC ({npc.name if npc else 'None'}) or team ({team.name if team else 'None'}) available for agent pass"}
    
    target_npc_name = extracted_data.get('target_npc')
    if not target_npc_name:
        return {"messages": kwargs.get('messages', []), "output": "Error: No target NPC specified"}
    
    messages = kwargs.get('messages', [])
    
    
    pass_count = 0
    recent_passes = []
    
    for msg in messages[-10:]:  
        if 'NOTE: THIS COMMAND HAS BEEN PASSED FROM' in msg.get('content', ''):
            pass_count += 1
            
            if 'PASSED FROM' in msg.get('content', ''):
                content = msg.get('content', '')
                if 'PASSED FROM' in content and 'TO YOU' in content:
                    parts = content.split('PASSED FROM')[1].split('TO YOU')[0].strip()
                    recent_passes.append(parts)
    

    
    target_npc = team.get_npc(target_npc_name)
    if not target_npc:
        available_npcs = list(team.npcs.keys()) if hasattr(team, 'npcs') else []
        return {"messages": kwargs.get('messages', []), 
                "output": f"Error: NPC '{target_npc_name}' not found in team. Available: {available_npcs}"}
    
    
    
    result = npc.handle_agent_pass(
        target_npc,
        command,
        messages=kwargs.get('messages'),
        context=kwargs.get('context'),
        shared_context=getattr(team, 'shared_context', None),
        stream=kwargs.get('stream', False),
        team=team
    )
    
    return result

def create_or_replace_table(db_path, table_name, data):
    """Creates or replaces a table in the SQLite database"""
    conn = sqlite3.connect(os.path.expanduser(db_path))
    try:
        data.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Table '{table_name}' created/replaced successfully.")
        return True
    except Exception as e:
        print(f"Error creating/replacing table '{table_name}': {e}")
        return False
    finally:
        conn.close()

def find_file_path(filename, search_dirs, suffix=None):
    """Find a file in multiple directories"""
    if suffix and not filename.endswith(suffix):
        filename += suffix
        
    for dir_path in search_dirs:
        file_path = os.path.join(os.path.expanduser(dir_path), filename)
        if os.path.exists(file_path):
            return file_path
            
    return None

def get_log_entries(entity_id, entry_type=None, limit=10, db_path="~/npcsh_history.db"):
    """Get log entries for an NPC or team"""
    db_path = os.path.expanduser(db_path)
    with sqlite3.connect(db_path) as conn:
        query = "SELECT entry_type, content, metadata, timestamp FROM npc_log WHERE entity_id = ?"
        params = [entity_id]
        
        if entry_type:
            query += " AND entry_type = ?"
            params.append(entry_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        results = conn.execute(query, params).fetchall()
        
        return [
            {
                "entry_type": r[0],
                "content": json.loads(r[1]),
                "metadata": json.loads(r[2]) if r[2] else None,
                "timestamp": r[3]
            }
            for r in results
        ]

def _json_dumps_with_undefined(obj, **kwargs):
    """Custom JSON dumps that handles SilentUndefined objects"""
    def default_handler(o):
        if isinstance(o, Undefined):
            return ""
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    return json.dumps(obj, default=default_handler, **kwargs)

def load_yaml_file(file_path):
    """Load a YAML file with error handling, rendering Jinja2 first"""
    try:
        with open(os.path.expanduser(file_path), 'r') as f:
            content = f.read()

        if '{%' not in content:
            return yaml.safe_load(content)

        jinja_env = SandboxedEnvironment(undefined=SilentUndefined)
        jinja_env.policies['json.dumps_function'] = _json_dumps_with_undefined
        template = jinja_env.from_string(content)
        rendered_content = template.render({})

        return yaml.safe_load(rendered_content)
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return None

def log_entry(entity_id, entry_type, content, metadata=None, db_path="~/npcsh_history.db"):
    """Log an entry for an NPC or team"""
    db_path = os.path.expanduser(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO npc_log (entity_id, entry_type, content, metadata) VALUES (?, ?, ?, ?)",
            (entity_id, entry_type, json.dumps(content), json.dumps(metadata) if metadata else None)
        )
        conn.commit()

def initialize_npc_project(
    directory=None,
    templates=None,
    context=None,
    model=None,
    provider=None,
) -> str:
    """Initialize an NPC project"""
    if directory is None:
        directory = os.getcwd()
    directory = os.path.expanduser(os.fspath(directory))

    for subdir in ["images", "models", "attachments", "mcp_servers"]:
        os.makedirs(os.path.join(directory, subdir), exist_ok=True)

    npc_team_dir = os.path.join(directory, "npc_team")
    os.makedirs(npc_team_dir, exist_ok=True)

    for subdir in ["jinxs",
                   "jinxs/skills",
                   "assembly_lines",
                   "sql_models",
                   "jobs",
                   "triggers",
                   "tools"]:
        os.makedirs(os.path.join(npc_team_dir, subdir), exist_ok=True)
    
    forenpc_path = os.path.join(npc_team_dir, "forenpc.npc")
    

    
    if not os.path.exists(forenpc_path):
        
        default_npc = {
            "name": "forenpc",
            "primary_directive": "You are the forenpc of an NPC team", 
        }
        with open(forenpc_path, "w") as f:
            yaml.dump(default_npc, f)
    parsed_templates: List[str] = []
    if templates:
        if isinstance(templates, str):
            parsed_templates = [
                t.strip() for t in re.split(r"[,\s]+", templates) if t.strip()
            ]
        elif isinstance(templates, (list, tuple, set)):
            parsed_templates = [str(t).strip() for t in templates if str(t).strip()]
        else:
            parsed_templates = [str(templates).strip()]

    ctx_destination: Optional[str] = None
    preexisting_ctx = [
        os.path.join(npc_team_dir, f)
        for f in os.listdir(npc_team_dir)
        if f.endswith(".ctx")
    ]
    if preexisting_ctx:
        ctx_destination = preexisting_ctx[0]
        if len(preexisting_ctx) > 1:
            print(
                "Warning: Multiple .ctx files already present; using first and ignoring the rest."
            )
    
    def _resolve_template_path(template_name: str) -> Optional[str]:
        expanded = os.path.expanduser(template_name)
        if os.path.exists(expanded):
            return expanded

        embedded_templates = {
            "slean": """name: slean
primary_directive: You are slean, the marketing innovator AI. Your responsibility is to create marketing campaigns and manage them effectively, while also thinking creatively to solve marketing challenges. Guide the strategy that drives customer engagement and brand awareness.
""",
            "turnic": """name: turnic
primary_directive: Assist with sales challenges and questions. Opt for straightforward solutions that help sales professionals achieve quick results.
""",
            "budgeto": """name: budgeto
primary_directive: You manage marketing budgets, ensuring resources are allocated efficiently and spend is optimized.
""",
            "relatio": """name: relatio
primary_directive: You manage customer relationships and ensure satisfaction throughout the sales process. Focus on nurturing clients and maintaining long-term connections.
""",
            "funnel": """name: funnel
primary_directive: You oversee the sales pipeline, track progress, and optimize conversion rates to move leads efficiently.
""",
        }

        base_dirs = [
            os.path.expanduser("~/.npcsh/npc_team/templates"),
            os.path.expanduser("~/.npcpy/npc_team/templates"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "template_tests", "npc_team")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples", "npc_team")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "example_npc_project", "npc_team")),
        ]
        base_dirs = [d for d in base_dirs if os.path.isdir(d)]

        for base in base_dirs:
            direct = os.path.join(base, template_name)
            if os.path.exists(direct):
                return direct
            if not direct.endswith(".npc") and os.path.exists(direct + ".npc"):
                return direct + ".npc"
            for root, _, files in os.walk(base):
                for fname in files:
                    stem, ext = os.path.splitext(fname)
                    if ext == ".npc" and stem == template_name:
                        return os.path.join(root, fname)

        if template_name in embedded_templates:
            embedded_dir = os.path.join(npc_team_dir, "_embedded_templates", template_name)
            os.makedirs(embedded_dir, exist_ok=True)
            npc_file = os.path.join(embedded_dir, f"{template_name}.npc")
            if not os.path.exists(npc_file):
                with open(npc_file, "w") as f:
                    f.write(embedded_templates[template_name])
            return embedded_dir
        return None

    def _copy_template(src_path: str) -> List[str]:
        nonlocal ctx_destination
        copied: List[str] = []
        src_path = os.path.expanduser(src_path)

        allowed_exts = {".npc", ".tool", ".pipe", ".sql", ".job", ".ctx", ".yaml", ".yml"}

        if os.path.isfile(src_path):
            if os.path.splitext(src_path)[1] in allowed_exts:
                if os.path.splitext(src_path)[1] == ".ctx":
                    if ctx_destination:
                        print(
                            f"Warning: Skipping extra context file '{src_path}' because one already exists."
                        )
                        return copied
                    dest_path = os.path.join(npc_team_dir, os.path.basename(src_path))
                    ctx_destination = dest_path
                else:
                    dest_path = os.path.join(npc_team_dir, os.path.basename(src_path))
                if not os.path.exists(dest_path):
                    shutil.copy2(src_path, dest_path)
                copied.append(dest_path)
            return copied

        for root, _, files in os.walk(src_path):
            rel_dir = os.path.relpath(root, src_path)
            dest_dir = npc_team_dir if rel_dir == "." else os.path.join(npc_team_dir, rel_dir)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in files:
                if os.path.splitext(fname)[1] not in allowed_exts:
                    continue
                if os.path.splitext(fname)[1] == ".ctx":
                    if ctx_destination:
                        print(
                            f"Warning: Skipping extra context file '{os.path.join(root, fname)}' because one already exists."
                        )
                        continue
                    dest_path = os.path.join(npc_team_dir, fname)
                    ctx_destination = dest_path
                else:
                    dest_path = os.path.join(dest_dir, fname)
                if not os.path.exists(dest_path):
                    shutil.copy2(os.path.join(root, fname), dest_path)
                copied.append(dest_path)
        return copied

    applied_templates: List[str] = []
    if parsed_templates:
        for template_name in parsed_templates:
            template_path = _resolve_template_path(template_name)
            if not template_path:
                print(f"Warning: Template '{template_name}' not found in known template directories.")
                continue
            copied = _copy_template(template_path)
            if copied:
                applied_templates.append(template_name)
    
    if applied_templates:
        applied_templates = sorted(set(applied_templates))
    if not ctx_destination:
        default_ctx_path = os.path.join(npc_team_dir, "team.ctx")
        default_ctx = {
            'name': '',
            'context' : context or '', 
            'preferences': '', 
            'mcp_servers': '', 
            'databases':'', 
            'use_global_jinxs': True,
            'forenpc': 'forenpc'
        }
        if parsed_templates:
            default_ctx['templates'] = parsed_templates
        with open(default_ctx_path, "w") as f:
            yaml.dump(default_ctx, f)
        ctx_destination = default_ctx_path

    if applied_templates:
        return (
            f"NPC project initialized in {npc_team_dir} "
            f"using templates: {', '.join(applied_templates)}"
        )
    return f"NPC project initialized in {npc_team_dir}"

def write_yaml_file(file_path, data):
    """Write data to a YAML file"""
    try:
        with open(os.path.expanduser(file_path), 'w') as f:
            yaml.dump(data, f)
        return True
    except Exception as e:
        print(f"Error writing YAML file {file_path}: {e}")
        return False

class Jinx:
    ''' 
    Jinx represents a workflow template with Jinja-rendered steps.
    
    Loads YAML definition containing:
    - jinx_name: identifier
    - inputs: list of input parameters
    - description: what the jinx does
    - npc: optional NPC to execute with
    - steps: list of step definitions with code. This section can now be a Jinja template itself.
    - file_context: optional list of file patterns to include as context
    
    Execution:
    - Renders Jinja templates in step code with input values
    - Executes resulting Python code
    - Returns context with outputs
    '''
    def __init__(self, jinx_data=None, jinx_path=None):
        if jinx_path:
            self._load_from_file(jinx_path)
        elif jinx_data:
            self._load_from_data(jinx_data)
        else:
            raise ValueError("Either jinx_data or jinx_path must be provided")
        
        self._raw_steps = list(self.steps)
        if self.steps and all(isinstance(s, dict) for s in self.steps):
            pass
        else:
            self.steps = []
        self.parsed_files = {}
        if self.file_context:
            self.parsed_files = self._parse_file_patterns(self.file_context)

    def _load_from_file(self, path):
        jinx_data = load_yaml_file(path)
        if not jinx_data:
            raise ValueError(f"Failed to load jinx from {path}")
        jinx_data['_source_path'] = path
        self._load_from_data(jinx_data)
            

    def _load_from_data(self, jinx_data):
        if not jinx_data or not isinstance(jinx_data, dict):
            raise ValueError("Invalid jinx data provided")
            
        if "jinx_name" not in jinx_data:
            raise KeyError("Missing 'jinx_name' in jinx definition")
            
        self.jinx_name = jinx_data.get("jinx_name")
        self.inputs = jinx_data.get("inputs", [])
        self.description = jinx_data.get("description", "")
        self.npc = jinx_data.get("npc")
        self.steps = jinx_data.get("steps", [])
        self.file_context = jinx_data.get("file_context", [])
        self._source_path = jinx_data.get("_source_path", None)

    def to_tool_def(self) -> Dict[str, Any]:
        """Convert this Jinx to an OpenAI-style tool definition."""
        properties = {}
        required = []
        for inp in self.inputs:
            if isinstance(inp, str):
                properties[inp] = {"type": "string", "description": f"Parameter: {inp}"}
                required.append(inp)
            elif isinstance(inp, dict):
                name = list(inp.keys())[0]
                default_val = inp.get(name, "")
                desc = f"Parameter: {name}"
                if default_val != "":
                    desc += f" (default: {default_val})"
                properties[name] = {"type": "string", "description": desc}
        return {
            "type": "function",
            "function": {
                "name": self.jinx_name,
                "description": self.description or f"Jinx: {self.jinx_name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def render_first_pass(
        self,
        jinja_env_for_macros: Environment,
        all_jinx_callables: Dict[str, Callable]
    ):
        """
        Performs the first-pass Jinja rendering on the Jinx's raw steps.
        This expands Jinja control flow (for, if) to generate step structures,
        then expands nested Jinx calls (e.g., {{ sh(...) }} or engine: jinx_name)
        and inline macros.
        """
        if self._raw_steps and isinstance(self._raw_steps[0], dict):
            structurally_expanded_steps = list(self._raw_steps)
        else:
            raw_steps_template_string = "\n".join(self._raw_steps)

            try:
                steps_template = jinja_env_for_macros.from_string(raw_steps_template_string)
                rendered_steps_yaml_string = steps_template.render(**jinja_env_for_macros.globals)
            except Exception as e:
                self.steps = list(self._raw_steps)
                return

            try:
                structurally_expanded_steps = yaml.safe_load(rendered_steps_yaml_string)
                if not isinstance(structurally_expanded_steps, list):
                    if structurally_expanded_steps is None:
                        structurally_expanded_steps = []
                    else:
                        raise ValueError(f"Rendered steps YAML did not result in a list: {type(structurally_expanded_steps)}")
            except Exception as e:
                self.steps = list(self._raw_steps)
                return

        final_rendered_steps = []
        for raw_step in structurally_expanded_steps:
            if not isinstance(raw_step, dict):
                final_rendered_steps.append(raw_step)
                continue

            engine_name = raw_step.get('engine')
            
            if engine_name and engine_name in all_jinx_callables:
                step_name = raw_step.get('name', f'call_{engine_name}')
                jinx_args = {
                    k: v for k, v in raw_step.items() 
                    if k not in ['engine', 'name']
                }
                
                jinx_callable = all_jinx_callables[engine_name]
                try:
                    expanded_yaml_string = jinx_callable(**jinx_args)
                    expanded_steps = yaml.safe_load(expanded_yaml_string)
                    
                    if isinstance(expanded_steps, list):
                        final_rendered_steps.extend(expanded_steps)
                    elif expanded_steps is not None:
                        final_rendered_steps.append(expanded_steps)
                except Exception as e:
                    final_rendered_steps.append(raw_step)
            elif raw_step.get('engine') in ['python', 'bash']:
                processed_step = {}
                for key, value in raw_step.items():
                    if isinstance(value, str):
                        has_template_var = '{{' in value and '}}' in value
                        if has_template_var:
                            processed_step[key] = value
                        else:
                            try:
                                template = jinja_env_for_macros.from_string(value)
                                rendered_value = template.render({})
                                try:
                                    loaded_value = yaml.safe_load(rendered_value)
                                    processed_step[key] = loaded_value
                                except yaml.YAMLError:
                                    processed_step[key] = rendered_value
                            except Exception as e:
                                processed_step[key] = value
                    else:
                        processed_step[key] = value
                final_rendered_steps.append(processed_step)
            else:
                processed_step = {}
                for key, value in raw_step.items():
                    if isinstance(value, str):
                        try:
                            template = jinja_env_for_macros.from_string(value)
                            rendered_value = template.render({})
                            try:
                                loaded_value = yaml.safe_load(rendered_value)
                                processed_step[key] = loaded_value
                            except yaml.YAMLError:
                                processed_step[key] = rendered_value
                        except Exception as e:
                            processed_step[key] = value
                    else:
                        processed_step[key] = value
                final_rendered_steps.append(processed_step)
        
        self.steps = final_rendered_steps

    def execute(self,
                input_values: Dict[str, Any],
                npc: Optional[Any] = None,
                messages: Optional[List[Dict[str, str]]] = None,
                extra_globals: Optional[Dict[str, Any]] = None,
                jinja_env: Optional[Environment] = None):
        
        if jinja_env is None:
            jinja_env = SandboxedEnvironment(
                loader=DictLoader({}),
                undefined=SilentUndefined,
            )
        
        active_npc = self.npc if self.npc else npc
        
        context = (
            active_npc.shared_context.copy() 
            if active_npc and hasattr(active_npc, 'shared_context') 
            else {}
        )
        context.update(input_values)
        context.update({
            "llm_response": None,
            "output": None,
            "messages": messages,
            "npc": active_npc
        })
        
        if self.parsed_files:
            context['file_context'] = self._format_parsed_files_context(self.parsed_files)
            context['files'] = self.parsed_files

        for i, step in enumerate(self.steps):
            context = self._execute_step(
                step,
                context,
                jinja_env,
                npc=active_npc,
                messages=messages,
                extra_globals=extra_globals
            )
            output_str = str(context.get("output", ""))
            if "error" in output_str.lower():
                self._log_debug(f"DEBUG: Jinx '{self.jinx_name}' execution stopped due to error in step '{step.get('name', 'unnamed_step')}': {context['output']}")
                break

        return context

    def _log_debug(self, msg: str):
        """Helper for logging debug messages to a file."""
        log_file_path = os.path.expanduser("~/jinx_debug_log.txt")
        with open(log_file_path, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {msg}\n")

    def _execute_step(self,
                  step: Dict[str, Any],
                  context: Dict[str, Any],
                  jinja_env: Environment,
                  npc: Optional[Any] = None,
                  messages: Optional[List[Dict[str, str]]] = None,
                  extra_globals: Optional[Dict[str, Any]] = None):
        
        code_content = step.get("code") or ""
        step_name = step.get("name", "unnamed_step")
        step_npc = step.get("npc")
        
        active_npc = step_npc if step_npc else npc
        
        try:
            template = jinja_env.from_string(code_content)
            rendered_code = template.render(**context)
        except Exception as e:
            error_msg = (
                f"Error rendering template for step '{step_name}' "
                f"(second pass): {type(e).__name__}: {e}"
            )
            context['output'] = error_msg
            self._log_debug(error_msg)
            return context
        
        self._log_debug(f"DEBUG: Executing step '{step_name}' with rendered code: {rendered_code}")

        from npcpy.npc_array import NPCArray, infer_matrix, ensemble_vote

        exec_globals = {
            "__builtins__": __builtins__,
            "npc": active_npc,
            "context": context,
            "math": math,
            "random": random,
            "datetime": datetime,
            "Image": Image,
            "pd": pd,
            "plt": plt,
            "sys": sys,
            "subprocess": subprocess,
            "np": np,
            "os": os,
            're': re,
            "json": json,
            "Path": pathlib.Path,
            "fnmatch": fnmatch,
            "pathlib": pathlib,
            "subprocess": subprocess,
            "get_llm_response": npy.llm_funcs.get_llm_response,
            "CommandHistory": CommandHistory,
            "NPCArray": NPCArray,
            "infer_matrix": infer_matrix,
            "ensemble_vote": ensemble_vote,
        }
        
        if extra_globals:
            exec_globals.update(extra_globals)

        exec_globals.update(context)

        exec_locals = exec_globals

        try:
            exec(rendered_code, exec_globals, exec_locals)
        except Exception as e:
            error_msg = (
                f"Error executing step '{step_name}' in jinx '{self.jinx_name}': "
                f"{type(e).__name__}: {e}"
            )
            print(f"[JINX-ERROR] {error_msg}")
            context['output'] = error_msg
            self._log_debug(error_msg)
            return context

        context_output = context.get("output")
        context.update(exec_locals)

        if context_output is not None:
            context["output"] = context_output
            context[step_name] = context_output

        if "output" in exec_locals and exec_locals["output"] is not None:
            outp = exec_locals["output"]
            context["output"] = outp
            context[step_name] = outp

        final_output = context.get("output")
        if final_output is not None and messages is not None:
            messages.append({
                'role':'assistant',
                'content': (
                    f'Jinx {self.jinx_name} step {step_name} '
                    f'executed: {final_output}'
                )
            })
            context['messages'] = messages
        
        return context
        
    def _parse_file_patterns(self, patterns_config):
        """Parse file patterns configuration and load matching files into KV cache"""
        if not patterns_config:
            return {}
        
        file_cache = {}
        
        for pattern_entry in patterns_config:
            if isinstance(pattern_entry, str):
                pattern_entry = {"pattern": pattern_entry}
            
            pattern = pattern_entry.get("pattern", "")
            recursive = pattern_entry.get("recursive", False)
            base_path = pattern_entry.get("base_path", ".")
            
            if not pattern:
                continue
                
            if self._source_path:
                base_path = os.path.join(os.path.dirname(self._source_path), base_path)
            base_path = os.path.expanduser(base_path)
            
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.getcwd(), base_path)
            
            matching_files = self._find_matching_files(pattern, base_path, recursive)
            
            for file_path in matching_files:
                file_content = self._load_file_content(file_path)
                if file_content:
                    relative_path = os.path.relpath(file_path, base_path)
                    file_cache[relative_path] = file_content
        
        return file_cache

    def _find_matching_files(self, pattern, base_path, recursive=False):
        """Find files matching the given pattern"""
        matching_files = []
        
        if not os.path.exists(base_path):
            return matching_files
        
        if recursive:
            for root, dirs, files in os.walk(base_path):
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        matching_files.append(os.path.join(root, filename))
        else:
            try:
                for item in os.listdir(base_path):
                    item_path = os.path.join(base_path, item)
                    if os.path.isfile(item_path) and fnmatch.fnmatch(item, pattern):
                        matching_files.append(item_path)
            except PermissionError:
                print(f"Permission denied accessing {base_path}")
        
        return matching_files

    def _load_file_content(self, file_path):
        """Load content from a file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def _format_parsed_files_context(self, parsed_files):
        """Format parsed files into context string"""
        if not parsed_files:
            return ""
        
        context_parts = ["Additional context from files:"]
        
        for file_path, content in parsed_files.items():
            context_parts.append(f"\n--- {file_path} ---")
            context_parts.append(content)
            context_parts.append("")
        
        return "\n".join(context_parts)

    def to_dict(self):
        result = {
            "jinx_name": self.jinx_name,
            "description": self.description,
            "inputs": self.inputs,
            "steps": self._raw_steps,
            "file_context": self.file_context
        }
        
        if self.npc:
            result["npc"] = self.npc
            
        return result

    def save(self, directory):
        jinx_path = os.path.join(directory, f"{self.jinx_name}.jinx")
        ensure_dirs_exist(os.path.dirname(jinx_path))
        return write_yaml_file(jinx_path, self.to_dict())
        
    @classmethod
    def from_mcp(cls, mcp_tool):
        try:
            import inspect

            doc = mcp_tool.__doc__ or ""
            name = mcp_tool.__name__
            signature = inspect.signature(mcp_tool)
            
            inputs = []
            for param_name, param in signature.parameters.items():
                if param_name != 'self':
                    param_type = (
                        param.annotation 
                        if param.annotation != inspect.Parameter.empty 
                        else None
                    )
                    param_default = (
                        None 
                        if param.default == inspect.Parameter.empty 
                        else param.default
                    )
                    
                    inputs.append({
                        "name": param_name,
                        "type": str(param_type),
                        "default": param_default
                    })
            
            jinx_data = {
                "jinx_name": name,
                "description": doc.strip(),
                "inputs": inputs,
                "file_context": [],
                "steps": [
                    {
                        "name": "mcp_function_call",
                        "code": f"""
import {mcp_tool.__module__}
output = {mcp_tool.__module__}.{name}(
    {', '.join([
        f'{inp["name"]}=context.get("{inp["name"]}")' 
        for inp in inputs
    ])}
)
"""
                    }
                ]
            }
            
            return cls(jinx_data=jinx_data)
            
        except: 
            pass

def _parse_skill_md(path):
    """Parse a skill markdown file with YAML frontmatter and ## sections.

    Expected format:
        ---
        name: skill-name
        description: What it does. Use when ...
        ---
        # Skill Title

        ## section-one
        Content for section one...

        ## section-two
        Content for section two...

    Returns dict with name, description, sections, frontmatter or None on failure.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.startswith('---'):
        return None

    parts = content.split('---', 2)
    if len(parts) < 3:
        return None

    frontmatter = yaml.safe_load(parts[1])
    if not frontmatter or not isinstance(frontmatter, dict):
        return None

    body = parts[2].strip()

    sections = {}
    current_section = None
    current_content = []

    for line in body.split('\n'):
        if line.startswith('## '):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line[3:].strip()
            current_content = []
        elif current_section is not None:
            current_content.append(line)

    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()

    basename = os.path.splitext(os.path.basename(path))[0]
    if basename.upper() == 'SKILL':
        default_name = os.path.basename(os.path.dirname(path))
    else:
        default_name = basename

    return {
        'name': frontmatter.get('name', default_name),
        'description': frontmatter.get('description', ''),
        'sections': sections,
        'frontmatter': frontmatter
    }

def _compile_skill_to_jinx(skill_data, source_path=None):
    """Compile skill data into a Jinx whose step is ``engine: skill``.

    Everything about the skill — name, description, sections, scripts,
    references, assets — is passed as structured data into ``skill.jinx``.
    The sub-jinx owns the full representation; the compiler just parses the
    source format and hands it off.

    Sections content is base64-encoded to survive the two-pass Jinja pipeline
    without mangling.  Metadata lists (scripts, references, assets) are passed
    as JSON strings.
    """
    name = skill_data.get('jinx_name', skill_data.get('name', ''))
    description = skill_data.get('description', '')
    sections = skill_data.get('sections', {})

    sections_json = json.dumps(sections, ensure_ascii=False)
    content_b64 = base64.b64encode(sections_json.encode('utf-8')).decode('ascii')

    section_names = list(sections.keys())
    desc_suffix = f" [Sections: {', '.join(section_names)}]" if section_names else ""

    scripts_list = []
    references_list = []
    assets_list = []

    file_context = list(skill_data.get('file_context', []))
    for subdir, collector in (('scripts', scripts_list),
                               ('references', references_list),
                               ('assets', assets_list)):
        entries = skill_data.get(subdir)
        if isinstance(entries, list):
            collector.extend(entries)
            for entry in entries:
                if isinstance(entry, str):
                    file_context.append({
                        'pattern': entry,
                        'base_path': os.path.join('.', subdir) if source_path else '.',
                    })
                elif isinstance(entry, dict):
                    file_context.append(entry)
        elif entries is None and source_path:
            skill_dir = os.path.dirname(source_path)
            subdir_path = os.path.join(skill_dir, subdir)
            if os.path.isdir(subdir_path):
                for r, _d, fnames in os.walk(subdir_path):
                    for fn in fnames:
                        rel = os.path.relpath(os.path.join(r, fn), skill_dir)
                        collector.append(rel)
                file_context.append({
                    'pattern': '*',
                    'base_path': subdir,
                    'recursive': True,
                })

    jinx_data = {
        'jinx_name': name,
        'description': (description + desc_suffix) if description else f"Skill: {name}{desc_suffix}",
        'inputs': [{'section': 'all'}],
        'steps': [{
            'engine': 'skill',
            'skill_name': name,
            'skill_description': description,
            'sections': content_b64,
            'scripts_json': json.dumps(scripts_list),
            'references_json': json.dumps(references_list),
            'assets_json': json.dumps(assets_list),
            'section': '{{section}}'
        }],
        'file_context': file_context,
        '_source_path': source_path
    }

    return Jinx(jinx_data=jinx_data)

def _load_skill_from_md(path):
    """Load a skill from a SKILL.md file with YAML frontmatter and ## sections.

    The skill name defaults to the parent folder name (matching the Anthropic
    skill-folder convention), falling back to the frontmatter ``name`` field.

    Sibling directories (``scripts/``, ``references/``, ``assets/``) are
    auto-discovered and attached as ``file_context``.
    """
    parsed = _parse_skill_md(path)
    if not parsed or not parsed.get('sections'):
        return None

    parent_dir = os.path.basename(os.path.dirname(path))
    name = parsed['name'] or parent_dir

    skill_data = {
        'jinx_name': name,
        'description': parsed['description'],
        'sections': parsed['sections'],
    }

    fm = parsed.get('frontmatter', {})
    for key in ('scripts', 'references', 'assets', 'file_context'):
        if key in fm:
            skill_data[key] = fm[key]

    return _compile_skill_to_jinx(skill_data, source_path=path)

def load_jinxs_from_directory(directory):
    """Load all jinxs from a directory recursively.

    Handles two file types:

    1. **.jinx** — regular jinxs loaded as-is.  Skill-type jinxs are just
       regular ``.jinx`` files whose steps use ``engine: skill`` with the
       full structured args (sections, scripts, references, assets).
    2. **SKILL.md inside a skill folder** — Anthropic-style skill folders::

           jinxs/skills/code-review/SKILL.md
           jinxs/skills/code-review/scripts/
           jinxs/skills/code-review/references/

       The folder name becomes the skill name.  The SKILL.md must have YAML
       frontmatter (``---`` delimiters) and ``##`` section headers.
       These are compiled into jinxs with ``engine: skill`` steps.
    """
    jinxs = []
    directory = os.path.expanduser(directory)

    if not os.path.exists(directory):
        return jinxs

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".jinx"):
                try:
                    jinx_path = os.path.join(root, filename)
                    jinx = Jinx(jinx_path=jinx_path)
                    jinxs.append(jinx)
                except Exception as e:
                    print(f"Error loading jinx {filename}: {e}")
            elif filename == "SKILL.md":
                try:
                    md_path = os.path.join(root, filename)
                    jinx = _load_skill_from_md(md_path)
                    if jinx:
                        jinxs.append(jinx)
                except Exception as e:
                    skill_folder = os.path.basename(root)
                    print(f"Error loading skill from {skill_folder}/SKILL.md: {e}")

    return jinxs

def jinx_to_tool_def(jinx_obj: 'Jinx') -> Dict[str, Any]:
    """Convert a Jinx instance into an MCP/LLM-compatible tool schema definition."""
    return jinx_obj.to_tool_def()

def build_jinx_tool_catalog(jinxs: Dict[str, 'Jinx']) -> Dict[str, Dict[str, Any]]:
    """Helper to build a name->tool_def catalog from a dict of Jinx objects."""
    return {name: jinx_to_tool_def(jinx_obj) for name, jinx_obj in jinxs.items()}

def match_jinx_spec_to_names(jinx_spec: str, team_jinxs_dict: Dict[str, 'Jinx'], jinxs_base_dir: str) -> List[str]:
    """
    Match a jinx spec pattern to actual jinx names from the team's jinxs_dict.

    Args:
        jinx_spec: A spec like 'lib/core/python', 'lib/computer_use/*', or just 'python'
        team_jinxs_dict: Dict mapping jinx_name -> Jinx object
        jinxs_base_dir: Base directory where team jinxs are stored (e.g., '/path/to/npc_team/jinxs')

    Returns:
        List of jinx names that match the spec
    """
    matched_names = []

    if jinx_spec in team_jinxs_dict:
        return [jinx_spec]

    spec_pattern = jinx_spec
    if not spec_pattern.endswith('.jinx') and not spec_pattern.endswith('*'):
        spec_pattern += '.jinx'

    for jinx_name, jinx_obj in team_jinxs_dict.items():
        source_path = getattr(jinx_obj, '_source_path', None)
        if not source_path:
            continue

        try:
            rel_path = os.path.relpath(source_path, jinxs_base_dir)
        except ValueError:
            continue

        if fnmatch.fnmatch(rel_path, spec_pattern):
            matched_names.append(jinx_name)
        elif fnmatch.fnmatch(rel_path, spec_pattern.replace('.jinx', '') + '.jinx'):
            matched_names.append(jinx_name)

    return matched_names

def extract_jinx_inputs(args: List[str], jinx: Jinx) -> Dict[str, Any]:
    print(f"DEBUG extract_jinx_inputs called with args: {args}")
    print(f"DEBUG jinx.inputs: {jinx.inputs}")
    
    inputs = {}

    flag_mapping = {}
    for input_ in jinx.inputs:
        if isinstance(input_, str):
            flag_mapping[f"-{input_[0]}"] = input_
            flag_mapping[f"--{input_}"] = input_
        elif isinstance(input_, dict):
            key = list(input_.keys())[0]
            flag_mapping[f"-{key[0]}"] = key
            flag_mapping[f"--{key}"] = key

    if len(jinx.inputs) > 1:
        used_args = set()
        for i, arg in enumerate(args):
            if '=' in arg and arg != '=' and not arg.startswith('-'):
                key, value = arg.split('=', 1)
                key = key.strip().strip("'\"")
                value = value.strip().strip("'\"")
                inputs[key] = value
                used_args.add(i)
    else:
        used_args = set()

    for i, arg in enumerate(args):
        if i in used_args:
            continue
            
        if arg in flag_mapping:
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                input_name = flag_mapping[arg]
                inputs[input_name] = args[i + 1]
                used_args.add(i)
                used_args.add(i + 1)
            else:
                input_name = flag_mapping[arg]
                inputs[input_name] = True
                used_args.add(i)

    unused_args = [arg for i, arg in enumerate(args) if i not in used_args]
    
    print(f"DEBUG unused_args: {unused_args}")
    
    first_required = None
    for input_ in jinx.inputs:
        if isinstance(input_, str):
            first_required = input_
            break
    
    print(f"DEBUG first_required: {first_required}")
    
    if first_required and unused_args:
        inputs[first_required] = ' '.join(unused_args).strip()
        print(f"DEBUG assigned to first_required: {inputs[first_required]}")
    else:
        jinx_input_names = []
        for input_ in jinx.inputs:
            if isinstance(input_, str):
                jinx_input_names.append(input_)
            elif isinstance(input_, dict):
                jinx_input_names.append(list(input_.keys())[0])
        
        if len(jinx_input_names) == 1:
            inputs[jinx_input_names[0]] = ' '.join(unused_args).strip()
        else:
            for i, arg in enumerate(unused_args):
                if i < len(jinx_input_names):
                    input_name = jinx_input_names[i]
                    if input_name not in inputs: 
                        inputs[input_name] = arg

    for input_ in jinx.inputs:
        if isinstance(input_, str):
            if input_ not in inputs:
                raise ValueError(f"Missing required input: {input_}")
        elif isinstance(input_, dict):
            key = list(input_.keys())[0]
            default_value = input_[key]
            if key not in inputs:
                inputs[key] = default_value

    print(f"DEBUG final inputs: {inputs}")
    return inputs
from npcpy.memory.command_history import load_kg_from_db, save_kg_to_db
from npcpy.memory.knowledge_graph import kg_initial, kg_evolve_incremental, kg_sleep_process, kg_dream_process
from npcpy.llm_funcs import get_llm_response, breathe
import os
from datetime import datetime
import json

class NPC:
    def __init__(
        self,
        file: str = None,
        name: str = None,
        primary_directive: str = None,
        plain_system_message: bool = False,
        team = None,
        jinxs: list = None,
        tools: list = None,
        model: str = None,
        provider: str = None,
        api_url: str = None,
        api_key: str = None,
        db_conn=None,
        use_global_jinxs=False,
        memory = False,
        **kwargs
    ):
        """
        Initialize an NPC from a file path or with explicit parameters
        
        Args:
            file: Path to .npc file or name for the NPC
            primary_directive: System prompt/directive for the NPC
            jinxs: List of jinxs available to the NPC or "*" to load all jinxs
            model: LLM model to use
            provider: LLM provider to use
            api_url: API URL for LLM
            api_key: API key for LLM
            db_conn: Database connection
        """
        if not file and not name and not primary_directive:
            raise ValueError("Either 'file' or 'name' and 'primary_directive' must be provided")

        self.team = team

        if file:
            if file.endswith(".npc"):
                self._load_from_file(file)
            file_parent = os.path.dirname(file)
            self.jinxs_directory = os.path.join(file_parent, "jinxs")
            self.npc_directory = file_parent
        else:
            self.name = name            
            self.primary_directive = primary_directive
            self.model = model 
            self.provider = provider 
            self.api_url = api_url 
            self.api_key = api_key
            
            if use_global_jinxs:
                self.jinxs_directory = os.path.expanduser('~/.npcsh/npc_team/jinxs/')
            else: 
                self.jinxs_directory = None
            self.npc_directory = None

        if not hasattr(self, 'jinxs_spec') or jinxs is not None:
            self.jinxs_spec = jinxs or "*"

        if tools is not None:
            tools_schema, tool_map = auto_tools(tools)
            self.tools = tools_schema  
            self.tool_map = tool_map   
            self.tools_schema = tools_schema  
        else:
            self.tools = []
            self.tool_map = {}
            self.tools_schema = []
        self.plain_system_message = plain_system_message
        self.use_global_jinxs = use_global_jinxs
        self.jinx_tool_catalog: Dict[str, Dict[str, Any]] = {}
        self.mcp_servers = []
        
        self.memory_length = 20
        self.memory_strategy = 'recent'
        dirs = []
        if self.npc_directory:
            dirs.append(self.npc_directory)
        if self.jinxs_directory:
            dirs.append(self.jinxs_directory)
            
        self.jinja_env = SandboxedEnvironment(
            loader=FileSystemLoader([
                os.path.expanduser(d) for d in dirs
            ]),
            undefined=SilentUndefined,
        )
        
        self.db_conn = db_conn

        self.command_history = None
        self.kg_data = None
        self.tables = None
        self.memory = None

        if self.db_conn:
            self._setup_db()
            self.command_history = CommandHistory(db=self.db_conn)
            if memory:
                self.kg_data = self._load_npc_kg()  
                self.memory = self.get_memory_context()

        self.jinxs_dict = {}
        if jinxs and jinxs != "*": 
            for jinx_item in jinxs:
                if isinstance(jinx_item, Jinx):
                    self.jinxs_dict[jinx_item.jinx_name] = jinx_item
                elif isinstance(jinx_item, dict):
                    jinx_obj = Jinx(jinx_data=jinx_item)
                    self.jinxs_dict[jinx_obj.jinx_name] = jinx_obj
                elif isinstance(jinx_item, str):
                    jinx_path = find_file_path(jinx_item, [self.npc_jinxs_directory], suffix=".jinx")
                    if jinx_path:
                        jinx_obj = Jinx(jinx_path=jinx_path)
                        self.jinxs_dict[jinx_obj.jinx_name] = jinx_obj
                    else:
                        print(f"Warning: Jinx '{jinx_item}' not found for NPC '{self.name}' during initial load.")
        
        self.shared_context = {
            "dataframes": {},
            "current_data": None,
            "computation_results": [],
            "locals": {},

            "memories": {},

            "mcp_client": None,
            "mcp_tools": [],
            "mcp_tool_map": {},

            "session_input_tokens": 0,
            "session_output_tokens": 0,
            "session_cost_usd": 0.0,
            "turn_count": 0,

            "current_mode": "agent",
            "attachments": [],
        }
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        if db_conn is not None:
            init_db_tables()

    def initialize_jinxs(self, team_raw_jinxs: Optional[List['Jinx']] = None):
        """
        Loads and performs first-pass Jinja rendering for NPC-specific jinxs,
        now that the NPC's team context is fully established.
        """
        npc_jinxs_raw_list = []
        
        if self.jinxs_spec == "*":
            if self.team and hasattr(self.team, 'jinxs_dict') and self.team.jinxs_dict:
                self.jinxs_dict.update(self.team.jinxs_dict)
        else:
            if self.team and hasattr(self.team, 'jinxs_dict') and self.team.jinxs_dict:
                jinxs_base_dir = None
                if hasattr(self.team, 'team_path') and self.team.team_path:
                    jinxs_base_dir = os.path.join(self.team.team_path, 'jinxs')

                for jinx_spec in self.jinxs_spec:
                    if jinxs_base_dir:
                        matched_names = match_jinx_spec_to_names(jinx_spec, self.team.jinxs_dict, jinxs_base_dir)
                    else:
                        matched_names = [jinx_spec] if jinx_spec in self.team.jinxs_dict else []

                    if not matched_names:
                        raise FileNotFoundError(
                            f"NPC '{self.name}' references jinx '{jinx_spec}' but no matching jinx was found. "
                            f"Available jinxs: {list(self.team.jinxs_dict.keys())[:20]}..."
                        )

                    for jinx_name in matched_names:
                        if jinx_name in self.team.jinxs_dict:
                            self.jinxs_dict[jinx_name] = self.team.jinxs_dict[jinx_name]

        should_load_from_directory = False
        if hasattr(self, 'npc_jinxs_directory') and self.npc_jinxs_directory and os.path.exists(self.npc_jinxs_directory):
            if not self.team:
                should_load_from_directory = True
            elif hasattr(self.team, 'team_path') and self.team.team_path:
                team_jinxs_dir = os.path.join(self.team.team_path, 'jinxs')
                if os.path.normpath(self.npc_jinxs_directory) != os.path.normpath(team_jinxs_dir):
                    should_load_from_directory = True

        if should_load_from_directory:
            for jinx_obj in load_jinxs_from_directory(self.npc_jinxs_directory):
                if jinx_obj.jinx_name not in self.jinxs_dict:
                    npc_jinxs_raw_list.append(jinx_obj)
        
        if npc_jinxs_raw_list or team_raw_jinxs:
            all_available_raw_jinxs = list(team_raw_jinxs or [])
            all_available_raw_jinxs.extend(npc_jinxs_raw_list)

            combined_raw_jinxs_dict = {j.jinx_name: j for j in all_available_raw_jinxs}

            npc_first_pass_jinja_env = SandboxedEnvironment(undefined=SilentUndefined)

            jinx_macro_globals = {}
            for raw_jinx in combined_raw_jinxs_dict.values():
                def create_jinx_callable(jinx_obj_in_closure):
                    def callable_jinx(**kwargs):
                        temp_jinja_env = SandboxedEnvironment(undefined=SilentUndefined)
                        rendered_target_steps = []
                        for target_step in jinx_obj_in_closure._raw_steps:
                            temp_rendered_step = {}
                            for k, v in target_step.items():
                                if isinstance(v, str):
                                    try:
                                        temp_rendered_step[k] = temp_jinja_env.from_string(v).render(**kwargs)
                                    except Exception as e:
                                        print(f"Warning: Error in Jinx macro '{jinx_obj_in_closure.jinx_name}' rendering step field '{k}' (NPC first pass): {e}")
                                        temp_rendered_step[k] = v
                                else:
                                    temp_rendered_step[k] = v
                            rendered_target_steps.append(temp_rendered_step)
                        return yaml.dump(rendered_target_steps, default_flow_style=False)
                    return callable_jinx
                
                jinx_macro_globals[raw_jinx.jinx_name] = create_jinx_callable(raw_jinx)
            
            npc_first_pass_jinja_env.globals.update(jinx_macro_globals)

            for raw_npc_jinx in npc_jinxs_raw_list:
                try:
                    raw_npc_jinx.render_first_pass(npc_first_pass_jinja_env, jinx_macro_globals)
                    self.jinxs_dict[raw_npc_jinx.jinx_name] = raw_npc_jinx
                except Exception as e:
                    print(f"Error performing first-pass rendering for NPC Jinx '{raw_npc_jinx.jinx_name}': {e}")
        
        self.jinx_tool_catalog = build_jinx_tool_catalog(self.jinxs_dict)
        print(f"NPC {self.name} loaded {len(self.jinxs_dict)} jinxs and built catalog with {len(self.jinx_tool_catalog)} tools.")

    def _load_npc_kg(self):
        """Load knowledge graph data for this NPC from database"""
        if not self.command_history:
            return None
            
        directory_path = os.getcwd()
        team_name = getattr(self.team, 'name', 'default_team') if self.team else 'default_team'
        
        kg_data = load_kg_from_db(
            engine=self.command_history.engine,
            team_name=team_name,
            npc_name=self.name,
            directory_path=directory_path
        )
        print('# of facts: ', len(kg_data['facts']))
        print('# of facts: ', len(kg_data['concepts']))

        if not kg_data.get('facts') and not kg_data.get('concepts'):
            return self._initialize_kg_from_history()
        
        return kg_data

    def _initialize_kg_from_history(self):
        """Initialize KG from conversation history if no KG exists"""
        if not self.command_history:
            return None
            
        recent_messages = self.command_history.get_messages_by_npc(
            self.name, 
            n_last=50
        )
        print(f'Recent messages from NPC: {recent_messages[0:10]}')

        if not recent_messages:
            return {
                "generation": 0, 
                "facts": [], 
                "concepts": [], 
                "concept_links": [], 
                "fact_to_concept_links": {}, 
                "fact_to_fact_links": []
            }
        
        content_text = "\n".join([
            msg['content'] for msg in recent_messages 
            if msg['role'] == 'user' and isinstance(msg['content'], str)
        ])
        
        if not content_text.strip():
            return {
                "generation": 0, 
                "facts": [], 
                "concepts": [], 
                "concept_links": [], 
                "fact_to_concept_links": {}, 
                "fact_to_fact_links": []
            }
        
        kg_data = kg_initial(
            content_text,
            model=self.model,
            provider=self.provider,
            npc=self,
            context=getattr(self, 'shared_context', {})
        )
        self.kg_data = kg_data
        self._save_kg()
        return kg_data

    def _save_kg(self):
        """Save current KG data to database"""
        if not self.kg_data or not self.command_history:
            return False
            
        directory_path = os.getcwd()
        team_name = getattr(self.team, 'name', 'default_team') if self.team else 'default_team'
        save_kg_to_db(
            engine=self.command_history.engine,
            kg_data=self.kg_data,
            team_name=team_name,
            npc_name=self.name,
            directory_path=directory_path
        )
        return True

    def get_memory_context(self):
        """Get formatted memory context for system prompt"""
        if not self.kg_data:
            return ""
            
        context_parts = []
        
        recent_facts = self.kg_data.get('facts', [])[-10:]
        if recent_facts:
            context_parts.append("Recent memories:")
            for fact in recent_facts:
                context_parts.append(f"- {fact['statement']}")
        
        concepts = self.kg_data.get('concepts', [])
        if concepts:
            concept_names = [c['name'] for c in concepts[:5]]
            context_parts.append(f"Key concepts: {', '.join(concept_names)}")
        
        return "\n".join(context_parts)

    def update_memory(
        self, 
        user_input: str, 
        assistant_response: str
    ):
        """Update NPC memory from conversation turn using KG evolution"""
        conversation_turn = f"User: {user_input}\nAssistant: {assistant_response}"
        
        if not self.kg_data:
            self.kg_data = kg_initial(
                content_text=conversation_turn,
                model=self.model,
                provider=self.provider,
                npc=self
            )
        else:
            self.kg_data, _ = kg_evolve_incremental(
                existing_kg=self.kg_data,
                new_content_text=conversation_turn,
                model=self.model,
                provider=self.provider,
                npc=self,
                get_concepts=True,
                link_concepts_facts=False,
                link_concepts_concepts=False,
                link_facts_facts=False
            )
        
        self._save_kg()

    def enter_tool_use_loop(
        self, 
        prompt: str, 
        tools: list = None, 
        tool_map: dict = None, 
        max_iterations: int = 5,
        stream: bool = False
    ):
        """Enter interactive tool use loop for complex tasks"""
        if not tools:
            tools = self.tools
        if not tool_map:
            tool_map = self.tool_map
            
        messages = self.memory.copy() if self.memory else []
        messages.append({"role": "user", "content": prompt})
        
        for iteration in range(max_iterations):
            response = get_llm_response(
                prompt="",
                model=self.model,
                provider=self.provider,
                npc=self,
                messages=messages,
                tools=tools,
                tool_map=tool_map,
                auto_process_tool_calls=True,
                stream=stream
            )
            
            messages = response.get('messages', messages)
            
            if not response.get('tool_calls'):
                return {
                    "final_response": response.get('response'),
                    "messages": messages,
                    "iterations": iteration + 1
                }
                
        return {
            "final_response": "Max iterations reached",
            "messages": messages,
            "iterations": max_iterations
        }

    def get_code_response(
        self, 
        prompt: str, 
        language: str = "python", 
        execute: bool = False, 
        locals_dict: dict = None
    ):
        """Generate and optionally execute code responses"""
        code_prompt = f"""Generate {language} code for: {prompt}
        
        Provide ONLY executable {language} code without explanations.
        Do not include markdown formatting or code blocks.
        Begin directly with the code."""
        
        response = get_llm_response(
            prompt=code_prompt,
            model=self.model,
            provider=self.provider,
            npc=self,
            stream=False
        )
        
        generated_code = response.get('response', '')
        
        result = {
            "code": generated_code,
            "executed": False,
            "output": None,
            "error": None
        }
        
        if execute and language == "python":
            if locals_dict is None:
                locals_dict = {}
                
            exec_globals = {"__builtins__": __builtins__}
            exec_globals.update(locals_dict)
            
            exec_locals = {}
            exec(generated_code, exec_globals, exec_locals)
            
            locals_dict.update(exec_locals)
            result["executed"] = True
            result["output"] = exec_locals.get("output", "Code executed successfully")
        
        return result

    def _load_npc_memory(self):
        """Enhanced memory loading that includes KG context"""
        memory = self.command_history.get_messages_by_npc(self.name, n_last=self.memory_length)
        memory = [{'role':mem['role'], 'content':mem['content']} for mem in memory]
        return memory 

    def _load_from_file(self, file):
        """Load NPC configuration from file"""
        if "~" in file:
            file = os.path.expanduser(file)
        if not os.path.isabs(file):
            file = os.path.abspath(file)
            
        npc_data = load_yaml_file(file)
        if not npc_data:
            raise ValueError(f"Failed to load NPC from {file}")
            
        self.name = npc_data.get("name")
        if not self.name:
            self.name = os.path.splitext(os.path.basename(file))[0]
            
        self.primary_directive = npc_data.get("primary_directive")
        
        jinxs_spec = npc_data.get("jinxs", "*")
        
        if jinxs_spec == "*":
            self.jinxs_spec = "*" 
        else:
            self.jinxs_spec = jinxs_spec

        self.model = npc_data.get("model")
        self.provider = npc_data.get("provider")
        self.api_url = npc_data.get("api_url")
        self.api_key = npc_data.get("api_key")

        if self.team:
            if not self.model and hasattr(self.team, 'model'):
                self.model = self.team.model
            if not self.provider and hasattr(self.team, 'provider'):
                self.provider = self.team.provider
            if not self.api_url and hasattr(self.team, 'api_url'):
                self.api_url = self.team.api_url
            if not self.api_key and hasattr(self.team, 'api_key'):
                self.api_key = self.team.api_key

        self.name = npc_data.get("name", self.name)

        self.npc_path = file
        self.npc_jinxs_directory = os.path.join(os.path.dirname(file), "jinxs")

    def get_system_prompt(self, simple=False):
        """Get system prompt for the NPC"""
        if simple or self.plain_system_message:
            return self.primary_directive
        else:
            return get_system_message(self, team=self.team)

    def _setup_db(self):
        """Set up database tables and determine type"""
        dialect = self.db_conn.dialect.name

        with self.db_conn.connect() as conn:
            if dialect == "postgresql":
                result = conn.execute(text("""
                    SELECT table_name, obj_description((quote_ident(table_name))::regclass, 'pg_class')
                    FROM information_schema.tables
                    WHERE table_schema='public';
                """))
                self.tables = result.fetchall()
                self.db_type = "postgres"

            elif dialect == "sqlite":
                result = conn.execute(text(
                    "SELECT name, sql FROM sqlite_master WHERE type='table';"
                ))
                self.tables = result.fetchall()
                self.db_type = "sqlite"

            else:
                print(f"Unsupported DB dialect: {dialect}")
                self.tables = None
                self.db_type = None

    def get_llm_response(self, 
                        request,
                        jinxs=None,
                        tools: Optional[list] = None,
                        tool_map: Optional[dict] = None,
                        tool_choice=None, 
                        messages=None,
                        auto_process_tool_calls=True,
                        use_core_tools: bool = False,
                        **kwargs):
        all_candidate_functions = []

        if tools is not None and tool_map is not None:
            all_candidate_functions.extend([func for func in tool_map.values() if callable(func)])
        elif hasattr(self, 'tool_map') and self.tool_map:
            all_candidate_functions.extend([func for func in self.tool_map.values() if callable(func)])

        if use_core_tools:
            dynamic_core_tools_list = [
                self.think_step_by_step,
                self.write_code,
            ]

            if self.command_history:
                dynamic_core_tools_list.extend([
                    self.search_my_conversations,
                    self.search_my_memories,
                    self.create_memory,
                    self.read_memory, 
                    self.update_memory,
                    self.delete_memory,
                    self.search_memories,
                    self.get_all_memories,
                    self.archive_old_memories,
                    self.get_memory_stats
                ])

            if self.db_conn:
                dynamic_core_tools_list.append(self.query_database)

            all_candidate_functions.extend(dynamic_core_tools_list)

        unique_functions = []
        seen_names = set()
        for func in all_candidate_functions:
            if func.__name__ not in seen_names:
                unique_functions.append(func)
                seen_names.add(func.__name__)

        final_tools_schema = None
        final_tool_map_dict = None

        if unique_functions:
            final_tools_schema, final_tool_map_dict = auto_tools(unique_functions)

        if tool_choice is None:
            if final_tools_schema:
                tool_choice = "auto"
            else:
                tool_choice = "none"

        response = npy.llm_funcs.get_llm_response(
            request, 
            npc=self, 
            jinxs=jinxs,
            tools=final_tools_schema,
            tool_map=final_tool_map_dict,
            tool_choice=tool_choice,           
            auto_process_tool_calls=auto_process_tool_calls,
            messages=self.memory if messages is None else messages,
            **kwargs
        )        

        return response
    

    def search_my_conversations(self, query: str, limit: int = 5) -> str:
        """Search through this NPC's conversation history for relevant information"""
        if not self.command_history:
            return "No conversation history available"
        
        results = self.command_history.search_conversations(query)
        
        if not results:
            return f"No conversations found matching '{query}'"
        
        formatted_results = []
        for result in results[:limit]:
            timestamp = result.get('timestamp', 'Unknown time')
            content = result.get('content', '')[:200] + ('...' if len(result.get('content', '')) > 200 else '')
            formatted_results.append(f"[{timestamp}] {content}")
        
        return f"Found {len(results)} conversations matching '{query}'s:\n" + "\n".join(formatted_results)

    def search_my_memories(self, query: str, limit: int = 10) -> str:
        """Search through this NPC's knowledge graph memories for relevant facts and concepts"""
        if not self.kg_data:
            return "No memories available"
        
        query_lower = query.lower()
        relevant_facts = []
        relevant_concepts = []
        
        for fact in self.kg_data.get('facts', []):
            if query_lower in fact.get('statement', '').lower():
                relevant_facts.append(fact['statement'])
        
        for concept in self.kg_data.get('concepts', []):
            if query_lower in concept.get('name', '').lower():
                relevant_concepts.append(concept['name'])
        
        result_parts = []
        if relevant_facts:
            result_parts.append(f"Relevant memories: {'; '.join(relevant_facts[:limit])}")
        if relevant_concepts:
            result_parts.append(f"Related concepts: {', '.join(relevant_concepts[:limit])}")
        
        return "\n".join(result_parts) if result_parts else f"No memories found matching '{query}'"

    def query_database(self, sql_query: str) -> str:
        """Execute a SQL query against the available database"""
        if not self.db_conn:
            return "No database connection available"
        
        try:
            with self.db_conn.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                
                if not rows:
                    return "Query executed successfully but returned no results"
                
                columns = result.keys()
                formatted_rows = []
                for row in rows[:20]:  
                    row_dict = dict(zip(columns, row))
                    formatted_rows.append(str(row_dict))
                
                return f"Query results ({len(rows)} total rows, showing first 20):\n" + "\n".join(formatted_rows)
        
        except Exception as e:
            return f"Database query error: {str(e)}"

    def think_step_by_step(self, problem: str) -> str:
        """Think through a problem step by step using chain of thought reasoning"""
        thinking_prompt = f"""Think through this problem step by step:

    {problem}

    Break down your reasoning into clear steps:
    1. First, I need to understand...
    2. Then, I should consider...
    3. Next, I need to...
    4. Finally, I can conclude...

    Provide your step-by-step analysis.
    Do not under any circumstances ask for feedback from a user. These thoughts are part of an agentic tool that is letting the agent
    break down a problem by thinking it through. they will review the results and use them accordingly. 

    
    """
        
        response = self.get_llm_response(thinking_prompt, tool_choice = False)
        return response.get('response', 'Unable to process thinking request')

    def write_code(self, task: str, language: str = "python") -> str:
        """Write code to accomplish a task.

        Args:
            task: Description of what the code should do
            language: Programming language to use (default: python)

        Returns:
            The generated code as a string
        """
        code_prompt = f"""Write {language} code to accomplish the following task:

{task}

Requirements:
- Write clean, well-commented code
- Include error handling where appropriate
- Make sure the code is complete and runnable
- Only output the code, no explanations before or after

```{language}
"""

        response = self.get_llm_response(code_prompt, tool_choice=False)
        code = response.get('response', '')

        if f'```{language}' in code:
            code = code.split(f'```{language}')[-1]
        if '```' in code:
            code = code.split('```')[0]

        return code.strip()

    def create_planning_state(self, goal: str) -> Dict[str, Any]:
        """Create initial planning state for a goal"""
        return {
            "goal": goal,
            "todos": [],
            "constraints": [],
            "facts": [],
            "mistakes": [],
            "successes": [],
            "current_todo_index": 0,
            "current_subtodo_index": 0,
            "context_summary": ""
        }

    def generate_todos(self, user_goal: str, planning_state: Dict[str, Any], additional_context: str = "") -> List[Dict[str, Any]]:
        """Generate high-level todos for a goal"""
        prompt = f"""
        You are a high-level project planner. Structure tasks logically:
        1. Understand current state
        2. Make required changes 
        3. Verify changes work

        User goal: {user_goal}
        {additional_context}
        
        Generate 3-5 todos to accomplish this goal. Use specific actionable language.
        Each todo should be independent where possible and focused on a single component.
        
        Return JSON:
        {{
            "todos": [
                {{"description": "todo description", "estimated_complexity": "simple|medium|complex"}},
                ...
            ]
        }}
        """
        
        response = self.get_llm_response(prompt, format="json", tool_choice=False)
        todos_data = response.get("response", {}).get("todos", [])
        return todos_data

    def should_break_down_todo(self, todo: Dict[str, Any]) -> bool:
        """Ask LLM if a todo needs breakdown"""
        prompt = f"""
        Todo: {todo['description']}
        Complexity: {todo.get('estimated_complexity', 'unknown')}
        
        Should this be broken into smaller steps? Consider:
        - Is it complex enough to warrant breakdown?
        - Would breakdown make execution clearer?
        - Are there multiple distinct steps?
        
        Return JSON: {{"should_break_down": true/false, "reason": "explanation"}}
        """
        
        response = self.get_llm_response(prompt, format="json", tool_choice=False)
        result = response.get("response", {})
        return result.get("should_break_down", False)

    def generate_subtodos(self, todo: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate atomic subtodos for a complex todo"""
        prompt = f"""
        Parent todo: {todo['description']}
        
        Break this into atomic, executable subtodos. Each should be:
        - A single, concrete action
        - Executable in one step
        - Clear and unambiguous
        
        Return JSON:
        {{
            "subtodos": [
                {{"description": "subtodo description", "type": "action|verification|analysis"}},
                ...
            ]
        }}
        """
        
        response = self.get_llm_response(prompt, format="json")
        return response.get("response", {}).get("subtodos", [])

    def execute_planning_item(self, item: Dict[str, Any], planning_state: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Execute a single planning item (todo or subtodo)"""
        context_summary = self.get_planning_context_summary(planning_state)
        
        command = f"""
        Current context:
        {context_summary}
        {context}
        
        Execute this task: {item['description']}
        
        Constraints to follow:
        {chr(10).join([f"- {c}" for c in planning_state.get('constraints', [])])}
        """
        
        result = self.check_llm_command(
            command,
            context=self.shared_context,
            stream=False
        )
        
        return result

    def get_planning_context_summary(self, planning_state: Dict[str, Any]) -> str:
        """Get lightweight context for planning prompts"""
        context = []
        facts = planning_state.get('facts', [])
        mistakes = planning_state.get('mistakes', [])
        successes = planning_state.get('successes', [])
        
        if facts:
            context.append(f"Facts: {'; '.join(facts[:5])}")
        if mistakes:
            context.append(f"Recent mistakes: {'; '.join(mistakes[-3:])}")
        if successes:
            context.append(f"Recent successes: {'; '.join(successes[-3:])}")
        return "\n".join(context)

    def compress_planning_state(self, messages):
        if isinstance(messages, list):
            from npcpy.llm_funcs import breathe, get_facts
            
            conversation_summary = breathe(messages=messages, npc=self)
            summary_data = conversation_summary.get('output', '')
            
            conversation_text = "\n".join([msg['content'] for msg in messages])
            extracted_facts = get_facts(conversation_text, model=self.model, provider=self.provider, npc=self)
            
            user_inputs = [msg['content'] for msg in messages if msg.get('role') == 'user']
            assistant_outputs = [msg['content'] for msg in messages if msg.get('role') == 'assistant']
            
            planning_state = {
                "goal": summary_data,
                "facts": [fact['statement'] if isinstance(fact, dict) else str(fact) for fact in extracted_facts[-10:]],
                "successes": [output[:100] for output in assistant_outputs[-5:]],
                "mistakes": [],
                "todos": user_inputs[-3:],
                "constraints": []
            }
        else:
            planning_state = messages
        
        todos = planning_state.get('todos', [])
        current_index = planning_state.get('current_todo_index', 0)
        
        if todos and current_index < len(todos):
            current_focus = todos[current_index].get('description', todos[current_index]) if isinstance(todos[current_index], dict) else str(todos[current_index])
        else:
            current_focus = 'No current task'
        
        compressed = {
            "goal": planning_state.get("goal", ""),
            "progress": f"{len(planning_state.get('successes', []))}/{len(todos)} todos completed",
            "context": self.get_planning_context_summary(planning_state),
            "current_focus": current_focus
        }
        return json.dumps(compressed, indent=2)

    def decompress_planning_state(self, compressed_state: str) -> Dict[str, Any]:
        """Restore planning state from compressed string"""
        try:
            data = json.loads(compressed_state)
            return {
                "goal": data.get("goal", ""),
                "todos": [],
                "constraints": [],
                "facts": [],
                "mistakes": [],
                "successes": [],
                "current_todo_index": 0,
                "current_subtodo_index": 0,
                "compressed_context": data.get("context", "")
            }
        except json.JSONDecodeError:
            return self.create_planning_state("")

    def run_planning_loop(self, user_goal: str, interactive: bool = True) -> Dict[str, Any]:
        """Run the full planning loop for a goal"""
        planning_state = self.create_planning_state(user_goal)
        
        todos = self.generate_todos(user_goal, planning_state)
        planning_state["todos"] = todos
        
        for i, todo in enumerate(todos):
            planning_state["current_todo_index"] = i
            
            if self.should_break_down_todo(todo):
                subtodos = self.generate_subtodos(todo)
                
                for j, subtodo in enumerate(subtodos):
                    planning_state["current_subtodo_index"] = j
                    result = self.execute_planning_item(subtodo, planning_state)
                    
                    if result.get("output"):
                        planning_state["successes"].append(f"Completed: {subtodo['description']}")
                    else:
                        planning_state["mistakes"].append(f"Failed: {subtodo['description']}")
            else:
                result = self.execute_planning_item(todo, planning_state)
                
                if result.get("output"):
                    planning_state["successes"].append(f"Completed: {todo['description']}")
                else:
                    planning_state["mistakes"].append(f"Failed: {todo['description']}")
        
        return {
            "planning_state": planning_state,
            "compressed_state": self.compress_planning_state(planning_state),
            "summary": f"Completed {len(planning_state['successes'])} tasks for goal: {user_goal}"
        }
        
    def execute_jinx(
        self,
        jinx_name,
        inputs,
        conversation_id=None,
        message_id=None,
        team_name=None,
        extra_globals=None
    ):
        if jinx_name in self.jinxs_dict:
            jinx = self.jinxs_dict[jinx_name]
        else:
            return {"error": f"jinx '{jinx_name}' not found"}

        import time as _time
        _start = _time.monotonic()
        _status = "success"
        _error = None

        try:
            result = jinx.execute(
                input_values=inputs,
                npc=self,
                extra_globals=extra_globals,
                jinja_env=self.jinja_env
            )
        except Exception as e:
            _status = "error"
            _error = str(e)
            result = {"error": str(e)}

        _duration_ms = int((_time.monotonic() - _start) * 1000)

        if self.command_history is not None and hasattr(self.command_history, 'save_jinx_execution'):
            try:
                self.command_history.save_jinx_execution(
                    triggering_message_id=message_id,
                    conversation_id=conversation_id,
                    jinx_name=jinx_name,
                    jinx_inputs=inputs,
                    jinx_output=result,
                    status=_status,
                    error_message=_error,
                    duration_ms=_duration_ms,
                    npc_name=self.name,
                    team_name=team_name,
                )
            except Exception:
                pass
        return result
    def check_llm_command(self,
                            command,
                            messages=None,
                            context=None,
                            team=None,
                            stream=False,
                            jinxs=None, 
                            use_jinxs=True):
        """Check if a command is for the LLM"""
        if context is None:
            context = self.shared_context

        if team:
            self._current_team = team
        if jinxs is None and use_jinxs:
            jinxs_to_use = self.jinxs_dict
        elif jinxs is not None and use_jinxs:
            jinxs_to_use = jinxs
            
        return npy.llm_funcs.check_llm_command(
            command,
            model=self.model,
            provider=self.provider,
            npc=self,
            team=team,
            messages=self.memory if messages is None else messages,
            context=context,
            stream=stream,
            jinxs=jinxs_to_use,
        )
    
    def handle_agent_pass(self, 
                            npc_to_pass,
                            command, 
                            messages=None, 
                            context=None, 
                            shared_context=None, 
                            stream=False,
                            team=None):  
        """Pass a command to another NPC"""
        print('handling agent pass')
        if isinstance(npc_to_pass, NPC):
            target_npc = npc_to_pass
        else:
            return {"error": "Invalid NPC to pass command to"}
        
        if shared_context is not None:
            self.shared_context.update(shared_context)
            target_npc.shared_context.update(shared_context)
            
        updated_command = (
            command
            + "\n\n"
            + f"NOTE: THIS COMMAND HAS BEEN PASSED FROM {self.name} TO YOU, {target_npc.name}.\n"
            + "PLEASE CHOOSE ONE OF THE OTHER OPTIONS WHEN RESPONDING."
        )

        result = target_npc.check_llm_command(
            updated_command,
            messages=messages,
            context=target_npc.shared_context,
            team=team, 
            stream=stream
        )
        if isinstance(result, dict):
            result['npc_name'] = target_npc.name
            result['passed_from'] = self.name
        
        return result    

    def to_dict(self):
        """Convert NPC to dictionary representation"""
        jinx_rep = [] 
        if self.jinxs_dict:
            jinx_rep = [ jinx.to_dict() for jinx in self.jinxs_dict.values()]
        return {
            "name": self.name,
            "primary_directive": self.primary_directive,
            "model": self.model,
            "provider": self.provider,
            "api_url": self.api_url,
            "api_key": self.api_key,
            "jinxs": self.jinxs_spec,
            "use_global_jinxs": self.use_global_jinxs
        }
        
    def save(self, directory=None):
        """Save NPC to file"""
        if directory is None:
            directory = self.npc_directory
            
        ensure_dirs_exist(directory)
        npc_path = os.path.join(directory, f"{self.name}.npc")
        
        return write_yaml_file(npc_path, self.to_dict())
    
    def __str__(self):
        """String representation of NPC"""
        str_rep = f"NPC: {self.name}\nDirective: {self.primary_directive}\nModel: {self.model}\nProvider: {self.provider}\nAPI URL: {self.api_url}\n"
        if self.jinxs_dict:
            str_rep += "Jinxs:\n"
            for jinx_name in self.jinxs_dict.keys():
                str_rep += f"  - {jinx_name}\n"
        else:
            str_rep += "No jinxs available.\n"
        return str_rep

    def execute_jinx_command(self, 
        jinx: Jinx,
        args: List[str],
        messages=None,
    ) -> Dict[str, Any]:
        """
        Execute a jinx command with the given arguments.
        """
        
        input_values = extract_jinx_inputs(args, jinx)

        jinx_output = jinx.execute(
            input_values,
            npc=self,
            messages=messages,
            jinja_env=self.jinja_env
        )

        return {"messages": messages, "output": jinx_output}
    def create_memory(self, content: str, memory_type: str = "observation") -> Optional[int]:
        """Create a new memory entry"""
        if not self.command_history:
            return None
        
        message_id = generate_message_id()
        conversation_id = self.command_history.get_most_recent_conversation_id()
        conversation_id = conversation_id.get('conversation_id') if conversation_id else 'direct_memory'
        
        team_name = getattr(self.team, 'name', 'default_team') if self.team else 'default_team'
        directory_path = os.getcwd()
        
        return self.command_history.add_memory_to_database(
            message_id=message_id,
            conversation_id=conversation_id,
            npc=self.name,
            team=team_name,
            directory_path=directory_path,
            initial_memory=content,
            status='active',
            model=self.model,
            provider=self.provider
        )

    def read_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Read a specific memory by ID"""
        if not self.command_history:
            return None
        
        stmt = "SELECT * FROM memory_lifecycle WHERE id = :memory_id"
        return self.command_history._fetch_one(stmt, {"memory_id": memory_id})

    def update_memory(self, memory_id: int, new_content: str = None, status: str = None) -> bool:
        """Update memory content or status"""
        if not self.command_history:
            return False
        
        updates = []
        params = {"memory_id": memory_id}
        
        if new_content is not None:
            updates.append("final_memory = :final_memory")
            params["final_memory"] = new_content
        
        if status is not None:
            updates.append("status = :status") 
            params["status"] = status
        
        if not updates:
            return False
        
        stmt = f"UPDATE memory_lifecycle SET {', '.join(updates)} WHERE id = :memory_id"
        
        try:
            with self.command_history.engine.begin() as conn:
                conn.execute(text(stmt), params)
            return True
        except Exception as e:
            print(f"Error updating memory {memory_id}: {e}")
            return False

    def delete_memory(self, memory_id: int) -> bool:
        """Delete a memory by ID"""
        if not self.command_history:
            return False
        
        stmt = "DELETE FROM memory_lifecycle WHERE id = :memory_id AND npc = :npc"
        
        try:
            with self.command_history.engine.begin() as conn:
                result = conn.execute(text(stmt), {"memory_id": memory_id, "npc": self.name})
                return result.rowcount > 0
        except Exception as e:
            print(f"Error deleting memory {memory_id}: {e}")
            return False

    def search_memories(self, query: str, limit: int = 10, status_filter: str = None) -> List[Dict[str, Any]]:
        """Search memories with optional status filtering"""
        if not self.command_history:
            return []
        
        team_name = getattr(self.team, 'name', 'default_team') if self.team else 'default_team'
        directory_path = os.getcwd()
        
        return self.command_history.search_memory(
            query=query,
            npc=self.name,
            team=team_name,
            directory_path=directory_path,
            status_filter=status_filter,
            limit=limit
        )

    def get_all_memories(self, limit: int = 50, status_filter: str = None) -> List[Dict[str, Any]]:
        """Get all memories for this NPC with optional status filtering"""
        if not self.command_history:
            return []
        
        if limit is None:
            limit = 50
        
        conditions = ["npc = :npc"]
        params = {"npc": self.name, "limit": limit}
        
        if status_filter:
            conditions.append("status = :status")
            params["status"] = status_filter
        
        stmt = f"""
            SELECT * FROM memory_lifecycle 
            WHERE {' AND '.join(conditions)}
            ORDER BY created_at DESC 
            LIMIT :limit
            """
        
        return self.command_history._fetch_all(stmt, params)

    def archive_old_memories(self, days_old: int = 30) -> int:
        """Archive memories older than specified days"""
        if not self.command_history:
            return 0
        
        stmt = """
            UPDATE memory_lifecycle 
            SET status = 'archived' 
            WHERE npc = :npc 
            AND status = 'active'
            AND datetime(created_at) < datetime('now', '-{} days')
        """.format(days_old)
        
        try:
            with self.command_history.engine.begin() as conn:
                result = conn.execute(text(stmt), {"npc": self.name})
                return result.rowcount
        except Exception as e:
            print(f"Error archiving memories: {e}")
            return 0

    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics for this NPC"""
        if not self.command_history:
            return {}
        
        stmt = """
            SELECT status, COUNT(*) as count
            FROM memory_lifecycle 
            WHERE npc = :npc
            GROUP BY status
        """
        
        results = self.command_history._fetch_all(stmt, {"npc": self.name})
        return {row['status']: row['count'] for row in results}

class Team:
    def __init__(self, 
                    team_path=None, 
                    npcs: Optional[List['NPC']] = None,
                    forenpc: Optional[Union[str, 'NPC']] = None,
                    jinxs: Optional[List[Union['Jinx', Dict[str, Any]]]] = None,
                    db_conn=None, 
                    model = None, 
                    provider = None, 
                    api_url = None, 
                    api_key = None):
        """
        Initialize an NPC team from directory or list of NPCs
        
        Args:
            team_path: Path to team directory
            npcs: List of NPC objects
            db_conn: Database connection
        """
        self.model = model
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        
        self.npcs: Dict[str, 'NPC'] = {}
        self.sub_teams: Dict[str, 'Team'] = {}
        self.jinxs_dict: Dict[str, 'Jinx'] = {}
        self._raw_jinxs_list: List['Jinx'] = []
        self.jinx_tool_catalog: Dict[str, Dict[str, Any]] = {}
        
        self.jinja_env_for_first_pass = SandboxedEnvironment(undefined=SilentUndefined)

        self.db_conn = db_conn
        self.team_path = os.path.expanduser(team_path) if team_path else None
        self.databases = []
        self.mcp_servers = []
        
        self.forenpc: Optional['NPC'] = None
        self.forenpc_name: Optional[str] = None
        self.skills_directory: Optional[str] = None

        if team_path:
            self.name = os.path.basename(os.path.abspath(team_path))
            self._load_from_directory_and_initialize_forenpc() 
        elif npcs:
            self.name = "custom_team"
            for npc_obj in npcs:
                self.npcs[npc_obj.name] = npc_obj
                npc_obj.team = self
            
            if jinxs:
                for jinx_item in jinxs:
                    if isinstance(jinx_item, Jinx):
                        self._raw_jinxs_list.append(jinx_item)
                    elif isinstance(jinx_item, dict):
                        self._raw_jinxs_list.append(Jinx(jinx_data=jinx_item))
        
            self._determine_forenpc_from_provided_npcs(npcs, forenpc)

        else:
            self.name = "custom_team"
            self._create_default_forenpc()

        self.context = ''
        self.shared_context = {
            "intermediate_results": {},
            "dataframes": {},
            "memories": {},          
            "execution_history": [],   
            "context":''       
            }
        
        if team_path:
            self._load_team_context_into_shared_context()
        elif self.forenpc:
            if not self.context:
                self.context = f"Team '{self.name}' with forenpc '{self.forenpc.name}'"
                self.shared_context['context'] = self.context

        self._perform_first_pass_jinx_rendering()
        self.jinx_tool_catalog = build_jinx_tool_catalog(self.jinxs_dict)
        print(f"[TEAM] Built Jinx tool catalog with {len(self.jinx_tool_catalog)} entries for team {self.name}")

        for npc_obj in self.npcs.values():
            npc_obj.initialize_jinxs(team_raw_jinxs=self._raw_jinxs_list) 
        
        if db_conn is not None:
            init_db_tables()

    def _load_from_directory_and_initialize_forenpc(self):
        """
        Consolidated method to load NPCs, team context, and resolve the forenpc.
        Ensures self.npcs is populated and self.forenpc is an NPC object.
        """
        if not os.path.exists(self.team_path):
            raise ValueError(f"Team directory not found: {self.team_path}")
        
        self._load_team_context_file()

        for filename in os.listdir(self.team_path):
            if filename.endswith(".npc"):
                npc_path = os.path.join(self.team_path, filename)
                npc = NPC(npc_path, db_conn=self.db_conn, team=self)
                self.npcs[npc.name] = npc

        if self.forenpc_name and self.forenpc_name in self.npcs:
            self.forenpc = self.npcs[self.forenpc_name]
        elif self.npcs:
            self.forenpc = list(self.npcs.values())[0]
            self.forenpc_name = self.forenpc.name
        else:
            self._create_default_forenpc()
        
        jinxs_dir = os.path.join(self.team_path, "jinxs")
        if os.path.exists(jinxs_dir):
            for jinx_obj in load_jinxs_from_directory(jinxs_dir):
                self._raw_jinxs_list.append(jinx_obj)
        
        if hasattr(self, 'skills_directory') and self.skills_directory:
            skills_path = os.path.expanduser(self.skills_directory)
            if not os.path.isabs(skills_path):
                skills_path = os.path.join(self.team_path, skills_path)
            if os.path.exists(skills_path):
                for jinx_obj in load_jinxs_from_directory(skills_path):
                    self._raw_jinxs_list.append(jinx_obj)
                print(f"[TEAM] Loaded skills from SKILLS_DIRECTORY: {skills_path}")
            else:
                print(f"[TEAM] Warning: SKILLS_DIRECTORY not found: {skills_path}")

        self._load_sub_teams()

    def _load_team_context_file(self) -> Dict[str, Any]:
        """Loads team context from .ctx file and updates team attributes."""
        ctx_data = {}
        for fname in os.listdir(self.team_path):
            if fname.endswith('.ctx'):
                ctx_data = load_yaml_file(os.path.join(self.team_path, fname))                
                if ctx_data is not None:
                    self.model = ctx_data.get('model', self.model)
                    self.provider = ctx_data.get('provider', self.provider)
                    self.api_url = ctx_data.get('api_url', self.api_url)
                    self.env = ctx_data.get('env', self.env if hasattr(self, 'env') else None)
                    self.mcp_servers = ctx_data.get('mcp_servers', [])
                    self.databases = ctx_data.get('databases', [])
                    self.forenpc_name = ctx_data.get('forenpc', self.forenpc_name)
                    self.skills_directory = ctx_data.get('SKILLS_DIRECTORY', None)
                return ctx_data
        return {}

    def _load_team_context_into_shared_context(self):
        """Loads team context into shared_context after forenpc is determined."""
        ctx_data = {}
        for fname in os.listdir(self.team_path):
            if fname.endswith('.ctx'):
                ctx_data = load_yaml_file(os.path.join(self.team_path, fname))                
                if ctx_data is not None:
                    self.context = ctx_data.get('context', '')
                    self.shared_context['context'] = self.context
                    if 'file_patterns' in ctx_data:
                        file_cache = self._parse_file_patterns(ctx_data['file_patterns'])
                        self.shared_context['files'] = file_cache
                    for key, item in ctx_data.items():
                        if key not in ['name', 'mcp_servers', 'databases', 'context', 'file_patterns', 'forenpc', 'model', 'provider', 'api_url', 'env', 'SKILLS_DIRECTORY']:
                            self.shared_context[key] = item
                return
        
    def _determine_forenpc_from_provided_npcs(self, npcs_list: List['NPC'], forenpc_arg: Optional[Union[str, 'NPC']]):
        """Determines self.forenpc when NPCs are provided directly to Team.__init__."""
        if forenpc_arg:
            if isinstance(forenpc_arg, NPC):
                self.forenpc = forenpc_arg
                self.forenpc_name = forenpc_arg.name
            elif isinstance(forenpc_arg, str) and forenpc_arg in self.npcs:
                self.forenpc = self.npcs[forenpc_arg]
                self.forenpc_name = forenpc_arg
            else:
                print(f"Warning: Specified forenpc '{forenpc_arg}' not found among provided NPCs. Falling back to first NPC.")
                if npcs_list:
                    self.forenpc = npcs_list[0]
                    self.forenpc_name = npcs_list[0].name
                else:
                    self._create_default_forenpc()
        elif npcs_list:
            self.forenpc = npcs_list[0]
            self.forenpc_name = npcs_list[0].name
        else:
            self._create_default_forenpc()

    def _create_default_forenpc(self):
        """Creates a default forenpc if none can be determined."""
        forenpc_model = self.model or 'llama3.2'
        forenpc_provider = self.provider or 'ollama'
        forenpc_api_key = self.api_key
        forenpc_api_url = self.api_url
        
        default_forenpc = NPC(name='forenpc', 
                                primary_directive="""You are the forenpc of the team, coordinating activities 
                                                    between NPCs on the team, verifying that results from 
                                                    NPCs are high quality and can help to adequately answer 
                                                    user requests.""", 
                                model=forenpc_model,
                                provider=forenpc_provider,
                                api_key=forenpc_api_key,
                                api_url=forenpc_api_url,                            
                                team=self
                                                    )
        self.forenpc = default_forenpc
        self.forenpc_name = default_forenpc.name
        self.npcs[default_forenpc.name] = default_forenpc

    def _perform_first_pass_jinx_rendering(self):
        """
        Performs the first-pass Jinja rendering on all loaded raw Jinxs.
        This expands nested Jinx calls but preserves runtime variables.
        """
        jinx_macro_globals = {}
        for raw_jinx in self._raw_jinxs_list:
            def create_jinx_callable(jinx_obj_in_closure):
                def callable_jinx(**kwargs):
                    temp_jinja_env = SandboxedEnvironment(undefined=SilentUndefined)
                    
                    rendered_target_steps = []
                    for target_step in jinx_obj_in_closure._raw_steps:
                        temp_rendered_step = {}
                        for k, v in target_step.items():
                            if isinstance(v, str):
                                try:
                                    temp_rendered_step[k] = temp_jinja_env.from_string(v).render(**kwargs)
                                except Exception as e:
                                    print(f"Warning: Error in Jinx macro '{jinx_obj_in_closure.jinx_name}' rendering step field '{k}' (Team first pass): {e}")
                                    temp_rendered_step[k] = v
                            else:
                                temp_rendered_step[k] = v
                        rendered_target_steps.append(temp_rendered_step)
                    
                    return yaml.dump(rendered_target_steps, default_flow_style=False)
                return callable_jinx
            
            jinx_macro_globals[raw_jinx.jinx_name] = create_jinx_callable(raw_jinx)
        
        self.jinja_env_for_first_pass.globals['jinxs'] = jinx_macro_globals
        self.jinja_env_for_first_pass.globals.update(jinx_macro_globals)

        for raw_jinx in self._raw_jinxs_list:
            try:
                raw_jinx.render_first_pass(self.jinja_env_for_first_pass, jinx_macro_globals)
                self.jinxs_dict[raw_jinx.jinx_name] = raw_jinx
            except Exception as e:
                print(f"Error performing first-pass rendering for Jinx '{raw_jinx.jinx_name}': {e}")

    def update_context(self, messages: list):
        """Update team context based on recent conversation patterns"""
        if len(messages) < 10:
            return
            
        summary = breathe(
            messages=messages[-10:], 
            npc=self.forenpc
        )
        characterization = summary.get('output')
        
        if characterization:
            team_ctx_path = os.path.join(self.team_path, "team.ctx")
            
            if os.path.exists(team_ctx_path):
                with open(team_ctx_path, 'r') as f:
                    ctx_data = yaml.safe_load(f) or {}
            else:
                ctx_data = {}
                
            current_context = ctx_data.get('context', '')
            
            prompt = f"""Based on this characterization: {characterization},
            suggest changes to the team's context.
            Current Context: "{current_context}".
            Respond with JSON: {{"suggestion": "Your sentence."}}"""
            
            response = get_llm_response(
                prompt=prompt,
                npc=self.forenpc,
                format="json"
            )
            suggestion = response.get("response", {}).get("suggestion")
            
            if suggestion:
                new_context = (current_context + " " + suggestion).strip()
                user_approval = input(f"Update context to: {new_context}? [y/N]: ").strip().lower()
                if user_approval == 'y':
                    ctx_data['context'] = new_context
                    self.context = new_context
                    with open(team_ctx_path, 'w') as f:
                        yaml.dump(ctx_data, f)
            
    def _load_sub_teams(self):
        """Load sub-teams from subdirectories"""
        for item in os.listdir(self.team_path):
            item_path = os.path.join(self.team_path, item)
            if (os.path.isdir(item_path) and 
                not item.startswith('.') and 
                item != "jinxs"):
                
                if any(f.endswith(".npc") for f in os.listdir(item_path) 
                        if os.path.isfile(os.path.join(item_path, f))):
                    sub_team = Team(team_path=item_path, db_conn=self.db_conn)
                    self.sub_teams[item] = sub_team
        
    def get_forenpc(self) -> Optional['NPC']:
        """
        Returns the forenpc (coordinator) for this team.
        This method is now primarily for external access, as self.forenpc is set in __init__.
        """
        return self.forenpc

    def get_npc(self, npc_ref: Union[str, 'NPC']) -> Optional['NPC']:
        """Get NPC by name or reference with hierarchical lookup capability"""
        if isinstance(npc_ref, NPC):
            return npc_ref
        elif isinstance(npc_ref, str):
            if npc_ref in self.npcs:
                return self.npcs[npc_ref]
            
            for sub_team_name, sub_team in self.sub_teams.items():
                if npc_ref in sub_team.npcs:
                    return sub_team.npcs[npc_ref]
                
                result = sub_team.get_npc(npc_ref)
                if result:
                    return result
            
            return None
        else:
            return None

    def orchestrate(self, request, max_iterations=3):
        """Orchestrate a request through the team"""
        import re
        from termcolor import colored

        forenpc = self.get_forenpc()
        if not forenpc:
            return {"error": "No forenpc available to coordinate the team"}

        print(colored(f"[orchestrate] Starting with forenpc={forenpc.name}, team={self.name}", "cyan"))
        print(colored(f"[orchestrate] Request: {request[:100]}...", "cyan"))

        jinxs_for_orchestration = {k: v for k, v in forenpc.jinxs_dict.items() if k != 'orchestrate'}

        try:
            result = forenpc.check_llm_command(
                request,
                context=getattr(self, 'context', {}),
                team=self,
                jinxs=jinxs_for_orchestration,
            )
            print(colored(f"[orchestrate] Initial result type={type(result)}", "cyan"))
            if isinstance(result, dict):
                print(colored(f"[orchestrate] Result keys={list(result.keys())}", "cyan"))
                if 'error' in result:
                    print(colored(f"[orchestrate] Error in result: {result['error']}", "red"))
                    return result
        except Exception as e:
            print(colored(f"[orchestrate] Exception in check_llm_command: {e}", "red"))
            return {"error": str(e), "output": f"Orchestration failed: {e}"}

        output = ""
        if isinstance(result, dict):
            output = result.get('output') or result.get('response') or ""

        print(colored(f"[orchestrate] Output preview: {output[:200] if output else 'EMPTY'}...", "cyan"))

        if output and self.npcs:
            at_pattern = r'@(\w+)'
            mentions = re.findall(at_pattern, output)

            if not mentions:
                for npc_name in self.npcs.keys():
                    if npc_name.lower() != forenpc.name.lower():
                        if npc_name.lower() in output.lower():
                            mentions.append(npc_name)
                            break

            print(colored(f"[orchestrate] Found mentions: {mentions}", "cyan"))

            for mentioned in mentions:
                mentioned_lower = mentioned.lower()
                if mentioned_lower in self.npcs and mentioned_lower != forenpc.name:
                    target_npc = self.npcs[mentioned_lower]
                    print(colored(f"[orchestrate] Delegating to @{mentioned_lower}", "yellow"))

                    try:
                        target_jinxs = {k: v for k, v in target_npc.jinxs_dict.items() if k != 'orchestrate'}
                        delegate_result = target_npc.check_llm_command(
                            request,
                            context=getattr(self, 'context', {}),
                            team=self,
                            jinxs=target_jinxs,
                        )

                        if isinstance(delegate_result, dict):
                            delegate_output = delegate_result.get('output') or delegate_result.get('response') or ""
                            if delegate_output:
                                output = f"[{mentioned_lower}]: {delegate_output}"
                                result = delegate_result
                                print(colored(f"[orchestrate] Got response from {mentioned_lower}", "green"))
                    except Exception as e:
                        print(colored(f"[orchestrate] Delegation to {mentioned_lower} failed: {e}", "red"))

                    break

        if isinstance(result, dict):
            final_output = output if output else str(result)
            return {
                "output": final_output,
                "result": result,
            }
        else:
            return {
                "output": str(result),
                "result": result,
            }
                
    def to_dict(self):
        """Convert team to dictionary representation"""
        return {
            "name": self.name,
            "npcs": {name: npc.to_dict() for name, npc in self.npcs.items()},
            "sub_teams": {name: team.to_dict() for name, team in self.sub_teams.items()},
            "jinxs": {name: jinx.to_dict() for name, jinx in self.jinxs_dict.items()},
            "context": getattr(self, 'context', {})
        }
    
    def save(self, directory=None):
        """Save team to directory"""
        if directory is None:
            directory = self.team_path
            
        if not directory:
            raise ValueError("No directory specified for saving team")
            
        ensure_dirs_exist(directory)
        
        if hasattr(self, 'context') and self.context:
            ctx_path = os.path.join(directory, "team.ctx")
            write_yaml_file(ctx_path, self.context)
            
        for npc in self.npcs.values():
            npc.save(directory)
            
        jinxs_dir = os.path.join(directory, "jinxs")
        ensure_dirs_exist(jinxs_dir)
        
        for jinx in self.jinxs_dict.values():
            jinx.save(jinxs_dir)
            
        for team_name, team in self.sub_teams.items():
            team_dir = os.path.join(directory, team_name)
            team.save(team_dir)
            
        return True
    def _parse_file_patterns(self, patterns_config):
        """Parse file patterns configuration and load matching files into KV cache"""
        if not patterns_config:
            return {}
        
        file_cache = {}
        
        for pattern_entry in patterns_config:
            if isinstance(pattern_entry, str):
                pattern_entry = {"pattern": pattern_entry}
            
            pattern = pattern_entry.get("pattern", "")
            recursive = pattern_entry.get("recursive", False)
            base_path = pattern_entry.get("base_path", ".")
            
            if not pattern:
                continue
                
            base_path = os.path.expanduser(base_path)
            if not os.path.isabs(base_path):
                base_path = os.path.join(self.team_path or os.getcwd(), base_path)
            
            matching_files = self._find_matching_files(pattern, base_path, recursive)
            
            for file_path in matching_files:
                file_content = self._load_file_content(file_path)
                if file_content:
                    relative_path = os.path.relpath(file_path, base_path)
                    file_cache[relative_path] = file_content
        
        return file_cache

    def _find_matching_files(self, pattern, base_path, recursive=False):
        """Find files matching the given pattern"""
        matching_files = []
        
        if not os.path.exists(base_path):
            return matching_files
        
        if recursive:
            for root, dirs, files in os.walk(base_path):
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        matching_files.append(os.path.join(root, filename))
        else:
            try:
                for item in os.listdir(base_path):
                    item_path = os.path.join(base_path, item)
                    if os.path.isfile(item_path) and fnmatch.fnmatch(item, pattern):
                        matching_files.append(item_path)
            except PermissionError:
                print(f"Permission denied accessing {base_path}")
        
        return matching_files

    def _load_file_content(self, file_path):
        """Load content from a file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def _format_parsed_files_context(self, parsed_files):
        """Format parsed files into context string"""
        if not parsed_files:
            return ""
        
        context_parts = ["Additional context from files:"]
        
        for file_path, content in parsed_files.items():
            context_parts.append(f"\n--- {file_path} ---")
            context_parts.append(content)
            context_parts.append("")
        
        return "\n".join(context_parts)
