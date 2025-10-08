import os
from pyexpat.errors import messages
import yaml
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import random
from datetime import datetime
import hashlib
import pathlib
import fnmatch
import subprocess
from typing import Any, Dict, List, Optional, Union
from jinja2 import Environment, FileSystemLoader, Template, Undefined
from sqlalchemy import create_engine, text
import npcpy as npy 
from npcpy.llm_funcs import DEFAULT_ACTION_SPACE
from npcpy.tools import auto_tools

from npcpy.npc_sysenv import (
    ensure_dirs_exist, 
    init_db_tables,
    get_system_message, 

    )
from npcpy.memory.command_history import CommandHistory, generate_message_id

class SilentUndefined(Undefined):
    def _fail_with_undefined_error(self, *args, **kwargs):
        return ""

import math
from PIL import Image


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


def load_yaml_file(file_path):
    """Load a YAML file with error handling"""
    try:
        with open(os.path.expanduser(file_path), 'r') as f:
            return yaml.safe_load(f)
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

    npc_team_dir = os.path.join(directory, "npc_team")
    os.makedirs(npc_team_dir, exist_ok=True)
    
    for subdir in ["jinxs", 
                   "assembly_lines", 
                   "sql_models", 
                   "jobs", 
                   "triggers"]:
        os.makedirs(os.path.join(npc_team_dir, subdir), exist_ok=True)
    
    forenpc_path = os.path.join(npc_team_dir, "forenpc.npc")
    

    
    if not os.path.exists(forenpc_path):
        
        default_npc = {
            "name": "forenpc",
            "primary_directive": "You are the forenpc of an NPC team", 
        }
        with open(forenpc_path, "w") as f:
            yaml.dump(default_npc, f)
    ctx_path = os.path.join(npc_team_dir, "team.ctx")
    if not os.path.exists(ctx_path):
        default_ctx = {
            'name': '',
            'context' : '', 
            'preferences': '', 
            'mcp_servers': '', 
            'databases':'', 
            'use_global_jinxs': True,
            'forenpc': 'forenpc'
        }
        with open(ctx_path, "w") as f:
            yaml.dump(default_ctx, f)
            
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
    
    Jinx is a class that provides methods for rendering jinja templates to execute
    natural language commands within the NPC ecosystem, python, and eventually
    other code languages.
    '''
    def __init__(self, jinx_data=None, jinx_path=None):
        """Initialize a jinx from data or file path"""
        if jinx_path:
            self._load_from_file(jinx_path)
        elif jinx_data:
            self._load_from_data(jinx_data)
        else:
            raise ValueError("Either jinx_data or jinx_path must be provided")
            
    def _load_from_file(self, path):
        """Load jinx from file"""
        jinx_data = load_yaml_file(path)
        if not jinx_data:
            raise ValueError(f"Failed to load jinx from {path}")
        self._load_from_data(jinx_data)
            
    def _load_from_data(self, jinx_data):
        """Load jinx from data dictionary"""
        if not jinx_data or not isinstance(jinx_data, dict):
            raise ValueError("Invalid jinx data provided")
            
        if "jinx_name" not in jinx_data:
            raise KeyError("Missing 'jinx_name' in jinx definition")
            
        self.jinx_name = jinx_data.get("jinx_name")
        self.inputs = jinx_data.get("inputs", [])
        self.description = jinx_data.get("description", "")
        self.steps = self._parse_steps(jinx_data.get("steps", []))
            
    def _parse_steps(self, steps):
        """Parse steps from jinx definition"""
        parsed_steps = []
        for i, step in enumerate(steps):
            if isinstance(step, dict):
                parsed_step = {
                    "name": step.get("name", f"step_{i}"),
                    "engine": step.get("engine", "natural"),
                    "code": step.get("code", "")
                }
                parsed_steps.append(parsed_step)
            else:
                raise ValueError(f"Invalid step format: {step}")
        return parsed_steps
        
    def execute(self,
                input_values, 
                jinxs_dict, 
                jinja_env = None,
                npc = None,
                messages=None):
        """Execute the jinx with given inputs"""
        if jinja_env is None:
            
            
            from jinja2 import DictLoader
            jinja_env = Environment(
                loader=DictLoader({}),  
                undefined=SilentUndefined,
            )
        
        context = (npc.shared_context.copy() if npc else {})
        context.update(input_values)
        context.update({
            "jinxs": jinxs_dict,
            "llm_response": None,
            "output": None, 
            "messages": messages,
        })
        
        
        for i, step in enumerate(self.steps):
            context = self._execute_step(
                step, 
                context,
                jinja_env, 
                npc=npc, 
                messages=messages, 

            )            

        return context
            
    def _execute_step(self,
                      step, 
                      context, 
                      jinja_env,
                      npc=None,
                      messages=None, 
):
        """Execute a single step of the jinx"""
        engine = step.get("engine", "natural")
        code = step.get("code", "")
        step_name = step.get("name", "unnamed_step")

        
        

        try:
            
            template = jinja_env.from_string(code)
            rendered_code = template.render(**context)
            
            engine_template = jinja_env.from_string(engine)
            rendered_engine = engine_template.render(**context)
        
        except Exception as e:
            print(f"Error rendering templates for step {step_name}: {e}")
            rendered_code = code
            rendered_engine = engine
                
        
        if rendered_engine == "natural":
            if rendered_code.strip():
                
                response = npc.get_llm_response(
                    rendered_code,
                    context=context,
                    messages=messages,
                )
               
                response_text = response.get("response", "")
                context['output'] = response_text
                context["llm_response"] = response_text
                context["results"] = response_text
                context[step_name] = response_text
                context['messages'] = response.get('messages')
        elif rendered_engine == "python":
            
            exec_globals = {
                "__builtins__": __builtins__,
                "npc": npc,
                "context": context,
                "pd": pd,
                "plt": plt,
                "np": np,
                "os": os,
                're': re, 
                "json": json,
                "Path": pathlib.Path,
                "fnmatch": fnmatch,
                "pathlib": pathlib,
                "subprocess": subprocess,
                "get_llm_response": npy.llm_funcs.get_llm_response, 
                
                }
            
            
            
            exec_locals = {}
            exec(rendered_code, exec_globals, exec_locals)
            
            
            context.update(exec_locals)
            
            
            if "output" in exec_locals:
                outp = exec_locals["output"]
                context["output"] = outp
                context[step_name] = outp
                messages.append({'role':'assistant', 
                                 'content': f'Jinx executed with following output: {outp}'})
                context['messages'] = messages
                
        else:
            
            context[step_name] = {"error": f"Unsupported engine: {rendered_engine}"}
            
        return context
        
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            "jinx_name": self.jinx_name,
            "description": self.description,
            "inputs": self.inputs,
            "steps": [
                {
                    "name": step.get("name", f"step_{i}"),
                    "engine": step.get("engine"),
                    "code": step.get("code")
                }
                for i, step in enumerate(self.steps)
            ]
        }
        
    def save(self, directory):
        """Save jinx to file"""
        jinx_path = os.path.join(directory, f"{self.jinx_name}.jinx")
        ensure_dirs_exist(os.path.dirname(jinx_path))
        return write_yaml_file(jinx_path, self.to_dict())
        
    @classmethod
    def from_mcp(cls, mcp_tool):
        """Convert an MCP tool to NPC jinx format"""
        
        try:
            import inspect

            
            doc = mcp_tool.__doc__ or ""
            name = mcp_tool.__name__
            signature = inspect.signature(mcp_tool)
            
            
            inputs = []
            for param_name, param in signature.parameters.items():
                if param_name != 'self':  
                    param_type = param.annotation if param.annotation != inspect.Parameter.empty else None
                    param_default = None if param.default == inspect.Parameter.empty else param.default
                    
                    inputs.append({
                        "name": param_name,
                        "type": str(param_type),
                        "default": param_default
                    })
            
            
            jinx_data = {
                "jinx_name": name,
                "description": doc.strip(),
                "inputs": inputs,
                "steps": [
                    {
                        "name": "mcp_function_call",
                        "engine": "python",
                        "code": f"""

import {mcp_tool.__module__}
output = {mcp_tool.__module__}.{name}(
    {', '.join([f'{inp["name"]}=context.get("{inp["name"]}")' for inp in inputs])}
)
"""
                    }
                ]
            }
            
            return cls(jinx_data=jinx_data)
            
        except: 
            pass    


def load_jinxs_from_directory(directory):
    """Load all jinxs from a directory"""
    jinxs = []
    directory = os.path.expanduser(directory)
    
    if not os.path.exists(directory):
        return jinxs
        
    for filename in os.listdir(directory):
        if filename.endswith(".jinx"):
            try:
                jinx_path = os.path.join(directory, filename)
                jinx = Jinx(jinx_path=jinx_path)
                jinxs.append(jinx)
            except Exception as e:
                print(f"Error loading jinx {filename}: {e}")
                
    return jinxs

def get_npc_action_space(npc=None, team=None):
    """Get action space for NPC including memory CRUD and core capabilities"""
    actions = DEFAULT_ACTION_SPACE.copy()
    
    if npc:
        core_tools = [
            npc.think_step_by_step,
            npc.write_code
        ]
        
        if npc.command_history:
            core_tools.extend([
                npc.search_my_conversations,
                npc.search_my_memories,
                npc.create_memory,
                npc.read_memory,
                npc.update_memory,
                npc.delete_memory,
                npc.search_memories,
                npc.get_all_memories,
                npc.archive_old_memories,
                npc.get_memory_stats
            ])
        
        if npc.db_conn:
            core_tools.append(npc.query_database)
        
        if hasattr(npc, 'tools') and npc.tools:
            core_tools.extend([func for func in npc.tool_map.values() if callable(func)])
        
        if core_tools:
            tools_schema, tool_map = auto_tools(core_tools)
            actions.update({
                f"use_{tool.__name__}": {
                    "description": f"Use {tool.__name__} capability",
                    "handler": tool,
                    "context": lambda **_: f"Available as automated capability",
                    "output_keys": {"result": {"description": "Tool execution result", "type": "string"}}
                }
                for tool in core_tools
            })
    
    if team and hasattr(team, 'npcs') and len(team.npcs) > 1:
        available_npcs = [name for name in team.npcs.keys() if name != (npc.name if npc else None)]
        
        def team_aware_handler(command, extracted_data, **kwargs):
            if 'team' not in kwargs or kwargs['team'] is None:
                kwargs['team'] = team
            return agent_pass_handler(command, extracted_data, **kwargs)
        
        actions["pass_to_npc"] = {
            "description": "Pass request to another NPC - only when task requires their specific expertise",
            "handler": team_aware_handler,
            "context": lambda npc=npc, team=team, **_: (
                f"Available NPCs: {', '.join(available_npcs)}. "
                f"Only pass when you genuinely cannot complete the task."
            ),
            "output_keys": {
                "target_npc": {
                    "description": "Name of the NPC to pass the request to",
                    "type": "string"
                }
            }
        }
    
    return actions
def extract_jinx_inputs(args: List[str], jinx: Jinx) -> Dict[str, Any]:
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
        self.team = team
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
        
        self.memory_length = 20
        self.memory_strategy = 'recent'
        dirs = []
        if self.npc_directory:
            dirs.append(self.npc_directory)
        if self.jinxs_directory:
            dirs.append(self.jinxs_directory)
            
        self.jinja_env = Environment(
            loader=FileSystemLoader([
                os.path.expanduser(d) for d in dirs
            ]),
            undefined=SilentUndefined,
        )
        
        self.db_conn = db_conn

        # these 4 get overwritten if the db conn 
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


            
        self.jinxs = self._load_npc_jinxs(jinxs or "*")
        
        self.shared_context = {
            "dataframes": {},
            "current_data": None,
            "computation_results": [],
            "memories":{}
        }
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        if db_conn is not None:
            init_db_tables()

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

    def _load_npc_jinxs(self, jinxs):
        """Load and process NPC-specific jinxs"""
        npc_jinxs = []
        
        if self.jinxs_directory is None:
            self.jinxs_dict = {}
            return None
        
        if jinxs == "*":
            npc_jinxs.extend(load_jinxs_from_directory(self.jinxs_directory))
            
            if os.path.exists(self.jinxs_directory):
                npc_jinxs.extend(load_jinxs_from_directory(self.jinxs_directory))                
            
            self.jinxs_dict = {jinx.jinx_name: jinx for jinx in npc_jinxs}
            
            return npc_jinxs

        for jinx in jinxs:
            if isinstance(jinx, Jinx):
                npc_jinxs.append(jinx)
            elif isinstance(jinx, dict):
                npc_jinxs.append(Jinx(jinx_data=jinx))
            
                jinx_path = None
                jinx_name = jinx
                if not jinx_name.endswith(".jinx"):
                    jinx_name += ".jinx"
                
                if hasattr(self, 'jinxs_directory') and os.path.exists(self.jinxs_directory):
                    candidate_path = os.path.join(self.jinxs_directory, jinx_name)
                    if os.path.exists(candidate_path):
                        jinx_path = candidate_path
                        
                if jinx_path:
                    jinx_obj = Jinx(jinx_path=jinx_path)
                    npc_jinxs.append(jinx_obj)
        
        self.jinxs_dict = {jinx.jinx_name: jinx for jinx in npc_jinxs}
        return npc_jinxs
        
    def get_llm_response(self, 
                            request,
                            jinxs=None,
                            tools=None,
                            tool_map=None,
                            tool_choice=None, 
                            messages=None,
                            auto_process_tool_calls=True,
                            **kwargs):
        """Get response from LLM with automatic tool integration including memory CRUD"""
        
        if tools is None and tool_map is None and tool_choice is None:
            core_tools = [
                self.think_step_by_step,
                self.write_code
            ]
            
            if self.command_history:
                core_tools.extend([
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
                core_tools.append(self.query_database)
            
            if hasattr(self, 'tools') and self.tools:
                core_tools.extend([func for func in self.tool_map.values() if callable(func)])
            
            if core_tools:
                tools, tool_map = auto_tools(core_tools)
        
        if tool_choice is None and tools:
            tool_choice = "auto"

        response = npy.llm_funcs.get_llm_response(
            request, 
            model=self.model, 
            provider=self.provider, 
            npc=self, 
            jinxs=jinxs,
            tools=tools, 
            tool_map=tool_map,
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
        
        return f"Found {len(results)} conversations matching '{query}':\n" + "\n".join(formatted_results)

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

    def write_code(self, task_description: str, language: str = "python", show=True) -> str:
        """Generate and execute code for a specific task, returning the result"""
        if language.lower() != "python":
            
            code_prompt = f"""Write {language} code for the following task:
    {task_description}

    Provide clean, working code with brief explanations for key parts:"""
            
            response = self.get_llm_response(code_prompt, tool_choice=False )
            return response.get('response', 'Unable to generate code')
        
        
        code_prompt = f"""Write Python code for the following task:
    {task_description}

    Requirements:
    - Provide executable Python code
    - Store the final result in a variable called 'output'
    - Include any necessary imports
    - Handle errors gracefully
    - The code should be ready to execute without modification

    Example format:
    ```python
    import pandas as pd
    # Your code here
    result = some_calculation()
    output = f"Task completed successfully: {{result}}"
    """
        response = self.get_llm_response(code_prompt, tool_choice= False)
        generated_code = response.get('response', '')

        
        if '```python' in generated_code:
            code_lines = generated_code.split('\n')
            start_idx = None
            end_idx = None
    
            for i, line in enumerate(code_lines):
                if '```python' in line:
                    start_idx = i + 1
                elif '```' in line and start_idx is not None:
                    end_idx = i
                    break
        
            if start_idx is not None:
                if end_idx is not None:
                    generated_code = '\n'.join(code_lines[start_idx:end_idx])
                else:
                    generated_code = '\n'.join(code_lines[start_idx:])

        try:
            
            exec_globals = {
                "__builtins__": __builtins__,
                "npc": self,
                "context": self.shared_context,
                "pd": pd,
                "plt": plt,
                "np": np,
                "os": os,
                "re": re,
                "json": json,
                "Path": pathlib.Path,
                "fnmatch": fnmatch,
                "pathlib": pathlib,
                "subprocess": subprocess,
                "datetime": datetime,
                "hashlib": hashlib,
                "sqlite3": sqlite3,
                "yaml": yaml,
                "random": random,
                "math": math,
            }
            
            exec_locals = {}
            
            
            exec(generated_code, exec_globals, exec_locals)
            
            if show:
                print('Executing code', generated_code)
            
            
            if "output" in exec_locals:
                result = exec_locals["output"]
                
                self.shared_context.update({k: v for k, v in exec_locals.items() 
                                        if not k.startswith('_') and not callable(v)})
                return f"Code executed successfully. Result: {result}"
            else:
                
                meaningful_vars = {k: v for k, v in exec_locals.items() 
                                if not k.startswith('_') and not callable(v)}
                
                self.shared_context.update(meaningful_vars)
                
                if meaningful_vars:
                    last_var = list(meaningful_vars.items())[-1]
                    return f"Code executed successfully. Last result: {last_var[0]} = {last_var[1]}"
                else:
                    return "Code executed successfully (no explicit output generated)"
                    
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}\n\nGenerated code was:\n{generated_code}"
            return error_msg



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
    
    def execute_jinx(self, jinx_name, inputs, conversation_id=None, message_id=None, team_name=None):
        """Execute a jinx by name"""
        
        if jinx_name in self.jinxs_dict:
            jinx = self.jinxs_dict[jinx_name]
        elif jinx_name in self.jinxs_dict:
            jinx = self.jinxs_dict[jinx_name]
        else:
            return {"error": f"jinx '{jinx_name}' not found"}
        
        result = jinx.execute(
            input_values=inputs,
            context=self.shared_context,
            jinja_env=self.jinja_env,
            npc=self
        )
        if self.db_conn is not None:
            self.db_conn.add_jinx_call(
                triggering_message_id=message_id,
                conversation_id=conversation_id,
                jinx_name=jinx_name,
                jinx_inputs=inputs,
                jinx_output=result,
                status="success",
                error_message=None,
                duration_ms=None,
                npc_name=self.name,
                team_name=team_name,
            )
        return result

    def check_llm_command(self,
                            command, 
                            messages=None,
                            context=None,
                            team=None,
                            stream=False):
        """Check if a command is for the LLM"""
        if context is None:
            context = self.shared_context
        
        if team:
            self._current_team = team
        
        actions = get_npc_action_space(npc=self, team=team)
        
        return npy.llm_funcs.check_llm_command(
            command,
            model=self.model,
            provider=self.provider,
            npc=self,
            team=team,
            messages=self.memory if messages is None else messages,
            context=context,
            stream=stream,
            actions=actions  
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
        if self.jinxs is not None:
            jinx_rep = [ jinx.to_dict() if isinstance(jinx, Jinx) else jinx for jinx in self.jinxs]
        return {
            "name": self.name,
            "primary_directive": self.primary_directive,
            "model": self.model,
            "provider": self.provider,
            "api_url": self.api_url,
            "api_key": self.api_key,
            "jinxs": jinx_rep, 
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
        if self.jinxs:
            str_rep += "Jinxs:\n"
            for jinx in self.jinxs:
                str_rep += f"  - {jinx.jinx_name}\n"
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
            jinx.jinx_name,
            npc=self,
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
                    npcs=None, 
                    forenpc=None,
                    jinxs=None,                   
                    db_conn=None, 
                    model = None, 
                    provider = None):
        """
        Initialize an NPC team from directory or list of NPCs
        
        Args:
            team_path: Path to team directory
            npcs: List of NPC objects
            db_conn: Database connection
        """
        self.model = model
        self.provider = provider
        
        self.npcs = {}
        self.sub_teams = {}
        self.jinxs_dict = jinxs or {}
        self.db_conn = db_conn
        self.team_path = os.path.expanduser(team_path) if team_path else None
        self.databases = []
        self.mcp_servers = []
        if forenpc is not None:
            self.forenpc = forenpc
        else:
            self.forenpc  = npcs[0] if npcs else None
        
        if team_path:
            self.name = os.path.basename(os.path.abspath(team_path))
        else:
            self.name = "custom_team"
        self.context = ''
        self.shared_context = {
            "intermediate_results": {},
            "dataframes": {},
            "memories": {},          
            "execution_history": [],   
            "npc_messages": {},
            "context":''       
            }
                
        if team_path:
            self._load_from_directory()
            
        elif npcs:
            for npc in npcs:
                self.npcs[npc.name] = npc

        self.jinja_env = Environment(undefined=SilentUndefined)
        
        if db_conn is not None:
            init_db_tables()

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
            
    def _load_from_directory(self):
        """Load team from directory"""
        if not os.path.exists(self.team_path):
            raise ValueError(f"Team directory not found: {self.team_path}")
        
        for filename in os.listdir(self.team_path):
            if filename.endswith(".npc"):
                npc_path = os.path.join(self.team_path, filename)
                npc = NPC(npc_path, db_conn=self.db_conn)
                self.npcs[npc.name] = npc
                    
        self.context = self._load_team_context()
        
        jinxs_dir = os.path.join(self.team_path, "jinxs")
        if os.path.exists(jinxs_dir):
            for jinx in load_jinxs_from_directory(jinxs_dir):
                self.jinxs_dict[jinx.jinx_name] = jinx
        
        self._load_sub_teams()

    def _load_team_context(self):
        """Load team context from .ctx file"""
        for fname in os.listdir(self.team_path):
            if fname.endswith('.ctx'):
                ctx_data = load_yaml_file(os.path.join(self.team_path, fname))                
                if ctx_data is not None:
                    if 'model' in ctx_data:
                        self.model = ctx_data['model']
                    else:
                        self.model = None
                    if 'provider' in ctx_data:
                        self.provider = ctx_data['provider']
                    else:
                        self.provider = None
                    if 'api_url' in ctx_data:
                        self.api_url = ctx_data['api_url']
                    else:
                        self.api_url = None
                    if 'env' in ctx_data:
                        self.env = ctx_data['env']
                    else:
                        self.env = None
                        
                    if 'mcp_servers' in ctx_data:
                        self.mcp_servers = ctx_data['mcp_servers']
                    else:
                        self.mcp_servers = []
                    if 'databases' in ctx_data:
                        self.databases = ctx_data['databases']
                    else:
                        self.databases = []
                    
                    base_context = ctx_data.get('context', '')
                    self.shared_context['context'] = base_context
                    if 'file_patterns' in ctx_data:
                        file_cache = self._parse_file_patterns(ctx_data['file_patterns'])
                        self.shared_context['files'] = file_cache
                    if 'preferences' in ctx_data:
                        self.preferences = ctx_data['preferences']
                    else:
                        self.preferences = []
                    if 'forenpc' in ctx_data:
                        self.forenpc = self.npcs[ctx_data['forenpc']]
                    else:
                        self.forenpc = self.npcs[list(self.npcs.keys())[0]] if self.npcs else None
                    for key, item in ctx_data.items():
                        if key not in ['name', 'mcp_servers', 'databases', 'context', 'file_patterns']:
                            self.shared_context[key] = item
                return ctx_data
        return {}
        
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
        
    def get_forenpc(self):
        """
        Get the forenpc (coordinator) for this team.
        The forenpc is set only if explicitly defined in the context.
                
        """
        if isinstance(self.forenpc, NPC):
            return self.forenpc
        if hasattr(self, 'context') and self.context and 'forenpc' in self.context:
            forenpc_ref = self.context['forenpc']
            
            if '{{ref(' in forenpc_ref:
                match = re.search(r"{{\s*ref\('([^']+)'\)\s*}}", forenpc_ref)
                if match:
                    forenpc_name = match.group(1)
                    if forenpc_name in self.npcs:
                        return self.npcs[forenpc_name]
            elif forenpc_ref in self.npcs:
                return self.npcs[forenpc_ref]
        else:
            forenpc_model=self.context.get('model', 'llama3.2'),
            forenpc_provider=self.context.get('provider', 'ollama'),
            forenpc_api_key=self.context.get('api_key', None),
            forenpc_api_url=self.context.get('api_url', None)
            
            forenpc = NPC(name='forenpc', 
                            primary_directive="""You are the forenpc of the team, coordinating activities 
                                                between NPCs on the team, verifying that results from 
                                                NPCs are high quality and can help to adequately answer 
                                                user requests.""", 
                            model=forenpc_model,
                            provider=forenpc_provider,
                            api_key=forenpc_api_key,
                            api_url=forenpc_api_url,                            
                                                )
            self.forenpc = forenpc
            self.npcs[forenpc.name] = forenpc
            return forenpc
        return None

    def get_npc(self, npc_ref):
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

    def orchestrate(self, request):
        """Orchestrate a request through the team"""
        forenpc = self.get_forenpc()
        if not forenpc:
            return {"error": "No forenpc available to coordinate the team"}
        
        log_entry(
            self.name,
            "orchestration_start",
            {"request": request}
        )
        
        result = forenpc.check_llm_command(request,
            context=getattr(self, 'context', {}),
            team = self, 
        )
        
        while True:
            completion_prompt= "" 
            if isinstance(result, dict):
                self.shared_context["execution_history"].append(result)
                
                if result.get("messages") and result.get("npc_name"):
                    if result["npc_name"] not in self.shared_context["npc_messages"]:
                        self.shared_context["npc_messages"][result["npc_name"]] = []
                    self.shared_context["npc_messages"][result["npc_name"]].extend(
                        result["messages"]
                    )
                
                completion_prompt += f"""Context:
                    User request '{request}', previous agent
                    
                    previous agent returned:
                    {result.get('output')}

                    
                Instructions:

                    Check whether the response is relevant to the user's request.

                """
                if self.npcs is None or len(self.npcs) == 0:
                    completion_prompt += f"""
                    The team has no members, so the forenpc must handle the request alone.
                    """
                else:
                    completion_prompt += f"""
                    
                    These are all the members of the team: {', '.join(self.npcs.keys())}

                    Therefore, if you are trying to evaluate whether a request was fulfilled relevantly,
                    consider that requests are made to the forenpc: {forenpc.name}
                    and that the forenpc must pass those along to the other npcs. 
                    """
                completion_prompt += f"""

                Mainly concern yourself with ensuring there are no
                glaring errors nor fundamental mishaps in the response.
                Do not consider stylistic hiccups as the answers being
                irrelevant. By providing responses back to for the user to
                comment on, they can can more efficiently iterate and resolve any issues by 
                prompting more clearly.
                natural language itself is very fuzzy so there will always be some level
                of misunderstanding, but as long as the response is clearly relevant 
                to the input request and along the user's intended direction,
                it is considered relevant.
                                

                If there is enough information to begin a fruitful conversation with the user, 
                please consider the request relevant so that we do not
                arbritarily stall business logic which is more efficiently
                determined by iterations than through unnecessary pedantry.

                It is more important to get a response to the user
                than to account for all edge cases, so as long as the response more or less tackles the
                initial problem to first order, consider it relevant.

                Return a JSON object with:
                    -'relevant' with boolean value
                    -'explanation' for irrelevance with quoted citations in your explanation noting why it is irrelevant to user input must be a single string.
                Return only the JSON object."""
            
            completion_check = npy.llm_funcs.get_llm_response(
                completion_prompt, 
                model=forenpc.model,
                provider=forenpc.provider,
                api_key=forenpc.api_key,
                api_url=forenpc.api_url,
                npc=forenpc,
                format="json"
            )
            
            if isinstance(completion_check.get("response"), dict):
                complete = completion_check["response"].get("relevant", False)
                explanation = completion_check["response"].get("explanation", "")
            else:
                complete = False
                explanation = "Could not determine completion status"
            
            if complete:
                debrief = npy.llm_funcs.get_llm_response(
                    f"""Context:
                    Original request: {request}
                    Execution history: {self.shared_context['execution_history']}

                    Instructions:
                    Provide summary of actions taken and recommendations.
                    Return a JSON object with:
                    - 'summary': Overview of what was accomplished
                    - 'recommendations': Suggested next steps
                    Return only the JSON object.""",
                    model=forenpc.model,
                    provider=forenpc.provider,
                    api_key=forenpc.api_key,
                    api_url=forenpc.api_url,
                    npc=forenpc,
                    format="json"
                )
                
                return {
                    "debrief": debrief.get("response"),
                    "output": result.get("output"),
                    "execution_history": self.shared_context["execution_history"],
                }
            else:
                updated_request = (
                    request
                    + "\n\nThe request has not yet been fully completed. "
                    + explanation
                    + "\nPlease address only the remaining parts of the request."
                )
                print('updating request', updated_request)
                
                result = forenpc.check_llm_command(
                    updated_request,
                    context=getattr(self, 'context', {}),
                    stream = False,
                    team = self
                    
                )
                
    def to_dict(self):
        """Convert team to dictionary representation"""
        return {
            "name": self.name,
            "npcs": {name: npc.to_dict() for name, npc in self.npcs.items()},
            "sub_teams": {name: team.to_dict() for name, team in self.sub_teams.items()},
            "jinxs": {name: jinx.to_dict() for name, jinx in self.jinxs.items()},
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
        
        for jinx in self.jinxs.values():
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

class Pipeline:
    def __init__(self, pipeline_data=None, pipeline_path=None, npc_team=None):
        """Initialize a pipeline from data or file path"""
        self.npc_team = npc_team
        self.steps = []
        
        if pipeline_path:
            self._load_from_path(pipeline_path)
        elif pipeline_data:
            self.name = pipeline_data.get("name", "unnamed_pipeline")
            self.steps = pipeline_data.get("steps", [])
        else:
            raise ValueError("Either pipeline_data or pipeline_path must be provided")
            
    def _load_from_path(self, path):
        """Load pipeline from file"""
        pipeline_data = load_yaml_file(path)
        if not pipeline_data:
            raise ValueError(f"Failed to load pipeline from {path}")
            
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.steps = pipeline_data.get("steps", [])
        self.pipeline_path = path
        
    def execute(self, initial_context=None):
        """Execute the pipeline with given context"""
        context = initial_context or {}
        results = {}
        
        
        init_db_tables()
        
        
        pipeline_hash = self._generate_hash()
        
        
        results_table = f"{self.name}_results"
        self._ensure_results_table(results_table)
        
        
        run_id = self._create_run_entry(pipeline_hash)
        
        
        context.update({
            "ref": lambda step_name: results.get(step_name),
            "source": self._fetch_data_from_source,
        })
        
        
        for step in self.steps:
            step_name = step.get("step_name")
            if not step_name:
                raise ValueError(f"Missing step_name in step: {step}")
                
            
            npc_name = self._render_template(step.get("npc", ""), context)
            npc = self._get_npc(npc_name)
            if not npc:
                raise ValueError(f"NPC {npc_name} not found for step {step_name}")
                
            
            task = self._render_template(step.get("task", ""), context)
            
            
            model = step.get("model", npc.model)
            provider = step.get("provider", npc.provider)
            
            
            mixa = step.get("mixa", False)
            if mixa:
                response = self._execute_mixa_step(step, context, npc, model, provider)
            else:
                
                source_matches = re.findall(r"{{\s*source\('([^']+)'\)\s*}}", task)
                if source_matches:
                    response = self._execute_data_source_step(step, context, source_matches, npc, model, provider)
                else:
                    
                    llm_response = npy.llm_funcs.get_llm_response(task, model=model, provider=provider, npc=npc)
                    response = llm_response.get("response", "")
            
            
            results[step_name] = response
            context[step_name] = response
            
            
            self._store_step_result(run_id, step_name, npc_name, model, provider, 
                                   {"task": task}, response, results_table)
            
        
        return {
            "results": results,
            "run_id": run_id
        }
        
    def _render_template(self, template_str, context):
        """Render a template with the given context"""
        if not template_str:
            return ""
            
        try:
            template = Template(template_str)
            return template.render(**context)
        except Exception as e:
            print(f"Error rendering template: {e}")
            return template_str
            
    def _get_npc(self, npc_name):
        """Get NPC by name from team"""
        if not self.npc_team:
            raise ValueError("No NPC team available")
            
        return self.npc_team.get_npc(npc_name)
        
    def _generate_hash(self):
        """Generate a hash for the pipeline"""
        if hasattr(self, 'pipeline_path') and self.pipeline_path:
            with open(self.pipeline_path, 'r') as f:
                content = f.read()
            return hashlib.sha256(content.encode()).hexdigest()
        else:
            
            content = json.dumps(self.steps)
            return hashlib.sha256(content.encode()).hexdigest()
            
    def _ensure_results_table(self, table_name):
        """Ensure results table exists"""
        db_path = "~/npcsh_history.db"
        with sqlite3.connect(os.path.expanduser(db_path)) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    step_name TEXT,
                    npc_name TEXT,
                    model TEXT,
                    provider TEXT,
                    inputs TEXT,
                    outputs TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
                )
            """)
            conn.commit()
            
    def _create_run_entry(self, pipeline_hash):
        """Create run entry in pipeline_runs table"""
        db_path = "~/npcsh_history.db"
        with sqlite3.connect(os.path.expanduser(db_path)) as conn:
            cursor = conn.execute(
                "INSERT INTO pipeline_runs (pipeline_name, pipeline_hash, timestamp) VALUES (?, ?, ?)",
                (self.name, pipeline_hash, datetime.now())
            )
            conn.commit()
            return cursor.lastrowid
            
    def _store_step_result(self, run_id, step_name, npc_name, model, provider, inputs, outputs, table_name):
        """Store step result in database"""
        db_path = "~/npcsh_history.db"
        with sqlite3.connect(os.path.expanduser(db_path)) as conn:
            conn.execute(
                f"""
                INSERT INTO {table_name} 
                (run_id, step_name, npc_name, model, provider, inputs, outputs)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step_name,
                    npc_name,
                    model,
                    provider,
                    json.dumps(self._clean_for_json(inputs)),
                    json.dumps(self._clean_for_json(outputs))
                )
            )
            conn.commit()
            
    def _clean_for_json(self, obj):
        """Clean an object for JSON serialization"""
        if isinstance(obj, dict):
            return {
                k: self._clean_for_json(v)
                for k, v in obj.items()
                if not k.startswith("_") and not callable(v)
            }
        elif isinstance(obj, list):
            return [self._clean_for_json(i) for i in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
            
    def _fetch_data_from_source(self, table_name):
        """Fetch data from a database table"""
        db_path = "~/npcsh_history.db"
        try:
            engine = create_engine(f"sqlite:///{os.path.expanduser(db_path)}")
            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
            return df.to_json(orient="records")
        except Exception as e:
            print(f"Error fetching data from {table_name}: {e}")
            return "[]"
            
    def _execute_mixa_step(self, step, context, npc, model, provider):
        """Execute a mixture of agents step"""
        
        task = self._render_template(step.get("task", ""), context)
        
        
        mixa_turns = step.get("mixa_turns", 5)
        num_generating_agents = len(step.get("mixa_agents", []))
        if num_generating_agents == 0:
            num_generating_agents = 3  
            
        num_voting_agents = len(step.get("mixa_voters", []))
        if num_voting_agents == 0:
            num_voting_agents = 3  
            
        
        round_responses = []
        
        
        return 
        
    def _execute_data_source_step(self, step, context, source_matches, npc, model, provider):
        """Execute a step with data source"""
        task_template = step.get("task", "")
        table_name = source_matches[0]
        
        try:
            
            db_path = "~/npcsh_history.db"
            engine = create_engine(f"sqlite:///{os.path.expanduser(db_path)}")
            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
            
            
            if step.get("batch_mode", False):
                
                data_str = df.to_json(orient="records")
                task = task_template.replace(f"{{{{ source('{table_name}') }}}}", data_str)
                task = self._render_template(task, context)
                
                
                response = npy.llm_funcs.get_llm_response(task, model=model, provider=provider, npc=npc)
                return response.get("response", "")
            else:
                
                results = []
                for _, row in df.iterrows():
                    
                    row_data = json.dumps(row.to_dict())
                    row_task = task_template.replace(f"{{{{ source('{table_name}') }}}}", row_data)
                    row_task = self._render_template(row_task, context)
                    
                    
                    response = npy.llm_funcs.get_llm_response(row_task, model=model, provider=provider, npc=npc)
                    results.append(response.get("response", ""))
                    
                return results
        except Exception as e:
            print(f"Error processing data source {table_name}: {e}")
            return f"Error: {str(e)}"


