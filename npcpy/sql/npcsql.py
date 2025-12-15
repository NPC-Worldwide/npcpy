import pandas as pd
import re
import os
from pathlib import Path
from typing import Dict, List, Set, Union, Any, Optional, Callable
from collections import defaultdict, deque
from sqlalchemy import create_engine, text, Engine, inspect
import inspect as py_inspect

try:
    from jinja2 import Environment, BaseLoader, DebugUndefined
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False

# --- Explicitly import llm_funcs as a module object ---
try:
    import npcpy.llm_funcs as llm_funcs
except ImportError:
    print("Warning: `npcpy.llm_funcs` not found. Providing mock AI functions for execution.")
    class MockLlmFuncs:
        def generate_text(self, prompt: str, npc=None, team=None, context="") -> Dict[str, str]:
            print(f"MOCK AI: generate_text('{prompt}')")
            return {"response": f"MOCK: Generated text for '{prompt}'"}
        def analyze_sentiment(self, text: str, npc=None, team=None, context="") -> Dict[str, str]:
            print(f"MOCK AI: analyze_sentiment('{text}')")
            return {"response": f"MOCK: Positive sentiment for '{text}'"}
        def summarize(self, text: str, npc=None, team=None, context="") -> Dict[str, str]:
            print(f"MOCK AI: summarize('{text}')")
            return {"response": f"MOCK: Summary of '{text}'"}
        def translate(self, text: str, source_lang='auto', target_lang='en', npc=None, team=None, context="") -> Dict[str, str]:
            print(f"MOCK AI: translate('{text}', '{source_lang}', '{target_lang}')")
            return {"response": f"MOCK: Translated '{text}' from {source_lang} to {target_lang}"}
        def extract_entities(self, text: str, npc=None, team=None, context="") -> Dict[str, str]:
            print(f"MOCK AI: extract_entities('{text}')")
            return {"response": f"MOCK: Entities from '{text}'"}
        def generate_embedding(self, text: str, model='default', npc=None, team=None, context="") -> Dict[str, str]:
            print(f"MOCK AI: generate_embedding('{text}', '{model}')")
            return {"response": f"MOCK: Embedding for '{text}'"}
    llm_funcs = MockLlmFuncs()

# Assuming these are available in the npcpy environment
from npcpy.memory.command_history import create_engine_from_path
try:
    from npcpy.npc_compiler import Team
except ImportError:
    print("Warning: `npcpy.npc_compiler.Team` not found. Providing mock Team class.")
    class Team:
        def __init__(self, team_path: str = "./npc_team/", npcs: Optional[List[Any]] = None):
            print(f"MOCK NPC: Team initialized for path: {team_path}")
            self.npcs = npcs if npcs is not None else []
        def get_npc(self, npc_ref: str):
            print(f"MOCK NPC: get_npc called for: {npc_ref}")
            return {"name": npc_ref, "type": "mock_npc"}


# --- PANDAS BACKEND CONFIGURATION ---
try:
    import modin.pandas as pd_modin
    import snowflake.snowpark.modin.plugin
    pd = pd_modin
    PANDAS_BACKEND = 'snowflake_modin'
except ImportError:
    try:
        import modin.pandas as pd_modin
        pd = pd_modin
        PANDAS_BACKEND = 'modin'
    except ImportError:
        import pandas as pd
        PANDAS_BACKEND = 'pandas'
# print(f"Using pandas backend: {PANDAS_BACKEND}") # Removed for cleaner output


# --- AI Function Mappings ---
class DatabaseAIFunctionMapper:
    @staticmethod
    def get_snowflake_cortex_mapping() -> Dict[str, Dict[str, Any]]:
        return {
            'get_llm_response': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda prompt, **kwargs: f"SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b', {prompt})"
            },

            
            'get_facts': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b', 
                    'Extract facts from this text. Return JSON with facts array containing statement, source_text, and type fields. Text: ' || {text})"""
            },
            'identify_groups': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Identify main groups these facts could be organized into. Return JSON with groups array. Facts: ' || {text})"""
            },
            'assign_groups_to_fact': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Assign this fact to relevant groups. Return JSON with groups array. Fact: ' || {text})"""
            },
            'generate_group_candidates': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Generate specific conceptual groups for these items. Return JSON with groups array. Items: ' || {text})"""
            },
            'remove_idempotent_groups': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Remove conceptually identical groups, favor specificity. Return JSON with distinct_groups array. Groups: ' || {text})"""
            },
            'zoom_in': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Infer new implied facts from existing facts. Return JSON with implied_facts array. Facts: ' || {text})"""
            },
            'generate_groups': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Generate conceptual groups for facts. Return JSON with groups array. Facts: ' || {text})"""
            },
            'remove_redundant_groups': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Remove redundant groups, merge similar concepts. Return JSON with groups array. Groups: ' || {text})"""
            },
            'criticize': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Provide critical analysis and constructive criticism. Input: ' || {text})"""
            },
            'synthesize': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Synthesize information from multiple perspectives. Input: ' || {text})"""
            },
            'breathe': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Condense conversation context into key extractions. Return JSON with high_level_objective, most_recent_task, accomplishments, failures. Conversation: ' || {text})"""
            },
            'abstract': {
                'cortex_function': 'COMPLETE',
                'transformer': lambda text, **kwargs: f"""SNOWFLAKE.CORTEX.COMPLETE('llama3.1-8b',
                    'Create more abstract categories from groups. Return JSON with groups array. Groups: ' || {text})"""
            }
        }

    
    @staticmethod
    def get_databricks_ai_mapping() -> Dict[str, Dict[str, Any]]:
        return {
            'generate_text': {
                'databricks_function': 'serving.predict',
                'transformer': lambda prompt, model='databricks-dolly', **kwargs: 
                    f"serving.predict('{model}', '{prompt}')"
            },
        }
    
    @staticmethod
    def get_bigquery_ai_mapping() -> Dict[str, Dict[str, Any]]:
        return {
            'generate_text': {
                'bigquery_function': 'ML.GENERATE_TEXT',
                'transformer': lambda prompt, model='text-bison', **kwargs:
                    f"ML.GENERATE_TEXT(MODEL `{model}`, '{prompt}')"
            },
        }

# --- Native Database AI Transformer (INCLUDED in the module) ---
class NativeDatabaseAITransformer:
    def __init__(self, database_type: str):
        self.database_type = database_type.lower()
        self.function_mappings = self._get_database_mappings()
    
    def _get_database_mappings(self) -> Dict[str, Dict[str, Any]]:
        mappings = {
            'snowflake': DatabaseAIFunctionMapper.get_snowflake_cortex_mapping(),
            'databricks': DatabaseAIFunctionMapper.get_databricks_ai_mapping(),
            'bigquery': DatabaseAIFunctionMapper.get_bigquery_ai_mapping()
        }
        return mappings.get(self.database_type, {})
    
    def transform_ai_function(self, function_name: str, **kwargs) -> str:
        mapping = self.function_mappings.get(function_name)
        if not mapping:
            raise ValueError(f"No native mapping found for function: {function_name} for database type {self.database_type}")
        
        transformer: Callable[..., str] = mapping.get('transformer')
        if not transformer:
            raise ValueError(f"No transformer found for function: {function_name} for database type {self.database_type}")
        
        if function_name == 'generate_text' and 'text' in kwargs:
            kwargs['prompt'] = kwargs.pop('text')
        
        return transformer(**kwargs)


# --- NQL Jinja Context ---
class NQLJinjaContext:
    """Provides Jinja template context for NQL models with access to NPCs, jinxs, and team."""

    def __init__(self, team=None, npc_operations=None):
        self.team = team
        self.npc_operations = npc_operations
        self._npc_cache = {}
        self._jinx_cache = {}

    def npc(self, name: str) -> dict:
        """Get NPC properties by name. Usage: {{ npc('sibiji').model }}"""
        if name in self._npc_cache:
            return self._npc_cache[name]

        if not self.team:
            return {'name': name, 'error': 'No team loaded'}

        npc_obj = self.team.get_npc(name)
        if not npc_obj:
            return {'name': name, 'error': f'NPC {name} not found'}

        # Build properties dict
        props = {
            'name': getattr(npc_obj, 'name', name),
            'model': getattr(npc_obj, 'model', 'gpt-4o-mini'),
            'provider': getattr(npc_obj, 'provider', 'openai'),
            'directive': getattr(npc_obj, 'primary_directive', ''),
            'jinxs': getattr(npc_obj, 'jinxs', []),
        }
        self._npc_cache[name] = props
        return props

    def jinx(self, name: str) -> dict:
        """Get jinx properties by name. Usage: {{ jinx('sample').description }}"""
        if name in self._jinx_cache:
            return self._jinx_cache[name]

        if not self.npc_operations or not self.npc_operations.jinx_map:
            return {'name': name, 'error': 'No jinxs loaded'}

        jinx_info = self.npc_operations.jinx_map.get(name.lower())
        if not jinx_info:
            return {'name': name, 'error': f'Jinx {name} not found'}

        props = {
            'name': jinx_info.get('name', name),
            'description': jinx_info.get('description', ''),
            'inputs': jinx_info.get('inputs', []),
        }
        self._jinx_cache[name] = props
        return props

    def get_team_context(self) -> dict:
        """Get team-level properties. Usage: {{ team.forenpc }}"""
        if not self.team:
            return {'error': 'No team loaded'}

        return {
            'name': getattr(self.team, 'name', 'npc_team'),
            'forenpc': getattr(self.team, 'forenpc_name', None),
            'npcs': [getattr(n, 'name', str(n)) for n in getattr(self.team, 'npcs', [])],
            'jinx_count': len(self.npc_operations.jinx_map) if self.npc_operations else 0,
        }

    def ref(self, model_name: str) -> str:
        """Reference another model. Usage: {{ ref('base_stats') }}"""
        return f"{{{{ ref('{model_name}') }}}}"

    def config(self, **kwargs) -> str:
        """Model configuration. Usage: {{ config(materialized='table') }}"""
        config_parts = [f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"
                       for k, v in kwargs.items()]
        return f"{{{{ config({', '.join(config_parts)}) }}}}"

    def env(self, var_name: str, default: str = '') -> str:
        """Get environment variable. Usage: {{ env('API_KEY') }}"""
        return os.environ.get(var_name, default)

    def build_jinja_env(self) -> 'Environment':
        """Build Jinja2 environment with NQL functions."""
        if not JINJA_AVAILABLE:
            raise ImportError("Jinja2 is required for template processing. Install with: pip install jinja2")

        env = Environment(
            loader=BaseLoader(),
            undefined=DebugUndefined,
            # Keep {{ ref(...) }} and {{ config(...) }} unprocessed for later
            variable_start_string='{%',
            variable_end_string='%}',
            block_start_string='{%%',
            block_end_string='%%}',
        )

        # Add custom functions
        env.globals['npc'] = self.npc
        env.globals['jinx'] = self.jinx
        env.globals['team'] = self.get_team_context()
        env.globals['env'] = self.env

        return env

    def render_template(self, content: str) -> str:
        """Render Jinja template in SQL content."""
        if not JINJA_AVAILABLE:
            return content

        # Only process if there are NQL Jinja expressions ({% ... %})
        if '{%' not in content:
            return content

        try:
            env = self.build_jinja_env()
            template = env.from_string(content)
            return template.render()
        except Exception as e:
            print(f"Warning: Jinja template error: {e}")
            return content


# --- NPCSQL Operations ---
class NPCSQLOperations:
    def __init__(
        self,
        npc_directory: str,
        db_engine: Union[str, Engine] = "~/npcsh_history.db"
    ):
        self.npc_directory = npc_directory

        if isinstance(db_engine, str):
            self.engine = create_engine_from_path(db_engine)
        else:
            self.engine = db_engine

        self.npc_loader = None
        self.jinx_map = {}  # Maps jinx names to jinx objects
        self.function_map = self._build_function_map()

    def _get_team(self):
        return (self.npc_loader
                if hasattr(self.npc_loader, 'npcs')
                else None)

    def _build_function_map(self):
        import types

        function_map = {}
        for name in dir(llm_funcs):
            if name.startswith('_'):
                continue
            obj = getattr(llm_funcs, name)
            if (isinstance(obj, types.FunctionType) or
                (isinstance(obj, types.MethodType) and obj.__self__ is not None)):
                function_map[name] = obj

        return function_map

    def load_team_jinxs(self, team):
        """Load jinxs from team to make them available as NQL functions."""
        if not team:
            return

        try:
            # Get all jinxs from the team's jinx catalog
            if hasattr(team, 'jinx_tool_catalog'):
                for tool in team.jinx_tool_catalog:
                    jinx_name = tool.get('name', '').lower()
                    if jinx_name and jinx_name not in self.function_map:
                        # Store reference to the jinx
                        self.jinx_map[jinx_name] = tool
                        # Add a placeholder to function_map so it's recognized
                        self.function_map[jinx_name] = f"__jinx__{jinx_name}"
                        print(f"NQL: Registered team jinx '{jinx_name}' as NQL function")
        except Exception as e:
            print(f"Warning: Could not load team jinxs: {e}")

    def _resolve_npc_reference(self, npc_ref: str):
        if not npc_ref or not self.npc_loader:
            return None
            
        if npc_ref.endswith('.npc'):
            npc_ref = npc_ref[:-4]
        
        npc = self.npc_loader.get_npc(npc_ref)
        if npc:
            return npc
            
        if ',' in npc_ref:
            npc_names = [
                name.strip() for name in npc_ref.split(',')
            ]
            npcs = [
                self.npc_loader.get_npc(name) 
                for name in npc_names
            ]
            npcs = [npc for npc in npcs if npc is not None]
            
            if npcs:
                temp_team = Team(npcs=npcs)
                return temp_team
                
        return None
        
    def _execute_jinx(self, jinx_name: str, query: str, npc_ref: str, context: str = "") -> str:
        """Execute a team jinx and return the result."""
        try:
            from npcpy.npc_compiler import execute_jinx

            jinx_info = self.jinx_map.get(jinx_name)
            if not jinx_info:
                return f"Error: Jinx '{jinx_name}' not found"

            # Build context for jinx execution
            jinx_context = {
                'input': query,
                'prompt': query,
                'text': query,
                'context': context,
            }

            # Get the jinx object from team
            team = self._get_team()
            if team and hasattr(team, 'get_jinx'):
                jinx = team.get_jinx(jinx_name)
                if jinx:
                    result = execute_jinx(jinx, jinx_context, team=team)
                    if isinstance(result, dict):
                        return result.get('output', str(result))
                    return str(result)

            return f"Error: Could not execute jinx '{jinx_name}'"
        except Exception as e:
            return f"Jinx error: {e}"

    def execute_ai_function(
        self,
        func_name: str,
        df: pd.DataFrame,
        **params
    ) -> pd.Series:
        if func_name not in self.function_map:
            raise ValueError(f"Unknown AI function: {func_name}")

        func = self.function_map[func_name]
        is_jinx = isinstance(func, str) and func.startswith("__jinx__")

        npc_ref = params.get('npc', '')
        resolved_npc = self._resolve_npc_reference(npc_ref)

        resolved_team = self._get_team()
        if not resolved_team and hasattr(resolved_npc, 'team'):
            resolved_team = resolved_npc.team

        total_rows = len(df)
        func_type = "jinx" if is_jinx else "function"
        print(f"NQL: Executing {func_type} '{func_name}' on {total_rows} rows with NPC '{npc_ref}'...")

        results = []
        for idx, (row_idx, row) in enumerate(df.iterrows()):
            query_template = params.get('query', '')
            column_name = params.get('column', '')

            column_value = str(row[column_name]) if column_name and column_name in row.index else column_name

            if query_template:
                row_data = {
                    col: str(row[col])
                    for col in df.columns
                }
                row_data['column_value'] = column_value
                query = query_template.format(**row_data)
            else:
                query = column_value

            print(f"  [{idx+1}/{total_rows}] Processing row {row_idx}...", end=" ", flush=True)

            try:
                if is_jinx:
                    # Execute as jinx
                    result_value = self._execute_jinx(
                        func_name,
                        query,
                        npc_ref,
                        params.get('context', '')
                    )
                else:
                    # Execute as llm_func
                    sig = py_inspect.signature(func)

                    # Extract model/provider from NPC if available
                    npc_model = None
                    npc_provider = None
                    if resolved_npc and hasattr(resolved_npc, 'model'):
                        npc_model = resolved_npc.model
                    if resolved_npc and hasattr(resolved_npc, 'provider'):
                        npc_provider = resolved_npc.provider

                    func_params = {
                        k: v for k, v in {
                            'prompt': query,
                            'text': query,
                            'npc': resolved_npc,
                            'team': resolved_team,
                            'context': params.get('context', ''),
                            'model': npc_model or 'gpt-4o-mini',
                            'provider': npc_provider or 'openai'
                        }.items() if k in sig.parameters
                    }

                    result = func(**func_params)
                    result_value = (result.get("response", "")
                            if isinstance(result, dict)
                            else str(result))

                print(f"OK ({len(str(result_value))} chars)")
            except Exception as e:
                print(f"ERROR: {e}")
                result_value = None

            results.append(result_value)

        print(f"NQL: Completed {func_name} on {total_rows} rows.")
        return pd.Series(results, index=df.index)
    

# --- SQL Model Definition ---
class SQLModel:
    def __init__(
        self,
        name: str,
        content: str,
        path: str,
        npc_directory: str,
        additional_functions: Optional[List[str]] = None
    ):
        self.name = name
        self.content = content
        self.path = path
        self.npc_directory = npc_directory
        self.additional_functions = additional_functions or []

        config_match = re.search(
            r'\{\{[\s]*config\((.*?)\)[\s]*\}\}',
            content,
            re.DOTALL
        )
        if config_match:
            self.config = self._parse_config(config_match.group(1))
        else:
            self.config = {'materialized': 'table'}

        self.dependencies = self._extract_dependencies()
        self.has_ai_function = self._check_ai_functions()

        # DEBUG print to confirm if AI functions are found
        self.ai_functions = self._extract_ai_functions()
        if self.ai_functions:
            print(f"DEBUG SQLModel: Model '{self.name}' extracted AI functions: {list(self.ai_functions.keys())}")
        else:
            print(f"DEBUG SQLModel: Model '{self.name}' has no AI functions found by _extract_ai_functions.")


    def _parse_config(self, config_str: str) -> Dict:
        config = {}
        for item in re.split(r',\s*(?=[a-zA-Z0-9_]+\s*=)', config_str): 
            if '=' in item:
                key, value = item.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'") 
                config[key] = value
        return config

    def _extract_dependencies(self) -> Set[str]:
        pattern = r"\{\{\s*ref\(['\"]([^'\"]+)['\"]\)\s*\}\}"
        return set(re.findall(pattern, self.content))
    
    def _check_ai_functions(self) -> bool:
        return "nql." in self.content

    def _extract_ai_functions(self) -> Dict[str, Dict]:
        """Extract AI function calls from SQL content with improved robustness."""
        import types

        ai_functions = {}
        # Pattern that captures: nql.function_name(args...) as alias
        pattern = r"nql\.(\w+)\s*\(((?:[^()]|\([^()]*\))*)\)(\s+as\s+(\w+))?"
        
        matches = re.finditer(pattern, self.content, flags=re.DOTALL | re.IGNORECASE)

        available_functions = []
        for name in dir(llm_funcs):
            if name.startswith('_'):
                continue
            obj = getattr(llm_funcs, name)
            if (isinstance(obj, types.FunctionType) or
                (isinstance(obj, types.MethodType) and obj.__self__ is not None)):
                available_functions.append(name.lower())  # Store as lowercase for comparison

        # Add any additional functions (e.g., team jinxs)
        for fn in self.additional_functions:
            fn_lower = fn.lower()
            if fn_lower not in available_functions:
                available_functions.append(fn_lower)
        
        for match in matches:
            full_call_string = match.group(0).strip()
            func_name = match.group(1).lower()  # Convert to lowercase for lookup
            
            if func_name in available_functions:
                params_str = match.group(2)
                
                # Simplified parameter extraction
                params_list = []
                balance = 0
                in_quote = None
                current_param_chars = []
                
                for char in params_str:
                    if char in ("'", '"'):
                        if in_quote == char:
                            in_quote = None
                        elif in_quote is None:
                            in_quote = char
                        current_param_chars.append(char)
                    elif char == '(' and in_quote is None:
                        balance += 1
                        current_param_chars.append(char)
                    elif char == ')' and in_quote is None:
                        balance -= 1
                        current_param_chars.append(char)
                    elif char == ',' and balance == 0 and in_quote is None:
                        params_list.append("".join(current_param_chars).strip())
                        current_param_chars = []
                    else:
                        current_param_chars.append(char)
                
                if current_param_chars:
                    params_list.append("".join(current_param_chars).strip())
            
                params = [p.strip().strip("'\"") for p in params_list]

                column_param = params[0] if len(params) > 0 else ""
                npc_param = params[1] if len(params) > 1 else ""
                query_param = params[2] if len(params) > 2 else ""
                context_param = params[3] if len(params) > 3 else None

                if npc_param.endswith(".npc"):
                    npc_param = npc_param[:-4]
                if self.npc_directory and npc_param.startswith(self.npc_directory):
                    npc_param = npc_param[len(self.npc_directory):].strip('/')

                # Extract alias if present (group 4 from the pattern)
                alias = match.group(4) if match.lastindex >= 4 and match.group(4) else f"{func_name}_result"

                ai_functions[func_name] = {
                    "column": column_param,
                    "npc": npc_param,
                    "query": query_param,
                    "context": context_param,
                    "full_call_string": full_call_string,
                    "original_func_name": match.group(1),  # Store original case
                    "alias": alias
                }
            else:
                print(f"DEBUG SQLModel: Function '{func_name}' not found in available LLM funcs ({available_functions}). Skipping this NQL call.")

        return ai_functions

# --- Model Compiler ---
class ModelCompiler:
    def __init__(
        self, 
        models_dir: str, 
        target_engine: Union[str, Engine],
        npc_directory: str = "./npc_team/", 
        external_engines: Optional[Dict[str, Engine]] = None,
        target_schema: Optional[str] = None
    ):
        self.models_dir = Path(os.path.expanduser(models_dir))
        
        if isinstance(target_engine, str):
            self.target_engine = create_engine_from_path(
                target_engine
            )
        else:
            self.target_engine = target_engine
            
        self.external_engines = external_engines or {}
        self.target_schema = target_schema
        self.models: Dict[str, SQLModel] = {}
        self.npc_operations = NPCSQLOperations(
            npc_directory, 
            self.target_engine
        )
        self.npc_directory = npc_directory
        
        try:
            self.npc_team = Team(team_path=npc_directory)
            self.npc_operations.npc_loader = self.npc_team
            # Load team jinxs as NQL functions
            self.npc_operations.load_team_jinxs(self.npc_team)
        except Exception as e:
            self.npc_team = None
            print(f"Warning: Could not load NPC team from {npc_directory}. AI functions relying on NPC context might fail: {e}")

        # Initialize Jinja context for template processing
        self.jinja_context = NQLJinjaContext(
            team=self.npc_team,
            npc_operations=self.npc_operations
        )
            
    def _get_engine(self, source_name: str) -> Engine:
        if source_name.lower() == 'local' or not self.external_engines:
            return self.target_engine
        
        for key, engine in self.external_engines.items():
            if key.lower() == source_name.lower():
                return engine
        return self.target_engine

    def _has_native_ai_functions(self, source_name: str) -> bool:
        ai_enabled_dbs = {'snowflake', 'databricks', 'bigquery'}
        return source_name.lower() in ai_enabled_dbs

    def discover_models(self):
        self.models = {}
        sql_files = list(self.models_dir.glob("**/*.sql"))

        # Get list of available jinx names for NQL function recognition
        additional_funcs = list(self.npc_operations.jinx_map.keys())

        for sql_file in sql_files:
            model_name = sql_file.stem
            with open(sql_file, "r") as f:
                content = f.read()

            # Process Jinja templates ({% npc(...) %}, {% team.forenpc %}, etc.)
            if JINJA_AVAILABLE and '{%' in content:
                content = self.jinja_context.render_template(content)

            self.models[model_name] = SQLModel(
                model_name,
                content,
                str(sql_file),
                str(sql_file.parent),
                additional_functions=additional_funcs
            )

        return self.models

    def build_dag(self) -> Dict[str, Set[str]]:
        dag = {}
        for model_name, model in self.models.items():
            dag[model_name] = model.dependencies
        return dag

    def topological_sort(self) -> List[str]:
        dag = self.build_dag()
        
        true_in_degree = {model_name: 0 for model_name in self.models.keys()}
        adj_list = defaultdict(list) 

        for model_name, model in self.models.items():
            for dependency in model.dependencies:
                if dependency not in self.models:
                    raise ValueError(f"Dependency '{dependency}' of model '{model_name}' not found in discovered models.")
                true_in_degree[model_name] += 1 
                adj_list[dependency].append(model_name) 

        queue = deque([model_name for model_name in self.models.keys() if true_in_degree[model_name] == 0])
        result = []
        
        while queue:
            current_model = queue.popleft()
            result.append(current_model)

            for dependent_model in adj_list[current_model]:
                true_in_degree[dependent_model] -= 1 
                if true_in_degree[dependent_model] == 0:
                    queue.append(dependent_model)

        if len(result) != len(self.models):
            raise ValueError("Circular dependency detected or some models not processed.")

        return result

    def _replace_model_references(self, sql_content: str) -> str:
        ref_pattern = (
            r"\{\{\s*ref\s*\(\s*['\"]([^'\"]+)['\"]\s*\)\s*\}\}"
        )

        def replace_ref(match):
            model_name = match.group(1)

            # First check if it's a model we're compiling
            if model_name in self.models:
                if self.target_schema:
                    return f"{self.target_schema}.{model_name}"
                return model_name

            # Otherwise, check if it's an existing table in the database
            if self._table_exists(model_name):
                if self.target_schema:
                    return f"{self.target_schema}.{model_name}"
                return model_name

            # If neither, raise an error
            raise ValueError(
                f"Model or table '{model_name}' referenced by '{{{{ ref('{model_name}') }}}}' not found during compilation."
            )

        replaced_sql = re.sub(ref_pattern, replace_ref, sql_content)
        return replaced_sql

    def _clean_sql_for_execution(self, sql_content: str) -> str:
        config_pattern = r'\{\{[\s]*config\((.*?)\)[\s]*\}\}'
        cleaned_sql = re.sub(config_pattern, '', sql_content, flags=re.DOTALL).strip()
        cleaned_sql = re.sub(r"--.*?\n", "\n", cleaned_sql)
        cleaned_sql = re.sub(r"/\*.*?\*/", "", cleaned_sql, flags=re.DOTALL)
        cleaned_sql = re.sub(r"\s+", " ", cleaned_sql).strip()
        return cleaned_sql
    
    def _execute_standard_sql(
        self, 
        sql_to_execute: str, 
        engine: Engine
    ) -> pd.DataFrame:
        return pd.read_sql(sql_to_execute, engine)

    def _execute_ai_model(self, cleaned_sql_content: str, model: SQLModel) -> pd.DataFrame:
        processed_sql = self._replace_model_references(cleaned_sql_content)

        db_type = self.target_engine.dialect.name.lower()
        print(f"DEBUG: Determined DB dialect: '{db_type}'")

        if self._has_native_ai_functions(db_type):
            print(f"DEBUG: Native AI functions ARE supported for '{db_type}'. Attempting native translation.")
            transformer = NativeDatabaseAITransformer(db_type)
            sql_to_execute_with_native_ai = processed_sql

            print("DEBUG: AI functions and NQL calls to replace (from model.ai_functions):")
            if model.ai_functions:
                for fn, params in model.ai_functions.items():
                    print(f"  Function: {fn}, Full Call String: '{params.get('full_call_string')}'")
            else:
                print("  (None found in model.ai_functions to replace natively)")

            # Replace NQL calls with native functions
            for func_name, params in model.ai_functions.items():
                original_nql_call = params.get('full_call_string')
                if not original_nql_call:
                    print(f"WARNING: 'full_call_string' not found for NQL function '{func_name}'. Skipping native replacement attempt.")
                    continue

                try:
                    column_ref = params.get('column', '')
                    
                    transform_kwargs = {
                        'text': column_ref,
                        'prompt': column_ref,
                        'query': params.get('query', ''),
                        'context': params.get('context', ''),
                        'npc': params.get('npc', '')
                    }
                    
                    native_func_call = transformer.transform_ai_function(
                        func_name,
                        **transform_kwargs
                    )
                    
                    print(f"DEBUG: Replacing '{original_nql_call}' with '{native_func_call}'")
                    
                    # NORMALIZE WHITESPACE in both the original call and the SQL
                    # This handles multiline NQL calls with varying indentation
                    normalized_original = re.sub(r'\s+', ' ', original_nql_call).strip()
                    normalized_sql = re.sub(r'\s+', ' ', sql_to_execute_with_native_ai).strip()
                    
                    # Find the normalized pattern in the normalized SQL
                    if normalized_original in normalized_sql:
                        # Now do the replacement on the ORIGINAL (non-normalized) SQL
                        # by creating a flexible regex pattern
                        # Escape special regex chars but allow flexible whitespace
                        pattern_parts = [re.escape(part) for part in original_nql_call.split()]
                        flexible_pattern = r'\s*'.join(pattern_parts)
                        pattern = re.compile(flexible_pattern, re.IGNORECASE | re.DOTALL)
                        
                        old_sql = sql_to_execute_with_native_ai
                        sql_to_execute_with_native_ai = pattern.sub(native_func_call, sql_to_execute_with_native_ai, count=1)
                        
                        if old_sql != sql_to_execute_with_native_ai:
                            print(f"DEBUG: Successfully replaced with flexible whitespace pattern.")
                        else:
                            print(f"ERROR: Flexible pattern replacement failed for '{func_name}'.")
                    else:
                        print(f"ERROR: Could not find normalized NQL call in SQL for '{func_name}'.")

                except ValueError as e:
                    print(f"WARNING: Native translation failed for '{func_name}': {e}. This AI function will NOT be natively translated.")
                except Exception as e: 
                    print(f"ERROR: An unexpected error occurred during native AI transformation for '{func_name}': {e}. This AI function will NOT be natively translated.")            # Check for remaining NQL calls
            if "nql." in sql_to_execute_with_native_ai.lower():
                print(f"WARNING: Some NQL calls remain after native translation attempts. Replacing remaining NQL calls with NULLs.")
                sql_to_execute_with_native_ai = self._replace_nql_calls_with_null(sql_to_execute_with_native_ai, model)

            print(f"DEBUG: Final SQL for native/mixed AI execution:\n{sql_to_execute_with_native_ai}\n")
            target_engine_for_native_ai = self.target_engine
            return pd.read_sql(sql_to_execute_with_native_ai, target_engine_for_native_ai)

        else: # Fallback path when native AI is not supported for the determined DB type
            print(f"DEBUG: Native AI functions are NOT supported for '{db_type}'. Entering Python fallback path.")
            sql_with_nql_as_null = self._replace_nql_calls_with_null(processed_sql, model)
            
            print(f"DEBUG: SQL to execute in pure fallback (NQL as NULLs for DB):\n{sql_with_nql_as_null}\n")

            target_engine_for_fallback = self.target_engine # Use target_engine directly
            df = pd.read_sql(sql_with_nql_as_null, target_engine_for_fallback)

            # Apply Python-driven AI functions on the DataFrame
            for func_name, params in model.ai_functions.items():
                try:
                    result_series = self.npc_operations.execute_ai_function(func_name, df, **params)
                    # Use the SQL alias if available, otherwise generate one
                    result_column_name = params.get('alias', f"{func_name}_result")
                    df[result_column_name] = result_series
                    print(f"DEBUG: AI function '{func_name}' result stored in column '{result_column_name}'.")
                except Exception as e:
                    print(f"ERROR: Executing AI function '{func_name}': {e}. Assigning NULL.")
                    result_column_name = params.get('alias', f"{func_name}_result")
                    df[result_column_name] = None
                    
            return df

    def _replace_nql_calls_with_null(self, sql_content: str, model: SQLModel) -> str:
        """
        Replaces nql.func(...) calls with NULL placeholders.
        This is used for the fallback path where we execute SQL first, then apply AI functions in Python.
        """
        modified_sql = sql_content

        # Pattern to match nql.function_name(...) with nested parentheses support
        # Also captures the 'as alias' part if present
        nql_pattern = r'nql\.(\w+)\s*\(((?:[^()]|\([^()]*\))*)\)(\s+as\s+(\w+))?'

        def replace_with_null(match):
            func_name = match.group(1)
            alias_part = match.group(3) or ''
            alias_name = match.group(4)

            # If no alias specified, generate one from function name
            if not alias_name:
                alias_name = f"{func_name}_result"
                alias_part = f" as {alias_name}"

            print(f"DEBUG: Replacing nql.{func_name}(...) with NULL{alias_part}")
            return f"NULL{alias_part}"

        modified_sql = re.sub(nql_pattern, replace_with_null, modified_sql, flags=re.IGNORECASE | re.DOTALL)

        return modified_sql

    def execute_model(self, model_name: str) -> pd.DataFrame:
        self.current_model = model_name
        model = self.models[model_name]

        cleaned_sql_content = self._clean_sql_for_execution(model.content)
        
        print(f"DEBUG: Cleaned SQL content for model '{model_name}':\n{cleaned_sql_content}\n")

        if model.has_ai_function:
            df = self._execute_ai_model(cleaned_sql_content, model)
        else:
            compiled_sql = self._replace_model_references(
                cleaned_sql_content
            )
            print(f"DEBUG: Compiled standard SQL for model '{model_name}':\n{compiled_sql}\n")
            df = self._execute_standard_sql(
                compiled_sql, 
                self.target_engine
            )

        self._materialize_to_db(model_name, df, model.config)
        return df

    def _materialize_to_db(
        self, 
        model_name: str, 
        df: pd.DataFrame,
        config: Dict
    ):
        materialization = config.get('materialized', 'table')
        
        table_name = model_name
        table_name_with_schema = (
            f"{self.target_schema}.{table_name}" 
            if self.target_schema 
            else table_name
        )
        
        with self.target_engine.begin() as conn:
            if self.target_schema:
                inspector = inspect(conn)
                if not inspector.has_schema(self.target_schema):
                    print(f"Creating schema '{self.target_schema}'...")
                    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {self.target_schema}"))
                    print(f"Schema '{self.target_schema}' created (if it didn't exist).")

            if materialization == 'view':
                print(
                    f"Warning: Materialization '{materialization}' requested for model '{model_name}'. "
                    f"Pandas `to_sql` does not directly create SQL VIEWS from DataFrames. "
                    f"Materializing as TABLE instead. You may need to manually create the view."
                )
                df.to_sql(
                    table_name,
                    self.target_engine, 
                    schema=self.target_schema,
                    index=False,
                    if_exists='replace'
                )
                print(f"Materialized model {model_name} as TABLE to {table_name_with_schema}")
            else:
                df.to_sql(
                    table_name,
                    self.target_engine, 
                    schema=self.target_schema,
                    index=False,
                    if_exists='replace'
                )
                print(f"Materialized model {model_name} as TABLE to {table_name_with_schema}")

    def _table_exists(self, table_name: str) -> bool:
        with self.target_engine.connect() as conn:
            inspector = inspect(conn)
            return inspector.has_table(table_name, schema=self.target_schema) or \
                   inspector.has_view(table_name, schema=self.target_schema)

    def run_all_models(self):
        self.discover_models()
        execution_order = self.topological_sort()

        print(f"Running models in order: {execution_order}")

        results = {}
        for model_name in execution_order:
            print(f"\nExecuting model: {model_name}")

            model = self.models[model_name]
            for dep in model.dependencies:
                if not self._table_exists(dep):
                    if dep not in results:
                        raise ValueError(
                            f"Dependency '{dep}' for model '{model_name}' not found in database or already processed models. "
                            f"Please ensure all dependencies are resolved and run first."
                        )

            results[model_name] = self.execute_model(model_name)

        return results
