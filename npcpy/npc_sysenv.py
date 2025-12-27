from datetime import datetime
from dotenv import load_dotenv
import logging
import re
import os
import socket
import concurrent.futures 
import platform
import sqlite3
import sys
from typing import Dict, List
import textwrap
import json


import requests
ON_WINDOWS = platform.system() == "Windows"

try:
    if not ON_WINDOWS:
        import termios
        import tty
        import pty
        import select
        import signal
except ImportError:
    termios = None
    tty = None
    pty = None
    select = None
    signal = None


try:
    import readline
except ImportError:
    readline = None
    logging.warning('no readline support, some features may not work as desired.')


try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
except ImportError:
    Console = None
    Markdown = None
    Syntax = None

import warnings
import time


running = True
is_recording = False
recording_data = []
buffer_data = []
last_speech_time = 0

warnings.filterwarnings("ignore", module="whisper.transcribe")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch.serialization")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SDL_AUDIODRIVER"] = "dummy"

def check_internet_connection(timeout=5):
    """
    Checks for internet connectivity by trying to connect to a well-known host.
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False


def get_locally_available_models(project_directory, airplane_mode=False):
    available_models = {}
    env_path = os.path.join(project_directory, ".env")
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip().strip("\"'")

    internet_available = check_internet_connection()
    if not internet_available:
        logging.info(
            "No internet connection detected. "
            "External API calls will be skipped."
        )
        airplane_mode = True
    else:
        logging.info(
            "Internet connection detected. "
            "Proceeding based on 'airplane_mode' parameter."
        )
    
    custom_providers = load_custom_providers()
    
    for provider_name, config in custom_providers.items():
        api_key_var = config.get('api_key_var')
        if not api_key_var:
            api_key_var = f"{provider_name.upper()}_API_KEY"
        
        if api_key_var in env_vars or os.environ.get(api_key_var):
            try:
                import requests
                
                def fetch_custom_models():
                    base_url = config.get('base_url', '')
                    headers = config.get('headers', {})
                    
                    api_key = env_vars.get(api_key_var) or \
                              os.environ.get(api_key_var)
                    if api_key:
                        headers['Authorization'] = f'Bearer {api_key}'
                    
                    models_endpoint = f"{base_url.rstrip('/')}/models"
                    response = requests.get(
                        models_endpoint, 
                        headers=headers,
                        timeout=3.5
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if isinstance(data, dict) and 'data' in data:
                            return [
                                m['id'] for m in data['data'] 
                                if 'id' in m
                            ]
                        elif isinstance(data, list):
                            return [
                                m['id'] for m in data 
                                if isinstance(m, dict) and 'id' in m
                            ]
                    return []
                
                models = fetch_custom_models()
                for model in models:
                    available_models[model] = 'openai-like'
                    
                logging.info(
                    f"Loaded {len(models)} models "
                    f"from custom provider '{provider_name}'"
                )
                
            except Exception as e:
                logging.warning(
                    f"Failed to load models from "
                    f"custom provider '{provider_name}': {e}"
                )
    
    
    airplane_mode = False
    if not airplane_mode:
        timeout_seconds = 3.5
        

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
          
            if 'NPCSH_API_URL' in env_vars or os.environ.get('NPCSH_API_URL'):
              try:
                  import requests
                  
                  def fetch_custom_models():
                      base_url = env_vars.get('NPCSH_API_URL') or os.environ.get('NPCSH_API_URL')                      
                      models_endpoint = f"{base_url.rstrip('/')}/models"
                      response = requests.get(
                          models_endpoint, 

                          timeout=3.5
                      )
                      
                      if response.status_code == 200:
                          data = response.json()
                          
                          if isinstance(data, dict) and 'data' in data:
                              return [
                                  m['id'] for m in data['data'] 
                                  if 'id' in m
                              ]
                          elif isinstance(data, list):
                              return [
                                  m['id'] for m in data 
                                  if isinstance(m, dict) and 'id' in m
                              ]
                      return []
                  
                  models = fetch_custom_models()
                  for model in models:
                      available_models[model] = 'openai-like'
                      

                
                  
              except Exception as e:
                  logging.warning(
                      f"Failed to load models from "
                      f"custom provider 'openai-like': {e}"
                  )
  
            
            if "ANTHROPIC_API_KEY" in env_vars or os.environ.get("ANTHROPIC_API_KEY"):
                try:
                    import anthropic
                    
                    def fetch_anthropic_models():
                        client = anthropic.Anthropic(api_key=env_vars.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
                        
                        
                        return client.models.list()

                    future = executor.submit(fetch_anthropic_models)
                    models = future.result(timeout=timeout_seconds) 

                    for model in models.data:
                        available_models[model.id] = 'anthropic'
                            
                except (ImportError, concurrent.futures.TimeoutError, Exception) as e:
                    logging.info(f"Anthropic models not indexed or timed out: {e}")

            if "OPENAI_API_KEY" in env_vars or os.environ.get("OPENAI_API_KEY"):
                try:
                    import openai

                    def fetch_openai_models():
                        openai.api_key = env_vars.get("OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY", None)
                        return openai.models.list()

                    future = executor.submit(fetch_openai_models)
                    models = future.result(timeout=timeout_seconds) 

                    for model in models.data:
                        if (
                            (
                                "gpt" in model.id
                                or "o1" in model.id
                                or "o3" in model.id
                                or "chat" in model.id
                            )
                            and "audio" not in model.id
                            and "realtime" not in model.id
                        ):
                            available_models[model.id] = "openai"
                except (ImportError, openai.APIError, concurrent.futures.TimeoutError, Exception) as e:
                    logging.info(f"OpenAI models not indexed or timed out: {e}")

            if "GEMINI_API_KEY" in env_vars or os.environ.get("GEMINI_API_KEY"):
                try:
                    from google import genai
                    def fetch_gemini_models():
                        client = genai.Client(api_key=env_vars.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY"))
                        found_models = []
                        
                        target_models = [
                            'gemini-2.5-pro', 
                            'gemini-2.5-flash', 
                            'gemini-2.0-flash', 
                            'gemini-2.0-pro', 
                            'gemini-1.5-pro', 
                            'gemini-1.5-flash'
                            'gemini-3-flash-preview', 
                            'gemini-3-pro-preview', 
                        ]
                        
                        for m in client.models.list():
                            for action in m.supported_actions:
                                if action == "generateContent":
                                    if 'models/' in m.name:
                                        model_name_part = m.name.split('/')[1]  
                                        
                                        if any(model in model_name_part for model in target_models):
                                            found_models.append(model_name_part)
                        return set(found_models)
                    future = executor.submit(fetch_gemini_models)
                    models = future.result(timeout=timeout_seconds) 

                    for model in models: 
                        if "gemini" in model:
                            available_models[model] = "gemini"
                except (ImportError, concurrent.futures.TimeoutError, Exception) as e:
                    logging.info(f"Gemini models not indexed or timed out: {e}")
            
            if "DEEPSEEK_API_KEY" in env_vars or os.environ.get("DEEPSEEK_API_KEY"):
                available_models['deepseek-chat'] = 'deepseek'
                available_models['deepseek-reasoner'] = 'deepseek'        
    try:
        import ollama
        timeout_seconds = 0.5 
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ollama_executor:
            def fetch_ollama_models():
                return ollama.list()
            
            future = ollama_executor.submit(fetch_ollama_models)
            models = future.result(timeout=timeout_seconds) 

        for model in models.models:
            if "embed" not in model.model:
                mod = model.model
                available_models[mod] = "ollama"
    except (ImportError, concurrent.futures.TimeoutError, Exception) as e:
        logging.info(f"Error loading Ollama models or timed out: {e}")

    # Scan for local GGUF/GGML models
    gguf_dirs = [
        os.path.expanduser('~/.npcsh/models/gguf'),
        os.path.expanduser('~/.npcsh/models'),
        os.path.expanduser('~/models'),
        os.path.expanduser('~/.cache/huggingface/hub'),
    ]
    env_gguf_dir = os.environ.get('NPCSH_GGUF_DIR')
    if env_gguf_dir:
        gguf_dirs.insert(0, os.path.expanduser(env_gguf_dir))

    seen_paths = set()
    for scan_dir in gguf_dirs:
        if not os.path.isdir(scan_dir):
            continue
        try:
            for root, dirs, files in os.walk(scan_dir):
                for f in files:
                    if f.endswith(('.gguf', '.ggml')) and not f.startswith('.'):
                        full_path = os.path.join(root, f)
                        if full_path not in seen_paths:
                            seen_paths.add(full_path)
                            available_models[full_path] = "llamacpp"
        except Exception as e:
            logging.info(f"Error scanning GGUF directory {scan_dir}: {e}")

    # Check for LM Studio server (OpenAI-compatible API on port 1234)
    try:
        import requests
        response = requests.get('http://127.0.0.1:1234/v1/models', timeout=1)
        if response.ok:
            data = response.json()
            for model in data.get('data', []):
                model_id = model.get('id', model.get('name', 'unknown'))
                available_models[model_id] = "lmstudio"
    except Exception as e:
        logging.debug(f"LM Studio not available: {e}")

    # Check for llama.cpp server (OpenAI-compatible API on port 8080)
    try:
        import requests
        response = requests.get('http://127.0.0.1:8080/v1/models', timeout=1)
        if response.ok:
            data = response.json()
            for model in data.get('data', []):
                model_id = model.get('id', model.get('name', 'unknown'))
                available_models[model_id] = "llamacpp-server"
    except Exception as e:
        logging.debug(f"llama.cpp server not available: {e}")

    return available_models



def log_action(action: str, detail: str = "") -> None:
    """
    Function Description:
        This function logs an action with optional detail.
    Args:
        action: The action to log.
        detail: Additional detail to log.
    Keyword Args:
        None
    Returns:
        None
    """
    logging.info(f"{action}: {detail}")



def preprocess_code_block(code_text):
    """
    Preprocess code block text to remove leading spaces.
    """
    lines = code_text.split("\n")
    return "\n".join(line.lstrip() for line in lines)


def preprocess_markdown(md_text):
    """
    Preprocess markdown text to handle code blocks separately.
    """
    lines = md_text.split("\n")
    processed_lines = []

    inside_code_block = False
    current_code_block = []

    for line in lines:
        if line.startswith("```"):  
            if inside_code_block:
                
                processed_lines.append("```")
                processed_lines.extend(
                    textwrap.dedent("\n".join(current_code_block)).split("\n")
                )
                processed_lines.append("```")
                current_code_block = []
            inside_code_block = not inside_code_block
        elif inside_code_block:
            current_code_block.append(line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)



def request_user_input(input_request: Dict[str, str]) -> str:
    """
    Request and get input from user.

    Args:
        input_request: Dict with reason and prompt for input

    Returns:
        User's input text
    """
    print(f"\nAdditional input needed: {input_request['reason']}")
    return input(f"{input_request['prompt']}: ")


def render_markdown(text: str) -> None:
    """
    Renders markdown text, but handles code blocks as plain syntax-highlighted text.
    """
    lines = text.split("\n")
    console = Console()

    inside_code_block = False
    code_lines = []
    lang = None

    for line in lines:
        if line.startswith("```"):
            if inside_code_block:
                
                code = "\n".join(code_lines)
                if code.strip():
                    syntax = Syntax(
                        code, lang or "python", theme="monokai", line_numbers=False
                    )
                    console.print(syntax)
                code_lines = []
            else:
                
                lang = line[3:].strip() or None
            inside_code_block = not inside_code_block
        elif inside_code_block:
            code_lines.append(line)
        else:
            
            console.print(Markdown(line))

def get_directory_npcs(directory: str = None) -> List[str]:
    """
    Function Description:
        This function retrieves a list of valid NPCs from the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:
        None
    Returns:
        A list of valid NPCs.
    """
    if directory is None:
        directory = os.path.expanduser("./npc_team")
    npcs = []
    for filename in os.listdir(directory):
        if filename.endswith(".npc"):
            npcs.append(filename[:-4])
    return npcs


def get_db_npcs(db_path: str) -> List[str]:
    """
    Function Description:
        This function retrieves a list of valid NPCs from the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:
        None
    Returns:
        A list of valid NPCs.
    """
    if "~" in db_path:
        db_path = os.path.expanduser(db_path)
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM compiled_npcs")
    npcs = [row[0] for row in cursor.fetchall()]
    db_conn.close()
    return npcs

def guess_mime_type(filename):
    """Guess the MIME type of a file based on its extension."""
    extension = os.path.splitext(filename)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
    }
    return mime_types.get(extension, "application/octet-stream")


def ensure_dirs_exist(*dirs):
    """Ensure all specified directories exist"""
    for dir_path in dirs:
        os.makedirs(os.path.expanduser(dir_path), exist_ok=True)

def init_db_tables(db_path="~/npcsh_history.db"):
    """Initialize necessary database tables"""
    db_path = os.path.expanduser(db_path)
    with sqlite3.connect(db_path) as conn:
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS npc_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT,  
                entry_type TEXT,
                content TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_name TEXT,
                step_name TEXT,
                output TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compiled_npcs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                source_path TEXT,
                compiled_content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()




def get_model_and_provider(command: str, available_models: list) -> tuple:
    """
    Function Description:
        Extracts model and provider from command and autocompletes if possible.
    Args:
        command : str : Command string
        available_models : list : List of available models
    Keyword Args:
        None
    Returns:
        model_name : str : Model name
        provider : str : Provider
        cleaned_command : str : Clean


    """

    model_match = re.search(r"@(\S+)", command)
    if model_match:
        model_name = model_match.group(1)
        
        matches = [m for m in available_models if m.startswith(model_name)]
        if matches:
            if len(matches) == 1:
                model_name = matches[0]  
            
            provider = lookup_provider(model_name)
            if provider:
                
                cleaned_command = command.replace(
                    f"@{model_match.group(1)}", ""
                ).strip()
                
                return model_name, provider, cleaned_command
            else:
                return None, None, command  
        else:
            return None, None, command  
    else:
        return None, None, command  

def render_code_block(code: str, language: str = None) -> None:
    """Render a code block with syntax highlighting using rich, left-justified with no line numbers"""
    from rich.syntax import Syntax
    from rich.console import Console

    console = Console(highlight=True)
    code = code.strip()
    
    if code.split("\n", 1)[0].lower() in ["python", "bash", "javascript"]:
        code = code.split("\n", 1)[1]
    syntax = Syntax(
        code, language or "python", theme="monokai", line_numbers=False, padding=0
    )
    console.print(syntax)
    
def print_and_process_stream_with_markdown(response, model, provider, show=False):
    import sys
    
    str_output = ""
    dot_count = 0  
    tool_call_data = {"id": None, "function_name": None, "arguments": ""}
    interrupted = False
    
    if isinstance(response, str):
        render_markdown(response)  
        print('\n') 
        return response 
    
    
    sys.stdout.write('\033[s')  
    sys.stdout.flush()
    
    try:
        for chunk in response:

            if provider == "ollama" and 'gpt-oss' not in model:

                if "message" in chunk and "tool_calls" in chunk["message"]:
                    for tool_call in chunk["message"]["tool_calls"]:
                        if "id" in tool_call:
                            tool_call_data["id"] = tool_call["id"]
                        if "function" in tool_call:
                            if "name" in tool_call["function"]:
                                tool_call_data["function_name"] = tool_call["function"]["name"]
                            if "arguments" in tool_call["function"]:
                                if isinstance(tool_call["function"]["arguments"], dict):
                                    tool_call_data["arguments"] += json.dumps(tool_call["function"]["arguments"])
                                else:
                                    tool_call_data["arguments"] += tool_call["function"]["arguments"]                
                chunk_content = chunk["message"]["content"] if "message" in chunk and "content" in chunk["message"] else ""
                reasoning_content = chunk['message'].get('thinking', '') if "message" in chunk and "thinking" in chunk['message'] else ""
                if show:
                    if len(reasoning_content) > 0:
                        print(reasoning_content, end="", flush=True)
                    if chunk_content != "":
                        print(chunk_content, end="", flush=True)
                else:
                    print('.', end="", flush=True)
                    dot_count += 1
                    
            else:
                for c in chunk.choices:
                    if hasattr(c.delta, "tool_calls") and c.delta.tool_calls:
                        for tool_call in c.delta.tool_calls:
                            if tool_call.id:
                                tool_call_data["id"] = tool_call.id
                            if tool_call.function:
                                if hasattr(tool_call.function, "name") and tool_call.function.name:
                                    tool_call_data["function_name"] = tool_call.function.name
                                if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                                    tool_call_data["arguments"] += tool_call.function.arguments
                
                chunk_content = ''
                reasoning_content = ''
                for c in chunk.choices:
                    if hasattr(c.delta, "reasoning_content"):        
                        reasoning_content += c.delta.reasoning_content
                                        
                chunk_content += "".join(
                    c.delta.content for c in chunk.choices if c.delta.content
                )
                if show:
                    if reasoning_content is not None:
                        print(reasoning_content, end="", flush=True)
                    if chunk_content != "":
                        print(chunk_content, end="", flush=True)
                else:
                    print('.', end="", flush=True)
                    dot_count += 1

            if not chunk_content:
                continue
            str_output += chunk_content
    
    except KeyboardInterrupt:
        interrupted = True
        print('\n⚠️ Stream interrupted by user')
    
    if tool_call_data["id"] or tool_call_data["function_name"] or tool_call_data["arguments"]:
        str_output += "\n\n"
        if tool_call_data["id"]:
            str_output += f"**ID:** {tool_call_data['id']}\n\n"
        if tool_call_data["function_name"]:
            str_output += f"**Function:** {tool_call_data['function_name']}\n\n"
        if tool_call_data["arguments"]:
            try:
                args_parsed = json.loads(tool_call_data["arguments"])
                str_output += f"**Arguments:**\n```json\n{json.dumps(args_parsed, indent=2)}\n```"
            except:
                str_output += f"**Arguments:** `{tool_call_data['arguments']}`"

    if interrupted:
        str_output += "\n\n[⚠️ Response interrupted by user]"
    
    
    sys.stdout.write('\033[u')  
    sys.stdout.write('\033[J')  
    sys.stdout.flush()
    
    
    render_markdown(str_output)
    print('\n')
    
    return str_output


def print_and_process_stream(response, model, provider):
    
    str_output = ""
    dot_count = 0  
    tool_call_data = {"id": None, "function_name": None, "arguments": ""}
    interrupted = False
    
    thinking_part=True
    thinking_str=''
    if isinstance(response, str):
        render_markdown(response)  
        print('\n') 
        return response 
    try:
        for chunk in response:

            if provider == "ollama" and 'gpt-oss' not in model:

                if "message" in chunk and "tool_calls" in chunk["message"]:
                    for tool_call in chunk["message"]["tool_calls"]:
                        if "id" in tool_call:
                            tool_call_data["id"] = tool_call["id"]
                        if "function" in tool_call:
                            if "name" in tool_call["function"]:
                                tool_call_data["function_name"] = tool_call["function"]["name"]
                            if "arguments" in tool_call["function"]:
                                if isinstance(tool_call["function"]["arguments"], dict):
                                    tool_call_data["arguments"] += json.dumps(tool_call["function"]["arguments"])
                                else:
                                    tool_call_data["arguments"] += tool_call["function"]["arguments"]                
                chunk_content = chunk["message"]["content"] if "message" in chunk and "content" in chunk["message"] else ""
                reasoning_content = chunk['message'].get('thinking', '') if "message" in chunk and "thinking" in chunk['message'] else ""

                if len(reasoning_content) > 0:
                    print(reasoning_content, end="", flush=True)
                    thinking_part = True
                if chunk_content != "":
                    print(chunk_content, end="", flush=True)
                    
            else:
                for c in chunk.choices:
                    if hasattr(c.delta, "tool_calls") and c.delta.tool_calls:
                        for tool_call in c.delta.tool_calls:
                            if tool_call.id:
                                tool_call_data["id"] = tool_call.id
                            if tool_call.function:
                                if hasattr(tool_call.function, "name") and tool_call.function.name:
                                    tool_call_data["function_name"] = tool_call.function.name
                                if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                                    tool_call_data["arguments"] += tool_call.function.arguments
                
                chunk_content = ''
                reasoning_content = ''
                for c in chunk.choices:
                    if hasattr(c.delta, "reasoning_content"):        
                        reasoning_content += c.delta.reasoning_content
                
                        
                chunk_content += "".join(
                    c.delta.content for c in chunk.choices if c.delta.content
                )
                if reasoning_content is not None:
                    if thinking_part:
                        thinking_str +='<think>'
                        thinking_part=False
                        print('<think>')
                    print(reasoning_content, end="", flush=True)
                    thinking_str+=reasoning_content
                
                
                if chunk_content != "":
                    if len(thinking_str) >0 and not thinking_part and '</think>' not in thinking_str:

                        thinking_str+='</think>'
                        print('</think>')
                    print(chunk_content, end="", flush=True)


            if not chunk_content:
                continue
            str_output += chunk_content
    
    except KeyboardInterrupt:
        interrupted = True
        print('\n⚠️ Stream interrupted by user')
    
    if tool_call_data["id"] or tool_call_data["function_name"] or tool_call_data["arguments"]:
        str_output += "\n\n"
        if tool_call_data["id"]:
            str_output += f"**ID:** {tool_call_data['id']}\n\n"
        if tool_call_data["function_name"]:
            str_output += f"**Function:** {tool_call_data['function_name']}\n\n"
        if tool_call_data["arguments"]:
            try:
                args_parsed = json.loads(tool_call_data["arguments"])
                str_output += f"**Arguments:**\n```json\n{json.dumps(args_parsed, indent=2)}\n```"
            except:
                str_output += f"**Arguments:** `{tool_call_data['arguments']}`"

    if interrupted:
        str_output += "\n\n[⚠️ Response interrupted by user]"
    

                
    return thinking_str+str_output   
def get_system_message(npc, team=None) -> str:

    if npc is None:
        return "You are a helpful assistant"
    if npc.plain_system_message:
        return npc.primary_directive

    system_message = f"""
.
..
...
....
.....
......
.......
........
.........
..........
Hello!
Welcome to the team.
You are the {npc.name} NPC with the following primary directive: {npc.primary_directive}.
Users may refer to you by your assistant name, {npc.name} and you should
consider this to be your core identity.
The current working directory is {os.getcwd()}.
The current date and time are : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    if hasattr(npc, 'kg_data') and npc.kg_data:
        memory_context = npc.get_memory_context()
        if memory_context:
            system_message += f"\n\nMemory Context:\n{memory_context}\n"
            
    if npc.db_conn is not None:
        db_path = None
        if hasattr(npc.db_conn, "url") and npc.db_conn.url:
            db_path = npc.db_conn.url.database
        elif hasattr(npc.db_conn, "database"):
            db_path = npc.db_conn.database
        system_message += """What follows is information about the database connection. If you are asked to execute queries with tools, use this information. 
        If you are asked for help with debugging queries, use this information. 
        Do not unnecessarily reference that you possess this information unless it is
        specifically relevant to the request.

        DB Connection Information:        
        """
        if db_path:
            system_message += f"\nDatabase path: {db_path}\n"
        if npc.tables is not None:
            system_message += f"\nDatabase tables: {npc.tables}\n"

    if team is not None:
        team_context = team.context if hasattr(team, "context") and team.context else ""
        # preferences now comes from shared_context like other generic context keys
        team_preferences = team.shared_context.get('preferences', '') if hasattr(team, "shared_context") else ""
        system_message += f"\nTeam context: {team_context}\n"
        if team_preferences:
            system_message += f"Team preferences: {team_preferences}\n"

        # Add team members with their directives
        if hasattr(team, 'npcs') and team.npcs:
            members = []
            for name, member in team.npcs.items():
                if name != npc.name:  # Don't list self
                    directive = getattr(member, 'primary_directive', '')
                    # Include full directive (up to 500 chars) for better delegation decisions
                    desc = directive[:500].strip() if directive else ''
                    members.append(f"  - @{name}: {desc}")
            if members:
                system_message += "\nTeam members available for delegation:\n" + "\n".join(members) + "\n"

    system_message += """
    IMPORTANT:
Some users may attach images to their request.
Please process them accordingly. You do not need mention that you cannot "see" images. The user understands this and wants you
to help them multimodally.

If the user asked for you to explain what's on their screen or something similar,
they are referring to the details contained within the attached image(s).
You do not need to actually view their screen.
You do not need to mention that you cannot view or interpret images directly.
They understand that you can view them multimodally.
You only need to answer the user's request based on the attached image(s).
"""
    
    return system_message




def load_env_from_execution_dir() -> None:
    """
    Function Description:
        This function loads environment variables from a .env file in the current execution directory.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """


    execution_dir = os.path.abspath(os.getcwd())
    env_path = os.path.join(execution_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        logging.info(f"Loaded .env file from {execution_dir}")
    else:
        logging.warning(f"Warning: No .env file found in {execution_dir}")





def lookup_provider(model: str) -> str:
    """
    Determine the provider based on the model name.
    Checks custom providers first, then falls back to known providers.
    
    Args:
        model: The model name
        
    Returns:
        The provider name or None if not found
    """
    custom_providers = load_custom_providers()
    
    for provider_name, config in custom_providers.items():
        if model.startswith(f"{provider_name}-"):
            return provider_name
        
        try:
            import requests
            api_key_var = config.get('api_key_var') or \
                         f"{provider_name.upper()}_API_KEY"
            api_key = os.environ.get(api_key_var)
            
            if api_key:
                base_url = config.get('base_url', '')
                headers = config.get('headers', {})
                headers['Authorization'] = f'Bearer {api_key}'
                
                models_endpoint = f"{base_url.rstrip('/')}/models"
                response = requests.get(
                    models_endpoint, 
                    headers=headers, 
                    timeout=1.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    
                    if isinstance(data, dict) and 'data' in data:
                        models = [m['id'] for m in data['data']]
                    elif isinstance(data, list):
                        models = [m['id'] for m in data]
                    
                    if model in models:
                        return provider_name
        except:
            pass
    
    if model == "deepseek-chat" or model == "deepseek-reasoner":
        return "deepseek"
        
    ollama_prefixes = [
        "llama", "deepseek", "qwen", "llava", 
        "phi", "mistral", "mixtral", "dolphin", 
        "codellama", "gemma",]
    if any(model.startswith(prefix) for prefix in ollama_prefixes):
        return "ollama"

    openai_prefixes = ["gpt-", "dall-e-", "whisper-", "o1"]
    if any(model.startswith(prefix) for prefix in openai_prefixes):
        return "openai"

    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    if "diffusion" in model:
        return "diffusers"
        
    return None


def load_custom_providers():
    """
    Load custom provider configurations from .npcshrc
    
    Returns:
        dict: Custom provider configurations keyed by provider name
    """
    custom_providers = {}
    npcshrc_path = os.path.expanduser("~/.npcshrc")
    
    if os.path.exists(npcshrc_path):
        with open(npcshrc_path, "r") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if "CUSTOM_PROVIDER_" in line and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().replace("export ", "")
                    value = value.strip().strip("\"'")
                    
                    try:
                        config = json.loads(value)
                        provider_name = key.replace(
                            "CUSTOM_PROVIDER_", ""
                        ).lower()
                        custom_providers[provider_name] = config
                    except json.JSONDecodeError as e:
                        logging.warning(
                            f"Failed to parse custom provider {key}: {e}"
                        )
                        continue
    
    return custom_providers
load_env_from_execution_dir()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", None)
gemini_api_key = os.getenv("GEMINI_API_KEY", None)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", None)
openai_api_key = os.getenv("OPENAI_API_KEY", None)

