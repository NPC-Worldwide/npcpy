import datetime
from flask import Flask, request, jsonify, Response
from flask_sse import sse
import redis
import threading
import uuid
import sys 
import traceback
import glob


from flask_cors import CORS
import os
import sqlite3
import json
from pathlib import Path
import yaml
from dotenv import load_dotenv

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

import base64
import shutil
import uuid

from npcpy.llm_funcs import gen_image                                                                                                                                                                      

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from npcpy.npc_sysenv import get_locally_available_models
from npcpy.memory.command_history import (
    CommandHistory,
    save_conversation_message,
    generate_message_id,
)
from npcpy.npc_compiler import  Jinx, NPC, Team 

from npcpy.llm_funcs import (
    get_llm_response, check_llm_command
)
from npcpy.npc_compiler import NPC
import base64

from npcpy.tools import auto_tools

import json
import os
from pathlib import Path
from flask_cors import CORS






cancellation_flags = {}
cancellation_lock = threading.Lock()


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
            "NPC_STUDIO_LICENSE_KEY": "",
            "default_folder": os.path.expanduser("~/.npcsh/"),
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
                        "NPC_STUDIO_LICENSE_KEY": "NPC_STUDIO_LICENSE_KEY",
                        "NPCSH_STREAM_OUTPUT": "NPCSH_STREAM_OUTPUT",
                        "NPC_STUDIO_DEFAULT_FOLDER": "default_folder",
                    }

                    if key in key_mapping:
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


@app.route("/api/jinxs/available", methods=["GET"])
def get_available_jinxs():
    """
    Get all available jinxs for a given NPC and/or team.
    Returns a list of jinx names that can be executed.
    """
    try:
        current_path = request.args.get('currentPath')
        npc_name = request.args.get('npc')
        
        jinx_names = set()  # Use set to avoid duplicates
        
        # Get team jinxs from project directory
        if current_path:
            team_jinxs_dir = os.path.join(current_path, 'npc_team', 'jinxs')
            if os.path.exists(team_jinxs_dir):
                for file in os.listdir(team_jinxs_dir):
                    if file.endswith('.jinx'):
                        jinx_names.add(file[:-5])  # Remove .jinx extension
        
        # Get global jinxs
        global_jinxs_dir = os.path.expanduser('~/.npcsh/npc_team/jinxs')
        if os.path.exists(global_jinxs_dir):
            for file in os.listdir(global_jinxs_dir):
                if file.endswith('.jinx'):
                    jinx_names.add(file[:-5])
        
        # Get NPC-specific jinxs if NPC is specified
        if npc_name:
            # Try to load the NPC and get its jinxs
            db_conn = get_db_connection()
            npc_object = load_npc_by_name_and_source(npc_name, 'project', db_conn, current_path)
            if not npc_object:
                npc_object = load_npc_by_name_and_source(npc_name, 'global', db_conn)
            
            if npc_object and hasattr(npc_object, 'jinxs_dict') and npc_object.jinxs_dict:
                jinx_names.update(npc_object.jinxs_dict.keys())
        
        return jsonify({
            'jinxs': sorted(list(jinx_names)),
            'error': None
        })
        
    except Exception as e:
        print(f"Error getting available jinxs: {str(e)}")
        traceback.print_exc()
        return jsonify({'jinxs': [], 'error': str(e)}), 500


@app.route("/api/jinx/execute", methods=["POST"])
def execute_jinx():
    """
    Execute a specific jinx with provided arguments.
    Streams the output back to the client.
    """
    data = request.json
    
    stream_id = data.get("streamId")
    if not stream_id:
        stream_id = str(uuid.uuid4())
    
    with cancellation_lock:
        cancellation_flags[stream_id] = False
    
    jinx_name = data.get("jinxName")
    jinx_args = data.get("jinxArgs", [])
    conversation_id = data.get("conversationId")
    model = data.get("model")
    provider = data.get("provider")
    npc_name = data.get("npc")
    npc_source = data.get("npcSource", "global")
    current_path = data.get("currentPath")
    
    if not jinx_name:
        return jsonify({"error": "jinxName is required"}), 400
    
    # Load project environment if applicable
    if current_path:
        load_project_env(current_path)
    
    # Load the NPC
    npc_object = None
    if npc_name:
        db_conn = get_db_connection()
        npc_object = load_npc_by_name_and_source(npc_name, npc_source, db_conn, current_path)
        if not npc_object and npc_source == 'project':
            npc_object = load_npc_by_name_and_source(npc_name, 'global', db_conn)
    
    # Try to find the jinx
    jinx = None
    
    # Check NPC's jinxs
    if npc_object and hasattr(npc_object, 'jinxs_dict') and jinx_name in npc_object.jinxs_dict:
        jinx = npc_object.jinxs_dict[jinx_name]
    
    # Check team jinxs
    if not jinx and current_path:
        team_jinx_path = os.path.join(current_path, 'npc_team', 'jinxs', f'{jinx_name}.jinx')
        if os.path.exists(team_jinx_path):
            jinx = Jinx(jinx_path=team_jinx_path)
    
    # Check global jinxs
    if not jinx:
        global_jinx_path = os.path.expanduser(f'~/.npcsh/npc_team/jinxs/{jinx_name}.jinx')
        if os.path.exists(global_jinx_path):
            jinx = Jinx(jinx_path=global_jinx_path)
    
    if not jinx:
        return jsonify({"error": f"Jinx '{jinx_name}' not found"}), 404
    
    # Extract inputs from args
    from npcpy.npc_compiler import extract_jinx_inputs
    input_values = extract_jinx_inputs(jinx_args, jinx)
    
    # Get conversation history
    command_history = CommandHistory(app.config.get('DB_PATH'))
    messages = fetch_messages_for_conversation(conversation_id)
    
    # Prepare jinxs_dict for execution
    all_jinxs = {}
    if npc_object and hasattr(npc_object, 'jinxs_dict'):
        all_jinxs.update(npc_object.jinxs_dict)
    
    def event_stream(current_stream_id):
        try:
            # Execute the jinx
            result = jinx.execute(
                input_values=input_values,
                jinxs_dict=all_jinxs,
                jinja_env=npc_object.jinja_env if npc_object else None,
                npc=npc_object,
                messages=messages
            )
            
            # Get output
            output = result.get('output', str(result))
            messages_updated = result.get('messages', messages)
            
            # Check for interruption
            with cancellation_lock:
                if cancellation_flags.get(current_stream_id, False):
                    yield f"data: {json.dumps({'type': 'interrupted'})}\n\n"
                    return
            
            # Stream the output in chunks for consistent UI experience
            if isinstance(output, str):
                chunk_size = 50  # Characters per chunk
                for i in range(0, len(output), chunk_size):
                    chunk = output[i:i + chunk_size]
                    chunk_data = {
                        "id": None,
                        "object": None,
                        "created": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "content": chunk,
                                "role": "assistant"
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
            else:
                # Non-string output, send as single chunk
                chunk_data = {
                    "id": None,
                    "object": None,
                    "created": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "content": str(output),
                            "role": "assistant"
                        },
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send completion message
            yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
            
            # Save to conversation history
            message_id = generate_message_id()
            save_conversation_message(
                command_history,
                conversation_id,
                "user",
                f"/{jinx_name} {' '.join(jinx_args)}",
                wd=current_path,
                model=model,
                provider=provider,
                npc=npc_name,
                message_id=message_id
            )
            
            message_id = generate_message_id()
            save_conversation_message(
                command_history,
                conversation_id,
                "assistant",
                str(output),
                wd=current_path,
                model=model,
                provider=provider,
                npc=npc_name,
                message_id=message_id
            )
            
        except Exception as e:
            print(f"Error executing jinx {jinx_name}: {str(e)}")
            traceback.print_exc()
            error_data = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
        
        finally:
            with cancellation_lock:
                if current_stream_id in cancellation_flags:
                    del cancellation_flags[current_stream_id]
    
    return Response(event_stream(stream_id), mimetype="text/event-stream")

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
            "NPC_STUDIO_LICENSE_KEY": "NPC_STUDIO_LICENSE_KEY",
            "NPCSH_STREAM_OUTPUT": "NPCSH_STREAM_OUTPUT",
            "default_folder": "NPC_STUDIO_DEFAULT_FOLDER",
        }

        os.makedirs(os.path.dirname(npcshrc_path), exist_ok=True)
        print(data)
        with open(npcshrc_path, "w") as f:
            
            for key, value in data.get("global_settings", {}).items():
                if key in key_mapping and value:
                    
                    if " " in str(value):
                        value = f'"{value}"'
                    f.write(f"export {key_mapping[key]}={value}\n")

            
            for key, value in data.get("global_vars", {}).items():
                if key and value:
                    if " " in str(value):
                        value = f'"{value}"'
                    f.write(f"export {key}={value}\n")

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
            if "claude-3-5-haiku-latest" in m:
                display_model = "claude-3.5-haiku"
            elif "claude-3-5-sonnet-latest" in m:
                display_model = "claude-3.5-sonnet"
            elif "gemini-1.5-flash" in m:
                display_model = "gemini-1.5-flash"  
            elif "gemini-2.0-flash-lite-preview-02-05" in m:
                display_model = "gemini-2.0-flash-lite-preview"

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
@app.route("/api/npc_team_global")
def get_npc_team_global():
    try:
        db_conn = get_db_connection()
        global_npc_directory = os.path.expanduser("~/.npcsh/npc_team")

        npc_data = []

        
        for file in os.listdir(global_npc_directory):
            if file.endswith(".npc"):
                npc_path = os.path.join(global_npc_directory, file)
                npc = NPC(file=npc_path, db_conn=db_conn)

                
                serialized_npc = {
                    "name": npc.name,
                    "primary_directive": npc.primary_directive,
                    "model": npc.model,
                    "provider": npc.provider,
                    "api_url": npc.api_url,
                    "use_global_jinxs": npc.use_global_jinxs,
                    "jinxs": [
                        {
                            "jinx_name": jinx.jinx_name,
                            "inputs": jinx.inputs,
                            "steps": [
                                {
                                    "name": step.get("name", f"step_{i}"),
                                    "engine": step.get("engine", "natural"),
                                    "code": step.get("code", "")
                                }
                                for i, step in enumerate(jinx.steps)
                            ]
                        }
                        for jinx in npc.jinxs
                    ],
                }
                npc_data.append(serialized_npc)

        return jsonify({"npcs": npc_data, "error": None})

    except Exception as e:
        print(f"Error loading global NPCs: {str(e)}")
        return jsonify({"npcs": [], "error": str(e)})


@app.route("/api/jinxs/global", methods=["GET"])
def get_global_jinxs():
    
    user_home = os.path.expanduser("~")
    jinxs_dir = os.path.join(user_home, ".npcsh", "npc_team", "jinxs")
    jinxs = []
    if os.path.exists(jinxs_dir):
        for file in os.listdir(jinxs_dir):
            if file.endswith(".jinx"):
                with open(os.path.join(jinxs_dir, file), "r") as f:
                    jinx_data = yaml.safe_load(f)
                    jinxs.append(jinx_data)
            print("file", file)

    return jsonify({"jinxs": jinxs})






@app.route("/api/jinxs/project", methods=["GET"])
def get_project_jinxs():
    current_path = request.args.get(
        "currentPath"
    )  
    if not current_path:
        return jsonify({"jinxs": []})

    if not current_path.endswith("npc_team"):
        current_path = os.path.join(current_path, "npc_team")

    jinxs_dir = os.path.join(current_path, "jinxs")
    jinxs = []
    if os.path.exists(jinxs_dir):
        for file in os.listdir(jinxs_dir):
            if file.endswith(".jinx"):
                with open(os.path.join(jinxs_dir, file), "r") as f:
                    jinx_data = yaml.safe_load(f)
                    jinxs.append(jinx_data)
    return jsonify({"jinxs": jinxs})


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


@app.route("/api/npc_team_project", methods=["GET"])
def get_npc_team_project():
    try:
        db_conn = get_db_connection()

        project_npc_directory = request.args.get("currentPath")
        if not project_npc_directory.endswith("npc_team"):
            project_npc_directory = os.path.join(project_npc_directory, "npc_team")

        npc_data = []

        for file in os.listdir(project_npc_directory):
            print(file)
            if file.endswith(".npc"):
                npc_path = os.path.join(project_npc_directory, file)
                npc = NPC(file=npc_path, db_conn=db_conn)

                
                serialized_npc = {
                    "name": npc.name,
                    "primary_directive": npc.primary_directive,
                    "model": npc.model,
                    "provider": npc.provider,
                    "api_url": npc.api_url,
                    "use_global_jinxs": npc.use_global_jinxs,
                    "jinxs": [
                        {
                            "jinx_name": jinx.jinx_name,
                            "inputs": jinx.inputs,
                            "steps": [
                                {
                                    "name": step.get("name", f"step_{i}"),
                                    "engine": step.get("engine", "natural"),
                                    "code": step.get("code", "")
                                }
                                for i, step in enumerate(jinx.steps)
                            ]
                        }
                        for jinx in npc.jinxs
                    ],
                }
                npc_data.append(serialized_npc)

        print(npc_data)
        return jsonify({"npcs": npc_data, "error": None})

    except Exception as e:
        print(f"Error fetching NPC team: {str(e)}")
        return jsonify({"npcs": [], "error": str(e)})
def get_last_used_model_and_npc_in_directory(directory_path):
    """
    Fetches the model and NPC from the most recent message in any conversation
    within the given directory.
    """
    engine = get_db_connection()
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT model, npc
                FROM conversation_history
                WHERE directory_path = :directory_path 
                AND model IS NOT NULL AND npc IS NOT NULL 
                AND model != '' AND npc != ''
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
            """)
            result = conn.execute(query, {"directory_path": directory_path}).fetchone()
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

def get_ctx_path(is_global, current_path=None):
    """Determines the path to the .ctx file."""
    if is_global:
        ctx_dir = os.path.join(os.path.expanduser("~/.npcsh/npc_team/"))
        ctx_files = glob.glob(os.path.join(ctx_dir, "*.ctx"))
        return ctx_files[0] if ctx_files else None
    else:
        if not current_path:
            return None
        
        ctx_dir = os.path.join(current_path, "npc_team")
        ctx_files = glob.glob(os.path.join(ctx_dir, "*.ctx"))
        return ctx_files[0] if ctx_files else None


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
        {"value": "dall-e-3", "display_name": "DALL-E 3"},
        {"value": "dall-e-2", "display_name": "DALL-E 2"},
        {"value": "gpt-image-1", "display_name": "GPT-Image-1"},
    ],
    "gemini": [
        {"value": "gemini-2.5-flash-image-preview", "display_name": "Gemini 2.5 Flash Image"},
        {"value": "imagen-3.0-generate-002", "display_name": "Imagen 3.0 Generate (Preview)"}, 
    ],
    "diffusers": [
        {"value": "runwayml/stable-diffusion-v1-5", "display_name": "Stable Diffusion v1.5"},
    ],
}

def get_available_image_models(current_path=None):
    """
    Retrieves available image generation models based on environment variables
    and predefined configurations.
    """
    
    if current_path:
        load_project_env(current_path) 
    
    all_image_models = []

    
    env_image_model = os.getenv("NPCSH_IMAGE_MODEL")
    env_image_provider = os.getenv("NPCSH_IMAGE_PROVIDER")

    if env_image_model and env_image_provider:
        all_image_models.append({
            "value": env_image_model,
            "provider": env_image_provider,
            "display_name": f"{env_image_model} | {env_image_provider} (Configured)"
        })

    
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
        elif provider_key == "diffusers":
            
            
            all_image_models.extend([
                {**model, "provider": provider_key, "display_name": f"{model['display_name']} | {provider_key}"}
                for model in models_list
            ])
        

    
    seen_models = set()
    unique_models = []
    for model_entry in all_image_models:
        key = (model_entry["value"], model_entry["provider"])
        if key not in seen_models:
            seen_models.add(key)
            unique_models.append(model_entry)

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
                            input_images.append(pil_img)
                            
                            
                            with open(image_path, 'rb') as f:
                                img_data = f.read()
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
    server_path = request.args.get("mcpServerPath")
    conversation_id = request.args.get("conversationId")
    npc_name = request.args.get("npc")
    
    if not server_path:
        return jsonify({"error": "mcpServerPath parameter is required."}), 400

    
    try:
        from npcsh.corca import MCPClientNPC
    except ImportError:
        return jsonify({"error": "MCP Client (npcsh.corca) not available. Ensure npcsh.corca is installed and importable."}), 500

    temp_mcp_client = None
    try:
        
        if conversation_id and npc_name and hasattr(app, 'corca_states'):
            state_key = f"{conversation_id}_{npc_name or 'default'}"
            if state_key in app.corca_states:
                existing_corca_state = app.corca_states[state_key]
                if hasattr(existing_corca_state, 'mcp_client') and existing_corca_state.mcp_client \
                   and existing_corca_state.mcp_client.server_script_path == server_path:
                    print(f"Using existing MCP client for {state_key} to fetch tools.")
                    temp_mcp_client = existing_corca_state.mcp_client
                    return jsonify({"tools": temp_mcp_client.available_tools_llm, "error": None})

        
        print(f"Creating a temporary MCP client to fetch tools for {server_path}.")
        temp_mcp_client = MCPClientNPC()
        if temp_mcp_client.connect_sync(server_path):
            return jsonify({"tools": temp_mcp_client.available_tools_llm, "error": None})
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


@app.route("/api/image_models", methods=["GET"]) 
def get_image_models_api():
    """
    API endpoint to retrieve available image generation models.
    """
    current_path = request.args.get("currentPath")
    try:
        image_models = get_available_image_models(current_path)
        return jsonify({"models": image_models, "error": None})
    except Exception as e:
        print(f"Error getting available image models: {str(e)}")
        traceback.print_exc()
        return jsonify({"models": [], "error": str(e)}), 500







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
    model = data.get("model", None)
    provider = data.get("provider", None)
    if provider is None:
        provider = available_models.get(model)
        
    npc_name = data.get("npc", None)
    npc_source = data.get("npcSource", "global")
    current_path = data.get("currentPath")
    
    if current_path:
        loaded_vars = load_project_env(current_path)
        print(f"Loaded project env variables for stream request: {list(loaded_vars.keys())}")
    
    npc_object = None
    team_object = None
    team = None  
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
    
    
    exe_mode = data.get('executionMode','chat')
    
    if exe_mode == 'chat':
        stream_response = get_llm_response(
            commandstr, 
            messages=messages, 
            images=images, 
            model=model,
            provider=provider, 
            npc=npc_object, 
            team=team_object,
            stream=True, 
            attachments=attachment_paths_for_llm,
            auto_process_tool_calls=True,
            **tool_args
        )
        messages = stream_response.get('messages', messages)

    elif exe_mode == 'npcsh':
        from npcsh._state import execute_command, initial_state
        from npcsh.routes import router
        initial_state.model = model
        initial_state.provider = provider
        initial_state.npc = npc_object
        initial_state.team = team_object
        initial_state.messages = messages
        initial_state.command_history = command_history
        
        state, stream_response = execute_command(
            commandstr, 
            initial_state, router=router)
        messages = state.messages        
        
    elif exe_mode == 'guac':
        from npcsh.guac import execute_guac_command
        from npcsh.routes import router
        from npcsh._state import initial_state
        from pathlib import Path
        import pandas as pd, numpy as np, matplotlib.pyplot as plt

        if not hasattr(app, 'guac_locals'):
            app.guac_locals = {}

        if conversation_id not in app.guac_locals:
            app.guac_locals[conversation_id] = {
                'pd': pd, 
                'np': np, 
                'plt': plt, 
                'datetime': datetime,
                'Path': Path, 
                'os': os, 
                'sys': sys, 
                'json': json
            }

        initial_state.model = model
        initial_state.provider = provider  
        initial_state.npc = npc_object
        initial_state.team = team_object
        initial_state.messages = messages
        initial_state.command_history = command_history
        
        state, stream_response = execute_guac_command(
            commandstr,
            initial_state, 
            app.guac_locals[conversation_id],
            "guac",
            Path.cwd() / "npc_team", 
            router
        )
        messages = state.messages
        
    elif exe_mode == 'corca':
        
        try:
            from npcsh.corca import execute_command_corca, create_corca_state_and_mcp_client, MCPClientNPC
            from npcsh._state import initial_state as state
        except ImportError:
            
            print("ERROR: npcsh.corca or MCPClientNPC not found. Corca mode is disabled.", file=sys.stderr)
            state = None 
            stream_response = {"output": "Corca mode is not available due to missing dependencies.", "messages": messages}
            
        
        if state is not None: 
            
            mcp_server_path_from_request = data.get("mcpServerPath")
            selected_mcp_tools_from_request = data.get("selectedMcpTools", [])
            
            
            effective_mcp_server_path = mcp_server_path_from_request
            if not effective_mcp_server_path and team_object and hasattr(team_object, 'team_ctx') and team_object.team_ctx:
                mcp_servers_list = team_object.team_ctx.get('mcp_servers', [])
                if mcp_servers_list and isinstance(mcp_servers_list, list):
                    first_server_obj = next((s for s in mcp_servers_list if isinstance(s, dict) and 'value' in s), None)
                    if first_server_obj:
                        effective_mcp_server_path = first_server_obj['value']
                elif isinstance(team_object.team_ctx.get('mcp_server'), str): 
                    effective_mcp_server_path = team_object.team_ctx.get('mcp_server')

            
            if not hasattr(app, 'corca_states'):
                app.corca_states = {}
            
            state_key = f"{conversation_id}_{npc_name or 'default'}"
            
            corca_state = None
            if state_key not in app.corca_states:
                
                corca_state = create_corca_state_and_mcp_client(
                    conversation_id=conversation_id,
                    command_history=command_history,
                    npc=npc_object,
                    team=team_object,
                    current_path=current_path,
                    mcp_server_path=effective_mcp_server_path
                )
                app.corca_states[state_key] = corca_state
            else:
                corca_state = app.corca_states[state_key]
                corca_state.npc = npc_object
                corca_state.team = team_object
                corca_state.current_path = current_path
                corca_state.messages = messages
                corca_state.command_history = command_history

                
                current_mcp_client_path = getattr(corca_state.mcp_client, 'server_script_path', None)

                if effective_mcp_server_path != current_mcp_client_path:
                    print(f"MCP server path changed/updated for {state_key}. Disconnecting old client (if any) and reconnecting to {effective_mcp_server_path or 'None'}.")
                    if corca_state.mcp_client and corca_state.mcp_client.session:
                        corca_state.mcp_client.disconnect_sync()
                        corca_state.mcp_client = None 

                    if effective_mcp_server_path:
                        new_mcp_client = MCPClientNPC()
                        if new_mcp_client.connect_sync(effective_mcp_server_path):
                            corca_state.mcp_client = new_mcp_client
                            print(f"Successfully reconnected MCP client for {state_key} to {effective_mcp_server_path}.")
                        else:
                            print(f"Failed to reconnect MCP client for {state_key} to {effective_mcp_server_path}. Corca will have no tools.")
                            corca_state.mcp_client = None
                    
                
            
            state, stream_response = execute_command_corca(
                commandstr,
                corca_state,
                command_history,
                selected_mcp_tools_names=selected_mcp_tools_from_request 
            )
            
            
            app.corca_states[state_key] = state
            messages = state.messages 


    user_message_filled = ''

    if isinstance(messages[-1].get('content'), list):
        for cont in messages[-1].get('content'):
            txt = cont.get('text')
            if txt is not None:
                user_message_filled +=txt       
    save_conversation_message(
        command_history, 
        conversation_id, 
        "user", 
        user_message_filled if len(user_message_filled)>0 else commandstr, 
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
        dot_count = 0
        interrupted = False
        tool_call_data = {"id": None, "function_name": None, "arguments": ""}

        try:
            if isinstance(stream_response, str) :
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
                yield f"data: {json.dumps(chunk_data)}"
                return
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
                yield f"data: {json.dumps(chunk_data)}"
                return
            for response_chunk in stream_response.get('response', stream_response.get('output')):
                with cancellation_lock:
                    if cancellation_flags.get(current_stream_id, False):
                        print(f"Cancellation flag triggered for {current_stream_id}. Breaking loop.")
                        interrupted = True
                        break

                print('.', end="", flush=True)
                dot_count += 1
                if "hf.co" in model or provider == 'ollama' and 'gpt-oss' not in model:
                    chunk_content = response_chunk["message"]["content"] if "message" in response_chunk and "content" in response_chunk["message"] else ""
                    if "message" in response_chunk and "tool_calls" in response_chunk["message"]:
                        for tool_call in response_chunk["message"]["tool_calls"]:
                            if "id" in tool_call:
                                tool_call_data["id"] = tool_call["id"]
                            if "function" in tool_call:
                                if "name" in tool_call["function"]:
                                    tool_call_data["function_name"] = tool_call["function"]["name"]
                                if "arguments" in tool_call["function"]:
                                    arg_val = tool_call["function"]["arguments"]
                                    if isinstance(arg_val, dict):
                                        arg_val = json.dumps(arg_val)
                                    tool_call_data["arguments"] += arg_val
                    if chunk_content:
                        complete_response.append(chunk_content)
                    chunk_data = {
                        "id": None, "object": None, 
                        "created": response_chunk["created_at"] or datetime.datetime.now(), 
                        "model": response_chunk["model"],
                        "choices": [{"index": 0, "delta": {"content": chunk_content, "role": response_chunk["message"]["role"]}, "finish_reason": response_chunk.get("done_reason")}]
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
                    for choice in response_chunk.choices:
                        if hasattr(choice.delta, "reasoning_content"):
                            reasoning_content += choice.delta.reasoning_content
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
            yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
            
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
            )

            with cancellation_lock:
                if current_stream_id in cancellation_flags:
                    del cancellation_flags[current_stream_id]
                    print(f"Cleaned up cancellation flag for stream ID: {current_stream_id}")
                    
    return Response(event_stream(stream_id), mimetype="text/event-stream")



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




@app.route("/api/execute", methods=["POST"])
def execute():
    data = request.json
    

    stream_id = data.get("streamId")
    if not stream_id:
        import uuid
        stream_id = str(uuid.uuid4())

    
    with cancellation_lock:
        cancellation_flags[stream_id] = False
    print(f"Starting execute stream with ID: {stream_id}")

    
    commandstr = data.get("commandstr")
    conversation_id = data.get("conversationId")
    model = data.get("model", 'llama3.2')
    provider = data.get("provider", 'ollama')
    if provider is None:
        provider = available_models.get(model)

        
    npc_name = data.get("npc", "sibiji")
    npc_source = data.get("npcSource", "global")
    team = data.get("team", None)
    current_path = data.get("currentPath")
    
    if current_path:
        loaded_vars = load_project_env(current_path)
        print(f"Loaded project env variables for stream request: {list(loaded_vars.keys())}")
    
    npc_object = None
    team_object = None
    
    
    if team:
        print(team)
        if hasattr(app, 'registered_teams') and team in app.registered_teams:
            team_object = app.registered_teams[team]
            print(f"Using registered team: {team}")
        else:
            print(f"Warning: Team {team} not found in registered teams")
    
    
    if npc_name:
        
        if team and hasattr(app, 'registered_teams') and team in app.registered_teams:
            team_object = app.registered_teams[team]
            print('team', team_object)
            
            if hasattr(team_object, 'npcs'):
                team_npcs = team_object.npcs
                if isinstance(team_npcs, dict):
                    if npc_name in team_npcs:
                        npc_object = team_npcs[npc_name]
                        print(f"Found NPC {npc_name} in registered team {team}")
                elif isinstance(team_npcs, list):
                    for npc in team_npcs:
                        if hasattr(npc, 'name') and npc.name == npc_name:
                            npc_object = npc
                            print(f"Found NPC {npc_name} in registered team {team}")
                            break
            
            if not npc_object and hasattr(team_object, 'forenpc') and hasattr(team_object.forenpc, 'name'):
                if team_object.forenpc.name == npc_name:
                    npc_object = team_object.forenpc
                    print(f"Found NPC {npc_name} as forenpc in team {team}")
        
        
        if not npc_object and hasattr(app, 'registered_npcs') and npc_name in app.registered_npcs:
            npc_object = app.registered_npcs[npc_name]
            print(f"Found NPC {npc_name} in registered NPCs")
        
        
        if not npc_object:
            db_conn = get_db_connection()
            npc_object = load_npc_by_name_and_source(npc_name, npc_source, db_conn, current_path)
            
            if not npc_object and npc_source == 'project':
                print(f"NPC {npc_name} not found in project directory, trying global...")
                npc_object = load_npc_by_name_and_source(npc_name, 'global', db_conn)
                
            if npc_object:
                print(f"Successfully loaded NPC {npc_name} from {npc_source} directory")
            else:
                print(f"Warning: Could not load NPC {npc_name}")

    attachments = data.get("attachments", [])
    command_history = CommandHistory(app.config.get('DB_PATH'))
    images = []
    attachments_loaded = []
    

    if attachments:
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

    messages = fetch_messages_for_conversation(conversation_id)
    if len(messages) == 0 and npc_object is not None:
        messages = [{'role': 'system', 'content': npc_object.get_system_prompt()}]
    elif len(messages)>0 and messages[0]['role'] != 'system' and npc_object is not None:
        messages.insert(0, {'role': 'system', 'content': npc_object.get_system_prompt()})
    elif len(messages) > 0 and npc_object is not None:
        messages[0]['content'] = npc_object.get_system_prompt()
    if npc_object is not None and messages and messages[0]['role'] == 'system':
        messages[0]['content'] = npc_object.get_system_prompt()

    message_id = generate_message_id()
    save_conversation_message(
        command_history, conversation_id, "user", commandstr,
        wd=current_path, model=model, provider=provider, npc=npc_name,
        team=team, attachments=attachments_loaded, message_id=message_id,
    )
    response_gen = check_llm_command(
        commandstr, messages=messages, images=images, model=model,
        provider=provider, npc=npc_object, team=team_object, stream=True
    )
    print(response_gen)
    
    message_id = generate_message_id()

    def event_stream(current_stream_id):
        complete_response = []
        dot_count = 0
        interrupted = False
        tool_call_data = {"id": None, "function_name": None, "arguments": ""}
        memory_data = None

        try:
            for response_chunk in stream_response.get('response', stream_response.get('output')):
                with cancellation_lock:
                    if cancellation_flags.get(current_stream_id, False):
                        print(f"Cancellation flag triggered for {current_stream_id}. Breaking loop.")
                        interrupted = True
                        break

                print('.', end="", flush=True)
                dot_count += 1
                
                if "hf.co" in model or provider == 'ollama':
                    chunk_content = response_chunk["message"]["content"] if "message" in response_chunk and "content" in response_chunk["message"] else ""
                    if "message" in response_chunk and "tool_calls" in response_chunk["message"]:
                        for tool_call in response_chunk["message"]["tool_calls"]:
                            if "id" in tool_call:
                                tool_call_data["id"] = tool_call["id"]
                            if "function" in tool_call:
                                if "name" in tool_call["function"]:
                                    tool_call_data["function_name"] = tool_call["function"]["name"]
                                if "arguments" in tool_call["function"]:
                                    arg_val = tool_call["function"]["arguments"]
                                    if isinstance(arg_val, dict):
                                        arg_val = json.dumps(arg_val)
                                    tool_call_data["arguments"] += arg_val
                    if chunk_content:
                        complete_response.append(chunk_content)
                    chunk_data = {
                        "id": None, "object": None, "created": response_chunk["created_at"], "model": response_chunk["model"],
                        "choices": [{"index": 0, "delta": {"content": chunk_content, "role": response_chunk["message"]["role"]}, "finish_reason": response_chunk.get("done_reason")}]
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
                    for choice in response_chunk.choices:
                        if hasattr(choice.delta, "reasoning_content"):
                            reasoning_content += choice.delta.reasoning_content
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
            
            conversation_turn_text = f"User: {commandstr}\nAssistant: {final_response_text}"
            
            try:
                memory_examples = command_history.get_memory_examples_for_context(
                    npc=npc_name,
                    team=team,
                    directory_path=current_path
                )
                
                memory_context = format_memory_context(memory_examples)
                
                facts = get_facts(
                    conversation_turn_text,
                    model=npc_object.model if npc_object else model,
                    provider=npc_object.provider if npc_object else provider,
                    npc=npc_object,
                    context=memory_context
                )
                
                if facts:
                    memories_for_approval = []
                    for i, fact in enumerate(facts):
                        memory_id = command_history.add_memory_to_database(
                            message_id=f"{conversation_id}_{datetime.now().strftime('%H%M%S')}_{i}",
                            conversation_id=conversation_id,
                            npc=npc_name or "default",
                            team=team or "default",
                            directory_path=current_path or "/",
                            initial_memory=fact['statement'],
                            status="pending_approval",
                            model=npc_object.model if npc_object else model,
                            provider=npc_object.provider if npc_object else provider
                        )
                        
                        memories_for_approval.append({
                            "memory_id": memory_id,
                            "content": fact['statement'],
                            "context": f"Type: {fact.get('type', 'unknown')}, Source: {fact.get('source_text', '')}",
                            "npc": npc_name or "default"
                        })
                    
                    memory_data = {
                        "type": "memory_approval",
                        "memories": memories_for_approval,
                        "conversation_id": conversation_id
                    }
                    
            except Exception as e:
                print(f"Memory generation error: {e}")

            if memory_data:
                yield f"data: {json.dumps(memory_data)}\n\n"

            yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
            
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
            )

            with cancellation_lock:
                if current_stream_id in cancellation_flags:
                    del cancellation_flags[current_stream_id]
                    print(f"Cleaned up cancellation flag for stream ID: {current_stream_id}")



    return Response(event_stream(stream_id), mimetype="text/event-stream")

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
                query = text("""
                SELECT DISTINCT conversation_id,
                       MIN(timestamp) as start_time,
                       MAX(timestamp) as last_message_timestamp,
                       GROUP_CONCAT(content) as preview
                FROM conversation_history
                WHERE directory_path = :path_without_slash OR directory_path = :path_with_slash
                GROUP BY conversation_id
                ORDER BY MAX(timestamp) DESC
                """)

                
                path_without_slash = path.rstrip('/')
                path_with_slash = path_without_slash + '/'
                
                result = conn.execute(query, {
                    "path_without_slash": path_without_slash,
                    "path_with_slash": path_with_slash
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
                        ch.*,
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

            return jsonify(
                {
                    "messages": [
                        {
                            "message_id": msg[1] if len(msg) > 1 else None,  
                            "role": msg[3] if len(msg) > 3 else None,
                            "content": msg[4] if len(msg) > 4 else None,
                            "timestamp": msg[5] if len(msg) > 5 else None,
                            "model": msg[6] if len(msg) > 6 else None,
                            "provider": msg[7] if len(msg) > 7 else None,
                            "npc": msg[8] if len(msg) > 8 else None,
                            "attachments": (
                                get_message_attachments(msg[1])
                                if len(msg) > 1 and msg[-1]  
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
