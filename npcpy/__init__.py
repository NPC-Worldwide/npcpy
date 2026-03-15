from . import npc_compiler
from . import npc_sysenv
from . import llm_funcs
from . import ml_funcs
from . import npc_array
from . import sql
from . import work
from . import gen
from . import data
from . import memory
from . import tools

# Core types
from .npc_compiler import NPC, Team, Jinx, Agent, ToolAgent, CodingAgent

# LLM
from .llm_funcs import get_llm_response, check_llm_command, execute_llm_command

# ML
from .ml_funcs import fit_model, predict_model, score_model, ensemble_predict

# Arrays / Ensembles
from .npc_array import NPCArray, ResponseTensor, LazyResult, infer_matrix, ensemble_vote

# Tools
from .tools import auto_tools, create_tool_schema, create_tool_map, extract_function_info

# Generation
from .gen.response import get_litellm_response, get_ollama_response, calculate_cost
from .gen.embeddings import get_embeddings
from .gen.image_gen import generate_image

# Memory
from .memory.command_history import CommandHistory
from .memory.knowledge_graph import kg_initial, kg_evolve_incremental, kg_search_facts

# System
from .npc_sysenv import (
    get_data_dir,
    get_config_dir,
    get_npcshrc_path,
    get_history_db_path,
    get_locally_available_models,
    render_markdown,
)