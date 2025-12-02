from . import npc_compiler
from . import npc_sysenv
from . import llm_funcs
from . import ml_funcs
from . import npc_array
from . import sql
from . import work
from . import gen

# Expose key classes at package level
from .npc_array import NPCArray, ResponseTensor, LazyResult, infer_matrix, ensemble_vote
from .npc_compiler import NPC, Team, Jinx
from .llm_funcs import get_llm_response, check_llm_command, execute_llm_command
from .ml_funcs import fit_model, predict_model, score_model, ensemble_predict