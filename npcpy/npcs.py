from npcpy.npc_compiler import NPC
import os 
from sqlalchemy import create_engine

sibiji_path = os.path.expanduser("~/.npcsh/npc_team/sibiji.npc")

if os.path.exists(os.path.expanduser('~/npcsh_history.db')):
    db = create_engine("sqlite:///"+os.path.expanduser('~/npcsh_history.db'))
else: 
    db = None
try:
    if not os.path.exists(sibiji_path):
        sibiji_path = __file__.getparent() / "npc_team/sibiji.npc"
    sibiji = NPC(file = sibiji_path, db_conn = db)
        
except Exception as e:
    print(f"Error finding sibiji.npc: {e}")
    sibiji = NPC(primary_directive='You are sibiji, the master planner for all NPCs and genius of the NPC team', 
                 model='llama3.2', 
                 provider='ollama', )
    

