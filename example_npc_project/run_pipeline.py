import os
import sys
import pandas as pd
import yaml

# Add the npcpy package to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from npcpy.npc_compiler import NPC, Team
from npcpy.sql.sql_model_compiler import create_model_compiler

def load_npc_from_file(npc_file_path):
    """
    Load an NPC from a YAML file
    """
    with open(npc_file_path, 'r') as f:
        npc_config = yaml.safe_load(f)
    
    return NPC(
        name=npc_config['name'],
        primary_directive=npc_config['primary_directive'],
        model=npc_config.get('model', 'mistral-large2'),
        provider=npc_config.get('provider', 'ollama')
    )

def create_seed_data():
    """
    Create sample seed data for SQL models
    """
    return {
        'customers': pd.DataFrame({
            'customer_id': range(1, 1001),
            'days_since_last_purchase': pd.np.random.randint(0, 120, 1000),
            'total_purchases': pd.np.random.randint(1, 50, 1000),
            'avg_order_value': pd.np.random.uniform(10, 500, 1000)
        })
    }

def main():
    # Set up project directories
    project_root = os.path.dirname(os.path.abspath(__file__))
    npc_team_dir = os.path.join(project_root, 'npc_team')
    sql_models_dir = os.path.join(project_root, 'sql_models')

    # Load Team Context
    with open(os.path.join(npc_team_dir, 'team.ctx'), 'r') as f:
        team_config = yaml.safe_load(f)

    # Create NPCs
    npcs = []
    for npc_file in os.listdir(npc_team_dir):
        if npc_file.endswith('.npc'):
            npc_path = os.path.join(npc_team_dir, npc_file)
            npcs.append(load_npc_from_file(npc_path))

    # Create Team
    team = Team(
        npcs=npcs,
        team_path=npc_team_dir
    )

    # Create SQL Model Compiler 
    # You can change engine_type to 'snowflake' or 'bigquery'
    model_compiler = create_model_compiler(
        models_dir=sql_models_dir, 
        engine_type='sqlite'
    )

    # Create seed data
    seed_data = create_seed_data()

    # Run all models
    model_results = model_compiler.run_all_models(seed_data)

    # Optional: Use NPC to analyze results
    data_analyst = team.get_npc('data_analyst')
    for model_name, result in model_results.items():
        print(f"\nAnalyzing {model_name} results:")
        analysis = data_analyst.get_llm_response(
            f"Analyze the following SQL model results for {model_name}:\n{result.to_string()}"
        )
        print(analysis.get('response', 'No analysis available'))

if __name__ == '__main__':
    main()