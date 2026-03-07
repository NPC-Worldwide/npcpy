import os
import yaml
import argparse

from npcpy.npc_compiler import NPC, Team
from npcpy.serve import start_flask_server

def create_simple_team():
    """Create a simple NPC team"""
    
    team_dir = os.path.expanduser("~/.npcsh/simple_team")
    os.makedirs(team_dir, exist_ok=True)
    os.makedirs(os.path.join(team_dir, "jinxes"), exist_ok=True)
    
    print(f"Creating team in: {team_dir}")
    
    
    coordinator_config = {
        "name": "coordinator",
        "primary_directive": "You coordinate tasks and delegate work to team members when needed. Handle general questions yourself.",
        "model": "llama3.2",
        "provider": "ollama"
    }
    
    
    developer_config = {
        "name": "developer",
        "primary_directive": "You are a software developer. Handle all coding and technical tasks directly.",
        "model": "llama3.2",
        "provider": "ollama"
    }
    
    
    writer_config = {
        "name": "writer",
        "primary_directive": "You are a technical writer. Handle all documentation and content creation directly.",
        "model": "llama3.2",
        "provider": "ollama"
    }
    
    configs = [coordinator_config, developer_config, writer_config]
    for config in configs:
        npc_path = os.path.join(team_dir, f"{config['name']}.npc")
        with open(npc_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    
    team_context = {
        "forenpc": "coordinator",
        "team_name": "Simple Team",
        "model": "llama3.2",
        "provider": "ollama"
    }
    
    ctx_path = os.path.join(team_dir, "team.ctx")
    with open(ctx_path, 'w') as f:
        yaml.dump(team_context, f, default_flow_style=False)
    
    
    simple_jinx = {
        "jinx_name": "hello_world",
        "description": "Simple greeting",
        "inputs": [{"name": "Name to greet"}],
        "steps": [
            {
                "name": "greet",
                "engine": "natural",
                "code": "Hello {{name}}! How can the team help you today?"
            }
        ]
    }
    
    jinx_path = os.path.join(team_dir, "jinxes", "hello_world.jinx")
    with open(jinx_path, 'w') as f:
        yaml.dump(simple_jinx, f, default_flow_style=False)
    
    print(f"✅ Created team with {len(configs)} NPCs")
    return team_dir

def create_wsgi_app():
    """WSGI app for gunicorn"""
    from npcpy.serve import app
    return app

def main():
    parser = argparse.ArgumentParser(description='Simple NPC Team Server')
    parser.add_argument('--port', type=int, default=5337)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--create-team', action='store_true')
    parser.add_argument('--gunicorn', action='store_true')
    
    args = parser.parse_args()
    
    if args.create_team:
        create_simple_team()
        print("Team created! Run without --create-team to start server")
        return
    
    if args.gunicorn:
        print(f"gunicorn -w 4 -b 0.0.0.0:{args.port} simple_server_example:create_wsgi_app")
        return
    
    team_dir = os.path.expanduser("~/.npcsh/simple_team")
    if not os.path.exists(team_dir):
        create_simple_team()
    
    try:
        team = Team(team_path=team_dir)
        print(f"✅ Team loaded: {list(team.npcs.keys())}")
        print(f"🚀 Starting server on port {args.port}")
        
        start_flask_server(port=args.port, debug=args.debug)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()