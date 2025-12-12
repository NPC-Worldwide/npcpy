"""
Build functions for NPC team deployment artifacts.
"""
import os
import yaml
from pathlib import Path


def get_team_name(team_path):
    """Get team name from ctx file or folder name."""
    team_path = Path(team_path)
    for ctx_file in team_path.glob("*.ctx"):
        try:
            with open(ctx_file, 'r') as f:
                ctx = yaml.safe_load(f)
            if ctx and ctx.get('name'):
                return ctx['name']
        except:
            pass

    name = team_path.name
    if name == 'npc_team':
        name = team_path.parent.name
    return name


def build_dockerfile(config, **kwargs):
    """Generate a Dockerfile for serving an NPC team."""
    team_path = config.get('team_path', './npc_team')
    port = config.get('port', 5337)
    team_name = get_team_name(team_path)

    dockerfile = f'''FROM python:3.11-slim

WORKDIR /app

# Install npcsh
RUN pip install --no-cache-dir npcsh

# Copy the NPC team
COPY {os.path.basename(team_path)} /app/npc_team

# Expose the port
EXPOSE {port}

# Set environment variables (override at runtime)
ENV NPCSH_CHAT_MODEL=gpt-4o-mini
ENV NPCSH_CHAT_PROVIDER=openai
ENV OPENAI_API_KEY=""
ENV ANTHROPIC_API_KEY=""

# Run the serve command
CMD ["npc", "serve", "--port", "{port}"]
'''
    return dockerfile


def build_docker_compose(config, **kwargs):
    """Generate Docker Compose setup for NPC team deployment."""
    team_path = config.get('team_path', './npc_team')
    output_dir = config.get('output_dir', './build')
    port = config.get('port', 5337)
    cors_origins = config.get('cors_origins', None)

    team_name = get_team_name(team_path)
    safe_name = team_name.lower().replace(' ', '_').replace('-', '_')

    os.makedirs(output_dir, exist_ok=True)

    # Generate Dockerfile
    dockerfile_content = build_dockerfile(config)
    dockerfile_path = os.path.join(output_dir, 'Dockerfile')
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)

    # Generate docker-compose.yml
    cors_env = f'\n      - CORS_ORIGINS={",".join(cors_origins)}' if cors_origins else ''

    compose = f'''version: '3.8'

services:
  {safe_name}:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - NPCSH_CHAT_MODEL=${{NPCSH_CHAT_MODEL:-gpt-4o-mini}}
      - NPCSH_CHAT_PROVIDER=${{NPCSH_CHAT_PROVIDER:-openai}}
      - OPENAI_API_KEY=${{OPENAI_API_KEY}}
      - ANTHROPIC_API_KEY=${{ANTHROPIC_API_KEY}}
      - GEMINI_API_KEY=${{GEMINI_API_KEY}}{cors_env}
    volumes:
      - ./{os.path.basename(team_path)}:/app/npc_team
    restart: unless-stopped
'''

    compose_path = os.path.join(output_dir, 'docker-compose.yml')
    with open(compose_path, 'w') as f:
        f.write(compose)

    # Generate .env.example
    env_example = '''# NPC Team Environment Variables
NPCSH_CHAT_MODEL=gpt-4o-mini
NPCSH_CHAT_PROVIDER=openai

# API Keys (set at least one)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
'''

    env_path = os.path.join(output_dir, '.env.example')
    with open(env_path, 'w') as f:
        f.write(env_example)

    # Copy npc_team to output
    import shutil
    dest_team = os.path.join(output_dir, os.path.basename(team_path))
    if os.path.exists(dest_team):
        shutil.rmtree(dest_team)
    shutil.copytree(team_path, dest_team)

    return {
        'output': f'''Docker deployment created in {output_dir}/

Files generated:
  - Dockerfile
  - docker-compose.yml
  - .env.example
  - {os.path.basename(team_path)}/

To deploy:
  cd {output_dir}
  cp .env.example .env
  # Edit .env with your API keys
  docker-compose up -d

API will be available at http://localhost:{port}
''',
        'messages': kwargs.get('messages', [])
    }


def build_flask_server(config, **kwargs):
    """Generate a standalone Flask server script."""
    team_path = config.get('team_path', './npc_team')
    output_dir = config.get('output_dir', './build')
    port = config.get('port', 5337)
    cors_origins = config.get('cors_origins', None)

    os.makedirs(output_dir, exist_ok=True)

    cors_line = f"cors_origins={cors_origins}" if cors_origins else "cors_origins=None"

    server_script = f'''#!/usr/bin/env python3
"""
Auto-generated NPC Team server.
Run with: python server.py
"""
from npcpy.serve import start_flask_server

if __name__ == "__main__":
    start_flask_server(
        port={port},
        {cors_line},
    )
'''

    script_path = os.path.join(output_dir, 'server.py')
    with open(script_path, 'w') as f:
        f.write(server_script)
    os.chmod(script_path, 0o755)

    # Copy npc_team
    import shutil
    dest_team = os.path.join(output_dir, os.path.basename(team_path))
    if os.path.exists(dest_team):
        shutil.rmtree(dest_team)
    shutil.copytree(team_path, dest_team)

    return {
        'output': f'''Flask server created in {output_dir}/

Files generated:
  - server.py
  - {os.path.basename(team_path)}/

To run:
  cd {output_dir}
  pip install npcsh
  python server.py

API will be available at http://localhost:{port}
''',
        'messages': kwargs.get('messages', [])
    }


def build_cli_executable(config, **kwargs):
    """Generate CLI wrapper scripts for team NPCs."""
    team_path = config.get('team_path', './npc_team')
    output_dir = config.get('output_dir', './build')

    team_path = Path(team_path)
    os.makedirs(output_dir, exist_ok=True)

    npc_files = list(team_path.glob("*.npc"))
    scripts = []

    for npc_file in npc_files:
        name = npc_file.stem
        script = f'''#!/usr/bin/env python3
"""CLI wrapper for {name} NPC."""
import sys
from npcsh.npcsh import main
sys.argv[0] = "{name}"
main()
'''
        script_path = os.path.join(output_dir, name)
        with open(script_path, 'w') as f:
            f.write(script)
        os.chmod(script_path, 0o755)
        scripts.append(name)

    return {
        'output': f'''CLI scripts created in {output_dir}/

Scripts: {", ".join(scripts)}

To use, add {output_dir} to your PATH or run directly:
  ./{scripts[0] if scripts else "npc_name"} "your prompt"
''',
        'messages': kwargs.get('messages', [])
    }


def build_static_site(config, **kwargs):
    """Generate static site documentation for the team."""
    team_path = config.get('team_path', './npc_team')
    output_dir = config.get('output_dir', './build')

    team_path = Path(team_path)
    os.makedirs(output_dir, exist_ok=True)

    team_name = get_team_name(team_path)

    # Get NPCs
    npcs = []
    for npc_file in team_path.glob("*.npc"):
        with open(npc_file, 'r') as f:
            npc_data = yaml.safe_load(f)
        npcs.append({
            'name': npc_file.stem,
            'directive': npc_data.get('primary_directive', '')[:200] + '...'
        })

    # Simple HTML page
    npc_list = '\n'.join([
        f'<li><strong>{n["name"]}</strong>: {n["directive"]}</li>'
        for n in npcs
    ])

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{team_name} - NPC Team</title>
    <style>
        body {{ font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        ul {{ line-height: 1.8; }}
    </style>
</head>
<body>
    <h1>{team_name}</h1>
    <h2>Available NPCs</h2>
    <ul>{npc_list}</ul>
</body>
</html>
'''

    html_path = os.path.join(output_dir, 'index.html')
    with open(html_path, 'w') as f:
        f.write(html)

    return {
        'output': f'Static site created at {html_path}',
        'messages': kwargs.get('messages', [])
    }
