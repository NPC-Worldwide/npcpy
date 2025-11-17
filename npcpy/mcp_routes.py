import os
import yaml
from flask import Blueprint, request, jsonify
from npcsh.corca import MCPClientNPC

mcp_bp = Blueprint('mcp_routes', __name__)

MCP_SERVERS_CONFIG = os.path.expanduser("~/.npcsh/mcp_servers.yml")

def load_mcp_servers():
    if not os.path.exists(MCP_SERVERS_CONFIG):
        return []
    with open(MCP_SERVERS_CONFIG, 'r') as f:
        return yaml.safe_load(f) or []

def save_mcp_servers(servers):
    os.makedirs(os.path.dirname(MCP_SERVERS_CONFIG), exist_ok=True)
    with open(MCP_SERVERS_CONFIG, 'w') as f:
        yaml.safe_dump(servers, f)

@mcp_bp.route('/servers', methods=['GET'])
def get_mcp_servers():
    """
    Get the list of saved MCP servers.
    """
    try:
        servers = load_mcp_servers()
        return jsonify({"servers": servers, "error": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@mcp_bp.route('/servers', methods=['POST'])
def save_mcp_server():
    """
    Save a new or updated MCP server configuration.
    """
    try:
        server_data = request.json
        servers = load_mcp_servers()
        
        # Check if server exists and update it, otherwise add it
        found = False
        for i, server in enumerate(servers):
            if server.get('name') == server_data.get('name'):
                servers[i] = server_data
                found = True
                break
        if not found:
            servers.append(server_data)
            
        save_mcp_servers(servers)
        return jsonify({"status": "success", "servers": servers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@mcp_bp.route('/test', methods=['POST'])
def test_mcp_server():
    """
    Test the connection to an MCP server and get its available tools.
    """
    server_config = request.json
    server_path = server_config.get("url") # Assuming url is the path to the script
    if not server_path:
        return jsonify({"error": "Server URL/path is required."}), 400

    client = None
    try:
        client = MCPClientNPC()
        if client.connect_sync(server_path):
            tools = client.available_tools_llm
            return jsonify({"status": "success", "tools": tools})
        else:
            return jsonify({"status": "error", "error": "Failed to connect to MCP server."})
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})
    finally:
        if client:
            client.disconnect_sync()
