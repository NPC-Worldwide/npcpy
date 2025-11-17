"""
Local model discovery for common hosting endpoints (LM Studio, llama.cpp, MLX).
"""
import socket
import requests
import logging
from typing import Dict, List, Tuple


def check_port_available(host: str = "localhost", port: int = 8080, timeout: float = 0.5) -> bool:
    """
    Check if a local port is responding.
    
    Args:
        host: The host to check (default: localhost)
        port: The port to check
        timeout: Connection timeout in seconds
        
    Returns:
        bool: True if port is responding, False otherwise
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.error, ConnectionRefusedError, OSError):
        return False


def discover_lm_studio_models(host: str = "localhost", port: int = 1234, timeout: float = 2.0) -> List[str]:
    """
    Discover models from LM Studio endpoint.
    
    Args:
        host: LM Studio host (default: localhost)
        port: LM Studio port (default: 1234)
        timeout: Request timeout in seconds
        
    Returns:
        List of available model IDs
    """
    try:
        if not check_port_available(host, port, timeout/4):
            return []
            
        response = requests.get(
            f"http://{host}:{port}/v1/models",
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and 'data' in data:
                return [model['id'] for model in data['data'] if 'id' in model]
            
    except Exception as e:
        logging.debug(f"Failed to discover LM Studio models on {host}:{port}: {e}")
    
    return []


def discover_llamacpp_models(host: str = "localhost", port: int = 8080, timeout: float = 2.0) -> List[str]:
    """
    Discover models from llama.cpp server endpoint.
    
    Args:
        host: llama.cpp server host (default: localhost)
        port: llama.cpp server port (default: 8080)
        timeout: Request timeout in seconds
        
    Returns:
        List of available model IDs
    """
    try:
        if not check_port_available(host, port, timeout/4):
            return []
            
        # Try the OpenAI-compatible endpoint first
        response = requests.get(
            f"http://{host}:{port}/v1/models",
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and 'data' in data:
                return [model['id'] for model in data['data'] if 'id' in model]
        
        # Fallback to llama.cpp specific endpoint
        response = requests.get(
            f"http://{host}:{port}/models",
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return [model['id'] for model in data if 'id' in model]
            elif isinstance(data, dict) and 'models' in data:
                return data['models']
                
    except Exception as e:
        logging.debug(f"Failed to discover llama.cpp models on {host}:{port}: {e}")
    
    return []


def discover_mlx_models(host: str = "localhost", port: int = 8080, timeout: float = 2.0) -> List[str]:
    """
    Discover models from MLX server endpoint.
    
    Args:
        host: MLX server host (default: localhost)
        port: MLX server port (default: 8080)
        timeout: Request timeout in seconds
        
    Returns:
        List of available model IDs
    """
    try:
        if not check_port_available(host, port, timeout/4):
            return []
            
        # MLX typically uses OpenAI-compatible API
        response = requests.get(
            f"http://{host}:{port}/v1/models",
            timeout=timeout
        )
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, dict) and 'data' in data:
                return [model['id'] for model in data['data'] if 'id' in model]
                
    except Exception as e:
        logging.debug(f"Failed to discover MLX models on {host}:{port}: {e}")
    
    return []


def discover_local_endpoints(timeout: float = 2.0) -> Dict[str, List[str]]:
    """
    Discover models from common local hosting endpoints.
    
    Args:
        timeout: Request timeout in seconds for each endpoint
        
    Returns:
        Dictionary mapping provider names to lists of discovered models
    """
    discovered = {}
    
    # LM Studio (port 1234)
    lm_studio_models = discover_lm_studio_models(timeout=timeout)
    if lm_studio_models:
        discovered['lm-studio'] = lm_studio_models
        logging.info(f"Discovered {len(lm_studio_models)} models from LM Studio")
    
    # llama.cpp server (port 8080)
    llamacpp_models = discover_llamacpp_models(timeout=timeout)
    if llamacpp_models:
        discovered['llama-cpp'] = llamacpp_models
        logging.info(f"Discovered {len(llamacpp_models)} models from llama.cpp")
    
    # MLX server (port 8080, but check different ports to avoid conflicts)
    # Try common alternative ports for MLX
    mlx_ports = [8000, 8001, 8002]
    for mlx_port in mlx_ports:
        mlx_models = discover_mlx_models(port=mlx_port, timeout=timeout)
        if mlx_models:
            discovered['mlx'] = mlx_models
            logging.info(f"Discovered {len(mlx_models)} models from MLX on port {mlx_port}")
            break
    
    return discovered