"""
Tool utilities for automatic schema generation from Python functions.
"""

import inspect
import json
from typing import Any, Dict, List, Callable, Union, get_type_hints, get_origin, get_args

from docstring_parser import parse as parse_docstring

def python_type_to_json_schema(py_type: type) -> Dict[str, Any]:
    """Convert Python type hints to JSON schema types."""
    
    if get_origin(py_type) is Union:
        args = get_args(py_type)
        
        if len(args) == 2 and type(None) in args:
            non_none_type = args[0] if args[1] is type(None) else args[1]
            return python_type_to_json_schema(non_none_type)
        
        return python_type_to_json_schema(args[0])
    
    
    if get_origin(py_type) is list:
        item_type = get_args(py_type)[0] if get_args(py_type) else str
        return {
            "type": "array",
            "items": python_type_to_json_schema(item_type)
        }
    
    
    if get_origin(py_type) is dict:
        return {"type": "object"}
    
    
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }
    
    return type_mapping.get(py_type, {"type": "string"})

def extract_function_info(func: Callable) -> Dict[str, Any]:
    """Extract function information including name, description, and parameters."""
    
    sig = inspect.signature(func)
    
    
    try:
        type_hints = get_type_hints(func)
    except Exception:
        type_hints = {}
    
    
    docstring = inspect.getdoc(func)
    parsed_doc = None
    if docstring:
        try:
            parsed_doc = parse_docstring(docstring)
        except Exception:
            pass
    
    
    func_name = func.__name__
    description = ""
    
    if parsed_doc and hasattr(parsed_doc, 'short_description') and parsed_doc.short_description:
        description = parsed_doc.short_description
        if hasattr(parsed_doc, 'long_description') and parsed_doc.long_description:
            description += f". {parsed_doc.long_description}"
    elif docstring:
        
        description = docstring.split('\n')[0].strip()
    
    
    properties = {}
    required = []
    param_descriptions = {}
    
    
    if parsed_doc and hasattr(parsed_doc, 'params'):
        for param in parsed_doc.params:
            param_descriptions[param.arg_name] = param.description or ""
    
    for param_name, param in sig.parameters.items():
        
        if param_name == 'self':
            continue
            
        
        param_type = type_hints.get(param_name, str)
        
        
        param_schema = python_type_to_json_schema(param_type)
        
        
        if param_name in param_descriptions:
            param_schema["description"] = param_descriptions[param_name]
        else:
            param_schema["description"] = f"The {param_name} parameter"
        
        properties[param_name] = param_schema
        
        
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    
    return {
        "name": func_name,
        "description": description or f"Call the {func_name} function",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }

def create_tool_schema(functions: List[Callable]) -> List[Dict[str, Any]]:
    """Create OpenAI-style tool schema from a list of functions."""
    schema = []
    
    for func in functions:
        func_info = extract_function_info(func)
        schema.append({
            "type": "function",
            "function": func_info
        })
    
    return schema

def create_tool_map(functions: List[Callable]) -> Dict[str, Callable]:
    """Create a tool map from a list of functions."""
    return {func.__name__: func for func in functions}

def auto_tools(functions: List[Callable]) -> tuple[List[Dict[str, Any]], Dict[str, Callable]]:
    """
    Automatically create both tool schema and tool map from functions.
    
    Args:
        functions: List of Python functions to convert to tools
        
    Returns:
        Tuple of (tools_schema, tool_map)
        
    Example:
        ```python
        def get_weather(location: str) -> str:
            '''Get weather information for a location'''
            return f"The weather in {location} is sunny and 75°F"
            
        def calculate_math(expression: str) -> str:
            '''Calculate a mathematical expression'''
            try:
                result = eval(expression)
                return f"The result of {expression} is {result}"
            except Exception:
                return "Invalid mathematical expression"
        
        
        tools_schema, tool_map = auto_tools([get_weather, calculate_math])
        
        
        response = get_llm_response(
            "What's the weather in Paris and what's 15 * 23?",
            model='gpt-4o-mini',
            provider='openai',
            tools=tools_schema,
            tool_map=tool_map
        )
        ```
    """
    schema = create_tool_schema(functions)
    tool_map = create_tool_map(functions)
    return schema, tool_map
