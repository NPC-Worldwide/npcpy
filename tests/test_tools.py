"""Test suite for auto_tools functionality."""

import os
import json
import pytest
from npcpy.tools import auto_tools


# =============================================================================
# Basic Tool Function Tests
# =============================================================================

def get_weather(location: str) -> str:
    """Get weather information for a location"""
    return f"The weather in {location} is sunny and 75°F"


def calculate_math(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except:
        return "Invalid mathematical expression"


def process_data(data: list, operation: str = "sum") -> float:
    """
    Process a list of numbers with various operations.

    Args:
        data: List of numbers to process
        operation: Operation to perform ('sum', 'avg', 'max', 'min')
    """
    if operation == "sum":
        return sum(data)
    elif operation == "avg":
        return sum(data) / len(data) if data else 0
    elif operation == "max":
        return max(data) if data else 0
    elif operation == "min":
        return min(data) if data else 0
    else:
        return 0


class TestAutoToolsBasic:
    """Test basic auto_tools functionality."""

    def test_auto_tools_generates_schema(self):
        """Test that auto_tools generates proper schema from functions."""
        tools_schema, tool_map = auto_tools([get_weather, calculate_math, process_data])

        assert len(tools_schema) == 3
        assert len(tool_map) == 3
        assert "get_weather" in tool_map
        assert "calculate_math" in tool_map
        assert "process_data" in tool_map

    def test_auto_tools_schema_structure(self):
        """Test that generated schema has correct structure."""
        tools_schema, _ = auto_tools([get_weather])

        assert len(tools_schema) == 1
        tool = tools_schema[0]

        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert tool["function"]["name"] == "get_weather"

    def test_auto_tools_function_calls_work(self):
        """Test that tool_map functions are callable."""
        _, tool_map = auto_tools([get_weather, calculate_math, process_data])

        # Test get_weather
        weather_result = tool_map["get_weather"]("Tokyo")
        assert "Tokyo" in weather_result
        assert "sunny" in weather_result.lower() or "weather" in weather_result.lower()

        # Test calculate_math
        math_result = tool_map["calculate_math"]("5 * 7")
        assert "35" in math_result

        # Test process_data
        data_result = tool_map["process_data"]([1, 2, 3, 4, 5], "avg")
        assert data_result == 3.0

    def test_auto_tools_with_single_function(self):
        """Test auto_tools with a single function."""
        tools_schema, tool_map = auto_tools([get_weather])

        assert len(tools_schema) == 1
        assert len(tool_map) == 1

    def test_auto_tools_with_empty_list(self):
        """Test auto_tools with empty function list."""
        tools_schema, tool_map = auto_tools([])

        assert len(tools_schema) == 0
        assert len(tool_map) == 0


class TestAutoToolsAdvanced:
    """Test advanced auto_tools functionality with complex functions."""

    def test_function_with_default_args(self):
        """Test that functions with default arguments work correctly."""
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet a person with optional greeting."""
            return f"{greeting}, {name}!"

        tools_schema, tool_map = auto_tools([greet])

        # Test with default
        result = tool_map["greet"]("Alice")
        assert "Hello" in result
        assert "Alice" in result

        # Test with custom greeting
        result = tool_map["greet"]("Bob", "Hi")
        assert "Hi" in result
        assert "Bob" in result

    def test_function_with_list_param(self):
        """Test that functions with list parameters work correctly."""
        tools_schema, tool_map = auto_tools([process_data])

        # Check schema has correct type for list parameter
        func_schema = tools_schema[0]["function"]
        assert "parameters" in func_schema

    def test_function_with_docstring_parsing(self):
        """Test that docstrings are properly parsed."""
        tools_schema, _ = auto_tools([process_data])

        func_schema = tools_schema[0]["function"]
        assert "description" in func_schema
        assert len(func_schema["description"]) > 0


class TestAutoToolsWithFileOperations:
    """Test auto_tools with file operation functions."""

    def test_file_operation_tools(self):
        """Test auto_tools with file operation functions."""
        def list_files(directory: str = ".") -> list:
            """List all files in a directory."""
            return os.listdir(directory)

        def read_file(filepath: str) -> str:
            """Read and return the contents of a file."""
            with open(filepath, 'r') as f:
                return f.read()

        tools_schema, tool_map = auto_tools([list_files, read_file])

        assert len(tools_schema) == 2
        assert "list_files" in tool_map
        assert "read_file" in tool_map

        # Test list_files works
        files = tool_map["list_files"](".")
        assert isinstance(files, list)


class TestAutoToolsSchemaValidation:
    """Test that generated schemas are valid for LLM tool calling."""

    def test_schema_has_required_fields(self):
        """Test that schema has all required fields for OpenAI-style tool calling."""
        def sample_tool(arg1: str, arg2: int = 10) -> str:
            """A sample tool for testing."""
            return f"{arg1}: {arg2}"

        tools_schema, _ = auto_tools([sample_tool])
        tool = tools_schema[0]

        # Check top-level structure
        assert "type" in tool
        assert "function" in tool

        # Check function structure
        func = tool["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func

        # Check parameters structure
        params = func["parameters"]
        assert "type" in params
        assert params["type"] == "object"
        assert "properties" in params

    def test_schema_json_serializable(self):
        """Test that generated schema is JSON serializable."""
        tools_schema, _ = auto_tools([get_weather, calculate_math, process_data])

        # Should not raise
        json_str = json.dumps(tools_schema)
        assert len(json_str) > 0

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert len(parsed) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
