"""
Test suite for documentation examples.

These tests verify that the examples in README.md and docs/ work correctly.
Tests are organized by category and can be run with or without LLM backends.

To run all tests:
    pytest tests/test_documentation_examples.py -v

To run only unit tests (no LLM required):
    pytest tests/test_documentation_examples.py -v -m "not requires_ollama"

To run integration tests with Ollama:
    pytest tests/test_documentation_examples.py -v -m "requires_ollama"
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path

# Mark for tests that require Ollama
requires_ollama = pytest.mark.requires_ollama


# =============================================================================
# Unit Tests (No LLM Required)
# =============================================================================

class TestNPCCreation:
    """Test NPC creation without LLM calls."""

    def test_basic_npc_creation(self):
        """Test that NPC can be created with basic parameters."""
        from npcpy.npc_compiler import NPC

        npc = NPC(
            name='Test Agent',
            primary_directive='You are a helpful test assistant.',
            model='llama3.2',
            provider='ollama'
        )

        assert npc.name == 'Test Agent'
        assert npc.primary_directive == 'You are a helpful test assistant.'
        assert npc.model == 'llama3.2'
        assert npc.provider == 'ollama'

    def test_npc_with_tools(self):
        """Test that NPC can be created with custom tools."""
        from npcpy.npc_compiler import NPC

        def sample_tool(x: int) -> int:
            """A sample tool that doubles the input."""
            return x * 2

        npc = NPC(
            name='Tool Agent',
            primary_directive='You can use tools.',
            model='llama3.2',
            provider='ollama',
            tools=[sample_tool]
        )

        assert npc.name == 'Tool Agent'
        assert len(npc.tools) > 0


class TestJinxCreation:
    """Test Jinx creation without LLM calls."""

    def test_basic_jinx_creation(self):
        """Test that Jinx can be created with jinx_data."""
        from npcpy.npc_compiler import Jinx

        jinx = Jinx(jinx_data={
            "jinx_name": "test_jinx",
            "description": "A test jinx",
            "inputs": ["input1"],
            "steps": [
                {
                    "name": "step1",
                    "engine": "python",
                    "code": "output = '{{ input1 }}'"
                }
            ]
        })

        assert jinx.jinx_name == "test_jinx"
        assert jinx.description == "A test jinx"
        assert "input1" in jinx.inputs

    def test_jinx_python_step_execution(self):
        """Test Jinx with Python engine step."""
        from npcpy.npc_compiler import Jinx

        jinx = Jinx(jinx_data={
            "jinx_name": "math_jinx",
            "description": "Math calculation jinx",
            "inputs": ["number1", "number2"],
            "steps": [
                {
                    "name": "calculate",
                    "engine": "python",
                    "code": """
number1 = int('{{ number1 }}')
number2 = int('{{ number2 }}')
output = number1 + number2
"""
                }
            ]
        })

        result = jinx.execute(
            input_values={"number1": "5", "number2": "7"}
        )

        assert result is not None
        # The result should contain the output from the calculate step
        output = result.get('output') or result.get('calculate')
        assert output == 12 or str(output) == '12'


class TestTeamCreation:
    """Test Team creation without LLM calls."""

    def test_basic_team_creation(self):
        """Test that Team can be created with NPCs."""
        from npcpy.npc_compiler import NPC, Team

        npc1 = NPC(
            name='Agent 1',
            primary_directive='First agent',
            model='llama3.2',
            provider='ollama'
        )

        npc2 = NPC(
            name='Agent 2',
            primary_directive='Second agent',
            model='llama3.2',
            provider='ollama'
        )

        coordinator = NPC(
            name='Coordinator',
            primary_directive='Coordinate the team',
            model='llama3.2',
            provider='ollama'
        )

        team = Team(
            npcs=[npc1, npc2],
            forenpc=coordinator
        )

        assert team is not None
        assert 'Agent 1' in team.npcs
        assert 'Agent 2' in team.npcs
        assert team.forenpc.name == 'Coordinator'


class TestAutoTools:
    """Test auto_tools functionality."""

    def test_auto_tools_generation(self):
        """Test that auto_tools generates correct schema."""
        from npcpy.tools import auto_tools

        def list_files(directory: str = ".") -> list:
            """List all files in a directory."""
            return os.listdir(directory)

        def read_file(filepath: str) -> str:
            """Read and return the contents of a file."""
            with open(filepath, 'r') as f:
                return f.read()

        tools_schema, tool_map = auto_tools([list_files, read_file])

        assert len(tools_schema) == 2
        assert 'list_files' in tool_map
        assert 'read_file' in tool_map

        # Verify schema structure
        for tool in tools_schema:
            assert 'type' in tool
            assert 'function' in tool
            assert 'name' in tool['function']
            assert 'description' in tool['function']


class TestNPCSaveLoad:
    """Test NPC save and load functionality."""

    def test_npc_save_and_load(self):
        """Test that NPC can be saved and loaded."""
        from npcpy.npc_compiler import NPC

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save
            npc = NPC(
                name='save_test_npc',
                primary_directive='Test NPC for saving',
                model='llama3.2',
                provider='ollama'
            )
            npc.save(temp_dir)

            # Verify file exists
            npc_file = os.path.join(temp_dir, "save_test_npc.npc")
            assert os.path.exists(npc_file)

            # Load and verify
            loaded_npc = NPC(file=npc_file)
            assert loaded_npc.name == "save_test_npc"
            assert loaded_npc.primary_directive == "Test NPC for saving"


class TestJinxSaveLoad:
    """Test Jinx save and load functionality."""

    def test_jinx_save_and_load(self):
        """Test that Jinx can be saved and loaded."""
        from npcpy.npc_compiler import Jinx

        with tempfile.TemporaryDirectory() as temp_dir:
            jinx = Jinx(jinx_data={
                "jinx_name": "save_test_jinx",
                "description": "Test jinx for saving",
                "inputs": ["input1"],
                "steps": [{"name": "step1", "engine": "python", "code": "output = 1"}]
            })
            jinx.save(temp_dir)

            jinx_file = os.path.join(temp_dir, "save_test_jinx.jinx")
            assert os.path.exists(jinx_file)

            loaded_jinx = Jinx(jinx_path=jinx_file)
            assert loaded_jinx.jinx_name == "save_test_jinx"


# =============================================================================
# Integration Tests (Require Ollama)
# =============================================================================

@requires_ollama
class TestNPCResponses:
    """Test NPC LLM response functionality."""

    def test_basic_npc_response(self):
        """Test basic NPC response (Simon Bolivar example from README)."""
        from npcpy.npc_compiler import NPC

        simon = NPC(
            name='Simon Bolivar',
            primary_directive='Liberate South America from the Spanish Royalists.',
            model='gemma3:4b',
            provider='ollama'
        )
        response = simon.get_llm_response(
            "What is the most important territory to retain in the Andes mountains?"
        )

        assert response is not None
        assert 'response' in response
        assert response['response'] is not None
        assert len(response['response']) > 0

    def test_npc_with_tools_response(self):
        """Test NPC with tools (list_files, read_file example from README)."""
        from npcpy.npc_compiler import NPC

        def list_files(directory: str = ".") -> list:
            """List all files in a directory."""
            return os.listdir(directory)

        def read_file(filepath: str) -> str:
            """Read and return the contents of a file."""
            with open(filepath, 'r') as f:
                return f.read()

        assistant = NPC(
            name='File Assistant',
            primary_directive='You are a helpful assistant who can list and read files.',
            model='llama3.2',
            provider='ollama',
            tools=[list_files, read_file],
        )

        response = assistant.get_llm_response(
            "List the files in the current directory.",
            auto_process_tool_calls=True,
        )

        assert response is not None
        assert 'tool_results' in response or 'response' in response


@requires_ollama
class TestLLMFuncs:
    """Test llm_funcs module."""

    def test_get_llm_response_basic(self):
        """Test basic get_llm_response without NPC."""
        from npcpy.llm_funcs import get_llm_response

        response = get_llm_response(
            "What is 2+2? Answer only with the number.",
            model='qwen3:4b',
            provider='ollama'
        )

        assert response is not None
        assert 'response' in response
        assert response['response'] is not None

    def test_get_llm_response_with_npc(self):
        """Test get_llm_response with NPC parameter."""
        from npcpy.npc_compiler import NPC
        from npcpy.llm_funcs import get_llm_response

        simon = NPC(
            name='Simon Bolivar',
            primary_directive='Liberate South America from the Spanish Royalists.',
            model='gemma3:4b',
            provider='ollama'
        )
        response = get_llm_response(
            "Who was the mythological chilean bird that guides lucky visitors to gold?",
            npc=simon
        )

        assert response is not None
        assert 'response' in response

    def test_streaming_response(self):
        """Test streaming LLM response."""
        from npcpy.llm_funcs import get_llm_response
        from npcpy.npc_sysenv import print_and_process_stream

        response = get_llm_response(
            "Count from 1 to 3",
            model='qwen3:4b',
            provider='ollama',
            stream=True
        )

        assert response is not None
        assert 'response' in response
        # Response should be a generator when streaming
        assert hasattr(response['response'], '__iter__')

    def test_json_format_response(self):
        """Test JSON format response."""
        from npcpy.llm_funcs import get_llm_response

        response = get_llm_response(
            "Return a json object with 'answer' as the key and 42 as the value",
            model='qwen3:4b',
            provider='ollama',
            format='json'
        )

        assert response is not None
        assert 'response' in response
        # Response should be a dict when format='json'
        assert isinstance(response['response'], dict)

    def test_get_llm_response_with_messages(self):
        """Test get_llm_response with conversation history."""
        from npcpy.llm_funcs import get_llm_response

        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        ]

        response = get_llm_response(
            prompt="What is my name?",
            messages=messages,
            model='llama3.2',
            provider='ollama'
        )

        assert response is not None
        assert 'response' in response


@requires_ollama
class TestTeamOrchestration:
    """Test Team orchestration functionality."""

    def test_team_simple_orchestration(self):
        """Test team orchestration with a simple query."""
        from npcpy.npc_compiler import NPC, Team

        npc1 = NPC(
            name='Analyst',
            primary_directive='You analyze information.',
            model='gemma3:4b',
            provider='ollama'
        )

        coordinator = NPC(
            name='Coordinator',
            primary_directive='You coordinate and summarize.',
            model='qwen3:4b',
            provider='ollama'
        )

        team = Team(
            npcs=[npc1],
            forenpc=coordinator
        )

        result = team.orchestrate("What is 2+2?")

        assert result is not None
        assert 'debrief' in result or 'output' in result


# =============================================================================
# Test Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "requires_ollama: mark test as requiring Ollama backend"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
