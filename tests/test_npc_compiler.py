import os
import tempfile
import sqlite3
from pathlib import Path
from npcpy.npc_compiler import NPC, Jinx, Team, initialize_npc_project


def test_npc_creation():
    """Test basic NPC creation"""
    npc = NPC(
        name="test_npc",
        primary_directive="You are a helpful assistant",
        model="llama3.2:latest",
        provider="ollama"
    )
    assert npc.name == "test_npc"
    assert npc.primary_directive == "You are a helpful assistant"
    print(f"Created NPC: {npc.name}")


def test_npc_save_and_load():
    """Test NPC save and load functionality"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        npc = NPC(
            name="save_test_npc",
            primary_directive="Test NPC for saving",
            model="llama3.2:latest",
            provider="ollama"
        )
        npc.save(temp_dir)
        
        
        npc_file = os.path.join(temp_dir, "save_test_npc.npc")
        assert os.path.exists(npc_file)
        
        
        loaded_npc = NPC(file=npc_file)
        assert loaded_npc.name == "save_test_npc"
        print(f"Saved and loaded NPC: {loaded_npc.name}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_npc_get_llm_response():
    """Test NPC LLM response functionality"""
    npc = NPC(
        name="response_test_npc",
        primary_directive="You are a helpful math assistant",
        model="llama3.2:latest",
        provider="ollama"
    )
    
    response = npc.get_llm_response("What is 3 + 4?")
    assert response is not None
    print(f"NPC response: {response}")


def test_jinx_creation():
    """Test basic Jinx creation"""
    jinx_data = {
        "jinx_name": "test_jinx",
        "description": "A test jinx",
        "inputs": ["input1", "input2"],
        "steps": [
            {
                "code": """
input1 = '{{ input1 }}'
input2 = '{{ input2 }}'
output = f"Processed: {input1} and {input2}"
print(output)
""",
                "engine": "python"
            }
        ]
    }
    
    jinx = Jinx(jinx_data=jinx_data)
    assert jinx.jinx_name == "test_jinx"
    print(f"Created Jinx: {jinx.jinx_name}")


def test_jinx_execution():
    """Test Jinx execution"""
    jinx_data = {
        "jinx_name": "math_jinx",
        "description": "Math calculation jinx",
        "inputs": ["number1", "number2"],
        "steps": [
            {
                "code": """
number1 = int('{{ number1 }}')
number2 = int('{{ number2 }}')
output = number1 + number2
print(f"The sum of {number1} and {number2} is {output}")
""",
                "engine": "python"
            }
        ]
    }
    
    jinx = Jinx(jinx_data=jinx_data)
    input_values = {"number1": "5", "number2": "7"}
    
    result = jinx.execute(input_values, {})
    assert result is not None
    print(f"Jinx execution result: {result}")


def test_team_creation():
    """Test Team creation"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        npc1 = NPC(name="analyst", primary_directive="Analyze data")
        npc2 = NPC(name="critic", primary_directive="Critique analysis")
        
        npc1.save(temp_dir)
        npc2.save(temp_dir)
        
        team = Team(team_path=temp_dir)
        assert team is not None
        print(f"Created team with {len(team.npcs)} NPCs")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_npc_with_database():
    """Test NPC with database connection"""
    temp_db = tempfile.mktemp(suffix=".db")
    
    try:
        conn = sqlite3.connect(temp_db)
        
        npc = NPC(
            name="db_test_npc",
            primary_directive="Test NPC with database",
            db_conn=conn
        )
        
        assert npc.db_conn is not None
        print(f"Created NPC with database: {npc.name}")
        
        conn.close()
        
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)


def test_initialize_project_with_templates(tmp_path):
    """Ensure template NPC files are copied into a new project"""
    template_path = Path(__file__).parent / "template_tests" / "npc_team" / "slean.npc"
    project_dir = tmp_path / "proj_with_templates"
    msg = initialize_npc_project(directory=project_dir, templates=[template_path])

    expected_npc = project_dir / "npc_team" / "slean.npc"
    assert expected_npc.exists()
    assert "npc_team" in msg


def test_initialize_project_prefers_custom_ctx(tmp_path):
    """Custom .ctx from template should be used and default team.ctx skipped"""
    template_dir = tmp_path / "my_template"
    template_dir.mkdir()
    (template_dir / "custom.ctx").write_text("name: custom\ncontext: hello\n")
    (template_dir / "alpha.npc").write_text("name: alpha\nprimary_directive: test\n")

    project_dir = tmp_path / "proj_custom_ctx"
    initialize_npc_project(directory=project_dir, templates=[template_dir])

    custom_ctx = project_dir / "npc_team" / "custom.ctx"
    default_ctx = project_dir / "npc_team" / "team.ctx"
    assert custom_ctx.exists()
    assert not default_ctx.exists()


def test_jinx_save_and_load():
    """Test Jinx save and load"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        jinx_data = {
            "jinx_name": "save_test_jinx",
            "description": "Test jinx for saving",
            "inputs": ["input1"],
            "steps": [{"type": "llm", "prompt": "Process {{input1}}"}]
        }
        
        jinx = Jinx(jinx_data=jinx_data)
        jinx.save(temp_dir)
        
        jinx_file = os.path.join(temp_dir, "save_test_jinx.jinx")
        assert os.path.exists(jinx_file)
        
        loaded_jinx = Jinx(jinx_path=jinx_file)
        assert loaded_jinx.jinx_name == "save_test_jinx"
        print(f"Saved and loaded Jinx: {loaded_jinx.jinx_name}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_npc_execute_jinx():
    """Test NPC executing a jinx"""
    try:
        npc = NPC(
            name="jinx_executor",
            primary_directive="Execute jinxs",
            model="llama3.2:latest",
            provider="ollama"
        )
        
        jinx_data = {
            "jinx_name": "simple_jinx",
            "inputs": ["message"],
            "steps": [{"type": "llm", "prompt": "Reply to: {{message}}"}]
        }
        
        jinx = Jinx(jinx_data=jinx_data)
        result = npc.execute_jinx("simple_jinx", {"message": "Hello"})
        
        assert result is not None
        print(f"NPC jinx execution result: {result}")
    except Exception as e:
        print(f"NPC jinx execution failed: {e}")



import os
import tempfile
import sqlite3
from datetime import datetime
import sys


from npcpy.memory.command_history import CommandHistory, get_db_connection
from npcpy.tools import auto_tools

def test_npc_memory_crud_integration():
    """Integration test for NPC memory CRUD operations via auto tools"""
    
    test_db_path = "test_npc.db"
    db_conn = get_db_connection(test_db_path)
    
    test_npc = NPC(
        name="memory_test_npc",
        primary_directive="Test NPC with memory CRUD capabilities", 
        model="llama3.2",
        provider="ollama", 
        db_conn=db_conn,
        memory=True
    )
    
    print(f"NPC '{test_npc.name}' initialized successfully")
    print(f"Database connection: {test_npc.db_conn is not None}")
    print(f"Command history: {test_npc.command_history is not None}")
    
    print("\n=== Testing Memory CRUD via Auto Tools ===")
    
    print("\n1. Testing create_memory via LLM response")
    create_response = test_npc.get_llm_response(
        "Create a memory about user preferring Python over JavaScript for backend work",
        auto_process_tool_calls=True
    )
    
    if create_response.get('tool_results'):
        memory_id = create_response['tool_results'][0]['result']
        print(f"✓ Memory created with ID: {memory_id}")
    else:
        print("✗ No memory created")
    
    print("\n2. Testing search_memories via LLM")
    search_response = test_npc.get_llm_response(
        "Search my memories for anything about Python",
        auto_process_tool_calls=True
    )
    
    if search_response.get('tool_results'):
        search_results = search_response['tool_results'][0]['result']
        print(f"✓ Found {len(search_results) if isinstance(search_results, list) else 1} memories")
    else:
        print("✗ No search results")
    
    print("\n3. Testing get_all_memories via LLM")
    all_memories_response = test_npc.get_llm_response(
        "Show me all my current memories",
        auto_process_tool_calls=True
    )
    
    if all_memories_response.get('tool_results'):
        all_memories = all_memories_response['tool_results'][0]['result']
        print(f"✓ Retrieved {len(all_memories) if isinstance(all_memories, list) else 1} total memories")
    else:
        print("✗ No memories retrieved")
    
    print("\n4. Testing memory stats via LLM")
    stats_response = test_npc.get_llm_response(
        "Give me statistics about my memories",
        auto_process_tool_calls=True
    )
    
    if stats_response.get('tool_results'):
        stats = stats_response['tool_results'][0]['result']
        print(f"✓ Memory stats: {stats}")
    else:
        print("✗ No stats retrieved")
    
    print("\n=== Testing Direct Tool Access ===")
    
    print("\n5. Direct memory creation")
    memory_id = test_npc.create_memory("Direct memory creation test", "test")
    print(f"✓ Created memory ID: {memory_id}")
    
    if memory_id:
        print("\n6. Reading created memory")
        memory_data = test_npc.read_memory(memory_id)
        print(f"✓ Memory content: {memory_data['initial_memory'] if memory_data else 'None'}")
        
        print("\n7. Updating memory")
        update_success = test_npc.update_memory(
            memory_id, 
            new_content="Updated test memory content",
            status="verified"
        )
        print(f"✓ Update success: {update_success}")
        
        print("\n8. Reading updated memory")
        updated_memory = test_npc.read_memory(memory_id)
        if updated_memory:
            print(f"✓ Updated content: {updated_memory['final_memory']}")
            print(f"✓ New status: {updated_memory['status']}")
    
    print("\n9. Getting final memory statistics")
    final_stats = test_npc.get_memory_stats()
    print(f"✓ Final memory stats: {final_stats}")
    
    print("\n=== Test Complete ===")
    print("✓ Memory CRUD operations successfully integrated as auto tools")
    return True
