import os
import sys

from npcpy.npc_compiler import NPC, Team
import tempfile
import yaml

def test_full_orchestration():
    """Test complete team orchestration"""
    print("=== Full Team Orchestration Test ===")
    
    
    coordinator = NPC(
        name="coordinator",
        primary_directive="""You are a project coordinator who manages workflows. 
        You should ONLY pass tasks to other team members if they require very specific 
        technical expertise you don't have. For general coordination and simple document 
        tasks, handle them yourself.""",
        model="llama3.2",
        provider="ollama"
    )
    
    writer = NPC(
        name="writer", 
        primary_directive="""You are a content writer who creates documents and content. 
        Handle all writing tasks directly - create documents, reports, and content as requested. 
        Only pass tasks if they require technical analysis or data processing.""",
        model="llama3.2",
        provider="ollama"
    )
    
    analyst = NPC(
        name="analyst",
        primary_directive="""You are a data analyst who analyzes data and creates insights.
        Handle all data analysis, statistics, and technical analysis tasks directly.""",
        model="llama3.2", 
        provider="ollama"
    )
    
    
    team = Team(npcs=[coordinator, writer, analyst], forenpc=coordinator)
    
    print(f"Team: {team.name}")
    print(f"NPCs: {list(team.npcs.keys())}")
    print(f"Forenpc: {team.get_forenpc().name}")
    
    
    request = """
    I need a project status report for our Q4 initiative. The report should include:
    1. Executive summary of progress
    2. Key milestones achieved
    3. Upcoming deliverables
    4. Risk assessment
    
    This is a standard business document that should be professional and concise.
    """
    
    print(f"\nRequest: {request}")
    print("\n" + "="*50)
    
    result = team.orchestrate(request)
    
    print("Final Result:")
    print(f"Output: {result.get('output', 'No output')}")
    print(f"Debrief: {result.get('debrief', 'No debrief')}")
    
    return result

if __name__ == "__main__":
    test_full_orchestration()