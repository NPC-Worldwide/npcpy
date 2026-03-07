import os
import sys
sys.path.append('/media/caug/extradrive1/npcww/npcpy')

from npcpy.npc_compiler import NPC, Team
import tempfile
import yaml
import pandas as pd
import numpy as np

def setup_specialized_team():
    """Create a team with specialized roles for advanced scenarios"""
    temp_dir = tempfile.mkdtemp()
    team_dir = os.path.join(temp_dir, "specialized_team")
    os.makedirs(team_dir, exist_ok=True)
    os.makedirs(os.path.join(team_dir, "jinxes"), exist_ok=True)
    
    
    security_config = {
        "name": "security_analyst",
        "primary_directive": """You are a cybersecurity analyst specializing in threat assessment, 
        vulnerability analysis, and security best practices. You analyze security incidents, 
        assess risks, and provide actionable security recommendations.""",
        "model": "llama3.2",
        "provider": "ollama"
    }
    
    
    devops_config = {
        "name": "devops_engineer", 
        "primary_directive": """You are a DevOps engineer expert in infrastructure automation, 
        CI/CD pipelines, monitoring, and system reliability. You design scalable solutions 
        and troubleshoot complex distributed systems.""",
        "model": "llama3.2",
        "provider": "ollama"
    }
    
    
    ux_config = {
        "name": "ux_designer",
        "primary_directive": """You are a UX designer focused on user experience research, 
        interface design, and usability optimization. You analyze user behavior and create 
        designs that improve user satisfaction and conversion rates.""",
        "model": "llama3.2", 
        "provider": "ollama"
    }
    
    
    pm_config = {
        "name": "product_manager",
        "primary_directive": """You are a product manager who coordinates cross-functional teams, 
        analyzes market requirements, and makes strategic product decisions. You balance technical 
        constraints with business objectives and user needs.""",
        "model": "llama3.2",
        "provider": "ollama"
    }
    
    configs = [security_config, devops_config, ux_config, pm_config]
    for config in configs:
        npc_path = os.path.join(team_dir, f"{config['name']}.npc")
        with open(npc_path, 'w') as f:
            yaml.dump(config, f)
    
    
    team_context = {
        "forenpc": "product_manager",
        "specialization": "technical_product_development"
    }
    
    ctx_path = os.path.join(team_dir, "team.ctx") 
    with open(ctx_path, 'w') as f:
        yaml.dump(team_context, f)
    
    
    log_analysis_jinx = {
        "jinx_name": "analyze_system_logs",
        "description": "Analyze system logs for security and performance issues",
        "inputs": [
            {"log_content": "system log content to analyze"},
            {"focus_area": "security"}
        ],
        "steps": [
            {
                "name": "parse_logs",
                "engine": "python",
                "code": """
import re
from collections import Counter

log_content = context.get('log_content', '')
focus_area = context.get('focus_area', 'general')


lines = log_content.split('\\n') if log_content else [
    "2024-01-15 10:23:45 ERROR Failed login attempt for user 'admin' from 192.168.1.100",
    "2024-01-15 10:24:12 WARNING Multiple failed login attempts detected", 
    "2024-01-15 10:25:33 INFO User 'john.doe' logged in successfully",
    "2024-01-15 10:26:45 ERROR Database connection timeout",
    "2024-01-15 10:27:12 CRITICAL System memory usage at 95%"
]


log_levels = Counter()
error_patterns = []
security_events = []

for line in lines:
    if 'ERROR' in line:
        log_levels['ERROR'] += 1
        error_patterns.append(line)
    elif 'WARNING' in line:
        log_levels['WARNING'] += 1
    elif 'CRITICAL' in line:
        log_levels['CRITICAL'] += 1
        error_patterns.append(line)
    
    if 'login' in line.lower() or 'authentication' in line.lower():
        security_events.append(line)

context['log_summary'] = {
    'total_lines': len(lines),
    'log_levels': dict(log_levels),
    'error_count': log_levels['ERROR'] + log_levels['CRITICAL'],
    'security_events': len(security_events)
}

output = f"Analyzed {len(lines)} log lines. Found {context['log_summary']['error_count']} errors/critical issues."
"""
            },
            {
                "name": "generate_analysis",
                "engine": "natural", 
                "code": """
Based on the log analysis results:
- Total lines processed: {{log_summary.total_lines}}
- Error/Critical issues: {{log_summary.error_count}}
- Security events detected: {{log_summary.security_events}}

Provide a detailed analysis focusing on {{focus_area}} aspects including:
1. Key findings and patterns
2. Risk assessment
3. Immediate action items
4. Long-term recommendations
5. Monitoring suggestions

Focus on actionable insights for the {{focus_area}} team.
"""
            }
        ]
    }
    
    jinx_path = os.path.join(team_dir, "jinxes", "analyze_system_logs.jinx")
    with open(jinx_path, 'w') as f:
        yaml.dump(log_analysis_jinx, f)
    
    return team_dir

def test_security_incident_response():
    """Test case: Security incident response coordination"""
    print("=== Advanced Test 1: Security Incident Response ===")
    
    team_dir = setup_specialized_team()
    team = Team(team_path=team_dir)
    
    request = """
    We've detected unusual network activity in our production environment:
    - 300% increase in failed login attempts from IP range 185.x.x.x
    - Database queries taking 5x longer than normal
    - Memory usage spiking to 90%+ on 3 application servers
    - Users reporting intermittent 503 errors
    
    I need immediate incident response coordination:
    1. Security assessment of the potential breach
    2. DevOps analysis of system performance issues  
    3. UX impact assessment for user experience
    4. Coordinated response plan with priorities
    
    This is urgent - we need both immediate containment and long-term prevention strategies.
    """
    
    result = team.orchestrate(request)
    print("Security Incident Result:", result)
    print("\n" + "="*60 + "\n")

def test_feature_development_planning():
    """Test case: Cross-functional feature development"""
    print("=== Advanced Test 2: Feature Development Planning ===")
    
    team_dir = setup_specialized_team()
    team = Team(team_path=team_dir)
    
    request = """
    We want to add real-time collaboration features to our document editing platform:
    - Multi-user editing with conflict resolution
    - Live cursor tracking and user presence indicators
    - Voice/video chat integration
    - Real-time commenting and suggestions
    
    I need comprehensive planning across all disciplines:
    1. UX research on collaboration patterns and user needs
    2. DevOps assessment of infrastructure requirements for real-time features
    3. Security analysis of multi-user data handling and privacy
    4. Product roadmap with technical feasibility and business impact
    
    Focus on a 6-month development timeline with MVP and full feature phases.
    """
    
    result = team.orchestrate(request)
    print("Feature Development Result:", result)
    print("\n" + "="*60 + "\n")

def test_system_architecture_review():
    """Test case: System architecture review and optimization"""
    print("=== Advanced Test 3: System Architecture Review ===")
    
    team_dir = setup_specialized_team()
    team = Team(team_path=team_dir)
    
    request = """
    Our microservices architecture is showing scalability issues:
    - API response times increased 40% over 6 months
    - Database connection pooling reaching limits
    - Service-to-service communication causing bottlenecks
    - Deployment complexity making releases risky
    
    I need a comprehensive architecture review:
    1. DevOps analysis of current infrastructure bottlenecks and scaling solutions
    2. Security review of service mesh and inter-service communication
    3. UX impact analysis of performance degradation on user experience
    4. Product strategy for architecture modernization roadmap
    
    Consider both short-term optimizations and long-term architectural evolution.
    """
    
    result = team.orchestrate(request)
    print("Architecture Review Result:", result)
    print("\n" + "="*60 + "\n")

def test_specialized_npc_direct_consultation():
    """Test case: Direct consultation with specialized NPCs"""
    print("=== Advanced Test 4: Specialized NPC Consultation ===")
    
    team_dir = setup_specialized_team()
    team = Team(team_path=team_dir)
    
    
    devops = team.get_npc("devops_engineer")
    devops_request = """
    We're seeing intermittent failures in our Kubernetes cluster:
    - Pods randomly restarting every 2-3 hours
    - Memory usage patterns seem normal
    - CPU utilization spikes coincide with restarts
    - Application logs show no obvious errors before restarts
    
    What diagnostic approach would you recommend to identify the root cause?
    """
    
    devops_result = devops.check_llm_command(devops_request, team=team)
    print("DevOps Consultation:", devops_result)
    
    
    security = team.get_npc("security_analyst")
    security_request = """
    We're implementing OAuth 2.0 with PKCE for our mobile app. The security considerations are:
    - Storing refresh tokens securely on mobile devices
    - Handling token rotation without user interruption
    - Protecting against token theft in compromised devices
    - Implementing proper token validation on the backend
    
    What security architecture would you recommend for this implementation?
    """
    
    security_result = security.check_llm_command(security_request, team=team)
    print("Security Consultation:", security_result)
    print("\n" + "="*60 + "\n")

def test_complex_agent_passing_chain():
    """Test case: Complex multi-step agent passing"""
    print("=== Advanced Test 5: Complex Agent Passing Chain ===")
    
    team_dir = setup_specialized_team()
    team = Team(team_path=team_dir)
    
    pm = team.get_npc("product_manager")
    
    request = """
    We need to evaluate implementing AI-powered code review in our development workflow.
    This requires expertise from multiple domains:
    
    1. First, have the DevOps engineer assess the technical integration challenges
    2. Then, the Security analyst should evaluate the AI model security implications  
    3. The UX designer should research developer experience and workflow impact
    4. Finally, I need a consolidated product decision with implementation timeline
    
    Each team member should build on the previous analysis. This is a complex technical 
    decision that requires cross-functional expertise.
    """
    
    result = pm.check_llm_command(request, team=team)
    print("Complex Agent Passing Result:", result)
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    print("Testing Advanced NPC Orchestration Scenarios...\n")
    
    test_security_incident_response()
    test_feature_development_planning()
    test_system_architecture_review()
    test_specialized_npc_direct_consultation()
    test_complex_agent_passing_chain()
    
    print("Advanced tests completed!")