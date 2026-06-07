import random
import json
from collections import Counter
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response

def generate_npcs(
    generation_theme: str,
    num_npcs: int,
    model: str,
    provider: str
) -> list:
    """
    Generates a list of unique NPC agents by iterating through LLM calls.
    One NPC is generated per call for robustness.
    """
    print(f"\n--- Generating {num_npcs} NPCs iteratively based on theme: '{generation_theme}' ---")
    npcs = []
    generated_names = []

    for i in range(num_npcs):
        avoidance_prompt = f"Avoid creating a persona similar to these already generated: {', '.join(generated_names)}" if generated_names else ""

        prompt = f"""
        Based on the creative theme "{generation_theme}", invent one completely original persona.
        Do not use any existing historical, literary, or public figures.
        {avoidance_prompt}

        Return the result as a single, valid JSON object with two keys: "name" and "primary_directive".

        Here is an example: """ + """
        {
            "primary_directive": "a directive for the persona", 
            "name": "a name for the persona"
        }
        """
        response = get_llm_response(prompt, model=model, provider=provider, format='json')
        npc_data = response['response']

        npc = NPC(
            name=npc_data['name'],
            primary_directive=npc_data['primary_directive'],
            model=None,
            provider='ollama'
        )
        npcs.append(npc)
        generated_names.append(npc.name)
        print(f"({i+1}/{num_npcs}) Successfully generated: {npc.name}")

    print("--- NPC Generation Complete ---")
    return npcs

def run_debate(agent1: NPC, argument1: str, agent2: NPC, argument2: str, turns: int = 2):
    """
    Runs a debate, correctly passing the 'request' argument to get_llm_response.
    """
    print(f"\n--- Debate ---")
    print(f"{agent1.name} champions: '{argument1[:100]}...'")
    print(f"{agent2.name} champions: '{argument2[:100]}...'")
    messages = [{'role': 'system', 'content': f"You are in a debate. Argue for your assigned position."}]
    opening_request = f"{agent1.name}, make your opening statement for the argument: '{argument1}'"
    response = agent1.get_llm_response(opening_request, messages=messages)
    messages.extend([
        {'role': 'user', 'content': opening_request},
        {'role': 'assistant', 'name': agent1.name, 'content': response['response']}
    ])
    current_agent, current_arg, other_agent, other_arg = (agent2, argument2, agent1, argument1)
    for _ in range(turns * 2 - 1):
        turn_request = f"{current_agent.name}, respond to {other_agent.name} and defend your position: '{current_arg}'"
        response = current_agent.get_llm_response(turn_request, messages=messages)
        messages.extend([
            {'role': 'user', 'content': turn_request},
            {'role': 'assistant', 'name': current_agent.name, 'content': response['response']}
        ])
        current_agent, other_agent = other_agent, current_agent
        current_arg, other_arg = other_arg, current_arg
    return messages

def run_synthesis_tournament(initial_contenders: list, initial_topic: str, model: str, provider: str):
    """
    Runs a tournament where the final synthesis is weighted by the judges' votes.
    """
    if len(initial_contenders) & (len(initial_contenders) - 1) != 0 or len(initial_contenders) == 0:
        print("Error: Number of contenders must be a power of 2.")
        return None

    print(f"===== Starting Synthesis Tournament on: {initial_topic} =====")
    evolving_ideas = {npc.name: (npc, initial_topic) for npc in initial_contenders}
    
    round_num = 1
    while len(evolving_ideas) > 1:
        print(f"\n\n======= ROUND {round_num} =======\n")
        
        contender_names = list(evolving_ideas.keys())
        random.shuffle(contender_names)
        next_round_ideas = {}

        for i in range(0, len(contender_names), 2):
            name1, name2 = contender_names[i], contender_names[i+1]
            agent1, argument1 = evolving_ideas[name1]
            agent2, argument2 = evolving_ideas[name2]

            debate_history = run_debate(agent1, argument1, agent2, argument2)
            transcript = "\n".join([f"{msg.get('name')}: {msg['content']}" for msg in debate_history if msg.get('role') == 'assistant'])

            judge_theme = f"Judges evaluating two opposing arguments: '{argument1[:50]}...' versus '{argument2[:50]}...'"
            judging_panel = generate_npcs(judge_theme, 3, model=model, provider=provider)

            
            votes = []
            print("\n--- The Jury Votes ---")
            vote_prompt = f"""
            After reviewing the debate transcript, which debater's argument was more foundational and persuasive: {agent1.name} or {agent2.name}?
            Respond with ONLY the name of the winner. Do not add any other text or explanation.

            TRANSCRIPT:
            ---
            {transcript}
            ---

            EXAMPLE RESPONSE:
            {agent1.name}
            """
            for judge in judging_panel:
                response_text = judge.get_llm_response(vote_prompt)['response'].strip()
                vote = agent1.name if agent1.name in response_text else agent2.name
                votes.append(vote)
                print(f"{judge.name} votes for: {vote}")

            
            vote_counts = Counter(votes)
            winner_name, winner_votes = vote_counts.most_common(1)[0]
            
            advancing_agent = agent1 if winner_name == agent1.name else agent2
            losing_agent = agent2 if winner_name == agent1.name else agent1
            loser_votes = len(votes) - winner_votes
            
            winning_argument = argument1 if winner_name == agent1.name else argument2
            losing_argument = argument2 if winner_name == agent1.name else winning_argument

            
            final_synthesis_prompt = f"""
            You are a Master Synthesizer. Your task is to create a new, superior argument by merging two competing ideas based on a jury's verdict.

            The jury's vote was {winner_votes} for {winner_name}'s argument and {loser_votes} for {losing_agent.name}'s argument.

            Winning Argument ({winner_name}): "{winning_argument}"
            Losing Argument ({losing_agent.name}): "{losing_argument}"

            Create a single, potent, synthesized argument. Because the vote was {winner_votes} to {loser_votes}, the new argument must be primarily based on the winning argument, but it MUST incorporate the most valuable and resilient points from the losing argument to make the final concept stronger. The final output should be ONLY the new, synthesized argument paragraph. Do not add any introductory phrases.

            EXAMPLE OUTPUT:
            A system's survival is paramount, as no component can thrive without the whole. However, this survival mandate is not absolute; it must be tempered by a foundational respect for the well-being of its components, integrating their feedback as a vital sign of systemic health, thereby proving that true resilience is a synthesis of centralized stability and distributed vitality.
            """
            
            compressed_idea = get_llm_response(final_synthesis_prompt, model=model, provider=provider)['response'].strip()
            
            print(f"\n--- Round Result ---")
            print(f"Vote: {winner_votes} for {winner_name}, {loser_votes} for {losing_agent.name}")
            print(f"Carrier Agent: {advancing_agent.name}")
            print(f"Weighted Synthesized Idea: '{compressed_idea[:150]}...'")
            
            next_round_ideas[advancing_agent.name] = (advancing_agent, compressed_idea)
        
        evolving_ideas = next_round_ideas
        round_num += 1
        
    final_agent, final_idea = list(evolving_ideas.values())[0]
    print(f"\n\n===== Tournament Complete! ======")
    print(f"The Final Carrier Agent is: {final_agent.name}")
    print(f"The Final Compressed Idea is: {final_idea}")
    return final_agent, final_idea

if __name__ == "__main__":
    raise ValueError("No model specified. Set model explicitly.")

    contender_theme = "Entities from a dimension where logic is a fluid, not a constant"
    contenders = generate_npcs(
        generation_theme=contender_theme, 
        num_npcs=4, 
        model=GENERATION_MODEL, 
        provider=GENERATION_PROVIDER
    )

    if contenders and len(contenders) >= 4:
        initial_topic = "Should a system prioritize its own survival above the well-being of its components?"
        run_synthesis_tournament(
            initial_contenders=contenders, 
            initial_topic=initial_topic,
            model=GENERATION_MODEL,
            provider=GENERATION_PROVIDER
        )
    else:
        print("Could not run tournament: requires at least 4 contenders to be generated.")