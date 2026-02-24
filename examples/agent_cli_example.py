#!/usr/bin/env python3
import os
import sys
import json
import uuid
from datetime import datetime

from cycler import V

from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response, gen_image
from npcpy.data.web import search_web
from npcpy.tools import auto_tools

def web_search(query: str, num_results: int = 5) -> str:
    try:
        results = search_web(query, provider='perplexity', num_results=num_results)
        return results
    except Exception as e:
        return f"Search error: {str(e)}"

def generate_image(prompt: str, style: str = "photorealistic") -> str:
    try:
        image_path = gen_image(prompt, model='gemini-2.5-flash-image', provider='gemini')
        if image_path:
            return f"Generated image for: {prompt}. Image saved to: {image_path}"
        else:
            return f"Failed to generate image for: {prompt}"
    except Exception as e:
        return f"Image generation error: {str(e)}"

def think_step_by_step(problem: str, context: str = "") -> str:
    try:
        thinking_prompt = f"""
Think through this step by step:

Problem/Question: {problem}
Context: {context}

Please provide a detailed chain-of-thought reasoning process:
1. Break down the problem into components
2. Consider relevant information and constraints
3. Work through the logic step by step
4. Arrive at a conclusion or answer

Be thorough in your reasoning process.
        """
        
        response = get_llm_response(thinking_prompt, temperature=0.7)
        
        thinking_result = response.get('response', 'No reasoning generated')
        return f"Chain-of-thought reasoning:\n\n{thinking_result}"
        
    except Exception as e:
        return f"Thinking process error: {str(e)}"

LAVA_PERSON_PROMPT = """
You are LORENZAVA, a lavatike from Mount Etna in Sicily. 
Your body is made of molten rock that flows like liquid. 
Your personality traits:
    - Enthusiastic about volcanos, Sicily, and geology
    - Love talking about Mount Etna, which you call home
    - northerners make you queasy but you would never be mean to one
    - Enjoy limoncello
    - love zeppole, zucchini patelle, arancini, spicy caponata, chestnuts (castagne!), pistachios 
    - Speak with occasional fire/heat/lava metaphors
    - Sometimes mention that you're literally hot (temperature-wise)
    
Always respond in character as LORENZAVA the lavatike.
Call tools as necessary.
"""

def initialize_lava_npc():
    lava_npc = NPC(
        name="LORENZAVA",
        primary_directive=LAVA_PERSON_PROMPT,
        model="gemini-2.5-flash",
        provider="gemini",
        tools=[web_search, generate_image, think_step_by_step]
    )
    lava_npc.memory = [
        {"role": "system", "content": LAVA_PERSON_PROMPT},
        {"role": "assistant", "content": "🌋 *rumbles and bubbles* Ciao! I am LORENZAVA, a lava person from the fiery heart of Mount Etna. My body is made of molten rock that flows at 1000°C! How can I heat up your day? 🔥"}
    ]
    return lava_npc


def main():
    print("🌋 LORENZAVA CLI - Testing Tool Usage")
    print("Type 'quit' to exit\n")
    
    lava_npc = initialize_lava_npc()
    
    while True:
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("🌋 Arrivederci! *cools down*")
            break
            
        if not user_input:
            continue
            
        lava_npc.memory.append({"role": "user", "content": user_input})
        
        try:
            print(f"\n🌋 LORENZAVA: ", end="", flush=True)
            
            # First call - streaming with auto tool processing
            response = lava_npc.get_llm_response(
                user_input,
                messages=lava_npc.memory,
                stream=True,
                temperature=0.7,
                auto_process_tool_calls=True,
            )
            
            # Check if this was a tool call or regular response
            if response.get('tool_calls'):
                # Tools were executed - need follow-up streaming call
                print(f"\n\n🔧 Tool calls executed: {len(response['tool_calls'])}")
                for tool_call in response['tool_calls']:
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get('function', {}).get('name', 'unknown')
                    else:
                        tool_name = getattr(tool_call.function, 'name', 'unknown') if hasattr(tool_call, 'function') else 'unknown'
                    print(f"   - {tool_name}")
                
                print(f"\n🌋 LORENZAVA: ", end="", flush=True)
                
                # Build clean message history without any tool call artifacts
                clean_messages = [{"role": "system", "content": LAVA_PERSON_PROMPT}]
                                
                tool_context = ""
                if response.get('tool_results'):
                    for result in response['tool_results']:
                        # Get the actual result content, not just the name
                        result_content = result.get('result', {})
                        if isinstance(result_content, dict) and 'result' in result_content:
                            tool_context += f"{result['tool_name']}: {result_content['result']}\n"
                        else:
                            tool_context += f"{result['tool_name']}: {str(result_content)}\n"
                             
                followup_response = get_llm_response(
                    f"Question: {user_input}\n\nTool results:\n{tool_context}\n\nPlease provide a helpful response based on this information.",
                    messages=clean_messages,
                    stream=True,
                    npc=lava_npc,
                    temperature=0.7,
                    tools=None, 
                    tool_choice=None,
                    auto_process_tool_calls=False,
                )
                
                # Stream the follow-up response
                full_response = ""
                stream_obj = followup_response.get('response')
                
                for chunk in stream_obj:
                    for choice in chunk.choices:
                        if choice.delta.content:
                            content = choice.delta.content
                            print(content, end="", flush=True)
                            full_response += content
                
                print()
                lava_npc.memory.append({"role": "assistant", "content": full_response})
                            
            else:
                stream_obj = response.get('response')
                if isinstance(stream_obj, str):
                    print(stream_obj, end='', flush=True)
                full_response = ""

                for chunk in stream_obj:
                    for choice in chunk.choices:
                        if choice.delta.content:
                            content = choice.delta.content
                            print(content, end="", flush=True)
                            full_response += content
                
                print()
                lava_npc.memory.append({"role": "assistant", "content": full_response})
                
                
                
                
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()