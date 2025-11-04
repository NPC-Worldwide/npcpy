# SIMPLE REVIEW ANALYSIS RL TRAINING - CLEAN VERSION
import random
from typing import Dict, Any, List

from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response
from npcpy.ft.rl import run_rl_training, RLConfig  # ONLY IMPORTS, NO REDEFINITIONS!

# ============================================
# TOOL DEFINITIONS - Product Review Analysis
# ============================================

def analyze_review_tone(review_text: str) -> str:
    """Analyzes the emotional tone of a product review."""
    prompt = f"""Analyze the emotional tone of this product review.

Review: {review_text}

Return JSON with: tone (positive/negative/neutral), confidence (0-1), 
key_phrases (list of strings), emotional_intensity (0-1)"""
    
    response = get_llm_response(
        prompt, 
        model='llama3.1:8b', 
        provider='ollama', 
        format='json'
    )
    return response['response']


def extract_product_features(review_text: str) -> str:
    """Extracts mentioned product features from review."""
    prompt = f"""Extract product features mentioned in this review.

Review: {review_text}

Return JSON with: features (list), feature_sentiments (dict mapping feature to positive/negative),
overall_quality_score (0-1)"""
    
    response = get_llm_response(
        prompt, 
        model='llama3.1:8b', 
        provider='ollama', 
        format='json'
    )
    return response['response']


def check_review_authenticity(review_text: str) -> str:
    """Checks if review appears authentic or potentially fake."""
    prompt = f"""Evaluate if this review seems authentic.

Review: {review_text}

Return JSON with: authenticity_score (0-1), red_flags (list), 
language_quality (low/medium/high), recommendation (trust/skeptical/reject)"""
    
    response = get_llm_response(
        prompt, 
        model='llama3.1:8b', 
        provider='ollama', 
        format='json'
    )
    return response['response']


TOOLS = [
    analyze_review_tone,
    extract_product_features,
    check_review_authenticity
]

# ============================================
# AGENT CONFIGURATIONS
# ============================================

AGENT_CONFIGS = [
    {
        "name": "Optimist",
        "primary_directive": """You are an optimistic review analyst. Use tools to analyze 
product reviews and recommend whether to feature them. Focus on finding positives.
Return final decision as JSON: {"decision": "FEATURE" or "REJECT", "reasoning": "explanation"}"""
    },
    {
        "name": "Skeptic",
        "primary_directive": """You are a skeptical review analyst. Use tools carefully.
Only recommend featuring genuinely helpful, authentic reviews.
Return final decision as JSON: {"decision": "FEATURE" or "REJECT", "reasoning": "explanation"}"""
    },
    {
        "name": "Balanced",
        "primary_directive": """You are a balanced review analyst. Use tools to gather
comprehensive information, then make fair judgment.
Return final decision as JSON: {"decision": "FEATURE" or "REJECT", "reasoning": "explanation"}"""
    }
]

# ============================================
# SAMPLE REVIEWS WITH GROUND TRUTH
# ============================================

SAMPLE_REVIEWS = [
    {
        "text": "This coffee maker changed my mornings! Brews perfectly every time, easy to clean, and looks great on my counter. Worth every penny.",
        "should_feature": True,
        "rating": 5
    },
    {
        "text": "AMAZING!!!! BEST PRODUCT EVER!!!! BUY NOW!!!! 5 STARS!!!!!",
        "should_feature": False,
        "rating": 5
    },
    {
        "text": "Decent blender for the price. Motor is a bit loud but gets the job done. Good for smoothies, struggles with ice.",
        "should_feature": True,
        "rating": 3
    },
    {
        "text": "Broke after 2 days. Customer service was unhelpful. Total waste of money.",
        "should_feature": True,
        "rating": 1
    },
    {
        "text": "This is product. I used product. Product is good. I like product very much. You buy product now.",
        "should_feature": False,
        "rating": 5
    },
    {
        "text": "The headphones have excellent noise cancellation and battery life is solid (20+ hours). Comfort could be better for long sessions. Bluetooth connectivity is reliable.",
        "should_feature": True,
        "rating": 4
    },
    {
        "text": "Bad bad bad bad bad. Not good. Very bad. Do not buy.",
        "should_feature": False,
        "rating": 1
    },
    {
        "text": "I've been using this vacuum for 6 months. Suction power hasn't decreased, easy to empty, and the attachments are actually useful unlike my old vacuum.",
        "should_feature": True,
        "rating": 5
    }
]

# ============================================
# REWARD FUNCTION (YOUR CUSTOM LOGIC)
# ============================================

def calculate_reward(trace: Dict[str, Any]) -> float:
    """Calculate reward based on whether agent made correct featuring decision."""
    
    final_output = trace.get('final_output', '').upper()
    
    if 'FEATURE' in final_output and 'REJECT' not in final_output:
        agent_decision = 'FEATURE'
    elif 'REJECT' in final_output:
        agent_decision = 'REJECT'
    else:
        return -1.0
    
    if not trace.get('completed', False):
        return -0.5
    
    should_feature = trace['task_metadata'].get('should_feature', False)
    correct_decision = 'FEATURE' if should_feature else 'REJECT'
    
    if agent_decision == correct_decision:
        return 1.0
    else:
        return 0.0

# ============================================
# TASK GENERATION (YOUR CUSTOM LOGIC)
# ============================================

def generate_review_tasks(num_tasks: int = 15) -> List[Dict[str, Any]]:
    """Generate review analysis tasks with ground truth."""
    tasks = []
    
    for i in range(num_tasks):
        review = random.choice(SAMPLE_REVIEWS)
        
        task = {
            'prompt': f"""Should we feature this product review on our website? 
Use tools to analyze the review quality, authenticity, and helpfulness.

Review: "{review['text']}"
Star Rating: {review['rating']}/5

Provide your final decision as JSON with "decision" (FEATURE or REJECT) and "reasoning".""",
            'should_feature': review['should_feature'],
            'review_text': review['text'],
            'rating': review['rating']
        }
        
        tasks.append(task)
    
    return tasks

# ============================================
# MAIN - ONLY CALLS NPCPY FUNCTIONS
# ============================================

def main():
    print("🔥 Starting Review Analysis RL Training")
    
    tasks = generate_review_tasks(num_tasks=12)
    
    agents = [
        NPC(
            name=config["name"],
            primary_directive=config["primary_directive"],
            tools=TOOLS,
            model='llama3.1:8b',
            provider='ollama'
        )
        for config in AGENT_CONFIGS
    ]
    
    rl_config = RLConfig(
        base_model_name="meta-llama/Llama-3.1-8B",
        adapter_path="./review_analyst_llama_adapter",
        max_iterations=4,
        min_reward_gap=0.5,
        num_train_epochs=5,
        learning_rate=5e-6
    )
    
    # ONLY NPCPY FUNCTION CALL - NO REDEFINITIONS!
    adapter_path = run_rl_training(
        tasks=tasks,
        agents=agents,
        reward_fn=calculate_reward,
        config=rl_config,
        save_traces=True
    )
    
    if adapter_path:
        print(f"\n✅ Done! Adapter: {adapter_path}")
    else:
        print("\n⚠️ Training failed")

if __name__ == "__main__":
    main()