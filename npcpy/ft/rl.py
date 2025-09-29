from dataclasses import dataclass

from datetime import datetime
import glob
import json
import os
import pandas as pd
try:
    from datasets import Dataset

    from peft import LoraConfig, PeftModel
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer
    )
    from trl import DPOTrainer, DPOConfig
except:
    Dataset = None
    PeftModel = None
    DPOConfig = None
    DPOTrainer = None
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

    
import random
from typing import List, Dict, Any, Optional, Callable
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response


@dataclass
class RLConfig:
    base_model_name: str = "Qwen/Qwen3-0.6B"
    adapter_path: str = "./rl_adapter"
    max_iterations: int = 8
    min_reward_gap: float = 0.4
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6
    beta: float = 0.5
    max_length: int = 512
    max_prompt_length: int = 256


class TaskExecutor:

    def __init__(
        self,
        agent: NPC,
        max_iterations: int = 8
    ):
        self.agent = agent
        self.max_iterations = max_iterations
    
    def execute_task(
        self,
        task_prompt: str
    ) -> Dict[str, Any]:

        messages = [
            {
                "role": "system",
                "content": self.agent.primary_directive
            }
        ]
        
        raw_responses = []
        current_prompt = task_prompt
        
        for i in range(self.max_iterations):
            response_obj = self.agent.get_llm_response(
                current_prompt,
                messages=messages,
                auto_process_tool_calls=True
            )
            
            raw_responses.append(response_obj)
            messages = response_obj.get('messages', messages)
            
            last_content = messages[-1].get('content', '')
            
            if self._is_complete(last_content):
                return {
                    "raw_responses": raw_responses,
                    "final_output": last_content,
                    "total_iterations": i + 1,
                    "completed": True
                }
            
            current_prompt = (
                "Continue or provide final answer."
            )
        
        return {
            "raw_responses": raw_responses,
            "final_output": messages[-1].get('content', ''),
            "total_iterations": self.max_iterations,
            "completed": False
        }
    
    def _is_complete(self, content: str) -> bool:

        completion_markers = [
            "final answer:",
            "conclusion:",
            "result:",
            "therefore",
            "in summary"
        ]
        content_lower = content.lower()
        return any(
            marker in content_lower 
            for marker in completion_markers
        )


def collect_traces(
    tasks: List[Dict[str, Any]],
    agents: List[NPC],
    reward_fn: Callable[[Dict], float],
    config: Optional[RLConfig] = None
) -> List[Dict[str, Any]]:

    if config is None:
        config = RLConfig()
    
    traces = []
    
    for task in tasks:
        task_prompt = task.get('prompt', task.get('input', ''))
        
        for agent in agents:
            executor = TaskExecutor(
                agent,
                max_iterations=config.max_iterations
            )
            
            result = executor.execute_task(task_prompt)
            
            trace = {
                "agent_name": agent.name,
                "task_prompt": task_prompt,
                "final_output": result['final_output'],
                "total_iterations": result['total_iterations'],
                "completed": result['completed'],
                "task_metadata": task
            }
            
            trace['reward'] = reward_fn(trace)
            
            traces.append(trace)
            
            print(
                f"Agent {agent.name}: "
                f"Reward={trace['reward']:.2f}"
            )
    
    return traces


def create_preference_pairs(
    traces: List[Dict[str, Any]],
    min_reward_gap: float = 0.4
) -> Dataset:

    df = pd.DataFrame(traces)
    df = df[df['reward'] > -1.0].copy()
    
    if len(df) < 2:
        return None
    
    df = df.sort_values('reward', ascending=False)
    
    top_quantile = df['reward'].quantile(
        0.8,
        interpolation='higher'
    )
    low_quantile = df['reward'].quantile(
        0.2,
        interpolation='lower'
    )
    
    high_traces = df[df['reward'] >= top_quantile]
    low_traces = df[df['reward'] <= low_quantile]
    
    pairs = []
    
    for _, high_trace in high_traces.iterrows():
        for _, low_trace in low_traces.iterrows():
            reward_gap = (
                high_trace['reward'] - low_trace['reward']
            )
            
            if reward_gap >= min_reward_gap:
                pairs.append({
                    "prompt": str(high_trace['task_prompt']),
                    "chosen": str(high_trace['final_output']),
                    "rejected": str(low_trace['final_output'])
                })
    
    if len(pairs) < 5:
        print(
            f"Warning: Only {len(pairs)} pairs found. "
            "May overfit."
        )
    
    return Dataset.from_list(pairs[:100])


def train_with_dpo(
    traces: List[Dict[str, Any]],
    config: Optional[RLConfig] = None
) -> str:

    if config is None:
        config = RLConfig()
    
    preference_dataset = create_preference_pairs(
        traces,
        min_reward_gap=config.min_reward_gap
    )
    
    if preference_dataset is None or len(preference_dataset) == 0:
        print("No valid preference pairs. Cannot train.")
        return None
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        torch_dtype=torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
    )
    
    training_args = DPOConfig(
        output_dir="./dpo_results",
        per_device_train_batch_size=(
            config.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            config.gradient_accumulation_steps
        ),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        weight_decay=0.1,
        beta=config.beta,
        logging_steps=2,
        save_steps=10,
        remove_unused_columns=False,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        dataloader_num_workers=0,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        warmup_steps=2,
        save_strategy="steps",
        save_total_limit=3
    )
    
    trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=preference_dataset,
        peft_config=peft_config
    )
    
    print("Starting DPO training...")
    trainer.train()
    
    trainer.save_model(config.adapter_path)
    print(f"Adapter saved to {config.adapter_path}")
    
    return config.adapter_path


def run_rl_training(
    tasks: List[Dict[str, Any]],
    agents: List[NPC],
    reward_fn: Callable[[Dict], float],
    config: Optional[RLConfig] = None,
    save_traces: bool = True
) -> str:

    if config is None:
        config = RLConfig()
    
    print(f"Collecting traces from {len(tasks)} tasks...")
    traces = collect_traces(
        tasks,
        agents,
        reward_fn,
        config
    )
    
    if save_traces:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        traces_file = f"rl_traces_{timestamp}.csv"
        df = pd.DataFrame(traces)
        df.to_csv(traces_file, index=False)
        print(f"Traces saved to {traces_file}")
    
    print("Training with DPO...")
    adapter_path = train_with_dpo(traces, config)
    
    return adapter_path


def load_rl_model(
    base_model_id: str,
    adapter_path: str
):

    print(f"Loading base model: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float32,
        device_map="auto",
        attn_implementation='eager'
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    
    return model, tokenizer