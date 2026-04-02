from dataclasses import dataclass, field
from typing import List

from datetime import datetime
import glob
import json
import os
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

try:
    from peft import LoraConfig, PeftModel
except ImportError:
    LoraConfig = None
    PeftModel = None

try:
    from trl import DPOTrainer, DPOConfig
except (ImportError, RuntimeError):
    DPOTrainer = None
    DPOConfig = None

try:
    import mlx.core as mx
    import mlx.optimizers as mlx_opt
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.trainer import TrainingArgs as MLXTrainingArgs, train as mlx_train
    from mlx_lm.tuner.utils import linear_to_lora_layers
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

import random
from typing import List, Dict, Any, Optional, Callable
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response
from npcpy.ft.sft import _resolve_mlx_model, _num_lora_layers

@dataclass
class RLConfig:
    base_model_name: str = "Qwen/Qwen3-0.6B"
    adapter_path: str = "./rl_adapter"
    device: str = "cpu"  # "cpu", "cuda", "mlx"
    max_iterations: int = 8
    min_reward_gap: float = 0.4
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6
    beta: float = 0.5
    max_length: int = 512
    max_prompt_length: int = 256
    use_4bit: bool = False
    use_8bit: bool = False
    fp16: bool = False
    bf16: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    max_pairs: int = 200
    warmup_steps: int = 5
    logging_steps: int = 5
    save_steps: int = 20

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

            print(f"Agent {agent.name}: Reward={trace['reward']:.2f}")

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
        print(f"Warning: Only {len(pairs)} pairs found. May overfit.")

    return Dataset.from_list(pairs)

def _train_dpo_mlx(
    pairs: List[Dict[str, str]],
    config: RLConfig,
) -> str:
    if not MLX_AVAILABLE:
        raise ImportError("MLX backend requires mlx and mlx-lm. pip install mlx mlx-lm")

    os.makedirs(config.adapter_path, exist_ok=True)

    mlx_model_name = _resolve_mlx_model(config.base_model_name)
    model, tokenizer = mlx_load(mlx_model_name)

    lora_cfg = {
        "rank": config.lora_r,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "scale": config.lora_alpha / config.lora_r,
    }
    linear_to_lora_layers(model, _num_lora_layers(config.lora_r), lora_cfg)

    # mlx-lm doesn't have native DPO, so SFT on chosen responses
    processed = []
    for p in pairs:
        text = f"{p['prompt']}\n{p['chosen']}"
        tokens = tokenizer.encode(text)
        if tokens[-1] != tokenizer.eos_token_id:
            tokens.append(tokenizer.eos_token_id)
        processed.append((tokens, 0))

    class _ProcessedDataset:
        def __init__(self, data):
            self._data = data
        def __getitem__(self, idx):
            return self._data[idx]
        def __len__(self):
            return len(self._data)

    train_dataset = _ProcessedDataset(processed)

    iters_per_epoch = max(1, len(pairs) // config.per_device_train_batch_size)
    total_iters = iters_per_epoch * config.num_train_epochs

    adapter_file = os.path.join(config.adapter_path, "adapters.safetensors")

    training_args = MLXTrainingArgs(
        batch_size=config.per_device_train_batch_size,
        iters=total_iters,
        val_batches=0,
        steps_per_report=config.logging_steps,
        steps_per_eval=0,
        steps_per_save=config.save_steps,
        max_seq_length=config.max_length,
        adapter_file=adapter_file,
        grad_checkpoint=True,
        grad_accumulation_steps=config.gradient_accumulation_steps,
    )

    optimizer = mlx_opt.AdamW(learning_rate=config.learning_rate)

    print(f"MLX DPO (SFT on chosen): {mlx_model_name}, {len(pairs)} pairs, {total_iters} iters")

    mlx_train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=None,
        args=training_args,
    )

    adapter_config = {
        "model": mlx_model_name,
        "fine_tune_type": "lora",
        "num_layers": _num_lora_layers(config.lora_r),
        "lora_parameters": {
            "rank": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "scale": config.lora_alpha / config.lora_r,
        },
    }
    with open(os.path.join(config.adapter_path, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"MLX adapter saved to {config.adapter_path}")
    return config.adapter_path

def _train_dpo_torch(
    traces: List[Dict[str, Any]],
    config: RLConfig,
) -> str:

    preference_dataset = create_preference_pairs(
        traces,
        min_reward_gap=config.min_reward_gap
    )

    if preference_dataset is None or len(preference_dataset) == 0:
        print("No valid preference pairs. Cannot train.")
        return None

    if config.max_pairs and len(preference_dataset) > config.max_pairs:
        preference_dataset = preference_dataset.select(range(config.max_pairs))

    print(f"Training with {len(preference_dataset)} preference pairs (device={config.device})")

    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True
    }

    if config.device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    if config.use_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for 4-bit. pip install bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        print("Using 4-bit quantization")
    elif config.use_8bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for 8-bit. pip install bitsandbytes")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True
        )
        print("Using 8-bit quantization")
    else:
        if config.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif config.fp16:
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        **model_kwargs
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=config.lora_target_modules
    )

    if config.use_4bit or config.use_8bit:
        optim = "paged_adamw_8bit"
    else:
        optim = "adamw_torch"

    use_fp16 = config.fp16 or config.use_4bit
    use_bf16 = config.bf16
    if config.device == "cpu":
        use_fp16 = False
        use_bf16 = False

    training_args = DPOConfig(
        output_dir="./dpo_results",
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        weight_decay=0.1,
        beta=config.beta,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        remove_unused_columns=False,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
        dataloader_num_workers=0,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=optim,
        warmup_steps=config.warmup_steps,
        save_strategy="steps",
        save_total_limit=2,
        no_cuda=(config.device == "cpu"),
    )

    trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=preference_dataset,
        peft_config=peft_config,
        processing_class=tokenizer
    )

    print("Starting DPO training...")
    trainer.train()

    os.makedirs(config.adapter_path, exist_ok=True)
    trainer.save_model(config.adapter_path)
    print(f"Adapter saved to {config.adapter_path}")

    return config.adapter_path

def train_with_dpo(
    traces: List[Dict[str, Any]],
    config: Optional[RLConfig] = None
) -> str:

    if config is None:
        config = RLConfig()

    if config.device == "mlx":
        # Extract pairs from traces first, then pass to MLX
        preference_dataset = create_preference_pairs(
            traces,
            min_reward_gap=config.min_reward_gap
        )
        if preference_dataset is None or len(preference_dataset) == 0:
            print("No valid preference pairs. Cannot train.")
            return None

        if config.max_pairs and len(preference_dataset) > config.max_pairs:
            preference_dataset = preference_dataset.select(range(config.max_pairs))

        pairs = [
            {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            for r in preference_dataset
        ]
        return _train_dpo_mlx(pairs, config)
    else:
        return _train_dpo_torch(traces, config)

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
    adapter_path: str,
    device: str = "cpu",
    use_4bit: bool = False,
    use_8bit: bool = False,
    merge_adapter: bool = True
):
    if device == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError("MLX backend requires mlx and mlx-lm. pip install mlx mlx-lm")
        mlx_model = _resolve_mlx_model(base_model_id)
        if adapter_path and os.path.exists(adapter_path):
            model, tokenizer = mlx_load(mlx_model, adapter_path=adapter_path)
        else:
            model, tokenizer = mlx_load(mlx_model)
        return model, tokenizer

    # torch path
    print(f"Loading base model: {base_model_id}")

    model_kwargs = {
        "trust_remote_code": True
    }

    if device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    if use_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for 4-bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif use_8bit:
        if BitsAndBytesConfig is None:
            raise ImportError("bitsandbytes required for 8-bit")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        **model_kwargs
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
        if merge_adapter and not (use_4bit or use_8bit):
            model = model.merge_and_unload()

    return model, tokenizer
