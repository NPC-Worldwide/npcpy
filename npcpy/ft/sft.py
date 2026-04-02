from dataclasses import dataclass, field
from datasets import Dataset
import json
import numpy as np
import os
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments
    )
    from trl import SFTTrainer
    from peft import LoraConfig
except Exception:
    torch = None
    SFTTrainer = None
    LoraConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None

try:
    import mlx.core as mx
    import mlx.optimizers as mlx_opt
    from mlx_lm import load as mlx_load, generate as mlx_generate
    from mlx_lm.tuner.trainer import TrainingArgs as MLXTrainingArgs, train as mlx_train
    from mlx_lm.tuner.utils import linear_to_lora_layers
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from typing import List, Dict, Any, Optional

# Map common HF model names to mlx-community equivalents
_MLX_MODEL_MAP = {
    "google/gemma-3-270m-it": "mlx-community/gemma-3-270m-it-4bit",
    "google/gemma-3-1b-it": "mlx-community/gemma-3-1b-it-4bit",
    "google/gemma-3-4b-it": "mlx-community/gemma-3-4b-it-4bit",
    "google/gemma-3-12b-it": "mlx-community/gemma-3-12b-it-4bit",
    "google/gemma-3-27b-it": "mlx-community/gemma-3-27b-it-4bit",
    "Qwen/Qwen3-0.6B": "mlx-community/Qwen3-0.6B-4bit",
    "Qwen/Qwen3-1.7B": "mlx-community/Qwen3-1.7B-4bit",
    "Qwen/Qwen3-4B": "mlx-community/Qwen3-4B-4bit",
    "Qwen/Qwen3-8B": "mlx-community/Qwen3-8B-4bit",
    "Qwen/Qwen3-14B": "mlx-community/Qwen3-14B-4bit",
    "Qwen/Qwen3-32B": "mlx-community/Qwen3-32B-4bit",
    "meta-llama/Llama-3.1-8B-Instruct": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "meta-llama/Llama-3.2-1B-Instruct": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "meta-llama/Llama-3.2-3B-Instruct": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mistralai/Mistral-7B-Instruct-v0.3": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
}

def _resolve_mlx_model(hf_name: str) -> str:
    if hf_name in _MLX_MODEL_MAP:
        return _MLX_MODEL_MAP[hf_name]
    if hf_name.startswith("mlx-community/") or os.path.exists(hf_name):
        return hf_name
    return f"mlx-community/{hf_name.split('/')[-1]}"

def _num_lora_layers(lora_r: int) -> int:
    if lora_r <= 8:
        return 16
    elif lora_r <= 32:
        return 24
    elif lora_r <= 64:
        return 32
    return 48

@dataclass
class SFTConfig:
    base_model_name: str = "google/gemma-3-270m-it"
    output_model_path: str = "models/sft_model"
    device: str = "cpu"  # "cpu", "cuda", "mlx"
    lora_r: int = 8
    lora_alpha: int = 16
    use_4bit: bool = False
    fp16: bool = False
    bf16: bool = False
    lora_dropout: float = 0.15
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    logging_steps: int = 10
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine_with_restarts"
    weight_decay: float = 0.01
    max_length: int = 512
    save_steps: int = 50

def format_training_examples(
    inputs: List[str],
    outputs: List[str],
    format_style: str = "gemma"
) -> List[Dict[str, str]]:

    formatted = []

    for inp, out in zip(inputs, outputs):
        if format_style == "gemma":
            text = f"<start_of_turn>user\n{inp}<end_of_turn>\n<start_of_turn>model\n{out}<end_of_turn>"
        elif format_style == "llama":
            text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{inp}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{out}<|eot_id|>"
        else:
            text = f"Input: {inp}\nOutput: {out}"

        formatted.append({"text": text})

    return formatted

def _run_sft_mlx(
    X: List[str],
    y: List[str],
    config: 'SFTConfig',
    format_style: str = "gemma",
) -> str:
    if not MLX_AVAILABLE:
        raise ImportError("MLX backend requires mlx and mlx-lm. pip install mlx mlx-lm")

    os.makedirs(config.output_model_path, exist_ok=True)

    mlx_model_name = _resolve_mlx_model(config.base_model_name)
    model, tokenizer = mlx_load(mlx_model_name)

    lora_cfg = {
        "rank": config.lora_r,
        "alpha": config.lora_alpha,
        "dropout": config.lora_dropout,
        "scale": config.lora_alpha / config.lora_r,
    }
    linear_to_lora_layers(model, _num_lora_layers(config.lora_r), lora_cfg)

    formatted = format_training_examples(X, y, format_style)

    # Build processed dataset: each item is (token_ids, offset)
    processed = []
    for rec in formatted:
        tokens = tokenizer.encode(rec["text"])
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

    iters_per_epoch = max(1, len(X) // config.per_device_train_batch_size)
    total_iters = iters_per_epoch * config.num_train_epochs

    adapter_file = os.path.join(config.output_model_path, "adapters.safetensors")

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

    print(f"MLX SFT: {mlx_model_name}, LoRA r={config.lora_r}, {len(X)} examples, {total_iters} iters")

    mlx_train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=None,
        args=training_args,
    )

    # save adapter config in the format mlx-lm's load_adapters expects
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
    with open(os.path.join(config.output_model_path, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"MLX adapter saved to {config.output_model_path}")
    return config.output_model_path

def _run_sft_torch(
    X: List[str],
    y: List[str],
    config: 'SFTConfig',
    validation_split: float = 0.0,
    format_style: str = "gemma",
) -> str:
    formatted_examples = format_training_examples(X, y, format_style)

    if validation_split > 0:
        split_idx = int(len(formatted_examples) * (1 - validation_split))
        train_examples = formatted_examples[:split_idx]
        val_examples = formatted_examples[split_idx:]
        print(f"Split: {len(train_examples)} train, {len(val_examples)} val")
    else:
        train_examples = formatted_examples

    dataset = Dataset.from_list(train_examples)

    model_kwargs = {
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }

    if config.device == "cuda":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        **model_kwargs
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    use_fp16 = config.fp16
    use_bf16 = config.bf16
    if config.device == "cpu":
        use_fp16 = False
        use_bf16 = False

    training_args = TrainingArguments(
        output_dir=config.output_model_path,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        fp16=use_fp16,
        bf16=use_bf16,
        lr_scheduler_type=config.lr_scheduler_type,
        group_by_length=True,
        save_steps=config.save_steps,
        weight_decay=config.weight_decay,
        no_cuda=(config.device == "cpu"),
    )

    def formatting_func(example):
        return example["text"]

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        processing_class=tokenizer,
        formatting_func=formatting_func
    )

    print(f"Training on {len(dataset)} examples (device={config.device})")
    trainer.train()

    trainer.save_model(config.output_model_path)
    print(f"Model saved to {config.output_model_path}")

    return config.output_model_path

def run_sft(
    X: List[str],
    y: List[str],
    config: Optional[SFTConfig] = None,
    validation_split: float = 0.0,
    format_style: str = "gemma"
) -> str:

    if config is None:
        config = SFTConfig()

    if len(X) != len(y):
        raise ValueError(
            f"X and y must have same length: {len(X)} vs {len(y)}"
        )

    if config.device == "mlx":
        return _run_sft_mlx(X, y, config, format_style)
    else:
        return _run_sft_torch(X, y, config, validation_split, format_style)

def load_sft_model(model_path: str, device: str = "cpu", base_model: str = None):

    if device == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError("MLX backend requires mlx and mlx-lm. pip install mlx mlx-lm")

        adapter_file = os.path.join(model_path, "adapters.safetensors")
        if os.path.exists(adapter_file):
            if base_model:
                mlx_model = _resolve_mlx_model(base_model)
            else:
                config_path = os.path.join(model_path, "adapter_config.json")
                if os.path.exists(config_path):
                    with open(config_path) as f:
                        cfg = json.load(f)
                    mlx_model = _resolve_mlx_model(cfg.get("model", ""))
                else:
                    raise ValueError(
                        f"Adapter at {model_path} but no base model specified. "
                        "Pass base_model= or ensure adapter_config.json exists."
                    )
            model, tokenizer = mlx_load(mlx_model, adapter_path=model_path)
        else:
            model, tokenizer = mlx_load(model_path)
        return model, tokenizer

    # torch path (cpu or cuda)
    device_map = "auto" if device == "cuda" else {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map=device_map,
        attn_implementation="eager"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def predict_sft(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    device: str = "cpu",
    format_style: str = "gemma",
) -> str:

    if format_style == "gemma":
        formatted_prompt = f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    elif format_style == "llama":
        formatted_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        formatted_prompt = f"Input: {prompt}\nOutput: "

    if device == "mlx":
        if not MLX_AVAILABLE:
            raise ImportError("MLX backend requires mlx and mlx-lm. pip install mlx mlx-lm")
        from mlx_lm.sample_utils import make_sampler
        sampler = make_sampler(temperature)
        response = mlx_generate(
            model,
            tokenizer,
            prompt=formatted_prompt,
            max_tokens=max_new_tokens,
            sampler=sampler,
        )
        return response

    # torch path
    dev = next(model.parameters()).device

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = inputs.input_ids.to(dev)
    attention_mask = inputs.attention_mask.to(dev)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )

    full_response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=False
    )

    if "<start_of_turn>model\n" in full_response:
        response = full_response.split(
            "<start_of_turn>model\n"
        )[-1]
        response = response.split("<end_of_turn>")[0].strip()
    else:
        response = tokenizer.decode(
            outputs[0][len(input_ids[0]):],
            skip_special_tokens=True
        )

    return response
