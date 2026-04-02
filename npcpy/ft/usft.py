from dataclasses import dataclass, field
import json
import os
try:
    from datasets import Dataset, load_dataset
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments
    )
    from trl import SFTTrainer
    from peft import LoraConfig
except Exception:
    Dataset = None
    load_dataset = None
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None
    SFTTrainer = None

try:
    import mlx.core as mx
    import mlx.optimizers as mlx_opt
    from mlx_lm import load as mlx_load
    from mlx_lm.tuner.trainer import TrainingArgs as MLXTrainingArgs, train as mlx_train
    from mlx_lm.tuner.utils import linear_to_lora_layers
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from typing import List, Optional
from npcpy.ft.sft import _resolve_mlx_model, _num_lora_layers

@dataclass
class USFTConfig:
    base_model_name: str = "Qwen/Qwen3-0.6B"
    output_model_path: str = "models/usft_model"
    device: str = "cpu"  # "cpu", "cuda", "mlx"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.15
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    logging_steps: int = 10
    optim: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    max_length: int = 512
    save_steps: int = 100

def _run_usft_mlx(
    texts: List[str],
    config: USFTConfig,
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

    processed = []
    for t in texts:
        tokens = tokenizer.encode(t)
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

    iters_per_epoch = max(1, len(texts) // config.per_device_train_batch_size)
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

    print(f"MLX USFT: {mlx_model_name}, {len(texts)} texts, {total_iters} iters")

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
    with open(os.path.join(config.output_model_path, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"MLX adapter saved to {config.output_model_path}")
    return config.output_model_path

def _run_usft_torch(
    texts: List[str],
    config: USFTConfig,
) -> str:
    dataset = Dataset.from_dict({"text": texts})

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

    if tokenizer.pad_token is None:
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

    use_bf16 = False
    if config.device == "cuda" and torch is not None:
        use_bf16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=config.output_model_path,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        optim=config.optim,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        fp16=False,
        bf16=use_bf16,
        lr_scheduler_type=config.lr_scheduler_type,
        save_steps=config.save_steps,
        weight_decay=config.weight_decay,
        no_cuda=(config.device == "cpu"),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        max_seq_length=config.max_length,
        dataset_text_field="text"
    )

    print(f"Starting USFT on {len(dataset)} texts (device={config.device})")
    trainer.train()

    trainer.save_model(config.output_model_path)
    print(f"Model saved to {config.output_model_path}")

    return config.output_model_path

def run_usft(
    texts: List[str],
    config: Optional[USFTConfig] = None
) -> str:

    if config is None:
        config = USFTConfig()

    if config.device == "mlx":
        return _run_usft_mlx(texts, config)
    else:
        return _run_usft_torch(texts, config)

def load_corpus_from_hf(dataset_name: str, split: str = "train"):

    ds = load_dataset(dataset_name, split=split)

    if "text" in ds.column_names:
        return ds["text"]
    elif "content" in ds.column_names:
        return ds["content"]
    else:
        return [str(item) for item in ds]
