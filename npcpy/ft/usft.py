from dataclasses import dataclass, field
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
except:
    Dataset = None
    load_dataset = None
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None
    SFTTrainer = None
    
from typing import List, Optional


@dataclass
class USFTConfig:
    base_model_name: str = "Qwen/Qwen3-0.6B"
    output_model_path: str = "models/usft_model"
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


def run_usft(
    texts: List[str],
    config: Optional[USFTConfig] = None
) -> str:
    
    if config is None:
        config = USFTConfig()
    
    dataset = Dataset.from_dict({"text": texts})
    
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name,
        trust_remote_code=True,
        attn_implementation="eager"
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
    
    training_args = TrainingArguments(
        output_dir=config.output_model_path,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=(
            config.per_device_train_batch_size
        ),
        gradient_accumulation_steps=(
            config.gradient_accumulation_steps
        ),
        optim=config.optim,
        logging_steps=config.logging_steps,
        learning_rate=config.learning_rate,
        fp16=False,
        bf16=torch.cuda.is_available(),
        lr_scheduler_type=config.lr_scheduler_type,
        save_steps=config.save_steps,
        weight_decay=config.weight_decay,
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        max_seq_length=config.max_length,
        dataset_text_field="text"
    )
    
    print(f"Starting USFT on {len(dataset)} texts")
    trainer.train()
    
    trainer.save_model(config.output_model_path)
    print(f"Model saved to {config.output_model_path}")
    
    return config.output_model_path


def load_corpus_from_hf(dataset_name: str, split: str = "train"):
    
    ds = load_dataset(dataset_name, split=split)
    
    if "text" in ds.column_names:
        return ds["text"]
    elif "content" in ds.column_names:
        return ds["content"]
    else:
        return [str(item) for item in ds]