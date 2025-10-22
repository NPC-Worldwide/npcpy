# structured fine tuning of LLMs to produce structured output
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
except:
    torch = None
    SFTTrainer = None
    LoraConfig = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TrainingArguments = None

from typing import List, Dict, Any, Optional


@dataclass
class SFTConfig:
    base_model_name: str = "google/gemma-3-270m-it"
    output_model_path: str = "models/sft_model"
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
            text = (
                f"<start_of_turn>user\n{inp}<end_of_turn>\n"
                f"<start_of_turn>model\n{out}<end_of_turn>"
            )
        elif format_style == "llama":
            text = (
                f"<|begin_of_text|><|start_header_id|>user"
                f"<|end_header_id|>\n\n{inp}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>"
                f"\n\n{out}<|eot_id|>"
            )
        else:
            text = f"Input: {inp}\nOutput: {out}"
        
        formatted.append({"text": text})
    
    return formatted


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
    
    formatted_examples = format_training_examples(
        X, y, format_style
    )
    
    if validation_split > 0:
        split_idx = int(len(formatted_examples) * (1 - validation_split))
        train_examples = formatted_examples[:split_idx]
        val_examples = formatted_examples[split_idx:]
        print(
            f"Split: {len(train_examples)} train, "
            f"{len(val_examples)} val"
        )
    else:
        train_examples = formatted_examples
        val_examples = []
    
    dataset = Dataset.from_list(train_examples)
    
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
        fp16=config.fp16,
        bf16=config.bf16,
        lr_scheduler_type=config.lr_scheduler_type,
        group_by_length=True,
        save_steps=config.save_steps,
        weight_decay=config.weight_decay,
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
    
    print(f"Training on {len(dataset)} examples")
    trainer.train()
    
    trainer.save_model(config.output_model_path)
    print(f"Model saved to {config.output_model_path}")
    
    return config.output_model_path


def load_sft_model(model_path: str):

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
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
    temperature: float = 0.7
) -> str:

    device = next(model.parameters()).device
    
    formatted_prompt = (
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )
    
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    
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