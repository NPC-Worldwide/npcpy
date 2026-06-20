"""
npcpy.ft — Fine-tuning and model serving utilities.

Submodules:
    sft  — Supervised fine-tuning (SFT) with LoRA via MLX, torch/cpu, or torch/cuda.
    rl   — Reinforcement learning: DPO, GRPO, PPO via MLX or torch.
    usft — Unsloth-based SFT (optional, for fast training on consumer GPUs).

Quick-start:
    from npcpy.ft import SFTConfig, run_sft, RLConfig, train_with_dpo
    from npcpy.ft import export_adapter, convert_to_mlx, merge_and_save, download_from_hub

    cfg = SFTConfig(base_model_name="mlx-community/Qwen3-4B-4bit", device="mlx")
    adapter = run_sft(X, y, config=cfg, format_style="qwen3")

    export_adapter(adapter_path, output_path="./npcsh_q4.gguf", format="gguf")

    merge_and_save(adapter_path, base_model="Qwen/Qwen3-4B", output_path="./npcsh_full")

    download_from_hub("npc-worldwide/enpisi-coder", "adapters/npcsh-sft", path_in_repo="adapters/npcsh-sft")
"""

def __getattr__(name):
    if name in (
        "SFTConfig",
        "run_sft",
        "load_sft_model",
        "predict_sft",
        "format_training_examples",
    ):
        from .sft import (
            SFTConfig,
            run_sft,
            load_sft_model,
            predict_sft,
            format_training_examples,
        )
        return locals()[name]
    if name in ("RLConfig", "train_with_dpo", "train_with_grpo", "train_with_ppo"):
        from .rl import RLConfig, train_with_dpo, train_with_grpo, train_with_ppo
        return locals()[name]
    if name in ("UnslothSFTConfig", "run_unsloth_sft"):
        try:
            from .usft import UnslothSFTConfig, run_unsloth_sft
        except Exception:
            UnslothSFTConfig = None
            run_unsloth_sft = None
        return locals()[name]
    if name in ("merge_and_save", "export_adapter", "convert_to_mlx", "upload_to_hub", "download_from_hub"):
        from .export import (
            merge_and_save,
            export_adapter,
            convert_to_mlx,
            upload_to_hub,
            download_from_hub,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SFTConfig",
    "run_sft",
    "load_sft_model",
    "predict_sft",
    "format_training_examples",
    "RLConfig",
    "train_with_dpo",
    "train_with_grpo",
    "train_with_ppo",
    "merge_and_save",
    "export_adapter",
    "convert_to_mlx",
    "upload_to_hub",
    "download_from_hub",
]
