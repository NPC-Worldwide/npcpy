# Model Export & Hub

The `npcpy.ft.export` module provides zero-subprocess helpers for moving trained adapters into serving formats and publishing them.

```python
from npcpy.ft import (
    merge_and_save,
    export_adapter,
    convert_to_mlx,
    upload_to_hub,
    download_from_hub,
)
```

## Merge LoRA into a full model

`merge_and_save` folds a LoRA adapter back into its base model and saves a standard transformers checkpoint.

```python
full_model_path = merge_and_save(
    "adapters/npcsh_sft_qwen3",
    base_model="Qwen/Qwen3-4B",
    output_path="models/npcsh_full_qwen3",
)
```

The base model is inferred from the adapter's `adapter_config.json` when not provided. MLX-community IDs are automatically mapped back to their unquantized HuggingFace equivalents for merging.

## Export to GGUF

`export_adapter` first merges the adapter, then converts the merged model to a llama.cpp GGUF file.

```python
gguf_path = export_adapter(
    "adapters/npcsh_sft_qwen3",
    output_path="models/npcsh_qwen3_q4_k_m.gguf",
    format="gguf",
    quantization="Q4_K_M",
)
```

Only `format="gguf"` is supported right now. The function requires `llama-cpp-python` or a llama.cpp installation with `convert_hf_to_gguf.py` available.

## Convert to MLX format

`convert_to_mlx` turns a standard transformers LoRA adapter into the MLX adapter layout expected by `mlx_lm`:

```python
mlx_adapter = convert_to_mlx(
    "adapters/npcsh_sft_qwen3",
    output_path="adapters/npcsh_sft_qwen3_mlx",
    base_model="Qwen/Qwen3-4B",
)
```

The output directory contains `adapters.safetensors` and `adapter_config.json` compatible with `mlx_lm.load(..., adapter_path=...)`.

## Upload to HuggingFace Hub

`upload_to_hub` pushes an adapter, merged model, or GGUF file to a Hub repo:

```python
url = upload_to_hub(
    "adapters/npcsh_sft_qwen3",
    repo_id="npc-worldwide/npcsh-sft-qwen3",
    token=os.environ.get("HF_TOKEN"),
    path_in_repo="adapter",
)
```

For a single GGUF file, set `path_in_repo="gguf"` or leave it at the repo root.

## Download from HuggingFace Hub

`download_from_hub` downloads a subfolder from a Hub repo directly into a local directory:

```python
local_path = download_from_hub(
    "npc-worldwide/npcsh-sft-qwen3",
    local_path="adapters/npcsh_sft_qwen3",
    path_in_repo="adapter",
)
```

This is used by `load_sft_model` when you pass a Hub repo ID instead of a local path.
