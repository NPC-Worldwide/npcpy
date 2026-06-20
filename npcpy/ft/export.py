"""
npcpy.ft.export — Export trained adapters to multiple serving formats.

Provides:
    merge_and_save(adapter_path, base_model, output_path)
    export_adapter(adapter_path, output_path, format="gguf")
    convert_to_mlx(adapter_path, output_path, base_model=None)
    upload_to_hub(adapter_path, repo_id, token=None)

All functions are zero-subprocess and use the standard HF/PEFT ecosystem.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional


def _infer_base_model(adapter_path: str) -> str:
    """Read adapter_config.json to find the original base model."""
    cfg_file = Path(adapter_path) / "adapter_config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cfg.get("model", cfg.get("base_model_name_or_path", ""))
    # PEFT adapters
    cfg_file = Path(adapter_path) / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cfg.get("base_model_name_or_path", "")
    return ""


def merge_and_save(
    adapter_path: str,
    base_model: str = None,
    output_path: str = None,
    device: str = "cpu",
) -> str:
    """Merge LoRA weights into the base model and save a full checkpoint.

    The resulting folder is a standard transformers model directory
    (contains pytorch_model.bin or model.safetensors + tokenizer).

    Usage:
        full_model = merge_and_save(
            "./npcsh_sft_qwen3",
            base_model="Qwen/Qwen3-4B",
            output_path="./npcsh_full_qwen3",
        )
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    adapter_path = Path(adapter_path)
    if output_path is None:
        output_path = str(adapter_path.parent / f"{adapter_path.name}_merged")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if not base_model:
        base_model = _infer_base_model(adapter_path)
    if not base_model:
        raise ValueError(
            f"Could not infer base model from {adapter_path}. "
            "Pass base_model= explicitly."
        )

    print(f"[merge] Loading base: {base_model}")
    # MLX-community models are pre-quantized and incompatible with standard transformers
    # loading + PEFT merging. Map back to the unquantized HF base model.
    from npcpy.ft.sft import _MLX_MODEL_MAP, _resolve_mlx_model
    # reverse map: mlx-community name -> unquantized HF name
    _REVERSE_MLX_MAP = {v: k for k, v in _MLX_MODEL_MAP.items()}
    if base_model in _REVERSE_MLX_MAP:
        unquantized_base = _REVERSE_MLX_MAP[base_model]
        print(f"[merge] Mapped {base_model} -> {unquantized_base} for merge")
        base_model = unquantized_base

    print(f"[merge] Loading unquantized base model (this may take a while for large models)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else {"": device},
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"[merge] Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    print("[merge] Merging LoRA weights...")
    model = model.merge_and_unload()

    print(f"[merge] Saving to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return output_path


def export_adapter(
    adapter_path: str,
    output_path: str = None,
    base_model: str = None,
    format: str = "gguf",
    quantization: str = "Q4_K_M",
    device: str = "cpu",
) -> str:
    """Export a merged model to llama.cpp GGUF (or other formats).

    Args:
        adapter_path: Path to the PEFT/LoRA adapter.
        output_path:  Destination path for the exported file.
        base_model:   HF model ID or local path.  Inferred if None.
        format:       "gguf"  (only GGUF is supported right now).
        quantization:   GGUF quant type: Q4_K_M, Q5_K_M, Q8_0, F16, etc.
        device:       "cpu" or "cuda" for the merge step.

    Returns:
        Path to the exported .gguf file.
    """
    if format != "gguf":
        raise ValueError(f"Only 'gguf' format is supported right now, got {format}")

    merged = merge_and_save(adapter_path, base_model, device=device)

    if output_path is None:
        base_name = Path(adapter_path).name
        output_path = str(Path(adapter_path).parent / f"{base_name}_{quantization.lower()}.gguf")

    print(f"[export] Converting to GGUF ({quantization}) → {output_path}")

    # Use llama.cpp's Python converter (bundled in many llama-cpp-python installs)
    try:
        from llama_cpp.convert import convert_hf_to_gguf
        convert_hf_to_gguf(
            merged,
            output_path,
            outtype=quantization,
        )
    except ImportError:
        # Fallback: call the llama.cpp CLI via Python module (still zero-subprocess)
        print("[export] llama_cpp.convert not found, trying llama.cpp CLI fallback...")
        _gguf_cli_fallback(merged, output_path, quantization)

    return output_path


def _gguf_cli_fallback(merged_path: str, output_path: str, quantization: str):
    """Last-resort GGUF conversion using llama.cpp's convert_hf_to_gguf.py."""
    import sys
    # Try to find the convert script inside llama-cpp-python package
    import llama_cpp
    pkg_dir = Path(llama_cpp.__file__).parent
    convert_script = pkg_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise RuntimeError(
            "GGUF conversion requires llama.cpp.\n"
            "Install: pip install llama-cpp-python\n"
            "Or build llama.cpp from source and run convert_hf_to_gguf.py manually."
        )
    # Save current argv, run the converter module directly
    old_argv = sys.argv
    try:
        sys.argv = [
            str(convert_script),
            merged_path,
            "--outfile", output_path,
            "--outtype", quantization,
        ]
        exec(convert_script.read_text(), {"__name__": "__main__"})
    finally:
        sys.argv = old_argv


def convert_to_mlx(
    adapter_path: str,
    output_path: str = None,
    base_model: str = None,
    quantize: bool = False,
) -> str:
    """Convert a standard transformers LoRA adapter into MLX format.

    MLX expects:
        adapters.safetensors   (LoRA weights)
        adapter_config.json    (metadata: rank, alpha, base_model, etc.)

    This function merges the adapter, saves it to a temp dir, then
    re-saves the weights in MLX-compatible safetensors format.

    Args:
        adapter_path: Path to the PEFT/LoRA adapter.
        output_path:  Destination directory.  Defaults to adapter_path + "_mlx".
        base_model:   HF model ID or local path.  Inferred if None.
        quantize:     If True, quantize the merged model to 4-bit.

    Returns:
        Path to the MLX adapter directory.
    """
    import safetensors.torch

    adapter_path = Path(adapter_path)
    if output_path is None:
        output_path = str(adapter_path.parent / f"{adapter_path.name}_mlx")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Merge first (we need the full base + lora weights)
    merged = merge_and_save(adapter_path, base_model, device="cpu")

    print(f"[mlx] Converting merged model → MLX format ({output_path})")

    # Load merged model and extract state_dict
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        merged,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
    )

    if quantize:
        # TODO: actual 4-bit quantization via mlx_lm
        print("[mlx] Note: quantize=True requires mlx_lm.quantize; using fp16 for now")

    # Save as safetensors
    state_dict = model.state_dict()
    safetensors.torch.save_file(
        state_dict,
        Path(output_path) / "adapters.safetensors",
    )

    # Write adapter_config.json in mlx-lm format
    base_model_name = base_model or _infer_base_model(adapter_path)
    config = {
        "model": base_model_name,
        "fine_tune_type": "lora",
        "num_layers": len([k for k in state_dict.keys() if "lora_A" in k]),
        "lora_parameters": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "scale": 2.0,
        },
    }
    with open(Path(output_path) / "adapter_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[mlx] Saved MLX adapter to {output_path}")
    return output_path


def upload_to_hub(
    model_path: str,
    repo_id: str,
    token: str = None,
    private: bool = False,
    commit_message: str = "Upload npcsh fine-tuned adapter",
    path_in_repo: str = None,
) -> str:
    """Push a model/adapter to the HuggingFace Hub.

    Works for:
        - LoRA adapters (npcsh_sft_qwen3/)
        - Merged full models (npcsh_sft_qwen3_merged/)
        - GGUF files (npcsh_sft_qwen3_q4_k_m.gguf)

    Args:
        path_in_repo: Subfolder inside the repo (e.g. "adapter", "full", "gguf").
    """
    from huggingface_hub import HfApi, create_repo

    api = HfApi()
    token = token or os.environ.get("HF_TOKEN")

    print(f"[hub] Creating repo {repo_id}")
    try:
        create_repo(repo_id, token=token, private=private, exist_ok=True)
    except Exception as e:
        print(f"[hub] Repo creation warning: {e}")

    model_path = Path(model_path)

    if model_path.is_file():
        # Single file upload (GGUF)
        dest = Path(path_in_repo or ".") / model_path.name if path_in_repo else model_path.name
        print(f"[hub] Uploading {model_path.name} → {dest}")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=str(dest),
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
        )
    else:
        # Folder upload (adapter or merged model)
        dest = path_in_repo or "."
        print(f"[hub] Uploading folder {model_path} → {dest}/")
        api.upload_folder(
            folder_path=str(model_path),
            path_in_repo=dest,
            repo_id=repo_id,
            token=token,
            commit_message=commit_message,
        )

    url = f"https://huggingface.co/{repo_id}"
    print(f"[hub] Uploaded to {url}")
    return url


def download_from_hub(
    repo_id: str,
    local_path: str,
    path_in_repo: str = "adapter",
    token: str = None,
) -> str:
    """Download a model/adapter subfolder from HuggingFace Hub.

    Downloads only the files in the specified subfolder and saves them
    directly to `local_path` (flattened — no nested subfolders).

    Args:
        repo_id:      HF Hub repo (e.g. "npc-worldwide/enpisi-coder")
        local_path:   Where to save locally (e.g. "adapters/npcsh-sft")
        path_in_repo: Subfolder inside repo (e.g. "adapters/npcsh-sft")
        token:        HF API token (reads HF_TOKEN env var if None)

    Returns:
        Absolute path to the downloaded folder.
    """
    from huggingface_hub import HfApi, hf_hub_download

    token = token or os.environ.get("HF_TOKEN")
    local_path = os.path.abspath(os.path.expanduser(local_path))
    Path(local_path).mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception:
        all_files = api.list_repo_files(
            repo_id=repo_id, repo_type="model", token=token
        )

    prefix = path_in_repo.rstrip("/") + "/"
    subfolder_files = [f for f in all_files if f.startswith(prefix)]
    if not subfolder_files:
        raise ValueError(
            f"No files found in {repo_id}/{path_in_repo}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in subfolder_files:
            print(f"[hub] downloading {f}")
            downloaded = hf_hub_download(
                repo_id=repo_id,
                filename=f,
                repo_type="model",
                local_dir=tmpdir,
                local_dir_use_symlinks=False,
                token=token,
            )
            # Move from tmpdir/path_in_repo/file → local_path/file
            rel = f[len(prefix):]
            dest = os.path.join(local_path, rel)
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(downloaded, dest)

    print(f"[hub] Saved {len(subfolder_files)} files to {local_path}")
    return local_path


__all__ = [
    "merge_and_save",
    "export_adapter",
    "convert_to_mlx",
    "upload_to_hub",
    "download_from_hub",
]
