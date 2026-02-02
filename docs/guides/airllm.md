# AirLLM -- 70B+ Models on Consumer Hardware

AirLLM lets you run large language models (70B parameters and beyond) on machines with limited RAM by splitting the model into per-layer files and loading them one at a time during inference. npcpy integrates AirLLM as a first-class provider so you can use `get_llm_response` with huge HuggingFace models the same way you use Ollama or OpenAI.

## Installation

```bash
# Base install
pip install npcpy[airllm]

# macOS: MLX backend (no compression needed on Apple Silicon)
pip install mlx

# Linux: 4-bit CUDA backend via bitsandbytes
pip install bitsandbytes
```

## Basic Usage

```python
from npcpy.llm_funcs import get_llm_response

result = get_llm_response(
    "Explain quantum entanglement in simple terms.",
    model="Qwen/Qwen2.5-7B-Instruct",
    provider="airllm",
    max_tokens=128,
)

print(result["response"])
```

The first call downloads and splits the model to disk (this is slow). Subsequent calls load from the split cache and are much faster.

## macOS MLX Backend

On macOS (Apple Silicon), AirLLM uses the MLX framework. No compression is applied by default because unified memory on Apple Silicon is typically sufficient.

Under the hood:
1. The tokenizer encodes with `return_tensors="np"`.
2. Input IDs are converted to `mx.array` for MLX.
3. `air_model.generate()` returns the decoded string directly.

```python
result = get_llm_response(
    "What is the capital of France?",
    model="Qwen/Qwen2.5-7B-Instruct",
    provider="airllm",
    max_tokens=50,
)
```

## Linux CUDA Backend

On Linux with a CUDA GPU, AirLLM defaults to **4-bit compression** via bitsandbytes to fit large models in VRAM.

```python
result = get_llm_response(
    "Summarize the theory of relativity.",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    provider="airllm",
    compression="4bit",   # default on Linux
    max_tokens=200,
)
```

To disable compression (requires more memory):

```python
result = get_llm_response(
    "Hello!",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    provider="airllm",
    compression=None,
    max_tokens=50,
)
```

## Supported Models

Any sharded HuggingFace model works, as long as it has a `model.safetensors.index.json` file in the repo. Common choices:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-72B-Instruct`
- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`

## Model Caching

Models are cached at the module level in `_AIRLLM_MODEL_CACHE`, keyed by `"model_name:compression"`. The first load splits the model files to disk (this can take minutes for 70B models). After that, the split files are reused instantly.

```python
# First call: slow (downloads + splits)
r1 = get_llm_response("Hello", model="Qwen/Qwen2.5-7B-Instruct", provider="airllm")

# Second call: fast (loads from cache)
r2 = get_llm_response("World", model="Qwen/Qwen2.5-7B-Instruct", provider="airllm")
```

## HuggingFace Token

For gated models (e.g., Llama), pass your HuggingFace token via the `hf_token` keyword argument or the `HF_TOKEN` environment variable.

```python
# Via kwarg
result = get_llm_response(
    "Hello",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    provider="airllm",
    hf_token="hf_xxxxxxxxxxxxx",
)

# Or via environment variable
import os
os.environ["HF_TOKEN"] = "hf_xxxxxxxxxxxxx"

result = get_llm_response(
    "Hello",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    provider="airllm",
)
```

## Chat Template Support

AirLLM automatically applies the tokenizer's chat template when available. Messages are formatted using `tokenizer.apply_chat_template()` before being fed to the model. If no chat template exists, it falls back to a simple `role: content` format.

```python
result = get_llm_response(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
    ],
    model="Qwen/Qwen2.5-7B-Instruct",
    provider="airllm",
)
```

## JSON Format

Pass `format="json"` to instruct the model to return a JSON object. A system-level instruction is appended to guide the model.

```python
result = get_llm_response(
    "List 3 programming languages as a JSON array under the key 'languages'.",
    model="Qwen/Qwen2.5-7B-Instruct",
    provider="airllm",
    format="json",
    max_tokens=100,
)
print(result["response"])  # {"languages": ["Python", "Rust", "Go"]}
```

## Additional Parameters

The following keyword arguments are forwarded to `airllm.AutoModel.from_pretrained`:

| Parameter           | Description                                           |
|--------------------|-------------------------------------------------------|
| `compression`      | `"4bit"` (default on Linux) or `None`                 |
| `max_tokens`       | Maximum tokens to generate (default: 256)             |
| `temperature`      | Sampling temperature (default: 0.7)                   |
| `hf_token`         | HuggingFace Hub token for gated repos                 |
| `delete_original`  | Delete original weights after splitting               |
| `max_seq_len`      | Maximum sequence length                               |
| `prefetching`      | Enable layer prefetching for speed                    |

## Limitations

- **No tool calling**: AirLLM does not support function/tool calling.
- **No streaming**: Responses are returned in full, not streamed.
- **First load is slow**: Model splitting to disk is disk-intensive and can take several minutes for large models.
- **One inference at a time**: The layer-by-layer approach is inherently sequential. Throughput is lower than GPU-native inference.
