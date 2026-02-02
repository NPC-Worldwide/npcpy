from npcpy.llm_funcs import get_llm_response

# Test 1: Small model sanity check (7B, fast download)
response = get_llm_response(
    "What is the capital of France?",
    model="Qwen/Qwen2.5-7B-Instruct",
    provider="airllm",
    max_tokens=50,
)
print("7B response:", response["response"])

# Test 2: Llama 3.1 70B Instruct (4-bit compression)
response = get_llm_response(
    "Explain quantum computing in simple terms.",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    provider="airllm",
    max_tokens=100,
)
print("Llama 70B response:", response["response"])

# Test 3: Qwen 2.5 72B Instruct (4-bit compression)
response = get_llm_response(
    "What are the key differences between Python and Rust?",
    model="Qwen/Qwen2.5-72B-Instruct",
    provider="airllm",
    max_tokens=100,
)
print("Qwen 72B response:", response["response"])

# Test 4: JSON format parsing
response = get_llm_response(
    'Return a JSON object with keys "capital" and "country" for France.',
    model="Qwen/Qwen2.5-7B-Instruct",
    provider="airllm",
    format="json",
    max_tokens=50,
)
print("JSON response:", response["response"])
