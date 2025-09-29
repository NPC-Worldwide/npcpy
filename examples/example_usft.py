from npcpy.ft.usft import run_usft, load_corpus_from_hf, USFTConfig
from npcpy.ft.sft import load_sft_model, predict_sft


def demo_usft_from_text():
    print("\n=== USFT FROM RAW TEXT ===")
    
    texts = [
        "The cat sat on the mat.",
        "Dogs are loyal companions.",
        "Birds fly in the sky.",
        "Fish swim in the ocean.",
        "Elephants have long trunks.",
    ] * 50
    
    config = USFTConfig(
        base_model_name="Qwen/Qwen3-0.6B",
        output_model_path="models/usft_animals",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=2e-5
    )
    
    print(f"Training on {len(texts)} texts...")
    model_path = run_usft(texts, config)
    
    print("\nTesting USFT model...")
    model, tokenizer = load_sft_model(model_path)
    
    prompt = "The elephant"
    response = predict_sft(
        model,
        tokenizer,
        prompt,
        max_new_tokens=20,
        temperature=0.7
    )
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    return model_path


def demo_usft_from_hf():
    print("\n=== USFT FROM HUGGINGFACE DATASET ===")
    
    print("Loading tiny_shakespeare from HuggingFace...")
    texts = load_corpus_from_hf(
        "tiny_shakespeare",
        split="train[:100]"
    )
    
    config = USFTConfig(
        base_model_name="Qwen/Qwen3-0.6B",
        output_model_path="models/usft_shakespeare",
        num_train_epochs=2,
        max_length=256
    )
    
    print(f"Training on {len(texts)} Shakespeare texts...")
    model_path = run_usft(texts, config)
    
    print("\nTesting Shakespeare-adapted model...")
    model, tokenizer = load_sft_model(model_path)
    
    prompt = "To be or not to be"
    response = predict_sft(
        model,
        tokenizer,
        prompt,
        max_new_tokens=30,
        temperature=0.8
    )
    
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    
    return model_path


def demo_usft_code():
    print("\n=== USFT ON CODE ===")
    
    code_samples = [
        "def hello_world():\n    print('Hello, World!')",
        "def add(a, b):\n    return a + b",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
        "class Dog:\n    def __init__(self, name):\n        self.name = name",
        "for i in range(10):\n    print(i)",
    ] * 30
    
    config = USFTConfig(
        base_model_name="Qwen/Qwen3-0.6B",
        output_model_path="models/usft_python",
        num_train_epochs=3,
        learning_rate=1e-5
    )
    
    print(f"Training on {len(code_samples)} Python samples...")
    model_path = run_usft(code_samples, config)
    
    return model_path


if __name__ == "__main__":
    print("USFT Demo")
    print("=" * 50)
    
    demo_usft_from_text()
    
    demo_usft_from_hf()
    
    demo_usft_code()
    
    print("\n" + "=" * 50)
    print("Demo complete!")