# Fine-Tuning & Evolution

npcpy includes a complete fine-tuning toolkit under `npcpy.ft` that covers supervised and unsupervised language model fine-tuning, diffusion model training, reinforcement learning from agent traces, genetic algorithms, and model ensembling. Every module uses LoRA adapters by default so you can train on consumer hardware without touching the full model weights.

## Supervised Fine-Tuning (SFT)

Supervised fine-tuning trains a language model on input-output pairs. npcpy wraps HuggingFace Transformers, TRL, and PEFT into three functions: `run_sft` to train, `load_sft_model` to reload, and `predict_sft` to generate.

```python
from npcpy.ft.sft import run_sft, load_sft_model, predict_sft, SFTConfig

# Prepare paired training data
X_train = [
    "translate to french: hello",
    "translate to french: goodbye",
    "translate to french: thank you",
]
y_train = [
    "bonjour",
    "au revoir",
    "merci",
]

# Train with default config (google/gemma-3-270m-it base, LoRA r=8, 20 epochs)
model_path = run_sft(X_train, y_train)

# Or customize the config
config = SFTConfig(
    base_model_name="google/gemma-3-270m-it",
    output_model_path="models/translator",
    num_train_epochs=10,
    learning_rate=3e-5,
    lora_r=8,
    lora_alpha=16,
)
model_path = run_sft(X_train, y_train, config=config)

# Load the trained model and run inference
model, tokenizer = load_sft_model(model_path)
response = predict_sft(model, tokenizer, "translate to french: thanks")
print(response)
```

`run_sft` formats the data into chat-style turns (Gemma format by default, Llama format also supported via `format_style="llama"`), applies a LoRA adapter, and trains with the TRL `SFTTrainer`. The function returns the path where the adapter was saved.

### SFTConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model_name` | `"google/gemma-3-270m-it"` | HuggingFace model ID |
| `output_model_path` | `"models/sft_model"` | Where to save the adapter |
| `lora_r` | `8` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha scaling |
| `lora_dropout` | `0.15` | Dropout for LoRA layers |
| `num_train_epochs` | `20` | Training epochs |
| `per_device_train_batch_size` | `2` | Batch size per device |
| `learning_rate` | `3e-5` | Optimizer learning rate |
| `max_length` | `512` | Maximum sequence length |

## Unsupervised Fine-Tuning (USFT)

Unsupervised fine-tuning continues pretraining a language model on raw text without paired examples. This is useful for adapting a model to a specific domain or writing style.

```python
from npcpy.ft.usft import run_usft, load_corpus_from_hf, USFTConfig

# Load a corpus from HuggingFace
texts = load_corpus_from_hf("tiny_shakespeare", split="train[:1000]")

# Train with custom config
config = USFTConfig(
    base_model_name="Qwen/Qwen3-0.6B",
    output_model_path="models/shakespeare",
    num_train_epochs=3,
    learning_rate=2e-5,
)
model_path = run_usft(texts, config=config)
```

`load_corpus_from_hf` automatically extracts text from a HuggingFace dataset. It looks for a `text` column first, then `content`, and falls back to stringifying each row. `run_usft` uses the same LoRA + SFTTrainer pipeline as SFT but trains on the raw text field directly.

### USFTConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_model_name` | `"Qwen/Qwen3-0.6B"` | HuggingFace model ID |
| `output_model_path` | `"models/usft_model"` | Where to save the adapter |
| `num_train_epochs` | `3` | Training epochs |
| `per_device_train_batch_size` | `4` | Batch size per device |
| `max_length` | `512` | Maximum sequence length |

## Diffusion Fine-Tuning

npcpy includes a from-scratch diffusion model trainer built on a simple UNet architecture. This is designed for learning small image domains rather than fine-tuning Stable Diffusion.

```python
from npcpy.ft.diff import train_diffusion, generate_image, DiffusionConfig

# Paths to your training images and optional captions
image_paths = ["data/img1.png", "data/img2.png", "data/img3.png"]
captions = ["a red circle", "a blue square", "a green triangle"]

# Configure training
config = DiffusionConfig(
    image_size=128,
    channels=256,
    num_epochs=100,
    batch_size=4,
    learning_rate=1e-5,
    checkpoint_frequency=10,
    output_model_path="diffusion_model",
)

# Train the diffusion model
model_path = train_diffusion(image_paths, captions, config=config)

# Generate new images from the trained model
generated = generate_image(
    model_path + "/model_final.pt",
    prompt="a white square",
    num_samples=1,
    image_size=128,
)
generated.save("output.png")
```

The trainer saves checkpoints at a configurable frequency and a final `model_final.pt` at the end. You can resume training from a checkpoint by passing `resume_from` to `train_diffusion`.

### DiffusionConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_size` | `128` | Height and width of training images |
| `channels` | `256` | UNet channel width |
| `timesteps` | `1000` | Number of diffusion timesteps |
| `num_epochs` | `100` | Training epochs |
| `batch_size` | `4` | Training batch size |
| `learning_rate` | `1e-5` | Optimizer learning rate |
| `checkpoint_frequency` | `10` | Save checkpoint every N steps |
| `num_channels` | `3` | Image channels (3 for RGB) |

### Progress Callbacks

You can monitor training by passing a callback function:

```python
def on_progress(info):
    print(f"Epoch {info['epoch']}/{info['total_epochs']} "
          f"Step {info['step']} Loss {info['loss']:.6f}")

model_path = train_diffusion(image_paths, captions, config=config,
                             progress_callback=on_progress)
```

## Reinforcement Learning with DPO

The RL module collects execution traces from NPC agents, scores them with a reward function, creates preference pairs, and trains a LoRA adapter using Direct Preference Optimization (DPO).

```python
from npcpy.ft.rl import collect_traces, run_rl_training, RLConfig
from npcpy.npc_compiler import NPC

# Define tasks with expected outputs
tasks = [
    {"prompt": "Solve 2+2", "expected": "4"},
    {"prompt": "Solve 10*5", "expected": "50"},
    {"prompt": "Solve 100/4", "expected": "25"},
]

# Create agents to collect traces from
agents = [
    NPC(
        name="mathbot",
        primary_directive="Be concise. Provide only the numeric answer.",
        model="qwen3:0.6b",
        provider="ollama",
    ),
]

# Define a reward function that scores each trace
def reward_fn(trace):
    expected = trace["task_metadata"]["expected"]
    output = trace["final_output"]
    return 1.0 if expected in output else 0.0

# Run the full pipeline: collect traces -> create preference pairs -> DPO training
adapter_path = run_rl_training(tasks, agents, reward_fn)
```

The pipeline works in three stages:

1. **Trace collection** -- each agent executes each task and the reward function scores the result. Traces are saved to a CSV file with a timestamp.
2. **Preference pair creation** -- traces are split into high-reward and low-reward groups. Pairs are formed where the reward gap exceeds a configurable threshold (`min_reward_gap`, default 0.4).
3. **DPO training** -- a LoRA adapter is trained on the preference pairs using TRL's `DPOTrainer`.

You can also run the stages separately for more control:

```python
from npcpy.ft.rl import collect_traces, train_with_dpo, RLConfig

config = RLConfig(
    base_model_name="Qwen/Qwen3-0.6B",
    adapter_path="./math_adapter",
    max_iterations=8,
    min_reward_gap=0.4,
    num_train_epochs=20,
    learning_rate=1e-6,
    beta=0.5,
)

# Step 1: Collect traces
traces = collect_traces(tasks, agents, reward_fn, config)

# Step 2: Train with DPO
adapter_path = train_with_dpo(traces, config)
```

### Loading an RL-Trained Model

```python
from npcpy.ft.rl import load_rl_model

model, tokenizer = load_rl_model(
    base_model_id="Qwen/Qwen3-0.6B",
    adapter_path="./math_adapter",
    merge_adapter=True,
)
```

When `merge_adapter=True` the LoRA weights are folded into the base model for faster inference.

## Genetic Evolution

The `ge` module provides a generic genetic algorithm that can evolve any type of individual. You supply four functions -- `initialize_fn`, `fitness_fn`, `mutate_fn`, and `crossover_fn` -- and the evolver handles selection, elitism, and generational progression.

```python
from npcpy.ft.ge import GeneticEvolver, GAConfig
import random

# Example: evolve a list of floats to maximize their sum
def initialize():
    return [random.uniform(-1, 1) for _ in range(10)]

def fitness(individual):
    return sum(individual)

def mutate(individual):
    idx = random.randint(0, len(individual) - 1)
    individual[idx] += random.gauss(0, 0.1)
    return individual

def crossover(parent1, parent2):
    split = random.randint(1, len(parent1) - 1)
    return parent1[:split] + parent2[split:]

config = GAConfig(
    population_size=20,
    generations=50,
    mutation_rate=0.15,
    crossover_rate=0.7,
    tournament_size=3,
    elitism_count=2,
)

evolver = GeneticEvolver(
    fitness_fn=fitness,
    mutate_fn=mutate,
    crossover_fn=crossover,
    initialize_fn=initialize,
    config=config,
)

best = evolver.run()
print(f"Best individual: {best}")
print(f"Best fitness: {fitness(best):.3f}")
```

### GAConfig Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `population_size` | `20` | Number of individuals per generation |
| `generations` | `50` | Number of generations to evolve |
| `mutation_rate` | `0.15` | Probability of mutating a child |
| `crossover_rate` | `0.7` | Probability of crossover vs cloning |
| `tournament_size` | `3` | Individuals per tournament selection |
| `elitism_count` | `2` | Top individuals preserved unchanged |

### Inspecting Evolution History

After a run, the evolver stores statistics for each generation:

```python
for i, gen in enumerate(evolver.history):
    print(f"Gen {i}: best={gen['best_fitness']:.3f} avg={gen['avg_fitness']:.3f}")
```

## Model Ensembler / Response Router

The model ensembler implements a System 1 / System 2 routing pattern inspired by dual-process theory. Fast, specialized models handle queries they are confident about (System 1 -- gut reaction), and a full reasoning model handles everything else (System 2 -- deliberate thought).

```python
from npcpy.ft.model_ensembler import ResponseRouter, create_model_genome

# Create a genome of specialized model genes
genome = create_model_genome(["math", "code", "factual"])

# Each gene has auto-generated trigger patterns:
# - math: ['calculate', 'solve', 'equation', 'number']
# - code: ['function', 'class', 'bug', 'debug', 'code']
# - factual: ['what is', 'who is', 'when did', 'where is']

# Create the router with confidence thresholds
router = ResponseRouter(
    fast_threshold=0.8,      # System 1 must exceed this to skip deliberation
    ensemble_threshold=0.6,  # Ensemble must exceed this to avoid full reasoning
)

# Route a query
result = router.route_query("What is 2+2?", genome)
print(result["response"])
print(f"Used fast path: {result['used_fast_path']}")
print(f"Confidence: {result['confidence']}")
```

### How Routing Works

1. **System 1 (Fast Path)** -- the router checks if any gene's trigger patterns match the query. If a matching gene has a trained SFT model and its confidence threshold exceeds `fast_threshold`, the response is returned immediately.
2. **Ensemble** -- if the fast path does not fire, all genes with trained models vote on the answer. If the average confidence exceeds `ensemble_threshold`, the highest-weight response is returned.
3. **System 2 (Full Reasoning)** -- if neither fast path nor ensemble is confident enough, the query falls through to a full LLM call via `get_llm_response`.

### Custom Model Genes

You can assign trained model paths to genes for real routing:

```python
from npcpy.ft.model_ensembler import ModelGene

math_gene = ModelGene(
    sft_path="models/math_sft",
    base_model="Qwen/Qwen3-0.6B",
    specialization="math",
    trigger_patterns=["calculate", "solve", "equation", "sum", "multiply"],
    confidence_threshold=0.85,
)
```

### Evolving the Router

The model ensembler integrates with the genetic evolution module. You can evolve genomes to find the best combination of specialists, thresholds, and trigger patterns:

```python
from npcpy.ft.model_ensembler import (
    evaluate_model_genome,
    mutate_model_genome,
    crossover_model_genomes,
    create_model_genome,
    ResponseRouter,
)
from npcpy.ft.ge import GeneticEvolver, GAConfig

router = ResponseRouter(fast_threshold=0.8)

test_cases = [
    {"query": "What is 2+2?", "ground_truth": "4"},
    {"query": "Write a hello world function", "ground_truth": "def hello"},
]

evolver = GeneticEvolver(
    fitness_fn=lambda genome: evaluate_model_genome(genome, test_cases, router),
    mutate_fn=mutate_model_genome,
    crossover_fn=crossover_model_genomes,
    initialize_fn=lambda: create_model_genome(["math", "code", "factual"]),
    config=GAConfig(population_size=10, generations=20),
)

best_genome = evolver.run()
```

Mutations can adjust confidence thresholds, add trigger patterns, remove genes, or duplicate genes with variant specializations. Crossover splices two genomes at a random point.
