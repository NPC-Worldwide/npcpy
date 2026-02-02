# NPCArray -- NumPy for AI

NPCArray is a vectorized abstraction for **populations of models**. It wraps LLMs, scikit-learn estimators, PyTorch modules, and NPC agents into a single NumPy-like container, then lets you broadcast prompts, chain transformations, and reduce results with one unified API. Operations are lazy: they build a computation graph that only executes when you call `.collect()` (or its alias `.compute()`).

```python
from npcpy.npc_array import NPCArray

models = NPCArray.from_llms(
    ['llama3.2', 'gemma3:1b'],
    providers='ollama'
)

result = models.infer("What is 2+2? Just the number.").collect()
print(result.shape)        # (2, 1) -- 2 models, 1 prompt
print(result.data[0, 0])   # response from llama3.2
print(result.data[1, 0])   # response from gemma3:1b
```

## Factory Methods

### from_llms

Create an array from LLM model names. A single string is treated as a length-1 array.

```python
# Multiple models, single provider broadcast to all
models = NPCArray.from_llms(['llama3.2', 'gemma3:1b'], providers='ollama')

# Per-model providers
models = NPCArray.from_llms(
    ['gpt-4', 'claude-3-opus', 'llama3.2'],
    providers=['openai', 'anthropic', 'ollama']
)

# Single model -- still array-like
model = NPCArray.from_llms('llama3.2')
print(model.shape)  # (1,)
```

### from_sklearn

Wrap fitted scikit-learn models.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

rf = RandomForestClassifier(n_estimators=50).fit(X_train, y_train)
lr = LogisticRegression().fit(X_train, y_train)

models = NPCArray.from_sklearn([rf, lr])
```

### from_torch

Wrap PyTorch `nn.Module` instances.

```python
models = NPCArray.from_torch([my_bert, my_gpt], device="cuda")
```

### from_npcs

Wrap NPC objects loaded from `.npc` files.

```python
from npcpy.npc_compiler import NPC

analyst = NPC(name="analyst", model="gpt-4", provider="openai")
writer  = NPC(name="writer",  model="claude-3-opus", provider="anthropic")

team = NPCArray.from_npcs([analyst, writer])
```

### meshgrid

Cartesian product over parameter ranges. Every combination becomes one entry.

```python
configs = NPCArray.meshgrid(
    model=['llama3.2', 'gemma3:1b'],
    temperature=[0.0, 0.5, 1.0]
)
print(configs.shape)  # (6,) -- 2 models x 3 temperatures

result = configs.infer("Complete: The quick brown fox").collect()
for i, spec in enumerate(configs.specs):
    print(f"{spec.model_ref} @ temp={spec.config['temperature']}: {result.data[i, 0][:60]}...")
```

### from_matrix

Explicit list of configurations. Useful in Jinx templates.

```python
matrix = [
    {'model': 'gpt-4',    'provider': 'openai',    'temperature': 0.7},
    {'model': 'llama3.2', 'provider': 'ollama',     'temperature': 0.5},
    {'model': 'claude-3-opus', 'provider': 'anthropic', 'temperature': 0.3},
]
models = NPCArray.from_matrix(matrix)
```

## Lazy Evaluation

All operations return a `LazyResult` that records work without performing it. Nothing touches a model until `.collect()` runs. This lets you build pipelines, inspect them, and only pay the cost once.

```python
pipeline = (
    models
    .infer("Return a JSON object with 'answer': 42")
    .map(lambda r: r.strip())
    .filter(lambda r: '{' in r)
)

# Nothing has executed yet. Inspect the plan:
pipeline.explain()
# Computation Graph:
#   +-- filter: shape=None, params={}
#     +-- map: shape=(2, 1), params={}
#       +-- infer: shape=(2, 1), params={'prompts': [...]}
#         +-- source: shape=(2,), params={...}

# Now materialize:
tensor = pipeline.collect()
```

## Chaining Operations

### infer / predict / forward

Queue the appropriate call for the model type.

```python
# LLMs
lazy = models.infer(["Capital of France?", "Capital of Japan?"])

# sklearn
lazy = sklearn_array.predict(X_test)

# PyTorch
lazy = torch_array.forward(input_tensor)
```

### map

Apply a function element-wise to every response.

```python
lengths = models.infer("Tell me a joke.").map(len).collect()
print(lengths.data)  # array of ints
```

### filter

Keep only responses matching a predicate.

```python
good = models.infer("Write a haiku.").filter(lambda r: len(r) > 20).collect()
```

### chain

Feed all responses through a synthesis function, optionally for multiple rounds (debate / refinement pattern).

```python
def debate_round(responses):
    combined = "\n".join(f"- {r}" for r in responses)
    return f"Consider these views:\n{combined}\n\nProvide a refined answer."

result = models.infer("Should AI be open source?").chain(debate_round, n_rounds=2).collect()
```

## Reduction Methods

Reduce collapses the model axis (axis=0) or the prompt axis (axis=1) into a single value per remaining dimension.

| Method       | What it does                                      |
|-------------|---------------------------------------------------|
| `vote`      | Majority voting (most common response wins)       |
| `consensus` | LLM-synthesized summary of all perspectives       |
| `mean`      | Numeric average (for numeric outputs)             |
| `concat`    | Join all responses with `\n---\n`                 |
| `best`      | Select by external score list                     |

```python
# Majority vote across 3 models
answer = (
    NPCArray.from_llms(['llama3.2', 'llama3.2', 'gemma3:1b'], providers='ollama')
    .infer("Is Python compiled or interpreted? One word.")
    .vote(axis=0)
    .collect()
)
print(answer.data[0])  # "interpreted"

# LLM consensus
synthesis = (
    models
    .infer("Pros and cons of microservices?")
    .consensus(axis=0, model='llama3.2')
    .collect()
)
print(synthesis.data[0])
```

## explain()

Print the computation graph before executing.

```python
lazy = models.infer(["p1", "p2"]).map(str.upper).vote()
lazy.explain()
```

## ResponseTensor

`.collect()` returns a `ResponseTensor` -- a thin wrapper around a NumPy object array with shape metadata.

```python
tensor = models.infer(["Hello", "World"]).collect()

tensor.shape          # (2, 2) -- 2 models x 2 prompts
tensor.data           # numpy ndarray of strings
tensor[0]             # ResponseTensor for model 0
tensor[:, 1]          # ResponseTensor for prompt 1
tensor.tolist()       # nested Python list
tensor.flatten()      # flat Python list
tensor.model_specs    # list of ModelSpec objects
tensor.prompts        # the prompts used
```

## Matrix Sampling via get_llm_response

The `matrix` and `n_samples` parameters on `get_llm_response` provide a lower-level way to sweep parameters without constructing an `NPCArray`.

```python
from npcpy.llm_funcs import get_llm_response

# 2 models x 2 temperatures = 4 runs
result = get_llm_response(
    "Write a creative opening line.",
    matrix={
        'model': ['llama3.2', 'gemma3:1b'],
        'temperature': [0.5, 1.0]
    }
)
for run in result['runs']:
    print(f"{run['combo']}: {run['response'][:60]}...")

# n_samples: repeat the same config multiple times
result = get_llm_response(
    "Give me a random number between 1 and 100.",
    model='llama3.2',
    provider='ollama',
    n_samples=3
)
for run in result['runs']:
    print(f"Sample {run['sample_index']}: {run['response']}")
```

## sklearn Integration

```python
from npcpy.npc_array import NPCArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np

np.random.seed(42)
X = np.random.randn(100, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

rf  = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
lr  = LogisticRegression(random_state=42).fit(X_train, y_train)
svc = SVC(random_state=42).fit(X_train, y_train)

models = NPCArray.from_sklearn([rf, lr, svc])

predictions = models.predict(X_test).collect()
print(predictions.shape)   # (3, 20)
print(predictions.data[0]) # RandomForest predictions
print(predictions.data[1]) # LogisticRegression predictions
print(predictions.data[2]) # SVC predictions
```

## evolve() -- Genetic Evolution of Model Populations

Evolve a population by fitness. Elites survive unchanged; the rest are bred via mutation and crossover.

```python
from npcpy.npc_array import NPCArray, ModelSpec
import random

# Initial random population
specs = [
    ModelSpec(model_type="llm", model_ref="llama3.2",
              config={"temperature": random.uniform(0, 1)})
    for _ in range(10)
]
population = NPCArray(specs)

# Score each config (your evaluation logic here)
fitness = [random.random() for _ in range(10)]

# Mutation function
def mutate(spec):
    new_temp = max(0.0, min(1.0, spec.config["temperature"] + random.uniform(-0.2, 0.2)))
    return ModelSpec(model_type="llm", model_ref=spec.model_ref,
                     config={"temperature": new_temp})

evolved = population.evolve(fitness, mutate_fn=mutate, elite_ratio=0.2)
```

## Polars Integration

Register a `df.npc` namespace on Polars DataFrames.

```python
import polars as pl
from npcpy.npc_array import NPCArray, register_polars_namespace, npc_udf

register_polars_namespace()

models = NPCArray.from_llms('llama3.2')

df = pl.DataFrame({"text": ["Hello", "How are you?", "What is AI?"]})

# Option 1: namespace
result = df.npc.infer(models, 'text', output_col='response')

# Option 2: UDF
result = df.with_columns(
    npc_udf('infer', models, pl.col('text')).alias('response')
)
```

## Standalone Utilities

Quick helpers that skip the array construction.

```python
from npcpy.npc_array import infer_matrix, ensemble_vote

# Matrix inference: (n_models, n_prompts) tensor
tensor = infer_matrix(
    prompts=["Hello", "Goodbye"],
    models=['llama3.2'],
    providers=['ollama']
)
print(tensor.shape)  # (1, 2)

# One-shot ensemble vote
answer = ensemble_vote(
    "Capital of France? One word.",
    models=['llama3.2', 'gemma3:1b'],
    providers=['ollama', 'ollama']
)
print(answer)  # "Paris"
```

## Use Cases

- **Benchmarking**: Sweep models and temperatures with `meshgrid`, compare outputs side-by-side.
- **A/B testing**: Run the same prompts through two model versions, then `vote()` or inspect differences.
- **Ensemble production inference**: Combine 3+ models with `vote()` or `consensus()` for higher-quality answers.
- **Prompt optimization**: Use `n_samples` to test prompt variants, score results, feed scores into `evolve()`.
- **Heterogeneous ensembles**: Mix LLMs with sklearn classifiers in the same pipeline via `from_matrix`.
