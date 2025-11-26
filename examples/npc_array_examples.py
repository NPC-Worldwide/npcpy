"""
NPCArray Examples - NumPy for AI

This module demonstrates the NPCArray system, which provides a NumPy-like interface
for working with populations of models (LLMs, sklearn, PyTorch) at scale.

Key concepts:
- Model arrays support vectorized operations
- Operations are lazy until .collect() is called (like Spark)
- Same interface works for single models (treated as length-1 arrays)
- Supports ensemble voting, consensus, evolution, and more
"""

# =============================================================================
# BASIC USAGE
# =============================================================================

def example_basic_inference():
    """Basic inference across multiple LLMs"""
    from npcpy.npc_array import NPCArray

    # Create array of LLMs
    models = NPCArray.from_llms(
        ['llama3.2', 'gemma3:1b'],
        providers=['ollama', 'ollama']
    )

    print(f"Model array shape: {models.shape}")  # (2,)

    # Inference across all models
    result = models.infer("What is 2+2? Just the number.").collect()

    print(f"Result shape: {result.shape}")  # (2, 1) - 2 models, 1 prompt
    print(f"Model 1 response: {result.data[0, 0]}")
    print(f"Model 2 response: {result.data[1, 0]}")


def example_multiple_prompts():
    """Inference with multiple prompts - full matrix"""
    from npcpy.npc_array import NPCArray

    models = NPCArray.from_llms(['llama3.2', 'gemma3:1b'])

    prompts = [
        "Capital of France?",
        "Capital of Japan?",
        "Capital of Brazil?"
    ]

    result = models.infer(prompts).collect()

    print(f"Result shape: {result.shape}")  # (2, 3) - 2 models × 3 prompts

    # Access like numpy
    print(f"Model 0, all prompts: {result[0].data}")
    print(f"All models, prompt 0: {result[:, 0].data}")


def example_single_model_as_array():
    """Single model treated as length-1 array - no special cases"""
    from npcpy.npc_array import NPCArray

    # Single model, but still array-like
    model = NPCArray.from_llms('llama3.2')

    print(f"Shape: {model.shape}")  # (1,)

    # Same interface
    result = model.infer("Hello!").collect()
    print(f"Result shape: {result.shape}")  # (1, 1)


# =============================================================================
# LAZY EVALUATION & CHAINING
# =============================================================================

def example_lazy_chain():
    """Build computation graph without executing"""
    from npcpy.npc_array import NPCArray
    import json

    models = NPCArray.from_llms(['llama3.2', 'gemma3:1b'])

    # Build lazy computation graph
    result = (
        models
        .infer("Return a JSON object with 'answer': 42")
        .map(lambda r: r.strip())  # Clean whitespace
        .filter(lambda r: '{' in r)  # Keep only JSON-like
    )

    # Nothing executed yet! Show the plan
    print("Computation graph:")
    result.explain()

    # Now execute
    computed = result.collect()
    print(f"\nResult: {computed.data}")


def example_map_filter():
    """Map and filter operations"""
    from npcpy.npc_array import NPCArray

    models = NPCArray.from_llms(['llama3.2', 'gemma3:1b'])

    # Get responses
    responses = models.infer([
        "Write a one-sentence story.",
        "Write a haiku."
    ])

    # Map to get lengths
    lengths = responses.map(len).collect()
    print(f"Response lengths: {lengths.data}")

    # Filter by length
    long_responses = responses.filter(lambda r: len(r) > 50).collect()
    print(f"Long responses: {long_responses.shape}")


# =============================================================================
# ENSEMBLE OPERATIONS
# =============================================================================

def example_voting():
    """Majority voting across models"""
    from npcpy.npc_array import NPCArray

    # Create diverse model ensemble
    models = NPCArray.from_llms(
        ['llama3.2', 'llama3.2', 'gemma3:1b'],  # Can repeat for sampling
        providers=['ollama', 'ollama', 'ollama']
    )

    # Get consensus via voting
    result = (
        models
        .infer("Is Python a compiled or interpreted language? One word.")
        .vote(axis=0)  # Vote across models
        .collect()
    )

    print(f"Voted answer: {result.data[0]}")


def example_consensus():
    """LLM-based consensus synthesis"""
    from npcpy.npc_array import NPCArray

    models = NPCArray.from_llms(['llama3.2', 'gemma3:1b'])

    # Get diverse perspectives
    perspectives = models.infer(
        "What are the pros and cons of microservices architecture?"
    )

    # Synthesize into consensus
    consensus = perspectives.consensus(axis=0, model='llama3.2').collect()

    print(f"Synthesized consensus:\n{consensus.data[0]}")


def example_variance():
    """Measure disagreement across models"""
    from npcpy.npc_array import NPCArray

    models = NPCArray.from_llms(['llama3.2', 'gemma3:1b'])

    # Question that might get different answers
    result = models.infer("What is the best programming language?").collect()

    # Check variance (semantic disagreement)
    variance = result.map(lambda responses: _compute_variance(responses))
    print(f"Model disagreement: {variance}")


def _compute_variance(responses):
    """Helper to compute semantic variance"""
    if not isinstance(responses, list):
        return 0.0
    # Simple: check word overlap
    word_sets = [set(str(r).lower().split()) for r in responses]
    if len(word_sets) < 2:
        return 0.0
    overlap = len(word_sets[0] & word_sets[1]) / len(word_sets[0] | word_sets[1])
    return 1.0 - overlap


# =============================================================================
# PARAMETER SWEEPS (MATRIX)
# =============================================================================

def example_meshgrid():
    """Cartesian product over parameters"""
    from npcpy.npc_array import NPCArray

    # Create all combinations
    configs = NPCArray.meshgrid(
        model=['llama3.2', 'gemma3:1b'],
        temperature=[0.0, 0.5, 1.0]
    )

    print(f"Config array shape: {configs.shape}")  # (6,) = 2 models × 3 temps

    # Run inference with each config
    result = configs.infer("Complete: The quick brown fox").collect()

    print(f"Result shape: {result.shape}")
    for i, spec in enumerate(configs.specs):
        print(f"  {spec.model_ref} @ temp={spec.config.get('temperature')}: {result.data[i, 0][:50]}...")


def example_sampling_with_matrix():
    """Using get_llm_response matrix directly"""
    from npcpy.llm_funcs import get_llm_response

    # Single prompt, multiple configs via matrix
    result = get_llm_response(
        "Write a creative opening line for a story.",
        matrix={
            'model': ['llama3.2', 'gemma3:1b'],
            'temperature': [0.5, 1.0]
        }
    )

    print(f"Number of runs: {len(result['runs'])}")
    for run in result['runs']:
        print(f"  {run['combo']}: {run['response'][:50]}...")


def example_n_samples():
    """Multiple samples from same config"""
    from npcpy.llm_funcs import get_llm_response

    # Get 3 samples from same model
    result = get_llm_response(
        "Give me a random number between 1 and 100.",
        model='llama3.2',
        provider='ollama',
        n_samples=3
    )

    print(f"Number of samples: {len(result['runs'])}")
    for run in result['runs']:
        print(f"  Sample {run['sample_index']}: {run['response']}")


def example_matrix_with_samples():
    """Combine matrix and n_samples for full exploration"""
    from npcpy.llm_funcs import get_llm_response

    # 2 models × 2 temperatures × 2 samples = 8 total runs
    result = get_llm_response(
        "Flip a coin: heads or tails?",
        matrix={
            'model': ['llama3.2', 'gemma3:1b'],
            'temperature': [0.0, 1.0]
        },
        n_samples=2
    )

    print(f"Total runs: {len(result['runs'])}")

    # Group by config
    from collections import defaultdict
    by_config = defaultdict(list)
    for run in result['runs']:
        key = f"{run['combo'].get('model')}@{run['combo'].get('temperature')}"
        by_config[key].append(run['response'])

    for config, responses in by_config.items():
        print(f"{config}: {responses}")


# =============================================================================
# SKLEARN / ML MODELS
# =============================================================================

def example_sklearn_ensemble():
    """NPCArray with sklearn models"""
    from npcpy.npc_array import NPCArray
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    import numpy as np

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Fit multiple models
    rf = RandomForestClassifier(n_estimators=10, random_state=42).fit(X_train, y_train)
    lr = LogisticRegression(random_state=42).fit(X_train, y_train)
    svc = SVC(random_state=42).fit(X_train, y_train)

    # Create array from fitted models
    models = NPCArray.from_sklearn([rf, lr, svc])

    print(f"Model array shape: {models.shape}")  # (3,)

    # Vectorized prediction
    predictions = models.predict(X_test).collect()

    print(f"Predictions shape: {predictions.shape}")
    print(f"RF: {predictions.data[0]}")
    print(f"LR: {predictions.data[1]}")
    print(f"SVC: {predictions.data[2]}")


def example_ml_grid_search():
    """Grid search with ml_funcs"""
    from npcpy.ml_funcs import fit_model, score_model
    import numpy as np

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Grid search via matrix
    result = fit_model(
        X_train, y_train,
        model='RandomForestClassifier',
        matrix={
            'n_estimators': [10, 50, 100],
            'max_depth': [3, 5, 10]
        },
        parallel=True
    )

    print(f"Number of fitted models: {len(result['models'])}")

    # Score all models
    for i, (model, params) in enumerate(zip(result['models'], [r['params'] for r in result['results']])):
        scores = score_model(X_test, y_test, model, metrics=['accuracy', 'f1'])
        print(f"  {params}: accuracy={scores['scores']['accuracy']:.3f}")


def example_ensemble_voting_ml():
    """Ensemble voting with ML models"""
    from npcpy.ml_funcs import fit_model, ensemble_predict
    import numpy as np

    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    # Fit multiple model types
    rf = fit_model(X_train, y_train, 'RandomForestClassifier', n_estimators=10)['model']
    lr = fit_model(X_train, y_train, 'LogisticRegression')['model']
    gb = fit_model(X_train, y_train, 'GradientBoostingClassifier')['model']

    # Ensemble vote
    result = ensemble_predict(X_test, [rf, lr, gb], method='vote')

    print(f"Ensemble predictions: {result['predictions']}")
    print(f"Individual predictions:\n{result['individual_predictions']}")


# =============================================================================
# DEBATE / CHAIN PATTERNS
# =============================================================================

def example_debate():
    """Multi-agent debate pattern"""
    from npcpy.npc_array import NPCArray

    # Create diverse "personalities" via prompts
    models = NPCArray.from_llms(['llama3.2', 'gemma3:1b'])

    topic = "Should AI systems be open source?"

    # Initial positions
    positions = models.infer(f"Take a clear stance on: {topic}").collect()

    print("Initial positions:")
    for i, pos in enumerate(positions.flatten()):
        print(f"  Model {i}: {pos[:100]}...")

    # Debate round - each sees others' positions
    def debate_round(responses):
        all_positions = "\n".join(f"Position {i}: {r}" for i, r in enumerate(responses))
        return f"Consider these perspectives:\n{all_positions}\n\nProvide a refined position."

    refined = models.infer([debate_round(positions.flatten())]).collect()

    print("\nRefined positions after debate:")
    for i, pos in enumerate(refined.flatten()):
        print(f"  Model {i}: {pos[:100]}...")


def example_chain():
    """Chain outputs through synthesis"""
    from npcpy.npc_array import NPCArray

    models = NPCArray.from_llms(['llama3.2', 'gemma3:1b'])

    # Use chain for iterative refinement
    def synthesize(responses):
        combined = "\n---\n".join(responses)
        return f"Synthesize these perspectives into one view:\n{combined}"

    result = (
        models
        .infer("What makes a good software engineer?")
        .chain(synthesize, n_rounds=2)
        .collect()
    )

    print(f"Final synthesized result:\n{result.data[0, 0]}")


# =============================================================================
# EVOLUTIONARY OPTIMIZATION
# =============================================================================

def example_evolution():
    """Evolve model configurations"""
    from npcpy.npc_array import NPCArray, ModelSpec
    import random

    # Initial population of configs
    initial_specs = [
        ModelSpec(model_type="llm", model_ref="llama3.2",
                  config={"temperature": random.uniform(0, 1)})
        for _ in range(5)
    ]

    population = NPCArray(initial_specs)

    print(f"Initial population: {population.shape}")

    # Simulate fitness scores
    fitness_scores = [random.random() for _ in range(5)]

    # Mutation function
    def mutate(spec):
        new_spec = ModelSpec(
            model_type=spec.model_type,
            model_ref=spec.model_ref,
            config={"temperature": min(1.0, max(0.0,
                spec.config.get("temperature", 0.5) + random.uniform(-0.2, 0.2)
            ))}
        )
        return new_spec

    # Evolve
    evolved = population.evolve(
        fitness_scores,
        mutate_fn=mutate,
        elite_ratio=0.2
    )

    # This is lazy - would need compute infrastructure
    print("Evolution operation queued")


# =============================================================================
# POLARS INTEGRATION
# =============================================================================

def example_polars_integration():
    """Use NPCArray with Polars DataFrames"""
    try:
        import polars as pl
        from npcpy.npc_array import NPCArray, npc_udf
    except ImportError:
        print("Polars not installed. pip install polars")
        return

    # Create DataFrame
    df = pl.DataFrame({
        'text': [
            "Hello world",
            "How are you?",
            "What is AI?"
        ],
        'category': ['greeting', 'question', 'question']
    })

    # Create model array
    models = NPCArray.from_llms('llama3.2')

    # Apply inference as UDF
    result = df.with_columns(
        pl.col('text').map_elements(
            lambda x: models.infer(x).collect().data[0, 0],
            return_dtype=pl.Utf8
        ).alias('response')
    )

    print(result)


# =============================================================================
# QUICK UTILITIES
# =============================================================================

def example_quick_functions():
    """Quick utility functions"""
    from npcpy.npc_array import infer_matrix, ensemble_vote

    # Quick matrix inference
    result = infer_matrix(
        prompts=["Hello", "Goodbye"],
        models=['llama3.2'],
        providers=['ollama']
    )
    print(f"Matrix result shape: {result.shape}")

    # Quick ensemble vote
    answer = ensemble_vote(
        "What is the capital of France? One word.",
        models=['llama3.2', 'gemma3:1b'],
        providers=['ollama', 'ollama']
    )
    print(f"Voted answer: {answer}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NPCArray Examples")
    print("=" * 60)

    # Run examples that don't require actual model calls for structure testing
    print("\n--- Basic Structure Examples ---")

    from npcpy.npc_array import NPCArray

    # Structure tests (no inference)
    arr = NPCArray.from_llms(['model1', 'model2'])
    print(f"Created array: {arr}")

    grid = NPCArray.meshgrid(model=['a', 'b'], temp=[0.1, 0.9])
    print(f"Meshgrid: {grid}")

    lazy = arr.infer(['prompt1', 'prompt2']).map(len).filter(lambda x: x > 0)
    print(f"Lazy chain type: {type(lazy)}")
    lazy.explain()

    print("\n--- To run actual inference examples, call individual functions ---")
    print("e.g., example_basic_inference()")
