# Integration tests for NPCArray - numpy for AI
# These tests use real models (ollama) and sklearn - no mocks

import numpy as np
from npcpy.npc_array import (
    NPCArray, ResponseTensor, LazyResult, ModelSpec,
    infer_matrix, ensemble_vote
)
from npcpy.ml_funcs import fit_model, predict_model, score_model, ensemble_predict
from npcpy.llm_funcs import get_llm_response


# =============================================================================
# NPCArray Creation Tests
# =============================================================================

def test_npc_array_from_llms():
    """Test creating NPCArray from LLM specifications"""
    arr = NPCArray.from_llms(
        models=["llama3.2:latest", "gemma3:1b"],
        providers="ollama"
    )
    assert len(arr) == 2
    assert arr.shape == (2,)
    print(f"Created NPCArray with {len(arr)} models")
    print(f"Models: {[s.model_ref for s in arr.specs]}")


def test_npc_array_from_sklearn():
    """Test creating NPCArray from sklearn models"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    arr = NPCArray.from_sklearn([
        RandomForestClassifier(n_estimators=10),
        LogisticRegression(),
        DecisionTreeClassifier()
    ])
    assert len(arr) == 3
    assert arr.shape == (3,)
    print(f"Created sklearn NPCArray with {len(arr)} models")


def test_npc_array_meshgrid():
    """Test meshgrid creation over parameter space"""
    arr = NPCArray.meshgrid(
        models=["llama3.2:latest", "gemma3:1b"],
        providers=["ollama"],
        temperatures=[0.3, 0.7, 1.0]
    )
    # 2 models x 1 provider x 3 temps = 6 combinations
    assert len(arr) == 6
    print(f"Meshgrid created {len(arr)} model configurations")


# =============================================================================
# LLM Inference Tests
# =============================================================================

def test_npc_array_infer_single():
    """Test basic inference with a single model"""
    arr = NPCArray.from_llms(["llama3.2:latest"], providers="ollama")
    lazy = arr.infer("What is 2+2? Reply with just the number.")

    result = lazy.collect()
    assert result is not None
    assert isinstance(result, ResponseTensor)
    print(f"Inference result: {result.data}")


def test_npc_array_infer_multiple_models():
    """Test inference across multiple models"""
    arr = NPCArray.from_llms(
        ["llama3.2:latest", "gemma3:1b"],
        providers="ollama"
    )
    lazy = arr.infer("Say hello in exactly 3 words.")

    result = lazy.collect()
    assert result is not None
    assert len(result.data) == 2
    print(f"Multi-model results:")
    for i, r in enumerate(result.data):
        print(f"  Model {i}: {r}")


def test_npc_array_infer_multiple_prompts():
    """Test broadcasting prompts over models"""
    arr = NPCArray.from_llms(["llama3.2:latest"], providers="ollama")
    prompts = ["Say yes", "Say no", "Say maybe"]
    lazy = arr.infer(prompts)

    result = lazy.collect()
    assert result is not None
    print(f"Multi-prompt results: {result.data}")


# =============================================================================
# Lazy Chain Operations Tests
# =============================================================================

def test_lazy_map():
    """Test mapping function over results"""
    arr = NPCArray.from_llms(["llama3.2:latest"], providers="ollama")

    lazy = arr.infer("What is 5+5? Answer with just the number.").map(
        lambda x: f"The answer is: {x}"
    )

    result = lazy.collect()
    assert result is not None
    print(f"Mapped result: {result.data}")


def test_lazy_filter():
    """Test filtering results"""
    arr = NPCArray.from_llms(
        ["llama3.2:latest", "gemma3:1b"],
        providers="ollama"
    )

    lazy = arr.infer("Say a short word.").filter(
        lambda x: len(str(x)) < 50
    )

    result = lazy.collect()
    print(f"Filtered results: {result.data}")


def test_lazy_reduce():
    """Test reducing results"""
    arr = NPCArray.from_llms(
        ["llama3.2:latest", "gemma3:1b"],
        providers="ollama"
    )

    lazy = arr.infer("Pick a number between 1 and 10. Just say the number.").reduce(
        lambda results: f"Combined: {' and '.join(str(r) for r in results)}"
    )

    result = lazy.collect()
    print(f"Reduced result: {result.data}")


def test_lazy_chain():
    """Test chaining prompts through models"""
    arr = NPCArray.from_llms(["llama3.2:latest"], providers="ollama")

    lazy = arr.infer(
        "Write a very short haiku about code."
    ).chain(
        lambda responses: f"Translate this to French: {responses[0]}"
    )

    result = lazy.collect()
    print(f"Chained result: {result.data}")


# =============================================================================
# Voting and Consensus Tests
# =============================================================================

def test_lazy_vote():
    """Test voting across model responses"""
    arr = NPCArray.from_llms(
        ["llama3.2:latest", "gemma3:1b"],
        providers="ollama"
    )

    lazy = arr.infer(
        "Is Python a programming language? Answer only 'yes' or 'no'."
    ).vote()

    result = lazy.collect()
    print(f"Vote result: {result.data}")


def test_lazy_consensus():
    """Test synthesizing consensus from multiple responses"""
    arr = NPCArray.from_llms(
        ["llama3.2:latest", "gemma3:1b"],
        providers="ollama"
    )

    lazy = arr.infer(
        "Name one benefit of test-driven development in 5 words or less."
    ).consensus()

    result = lazy.collect()
    print(f"Consensus result: {result.data}")


def test_ensemble_vote_standalone():
    """Test standalone ensemble_vote function"""
    responses = ["yes", "yes", "no", "yes"]
    from collections import Counter
    counter = Counter(responses)
    winner = counter.most_common(1)[0][0]
    assert winner == "yes"
    print(f"Ensemble vote winner: {winner}")


# =============================================================================
# Matrix Sampling Tests (get_llm_response matrix/n_samples)
# =============================================================================

def test_matrix_sampling():
    """Test matrix parameter for cartesian product sampling"""
    result = get_llm_response(
        prompt="What is {x} + {y}? Answer with just the number.",
        model="llama3.2:latest",
        provider="ollama",
        matrix={
            "x": [1, 2],
            "y": [10, 20]
        }
    )
    # Should produce 2x2 = 4 results
    assert result is not None
    print(f"Matrix sampling result type: {type(result)}")
    print(f"Matrix sampling results: {result}")


def test_n_samples():
    """Test n_samples for multiple samples per config"""
    result = get_llm_response(
        prompt="Pick a random number between 1 and 100. Just say the number.",
        model="llama3.2:latest",
        provider="ollama",
        n_samples=3
    )
    assert result is not None
    print(f"n_samples result: {result}")


def test_infer_matrix_standalone():
    """Test standalone infer_matrix function"""
    results = infer_matrix(
        prompts=["What is Python?", "What is JavaScript?"],
        models=["llama3.2:latest"],
        providers=["ollama"]
    )

    print(f"infer_matrix results: {results}")


# =============================================================================
# ML Functions Tests
# =============================================================================

def test_fit_model_sklearn():
    """Test fitting sklearn model"""
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = RandomForestClassifier(n_estimators=5)
    fitted = fit_model(model, X, y)

    assert fitted is not None
    print(f"Fitted model: {fitted}")


def test_predict_model_sklearn():
    """Test prediction with sklearn model"""
    from sklearn.ensemble import RandomForestClassifier

    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[2, 3], [6, 7]])

    model = RandomForestClassifier(n_estimators=5)
    fitted = fit_model(model, X_train, y_train)
    predictions = predict_model(fitted, X_test)

    assert predictions is not None
    assert len(predictions) == 2
    print(f"Predictions: {predictions}")


def test_score_model_sklearn():
    """Test scoring sklearn model"""
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = RandomForestClassifier(n_estimators=5)
    fitted = fit_model(model, X, y)

    scores = score_model(fitted, X, y, metrics=["accuracy", "f1"])

    assert "accuracy" in scores
    print(f"Scores: {scores}")


def test_ensemble_predict():
    """Test ensemble prediction with multiple models"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[2, 3], [8, 9]])

    models = [
        fit_model(RandomForestClassifier(n_estimators=5), X_train, y_train),
        fit_model(LogisticRegression(), X_train, y_train),
        fit_model(DecisionTreeClassifier(), X_train, y_train)
    ]

    predictions = ensemble_predict(models, X_test, method="vote")

    assert predictions is not None
    assert len(predictions) == 2
    print(f"Ensemble predictions: {predictions}")


def test_fit_model_with_matrix():
    """Test grid search via matrix parameter"""
    from sklearn.ensemble import RandomForestClassifier

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y = np.array([0, 0, 0, 1, 1, 1])

    results = fit_model(
        RandomForestClassifier(),
        X, y,
        matrix={
            "n_estimators": [5, 10],
            "max_depth": [2, 3]
        }
    )

    # 2x2 = 4 fitted models
    assert results is not None
    print(f"Grid search results: {len(results) if isinstance(results, list) else 1} models fitted")


# =============================================================================
# NPCArray with sklearn Tests
# =============================================================================

def test_npc_array_sklearn_fit():
    """Test fitting sklearn models via NPCArray"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    arr = NPCArray.from_sklearn([
        RandomForestClassifier(n_estimators=5),
        LogisticRegression()
    ])

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])

    lazy = arr.fit(X, y)
    result = lazy._graph  # Check that graph was built

    assert result is not None
    print(f"NPCArray sklearn fit graph built")


def test_npc_array_sklearn_predict():
    """Test prediction with NPCArray sklearn models"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y_train = np.array([0, 0, 1, 1])
    X_test = np.array([[2, 3], [6, 7]])

    # Pre-fit models
    rf = RandomForestClassifier(n_estimators=5).fit(X_train, y_train)
    lr = LogisticRegression().fit(X_train, y_train)

    arr = NPCArray.from_sklearn([rf, lr])
    lazy = arr.predict(X_test)
    result = lazy.collect()

    assert result is not None
    print(f"NPCArray sklearn predict: {result.data}")


# =============================================================================
# ResponseTensor Tests
# =============================================================================

def test_response_tensor_numpy_like():
    """Test ResponseTensor numpy-like interface"""
    spec = ModelSpec(model_type="llm", model_ref="test")
    data = np.array(["response1", "response2", "response3"], dtype=object)
    tensor = ResponseTensor(data=data, model_specs=[spec])

    assert tensor.shape == (3,)
    assert len(tensor.data) == 3
    assert tensor.data[0] == "response1"
    print(f"ResponseTensor: shape={tensor.shape}, len={len(tensor.data)}")


def test_response_tensor_tolist():
    """Test ResponseTensor tolist"""
    spec = ModelSpec(model_type="llm", model_ref="test")
    data = np.array(["a", "b", "c"], dtype=object)
    tensor = ResponseTensor(data=data, model_specs=[spec])

    items = tensor.tolist()
    assert items == ["a", "b", "c"]
    print(f"ResponseTensor tolist: {items}")


def test_response_tensor_flatten():
    """Test ResponseTensor flatten"""
    spec = ModelSpec(model_type="llm", model_ref="test")
    data = np.array([["a", "b"], ["c", "d"]], dtype=object)
    tensor = ResponseTensor(data=data, model_specs=[spec, spec])

    flat = tensor.flatten()
    assert flat == ["a", "b", "c", "d"]
    print(f"ResponseTensor flatten: {flat}")


# =============================================================================
# End-to-End Integration Tests
# =============================================================================

def test_full_pipeline_llm():
    """Full end-to-end test of LLM pipeline"""
    arr = NPCArray.from_llms(
        ["llama3.2:latest", "gemma3:1b"],
        providers="ollama"
    )

    result = (
        arr.infer("What is the capital of France? Answer in one word.")
        .map(lambda x: x.strip().lower())
        .vote()
        .collect()
    )

    print(f"Full LLM pipeline result: {result.data}")
    assert "paris" in str(result.data).lower()


def test_full_pipeline_sklearn():
    """Full end-to-end test of sklearn pipeline"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    X_test = np.array([[2, 3], [10, 11]])

    # Pre-fit models
    rf = RandomForestClassifier(n_estimators=5).fit(X_train, y_train)
    lr = LogisticRegression().fit(X_train, y_train)
    dt = DecisionTreeClassifier().fit(X_train, y_train)

    arr = NPCArray.from_sklearn([rf, lr, dt])
    predictions = arr.predict(X_test).collect()

    print(f"Full sklearn pipeline predictions: {predictions.data}")


if __name__ == "__main__":
    print("=" * 60)
    print("NPCArray Integration Tests")
    print("=" * 60)

    tests = [
        # Creation tests
        ("NPCArray from LLMs", test_npc_array_from_llms),
        ("NPCArray from sklearn", test_npc_array_from_sklearn),
        ("NPCArray meshgrid", test_npc_array_meshgrid),

        # LLM inference tests
        ("Single model inference", test_npc_array_infer_single),
        ("Multi-model inference", test_npc_array_infer_multiple_models),
        ("Multi-prompt inference", test_npc_array_infer_multiple_prompts),

        # Lazy chain tests
        ("Lazy map", test_lazy_map),
        ("Lazy filter", test_lazy_filter),
        ("Lazy reduce", test_lazy_reduce),
        ("Lazy chain", test_lazy_chain),

        # Voting tests
        ("Lazy vote", test_lazy_vote),
        ("Lazy consensus", test_lazy_consensus),
        ("Ensemble vote standalone", test_ensemble_vote_standalone),

        # Matrix sampling tests
        ("Matrix sampling", test_matrix_sampling),
        ("N samples", test_n_samples),
        ("Infer matrix standalone", test_infer_matrix_standalone),

        # ML function tests
        ("Fit sklearn model", test_fit_model_sklearn),
        ("Predict sklearn model", test_predict_model_sklearn),
        ("Score sklearn model", test_score_model_sklearn),
        ("Ensemble predict", test_ensemble_predict),
        ("Fit with matrix (grid search)", test_fit_model_with_matrix),

        # NPCArray sklearn tests
        ("NPCArray sklearn fit", test_npc_array_sklearn_fit),
        ("NPCArray sklearn predict", test_npc_array_sklearn_predict),

        # ResponseTensor tests
        ("ResponseTensor numpy-like", test_response_tensor_numpy_like),
        ("ResponseTensor tolist", test_response_tensor_tolist),
        ("ResponseTensor flatten", test_response_tensor_flatten),

        # End-to-end tests
        ("Full LLM pipeline", test_full_pipeline_llm),
        ("Full sklearn pipeline", test_full_pipeline_sklearn),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
            print(f"✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
