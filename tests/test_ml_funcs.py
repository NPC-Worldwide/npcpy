# Tests for ml_funcs model serialization (Issue #198 - no pickle)

import os
import tempfile
import pytest
import numpy as np

from npcpy.ml_funcs import serialize_model, deserialize_model


# =============================================================================
# Serialization Tests (Issue #198 - Pickle Removed)
# =============================================================================

class TestModelSerialization:
    """Test model serialization without pickle."""

    def test_serialize_model_joblib(self):
        """Test serializing a model with joblib format."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=5)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name

        try:
            serialize_model(model, path, format='joblib')
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
            print(f"Model serialized to {path}")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_deserialize_model_joblib(self):
        """Test deserializing a model with joblib format."""
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=5)
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name

        try:
            serialize_model(model, path, format='joblib')
            loaded_model = deserialize_model(path, format='joblib')

            # Verify the loaded model works
            predictions = loaded_model.predict([[2, 3]])
            assert predictions is not None
            print(f"Model deserialized and made prediction: {predictions}")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_deserialize_auto_format_detection(self):
        """Test that format is auto-detected from file extension."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        model.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name

        try:
            serialize_model(model, path, format='joblib')
            # Use auto-detection (default)
            loaded_model = deserialize_model(path)

            assert loaded_model is not None
            predictions = loaded_model.predict([[4, 5]])
            assert predictions is not None
            print(f"Auto-format detection worked, prediction: {predictions}")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_serialize_unsupported_format_raises(self):
        """Test that unsupported format raises ValueError."""
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression()

        with tempfile.NamedTemporaryFile(suffix='.model', delete=False) as f:
            path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                serialize_model(model, path, format='pickle')
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_deserialize_unknown_extension_raises(self):
        """Test that unknown extension with auto-format raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix='.unknown', delete=False) as f:
            f.write(b'dummy data')
            path = f.name

        try:
            with pytest.raises(ValueError, match="Cannot auto-detect format"):
                deserialize_model(path, format='auto')
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestNoPickleUsage:
    """Ensure pickle is not used anywhere in ml_funcs."""

    def test_no_pickle_import_in_ml_funcs(self):
        """Verify pickle is not imported in ml_funcs module."""
        import npcpy.ml_funcs as ml_funcs
        import sys

        # Check that pickle is not in the module's namespace
        assert 'pickle' not in dir(ml_funcs), "pickle should not be imported in ml_funcs"

        # Also check that pickle isn't loaded as a submodule
        ml_funcs_file = ml_funcs.__file__
        with open(ml_funcs_file, 'r') as f:
            content = f.read()

        # Should only find "pickle" in docstrings mentioning "no pickle"
        import re
        pickle_imports = re.findall(r'^import pickle|^from pickle', content, re.MULTILINE)
        assert len(pickle_imports) == 0, f"Found pickle imports: {pickle_imports}"

        print("No pickle imports found in ml_funcs")


class TestSafetensorsFormat:
    """Test safetensors format for PyTorch models."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not installed"),
        reason="PyTorch required"
    )
    @pytest.mark.skipif(
        not pytest.importorskip("safetensors", reason="safetensors not installed"),
        reason="safetensors required"
    )
    def test_serialize_torch_model_safetensors(self):
        """Test serializing PyTorch model with safetensors."""
        import torch
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 2)

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()

        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
            path = f.name

        try:
            serialize_model(model, path, format='safetensors')
            assert os.path.exists(path)
            print(f"PyTorch model serialized with safetensors to {path}")
        finally:
            if os.path.exists(path):
                os.remove(path)


# =============================================================================
# Round-Trip Tests
# =============================================================================

class TestRoundTrip:
    """Test complete serialize/deserialize round trips."""

    def test_round_trip_preserves_predictions(self):
        """Test that serialized and deserialized model gives same predictions."""
        from sklearn.ensemble import RandomForestClassifier

        # Create and train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]])
        y_train = np.array([0, 0, 0, 1, 1, 1])
        model.fit(X_train, y_train)

        # Get predictions before serialization
        X_test = np.array([[2, 3], [8, 9]])
        original_predictions = model.predict(X_test)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            path = f.name

        try:
            # Serialize and deserialize
            serialize_model(model, path, format='joblib')
            loaded_model = deserialize_model(path)

            # Get predictions after deserialization
            loaded_predictions = loaded_model.predict(X_test)

            # Predictions should be identical
            np.testing.assert_array_equal(original_predictions, loaded_predictions)
            print(f"Round-trip preserved predictions: {original_predictions}")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_round_trip_multiple_model_types(self):
        """Test round trip with different sklearn model types."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])

        models = [
            ("LogisticRegression", LogisticRegression()),
            ("DecisionTree", DecisionTreeClassifier()),
            ("SVC", SVC()),
        ]

        for name, model in models:
            model.fit(X, y)

            with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
                path = f.name

            try:
                serialize_model(model, path, format='joblib')
                loaded = deserialize_model(path)

                original_pred = model.predict([[4, 5]])
                loaded_pred = loaded.predict([[4, 5]])

                assert original_pred[0] == loaded_pred[0], f"{name} predictions differ"
                print(f"✓ {name} round-trip successful")
            finally:
                if os.path.exists(path):
                    os.remove(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
