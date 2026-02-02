# ML Functions

`npcpy.ml_funcs` provides a NumPy-like interface for traditional machine learning operations. Where `npcpy.llm_funcs` handles LLM calls, `ml_funcs` handles sklearn models, PyTorch training, time series forecasting, ensembles, and serialization. The API follows the same patterns -- a single call does a single operation, a `matrix` parameter enables grid search, and `n_samples` enables repeated training with different seeds.

## fit_model

`fit_model` is the primary entry point for training sklearn-compatible models. Pass the model name as a string and any hyperparameters as keyword arguments.

```python
from npcpy.ml_funcs import fit_model
from sklearn.datasets import make_classification

# Generate sample data
X_train, y_train = make_classification(n_samples=200, n_features=10, random_state=42)

# Fit a single model
result = fit_model(X_train, y_train, model="RandomForestClassifier", n_estimators=100)

print(result["model"])         # The fitted RandomForestClassifier
print(result["scores"])        # Training score
print(len(result["models"]))   # 1
```

The returned dict always contains:

- `model` -- the first (or only) fitted model
- `models` -- list of all fitted models
- `scores` -- training scores (when available)
- `results` -- detailed results for grid search / multi-sample runs

### Passing a Model Instance

You can also pass an already-constructed sklearn estimator:

```python
from sklearn.svm import SVC

svc = SVC(kernel="rbf", C=1.0)
result = fit_model(X_train, y_train, model=svc)
print(result["model"])
```

## Grid Search with the matrix Parameter

The `matrix` parameter accepts a dict mapping hyperparameter names to lists of values. `fit_model` trains one model for every combination in the Cartesian product.

```python
result = fit_model(
    X_train, y_train,
    model="RandomForestClassifier",
    matrix={
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 5, 10],
    },
)

print(f"Fitted {len(result['models'])} configurations")  # 9

# Find the best model by training score
best_idx = result["scores"].index(max(result["scores"]))
best_model = result["models"][best_idx]
best_params = result["results"][best_idx]["params"]
print(f"Best params: {best_params}")
print(f"Best score: {result['scores'][best_idx]:.4f}")
```

Grid search runs in parallel by default. Set `parallel=False` to disable threading.

### Multi-Sample Training

Use `n_samples` to train multiple models with different random seeds:

```python
result = fit_model(
    X_train, y_train,
    model="RandomForestClassifier",
    n_estimators=100,
    n_samples=5,
)

print(f"Trained {len(result['models'])} models with different seeds")
print(f"Score range: {min(result['scores']):.4f} - {max(result['scores']):.4f}")
```

You can combine `matrix` and `n_samples` -- each grid combination is trained `n_samples` times.

## Supported Models

`ml_funcs` ships with 27 registered sklearn models plus XGBoost.

### Classification (9 models)

| Name | sklearn Class |
|------|--------------|
| `LogisticRegression` | `sklearn.linear_model.LogisticRegression` |
| `RandomForestClassifier` | `sklearn.ensemble.RandomForestClassifier` |
| `GradientBoostingClassifier` | `sklearn.ensemble.GradientBoostingClassifier` |
| `SVC` | `sklearn.svm.SVC` |
| `KNeighborsClassifier` | `sklearn.neighbors.KNeighborsClassifier` |
| `DecisionTreeClassifier` | `sklearn.tree.DecisionTreeClassifier` |
| `AdaBoostClassifier` | `sklearn.ensemble.AdaBoostClassifier` |
| `GaussianNB` | `sklearn.naive_bayes.GaussianNB` |
| `MLPClassifier` | `sklearn.neural_network.MLPClassifier` |

### Regression (10 models)

| Name | sklearn Class |
|------|--------------|
| `LinearRegression` | `sklearn.linear_model.LinearRegression` |
| `Ridge` | `sklearn.linear_model.Ridge` |
| `Lasso` | `sklearn.linear_model.Lasso` |
| `ElasticNet` | `sklearn.linear_model.ElasticNet` |
| `RandomForestRegressor` | `sklearn.ensemble.RandomForestRegressor` |
| `GradientBoostingRegressor` | `sklearn.ensemble.GradientBoostingRegressor` |
| `SVR` | `sklearn.svm.SVR` |
| `KNeighborsRegressor` | `sklearn.neighbors.KNeighborsRegressor` |
| `DecisionTreeRegressor` | `sklearn.tree.DecisionTreeRegressor` |
| `MLPRegressor` | `sklearn.neural_network.MLPRegressor` |

### Clustering (3 models)

| Name | sklearn Class |
|------|--------------|
| `KMeans` | `sklearn.cluster.KMeans` |
| `DBSCAN` | `sklearn.cluster.DBSCAN` |
| `AgglomerativeClustering` | `sklearn.cluster.AgglomerativeClustering` |

### Dimensionality Reduction (3 models)

| Name | Library |
|------|---------|
| `PCA` | `sklearn.decomposition.PCA` |
| `TSNE` | `sklearn.manifold.TSNE` |
| `UMAP` | `umap.UMAP` |

### XGBoost

Model names starting with `xgb` are routed to XGBoost:

```python
result = fit_model(X_train, y_train, model="xgb_classifier", n_estimators=100)
```

Use `xgb_classifier` for classification and `xgb_regressor` (or any name starting with `xgb` that does not contain `classifier`) for regression.

## score_model

`score_model` evaluates fitted models against test data using named metrics.

```python
from npcpy.ml_funcs import score_model
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=300, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

result = fit_model(X_train, y_train, model="RandomForestClassifier", n_estimators=100)

scores = score_model(
    X_test, y_test,
    model=result["model"],
    metrics=["accuracy", "f1", "precision", "recall"],
)

print(scores["scores"])
# {'accuracy': 0.92, 'f1': 0.91, 'precision': 0.93, 'recall': 0.90}
```

### Available Metrics

| Metric | Use Case |
|--------|----------|
| `accuracy` | Classification overall correctness |
| `f1` | Classification F1 (weighted average) |
| `precision` | Classification precision (weighted average) |
| `recall` | Classification recall (weighted average) |
| `mse` | Regression mean squared error |
| `mae` | Regression mean absolute error |
| `r2` | Regression R-squared |

### Scoring Multiple Models

Pass a list of models to score them all at once:

```python
scores = score_model(
    X_test, y_test,
    model=result["models"],  # list from grid search
    metrics=["accuracy", "f1"],
)

# scores["all_scores"] is a list of dicts, one per model
for i, s in enumerate(scores["all_scores"]):
    print(f"Model {i}: accuracy={s['accuracy']:.4f} f1={s['f1']:.4f}")
```

## ensemble_predict

`ensemble_predict` combines predictions from multiple models using voting, averaging, or weighted averaging.

```python
from npcpy.ml_funcs import ensemble_predict

# Train multiple models via grid search
result = fit_model(
    X_train, y_train,
    model="RandomForestClassifier",
    matrix={"n_estimators": [10, 50, 100]},
)

# Majority voting (classification)
predictions = ensemble_predict(X_test, result["models"], method="vote")
print(predictions["predictions"])       # array of predicted classes
print(predictions["method"])            # "vote"

# Simple averaging (regression or probabilities)
predictions = ensemble_predict(X_test, result["models"], method="average")

# Weighted averaging
predictions = ensemble_predict(
    X_test,
    result["models"],
    method="weighted",
    weights=[0.2, 0.3, 0.5],
)
```

The returned dict contains:

- `predictions` -- the ensemble predictions
- `individual_predictions` -- a 2D array of each model's predictions
- `method` -- the method used

### Ensemble Methods

| Method | Description |
|--------|-------------|
| `vote` | Majority voting across models (classification). Uses `scipy.stats.mode`. |
| `average` | Simple mean of all model predictions (regression). |
| `weighted` | Weighted sum of predictions. Pass `weights` as a list of floats. |

## cross_validate

`cross_validate` runs k-fold cross-validation and returns per-fold scores.

```python
from npcpy.ml_funcs import cross_validate

X, y = make_classification(n_samples=300, n_features=10, random_state=42)

cv_result = cross_validate(
    X, y,
    model="RandomForestClassifier",
    cv=5,
    metrics=["accuracy", "f1"],
    n_estimators=100,
)

for metric, stats in cv_result.items():
    print(f"{metric}: mean={stats['mean']:.4f} +/- {stats['std']:.4f}")
    print(f"  Per-fold: {stats['scores']}")
```

You can pass model hyperparameters as keyword arguments -- they are forwarded to the model constructor.

## predict_model

`predict_model` runs inference on one or more fitted models.

```python
from npcpy.ml_funcs import predict_model

result = predict_model(X_test, model=fitted_model)
print(result["predictions"])  # predictions from the model

# Predict probabilities
result = predict_model(X_test, model=fitted_model, method="predict_proba")
print(result["predictions"])  # probability matrix

# Transform (for PCA, TSNE, etc.)
result = predict_model(X_new, model=pca_model, method="transform")
print(result["predictions"])  # transformed features
```

## PyTorch Functions

### fit_torch

`fit_torch` trains a PyTorch `nn.Module` with a standard training loop.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from npcpy.ml_funcs import fit_torch

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()

# Create data loaders
X_tensor = torch.randn(200, 10)
y_tensor = torch.randint(0, 2, (200,))
train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32)

# Train
result = fit_torch(
    model,
    train_loader,
    epochs=10,
    optimizer="Adam",
    lr=0.001,
    criterion="CrossEntropyLoss",
    device="cpu",
)

print(f"Final train loss: {result['final_train_loss']:.4f}")
print(f"Loss history: {result['history']['train_loss']}")
trained_model = result["model"]
```

### fit_torch Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | required | `nn.Module` instance |
| `train_loader` | required | PyTorch `DataLoader` |
| `epochs` | `10` | Number of training epochs |
| `optimizer` | `"Adam"` | Optimizer class name from `torch.optim` |
| `lr` | `0.001` | Learning rate |
| `criterion` | `"CrossEntropyLoss"` | Loss class name from `torch.nn` |
| `device` | `"cpu"` | Device to train on (`"cpu"` or `"cuda"`) |
| `val_loader` | `None` | Optional validation `DataLoader` |

### forward_torch

`forward_torch` runs a forward pass without the training loop.

```python
from npcpy.ml_funcs import forward_torch

inputs = torch.randn(5, 10)
result = forward_torch(trained_model, inputs, device="cpu")

print(result["outputs"])        # raw tensor output
print(result["output_numpy"])   # numpy array
```

Set `grad=True` to compute gradients (for analysis or custom training):

```python
result = forward_torch(trained_model, inputs, grad=True)
```

## Time Series Functions

### fit_timeseries

`fit_timeseries` fits ARIMA, SARIMA, or exponential smoothing models using statsmodels.

```python
import numpy as np
from npcpy.ml_funcs import fit_timeseries

# Generate sample time series
np.random.seed(42)
series = np.cumsum(np.random.randn(200)) + 100

# Fit ARIMA
result = fit_timeseries(series, method="arima", order=(1, 1, 1))
print(f"AIC: {result['aic']:.2f}")
print(f"BIC: {result['bic']:.2f}")

# Fit SARIMA with seasonal component
result = fit_timeseries(
    series,
    method="sarima",
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
)

# Fit exponential smoothing
result = fit_timeseries(series, method="exp_smoothing")
print(f"SSE: {result['sse']:.2f}")
```

### forecast_timeseries

`forecast_timeseries` generates predictions from a fitted time series model.

```python
from npcpy.ml_funcs import forecast_timeseries

model = result["model"]

# Forecast 30 periods ahead
forecast = forecast_timeseries(model, horizon=30)

print(forecast["forecast"])       # point forecasts
if "conf_int" in forecast:
    print(forecast["conf_int"])   # confidence intervals
```

### Supported Time Series Methods

| Method | Model | Requirements |
|--------|-------|-------------|
| `arima` | ARIMA(p,d,q) | statsmodels |
| `sarima` | SARIMAX with seasonal order | statsmodels |
| `exp_smoothing` | Holt-Winters exponential smoothing | statsmodels |

## Serialization

`serialize_model` and `deserialize_model` save and load models using safe formats (no pickle).

```python
from npcpy.ml_funcs import serialize_model, deserialize_model

# Save with joblib (default, works with sklearn models)
serialize_model(result["model"], "model.joblib")

# Load it back
loaded_model = deserialize_model("model.joblib")

# Save PyTorch models with safetensors
serialize_model(torch_model, "model.safetensors", format="safetensors")

# Load safetensors (returns state dict)
state_dict = deserialize_model("model.safetensors")
```

### Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| `joblib` | `.joblib` | sklearn models, XGBoost, general Python objects |
| `safetensors` | `.safetensors` | PyTorch models (saves/loads `state_dict`) |

Auto-detection works based on file extension. To be explicit, pass `format="joblib"` or `format="safetensors"`.

## Utility Functions

### get_model_params

```python
from npcpy.ml_funcs import get_model_params

params = get_model_params(result["model"])
print(params)  # {'n_estimators': 100, 'max_depth': None, ...}
```

### set_model_params

```python
from npcpy.ml_funcs import set_model_params

updated_model = set_model_params(result["model"], {"n_estimators": 200})
```

## Full Workflow Example

Here is a complete example that trains, evaluates, ensembles, and serializes models:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from npcpy.ml_funcs import (
    fit_model, score_model, ensemble_predict,
    cross_validate, serialize_model,
)

# Data
X, y = make_classification(n_samples=500, n_features=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Grid search
result = fit_model(
    X_train, y_train,
    model="RandomForestClassifier",
    matrix={
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
    },
)
print(f"Trained {len(result['models'])} configurations")

# Score all models
scores = score_model(X_test, y_test, model=result["models"], metrics=["accuracy", "f1"])
for i, s in enumerate(scores["all_scores"]):
    print(f"  Config {i}: accuracy={s['accuracy']:.4f} f1={s['f1']:.4f}")

# Ensemble the top 3
top_indices = sorted(range(len(scores["all_scores"])),
                     key=lambda i: scores["all_scores"][i]["accuracy"],
                     reverse=True)[:3]
top_models = [result["models"][i] for i in top_indices]

ensemble = ensemble_predict(X_test, top_models, method="vote")
ensemble_scores = score_model(X_test, y_test, model=top_models[0], metrics=["accuracy"])
print(f"Ensemble (vote): {len(top_models)} models")

# Cross-validate the best model
cv = cross_validate(X, y, model="RandomForestClassifier",
                    cv=5, metrics=["accuracy", "f1"], n_estimators=200)
for metric, stats in cv.items():
    print(f"  CV {metric}: {stats['mean']:.4f} +/- {stats['std']:.4f}")

# Save the best model
serialize_model(top_models[0], "best_model.joblib")
print("Model saved to best_model.joblib")
```
