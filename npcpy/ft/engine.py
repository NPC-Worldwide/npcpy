"""
Compute engine abstraction for evolutionary algorithms.

Allows NEAT and other GA operations to run on different backends:
  - numpy (default, CPU)
  - jax (GPU/TPU via JAX)
  - mlx (Apple Silicon via MLX)
  - cuda (NVIDIA GPU via PyTorch)

Usage:
    engine = get_engine("jax")
    arr = engine.array([1.0, 2.0, 3.0])
    result = engine.tanh(arr)
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Sequence
import numpy as np


class Engine(ABC):
    """Abstract compute backend for array operations."""

    name: str = "abstract"

    # --- Array creation ---

    @abstractmethod
    def array(self, data, dtype=None):
        """Create an array from data."""

    @abstractmethod
    def zeros(self, shape, dtype=None):
        """Create a zero-filled array."""

    @abstractmethod
    def ones(self, shape, dtype=None):
        """Create a ones-filled array."""

    @abstractmethod
    def randn(self, *shape, key=None):
        """Standard normal random array."""

    @abstractmethod
    def rand(self, *shape, key=None):
        """Uniform [0,1) random array."""

    @abstractmethod
    def arange(self, start, stop=None, step=1):
        """Range array."""

    # --- Math ops ---

    @abstractmethod
    def dot(self, a, b):
        """Matrix/vector dot product."""

    @abstractmethod
    def tanh(self, x):
        """Hyperbolic tangent activation."""

    @abstractmethod
    def relu(self, x):
        """ReLU activation."""

    @abstractmethod
    def sigmoid(self, x):
        """Sigmoid activation."""

    @abstractmethod
    def softmax(self, x, axis=-1):
        """Softmax along axis."""

    @abstractmethod
    def sum(self, x, axis=None):
        """Sum reduction."""

    @abstractmethod
    def mean(self, x, axis=None):
        """Mean reduction."""

    @abstractmethod
    def max(self, x, axis=None):
        """Max reduction."""

    @abstractmethod
    def argmax(self, x, axis=None):
        """Argmax."""

    @abstractmethod
    def abs(self, x):
        """Absolute value."""

    @abstractmethod
    def clip(self, x, a_min, a_max):
        """Clip values."""

    @abstractmethod
    def sqrt(self, x):
        """Square root."""

    @abstractmethod
    def exp(self, x):
        """Exponential."""

    # --- Array manipulation ---

    @abstractmethod
    def concatenate(self, arrays, axis=0):
        """Concatenate arrays."""

    @abstractmethod
    def stack(self, arrays, axis=0):
        """Stack arrays along new axis."""

    @abstractmethod
    def reshape(self, x, shape):
        """Reshape array."""

    @abstractmethod
    def where(self, condition, x, y):
        """Element-wise conditional."""

    # --- Conversion ---

    @abstractmethod
    def to_numpy(self, x) -> np.ndarray:
        """Convert to numpy array."""

    @abstractmethod
    def from_numpy(self, x: np.ndarray):
        """Convert from numpy array."""


class NumPyEngine(Engine):
    """NumPy CPU backend."""

    name = "numpy"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype or np.float64)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or np.float64)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype or np.float64)

    def randn(self, *shape, key=None):
        return self.rng.standard_normal(shape)

    def rand(self, *shape, key=None):
        return self.rng.random(shape)

    def arange(self, start, stop=None, step=1):
        return np.arange(start, stop, step)

    def dot(self, a, b):
        return np.dot(a, b)

    def tanh(self, x):
        return np.tanh(x)

    def relu(self, x):
        return np.maximum(x, 0)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def softmax(self, x, axis=-1):
        e = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e / np.sum(e, axis=axis, keepdims=True)

    def sum(self, x, axis=None):
        return np.sum(x, axis=axis)

    def mean(self, x, axis=None):
        return np.mean(x, axis=axis)

    def max(self, x, axis=None):
        return np.max(x, axis=axis)

    def argmax(self, x, axis=None):
        return np.argmax(x, axis=axis)

    def abs(self, x):
        return np.abs(x)

    def clip(self, x, a_min, a_max):
        return np.clip(x, a_min, a_max)

    def sqrt(self, x):
        return np.sqrt(x)

    def exp(self, x):
        return np.exp(x)

    def concatenate(self, arrays, axis=0):
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis=0):
        return np.stack(arrays, axis=axis)

    def reshape(self, x, shape):
        return np.reshape(x, shape)

    def where(self, condition, x, y):
        return np.where(condition, x, y)

    def to_numpy(self, x):
        return np.asarray(x)

    def from_numpy(self, x):
        return np.asarray(x)


class JAXEngine(Engine):
    """JAX backend for GPU/TPU acceleration."""

    name = "jax"

    def __init__(self, seed=0):
        import jax
        import jax.numpy as jnp
        from jax import random
        self._jnp = jnp
        self._jax = jax
        self._random = random
        self._key = random.PRNGKey(seed)

    def _next_key(self):
        self._key, subkey = self._random.split(self._key)
        return subkey

    def array(self, data, dtype=None):
        return self._jnp.array(data, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return self._jnp.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return self._jnp.ones(shape, dtype=dtype)

    def randn(self, *shape, key=None):
        k = key if key is not None else self._next_key()
        return self._random.normal(k, shape)

    def rand(self, *shape, key=None):
        k = key if key is not None else self._next_key()
        return self._random.uniform(k, shape)

    def arange(self, start, stop=None, step=1):
        return self._jnp.arange(start, stop, step)

    def dot(self, a, b):
        return self._jnp.dot(a, b)

    def tanh(self, x):
        return self._jnp.tanh(x)

    def relu(self, x):
        return self._jax.nn.relu(x)

    def sigmoid(self, x):
        return self._jax.nn.sigmoid(x)

    def softmax(self, x, axis=-1):
        return self._jax.nn.softmax(x, axis=axis)

    def sum(self, x, axis=None):
        return self._jnp.sum(x, axis=axis)

    def mean(self, x, axis=None):
        return self._jnp.mean(x, axis=axis)

    def max(self, x, axis=None):
        return self._jnp.max(x, axis=axis)

    def argmax(self, x, axis=None):
        return self._jnp.argmax(x, axis=axis)

    def abs(self, x):
        return self._jnp.abs(x)

    def clip(self, x, a_min, a_max):
        return self._jnp.clip(x, a_min, a_max)

    def sqrt(self, x):
        return self._jnp.sqrt(x)

    def exp(self, x):
        return self._jnp.exp(x)

    def concatenate(self, arrays, axis=0):
        return self._jnp.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis=0):
        return self._jnp.stack(arrays, axis=axis)

    def reshape(self, x, shape):
        return self._jnp.reshape(x, shape)

    def where(self, condition, x, y):
        return self._jnp.where(condition, x, y)

    def to_numpy(self, x):
        import numpy as np
        return np.asarray(x)

    def from_numpy(self, x):
        return self._jnp.array(x)


class MLXEngine(Engine):
    """Apple Silicon MLX backend."""

    name = "mlx"

    def __init__(self, seed=None):
        import mlx.core as mx
        self._mx = mx
        if seed is not None:
            mx.random.seed(seed)

    def array(self, data, dtype=None):
        return self._mx.array(data)

    def zeros(self, shape, dtype=None):
        return self._mx.zeros(shape)

    def ones(self, shape, dtype=None):
        return self._mx.ones(shape)

    def randn(self, *shape, key=None):
        return self._mx.random.normal(shape)

    def rand(self, *shape, key=None):
        return self._mx.random.uniform(shape=shape)

    def arange(self, start, stop=None, step=1):
        return self._mx.arange(start, stop, step)

    def dot(self, a, b):
        return self._mx.matmul(a, b) if a.ndim > 1 else (a * b).sum()

    def tanh(self, x):
        return self._mx.tanh(x)

    def relu(self, x):
        return self._mx.maximum(x, 0)

    def sigmoid(self, x):
        return self._mx.sigmoid(x)

    def softmax(self, x, axis=-1):
        return self._mx.softmax(x, axis=axis)

    def sum(self, x, axis=None):
        return self._mx.sum(x, axis=axis)

    def mean(self, x, axis=None):
        return self._mx.mean(x, axis=axis)

    def max(self, x, axis=None):
        return self._mx.max(x, axis=axis)

    def argmax(self, x, axis=None):
        return self._mx.argmax(x, axis=axis)

    def abs(self, x):
        return self._mx.abs(x)

    def clip(self, x, a_min, a_max):
        return self._mx.clip(x, a_min, a_max)

    def sqrt(self, x):
        return self._mx.sqrt(x)

    def exp(self, x):
        return self._mx.exp(x)

    def concatenate(self, arrays, axis=0):
        return self._mx.concatenate(arrays, axis=axis)

    def stack(self, arrays, axis=0):
        return self._mx.stack(arrays, axis=axis)

    def reshape(self, x, shape):
        return self._mx.reshape(x, shape)

    def where(self, condition, x, y):
        return self._mx.where(condition, x, y)

    def to_numpy(self, x):
        import numpy as np
        return np.array(x)

    def from_numpy(self, x):
        return self._mx.array(x)


class CUDAEngine(Engine):
    """PyTorch CUDA backend."""

    name = "cuda"

    def __init__(self, device=None, seed=None):
        import torch
        self._torch = torch
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)

    def array(self, data, dtype=None):
        t = self._torch.tensor(data, dtype=dtype or self._torch.float64)
        return t.to(self.device)

    def zeros(self, shape, dtype=None):
        return self._torch.zeros(shape, dtype=dtype or self._torch.float64, device=self.device)

    def ones(self, shape, dtype=None):
        return self._torch.ones(shape, dtype=dtype or self._torch.float64, device=self.device)

    def randn(self, *shape, key=None):
        return self._torch.randn(*shape, dtype=self._torch.float64, device=self.device)

    def rand(self, *shape, key=None):
        return self._torch.rand(*shape, dtype=self._torch.float64, device=self.device)

    def arange(self, start, stop=None, step=1):
        if stop is None:
            return self._torch.arange(start, step=step, device=self.device)
        return self._torch.arange(start, stop, step, device=self.device)

    def dot(self, a, b):
        return self._torch.matmul(a, b) if a.dim() > 1 else (a * b).sum()

    def tanh(self, x):
        return self._torch.tanh(x)

    def relu(self, x):
        return self._torch.relu(x)

    def sigmoid(self, x):
        return self._torch.sigmoid(x)

    def softmax(self, x, axis=-1):
        return self._torch.softmax(x, dim=axis)

    def sum(self, x, axis=None):
        return self._torch.sum(x) if axis is None else self._torch.sum(x, dim=axis)

    def mean(self, x, axis=None):
        return self._torch.mean(x) if axis is None else self._torch.mean(x, dim=axis)

    def max(self, x, axis=None):
        if axis is None:
            return self._torch.max(x)
        return self._torch.max(x, dim=axis).values

    def argmax(self, x, axis=None):
        if axis is None:
            return self._torch.argmax(x)
        return self._torch.argmax(x, dim=axis)

    def abs(self, x):
        return self._torch.abs(x)

    def clip(self, x, a_min, a_max):
        return self._torch.clamp(x, a_min, a_max)

    def sqrt(self, x):
        return self._torch.sqrt(x)

    def exp(self, x):
        return self._torch.exp(x)

    def concatenate(self, arrays, axis=0):
        return self._torch.cat(arrays, dim=axis)

    def stack(self, arrays, axis=0):
        return self._torch.stack(arrays, dim=axis)

    def reshape(self, x, shape):
        return self._torch.reshape(x, shape)

    def where(self, condition, x, y):
        return self._torch.where(condition, x, y)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def from_numpy(self, x):
        return self._torch.tensor(x, dtype=self._torch.float64, device=self.device)


_ENGINE_REGISTRY = {
    "numpy": NumPyEngine,
    "cpu": NumPyEngine,
    "jax": JAXEngine,
    "mlx": MLXEngine,
    "cuda": CUDAEngine,
    "torch": CUDAEngine,
}

_cached_engines = {}


def get_engine(name: str = "numpy", seed: Optional[int] = None, **kwargs) -> Engine:
    """
    Get a compute engine by name.

    Args:
        name: One of "numpy", "cpu", "jax", "mlx", "cuda", "torch"
        seed: Optional random seed
        **kwargs: Additional engine-specific arguments

    Returns:
        Engine instance
    """
    name = name.lower()
    if name not in _ENGINE_REGISTRY:
        available = ", ".join(_ENGINE_REGISTRY.keys())
        raise ValueError(f"Unknown engine '{name}'. Available: {available}")

    cache_key = (name, seed)
    if cache_key not in _cached_engines:
        engine_cls = _ENGINE_REGISTRY[name]
        try:
            _cached_engines[cache_key] = engine_cls(seed=seed, **kwargs)
        except ImportError as e:
            raise ImportError(
                f"Engine '{name}' requires additional dependencies: {e}"
            ) from e

    return _cached_engines[cache_key]


def list_engines() -> list:
    """List available engine names."""
    return list(_ENGINE_REGISTRY.keys())
