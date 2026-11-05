"""chunk-bench: a small, reproducible benchmark for RAG chunking strategies."""

from .datasets import load_dataset
from .runner import run_benchmark
from .strategies import available_strategies, get_strategy, register_strategy

__version__ = "0.1.0"

__all__ = [
    "available_strategies",
    "get_strategy",
    "load_dataset",
    "register_strategy",
    "run_benchmark",
]
