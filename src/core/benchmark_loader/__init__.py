"""Benchmark loaders: register concrete loaders on import."""

from .base_loader import (
    BenchmarkType,
    BenchmarkConfig,
    BaseBenchmarkLoader,
    BenchmarkLoaderFactory,
    load_benchmark,
    get_benchmark_info,
    create_livecodebench_config,
    create_gsm8k_config,
)

from . import livecodebench_loader
from . import gsm8k_loader

__all__ = [
    "BenchmarkType",
    "BenchmarkConfig",
    "BaseBenchmarkLoader",
    "BenchmarkLoaderFactory",
    "load_benchmark",
    "get_benchmark_info",
    "create_livecodebench_config",
    "create_gsm8k_config",
    "livecodebench_loader",
    "gsm8k_loader",
]
