"""Benchmark-specific prompt adapters for DirectAnswer."""

from src.baselines.direct_solve.prompt_adapters.base_adapter import PromptAdapter
from src.baselines.direct_solve.prompt_adapters.livecodebench_adapter import (
    LiveCodeBenchPromptAdapter,
)


def get_adapter_for_benchmark(benchmark: str) -> PromptAdapter:
    if benchmark == "LiveCodeBench":
        return LiveCodeBenchPromptAdapter()
    raise ValueError(
        f"Unsupported benchmark: {benchmark}. Currently only 'LiveCodeBench' is supported."
    )


__all__ = [
    "PromptAdapter",
    "LiveCodeBenchPromptAdapter",
    "get_adapter_for_benchmark",
]
