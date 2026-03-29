"""Abstract benchmark loader, config, factory, and convenience helpers."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from ..problem import Problem


class BenchmarkType(Enum):
    LIVECODEBENCH = "LiveCodeBench"
    HUMANEVAL = "HumanEval"
    MBPP = "MBPP"
    LEETCODE_HARD = "LeetCodeHard"
    GSM8K = "GSM8K"


@dataclass
class BenchmarkConfig:
    benchmark_type: BenchmarkType
    release_version: Optional[str] = None
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None
    question_ids: Optional[List[str]] = None
    extra_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.start_date:
            try:
                datetime.strptime(self.start_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("start_date must be in YYYY-MM-DD format")

        if self.end_date:
            try:
                datetime.strptime(self.end_date, "%Y-%m-%d")
            except ValueError:
                raise ValueError("end_date must be in YYYY-MM-DD format")


class BaseBenchmarkLoader(ABC):

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> None:
        pass

    @abstractmethod
    def load_benchmark(self) -> List[Problem]:
        pass

    @abstractmethod
    def get_benchmark_info(self) -> Dict[str, Any]:
        pass


class BenchmarkLoaderFactory:

    _loaders = {}

    @classmethod
    def register_loader(cls, benchmark_type: BenchmarkType, loader_class: type):
        cls._loaders[benchmark_type] = loader_class

    @classmethod
    def create_loader(cls, config: BenchmarkConfig) -> BaseBenchmarkLoader:
        if config.benchmark_type not in cls._loaders:
            raise ValueError(f"Unsupported benchmark type: {config.benchmark_type}")

        loader_class = cls._loaders[config.benchmark_type]
        return loader_class(config)

    @classmethod
    def get_supported_benchmarks(cls) -> List[BenchmarkType]:
        return list(cls._loaders.keys())


def load_benchmark(config: BenchmarkConfig) -> List[Problem]:
    loader = BenchmarkLoaderFactory.create_loader(config)
    return loader.load_benchmark()


def get_benchmark_info(config: BenchmarkConfig) -> Dict[str, Any]:
    loader = BenchmarkLoaderFactory.create_loader(config)
    return loader.get_benchmark_info()


def create_livecodebench_config(
    release_version: str = "release_latest",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    **kwargs
) -> BenchmarkConfig:
    return BenchmarkConfig(
        benchmark_type=BenchmarkType.LIVECODEBENCH,
        release_version=release_version,
        start_date=start_date,
        end_date=end_date,
        extra_params=kwargs
    )


def create_gsm8k_config(
    split: str = "test",
    **kwargs
) -> BenchmarkConfig:
    """split: 'test' or 'train' -> data/test.jsonl or data/train.jsonl."""
    return BenchmarkConfig(
        benchmark_type=BenchmarkType.GSM8K,
        extra_params={"split": split, **kwargs}
    )
