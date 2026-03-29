"""LiveCodeBench loader: wraps lcb_runner and maps rows to Problem."""

import sys
from typing import Dict, List, Any
from pathlib import Path

from ...utils.path_manager import get_src_dir
from .base_loader import (
    BaseBenchmarkLoader,
    BenchmarkConfig,
    BenchmarkType,
    BenchmarkLoaderFactory
)
from ..problem import Problem


def _get_livecodebench_path() -> Path:
    src_dir = get_src_dir()
    lcb_path = src_dir / "benchmark_repo" / "LiveCodeBench"
    if not lcb_path.exists():
        raise RuntimeError(f"LiveCodeBench not found at {lcb_path}")
    return lcb_path


def _import_lcb_modules():
    lcb_path = _get_livecodebench_path()
    lcb_path_str = str(lcb_path)

    if lcb_path_str not in sys.path:
        sys.path.insert(0, lcb_path_str)

    try:
        from lcb_runner.benchmarks import (
            load_code_generation_dataset_streaming,
            load_code_generation_dataset_not_fast,
            CodeGenerationProblem
        )
        return {
            'load_code_generation_dataset_streaming': load_code_generation_dataset_streaming,
            'load_code_generation_dataset_not_fast': load_code_generation_dataset_not_fast,
            'CodeGenerationProblem': CodeGenerationProblem
        }
    except ImportError as e:
        raise ImportError(
            f"Failed to import LiveCodeBench modules. "
            f"Make sure LiveCodeBench is properly installed at {lcb_path}. "
            f"Original error: {e}"
        )


class LiveCodeBenchLoader(BaseBenchmarkLoader):

    def _validate_config(self) -> None:
        if self.config.benchmark_type != BenchmarkType.LIVECODEBENCH:
            raise ValueError(
                f"Invalid benchmark type for LiveCodeBench loader: {self.config.benchmark_type}"
            )

        if self.config.release_version is None:
            self.config.release_version = "release_latest"

    def load_benchmark(self) -> List[Problem]:
        lcb_modules = _import_lcb_modules()
        load_streaming = lcb_modules['load_code_generation_dataset_streaming']
        load_not_fast = lcb_modules['load_code_generation_dataset_not_fast']
        CodeGenerationProblem = lcb_modules['CodeGenerationProblem']

        if self.config.extra_params and self.config.extra_params.get("not_fast", False):
            raw_dataset = load_not_fast(self.config.release_version)
        else:
            raw_dataset = load_streaming(
                self.config.release_version,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )

        problem_instances = []
        for raw_problem in raw_dataset:
            problem_instance = self._convert_to_problem(raw_problem)
            problem_instances.append(problem_instance)

        return problem_instances

    def _convert_to_problem(self, raw_problem: 'CodeGenerationProblem') -> Problem:
        public_test_cases = []
        for test in raw_problem.public_test_cases:
            public_test_cases.append({
                "input": test.input,
                "output": test.output,
                "test_type": test.testtype.value if hasattr(test.testtype, 'value') else str(test.testtype),
                "is_public": True
            })

        private_test_cases = []
        for test in raw_problem.private_test_cases:
            private_test_cases.append({
                "input": test.input,
                "output": test.output,
                "test_type": test.testtype.value if hasattr(test.testtype, 'value') else str(test.testtype),
                "is_public": False
            })

        problem = Problem(
            question_id=raw_problem.question_id,
            question_title=raw_problem.question_title,
            question_content=raw_problem.question_content,
            difficulty=raw_problem.difficulty.value if hasattr(raw_problem.difficulty, 'value') else str(raw_problem.difficulty),
            benchmark="LiveCodeBench",
            starter_code=raw_problem.starter_code,
            public_test_cases=public_test_cases,
            private_test_cases=private_test_cases,
            platform=raw_problem.platform.value if hasattr(raw_problem.platform, 'value') else str(raw_problem.platform),
            contest_id=raw_problem.contest_id,
            contest_date=raw_problem.contest_date,
            metadata=raw_problem.metadata if isinstance(raw_problem.metadata, dict) else {}
        )

        return problem

    def get_benchmark_info(self) -> Dict[str, Any]:
        dataset = self.load_benchmark()

        total_problems = len(dataset)
        difficulty_stats = {}
        platform_stats = {}
        date_range = {"min": None, "max": None}

        for problem in dataset:
            difficulty = problem.difficulty
            difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1

            platform = problem.platform
            platform_stats[platform] = platform_stats.get(platform, 0) + 1

            if problem.contest_date:
                if date_range["min"] is None or problem.contest_date < date_range["min"]:
                    date_range["min"] = problem.contest_date
                if date_range["max"] is None or problem.contest_date > date_range["max"]:
                    date_range["max"] = problem.contest_date

        return {
            "benchmark_type": "LiveCodeBench",
            "release_version": self.config.release_version,
            "total_problems": total_problems,
            "difficulty_distribution": difficulty_stats,
            "platform_distribution": platform_stats,
            "date_range": {
                "min": date_range["min"].isoformat() if date_range["min"] else None,
                "max": date_range["max"].isoformat() if date_range["max"] else None
            },
            "filter_config": {
                "start_date": self.config.start_date,
                "end_date": self.config.end_date,
                "extra_params": self.config.extra_params
            }
        }


BenchmarkLoaderFactory.register_loader(BenchmarkType.LIVECODEBENCH, LiveCodeBenchLoader)
