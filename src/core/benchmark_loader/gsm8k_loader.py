"""GSM8K loader: JSONL rows with question/answer; gold from trailing '#### <number>'."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from ...utils.path_manager import get_src_dir
from .base_loader import (
    BaseBenchmarkLoader,
    BenchmarkConfig,
    BenchmarkType,
    BenchmarkLoaderFactory,
)
from ..problem import Problem


# Same as benchmark_repo/GSM8K/grade_school_math/dataset.py
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def _parse_gold_answer(answer_str: str) -> Any:
    match = ANS_RE.search(answer_str)
    if not match:
        return None
    s = match.group(1).strip().replace(",", "")
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return None


def _get_gsm8k_data_path(split: str) -> Path:
    src_dir = get_src_dir()
    path = src_dir / "benchmark_repo" / "GSM8K" / "grade_school_math" / "data" / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"GSM8K data not found: {path}")
    return path


class GSM8KLoader(BaseBenchmarkLoader):

    def _validate_config(self) -> None:
        if self.config.benchmark_type != BenchmarkType.GSM8K:
            raise ValueError(
                f"Invalid benchmark type for GSM8K loader: {self.config.benchmark_type}"
            )
        split = (self.config.extra_params or {}).get("split", "test")
        if split not in ("test", "train"):
            raise ValueError(f"GSM8K split must be 'test' or 'train', got: {split}")

    def load_benchmark(self) -> List[Problem]:
        split = (self.config.extra_params or {}).get("split", "test")
        path = _get_gsm8k_data_path(split)

        problems: List[Problem] = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                question = row.get("question", "")
                answer_raw = row.get("answer", "")
                gold = _parse_gold_answer(answer_raw)
                if gold is None:
                    continue
                problem = Problem(
                    question_id=f"gsm8k-{idx}",
                    question_title=f"GSM8K #{idx}",
                    question_content=question.strip(),
                    difficulty="all",
                    benchmark="GSM8K",
                    starter_code="",
                    public_test_cases=[],
                    private_test_cases=[],
                    platform="GSM8K",
                    contest_id="",
                    contest_date=None,
                    metadata={"gold_answer": gold},
                )
                problems.append(problem)

        return problems

    def get_benchmark_info(self) -> Dict[str, Any]:
        problems = self.load_benchmark()
        split = (self.config.extra_params or {}).get("split", "test")
        return {
            "benchmark_type": "GSM8K",
            "split": split,
            "total_problems": len(problems),
            "filter_config": self.config.extra_params,
        }


BenchmarkLoaderFactory.register_loader(BenchmarkType.GSM8K, GSM8KLoader)
