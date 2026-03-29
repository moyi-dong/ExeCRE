"""LiveCodeBench -> chat messages."""

from typing import Any, Dict, List

from src.core.problem import Problem
from src.baselines.direct_solve.prompt_adapters.base_adapter import PromptAdapter
from src.baselines.direct_solve.prompts import format_livecodebench_prompt


class LiveCodeBenchPromptAdapter(PromptAdapter):
    def __init__(self):
        super().__init__("LiveCodeBench")

    def format_prompt(self, problem: Problem) -> List[Dict[str, Any]]:
        return format_livecodebench_prompt(problem)
