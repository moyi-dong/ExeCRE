"""Direct LLM code generation with per-benchmark prompt adapters."""

from typing import Union
from src.baselines.base_solver import BaseSolver
from src.core.problem import Problem
from src.core.solution import Solution
from src.baselines.direct_solve.prompt_adapters import get_adapter_for_benchmark
from src.engine import get_engine
from src.utils import extract_code


class DirectAnswerSolver(BaseSolver):
    """Single-shot codegen via ``get_engine`` + ``extract_code``."""

    def __init__(
        self,
        name: str = "DirectAnswer",
        model_name: str = "gpt-4o",
        **kwargs,
    ):
        super().__init__(name, model_name=model_name, **kwargs)
        self.model_name = model_name
        self._engine = None
        self._adapter = None
        self.engine_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["temperature", "max_tokens", "top_p", "system_prompt"]
        }

    def solve(self, problem: Problem) -> Solution:
        try:
            benchmark = problem.benchmark or "LiveCodeBench"
            adapter = self._get_adapter(benchmark)
            chat_messages = adapter.format_prompt(problem)
            engine = self._get_engine()

            system_prompt = None
            user_prompt = None
            for msg in chat_messages:
                if msg.get("role") == "system":
                    system_prompt = msg.get("content", "")
                elif msg.get("role") == "user":
                    user_prompt = msg.get("content", "")

            if user_prompt is None:
                raise ValueError("chat prompt must include a user message")

            model_output = engine.generate(
                user_prompt,
                system_prompt=system_prompt,
                **self.engine_kwargs,
            )

            solution_code = extract_code(model_output)

            if len(solution_code) == 0:
                retry_prompt = user_prompt + "\nPlease provide the solution without comments."
                model_output = engine.generate(
                    retry_prompt,
                    system_prompt=system_prompt,
                    **self.engine_kwargs,
                )
                solution_code = extract_code(model_output)

            is_normal_end = len(solution_code) > 0

            return Solution(
                code=solution_code,
                problem_id=problem.question_id,
                is_normal_end=is_normal_end,
            )

        except Exception as e:
            print(f"Error in DirectAnswerSolver.solve() for problem {problem.question_id}: {e}")
            return Solution(
                code="",
                problem_id=problem.question_id,
                is_normal_end=False,
            )

    def _get_engine(self):
        if self._engine is None:
            self._engine = get_engine(self.model_name, **self.engine_kwargs)
        return self._engine

    def _get_adapter(self, benchmark: str):
        if self._adapter is None or self._adapter.benchmark != benchmark:
            self._adapter = get_adapter_for_benchmark(benchmark)
        return self._adapter
