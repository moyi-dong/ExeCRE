"""LLM brute-force / simulation style solutions (correctness over efficiency)."""

import re
from src.baselines.base_solver import BaseSolver
from src.core.problem import Problem
from src.core.solution import Solution
from src.baselines.bruteforce_solve.prompts import (
    SYSTEM_PROMPT_FOR_SIMULATION_CODE,
    get_simulation_question_template_answer,
)
from src.engine import get_engine
from src.utils import extract_code


def clean_code_blocks(text: str) -> str:
    """First ```python``` / ``` block body, else full stripped text."""
    text = text.strip()
    pattern = r"```(?:python|py)?\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return match.group(1).strip()
    return text


class BruteforceSolver(BaseSolver):
    """Simulation-oriented codegen with bruteforce-oriented system prompt."""

    def __init__(
        self,
        name: str = "Bruteforce",
        model_name: str = "gpt-4o",
        **kwargs,
    ):
        super().__init__(name, model_name=model_name, **kwargs)
        self.model_name = model_name
        self._engine = None
        self.engine_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["temperature", "max_tokens", "top_p", "system_prompt"]
        }

    def solve(self, problem: Problem) -> Solution:
        try:
            system_prompt = SYSTEM_PROMPT_FOR_SIMULATION_CODE
            user_prompt = get_simulation_question_template_answer(problem)
            engine = self._get_engine()

            model_output = engine.generate(
                user_prompt,
                system_prompt=system_prompt,
                **self.engine_kwargs,
            )

            solution_code = clean_code_blocks(model_output)

            if len(solution_code) == 0:
                solution_code = extract_code(model_output)

            if len(solution_code) == 0:
                retry_prompt = user_prompt + "\nPlease provide the solution without comments."
                model_output = engine.generate(
                    retry_prompt,
                    system_prompt=system_prompt,
                    **self.engine_kwargs,
                )
                solution_code = clean_code_blocks(model_output)
                if len(solution_code) == 0:
                    solution_code = extract_code(model_output)

            is_normal_end = len(solution_code) > 0

            return Solution(
                code=solution_code,
                problem_id=problem.question_id,
                is_normal_end=is_normal_end,
            )

        except Exception as e:
            print(f"Error in BruteforceSolver.solve() for problem {problem.question_id}: {e}")
            return Solution(
                code="",
                problem_id=problem.question_id,
                is_normal_end=False,
            )

    def _get_engine(self):
        if self._engine is None:
            self._engine = get_engine(self.model_name, **self.engine_kwargs)
        return self._engine
