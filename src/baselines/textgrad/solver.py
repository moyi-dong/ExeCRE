"""TextGrad iterative refinement on public tests."""

import sys
from pathlib import Path
from typing import List, Optional, Union

_textgrad_parent_path = Path(__file__).parent
if str(_textgrad_parent_path) not in sys.path:
    sys.path.insert(0, str(_textgrad_parent_path))

from src.baselines.base_solver import BaseSolver
from src.core.problem import Problem
from src.core.solution import Solution

import textgrad
from textgrad.variable import Variable
from textgrad.optimizer.optimizer import TextualGradientDescent
from textgrad.engine import get_engine as tg_get_engine

from src.baselines.textgrad.py_eval import evaluate
from src.baselines.textgrad.prompts import CODE_INSTANCE_ROLE_DESCRIPTION, CodeTestTimewithTests


class TextgradSolver(BaseSolver):
    """Public-test feedback loop with TextualGradientDescent."""

    def __init__(
        self,
        name: str = "Textgrad",
        model_name: str = "openai-gpt-4o",
        max_iters: int = 4,
        direct_solve_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(name, model_name=model_name, max_iters=max_iters, **kwargs)
        self.model_name = model_name
        self.max_iters = max_iters
        self.direct_solve_dir = Path(direct_solve_dir) if direct_solve_dir else None

        self._engine = None
        self._direct_solver = None

    def _get_engine(self):
        if self._engine is None:
            self._engine = tg_get_engine(self.model_name)
            textgrad.set_backward_engine(self._engine, override=True)
        return self._engine

    def _get_direct_solver(self):
        if self._direct_solver is None:
            from src.baselines.direct_solve.solver import DirectAnswerSolver

            self._direct_solver = DirectAnswerSolver(model_name=self.model_name)
        return self._direct_solver

    def _read_initial_code_from_dir(self, problem: Problem) -> Optional[str]:
        """First Solution_Code row from ``direct_solve_dir / {id}.csv``."""
        if self.direct_solve_dir is None:
            return None

        csv_path = self.direct_solve_dir / f"{problem.question_id}.csv"

        if not csv_path.exists():
            return None

        try:
            import csv

            with open(csv_path, "r", encoding="utf-8-sig") as csvfile:
                reader = csv.DictReader(csvfile)
                first_row = next(reader, None)

                if first_row is None:
                    return None

                solution_code = first_row.get("Solution_Code", "")

                if solution_code and solution_code.strip():
                    is_normal_end = first_row.get("Is_Normal_End", "True")
                    if str(is_normal_end).lower() == "true":
                        return solution_code.strip()

                return None

        except Exception as e:
            print(f"Failed to read initial code {csv_path}: {e}")
            return None

    def _get_initial_code(self, problem: Problem) -> tuple[str, bool]:
        initial_code = self._read_initial_code_from_dir(problem)
        if initial_code is not None:
            print(f"Loaded initial code from direct_solve_dir: {problem.question_id}")
            return initial_code, True

        print(f"Running DirectAnswerSolver for initial code: {problem.question_id}")
        direct_solver = self._get_direct_solver()
        initial_solution = direct_solver.solve(problem)

        return initial_solution.code, initial_solution.is_normal_end

    def _optimization_one_iteration(
        self,
        optimizer: TextualGradientDescent,
        instance_var: Variable,
        problem_content: str,
        test_string: str,
    ) -> None:
        engine = self._get_engine()

        optimizer.zero_grad()
        loss_fn = CodeTestTimewithTests(engine=engine)
        test_time_loss = loss_fn(problem_content, instance_var, test_string)
        test_time_loss.backward()
        optimizer.step()

    def solve(self, problem: Problem) -> List[Solution]:
        """Return one ``Solution`` per round; last marked ``is_final``."""
        results = []

        try:
            engine = self._get_engine()

            initial_code, is_normal_end = self._get_initial_code(problem)

            if not initial_code or not initial_code.strip():
                return [
                    Solution(
                        code="",
                        problem_id=problem.question_id,
                        is_normal_end=False,
                        round_index=0,
                        is_final=True,
                    )
                ]

            instance_var = Variable(
                initial_code,
                requires_grad=True,
                role_description=CODE_INSTANCE_ROLE_DESCRIPTION,
            )

            optimizer = TextualGradientDescent(
                engine=engine,
                parameters=[instance_var],
                constraints=[
                    "Do not add asserts to the code",
                    "Code must contain imports",
                ],
            )

            passed, test_string = evaluate(
                instance_var.value,
                problem.public_test_cases,
                problem.metadata,
            )

            initial_solution = Solution(
                code=instance_var.value,
                problem_id=problem.question_id,
                is_normal_end=is_normal_end,
                round_index=0,
                is_final=False,
                local_passed=passed,
                local_result_type=test_string,
            )
            results.append(initial_solution)

            for iter_idx in range(self.max_iters):
                if iter_idx != 0 and passed:
                    break

                print(f"{problem.question_id} iter {iter_idx + 1} before_opt passed={passed}")

                self._optimization_one_iteration(
                    optimizer,
                    instance_var,
                    problem.question_content,
                    test_string,
                )

                passed, test_string = evaluate(
                    instance_var.value,
                    problem.public_test_cases,
                    problem.metadata,
                )

                round_solution = Solution(
                    code=instance_var.value,
                    problem_id=problem.question_id,
                    is_normal_end=True,
                    round_index=iter_idx + 1,
                    is_final=False,
                    local_passed=passed,
                    local_result_type=test_string,
                )
                results.append(round_solution)

            if results:
                results[-1].is_final = True

            return results

        except Exception as e:
            print(f"TextgradSolver.solve() error {problem.question_id}: {e}")
            return [
                Solution(
                    code="",
                    problem_id=problem.question_id,
                    is_normal_end=False,
                    round_index=0,
                    is_final=True,
                )
            ]
