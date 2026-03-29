"""Abstract base for baseline solvers: Problem in, Solution(s) out, CSV save helper."""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import csv
import json
from datetime import datetime

from src.core.problem import Problem
from src.core.solution import Solution


class BaseSolver(ABC):
    """Subclass and implement ``solve``; optional multi-round solvers return a list with ``round_index`` / ``is_final``."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.config = kwargs

    @abstractmethod
    def solve(self, problem: Problem) -> Union[Solution, List[Solution]]:
        raise NotImplementedError("subclasses must implement solve()")

    def save_solution(
        self,
        problem: Problem,
        solution: Union[Solution, List[Solution]],
        save_dir: Path,
    ) -> Path:
        """Write one CSV per question: ``{question_id}.csv`` (one row per round if list)."""
        save_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(solution, Solution):
            solutions = [solution]
        elif isinstance(solution, list):
            if len(solution) == 0:
                raise ValueError("solution list must not be empty")
            solutions = solution
        else:
            raise ValueError(f"solution must be Solution or List[Solution], got {type(solution)}")

        rows = [self._solution_to_dict(problem, sol) for sol in solutions]
        csv_path = save_dir / f"{problem.question_id}.csv"

        if len(rows) > 0:
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return csv_path

    def _solution_to_dict(self, problem: Problem, solution: Solution) -> Dict[str, Any]:
        """Flatten ``Solution`` to uppercase_underscore keys for CSV (complex values JSON-encoded)."""

        def serialize_value(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, list):
                return json.dumps(value, ensure_ascii=False) if value else None
            if isinstance(value, dict):
                return json.dumps(value, ensure_ascii=False) if value else None
            if isinstance(value, datetime):
                return value.isoformat() if value else None
            return str(value)

        result = {
            "Question_Id": problem.question_id,
            "Question_Title": problem.question_title,
            "Platform": problem.platform or "",
            "Difficulty": problem.difficulty or "",
            "Contest_Date": serialize_value(problem.contest_date),
            "Solution_Code": solution.code or "",
            "Is_Normal_End": solution.is_normal_end,
            "Passed": solution.passed,
            "Result_Type": solution.result_type or "",
            "Error_Case_Indice": serialize_value(solution.error_case_indice),
            "Error_Case_Contents": serialize_value(solution.error_case_contents),
            "Round_Index": solution.round_index if solution.round_index is not None else 0,
            "Is_Final": solution.is_final,
            "Local_Passed": solution.local_passed if solution.local_passed is not None else "",
            "Local_Result_Type": solution.local_result_type or "",
            "Schema": solution.schema or "",
            "Simulation_Code": solution.simulation_code or "",
            "Metadata": serialize_value(solution.metadata),
        }

        if solution.execution_results:
            result["Execution_Results"] = serialize_value(solution.execution_results)
        else:
            result["Execution_Results"] = None

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
