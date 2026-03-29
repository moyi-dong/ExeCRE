"""GSM8K PM + randomized inputs + EM4C; adopt PM answer if alpha passes else base LLM."""

import csv
import json
import random
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.baselines.base_solver import BaseSolver
from src.core.problem import Problem
from src.core.solution import Solution
from src.engine import get_engine

from src.EM4C.EM4C import step6_calculate_confidence
from src.EM4C.label_processors.majority_01 import Majority01Processor
from src.EM4C.confidence_calculators.dawid_skene import DawidSkeneCalculator

from src.baselines.gsm8k.pm_solver import (
    GSM8KPMSolver,
    PM_SYSTEM_PROMPT,
    _PM_FEW_SHOT_EXAMPLES,
    parse_pm_output,
    safe_execute,
    floatify_ans,
    _extract_code,
)

VERBOSE = True

# Inline base-style GSM8K prompt when falling back from PM path.
GSM8K_SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step. "
    "At the end of your response, write your final numeric answer on a new line "
    "in the format: Answer: <number>"
)


def _perturb_value(value, perturbation_range: float = 0.5) -> Any:
    """Scale int/float around original value; bool/None unchanged."""
    if value is None:
        return value

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value == 0:
            return random.randint(-5, 5)
        lo = value - max(1, int(abs(value) * perturbation_range))
        hi = value + max(1, int(abs(value) * perturbation_range))
        if lo > hi:
            lo, hi = hi, lo
        # keep positive if original is positive (avoid negative counts, etc.)
        if value > 0:
            lo = max(1, lo)
        return random.randint(lo, hi)

    if isinstance(value, float):
        if value == 0.0:
            return random.uniform(-5.0, 5.0)
        delta = abs(value) * perturbation_range
        lo = value - delta
        hi = value + delta
        if lo > hi:
            lo, hi = hi, lo
        if value > 0:
            lo = max(0.01, lo)
        return round(random.uniform(lo, hi), 4)

    return value


def generate_random_inputs(
    original_values: List,
    count: int = 300,
    perturbation_range: float = 0.5,
) -> List[List]:
    """Build ``count`` perturbed argument lists from ``original_values``."""
    inputs: List[List] = []
    for _ in range(count):
        row = [_perturb_value(v, perturbation_range) for v in original_values]
        inputs.append(row)
    return inputs


def _exec_solve_with_args(function_code: str, args: List) -> Any:
    """Run ``solve(*args)`` from PM ``function_code`` via ``safe_execute``; None on failure."""
    full_code = function_code + "\n" + f"ans = solve({', '.join(repr(a) for a in args)})"
    result = safe_execute(full_code)
    return floatify_ans(result)


class GSM8KExeCRESolver(BaseSolver):
    """PM candidates, random inputs, EM4C confidence, optional base fallback."""

    def __init__(
        self,
        name: str = "GSM8KExeCRE",
        model_name: str = "gpt-4o",
        max_solution_candidates: int = 10,
        em4c_test_case_count: int = 300,
        allowed_error_ratio: float = 0.3,
        alpha_threshold: float = 0.90,
        sampling_temperature: float = 0.8,
        perturbation_range: float = 0.5,
        fallback_to_base: bool = True,
        direct_solve_dir: Optional[Union[str, Path]] = None,
        bruteforce_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(name, model_name=model_name, **kwargs)
        self.model_name = model_name
        self._engine = None

        self.max_solution_candidates = max_solution_candidates
        self.em4c_test_case_count = em4c_test_case_count
        self.allowed_error_ratio = allowed_error_ratio
        self.alpha_threshold = alpha_threshold
        self.sampling_temperature = sampling_temperature
        self.perturbation_range = perturbation_range
        self.fallback_to_base = fallback_to_base

        self.direct_solve_dir = Path(direct_solve_dir) if direct_solve_dir else None
        self.bruteforce_dir = Path(bruteforce_dir) if bruteforce_dir else None

        self.engine_kwargs = {
            k: v for k, v in kwargs.items()
            if k in ("temperature", "max_tokens", "top_p")
        }

    # ===================== public API =====================

    def solve(self, problem: Problem) -> Solution:
        try:
            return self._solve_inner(problem)
        except Exception as e:
            print(f"[GSM8KExeCRE] Error for {problem.question_id}: {e}")
            import traceback; traceback.print_exc()
            return Solution(
                code="",
                problem_id=problem.question_id,
                is_normal_end=False,
                metadata={"error": str(e)},
            )

    # ===================== core pipeline =====================

    def _solve_inner(self, problem: Problem) -> Solution:
        candidates = self._get_candidates(problem)
        if not candidates:
            if VERBOSE:
                print(f"[GSM8KExeCRE] {problem.question_id}: no candidates; fallback")
            return self._fallback_base(problem, alpha=0.0, meta={"reason": "no_candidates"})

        parsed_list = [parse_pm_output(c) for c in candidates]

        ref = next((p for p in parsed_list if p["param_names"] and p["original_values"]), None)
        if ref is None:
            if VERBOSE:
                print(f"[GSM8KExeCRE] {problem.question_id}: parse failed; fallback")
            return self._fallback_base(problem, alpha=0.0, meta={"reason": "parse_failed"})

        param_names = ref["param_names"]
        original_values = ref["original_values"]
        function_codes = [p["function_code"] or c for p, c in zip(parsed_list, candidates)]

        if VERBOSE:
            print(
                f"[GSM8KExeCRE] {problem.question_id}: "
                f"{len(candidates)} candidates, params={param_names}, original={original_values}"
            )

        valid_indices = []
        for i, fc in enumerate(function_codes):
            result = _exec_solve_with_args(fc, original_values)
            if result is not None:
                valid_indices.append(i)
        if not valid_indices:
            if VERBOSE:
                print(f"[GSM8KExeCRE] {problem.question_id}: none run on original inputs")
            return self._fallback_base(problem, alpha=0.0, meta={"reason": "all_exec_failed"})

        function_codes = [function_codes[i] for i in valid_indices]
        candidates = [candidates[i] for i in valid_indices]

        if VERBOSE:
            print(f"[GSM8KExeCRE] {problem.question_id}: {len(function_codes)} ok on original inputs")

        random_inputs = generate_random_inputs(
            original_values,
            count=self.em4c_test_case_count,
            perturbation_range=self.perturbation_range,
        )

        matrix, error_ratio = self._build_execution_matrix(
            function_codes, random_inputs
        )

        if VERBOSE:
            print(
                f"[GSM8KExeCRE] {problem.question_id}: "
                f"matrix {len(matrix)}×{len(function_codes)}, error_ratio={error_ratio:.3f}"
            )

        if error_ratio > self.allowed_error_ratio:
            if VERBOSE:
                print(
                    f"[GSM8KExeCRE] {problem.question_id}: "
                    f"error_ratio too high ({error_ratio:.3f} > {self.allowed_error_ratio})"
                )
            return self._fallback_base(
                problem, alpha=0.0,
                meta={"reason": "high_error_ratio", "error_ratio": error_ratio},
            )

        if not matrix:
            return self._fallback_base(
                problem, alpha=0.0, meta={"reason": "empty_matrix"},
            )

        label_processor = Majority01Processor()
        confidence_calculator = DawidSkeneCalculator()
        try:
            best_code, alpha, em4c_meta = step6_calculate_confidence(
                matrix=matrix,
                codes=function_codes,
                label_processor=label_processor,
                confidence_calculator=confidence_calculator,
            )
        except Exception as e:
            if VERBOSE:
                print(f"[GSM8KExeCRE] {problem.question_id}: EM4C failed: {e}")
            return self._fallback_base(
                problem, alpha=0.0, meta={"reason": "em4c_error", "error": str(e)},
            )

        best_idx = function_codes.index(best_code) if best_code in function_codes else 0

        if VERBOSE:
            print(f"[GSM8KExeCRE] {problem.question_id}: "
                  f"alpha={alpha:.4f}, best_idx={best_idx}")

        if alpha >= self.alpha_threshold:
            answer = _exec_solve_with_args(function_codes[best_idx], original_values)
            if answer is None:
                return self._fallback_base(
                    problem, alpha=alpha,
                    meta={"reason": "best_exec_failed", "best_idx": best_idx},
                )

            code_text = candidates[best_idx]
            output_text = f"{code_text}\n\n#### {answer}"

            if VERBOSE:
                print(f"[GSM8KExeCRE] {problem.question_id}: use PM alpha={alpha:.4f}, answer={answer}")

            return Solution(
                code=output_text,
                problem_id=problem.question_id,
                is_normal_end=True,
                metadata={
                    "alpha": alpha,
                    "best_idx": best_idx,
                    "fallback": False,
                    "executed_answer": answer,
                    "error_ratio": error_ratio,
                    "num_candidates": len(function_codes),
                    "num_random_inputs": len(matrix),
                    "param_names": param_names,
                    "original_values": original_values,
                },
            )
        else:
            if VERBOSE:
                print(f"[GSM8KExeCRE] {problem.question_id}: "
                      f"alpha={alpha:.4f} < {self.alpha_threshold}, fallback to Base")
            return self._fallback_base(
                problem, alpha=alpha,
                meta={
                    "reason": "low_alpha",
                    "best_idx": best_idx,
                    "error_ratio": error_ratio,
                    "num_candidates": len(function_codes),
                },
            )

    # ===================== helpers =====================

    def _get_engine(self):
        if self._engine is None:
            self._engine = get_engine(self.model_name, **self.engine_kwargs)
        return self._engine

    def _get_candidates(self, problem: Problem) -> List[str]:
        """Up to ``max_solution_candidates`` PM strings: bruteforce groups, seed CSV, then sampled."""
        codes: List[str] = []

        if self.bruteforce_dir is not None:
            codes = self._read_candidates_from_bruteforce(problem)

        if len(codes) < self.max_solution_candidates and self.direct_solve_dir is not None:
            seed = self._read_code_from_dir(self.direct_solve_dir, problem)
            if seed and seed not in codes:
                codes.append(seed)

        need = self.max_solution_candidates - len(codes)
        if need > 0:
            generated = self._generate_pm_candidates(problem, need)
            for g in generated:
                if g not in codes:
                    codes.append(g)

        return codes[:self.max_solution_candidates]

    def _generate_pm_candidates(self, problem: Problem, count: int) -> List[str]:
        """Sample ``count`` PM completions at ``sampling_temperature``."""
        engine = self._get_engine()
        user_prompt = GSM8KPMSolver._build_prompt(problem.question_content.strip())
        results: List[str] = []

        for _ in range(count):
            try:
                response = engine.generate(
                    user_prompt,
                    system_prompt=PM_SYSTEM_PROMPT,
                    temperature=self.sampling_temperature,
                    max_tokens=self.engine_kwargs.get("max_tokens", 2000),
                    top_p=self.engine_kwargs.get("top_p", 0.99),
                )
                code_text = _extract_code(response.strip())
                if code_text:
                    results.append(code_text)
            except Exception as e:
                if VERBOSE:
                    print(f"[GSM8KExeCRE] candidate gen failed: {e}")
        return results

    def _read_candidates_from_bruteforce(self, problem: Problem) -> List[str]:
        """Collect Solution_Code rows from ``group_*`` CSVs under bruteforce_dir."""
        if self.bruteforce_dir is None or not self.bruteforce_dir.exists():
            return []

        search_dir = self.bruteforce_dir
        if search_dir.name.startswith("group_"):
            search_dir = search_dir.parent

        codes: List[str] = []
        group_dirs: List[Tuple[int, Path]] = []

        for item in search_dir.iterdir():
            if item.is_dir():
                name = item.name
                try:
                    n = int(name.replace("group_", ""))
                    group_dirs.append((n, item))
                except ValueError:
                    continue
        group_dirs.sort()

        for _, gdir in group_dirs:
            csv_path = gdir / f"{problem.question_id}.csv"
            if not csv_path.exists():
                continue
            try:
                with open(csv_path, "r", encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        code = (row.get("Solution_Code") or "").strip()
                        is_ok = str(row.get("Is_Normal_End", "False")).lower() == "true"
                        if code and is_ok and code not in codes:
                            codes.append(code)
                            if len(codes) >= self.max_solution_candidates:
                                return codes
            except Exception:
                continue
        return codes

    @staticmethod
    def _read_code_from_dir(directory: Path, problem: Problem) -> Optional[str]:
        """First valid Solution_Code row from ``directory / {id}.csv``."""
        csv_path = directory / f"{problem.question_id}.csv"
        if not csv_path.exists():
            return None
        try:
            with open(csv_path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
                if row is None:
                    return None
                code = (row.get("Solution_Code") or "").strip()
                if code and str(row.get("Is_Normal_End", "False")).lower() == "true":
                    return code
        except Exception:
            pass
        return None

    def _build_execution_matrix(
        self,
        function_codes: List[str],
        random_inputs: List[List],
    ) -> Tuple[List[List[Any]], float]:
        """Return (matrix, fraction of None cells)."""
        matrix: List[List[Any]] = []
        total_exec = 0
        error_exec = 0

        for inputs in random_inputs:
            row: List[Any] = []
            for fc in function_codes:
                total_exec += 1
                result = _exec_solve_with_args(fc, inputs)
                if result is None:
                    error_exec += 1
                row.append(result)
            matrix.append(row)

        error_ratio = error_exec / total_exec if total_exec > 0 else 1.0
        return matrix, error_ratio

    def _fallback_base(
        self,
        problem: Problem,
        alpha: float = 0.0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Solution:
        """Plain LLM math answer when PM path is rejected or disabled."""
        if not self.fallback_to_base:
            return Solution(
                code="",
                problem_id=problem.question_id,
                is_normal_end=False,
                metadata={"alpha": alpha, "fallback": True, **(meta or {})},
            )

        try:
            engine = self._get_engine()
            user_prompt = f"Question: {problem.question_content.strip()}"
            response = engine.generate(
                user_prompt,
                system_prompt=GSM8K_SYSTEM_PROMPT,
                **self.engine_kwargs,
            )
            base_sol = Solution(
                code=response.strip(),
                problem_id=problem.question_id,
                is_normal_end=bool(response and response.strip()),
            )
        except Exception as e:
            print(
                f"Error in GSM8KExeCRESolver._fallback_base() for "
                f"{problem.question_id}: {e}"
            )
            base_sol = Solution(
                code="",
                problem_id=problem.question_id,
                is_normal_end=False,
                metadata={"error": str(e)},
            )
        base_sol.metadata = {
            **(base_sol.metadata or {}),
            "alpha": alpha,
            "fallback": True,
            **(meta or {}),
        }
        return base_sol
