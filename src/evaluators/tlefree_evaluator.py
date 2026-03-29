"""TLE-free correctness: TLE cases are skipped; WA/runtime errors still fail."""

import json
from typing import Dict, Any, List, Optional

from .lcb_coderunner import run_code_capture
from ..core.solution import Solution
from ..core.problem import Problem


def tlefree_evaluate_simulation_code(
    code: str,
    test_cases: List[Dict[str, Any]],
    fn_name: Optional[str] = None,
    timeout: float = 6
) -> bool:
    """Run all cases via `run_code_capture`; ignore TLE; require match on non-TLE runs."""
    for i, test_case in enumerate(test_cases):
        input_data = test_case.get('input', '')
        expected_output = test_case.get('output', '')

        success, output, error = run_code_capture(
            fn_name=fn_name,
            test_case=input_data,
            code=code,
            timeout=timeout,
            test_index=i
        )

        if not success and error == "Time Limit Exceeded":
            continue
        elif not success:
            return False

        if fn_name:
            try:
                expected_obj = json.loads(expected_output)
                if output != expected_obj:
                    return False
            except (json.JSONDecodeError, TypeError):
                if str(output).strip() != expected_output.strip():
                    return False
        else:
            if output.strip() != expected_output.strip():
                return False

    return True


def tlefree_evaluate(
    solution: Solution,
    problem: Problem,
    timeout: float = 0.8
) -> bool:
    """Evaluate `solution` on `problem.public_test_cases` with TLE ignored."""
    code = solution.code
    test_cases = problem.public_test_cases
    fn_name = problem.metadata.get('func_name', None)

    return tlefree_evaluate_simulation_code(
        code=code,
        test_cases=test_cases,
        fn_name=fn_name,
        timeout=timeout
    )
