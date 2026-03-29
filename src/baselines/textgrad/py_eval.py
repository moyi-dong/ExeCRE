"""Wrap LCB evaluation in TextGrad-style pass/fail + feedback strings."""

import json
from typing import Any, Dict, List, Optional, Tuple

from src.evaluators.lcb_evaluator import lcb_evaluate_single_generation


def evaluate(
    code: str,
    tests: List[Dict[str, Any]],
    problem_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Run ``code`` on ``tests`` via LCB; return (all_passed, markdown feedback)."""
    inputs = [test["input"] for test in tests]
    outputs = [test["output"] for test in tests]

    input_output_dict: Dict[str, Any] = {"inputs": inputs, "outputs": outputs}

    if problem_metadata and "func_name" in problem_metadata:
        input_output_dict["fn_name"] = problem_metadata["func_name"]

    test_cases = {"input_output": json.dumps(input_output_dict)}
    results, metadata = lcb_evaluate_single_generation(code, test_cases)

    success_tests: List[Dict[str, Any]] = []
    failed_tests: List[Dict[str, Any]] = []
    failed_errors: List[str] = []

    for i, result in enumerate(results):
        if result is True:
            success_tests.append(tests[i])
        else:
            failed_tests.append(tests[i])

            if isinstance(result, int) and result < 0:
                if result == -2:
                    error_msg = (
                        f"Wrong answer. Output was {metadata['output']}, "
                        f"but expected value was: {metadata['expected']}"
                    )
                else:
                    error_msg = f"{metadata['error_message']}."
            else:
                error_msg = f"Test failed with unexpected result: {metadata['error_message']}"

            failed_errors.append(error_msg)

    feedback = "**Tests that the code passed:**\n"
    if len(success_tests) == 0:
        feedback += "\nNo tests passed.\n"
    else:
        for test in success_tests:
            feedback += f"\nInput:{test['input']} Expected output:{test['output']}"
    feedback += "\n\n**Tests that the code failed:**\n"
    if len(failed_tests) == 0:
        feedback += "\nNo tests failed.\n"
    else:
        for i, test in enumerate(failed_tests):
            feedback += (
                f"\nInput:{test['input']} Expected output:{test['output']} "
                f"# ERROR: {failed_errors[i]}"
            )

    passed = all(r is True for r in results)
    return passed, feedback
