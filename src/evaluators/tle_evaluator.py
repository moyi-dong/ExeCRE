"""TLE check: random schema-generated inputs; fails on first timeout or hang."""

import time
import signal
from typing import Optional, Tuple

from src.input_generators.schema.schema import generate_data_from_schema

from .lcb_coderunner import run_code_capture
from ..config.experiment_config import EvaluatorConfig


def timeout_handler(signum, frame):
    raise TimeoutError("Data generation timeout")


def tle_evaluate(
    code: str,
    schema: str,
    problem_metadata: Optional[dict] = None,
    config: Optional[EvaluatorConfig] = None
) -> Tuple[bool, str]:
    """Return (passed, feedback). Passes only if no TLE/hang over sampled cases; ignores correctness."""
    if config is None:
        config = EvaluatorConfig()

    total_start_time = time.time()
    longest_case_length = 0
    for i in range(config.test_case_count):
        if i % (config.test_case_count // 10 + 1) == 0:
            print(f"time: {time.time() - total_start_time:.2f}/{config.total_generation_timeout}")
            print(f"t_t [{i}/{config.test_case_count}]")
        if time.time() - total_start_time > config.total_generation_timeout:
            break

        try:
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
            test_case = generate_data_from_schema(schema)
            longest_case_length = max(longest_case_length, len(str(test_case)))
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        except TimeoutError:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            raise TimeoutError(f"Data generation timeout after 10s for test case {i+1}")
        except Exception as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            raise RuntimeError(f"Data generation error for test case {i+1}: {str(e)}")

        fn_name = None
        if problem_metadata:
            if isinstance(problem_metadata, dict):
                fn_name = problem_metadata.get('func_name', None)
            else:
                fn_name = getattr(problem_metadata, 'metadata', {}).get('func_name', None) if hasattr(problem_metadata, 'metadata') else None

        try:
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(config.single_test_timeout) + 1)
            passed, output, errormsg = run_code_capture(
                fn_name,
                test_case,
                code,
                timeout=config.single_test_timeout
            )
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
        except TimeoutError:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            total_execution_time = time.time() - total_start_time
            feedback = f"**TLE Test Result:**\n\n"
            feedback += f"Status: FAILED (Code execution stuck/hung)\n"
            feedback += f"Failed at Test(length={len(str(test_case))}): {i+1}\n"
            feedback += f"Test Case Size: {len(str(test_case))} chars\n"
            feedback += f"Single Test Timeout: {config.single_test_timeout}s\n"
            feedback += f"Total Time Used: {total_execution_time:.3f}s\n"
            feedback += f"Total Timeout Limit: {config.total_generation_timeout}s"
            return False, feedback
        except Exception as e:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            raise RuntimeError(f"Code execution error for test case {i+1}: {str(e)}")

        if errormsg and "Time Limit Exceeded" in str(errormsg):
            total_execution_time = time.time() - total_start_time
            feedback = f"**TLE Test Result:**\n\n"
            feedback += f"Status: FAILED (Time Limit Exceeded)\n"
            feedback += f"Failed at Test(length={len(str(test_case))}): {i+1}\n"
            feedback += f"Test Case Size: {len(str(test_case))} chars\n"
            feedback += f"Single Test Timeout: {config.single_test_timeout}s\n"
            feedback += f"Total Time Used: {total_execution_time:.3f}s\n"
            feedback += f"Total Timeout Limit: {config.total_generation_timeout}s"
            return False, feedback

    total_execution_time = time.time() - total_start_time
    feedback = f"**TLE Test Result:**\n\n"
    feedback += f"Status: PASSED (No Timeout)\n"
    feedback += f"Total Tests Completed(longest test length={longest_case_length}): {config.test_case_count}\n"
    feedback += f"Total Time Used: {total_execution_time:.3f}s\n"
    feedback += f"Total Timeout Limit: {config.total_generation_timeout}s\n"
    feedback += f"Single Test Timeout: {config.single_test_timeout}s"

    return True, feedback
