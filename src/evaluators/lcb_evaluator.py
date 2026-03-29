"""LiveCodeBench-backed evaluation; reload `os` after `lcb_evaluate_single_generation` (reliability_guard)."""

import os
import importlib
import json
import sys
from typing import List, Dict, Any, Tuple
from pathlib import Path

from ..utils.path_manager import get_src_dir


def _restore_os_module():
    importlib.reload(os)


def _setup_lcb_path():
    src_dir = get_src_dir()
    lcb_runner_path = src_dir / "benchmark_repo" / "LiveCodeBench" / "lcb_runner"
    lcb_runner_path_str = str(lcb_runner_path)

    if lcb_runner_path_str not in sys.path:
        sys.path.insert(0, lcb_runner_path_str)

    return lcb_runner_path


_lcb_path = _setup_lcb_path()

try:
    from lcb_runner.evaluation.testing_util import run_test
    from lcb_runner.evaluation.compute_code_generation_metrics import evaluate_generations
except ImportError as e:
    raise ImportError(
        f"Failed to import LiveCodeBench evaluation modules. "
        f"Make sure LiveCodeBench is properly installed at {_lcb_path}. "
        f"Original error: {e}"
    )

# Lower import_string recursion limit (50000 -> 1000) to avoid C stack overflow when exec() in workers.
try:
    from lcb_runner.evaluation import testing_util as _testing_util_module
    if hasattr(_testing_util_module, 'import_string') and 'setrecursionlimit(50000)' in _testing_util_module.import_string:
        _testing_util_module.import_string = _testing_util_module.import_string.replace(
            "sys.setrecursionlimit(50000)", "sys.setrecursionlimit(1000)"
        )
except Exception:
    pass


def lcb_evaluate_single_generation(
    solution_code: str,
    test_cases: Dict[str, Any],
    timeout: int = 6,
    debug: bool = False
) -> Tuple[List, Dict[str, Any]]:
    """Run LCB `run_test` for one solution; restores `os` in `finally`."""
    try:
        return run_test(
            sample=test_cases,
            test=solution_code,
            debug=debug,
            timeout=timeout
        )
    finally:
        _restore_os_module()


def lcb_evaluate_generations(
    samples_list: List[Dict[str, Any]],
    generations_list: List[List[str]],
    debug: bool = False,
    num_process_evaluate: int = 12,
    timeout: int = 6
) -> Tuple[Dict, Dict]:
    """Batch evaluation via multiprocess `evaluate_generations`."""
    return evaluate_generations(
        samples_list=samples_list,
        generations_list=generations_list,
        debug=debug,
        num_process_evaluate=num_process_evaluate,
        timeout=timeout
    )
