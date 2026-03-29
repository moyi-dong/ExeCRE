"""Sequential LCB evaluation; reloads `os` after each run (reliability_guard patches it)."""

import os
import importlib
import signal
import resource
from typing import List, Dict, Any, Tuple, Callable, Optional
from tqdm import tqdm

from .lcb_evaluator import lcb_evaluate_single_generation


def _restore_os_module():
    importlib.reload(os)


def safe_single_process_evaluation(
    samples_list: List[Dict[str, Any]],
    generations_list: List[List[str]],
    timeout: int,
    problem_ids: List[str] = None,
    on_problem_complete: Optional[Callable[[int, List, List[Dict[str, Any]]], None]] = None
) -> Tuple[Dict[int, List], Dict[int, List[Dict[str, Any]]]]:
    """Evaluate each generation with `lcb_evaluate_single_generation`; restores `os` after each call."""
    all_results = {}
    all_metadatas = {}

    print(f"Starting single-process evaluation for {len(samples_list)} problem(s)...")

    total_evaluations = sum(len(gens) for gens in generations_list)

    with tqdm(total=total_evaluations, desc="Single-process eval", unit="run") as pbar:
        for i, (sample, generations) in enumerate(zip(samples_list, generations_list)):
            pid = problem_ids[i] if problem_ids else f"#{i}"
            try:
                problem_results = []
                problem_metadatas = []

                for round_idx, code in enumerate(generations):
                    try:
                        if not code or not code.strip():
                            problem_results.append([])
                            problem_metadatas.append({"error": "Empty code"})
                            pbar.update(1)
                            continue

                        result, metadata = lcb_evaluate_single_generation(
                            solution_code=code,
                            test_cases=sample,
                            timeout=timeout,
                            debug=False
                        )
                        problem_results.append(result)
                        problem_metadatas.append(metadata)
                        _restore_os_module()
                        pbar.update(1)
                        pbar.set_description(
                            f"Single-process [{pid}] ({i+1}/{len(samples_list)}) r{round_idx+1}"
                        )

                    except Exception as e:
                        print(f"Warning: problem {pid} round {round_idx+1} failed: {e}")
                        problem_results.append([])
                        problem_metadatas.append({"error": str(e)})
                        pbar.update(1)

                all_results[i] = problem_results
                all_metadatas[i] = problem_metadatas
                if on_problem_complete is not None:
                    on_problem_complete(i, problem_results, problem_metadatas)

            except Exception as e:
                print(f"Error: problem {pid} failed entirely: {e}")
                all_results[i] = []
                all_metadatas[i] = []
                if on_problem_complete is not None:
                    on_problem_complete(i, all_results[i], all_metadatas[i])
                pbar.update(len(generations))

    print("Single-process evaluation finished.")
    return all_results, all_metadatas


def safe_single_process_evaluation_simple(
    samples_list: List[Dict[str, Any]],
    generations_list: List[List[str]],
    timeout: int,
    problem_ids: List[str] = None,
    on_problem_complete: Optional[Callable[[int, List, List[Dict[str, Any]]], None]] = None
) -> Tuple[Dict[int, List], Dict[int, List[Dict[str, Any]]]]:
    """Like `safe_single_process_evaluation` but uses SIGALRM + `resource` limits (Unix; not Windows)."""
    all_results = {}
    all_metadatas = {}

    print(f"Starting simple single-process evaluation for {len(samples_list)} problem(s)...")

    def set_resource_limits():
        try:
            resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_STACK, (8 * 1024 * 1024, 8 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_CPU, (timeout + 5, timeout + 5))
        except Exception as e:
            print(f"Warning: could not set resource limits: {e}")

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Evaluation timed out ({timeout}s)")

    total_evaluations = sum(len(gens) for gens in generations_list)

    with tqdm(total=total_evaluations, desc="Simple single-process", unit="run") as pbar:
        for i, (sample, generations) in enumerate(zip(samples_list, generations_list)):
            pid = problem_ids[i] if problem_ids else f"#{i}"
            try:
                problem_results = []
                problem_metadatas = []

                for round_idx, code in enumerate(generations):
                    try:
                        if not code or not code.strip():
                            problem_results.append([])
                            problem_metadatas.append({"error": "Empty code"})
                            pbar.update(1)
                            continue

                        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                        old_limits = {}

                        try:
                            old_limits['as'] = resource.getrlimit(resource.RLIMIT_AS)
                            old_limits['stack'] = resource.getrlimit(resource.RLIMIT_STACK)
                            old_limits['cpu'] = resource.getrlimit(resource.RLIMIT_CPU)

                            set_resource_limits()
                            signal.alarm(timeout)

                            result, metadata = lcb_evaluate_single_generation(
                                solution_code=code,
                                test_cases=sample,
                                timeout=timeout,
                                debug=False
                            )

                            problem_results.append(result)
                            problem_metadatas.append(metadata)

                        except TimeoutError as e:
                            problem_results.append([])
                            problem_metadatas.append({"error": str(e)})
                        except Exception as e:
                            problem_results.append([])
                            problem_metadatas.append({"error": f"Runtime error: {e}"})
                        finally:
                            signal.signal(signal.SIGALRM, old_handler)
                            signal.alarm(0)

                            try:
                                resource.setrlimit(resource.RLIMIT_AS, old_limits['as'])
                                resource.setrlimit(resource.RLIMIT_STACK, old_limits['stack'])
                                resource.setrlimit(resource.RLIMIT_CPU, old_limits['cpu'])
                            except Exception:
                                pass

                            _restore_os_module()

                        pbar.update(1)
                        pbar.set_description(
                            f"Simple SP [{pid}] ({i+1}/{len(samples_list)}) r{round_idx+1}"
                        )

                    except Exception as e:
                        print(f"Warning: problem {pid} round {round_idx+1} failed: {e}")
                        problem_results.append([])
                        problem_metadatas.append({"error": str(e)})
                        pbar.update(1)

                all_results[i] = problem_results
                all_metadatas[i] = problem_metadatas
                if on_problem_complete is not None:
                    on_problem_complete(i, problem_results, problem_metadatas)

            except Exception as e:
                print(f"Error: problem {pid} failed entirely: {e}")
                all_results[i] = []
                all_metadatas[i] = []
                if on_problem_complete is not None:
                    on_problem_complete(i, all_results[i], all_metadatas[i])
                pbar.update(len(generations))

    print("Simple single-process evaluation finished.")
    return all_results, all_metadatas
