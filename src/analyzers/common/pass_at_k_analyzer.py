"""Compute pass@k from CSV group results (LiveCodeBench-style metrics)."""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

from src.config.experiment_config import ExperimentConfig
from src.utils.path_manager import get_group_dir
from src.benchmark_repo.LiveCodeBench.lcb_runner.evaluation.pass_k_utils import (
    compute_metrics_from_results
)
from src.core.problem import Problem


def _compute_group_pass_rates(
    results: Dict[int, Dict[str, bool]]
) -> List[float]:
    """Per-group pass rate; `results` is {group_n: {question_id: passed}}."""
    group_rates = []
    for group_n in sorted(results.keys()):
        group_results = results[group_n]
        if group_results:
            rate = sum(1 for v in group_results.values() if v) / len(group_results)
            group_rates.append(rate)
    return group_rates


def _enrich_metrics_with_group_stats(
    metrics: Dict[str, Any],
    results: Dict[int, Dict[str, bool]]
) -> None:
    """Attach n_groups and group_std to metrics (in place)."""
    group_rates = _compute_group_pass_rates(results)
    if group_rates:
        metrics["n_groups"] = len(group_rates)
        metrics["group_std"] = float(np.std(group_rates))


def analyze_pass_at_k(
    config: ExperimentConfig,
    k_list: List[int] = [1, 5, 10],
    output_file: Optional[Path] = None,
    test_dataset: Optional[List] = None,
    print_results: bool = True
) -> Dict[str, Any]:
    """Run pass@k for `config`; optional `test_dataset` limits question IDs."""
    experiment_id = config.experiment.experiment_id
    benchmark = config.experiment.benchmark
    model = config.model.model
    baseline = config.experiment.baseline
    config_hash = config.get_config_hash()
    n = config.experiment.n

    question_ids = None
    if test_dataset is not None:
        question_ids = {problem.question_id for problem in test_dataset}
    else:
        from src.experiments import initialize_and_load_dataset
        loaded_dataset = initialize_and_load_dataset(config)
        if loaded_dataset:
            question_ids = {problem.question_id for problem in loaded_dataset}

    results = load_results_from_csv(
        experiment_id=experiment_id,
        benchmark=benchmark,
        model=model,
        baseline=baseline,
        config_hash=config_hash,
        n=n,
        question_ids=question_ids
    )

    if not results:
        return {}

    lcb_results = convert_to_lcb_format(results)

    if not lcb_results:
        return {}

    metrics = compute_metrics_from_results(lcb_results, k_list=k_list)
    _enrich_metrics_with_group_stats(metrics, results)

    if print_results:
        print_pass_at_k_results(metrics, config)

    if output_file:
        save_pass_at_k_results(metrics, output_file, config)

    return metrics


def analyze_pass_at_k_from_config(
    config_path: Path,
    k_list: List[int] = [1, 5, 10],
    output_file: Optional[Path] = None
) -> Dict[str, Any]:
    """Load config from path and run analyze_pass_at_k."""
    config = ExperimentConfig.from_file(config_path)
    return analyze_pass_at_k(config, k_list=k_list, output_file=output_file)


def analyze_pass_at_k_by_difficulty(
    config: ExperimentConfig,
    k_list: List[int] = [1, 5, 10],
    test_dataset: Optional[List[Problem]] = None
) -> Dict[str, Any]:
    """Return {"overall": {...}, "by_difficulty": {diff: {problem_count, pass@k...}}}."""
    if test_dataset is None:
        from src.experiments import initialize_and_load_dataset
        test_dataset = initialize_and_load_dataset(config)
        if not test_dataset:
            return {
                "overall": {},
                "by_difficulty": {}
            }

    difficulty_groups = defaultdict(list)
    for problem in test_dataset:
        difficulty = problem.difficulty
        difficulty_groups[difficulty].append(problem)

    overall_metrics = analyze_pass_at_k(
        config=config,
        k_list=k_list,
        test_dataset=test_dataset,
        print_results=False
    )

    overall_pass_at_k = {}
    for key, value in overall_metrics.items():
        if key != "detail" and isinstance(value, (int, float)):
            overall_pass_at_k[key] = value

    by_difficulty = {}

    experiment_id = config.experiment.experiment_id
    benchmark = config.experiment.benchmark
    model = config.model.model
    baseline = config.experiment.baseline
    config_hash = config.get_config_hash()
    n = config.experiment.n

    for difficulty, problems in difficulty_groups.items():
        question_ids = {problem.question_id for problem in problems}

        results = load_results_from_csv(
            experiment_id=experiment_id,
            benchmark=benchmark,
            model=model,
            baseline=baseline,
            config_hash=config_hash,
            n=n,
            question_ids=question_ids
        )

        if not results:
            by_difficulty[difficulty] = {
                "problem_count": len(problems),
                **{f"pass@{k}": 0.0 for k in k_list}
            }
            continue

        lcb_results = convert_to_lcb_format(results)

        if not lcb_results:
            by_difficulty[difficulty] = {
                "problem_count": len(problems),
                **{f"pass@{k}": 0.0 for k in k_list}
            }
            continue

        metrics = compute_metrics_from_results(lcb_results, k_list=k_list)
        _enrich_metrics_with_group_stats(metrics, results)

        difficulty_pass_at_k = {
            "problem_count": len(problems)
        }
        for key, value in metrics.items():
            if key != "detail" and isinstance(value, (int, float)):
                difficulty_pass_at_k[key] = value

        by_difficulty[difficulty] = difficulty_pass_at_k

    return {
        "overall": overall_pass_at_k,
        "by_difficulty": by_difficulty
    }


def load_results_from_csv(
    experiment_id: str,
    benchmark: str,
    model: str,
    baseline: str,
    config_hash: str,
    n: int,
    question_ids: Optional[Set[str]] = None
) -> Dict[int, Dict[str, bool]]:
    """Load per-group pass/fail from CSVs under each group dir -> {group_n: {qid: passed}}."""
    results = {}
    is_trusttest = baseline.lower() in ["trusttest", "trusttestem", "execre"]

    if is_trusttest:
        from src.analyzers.e1_pass_at_k.round_accuracy_analyzer import (
            select_best_round_for_trusttest
        )

    for group_n in range(1, n + 1):
        group_dir = get_group_dir(
            experiment_id=experiment_id,
            benchmark=benchmark,
            model=model,
            baseline=baseline,
            config_hash=config_hash,
            group_n=group_n
        )

        if not group_dir.exists():
            continue

        group_results = {}

        csv_files = list(group_dir.glob("*.csv"))

        if not csv_files:
            continue

        for csv_file in csv_files:
            question_id = csv_file.stem

            if question_ids is not None and question_id not in question_ids:
                continue

            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                    if not rows:
                        continue

                    if is_trusttest:
                        _, best_passed = select_best_round_for_trusttest(rows, max_round=None)
                        if best_passed is not None:
                            group_results[question_id] = best_passed
                        else:
                            last_row = rows[-1]
                            passed_str = last_row.get('Passed', '').upper()
                            group_results[question_id] = (passed_str == 'TRUE')
                    else:
                        # Row pick: Is_Final (last first), else last Is_Normal_End, else last row
                        selected_row = None
                        for row in reversed(rows):
                            is_final = row.get('Is_Final', '').upper()
                            if is_final == 'TRUE' or is_final == 'True':
                                selected_row = row
                                break
                        if selected_row is None:
                            for row in reversed(rows):
                                is_normal_end = row.get('Is_Normal_End', '').upper()
                                if is_normal_end == 'TRUE' or is_normal_end == 'True':
                                    selected_row = row
                                    break
                        if selected_row is None:
                            selected_row = rows[-1]

                        passed_str = selected_row.get('Passed', '').upper()
                        passed = (passed_str == 'TRUE')

                        group_results[question_id] = passed

            except Exception as e:
                print(f"    Failed to read CSV {csv_file.name}: {e}")
                continue

        if group_results:
            results[group_n] = group_results

    return results


def convert_to_lcb_format(
    results: Dict[int, Dict[str, bool]]
) -> Dict[str, List[List[int]]]:
    """Map {group_n: {qid: passed}} to LCB {qid: [[1] or [-1], ...]} per generation."""
    lcb_results = defaultdict(list)

    all_question_ids = set()
    for group_results in results.values():
        all_question_ids.update(group_results.keys())

    for question_id in sorted(all_question_ids):
        generations = []

        for group_n in sorted(results.keys()):
            group_results = results[group_n]

            if question_id not in group_results:
                continue

            passed = group_results[question_id]

            if passed:
                generation = [1]
            else:
                generation = [-1]

            generations.append(generation)

        if generations:
            lcb_results[question_id] = generations

    return dict(lcb_results)


def print_pass_at_k_results(
    metrics: Dict[str, Any],
    config: ExperimentConfig
) -> None:
    """Print pass@k lines plus any other numeric metrics."""
    def extract_k_value(key: str) -> int:
        if key.startswith("pass@"):
            try:
                return int(key.split("@")[1])
            except (ValueError, IndexError):
                return 0
        return 0

    n_groups = metrics.get("n_groups")
    group_std = metrics.get("group_std")

    pass_at_k_items = []
    other_items = []
    group_stat_keys = {"n_groups", "group_std"}

    for key, value in metrics.items():
        if key == "detail" or key in group_stat_keys:
            continue
        if isinstance(value, (int, float)):
            if key.startswith("pass@"):
                pass_at_k_items.append((key, value))
            else:
                other_items.append((key, value))

    pass_at_k_items.sort(key=lambda x: extract_k_value(x[0]))

    for key, value in pass_at_k_items:
        suffix = ""
        if n_groups is not None and group_std is not None:
            suffix = f" (n_groups={n_groups}, group_std={group_std:.4f})"
        print(f"{key}: {value:.4f}{suffix}")

    for key, value in sorted(other_items):
        print(f"{key}: {value:.4f}")


def print_pass_at_k_by_difficulty_results(
    results: Dict[str, Any]
) -> None:
    """Pretty-print output of analyze_pass_at_k_by_difficulty."""
    def extract_k_value(key: str) -> int:
        if key.startswith("pass@"):
            try:
                return int(key.split("@")[1])
            except (ValueError, IndexError):
                return 0
        return 0

    def sort_pass_at_k_items(items: List[tuple]) -> List[tuple]:
        return sorted(items, key=lambda x: extract_k_value(x[0]))

    group_stat_keys = {"n_groups", "group_std", "problem_count"}

    def _format_pass_at_k_line(key: str, value: float, data: Dict) -> str:
        n_groups = data.get("n_groups")
        group_std = data.get("group_std")
        suffix = ""
        if n_groups is not None and group_std is not None:
            suffix = f" (n_groups={n_groups}, group_std={group_std:.4f})"
        return f"  {key}: {value:.4f}{suffix}"

    print("\n" + "=" * 40)
    print("Overall pass@k:")
    print("-" * 40)

    overall = results.get("overall", {})
    if overall:
        overall_items = [
            (k, v) for k, v in overall.items()
            if k not in group_stat_keys and isinstance(v, (int, float)) and k.startswith("pass@")
        ]
        sorted_overall = sort_pass_at_k_items(overall_items)
        for key, value in sorted_overall:
            print(_format_pass_at_k_line(key, value, overall))
    else:
        print("  (no data)")

    by_difficulty = results.get("by_difficulty", {})
    if by_difficulty:
        print("\n" + "=" * 40)
        print("Pass@k by difficulty:")
        print("-" * 40)

        sorted_difficulties = sorted(by_difficulty.items())

        for difficulty, metrics in sorted_difficulties:
            problem_count = metrics.get("problem_count", 0)
            print(f"\n{difficulty} ({problem_count} problems)")

            pass_at_k_items = [
                (k, v) for k, v in metrics.items()
                if k not in group_stat_keys and isinstance(v, (int, float)) and k.startswith("pass@")
            ]
            sorted_items = sort_pass_at_k_items(pass_at_k_items)

            for key, value in sorted_items:
                print(_format_pass_at_k_line(key, value, metrics))
    else:
        print("\nBy difficulty: (no data)")

    print("=" * 40)


def save_pass_at_k_results(
    metrics: Dict[str, Any],
    output_file: Path,
    config: ExperimentConfig
) -> None:
    """Write JSON (+ sidecar CSV summary without detail)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "config": {
            "experiment_id": config.experiment.experiment_id,
            "benchmark": config.experiment.benchmark,
            "model": config.model.model,
            "baseline": config.experiment.baseline,
            "config_hash": config.get_config_hash(),
            "n": config.experiment.n,
        },
        "metrics": metrics
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved: {output_file}")

    csv_file = output_file.with_suffix('.csv')
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in metrics.items():
            if key == "detail":
                continue
            if isinstance(value, (int, float)):
                writer.writerow([key, f"{value:.6f}"])
            else:
                writer.writerow([key, str(value)])

    print(f"Saved CSV: {csv_file}")
