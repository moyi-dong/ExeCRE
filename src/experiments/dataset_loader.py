"""Load and filter benchmark datasets for experiments."""

from typing import List
from src.config import ExperimentConfig
from src.core.benchmark_loader import (
    load_benchmark,
    create_livecodebench_config,
    create_gsm8k_config,
)
from src.core.problem import Problem


def initialize_and_load_dataset(config: ExperimentConfig) -> List[Problem]:
    """
    Load benchmark problems from config and apply filters.

    Filters: specific_question_id, specific_question_ids, test_count, difficulty
    (difficulty applies to benchmarks that expose it, e.g. LiveCodeBench).

    LiveCodeBench memory: default streaming load (not_fast=False); set
    not_fast=True to load the full dataset (higher memory).

    Returns an empty list on load failure or if filters match nothing.
    """
    print(f"Model: {config.model.model}")
    print(f"Experiment ID: {config.experiment.experiment_id}")
    print(f"Benchmark: {config.experiment.benchmark}")

    if config.experiment.benchmark == "GSM8K":
        print("\nLoading GSM8K...")
        benchmark_config = create_gsm8k_config(split="test")
        dataset = load_benchmark(benchmark_config)
    else:
        print("\nLoading dataset...")
        extra_params = {}
        if config.experiment.not_fast:
            extra_params["not_fast"] = True
            print("Warning: full dataset mode (not_fast=True), higher memory use")
        else:
            print("Streaming load (default), lower memory use")

        benchmark_config = create_livecodebench_config(
            release_version=config.experiment.release_version,
            start_date=config.experiment.start_date,
            end_date=config.experiment.end_date,
            **extra_params
        )
        dataset = load_benchmark(benchmark_config)
    print(f"Loaded {len(dataset)} problem(s)")

    if config.experiment.specific_question_id is not None:
        target_problem = None
        for problem in dataset:
            if problem.question_id == config.experiment.specific_question_id:
                target_problem = problem
                break

        if target_problem is None:
            print(f"Error: no problem with id '{config.experiment.specific_question_id}'")
            print("Sample question IDs:")
            for problem in dataset[:10]:
                print(f"  {problem.question_id}")
            if len(dataset) > 10:
                print(f"  ... and {len(dataset) - 10} more")
            return []

        print(f"Found: {target_problem.question_id} - {target_problem.question_title}")
        test_dataset = [target_problem]
        print("Test set: 1 selected problem")
    elif config.experiment.specific_question_ids is not None:
        target_problems = []
        not_found_ids = []

        for question_id in config.experiment.specific_question_ids:
            found = False
            for problem in dataset:
                if problem.question_id == question_id:
                    target_problems.append(problem)
                    found = True
                    break
            if not found:
                not_found_ids.append(question_id)

        if not_found_ids:
            print(f"Error: question IDs not found: {not_found_ids}")
            print("Sample question IDs:")
            for problem in dataset[:10]:
                print(f"  {problem.question_id}")
            if len(dataset) > 10:
                print(f"  ... and {len(dataset) - 10} more")
            return []

        if not target_problems:
            print("Error: no matching problems for the given IDs")
            return []

        print(f"Found {len(target_problems)} problem(s):")
        for problem in target_problems:
            print(f"  - {problem.question_id}: {problem.question_title}")

        test_dataset = target_problems
        print(f"Test set: {len(test_dataset)} selected problem(s)")
    else:
        if config.experiment.test_count is not None:
            test_count = min(config.experiment.test_count, len(dataset))
            print(f"Test set: first {test_count} (requested: {config.experiment.test_count})")
        else:
            test_count = len(dataset)
            print(f"Test set: all {test_count} problem(s)")

        test_dataset = dataset[:test_count]

    if config.experiment.difficulty != "all":
        print(f"\nFilter by difficulty: {config.experiment.difficulty}")
        original_count = len(test_dataset)
        test_dataset = [problem for problem in test_dataset if problem.difficulty == config.experiment.difficulty]
        filtered_count = len(test_dataset)
        print(f"After difficulty filter: {original_count} -> {filtered_count}")

        if filtered_count == 0:
            print(f"Error: no problems with difficulty '{config.experiment.difficulty}'")
            print("Difficulty counts in full dataset:")
            difficulty_counts = {}
            for problem in dataset:
                difficulty_counts[problem.difficulty] = difficulty_counts.get(problem.difficulty, 0) + 1
            for difficulty, count in difficulty_counts.items():
                print(f"  {difficulty}: {count}")
            return []

    return test_dataset
