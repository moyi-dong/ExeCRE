"""
Re-evaluate e1 solution CSVs with the TLEfree evaluator and write e3 CSVs
(same columns as e1 plus TLEfree_Passed). Missing/empty e1 CSVs are counted in the final report.
"""

import csv
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.config import ExperimentConfig
from src.utils.parser import get_config_from_args
from src.utils.config_printer import print_config_summary
from src.utils.path_manager import get_group_dir, get_results_dir
from src.experiments import initialize_and_load_dataset
from src.core.problem import Problem
from src.evaluators.tlefree_evaluator import tlefree_evaluate_simulation_code


@dataclass
class EvaluationStats:
    """Aggregated counts for one run or one group."""
    total_problems: int = 0
    processed_problems: int = 0
    missing_csv_count: int = 0
    empty_csv_count: int = 0
    evaluation_errors: int = 0
    missing_csv_ids: List[str] = field(default_factory=list)
    empty_csv_ids: List[str] = field(default_factory=list)
    error_ids: List[str] = field(default_factory=list)


def log_run(log_path: Path, message: str) -> None:
    """Append a timestamped line to run_log.txt."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def check_existing_tlefree_evaluation(problem: Problem, e3_save_dir: Path, force_reevaluate: bool = False) -> bool:
    """Return True if e3 CSV exists and every row has TLEfree_Passed set (resume skip)."""
    if force_reevaluate:
        return False

    csv_path = e3_save_dir / f"{problem.question_id}.csv"

    if not csv_path.exists():
        return False

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

            if not rows:
                return False

            for row in rows:
                tlefree_passed = row.get('TLEfree_Passed', '')
                if not tlefree_passed:
                    return False

            return True

    except Exception as e:
        print(f"⚠️ Failed to read evaluation CSV {csv_path}: {e}")
        return False


def format_id_list(ids: List[str], max_show: int = 5) -> str:
    """Join ids for display; truncate with count if longer than max_show."""
    if not ids:
        return ""
    if len(ids) <= max_show:
        return ", ".join(ids)
    else:
        return ", ".join(ids[:max_show]) + f", ... ({len(ids)} total)"


def process_single_csv(
    e1_csv_path: Path,
    e3_csv_path: Path,
    problem: Problem,
    timeout: float = 0.8,
    group_n: int = 0
) -> Tuple[bool, str]:
    """Read e1 CSV, add TLEfree_Passed per row, write e3 CSV."""
    if not e1_csv_path.exists():
        return False, "csv_not_found"

    try:
        with open(e1_csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        if not rows:
            return False, "csv_empty"

        test_cases = problem.public_test_cases + problem.private_test_cases
        fn_name = problem.metadata.get('func_name', None)

        for row in rows:
            solution_code = row.get('Solution_Code', '')
            is_normal_end = row.get('Is_Normal_End', '').lower() == 'true'

            if is_normal_end and solution_code and solution_code.strip():
                try:
                    tlefree_passed = tlefree_evaluate_simulation_code(
                        code=solution_code,
                        test_cases=test_cases,
                        fn_name=fn_name,
                        timeout=timeout
                    )
                    row['TLEfree_Passed'] = "TRUE" if tlefree_passed else "FALSE"
                except Exception as e:
                    row['TLEfree_Passed'] = "FALSE"
            else:
                row['TLEfree_Passed'] = "FALSE"

        e3_csv_path.parent.mkdir(parents=True, exist_ok=True)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(e3_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return True, ""

    except Exception as e:
        return False, f"evaluation_error: {str(e)}"


def process_single_problem_tlefree(task_data: Tuple) -> Tuple[str, bool, str]:
    """Worker for multiprocessing: same logic as process_single_csv with plain task tuple."""
    question_id, e1_csv_path_str, e3_csv_path_str, test_cases, fn_name, timeout, group_n = task_data

    e1_csv_path = Path(e1_csv_path_str)
    e3_csv_path = Path(e3_csv_path_str)

    if not e1_csv_path.exists():
        return question_id, False, "csv_not_found"

    try:
        with open(e1_csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        if not rows:
            return question_id, False, "csv_empty"

        for row in rows:
            solution_code = row.get('Solution_Code', '')
            is_normal_end = row.get('Is_Normal_End', '').lower() == 'true'

            if is_normal_end and solution_code and solution_code.strip():
                try:
                    tlefree_passed = tlefree_evaluate_simulation_code(
                        code=solution_code,
                        test_cases=test_cases,
                        fn_name=fn_name,
                        timeout=timeout
                    )
                    row['TLEfree_Passed'] = "TRUE" if tlefree_passed else "FALSE"
                except Exception as e:
                    row['TLEfree_Passed'] = "FALSE"
            else:
                row['TLEfree_Passed'] = "FALSE"

        e3_csv_path.parent.mkdir(parents=True, exist_ok=True)

        if rows:
            fieldnames = list(rows[0].keys())
            with open(e3_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        return question_id, True, ""

    except Exception as e:
        return question_id, False, f"evaluation_error: {str(e)}"


def tlefree_evaluate_for_all_groups(
    test_dataset: List[Problem],
    config: ExperimentConfig,
    run_log_path: Optional[Path] = None
) -> EvaluationStats:
    """For each group, copy e1 -> e3 per-problem CSV with TLEfree_Passed."""
    print("\nStarting TLEfree evaluation...")

    if run_log_path:
        log_run(run_log_path, "Starting TLEfree evaluation phase")

    total_stats = EvaluationStats()

    timeout = config.evaluator.timeout if hasattr(config.evaluator, 'timeout') else 0.8

    for group_n in range(1, config.experiment.n + 1):
        print(f"\n=== Processing group {group_n} ===")

        e1_save_dir = get_group_dir(
            experiment_id="e1",
            benchmark=config.experiment.benchmark,
            model=config.model.model,
            baseline=config.experiment.baseline,
            config_hash=config.get_config_hash(),
            group_n=group_n
        )

        e3_save_dir = get_group_dir(
            experiment_id="e3",
            benchmark=config.experiment.benchmark,
            model=config.model.model,
            baseline=config.experiment.baseline,
            config_hash=config.get_config_hash(),
            group_n=group_n
        )

        print(f"📁 e1 source dir: {e1_save_dir}")
        print(f"📁 e3 output dir: {e3_save_dir}")

        group_stats = EvaluationStats()
        group_stats.total_problems = len(test_dataset)

        need_process = []
        skipped_ids = []
        force_reevaluate_ids = []

        force_reevaluate = getattr(config, 'force_reevaluate', False)

        for problem in test_dataset:
            if check_existing_tlefree_evaluation(problem, e3_save_dir, force_reevaluate):
                skipped_ids.append(problem.question_id)
            else:
                need_process.append(problem)
                if force_reevaluate:
                    force_reevaluate_ids.append(problem.question_id)

        if skipped_ids:
            skip_info = format_id_list(skipped_ids, max_show=5)
            print(f"✅ Skipped (already evaluated): {len(skipped_ids)}" + (f" ({skip_info})" if skip_info else ""))

        if force_reevaluate_ids:
            force_info = format_id_list(force_reevaluate_ids, max_show=5)
            print(f"🔄 Force re-evaluate: {len(force_reevaluate_ids)}" + (f" ({force_info})" if force_info else ""))
        elif need_process:
            process_info = format_id_list([p.question_id for p in need_process], max_show=5)
            print(f"🆕 Pending: {len(need_process)}" + (f" ({process_info})" if process_info else ""))

        print(f"\nScan: {len(skipped_ids)} already done, {len(need_process)} pending")

        group_stats.processed_problems += len(skipped_ids)

        if not need_process:
            print("✅ All problems already processed; nothing to re-run")
            if run_log_path:
                log_run(run_log_path, f"Group {group_n} skipped (all complete)")
            total_stats.total_problems += group_stats.total_problems
            total_stats.processed_problems += group_stats.processed_problems
            continue

        tasks = []
        for problem in need_process:
            e1_csv_path = e1_save_dir / f"{problem.question_id}.csv"
            e3_csv_path = e3_save_dir / f"{problem.question_id}.csv"
            test_cases = problem.public_test_cases + problem.private_test_cases
            fn_name = problem.metadata.get('func_name', None)

            task_data = (
                problem.question_id,
                str(e1_csv_path),
                str(e3_csv_path),
                test_cases,
                fn_name,
                timeout,
                group_n
            )
            tasks.append(task_data)

        def update_stats_from_result(question_id: str, success: bool, error_msg: str):
            if success:
                group_stats.processed_problems += 1
            elif error_msg == "csv_not_found":
                group_stats.missing_csv_count += 1
                group_stats.missing_csv_ids.append(question_id)
            elif error_msg == "csv_empty":
                group_stats.empty_csv_count += 1
                group_stats.empty_csv_ids.append(question_id)
            else:
                group_stats.evaluation_errors += 1
                group_stats.error_ids.append(question_id)

        if config.multiprocess > 1:
            print(f"🚀 Multiprocessing, workers: {config.multiprocess}")

            with ProcessPoolExecutor(max_workers=config.multiprocess) as executor:
                future_to_task = {
                    executor.submit(process_single_problem_tlefree, task): task
                    for task in tasks
                }

                with tqdm(total=len(tasks), desc=f"TLEfree eval (group {group_n})", unit="prob") as pbar:
                    for future in as_completed(future_to_task):
                        try:
                            question_id, success, error_msg = future.result()

                            update_stats_from_result(question_id, success, error_msg)

                            status = "✅" if success else "⚠️"
                            pbar.set_postfix_str(f"{status} {question_id}")

                        except Exception as e:
                            task = future_to_task[future]
                            question_id = task[0]
                            print(f"❌ Task failed {question_id}: {e}")
                            update_stats_from_result(question_id, False, f"execution_error: {str(e)}")

                        pbar.update(1)
        else:
            print("🔧 Single-process mode")

            for problem in tqdm(need_process, desc=f"TLEfree eval (group {group_n})", unit="prob"):
                e1_csv_path = e1_save_dir / f"{problem.question_id}.csv"
                e3_csv_path = e3_save_dir / f"{problem.question_id}.csv"

                success, error_msg = process_single_csv(
                    e1_csv_path=e1_csv_path,
                    e3_csv_path=e3_csv_path,
                    problem=problem,
                    timeout=timeout,
                    group_n=group_n
                )

                update_stats_from_result(problem.question_id, success, error_msg)

        print(f"\nGroup {group_n} done:")
        print(f"  ✅ Processed: {group_stats.processed_problems} (incl. skipped: {len(skipped_ids)})")
        if group_stats.missing_csv_count > 0:
            print(f"  ⚠️ Missing CSV: {group_stats.missing_csv_count}")
        if group_stats.empty_csv_count > 0:
            print(f"  ⚠️ Empty CSV: {group_stats.empty_csv_count}")
        if group_stats.evaluation_errors > 0:
            print(f"  ❌ Evaluation errors: {group_stats.evaluation_errors}")

        total_stats.total_problems += group_stats.total_problems
        total_stats.processed_problems += group_stats.processed_problems
        total_stats.missing_csv_count += group_stats.missing_csv_count
        total_stats.empty_csv_count += group_stats.empty_csv_count
        total_stats.evaluation_errors += group_stats.evaluation_errors
        total_stats.missing_csv_ids.extend(group_stats.missing_csv_ids)
        total_stats.empty_csv_ids.extend(group_stats.empty_csv_ids)
        total_stats.error_ids.extend(group_stats.error_ids)

        if run_log_path:
            log_run(run_log_path,
                f"Group {group_n} done | ok: {group_stats.processed_problems}, "
                f"missing: {group_stats.missing_csv_count}, "
                f"empty: {group_stats.empty_csv_count}, "
                f"errors: {group_stats.evaluation_errors}")

    return total_stats


def print_final_report(stats: EvaluationStats) -> None:
    """Print aggregate counts and issue IDs."""
    print("\n" + "=" * 50)
    print("TLEfree evaluation — final report")
    print("=" * 50)

    print(f"\n📊 Summary:")
    print(f"  Total problems: {stats.total_problems}")
    print(f"  Processed: {stats.processed_problems}")
    print(f"  Success rate: {stats.processed_problems / stats.total_problems * 100:.1f}%" if stats.total_problems > 0 else "  Success rate: N/A")

    total_issues = stats.missing_csv_count + stats.empty_csv_count + stats.evaluation_errors
    if total_issues > 0:
        print(f"\n⚠️ Issues ({total_issues} total):")

        if stats.missing_csv_count > 0:
            print(f"\n  📂 Missing CSV files: {stats.missing_csv_count}")
            if len(stats.missing_csv_ids) <= 10:
                for qid in stats.missing_csv_ids:
                    print(f"      - {qid}")
            else:
                for qid in stats.missing_csv_ids[:5]:
                    print(f"      - {qid}")
                print(f"      ... and {len(stats.missing_csv_ids)} more")

        if stats.empty_csv_count > 0:
            print(f"\n  📄 Empty CSV files: {stats.empty_csv_count}")
            if len(stats.empty_csv_ids) <= 10:
                for qid in stats.empty_csv_ids:
                    print(f"      - {qid}")
            else:
                for qid in stats.empty_csv_ids[:5]:
                    print(f"      - {qid}")
                print(f"      ... and {len(stats.empty_csv_ids)} more")

        if stats.evaluation_errors > 0:
            print(f"\n  ❌ Evaluation errors: {stats.evaluation_errors}")
            if len(stats.error_ids) <= 10:
                for qid in stats.error_ids:
                    print(f"      - {qid}")
            else:
                for qid in stats.error_ids[:5]:
                    print(f"      - {qid}")
                print(f"      ... and {len(stats.error_ids)} more")
    else:
        print("\n✅ All problems processed successfully.")

    print("\n" + "=" * 50)


def main():
    config: ExperimentConfig = get_config_from_args()

    print_config_summary(config)
    print("\n🔄 TLEfree compare (re-evaluate e1 → e3)")
    print("  Source experiment: e1")
    print("  Output experiment: e3")

    e3_results_dir = get_results_dir(
        experiment_id="e3",
        benchmark=config.experiment.benchmark,
        model=config.model.model,
        baseline=config.experiment.baseline,
        config_hash=config.get_config_hash()
    )
    e3_results_dir.mkdir(parents=True, exist_ok=True)

    run_log_path = e3_results_dir / "run_log.txt"
    log_run(run_log_path, "=" * 40)
    log_run(run_log_path, f"TLEfree evaluation start | Hash: {config.get_config_hash()}")

    test_dataset = initialize_and_load_dataset(config)

    if not test_dataset:
        print("❌ Dataset load failed or empty; exiting")
        log_run(run_log_path, "Failed: empty dataset")
        return

    print(f"\n✅ Dataset ready: {len(test_dataset)} problems")
    log_run(run_log_path, f"Dataset: {len(test_dataset)} problems")

    stats = tlefree_evaluate_for_all_groups(test_dataset, config, run_log_path)

    print_final_report(stats)

    log_run(run_log_path,
        f"Done | ok: {stats.processed_problems}, "
        f"issues: {stats.missing_csv_count + stats.empty_csv_count + stats.evaluation_errors}")

    print(f"\n📁 Results written under e3 group dirs")
    print(f"📁 Log file: {run_log_path}")


if __name__ == "__main__":
    main()
