# solution pipeline: load dataset, run solvers (optional multiprocessing),
# save per-problem CSVs with resume, evaluate, then pass@k.

import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm
from loguru import logger

from src.config import ExperimentConfig, load_config_from_file
from src.utils.parser import get_config_from_args
from src.utils.config_printer import print_config_summary
from src.utils.path_manager import get_group_dir, get_results_dir, get_direct_answer_group_dir, get_Bruteforce_group_dir
from src.experiments import initialize_and_load_dataset
from src.core.problem import Problem
from src.core.solution import Solution
from src.baselines.solver_factory import create_solver
from src.baselines.base_solver import BaseSolver
from src.evaluators import (
    lcb_evaluate_generations,
    safe_single_process_evaluation,
    safe_single_process_evaluation_simple
)
from src.analyzers.common.pass_at_k_analyzer import (
    analyze_pass_at_k_by_difficulty,
    print_pass_at_k_by_difficulty_results,
)


def log_run(log_path: Path, message: str) -> None:
    """Append a line to the run log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] {message}\n")


def process_single_problem(problem_data: Tuple[Problem, Dict[str, Any]]) -> Tuple[str, Union[Solution, List[Solution]]]:
    """Worker for ProcessPoolExecutor: build solver in child process, return (question_id, solution)."""
    problem, solver_kwargs = problem_data
    
    try:
        solver_type = solver_kwargs.pop('solver_type', 'DirectAnswer')
        save_dir = solver_kwargs.pop('save_dir', None)
        if save_dir:
            save_dir = Path(save_dir)
        solver = create_solver(solver_type, **solver_kwargs)
        
        import inspect
        solve_signature = inspect.signature(solver.solve)
        if 'save_dir' in solve_signature.parameters:
            solution = solver.solve(problem, save_dir=save_dir)
        else:
            solution = solver.solve(problem)
        
        return (problem.question_id, solution)
        
    except Exception as e:
        print(f"Error on problem {problem.question_id}: {e}")
        return (problem.question_id, Solution(
            code="",
            problem_id=problem.question_id,
            is_normal_end=False,
            metadata={"error": str(e)}
        ))


def get_solution_status(solution: Union[Solution, List[Solution]]) -> bool:
    """True if generation ended normally; for a list, uses the last round."""
    if isinstance(solution, list):
        return solution[-1].is_normal_end if solution else False
    return solution.is_normal_end


def create_eval_sample(problem: Problem) -> Dict[str, Any]:
    """Build evaluator sample dict with JSON `input_output` from public+private tests."""
    return {
        "input_output": json.dumps({
            "inputs": [tc["input"] for tc in problem.public_test_cases + problem.private_test_cases],
            "outputs": [tc["output"] for tc in problem.public_test_cases + problem.private_test_cases],
            "fn_name": problem.metadata.get("func_name", None),
        })
    }


def check_existing_evaluation(problem: Problem, save_dir: Path, force_reevaluate: bool = False) -> bool:
    """Return True if CSV already has Passed and Result_Type filled for all rows (resume)."""
    if force_reevaluate:
        return False
    
    csv_path = save_dir / f"{problem.question_id}.csv"
    
    if not csv_path.exists():
        return False
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            if not rows:
                return False
            
            all_evaluated = True
            for row in rows:
                passed = row.get('Passed', '')
                result_type = row.get('Result_Type', '')
                if not passed or not result_type:
                    all_evaluated = False
                    break
            
            return all_evaluated
            
    except Exception as e:
        print(f"Failed to read evaluation CSV {csv_path}: {e}")
        return False


def read_solutions_from_csv(csv_path: Path) -> List[str]:
    """Read per-round Solution_Code rows from a problem CSV."""
    solution_codes = []
    
    if not csv_path.exists():
        return solution_codes
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)
            
            for row in rows:
                if row.get('Is_Normal_End', '').lower() == 'true':
                    solution_code = row.get('Solution_Code', '')
                    if solution_code and solution_code.strip():
                        solution_codes.append(solution_code)
                    else:
                        solution_codes.append('')
                else:
                    solution_codes.append('')
    
    except Exception as e:
        print(f"Failed to read CSV {csv_path}: {e}")
    
    return solution_codes


def check_existing_result(problem: Problem, save_dir: Path, force_regenerate: bool = False) -> bool:
    """Return True if first CSV row is normal end with non-empty code (skip regenerate)."""
    if force_regenerate:
        return False
    
    csv_path = save_dir / f"{problem.question_id}.csv"
    
    if not csv_path.exists():
        return False
    
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            first_row = next(reader, None)
            
            if first_row is None:
                return False
            
            is_normal_end = first_row.get('Is_Normal_End', 'False')
            if str(is_normal_end).lower() == 'true':
                solution_code = first_row.get('Solution_Code', '')
                if solution_code and solution_code.strip():
                    return True
            
            return False
            
    except Exception as e:
        print(f"Failed to read result CSV {csv_path}: {e}")
        return False


def save_single_result(
    problem: Problem, 
    solution: Union[Solution, List[Solution]], 
    save_dir: Path,
    solver: Optional[BaseSolver] = None
) -> Path:
    """Persist one problem's solution(s) via solver.save_solution or a manual CSV matching BaseSolver layout."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if solver is not None:
        return solver.save_solution(problem, solution, save_dir)
    
    csv_path = save_dir / f"{problem.question_id}.csv"
    
    if isinstance(solution, Solution):
        solutions = [solution]
    else:
        solutions = solution

    def serialize_value(value: Any) -> Any:
        """JSON/datetime serialization aligned with BaseSolver.save_solution."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, list):
            return json.dumps(value, ensure_ascii=False) if value else None
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False) if value else None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)
    
    # Generation phase: eval columns empty; filled in evaluate step.
    rows = []
    for sol in solutions:
        row = {
            "Question_Id": problem.question_id,
            "Question_Title": problem.question_title,
            "Platform": problem.platform or "",
            "Difficulty": problem.difficulty or "",
            "Contest_Date": serialize_value(problem.contest_date),
            "Solution_Code": sol.code or "",
            "Is_Normal_End": sol.is_normal_end,
            "Round_Index": sol.round_index if sol.round_index is not None else 0,
            "Is_Final": sol.is_final,
            "Passed": "",
            "Result_Type": "",
            "Error_Case_Indice": "",
            "Error_Case_Contents": "",
            "Local_Passed": sol.local_passed if sol.local_passed is not None else "",
            "Local_Result_Type": sol.local_result_type or "",
            "Schema": sol.schema or "",
            "Simulation_Code": sol.simulation_code or "",
            "Metadata": serialize_value(sol.metadata),
            "Execution_Results": serialize_value(sol.execution_results),
        }
        rows.append(row)
    
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    return csv_path


def update_single_evaluation_csv(
    problem: Problem,
    save_dir: Path,
    problem_results: List,
    problem_metadatas: List[Dict[str, Any]]
) -> bool:
    """Merge per-round evaluation into the problem CSV."""
    csv_path = save_dir / f"{problem.question_id}.csv"

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return False

    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
            reader = csv.DictReader(csvfile)
            rows = list(reader)

        if not rows:
            print(f"Empty CSV: {problem.question_id}")
            return False

        if len(problem_results) != len(rows):
            print(
                f"Mismatch eval rounds ({len(problem_results)}) vs CSV rows ({len(rows)}): {problem.question_id}"
            )
            num_to_process = min(len(problem_results), len(rows))
        else:
            num_to_process = len(rows)

        for round_idx in range(num_to_process):
            if round_idx < len(problem_results) and round_idx < len(problem_metadatas):
                evaluation_result = problem_results[round_idx]
                metadata = problem_metadatas[round_idx]

                passed = all(result is True for result in evaluation_result) if evaluation_result else False

                if passed:
                    result_type = "Accepted"
                else:
                    error_message = metadata.get("error_message", "Wrong Answer")
                    if error_message.startswith("Wrong answer"):
                        result_type = "Wrong Answer"
                    else:
                        result_type = error_message

                error_case_indice = str(evaluation_result) if evaluation_result else ""
                error_case_contents = "inputs:" + metadata.get("inputs", "") + \
                                    "\noutputs:" + metadata.get("output", "") + \
                                    "\nexpected:" + metadata.get("expected", "")

                rows[round_idx].update({
                    "Passed": "TRUE" if passed else "FALSE",
                    "Result_Type": result_type,
                    "Error_Case_Indice": error_case_indice,
                    "Error_Case_Contents": error_case_contents
                })

        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        first_result_type = rows[0].get("Result_Type", "Unknown")
        print(f"Updated {problem.question_id} (round1: {first_result_type}, {num_to_process} rounds)")
        return True

    except Exception as e:
        print(f"Failed to update CSV {csv_path}: {e}")
        return False


def generate_solutions_for_all(
    test_dataset: List[Problem], 
    config: ExperimentConfig, 
    group_n: int,
    run_log_path: Optional[Path] = None
) -> None:
    """Run solvers for all problems in one group; optional multiprocessing; resume via CSV."""
    start_time = datetime.now()
    print(f"\n=== Group {group_n} generation start ===")
    
    save_dir = get_group_dir(
        experiment_id=config.experiment.experiment_id,
        benchmark=config.experiment.benchmark,
        model=config.model.model,
        baseline=config.experiment.baseline,
        config_hash=config.get_config_hash(),
        group_n=group_n
    )
    print(f"Results directory: {save_dir}")
    
    need_process = []
    skip_count = 0
    skipped_ids = []
    need_process_ids = []
    force_regenerate_ids = []
    
    for i, problem in enumerate(test_dataset):
        if check_existing_result(problem, save_dir, config.force_regenerate):
            skipped_ids.append(problem.question_id)
            skip_count += 1
        else:
            need_process.append((i, problem))
            need_process_ids.append(problem.question_id)
            if config.force_regenerate:
                force_regenerate_ids.append(problem.question_id)
    
    def format_id_list(ids, max_show=5):
        """Compact ID list for logging."""
        if not ids:
            return ""
        if len(ids) <= max_show:
            return ", ".join(ids)
        else:
            return ", ".join(ids[:max_show]) + f", ... ({len(ids)} total)"
    
    if skipped_ids:
        skip_info = format_id_list(skipped_ids, max_show=5)
        print(f"Skipped (existing): {len(skipped_ids)}" + (f" ({skip_info})" if skip_info else ""))
    
    if need_process_ids:
        if force_regenerate_ids:
            force_info = format_id_list(force_regenerate_ids, max_show=5)
            print(f"Force regenerate: {len(force_regenerate_ids)}" + (f" ({force_info})" if force_info else ""))
        else:
            process_info = format_id_list(need_process_ids, max_show=5)
            print(f"To process: {len(need_process_ids)}" + (f" ({process_info})" if process_info else ""))
    
    print(f"\nResume check: {skip_count} done, {len(need_process)} pending")
    
    if run_log_path:
        log_run(
            run_log_path,
            f"Group {group_n} start | total: {len(test_dataset)}, skipped: {skip_count}, pending: {len(need_process)}",
        )
    
    if not need_process:
        print("All problems already generated.")
        if run_log_path:
            log_run(run_log_path, f"Group {group_n} skipped (all complete)")
        return
    
    baseline = config.experiment.baseline
    if baseline == "DirectAnswer":
        solver_type = "DirectAnswer"
    elif baseline == "Bruteforce":
        solver_type = "Bruteforce"
    elif baseline == "Textgrad":
        solver_type = "Textgrad"
    elif baseline in ("ExeCRE", "TrustTestEM"):
        solver_type = "ExeCRE"
    elif baseline == "GSM8KPM":
        solver_type = "GSM8KPM"
    elif baseline == "GSM8KExeCRE":
        solver_type = "GSM8KExeCRE"
    else:
        solver_type = "DirectAnswer"
    
    solver_kwargs = {
        'solver_type': solver_type,
        'model_name': config.model.model,
        'temperature': config.model.temperature,
        'max_tokens': config.model.max_tokens,
        'top_p': config.model.top_p,
    }
    
    if solver_type == "Textgrad":
        solver_kwargs['max_iters'] = config.max_rounds
        solver_kwargs['direct_solve_dir'] = get_direct_answer_group_dir(config, group_n)
    
    if solver_type == "ExeCRE":
        solver_kwargs['max_iters'] = config.max_rounds
        solver_kwargs['direct_solve_dir'] = get_direct_answer_group_dir(config, group_n)
        solver_kwargs['bruteforce_dir'] = get_Bruteforce_group_dir(config, group_n, temperature=config.model.sampling_temperature)
        solver_kwargs['test_case_count'] = config.evaluator.test_case_count
        solver_kwargs['total_generation_timeout'] = config.evaluator.total_generation_timeout
        solver_kwargs['single_test_timeout'] = config.evaluator.single_test_timeout
        if config.experiment.eval_score is not None:
            solver_kwargs['eval_score_threshold'] = config.experiment.eval_score
        if config.experiment.error_score is not None:
            solver_kwargs['error_score_threshold'] = config.experiment.error_score
        em4c_config = getattr(config, 'em4c', None)
        if em4c_config is None:
            em4c_config = {}
        solver_kwargs['max_schema_candidates'] = em4c_config.get('max_schema_candidates', 5)
        solver_kwargs['max_solution_candidates'] = em4c_config.get('max_solution_candidates', 10)
        solver_kwargs['em4c_test_case_count'] = em4c_config.get('em4c_test_case_count', 300)
        solver_kwargs['allowed_error_ratio'] = em4c_config.get('allowed_error_ratio', 0.1)
        solver_kwargs['alpha_threshold'] = em4c_config.get('alpha_threshold', 0.95)
        solver_kwargs['skip_schema_oracle_phase2'] = em4c_config.get('skip_schema_oracle_phase2', True)
    
    if solver_type == "GSM8KExeCRE":
        gsm8k_ec = getattr(config, 'gsm8k_execre', None)
        if gsm8k_ec is None:
            gsm8k_ec = {}
        solver_kwargs['max_solution_candidates'] = gsm8k_ec.get('max_solution_candidates', 10)
        solver_kwargs['em4c_test_case_count'] = gsm8k_ec.get('em4c_test_case_count', 300)
        solver_kwargs['allowed_error_ratio'] = gsm8k_ec.get('allowed_error_ratio', 0.3)
        solver_kwargs['alpha_threshold'] = gsm8k_ec.get('alpha_threshold', 0.90)
        solver_kwargs['sampling_temperature'] = gsm8k_ec.get(
            'sampling_temperature',
            getattr(config.model, 'sampling_temperature', 0.8),
        )
        solver_kwargs['perturbation_range'] = gsm8k_ec.get('perturbation_range', 0.5)
        solver_kwargs['fallback_to_base'] = gsm8k_ec.get('fallback_to_base', True)
        solver_kwargs['bruteforce_dir'] = get_Bruteforce_group_dir(
            config, group_n,
            temperature=solver_kwargs['sampling_temperature'],
        )

    tasks = []
    for _, problem in need_process:
        task_kwargs = solver_kwargs.copy()
        if solver_type == "ExeCRE":
            task_kwargs['save_dir'] = str(save_dir)
        tasks.append((problem, task_kwargs))
    
    if config.multiprocess > 1:
        print(f"Multiprocess workers: {config.multiprocess}")
        
        with ProcessPoolExecutor(max_workers=config.multiprocess) as executor:
            future_to_idx = {
                executor.submit(process_single_problem, task): idx
                for idx, (_, task) in enumerate(zip(need_process, tasks))
            }
            
            with tqdm(total=len(tasks), desc=f"Generate group {group_n}", unit="prob") as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        question_id, solution = future.result()
                        
                        idx = future_to_idx[future]
                        _, problem = need_process[idx]
                        
                        save_single_result(problem, solution, save_dir)
                        
                        status = "ok" if get_solution_status(solution) else "warn"
                        pbar.set_postfix_str(f"{status} {question_id}")
                        
                    except Exception as e:
                        print(f"Task failed: {e}")
                    
                    pbar.update(1)
    else:
        print("Single-process generation")
        
        solver_type = solver_kwargs.pop('solver_type')
        solver = create_solver(solver_type, **solver_kwargs)
        
        for _, problem in tqdm(need_process, desc=f"Generate group {group_n}", unit="prob"):
            try:
                import inspect
                solve_signature = inspect.signature(solver.solve)
                if 'save_dir' in solve_signature.parameters and solver_type == "ExeCRE":
                    solution = solver.solve(problem, save_dir=save_dir)
                else:
                    solution = solver.solve(problem)
                
                save_single_result(problem, solution, save_dir, solver)
                
                status = "ok" if get_solution_status(solution) else "warn"
                print(f"{status} {problem.question_id}")
                
            except Exception as e:
                print(f"Failed {problem.question_id}: {e}")
                failed_solution = Solution(
                    code="",
                    problem_id=problem.question_id,
                    is_normal_end=False,
                    metadata={"error": str(e)}
                )
                save_single_result(problem, failed_solution, save_dir)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\nGroup {group_n} generation done")
    print(f"Elapsed: {duration}")
    print(f"Saved under: {save_dir}")
    
    if run_log_path:
        log_run(run_log_path, f"Group {group_n} done | processed: {len(need_process)}, elapsed: {duration}")


def _evaluate_gsm8k_group(
    test_dataset: List[Problem],
    need_reevaluate: List[int],
    eval_generations: List[List[str]],
    save_dir: Path,
    group_n: int,
) -> None:
    """GSM8K: parse answers from Solution_Code vs gold_answer; no code execution."""
    from src.baselines.gsm8k.answer_parser import parse_gsm8k_answer, gsm8k_equal

    print("GSM8K answer-matching evaluation...")
    correct, wrong = 0, 0

    for eval_idx, dataset_idx in enumerate(tqdm(
        need_reevaluate, desc=f"GSM8K eval group {group_n}", unit="prob"
    )):
        problem = test_dataset[dataset_idx]
        csv_path = save_dir / f"{problem.question_id}.csv"
        gold = problem.metadata.get("gold_answer")

        if not csv_path.exists():
            print(f"CSV not found: {csv_path}")
            continue

        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                rows = list(reader)

            if not rows:
                print(f"Empty CSV: {problem.question_id}")
                continue

            for row in rows:
                text = row.get("Solution_Code", "")
                prediction = parse_gsm8k_answer(text)
                passed = gsm8k_equal(prediction, gold) if gold is not None else False

                if passed:
                    correct += 1
                else:
                    wrong += 1

                row.update({
                    "Passed": "TRUE" if passed else "FALSE",
                    "Result_Type": "Accepted" if passed else "Wrong Answer",
                    "Error_Case_Indice": "",
                    "Error_Case_Contents": (
                        f"prediction={prediction}, gold={gold}"
                    ),
                })

            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = list(rows[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        except Exception as e:
            print(f"Failed to update CSV {csv_path}: {e}")

    total = correct + wrong
    acc = correct / total if total > 0 else 0.0
    print(f"GSM8K eval done: {correct}/{total} = {acc:.2%}")


def evaluate_all_groups(
    test_dataset: List[Problem], 
    config: ExperimentConfig,
    run_log_path: Optional[Path] = None
) -> None:
    """Evaluate all groups from CSV generations; resume when eval columns are complete."""
    print("\nEvaluating generations...")
    
    if run_log_path:
        log_run(run_log_path, "Evaluation phase start")
    
    start_from = config.start_from
    if start_from > 1:
        print(f"Evaluating from group {start_from} (skipping groups 1..{start_from - 1})")
    
    for group_n in range(start_from, config.experiment.n + 1):
        print(f"\n=== Evaluate group {group_n} ===")
        
        save_dir = get_group_dir(
            experiment_id=config.experiment.experiment_id,
            benchmark=config.experiment.benchmark,
            model=config.model.model,
            baseline=config.experiment.baseline,
            config_hash=config.get_config_hash(),
            group_n=group_n
        )
        print(f"Eval directory: {save_dir}")
        
        print("Loading generations for evaluation...")
        all_generations = []
        all_eval_samples = []
        need_reevaluate = []
        
        existing_eval_ids = []
        force_reevaluate_ids = []
        new_problem_ids = []
        incomplete_eval_ids = []
        no_code_ids = []
        
        for i, problem in enumerate(test_dataset):
            csv_path = save_dir / f"{problem.question_id}.csv"
            
            if check_existing_evaluation(problem, save_dir, config.force_reevaluate):
                if not config.force_reevaluate:
                    existing_eval_ids.append(problem.question_id)
                else:
                    need_reevaluate.append(i)
                    force_reevaluate_ids.append(problem.question_id)
            else:
                need_reevaluate.append(i)
                if not csv_path.exists():
                    new_problem_ids.append(problem.question_id)
                else:
                    incomplete_eval_ids.append(problem.question_id)
            
            solution_codes = read_solutions_from_csv(csv_path)
            all_generations.append(solution_codes)
            all_eval_samples.append(create_eval_sample(problem))
            
            if not solution_codes:
                no_code_ids.append(problem.question_id)
        
        def format_id_list(ids, max_show=5):
            """Compact ID list for logging."""
            if not ids:
                return ""
            if len(ids) <= max_show:
                return ", ".join(ids)
            else:
                return ", ".join(ids[:max_show]) + f", ... ({len(ids)} total)"
        
        if existing_eval_ids:
            eval_info = format_id_list(existing_eval_ids, max_show=5)
            print(f"Reuse existing eval: {len(existing_eval_ids)}" + (f" ({eval_info})" if eval_info else ""))
        
        if force_reevaluate_ids:
            force_info = format_id_list(force_reevaluate_ids, max_show=5)
            print(f"Force re-eval: {len(force_reevaluate_ids)}" + (f" ({force_info})" if force_info else ""))
        
        if new_problem_ids:
            new_info = format_id_list(new_problem_ids, max_show=10)
            print(f"New problems to eval: {len(new_problem_ids)}" + (f" ({new_info})" if new_info else ""))
        
        if incomplete_eval_ids:
            incomplete_info = format_id_list(incomplete_eval_ids, max_show=10)
            print(
                f"Incomplete eval: {len(incomplete_eval_ids)}"
                + (f" ({incomplete_info})" if incomplete_info else "")
            )
        
        if no_code_ids:
            for problem_id in no_code_ids:
                print(f"No code read: {problem_id}")
        
        print(f"\nResume check: {len(need_reevaluate)} problems need evaluation")
        
        if not need_reevaluate:
            print("All problems already evaluated.")
            if run_log_path:
                log_run(run_log_path, f"Group {group_n} eval skipped (all complete)")
            continue
        
        eval_generations = [all_generations[i] for i in need_reevaluate]
        eval_samples = [all_eval_samples[i] for i in need_reevaluate]
        eval_problem_ids = [test_dataset[i].question_id for i in need_reevaluate]
        
        print(f"Ready to eval {len(eval_generations)} problems")
        
        if config.experiment.benchmark == "GSM8K":
            _evaluate_gsm8k_group(
                test_dataset, need_reevaluate, eval_generations, save_dir, group_n
            )
        else:
            if config.evaluator.evaluation_mode == "single_process":
                print("Single-process evaluation...")

                def on_problem_complete(local_idx: int, problem_results: List, problem_metadatas: List[Dict[str, Any]]) -> None:
                    problem = test_dataset[need_reevaluate[local_idx]]
                    update_single_evaluation_csv(problem, save_dir, problem_results, problem_metadatas)

                all_results, all_metadatas = safe_single_process_evaluation(
                    eval_samples,
                    eval_generations,
                    config.evaluator.timeout,
                    problem_ids=eval_problem_ids,
                    on_problem_complete=on_problem_complete
                )
            elif config.evaluator.evaluation_mode == "single_process_safe":
                print("Single-process safe evaluation...")

                def on_problem_complete(local_idx: int, problem_results: List, problem_metadatas: List[Dict[str, Any]]) -> None:
                    problem = test_dataset[need_reevaluate[local_idx]]
                    update_single_evaluation_csv(problem, save_dir, problem_results, problem_metadatas)

                all_results, all_metadatas = safe_single_process_evaluation_simple(
                    eval_samples,
                    eval_generations,
                    config.evaluator.timeout,
                    problem_ids=eval_problem_ids,
                    on_problem_complete=on_problem_complete
                )
            else:
                print("Multiprocess evaluation...")
                all_results, all_metadatas = lcb_evaluate_generations(
                    samples_list=eval_samples,
                    generations_list=eval_generations,
                    debug=config.evaluator.debug,
                    num_process_evaluate=config.evaluator.num_process_evaluate,
                    timeout=config.evaluator.timeout
                )
            
            print("Batch evaluation finished")

            if config.evaluator.evaluation_mode in {"single_process", "single_process_safe"}:
                print("Per-problem CSVs updated during single-process eval")
            else:
                print("Writing evaluation to CSV...")

                for idx, i in enumerate(tqdm(need_reevaluate, desc=f"Write CSV group {group_n}", unit="prob")):
                    problem = test_dataset[i]
                    problem_results = all_results.get(idx, [])
                    problem_metadatas = all_metadatas.get(idx, [])
                    update_single_evaluation_csv(problem, save_dir, problem_results, problem_metadatas)
        
        print(f"\nGroup {group_n} evaluation done")
        if run_log_path:
            log_run(run_log_path, f"Group {group_n} eval done | count: {len(need_reevaluate)}")
    
    print("\nEvaluation finished; CSVs updated per problem.")
    if run_log_path:
        log_run(run_log_path, "Evaluation phase complete")


def main():
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    config: ExperimentConfig = get_config_from_args()
    
    print_config_summary(config)
    
    results_dir = get_results_dir(
        experiment_id=config.experiment.experiment_id,
        benchmark=config.experiment.benchmark,
        model=config.model.model,
        baseline=config.experiment.baseline,
        config_hash=config.get_config_hash()
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    
    config_path = results_dir / "config.json"
    if not config_path.exists():
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config.to_json())
        print(f"Config saved: {config_path}")
    
    run_log_path = results_dir / "run_log.txt"
    log_run(run_log_path, "=" * 40)
    log_run(run_log_path, f"Run start | hash: {config.get_config_hash()}")
    
    test_dataset = initialize_and_load_dataset(config)
    
    if not test_dataset:
        print("Dataset load failed or empty; exiting.")
        log_run(run_log_path, "Run failed: empty dataset")
        return
    
    print(f"\nDataset ready: {len(test_dataset)} problems")
    log_run(run_log_path, f"Dataset size: {len(test_dataset)}")
    
    start_from = config.start_from
    if not config.skip_generate:
        if start_from > 1:
            print(f"\nGenerating... (groups {start_from}..{config.experiment.n})")
            log_run(run_log_path, f"Skip generate groups 1..{start_from - 1}, start at {start_from}")
        else:
            print(f"\nGenerating... ({config.experiment.n} groups)")
        
        for group_n in range(start_from, config.experiment.n + 1):
            generate_solutions_for_all(test_dataset, config, group_n, run_log_path)
        
        print(f"\nGeneration done for groups {start_from}..{config.experiment.n}")
        log_run(run_log_path, f"Generation done groups {start_from}..{config.experiment.n}")
    else:
        print("\nSkipping generation")
        log_run(run_log_path, "Skipped generation")
    
    if not config.skip_evaluate:
        evaluate_all_groups(test_dataset, config, run_log_path)
    else:
        print("\nSkipping evaluation")
        log_run(run_log_path, "Skipped evaluation")
    
    print("\n" + "=" * 40)
    print("Computing pass@k...")
    results = analyze_pass_at_k_by_difficulty(config, test_dataset=test_dataset)
    print_pass_at_k_by_difficulty_results(results)


if __name__ == "__main__":
    main()
