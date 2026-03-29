"""
ExeCRE: TextGrad refinement with EM4C schema/simulation selection and Sample / Rand / TLE checks.
Default labels: Majority01; confidence: Dawid-Skene.
"""

import sys
import os
import csv
import json
import time
from pathlib import Path
from typing import List, Optional, Union, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from src.EM4C.EM4C import step6_calculate_confidence
from src.EM4C.label_processors.majority_01 import Majority01Processor
from src.EM4C.confidence_calculators.dawid_skene import DawidSkeneCalculator

_textgrad_parent_path = Path(__file__).parent.parent / "textgrad"
if str(_textgrad_parent_path) not in sys.path:
    sys.path.insert(0, str(_textgrad_parent_path))

from src.baselines.base_solver import BaseSolver
from src.core.problem import Problem
from src.core.solution import Solution

import textgrad
from textgrad.variable import Variable
from textgrad.optimizer.optimizer import TextualGradientDescent
from src.engine import get_engine as tg_get_engine

from src.baselines.textgrad.py_eval import evaluate
from src.baselines.textgrad.prompts import CODE_INSTANCE_ROLE_DESCRIPTION, CodeTestTimewithTests

from src.test_generators.schema_oracle.generator import SchemaOracleTestGenerator
from src.test_generators.schema_oracle.config import SchemaOracleConfig

from src.evaluators.tlefree_evaluator import tlefree_evaluate_simulation_code
from src.evaluators.tle_evaluator import tle_evaluate

from src.input_generators.schema.schema import generate_data_from_schema
from src.input_generators.schema import config as schema_config_module

from src.evaluators.lcb_coderunner import run_code_capture


class TrustTestSolver(BaseSolver):
    """Multi-round solver: SchemaOracle + EM4C, then Sample / Rand / TLE via TextGrad."""

    VERBOSE_OUTPUT = True

    def __init__(
        self,
        name: str = "TrustTest",
        model_name: str = "openai-gpt-4o",
        max_iters: int = 4,
        direct_solve_dir: Optional[Union[str, Path]] = None,
        bruteforce_dir: Optional[Union[str, Path]] = None,
        eval_score_threshold: float = 0.8,
        error_score_threshold: float = 0.1,
        test_case_count: int = 100000,
        total_generation_timeout: float = 60.0,
        single_test_timeout: float = 0.8,
        iteration_timeout: float = 300.0,
        skip_schema_oracle_phase2: bool = False,
        use_existing_intermediate_results: bool = True,
        intermediate_results_group: Optional[int] = 1,
        **kwargs
    ):
        super().__init__(name, model_name=model_name, max_iters=max_iters, **kwargs)
        self.model_name = model_name
        self.max_iters = max_iters
        self.direct_solve_dir = Path(direct_solve_dir) if direct_solve_dir else None
        self.bruteforce_dir = Path(bruteforce_dir) if bruteforce_dir else None
        self.eval_score_threshold = eval_score_threshold
        self.error_score_threshold = error_score_threshold

        self.test_case_count = test_case_count
        self.total_generation_timeout = total_generation_timeout
        self.single_test_timeout = single_test_timeout
        self.iteration_timeout = iteration_timeout

        self.skip_schema_oracle_phase2 = skip_schema_oracle_phase2

        self.use_existing_intermediate_results = use_existing_intermediate_results
        self.intermediate_results_group = intermediate_results_group

        self._engine = None
        self._direct_solver = None
        self._test_generator: Optional[SchemaOracleTestGenerator] = None

        self._current_schema: Optional[str] = None
        self._current_simulation_code: Optional[str] = None
        self._current_error_score: float = 1.0
        self._current_eval_score: float = 0.0

        self._current_alpha: float = 0.0
        self._current_save_dir: Optional[Path] = None
        self._schema_candidates: List[str] = []
        self._solution_candidates: List[str] = []
        self._selected_schema_index: int = -1
        self._selected_solution_index: int = -1
        self._em4c_metadata: Optional[Dict[str, Any]] = None

        self.max_schema_candidates = kwargs.get('max_schema_candidates', 5)
        self.max_solution_candidates = kwargs.get('max_solution_candidates', 10)
        self.em4c_test_case_count = kwargs.get('em4c_test_case_count', 300)
        self.allowed_error_ratio = kwargs.get('allowed_error_ratio', 0.3)
        self.alpha_threshold = kwargs.get('alpha_threshold', 0.90)

        if self.VERBOSE_OUTPUT:
            if self.direct_solve_dir is not None:
                print(f"DirectAnswer dir: {self.direct_solve_dir}")
            if self.bruteforce_dir is not None:
                print(f"Bruteforce dir: {self.bruteforce_dir}")
            print(
                "SchemaOracle phase2: "
                + (
                    "skipped (EM4C only)"
                    if self.skip_schema_oracle_phase2
                    else "run (EM4C overrides selection)"
                )
            )
            if self.use_existing_intermediate_results and self.intermediate_results_group is not None:
                print(f"Reuse intermediate results: group_{self.intermediate_results_group}")

    def _get_engine(self):
        """Lazily init TextGrad backward engine."""
        if self._engine is None:
            self._engine = tg_get_engine(self.model_name)
            textgrad.set_backward_engine(self._engine, override=True)
        return self._engine
    
    def _get_direct_solver(self):
        """Lazy DirectAnswerSolver."""
        if self._direct_solver is None:
            from src.baselines.direct_solve.solver import DirectAnswerSolver
            self._direct_solver = DirectAnswerSolver(model_name=self.model_name)
        return self._direct_solver
    
    def _read_initial_code_from_dir(self, problem: Problem) -> Optional[str]:
        """Load initial code from ``direct_solve_dir`` CSV if present."""
        if self.direct_solve_dir is None:
            return None
        
        csv_path = self.direct_solve_dir / f"{problem.question_id}.csv"
        
        if not csv_path.exists():
            return None
        
        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)
                first_row = next(reader, None)
                
                if first_row is None:
                    return None

                solution_code = first_row.get('Solution_Code', '')

                if solution_code and solution_code.strip():
                    is_normal_end = first_row.get('Is_Normal_End', 'True')
                    if str(is_normal_end).lower() == 'true':
                        return solution_code.strip()
                
                return None
                
        except Exception as e:
            print(f"Failed to read initial code {csv_path}: {e}")
            return None

    def _read_solution_candidates_from_bruteforce_dir(self, problem: Problem) -> List[str]:
        """Collect candidate simulation_code from bruteforce_dir (group_* CSVs)."""
        if self.bruteforce_dir is None:
            return []
        
        if not self.bruteforce_dir.exists():
            if self.VERBOSE_OUTPUT:
                print(f"Bruteforce dir missing: {self.bruteforce_dir}")
            return []
        
        solution_codes = []
        
        try:
            if self.VERBOSE_OUTPUT:
                print(f"Reading bruteforce candidates from: {self.bruteforce_dir}")

            search_dir = self.bruteforce_dir
            dir_name = search_dir.name
            if dir_name.startswith('group_'):
                search_dir = search_dir.parent
                if self.VERBOSE_OUTPUT:
                    print(f"   Using parent of group dir: {search_dir}")

            group_dirs = []
            if not search_dir.exists():
                if self.VERBOSE_OUTPUT:
                    print(f"Search dir missing: {search_dir}")
                return []
            
            for item in search_dir.iterdir():
                if item.is_dir():
                    item_name = item.name
                    if item_name.startswith('group_'):
                        try:
                            group_n = int(item_name[6:])
                            group_dirs.append((group_n, item))
                        except ValueError:
                            continue
                    else:
                        try:
                            group_n = int(item_name)
                            group_dirs.append((group_n, item))
                        except ValueError:
                            continue
            
            group_dirs.sort(key=lambda x: x[0])

            if self.VERBOSE_OUTPUT:
                print(f"   Found {len(group_dirs)} group dirs: {[f'group_{g[0]}' for g in group_dirs]}")

            for group_n, group_dir in group_dirs:
                csv_path = group_dir / f"{problem.question_id}.csv"
                
                if not csv_path.exists():
                    continue
                
                try:
                    with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
                        reader = csv.DictReader(csvfile)
                        
                        for row in reader:
                            solution_code = row.get('Solution_Code', '')

                            if not solution_code or not solution_code.strip():
                                continue

                            is_normal_end = row.get('Is_Normal_End', 'False')
                            if str(is_normal_end).lower() != 'true':
                                continue

                            code = solution_code.strip()
                            if code not in solution_codes:
                                solution_codes.append(code)

                                if len(solution_codes) >= self.max_solution_candidates:
                                    break

                except Exception as e:
                    if self.VERBOSE_OUTPUT:
                        print(f"Failed to read bruteforce CSV {csv_path}: {e}")
                    continue

                if len(solution_codes) >= self.max_solution_candidates:
                    break

            if self.VERBOSE_OUTPUT:
                print(f"Loaded {len(solution_codes)} bruteforce candidates")

            return solution_codes[:self.max_solution_candidates]

        except Exception as e:
            if self.VERBOSE_OUTPUT:
                print(f"Bruteforce read failed {self.bruteforce_dir}: {e}")
            return []
    
    def _get_initial_code(self, problem: Problem) -> Tuple[str, bool]:
        """Return (code, ok): prefer direct_solve_dir else DirectAnswerSolver."""
        initial_code = self._read_initial_code_from_dir(problem)
        if initial_code is not None:
            if self.VERBOSE_OUTPUT:
                print(f"Loaded initial code from direct_solve_dir: {problem.question_id}")
            return initial_code, True

        if self.VERBOSE_OUTPUT:
            print(f"Running DirectAnswerSolver for initial code: {problem.question_id}")
        direct_solver = self._get_direct_solver()
        initial_solution = direct_solver.solve(problem)
        
        return initial_solution.code, initial_solution.is_normal_end
    
    def _load_existing_intermediate_results(self, problem: Problem) -> Optional[Dict[str, Any]]:
        """Load schema, simulation_code, and EM4C metadata from a prior run (group_N / metadata.json)."""
        if not self.use_existing_intermediate_results or self.intermediate_results_group is None:
            return None

        if self._current_save_dir is None:
            if self.VERBOSE_OUTPUT:
                print("[EM4C] save_dir unset; cannot load intermediate results")
            return None

        try:
            current_save_dir = Path(self._current_save_dir)

            if current_save_dir.name.startswith('group_'):
                base_dir = current_save_dir.parent
                intermediate_group_dir = base_dir / f"group_{self.intermediate_results_group}"
            else:
                intermediate_group_dir = current_save_dir

            question_dir = intermediate_group_dir / str(problem.question_id)
            metadata_file = question_dir / "metadata.json"

            if not metadata_file.exists():
                if self.VERBOSE_OUTPUT:
                    print(f"[EM4C] intermediate metadata missing: {metadata_file}")
                return None

            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            required_fields = ['schema', 'simulation_code', 'alpha', 'selected_schema_index', 'selected_solution_index']
            for field in required_fields:
                if field not in metadata:
                    if self.VERBOSE_OUTPUT:
                        print(f"[EM4C] intermediate metadata missing field: {field}")
                    return None

            result = {
                'schema': metadata['schema'],
                'simulation_code': metadata['simulation_code'],
                'alpha': float(metadata['alpha']),
                'selected_schema_index': int(metadata['selected_schema_index']),
                'selected_solution_index': int(metadata['selected_solution_index']),
                'em4c_metadata': metadata.get('em4c_metadata', {}),
                'error_score': 1.0,
            }

            try:
                schema_dir = question_dir / "EM4C" / f"schema_{result['selected_schema_index']}"
                execution_matrix_file = schema_dir / "execution_matrix.json"
                if execution_matrix_file.exists():
                    with open(execution_matrix_file, 'r', encoding='utf-8') as f:
                        matrix_data = json.load(f)
                        error_ratio = matrix_data.get('error_ratio', 1.0)
                        result['error_score'] = error_ratio
            except Exception as e:
                if self.VERBOSE_OUTPUT:
                    print(f"[EM4C] could not read error_ratio ({e}); using default")

            if self.VERBOSE_OUTPUT:
                print(
                    f"[EM4C] loaded intermediate group_{self.intermediate_results_group}, "
                    f"alpha={result['alpha']:.4f}, "
                    f"schema_idx={result['selected_schema_index']}, "
                    f"solution_idx={result['selected_solution_index']}"
                )

            return result

        except Exception as e:
            if self.VERBOSE_OUTPUT:
                print(f"[EM4C] failed to load intermediate results: {e}")
                import traceback
                traceback.print_exc()
            return None
    
    def _init_test_generator(self, problem: Problem) -> bool:
        """Build schema/simulation candidates (reuse disk, bruteforce cache, or LLM) and run EM4C selection."""
        try:
            if self.use_existing_intermediate_results and self.intermediate_results_group is not None:
                existing_results = self._load_existing_intermediate_results(problem)
                if existing_results is not None:
                    self._current_schema = existing_results['schema']
                    self._current_simulation_code = existing_results['simulation_code']
                    self._current_alpha = existing_results['alpha']
                    self._selected_schema_index = existing_results['selected_schema_index']
                    self._selected_solution_index = existing_results['selected_solution_index']
                    self._em4c_metadata = existing_results['em4c_metadata']
                    self._current_error_score = existing_results['error_score']
                    self._current_eval_score = existing_results['alpha']

                    if self._current_alpha < self.alpha_threshold:
                        if self.VERBOSE_OUTPUT:
                            print(
                                f"EM4C alpha {self._current_alpha:.4f} < threshold {self.alpha_threshold:.4f}; "
                                "no valid simulation_code"
                            )
                        return False

                    if self.VERBOSE_OUTPUT:
                        print(
                            f"[EM4C] init from cache: alpha={self._current_alpha:.4f} "
                            f"(>= {self.alpha_threshold:.4f}), error_score={self._current_error_score:.3f}, "
                            f"schema_idx={self._selected_schema_index}, "
                            f"solution_idx={self._selected_solution_index}"
                        )

                    return True
                else:
                    if self.VERBOSE_OUTPUT:
                        print("[EM4C] cache miss; generating candidates")

            engine = self._get_engine()

            schema_oracle_config = SchemaOracleConfig()
            schema_oracle_config.max_schema_candidates = self.max_schema_candidates
            schema_oracle_config.max_simulation_candidates = self.max_solution_candidates
            
            self._test_generator = SchemaOracleTestGenerator(
                engine=engine,
                schema_oracle_config=schema_oracle_config
            )
            
            bruteforce_solutions = []
            if self.bruteforce_dir is not None:
                bruteforce_solutions = self._read_solution_candidates_from_bruteforce_dir(problem)

            if self.VERBOSE_OUTPUT:
                if bruteforce_solutions:
                    print(
                        f"[EM4C] bruteforce seeds={len(bruteforce_solutions)}; "
                        f"target schemas={self.max_schema_candidates}, "
                        f"need {max(0, self.max_solution_candidates - len(bruteforce_solutions))} more solutions"
                    )
                else:
                    print(
                        f"[EM4C] generating candidates: {self.max_schema_candidates} schemas, "
                        f"{self.max_solution_candidates} solutions"
                    )

            if self.VERBOSE_OUTPUT:
                print("[EM4C] regenerating schemas (cache bypass)")

            from src.test_generators.schema_oracle.generator import CandidatesArtifact
            artifact = CandidatesArtifact()

            artifact.schema_candidates = self._test_generator._generate_schema_candidates(problem)

            if bruteforce_solutions and len(bruteforce_solutions) >= self.max_solution_candidates:
                from src.test_generators.schema_oracle.generator import SolutionCandidate
                artifact.solution_candidates = [
                    SolutionCandidate(
                        code=code,
                        index=i,
                        is_valid=True,
                        metadata={"source": "bruteforce_dir"}
                    )
                    for i, code in enumerate(bruteforce_solutions[:self.max_solution_candidates])
                ]
            else:
                if bruteforce_solutions:
                    from src.test_generators.schema_oracle.generator import SolutionCandidate
                    artifact.solution_candidates = [
                        SolutionCandidate(
                            code=code,
                            index=i,
                            is_valid=True,
                            metadata={"source": "bruteforce_dir"}
                        )
                        for i, code in enumerate(bruteforce_solutions)
                    ]
                    need_count = self.max_solution_candidates - len(bruteforce_solutions)
                    if need_count > 0:
                        generated = self._test_generator._generate_solution_candidates(
                            problem, 
                            count=need_count, 
                            start_index=len(bruteforce_solutions)
                        )
                        artifact.solution_candidates.extend(generated)
                else:
                    artifact.solution_candidates = self._test_generator._generate_solution_candidates(problem)

            self._test_generator._candidates_artifact = artifact
            self._test_generator._problem = problem

            self._test_generator._save_schemas_separately(problem, artifact.schema_candidates)
            self._test_generator._save_solutions_separately(problem, artifact.solution_candidates)

            if self.skip_schema_oracle_phase2:
                if self.VERBOSE_OUTPUT:
                    print("[EM4C] skip SchemaOracle phase2 (majority vote); EM4C selects")
            else:
                if self.VERBOSE_OUTPUT:
                    print("[EM4C] run SchemaOracle phase2 (majority vote); EM4C will override")
                success = True
                try:
                    self._test_generator._execute_phase2_and_save(problem)
                    if self.VERBOSE_OUTPUT:
                        print("[EM4C] phase2 result discarded; EM4C selection follows")
                except Exception as e:
                    if self.VERBOSE_OUTPUT:
                        print(f"phase2 failed: {e}")
                    success = False

                if not success:
                    if self.VERBOSE_OUTPUT:
                        print("TestGenerator init failed; no candidates")
                    return False

            self._schema_candidates = [
                sc.schema_json for sc in self._test_generator.schema_candidates if sc.is_valid
            ]
            self._solution_candidates = [
                sol.code for sol in self._test_generator.solution_candidates if sol.is_valid
            ]
            
            if self.VERBOSE_OUTPUT:
                print(
                    f"[EM4C] candidates ready: {len(self._schema_candidates)} schemas, "
                    f"{len(self._solution_candidates)} solutions"
                )

            if not self._schema_candidates or not self._solution_candidates:
                if self.VERBOSE_OUTPUT:
                    print("not enough valid candidates")
                return False

            (best_schema, best_solution, best_alpha, best_schema_idx, 
             best_solution_idx, em4c_metadata, all_schema_results) = self._em4c_select_best_combination(
                schema_candidates=self._schema_candidates,
                solution_candidates=self._solution_candidates,
                problem=problem
            )
            
            if best_alpha < self.alpha_threshold:
                if self.VERBOSE_OUTPUT:
                    print(
                        f"EM4C alpha {best_alpha:.4f} < threshold {self.alpha_threshold:.4f}; "
                        "no valid simulation_code"
                    )
                self._current_schema = best_schema
                self._current_simulation_code = best_solution
                self._current_alpha = best_alpha
                self._selected_schema_index = best_schema_idx
                self._selected_solution_index = best_solution_idx
                self._em4c_metadata = em4c_metadata

                if best_schema_idx >= 0 and best_schema_idx < len(all_schema_results):
                    self._current_error_score = all_schema_results[best_schema_idx].get("error_ratio", 1.0)
                else:
                    self._current_error_score = 1.0
                self._current_eval_score = best_alpha

                self._save_em4c_intermediate_results(
                    problem=problem,
                    schema_candidates=self._schema_candidates,
                    solution_candidates=self._solution_candidates,
                    all_schema_results=all_schema_results,
                    selected_schema_index=best_schema_idx,
                    selected_solution_index=best_solution_idx,
                    alpha=best_alpha,
                    em4c_metadata=em4c_metadata
                )

                return False

            self._current_schema = best_schema
            self._current_simulation_code = best_solution
            self._current_alpha = best_alpha
            self._selected_schema_index = best_schema_idx
            self._selected_solution_index = best_solution_idx
            self._em4c_metadata = em4c_metadata

            if best_schema_idx >= 0 and best_schema_idx < len(all_schema_results):
                self._current_error_score = all_schema_results[best_schema_idx].get("error_ratio", 1.0)
            else:
                self._current_error_score = 1.0
            self._current_eval_score = best_alpha

            self._save_em4c_intermediate_results(
                problem=problem,
                schema_candidates=self._schema_candidates,
                solution_candidates=self._solution_candidates,
                all_schema_results=all_schema_results,
                selected_schema_index=best_schema_idx,
                selected_solution_index=best_solution_idx,
                alpha=best_alpha,
                em4c_metadata=em4c_metadata
            )
            
            if self.VERBOSE_OUTPUT:
                print(
                    f"[EM4C] init ok: alpha={self._current_alpha:.4f} (>= {self.alpha_threshold:.4f}), "
                    f"error_score={self._current_error_score:.3f}, "
                    f"schema_idx={self._selected_schema_index}, "
                    f"solution_idx={self._selected_solution_index}"
                )
            
            return True
            
        except Exception as e:
            if self.VERBOSE_OUTPUT:
                print(f"[EM4C] init error: {e}")
                import traceback
                traceback.print_exc()
            return False
    
    def _is_test_generator_valid(self) -> bool:
        """True if alpha and eval/error thresholds allow Rand/TLE."""
        if self._current_alpha < self.alpha_threshold:
            return False

        return (
            self._current_eval_score >= self.eval_score_threshold
            and self._current_error_score <= self.error_score_threshold
        )

    # --- EM4C ---
    
    def _build_execution_matrix(
        self,
        schema: str,
        solution_codes: List[str],
        problem: Problem,
        test_case_count: int = 100
    ) -> Tuple[List[List[Any]], float]:
        """Run schema-generated inputs on each solution; return (matrix, runtime-error ratio)."""
        self._apply_rand_data_range_config()
        
        fn_name = problem.metadata.get('func_name', None)
        matrix: List[List[Any]] = []
        total_executions = 0
        error_executions = 0.0
        
        for test_idx in range(test_case_count):
            try:
                test_input = generate_data_from_schema(schema)
                if not test_input:
                    continue

                row_results: List[Any] = []
                for sol_idx, solution_code in enumerate(solution_codes):
                    total_executions += 1
                    try:
                        success, output, error = run_code_capture(
                            fn_name=fn_name,
                            test_case=test_input,
                            code=solution_code,
                            timeout=self.single_test_timeout
                        )
                        
                        if success:
                            row_results.append(output)
                        elif error == "Time Limit Exceeded":
                            row_results.append("__TLE__")
                        else:
                            error_executions += 1
                            row_results.append(None)
                    except Exception as e:
                        error_executions += 1
                        row_results.append(None)
                
                matrix.append(row_results)

            except Exception as e:
                continue

        error_ratio = error_executions / total_executions if total_executions > 0 else 1.0
        
        return matrix, error_ratio
    
    def _em4c_select_best_combination(
        self,
        schema_candidates: List[str],
        solution_candidates: List[str],
        problem: Problem
    ) -> Tuple[str, str, float, int, int, Dict[str, Any], List[Dict[str, Any]]]:
        """Pick highest-alpha (schema, solution) via execution matrix + EM4C."""
        if not schema_candidates or not solution_candidates:
            return "", "", 0.0, -1, -1, {}, []
        
        all_schema_results: List[Dict[str, Any]] = []
        
        best_alpha = -1.0
        best_schema_idx = -1
        best_solution_idx = -1
        best_em4c_metadata: Dict[str, Any] = {}
        
        if self.VERBOSE_OUTPUT:
            print(f"[EM4C] evaluating {len(schema_candidates)} schemas")

        for schema_idx, schema in enumerate(schema_candidates):
            if self.VERBOSE_OUTPUT:
                print(f"[EM4C] schema {schema_idx + 1}/{len(schema_candidates)}")

            matrix, error_ratio = self._build_execution_matrix(
                schema=schema,
                solution_codes=solution_candidates,
                problem=problem,
                test_case_count=self.em4c_test_case_count
            )

            schema_result: Dict[str, Any] = {
                "schema_index": schema_idx,
                "schema": schema,
                "execution_matrix": matrix,
                "error_ratio": error_ratio,
                "selected_index": -1,
                "alpha": 0.0,
                "metadata": {}
            }

            if error_ratio > self.allowed_error_ratio:
                if self.VERBOSE_OUTPUT:
                    print(
                        f"[EM4C]   schema {schema_idx + 1} error_ratio {error_ratio:.3f} "
                        f"> {self.allowed_error_ratio}; skip"
                    )
                schema_result["alpha"] = -1.0
                all_schema_results.append(schema_result)
                continue

            if not matrix:
                if self.VERBOSE_OUTPUT:
                    print(f"[EM4C]   schema {schema_idx + 1} empty matrix; skip")
                schema_result["alpha"] = -1.0
                all_schema_results.append(schema_result)
                continue

            try:
                label_processor = Majority01Processor()
                confidence_calculator = DawidSkeneCalculator()
                
                selected_code, alpha, metadata = step6_calculate_confidence(
                    matrix=matrix,
                    codes=solution_candidates,
                    label_processor=label_processor,
                    confidence_calculator=confidence_calculator
                )

                selected_idx = solution_candidates.index(selected_code) if selected_code in solution_candidates else 0
                
                schema_result["selected_index"] = selected_idx
                schema_result["alpha"] = alpha
                schema_result["metadata"] = metadata or {}
                
                if self.VERBOSE_OUTPUT:
                    print(f"[EM4C]   Schema {schema_idx + 1}: alpha={alpha:.4f}, "
                          f"selected_solution={selected_idx}, error_ratio={error_ratio:.3f}")

                if alpha > best_alpha:
                    best_alpha = alpha
                    best_schema_idx = schema_idx
                    best_solution_idx = selected_idx
                    best_em4c_metadata = metadata or {}
                    
            except Exception as e:
                if self.VERBOSE_OUTPUT:
                    print(f"[EM4C]   schema {schema_idx + 1} EM4C failed: {e}")
                schema_result["alpha"] = -1.0
            
            all_schema_results.append(schema_result)
        
        if best_schema_idx >= 0 and best_solution_idx >= 0:
            best_schema = schema_candidates[best_schema_idx]
            best_solution = solution_candidates[best_solution_idx]
        else:
            best_schema = schema_candidates[0] if schema_candidates else ""
            best_solution = solution_candidates[0] if solution_candidates else ""
            best_schema_idx = 0
            best_solution_idx = 0
            best_alpha = 0.0
        
        if self.VERBOSE_OUTPUT:
            print(
                f"[EM4C] picked schema {best_schema_idx + 1}, "
                f"solution {best_solution_idx + 1}, alpha={best_alpha:.4f}"
            )
        
        return (best_schema, best_solution, best_alpha, best_schema_idx, 
                best_solution_idx, best_em4c_metadata, all_schema_results)
    
    def _save_em4c_intermediate_results(
        self,
        problem: Problem,
        schema_candidates: List[str],
        solution_candidates: List[str],
        all_schema_results: List[Dict[str, Any]],
        selected_schema_index: int,
        selected_solution_index: int,
        alpha: float,
        em4c_metadata: Dict[str, Any]
    ) -> None:
        """Persist EM4C artifacts under ``save_dir / question_id / EM4C / ...``."""
        if self._current_save_dir is None:
            if self.VERBOSE_OUTPUT:
                print("[EM4C] save_dir unset; skip writing intermediates")
            return

        question_dir = self._current_save_dir / str(problem.question_id)
        question_dir.mkdir(parents=True, exist_ok=True)

        em4c_dir = question_dir / "EM4C"
        em4c_dir.mkdir(parents=True, exist_ok=True)

        solutions_dir = em4c_dir / "bruteforce_solutions"
        solutions_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, solution_code in enumerate(solution_candidates):
            solution_file = solutions_dir / f"solution_{idx}.py"
            try:
                with open(solution_file, 'w', encoding='utf-8') as f:
                    f.write(solution_code)
            except Exception as e:
                if self.VERBOSE_OUTPUT:
                    print(f"[EM4C] failed to write solution_{idx}.py: {e}")

        tlefree_file = solutions_dir / "tlefree_evaluation.json"
        try:
            tlefree_data = {f"solution_{idx}": None for idx in range(len(solution_candidates))}
            with open(tlefree_file, 'w', encoding='utf-8') as f:
                json.dump(tlefree_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.VERBOSE_OUTPUT:
                print(f"[EM4C] failed to write tlefree_evaluation.json: {e}")

        for schema_result in all_schema_results:
            schema_idx = schema_result["schema_index"]
            schema_dir = em4c_dir / f"schema_{schema_idx}"
            schema_dir.mkdir(parents=True, exist_ok=True)

            schema_file = schema_dir / "schema.json"
            try:
                schema_content = schema_result.get("schema", "")
                if isinstance(schema_content, str):
                    try:
                        schema_obj = json.loads(schema_content)
                        with open(schema_file, 'w', encoding='utf-8') as f:
                            json.dump(schema_obj, f, ensure_ascii=False, indent=2)
                    except json.JSONDecodeError:
                        with open(schema_file, 'w', encoding='utf-8') as f:
                            f.write(schema_content)
                else:
                    with open(schema_file, 'w', encoding='utf-8') as f:
                        json.dump(schema_content, f, ensure_ascii=False, indent=2)
            except Exception as e:
                if self.VERBOSE_OUTPUT:
                    print(f"[EM4C] failed schema_{schema_idx}/schema.json: {e}")

            matrix_file = schema_dir / "execution_matrix.json"
            try:
                matrix_data = {
                    "matrix": schema_result.get("execution_matrix", []),
                    "error_ratio": schema_result.get("error_ratio", 0.0),
                    "num_test_cases": len(schema_result.get("execution_matrix", [])),
                    "num_solutions": len(solution_candidates)
                }
                with open(matrix_file, 'w', encoding='utf-8') as f:
                    json.dump(matrix_data, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                if self.VERBOSE_OUTPUT:
                    print(f"[EM4C] failed schema_{schema_idx}/execution_matrix.json: {e}")

            selected_file = schema_dir / "selected_solution.json"
            try:
                selected_data = {
                    "selected_index": schema_result.get("selected_index", -1),
                    "alpha": schema_result.get("alpha", 0.0),
                    "metadata": schema_result.get("metadata", {})
                }
                with open(selected_file, 'w', encoding='utf-8') as f:
                    json.dump(selected_data, f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                if self.VERBOSE_OUTPUT:
                    print(f"[EM4C] failed schema_{schema_idx}/selected_solution.json: {e}")

        metadata_file = question_dir / "metadata.json"
        try:
            metadata = {
                "schema": schema_candidates[selected_schema_index] if selected_schema_index >= 0 else "",
                "simulation_code": solution_candidates[selected_solution_index] if selected_solution_index >= 0 else "",
                "alpha": alpha,
                "selected_schema_index": selected_schema_index,
                "selected_solution_index": selected_solution_index,
                "em4c_metadata": em4c_metadata,
                "num_schema_candidates": len(schema_candidates),
                "num_solution_candidates": len(solution_candidates)
            }
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            if self.VERBOSE_OUTPUT:
                print(f"[EM4C] failed metadata.json: {e}")

        if self.VERBOSE_OUTPUT:
            print(f"[EM4C] wrote intermediates to {question_dir}")

    def _is_test_passed(self, passed_value: Optional[bool], default_value: bool = True) -> bool:
        """Treat None as ``default_value`` for pass/fail aggregation."""
        if passed_value is None:
            return default_value
        return bool(passed_value)

    def _trusttest_evaluate(self, code: str, problem: Problem) -> Dict[str, Any]:
        """Sample (public) -> Rand (schema) -> TLE."""
        results = {
            'Sample_passed': None,
            'Sample_test_string': "Untested",
            'Rand_passed': None,
            'Rand_test_string': "Untested",
            'TLE_passed': None,
            'TLE_test_string': "Untested"
        }
        try:
            results['Sample_passed'], results['Sample_test_string'] = evaluate(
                code, problem.public_test_cases, problem.metadata
            )
            if not results['Sample_passed']:
                return results
        except Exception as e:
            results['Sample_passed'] = None
            results['Sample_test_string'] = f"Untested, Sample Test Error: {str(e)}"
            return results
        
        if self._is_test_generator_valid() and self._current_schema:
            try:
                self._apply_rand_data_range_config()

                rand_passed, rand_test_string = self._run_rand_test(
                    code, problem,
                    test_case_count=self.test_case_count,
                    single_test_timeout=self.single_test_timeout,
                    total_generation_timeout=self.total_generation_timeout
                )
                results['Rand_passed'] = rand_passed
                results['Rand_test_string'] = rand_test_string
                
                if not results['Rand_passed']:
                    if self.VERBOSE_OUTPUT:
                        print(f"failed Some Rand tests in id{problem.question_id}")
                    return results
                    
            except Exception as e:
                results['Rand_passed'] = None
                results['Rand_test_string'] = f"Untested, Rand Test Error: {str(e)}"
                if self.VERBOSE_OUTPUT:
                    print(f"error Some Rand tests in id{problem.question_id}")
                return results
        else:
            results['Rand_passed'] = None
            if self._current_alpha < self.alpha_threshold:
                results['Rand_test_string'] = (
                    f"Untested, alpha={self._current_alpha:.4f} < {self.alpha_threshold:.4f}, "
                    f"No Valid Simulation Code"
                )
            else:
                results['Rand_test_string'] = (
                    f"Untested, error_score={self._current_error_score:.3f}, "
                    f"eval_score={self._current_eval_score:.3f}, No Valid Schema"
                )
            return results
        
        if self._is_test_generator_valid() and self._current_schema:
            try:
                self._apply_tle_data_range_config()

                tle_passed, tle_test_string = tle_evaluate(
                    code, self._current_schema, problem.metadata
                )
                results['TLE_passed'] = tle_passed
                results['TLE_test_string'] = tle_test_string
                
            except TimeoutError as e:
                results['TLE_passed'] = None
                results['TLE_test_string'] = f"Untested, Data Generation Timeout: {str(e)}"
            except RuntimeError as e:
                results['TLE_passed'] = None
                results['TLE_test_string'] = f"Untested, Data Generation Error: {str(e)}"
            except Exception as e:
                results['TLE_passed'] = None
                results['TLE_test_string'] = f"Untested, Unexpected Error: {str(e)}"
        else:
            results['TLE_passed'] = None
            if self._current_alpha < self.alpha_threshold:
                results['TLE_test_string'] = (
                    f"Untested, alpha={self._current_alpha:.4f} < {self.alpha_threshold:.4f}, "
                    f"No Valid Simulation Code"
                )
            else:
                results['TLE_test_string'] = (
                    f"Untested, error_score={self._current_error_score:.3f}, "
                    f"eval_score={self._current_eval_score:.3f}, No Valid Schema"
                )
        
        return results
    
    def _apply_rand_data_range_config(self) -> None:
        """Small ranges for Rand schema generation."""
        schema_default_config = schema_config_module.default_config
        schema_default_config.boundary_bias = False
        schema_default_config.default_range_limits = {
            'list': (1, 20),
            'matrix': (1, 20),
            'group': (1, 20),
        }
        schema_default_config.variable_default_limits = {
            'int': (-20, 20),
            'float': (-20.0, 20.0),
            'char': None,
            'string': None,
        }
    
    def _apply_tle_data_range_config(self) -> None:
        """Large ranges for TLE schema generation."""
        schema_default_config = schema_config_module.default_config
        schema_default_config.boundary_bias = True
        schema_default_config.default_range_limits = {
            'list': (1, int(1e6)),
            'matrix': (1, int(1e6)),
            'group': (1, int(1e6)),
        }
        schema_default_config.variable_default_limits = {
            'int': (int(-1e18+7), int(1e18-7)),
            'float': (-1e18+7, 1e18-7),
            'char': None,
            'string': None,
        }
    
    def _run_rand_test(
        self, 
        code: str, 
        problem: Problem,
        test_case_count: int = 100000,
        single_test_timeout: float = 0.8,
        total_generation_timeout: float = 60.0
    ) -> Tuple[bool, str]:
        """Random small tests vs simulation reference (schema-driven)."""
        if not self._current_schema or not self._current_simulation_code:
            return False, "No valid schema or simulation code"
        
        fn_name = problem.metadata.get('func_name', None)
        passed_count = 0
        failed_count = 0
        total_generated = 0
        start_time = time.time()
        
        for i in range(test_case_count):
            if time.time() - start_time > total_generation_timeout:
                break

            try:
                test_input = generate_data_from_schema(self._current_schema)
                if not test_input:
                    continue
                total_generated += 1

                sim_success, sim_output, sim_error = run_code_capture(
                    fn_name, test_input, self._current_simulation_code, timeout=single_test_timeout
                )
                
                if not sim_success and sim_error != "Time Limit Exceeded":
                    continue

                code_success, code_output, code_error = run_code_capture(
                    fn_name, test_input, code, timeout=single_test_timeout
                )
                
                if not code_success:
                    if code_error == "Time Limit Exceeded":
                        continue
                    else:
                        failed_count += 1
                        if failed_count >= 3:
                            return False, f"Failed at test {i+1}: {code_error}"
                        continue

                if fn_name:
                    if code_output != sim_output:
                        failed_count += 1
                        if failed_count >= 3:
                            return False, f"Wrong answer at test {i+1}"
                else:
                    if str(code_output).strip() != str(sim_output).strip():
                        failed_count += 1
                        if failed_count >= 3:
                            return False, f"Wrong answer at test {i+1}"
                
                passed_count += 1
                
            except Exception as e:
                continue
        
        elapsed_time = time.time() - start_time
        
        if total_generated == 0:
            return False, "Failed to generate any test cases"

        if self.VERBOSE_OUTPUT:
            print(
                f"   Rand stats: target={test_case_count}, generated={total_generated}, "
                f"passed={passed_count}, failed={failed_count}, "
                f"time={elapsed_time:.2f}s/{total_generation_timeout}s"
            )
        
        if failed_count > 0:
            return False, f"Passed {passed_count}/{total_generated}, Failed {failed_count}"
        
        return True, f"Passed all {passed_count} tests"
    
    def _optimization_one_iteration(
        self,
        optimizer: TextualGradientDescent,
        instance_var: Variable,
        problem_content: str,
        test_string: str
    ) -> None:
        """One TextGrad optimizer step from ``test_string`` feedback."""
        engine = self._get_engine()

        if self.VERBOSE_OUTPUT:
            print("      [1/3] LLM loss eval...")
        optimizer.zero_grad()
        loss_fn = CodeTestTimewithTests(engine=engine)
        test_time_loss = loss_fn(problem_content, instance_var, test_string)

        if self.VERBOSE_OUTPUT:
            print("      [2/3] backward...")
        test_time_loss.backward()

        if self.VERBOSE_OUTPUT:
            print("      [3/3] optimizer.step...")
        optimizer.step()

        if self.VERBOSE_OUTPUT:
            print("      iteration done")
    
    def _run_iteration_with_timeout(
        self,
        optimizer: TextualGradientDescent,
        instance_var: Variable,
        problem_content: str,
        test_string: str
    ) -> bool:
        """Run ``_optimization_one_iteration`` with ``iteration_timeout``."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                self._optimization_one_iteration,
                optimizer, instance_var, problem_content, test_string
            )
            try:
                future.result(timeout=self.iteration_timeout)
                return True
            except FuturesTimeoutError:
                if self.VERBOSE_OUTPUT:
                    print(f"      iteration timeout ({self.iteration_timeout}s); skip")
                return False
            except Exception as e:
                if self.VERBOSE_OUTPUT:
                    print(f"      iteration error: {e}")
                return False

    def _format_local_passed(self, test_results: Dict[str, Any]) -> str:
        """Serialize Sample/Rand/TLE pass flags for CSV."""
        def format_value(value: Optional[bool]) -> str:
            if value is None:
                return "None"
            return "True" if value else "False"
        
        sample_passed = format_value(test_results.get('Sample_passed'))
        rand_passed = format_value(test_results.get('Rand_passed'))
        tle_passed = format_value(test_results.get('TLE_passed'))
        
        return f"Sample_Passed: {sample_passed}, Rand_Passed: {rand_passed}, TLE_Passed: {tle_passed}"
    
    def _count_passed_stages(self, test_results: Dict[str, Any]) -> int:
        """Count how many of Sample/Rand/TLE passed (strict False for None)."""
        count = 0
        if self._is_test_passed(test_results.get('Sample_passed'), default_value=False):
            count += 1
        if self._is_test_passed(test_results.get('Rand_passed'), default_value=False):
            count += 1
        if self._is_test_passed(test_results.get('TLE_passed'), default_value=False):
            count += 1
        return count
    
    def _select_best_solution_index(self, results: List[Solution]) -> int:
        """Index of solution with most passed stages."""
        if not results:
            return 0
        
        def get_passed_count(sol: Solution) -> int:
            count = 0
            metadata = sol.metadata or {}
            if self._is_test_passed(metadata.get('Sample_passed'), default_value=False):
                count += 1
            if self._is_test_passed(metadata.get('Rand_passed'), default_value=False):
                count += 1
            if self._is_test_passed(metadata.get('TLE_passed'), default_value=False):
                count += 1
            return count
        
        best_idx = 0
        best_count = get_passed_count(results[0])
        
        for i in range(1, len(results)):
            current_count = get_passed_count(results[i])
            if current_count > best_count:
                best_count = current_count
                best_idx = i
        
        return best_idx
    
    def solve(self, problem: Problem, save_dir: Optional[Path] = None) -> List[Solution]:
        """Run EM4C init + TextGrad rounds; ``is_final`` on best stage coverage."""
        results = []

        self._current_save_dir = save_dir

        try:
            engine = self._get_engine()

            initial_code, is_normal_end = self._get_initial_code(problem)

            if not initial_code or not initial_code.strip():
                return [Solution(
                    code="",
                    problem_id=problem.question_id,
                    is_normal_end=False,
                    round_index=0,
                    is_final=True
                )]

            self._init_test_generator(problem)

            instance_var = Variable(
                initial_code,
                requires_grad=True,
                role_description=CODE_INSTANCE_ROLE_DESCRIPTION
            )

            optimizer = TextualGradientDescent(
                engine=engine,
                parameters=[instance_var],
                constraints=[
                    "Do not add asserts to the code",
                    "Code must contain imports"
                ]
            )

            test_results = self._trusttest_evaluate(instance_var.value, problem)

            if self.VERBOSE_OUTPUT:
                print(
                    f"[{problem.question_id}] initial eval: "
                    f"Sample={test_results['Sample_passed']}, "
                    f"Rand={test_results['Rand_passed']}, "
                    f"TLE={test_results['TLE_passed']}"
                )

            all_passed = (
                self._is_test_passed(test_results['Sample_passed']) and
                self._is_test_passed(test_results['Rand_passed']) and
                self._is_test_passed(test_results['TLE_passed'])
            )

            initial_solution = Solution(
                code=instance_var.value,
                problem_id=problem.question_id,
                is_normal_end=is_normal_end,
                round_index=0,
                is_final=False,
                local_passed=self._format_local_passed(test_results),
                local_result_type=(
                    f"Sample: {test_results['Sample_test_string']}, "
                    f"Rand: {test_results['Rand_test_string']}, "
                    f"TLE: {test_results['TLE_test_string']}"
                ),
                schema=self._current_schema,
                simulation_code=self._current_simulation_code,
                metadata={
                    'Sample_passed': test_results['Sample_passed'],
                    'Rand_passed': test_results['Rand_passed'],
                    'TLE_passed': test_results['TLE_passed'],
                    'error_score': self._current_error_score,
                    'eval_score': self._current_eval_score,
                    'alpha': self._current_alpha,
                    'selected_schema_index': self._selected_schema_index,
                    'selected_solution_index': self._selected_solution_index
                }
            )
            results.append(initial_solution)

            for iter_idx in range(self.max_iters):
                if all_passed:
                    if self.VERBOSE_OUTPUT:
                        print(f"[{problem.question_id}] all stages passed; stop")
                    break

                if self.VERBOSE_OUTPUT:
                    print(
                        f"[{problem.question_id}] iter {iter_idx + 1}: "
                        f"Sample={test_results['Sample_passed']}, "
                        f"Rand={test_results['Rand_passed']}, "
                        f"TLE={test_results['TLE_passed']}"
                    )

                failed_test_string = None
                if not self._is_test_passed(test_results['Sample_passed']):
                    failed_test_string = test_results['Sample_test_string']
                elif not self._is_test_passed(test_results['Rand_passed']):
                    failed_test_string = test_results['Rand_test_string']
                elif not self._is_test_passed(test_results['TLE_passed']):
                    failed_test_string = test_results['TLE_test_string']
                
                if failed_test_string is None:
                    break

                iteration_success = self._run_iteration_with_timeout(
                    optimizer,
                    instance_var,
                    problem.question_content,
                    failed_test_string
                )

                if not iteration_success:
                    continue

                test_results = self._trusttest_evaluate(instance_var.value, problem)

                all_passed = (
                    self._is_test_passed(test_results['Sample_passed']) and
                    self._is_test_passed(test_results['Rand_passed']) and
                    self._is_test_passed(test_results['TLE_passed'])
                )

                round_solution = Solution(
                    code=instance_var.value,
                    problem_id=problem.question_id,
                    is_normal_end=True,
                    round_index=iter_idx + 1,
                    is_final=False,
                    local_passed=self._format_local_passed(test_results),
                    local_result_type=(
                        f"Sample: {test_results['Sample_test_string']}, "
                        f"Rand: {test_results['Rand_test_string']}, "
                        f"TLE: {test_results['TLE_test_string']}"
                    ),
                    schema=self._current_schema,
                    simulation_code=self._current_simulation_code,
                    metadata={
                        'Sample_passed': test_results['Sample_passed'],
                        'Rand_passed': test_results['Rand_passed'],
                        'TLE_passed': test_results['TLE_passed'],
                        'error_score': self._current_error_score,
                        'eval_score': self._current_eval_score,
                        'alpha': self._current_alpha,
                        'selected_schema_index': self._selected_schema_index,
                        'selected_solution_index': self._selected_solution_index
                    }
                )
                results.append(round_solution)
            
            if results:
                best_idx = self._select_best_solution_index(results)
                results[best_idx].is_final = True
                
                if self.VERBOSE_OUTPUT:
                    best_metadata = results[best_idx].metadata or {}
                    print(
                        f"[{problem.question_id}] best round={results[best_idx].round_index}, "
                        f"Sample={best_metadata.get('Sample_passed')}, "
                        f"Rand={best_metadata.get('Rand_passed')}, "
                        f"TLE={best_metadata.get('TLE_passed')}, "
                        f"alpha={best_metadata.get('alpha', 0.0):.4f}"
                    )
            
            return results
            
        except Exception as e:
            print(f"TrustTestSolver.solve() error {problem.question_id}: {e}")
            import traceback
            traceback.print_exc()
            return [Solution(
                code="",
                problem_id=problem.question_id,
                is_normal_end=False,
                round_index=0,
                is_final=True
            )]
