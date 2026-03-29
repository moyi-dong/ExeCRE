"""
Schema Oracle test generator.

Phase 1: LLM schemas + brute-force solutions (validated). Phase 2: run cross-checks, majority-vote best pair.
Phase 3: sample inputs from best schema, oracle output from best solution → ``TestCase``.

Cache: ``results/schemas/{benchmark}/{model}/{question_id}/`` with ``schemas/``, ``solutions/``, ``candidates_artifact.json``.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import Counter

from ..base_generator import BaseTestGenerator, TestCase
from src.core.problem import Problem
from .config import SchemaOracleConfig, default_config

from src.input_generators.schema.prompts import LLM_SCHEMA_PROMPT
from src.input_generators.schema.schema import generate_data_from_schema
from src.input_generators.schema import config as schema_config_module  # mutates range limits for sampling

from src.baselines.bruteforce_solve.prompts import (
    SYSTEM_PROMPT_FOR_SIMULATION_CODE,
    get_simulation_question_template_answer
)
from src.utils import extract_code

from src.evaluators.lcb_coderunner import run_code_capture

if TYPE_CHECKING:
    from src.config.experiment_config import ExperimentConfig


@dataclass
class SchemaCandidate:
    """One schema candidate."""

    schema_json: str
    index: int
    is_valid: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionCandidate:
    """One solution candidate."""
    code: str
    index: int
    is_valid: bool = False
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionRunResult:
    """Run outcome for one solution on one input."""
    solution_index: int
    success: bool
    output: Any
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "solution_index": self.solution_index,
            "success": self.success,
            "output": self.output,
            "error": self.error
        }


@dataclass
class VotingTestCase:
    """Per-input votes across solutions."""
    test_input: str
    results: List[SolutionRunResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_input": self.test_input,
            "results": [r.to_dict() for r in self.results]
        }


@dataclass
class SchemaVotingDetail:
    """Vote log for one schema."""
    schema_index: int
    test_cases: List[VotingTestCase] = field(default_factory=list)
    best_solution_index: int = -1
    best_score: float = 0.0
    error_ratio: float = 0.0
    actual_test_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_index": self.schema_index,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
            "best_solution_index": self.best_solution_index,
            "best_score": self.best_score,
            "error_ratio": self.error_ratio,
            "actual_test_count": self.actual_test_count
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaVotingDetail":
        detail = cls(schema_index=data["schema_index"])
        detail.best_solution_index = data.get("best_solution_index", -1)
        detail.best_score = data.get("best_score", 0.0)
        detail.error_ratio = data.get("error_ratio", 0.0)
        detail.actual_test_count = data.get("actual_test_count", 0)
        
        for tc_data in data.get("test_cases", []):
            results = [
                SolutionRunResult(
                    solution_index=r["solution_index"],
                    success=r["success"],
                    output=r["output"],
                    error=r.get("error")
                )
                for r in tc_data.get("results", [])
            ]
            detail.test_cases.append(VotingTestCase(
                test_input=tc_data["test_input"],
                results=results
            ))
        
        return detail


@dataclass
class CandidatesArtifact:
    """Serializable cache: candidates plus phase-2 best pair and scores."""
    schema_candidates: List[SchemaCandidate] = field(default_factory=list)
    solution_candidates: List[SolutionCandidate] = field(default_factory=list)
    best_schema: Optional[str] = None
    best_solution: Optional[str] = None
    error_score: float = 1.0
    eval_score: float = 0.0
    avg_test_cases: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_candidates": [
                {
                    "schema_json": sc.schema_json,
                    "index": sc.index,
                    "is_valid": sc.is_valid,
                    "metadata": sc.metadata
                }
                for sc in self.schema_candidates
            ],
            "solution_candidates": [
                {
                    "code": sol.code,
                    "index": sol.index,
                    "is_valid": sol.is_valid,
                    **({"passed": sol.passed} if sol.passed is not None else {}),
                    "metadata": sol.metadata
                }
                for sol in self.solution_candidates
            ],
            "best_schema": self.best_schema,
            "best_solution": self.best_solution,
            "error_score": self.error_score,
            "eval_score": self.eval_score,
            "avg_test_cases": self.avg_test_cases
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CandidatesArtifact":
        artifact = cls()

        # schema candidates
        for sc_data in data.get("schema_candidates", []):
            sc = SchemaCandidate(
                schema_json=sc_data["schema_json"],
                index=sc_data["index"],
                is_valid=sc_data.get("is_valid", False),
                metadata=sc_data.get("metadata", {})
            )
            artifact.schema_candidates.append(sc)

        # solution candidates
        for sol_data in data.get("solution_candidates", []):
            sol = SolutionCandidate(
                code=sol_data["code"],
                index=sol_data["index"],
                is_valid=sol_data.get("is_valid", False),
                passed=sol_data.get("passed"),
                metadata=sol_data.get("metadata", {})
            )
            artifact.solution_candidates.append(sol)
        
        artifact.best_schema = data.get("best_schema")
        artifact.best_solution = data.get("best_solution")
        artifact.error_score = data.get("error_score", 1.0)
        artifact.eval_score = data.get("eval_score", 0.0)
        artifact.avg_test_cases = data.get("avg_test_cases", 0.0)
        
        return artifact


class SchemaOracleTestGenerator(BaseTestGenerator):
    """Many schemas × many brute-force solutions → majority-vote best pair → ``generate()`` samples tests.

    Config merge order: ``kwargs`` > ``schema_oracle_config`` > ``ExperimentConfig`` > defaults.
    Cache: ``results/schemas/{benchmark}/{model}/{question_id}/``.
    """

    ARTIFACT_FILENAME = "candidates_artifact.json"
    VERBOSE_OUTPUT = True

    def __init__(
        self,
        engine: Any = None,
        config: Optional["ExperimentConfig"] = None,
        schema_oracle_config: Optional[SchemaOracleConfig] = None,
        **kwargs
    ):
        """``engine`` required; optional ``config`` / ``schema_oracle_config``; ``kwargs`` override fields."""
        super().__init__(name="schema_oracle", engine=engine, config=config, **kwargs)

        self.schema_oracle_config = self._init_schema_oracle_config(
            config, schema_oracle_config, kwargs
        )

        self._candidates_artifact: Optional[CandidatesArtifact] = None
    
    def _init_schema_oracle_config(
        self,
        experiment_config: Optional["ExperimentConfig"],
        schema_oracle_config: Optional[SchemaOracleConfig],
        kwargs: Dict[str, Any]
    ) -> SchemaOracleConfig:
        """Merge config sources; ``kwargs`` applied last via ``update()``."""
        if schema_oracle_config is not None:
            config = schema_oracle_config
        elif experiment_config is not None:
            config = SchemaOracleConfig.from_experiment_config(experiment_config)
        else:
            config = SchemaOracleConfig()

        if kwargs:
            config.update(**kwargs)

        return config

    def _get_cache_key(self, problem: Problem) -> str:
        """``{benchmark}/{model}/{question_id}``."""
        model_name = self.engine.model_string if self.engine else "unknown"
        return f"{problem.benchmark}/{model_name}/{problem.question_id}"
    
    def get_cache_path(self, problem: Problem) -> Optional[Path]:
        """Default under ``SCHEMA_ORACLE_CACHE_ROOT``; or ``cache_dir / name / cache_key`` when batch tools set it."""
        raw = getattr(self, "raw_cache_dir", False)
        base = getattr(self, "cache_dir", None)
        if base is not None and raw:
            return Path(base) / self.name / self._get_cache_key(problem)
        from .paths import SCHEMA_ORACLE_CACHE_ROOT
        model_name = self.engine.model_string if self.engine else "unknown"
        return SCHEMA_ORACLE_CACHE_ROOT / problem.benchmark / model_name / problem.question_id
    
    def initialize(self, problem: Problem, force_recalculate: bool = False) -> bool:
        """Phases 1–2 for ``problem``: load/supplement cache or generate; optionally rerun phase 2 only."""
        if self.engine is None:
            print("[schema_generator:SchemaOracleTestGenerator] error: no LLM engine")
            return False

        self._problem = problem

        if self._artifact_exists(problem):
            cached_data = self._load_artifact(problem)
            if cached_data:
                try:
                    self._candidates_artifact = CandidatesArtifact.from_dict(cached_data)
                    self._artifact = cached_data
                    
                    if self.VERBOSE_OUTPUT:
                        print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] cache load: "
                              f"{len(self._candidates_artifact.schema_candidates)} schemas, "
                              f"{len(self._candidates_artifact.solution_candidates)} solutions")

                    supplemented = self._supplement_candidates(problem)

                    if supplemented:
                        self._execute_phase2_and_save(problem)
                        return True

                    if self._candidates_artifact.best_schema is not None and not force_recalculate:
                        self._initialized = True
                        if self.VERBOSE_OUTPUT:
                            print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] cache ok: "
                                  f"error_score={self._candidates_artifact.error_score:.3f}, "
                                  f"eval_score={self._candidates_artifact.eval_score:.3f}")
                        return True
                    else:
                        if self.VERBOSE_OUTPUT:
                            if force_recalculate:
                                print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] force phase2...")
                            else:
                                print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] no phase2 in cache, running...")
                        self._execute_phase2_and_save(problem)
                        return True

                except Exception as e:
                    print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] cache load failed: {e}")

        try:
            artifact = self._phase1_get_candidates(problem)
            
            if not artifact.schema_candidates and not artifact.solution_candidates:
                print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] no valid candidates")
                return False

            self._candidates_artifact = artifact

            if self.VERBOSE_OUTPUT:
                print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] phase1 done: "
                      f"{len(artifact.schema_candidates)} schemas, "
                      f"{len(artifact.solution_candidates)} solutions")

            self._save_schemas_separately(problem, artifact.schema_candidates)
            self._save_solutions_separately(problem, artifact.solution_candidates)

            self._execute_phase2_and_save(problem)

            return True

        except Exception as e:
            print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] init failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _execute_phase2_and_save(self, problem: Problem) -> None:
        """Run phase 2 and persist artifact."""
        if self.VERBOSE_OUTPUT:
            print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] phase2: majority voting...")

        best_schema, best_solution, error_score, eval_score, avg_test_cases, error_msg = self._phase2_select_best_candidates()

        self._candidates_artifact.best_schema = best_schema
        self._candidates_artifact.best_solution = best_solution
        self._candidates_artifact.error_score = error_score
        self._candidates_artifact.eval_score = eval_score
        self._candidates_artifact.avg_test_cases = avg_test_cases

        self._artifact = self._candidates_artifact.to_dict()
        self._save_artifact(problem, self._artifact)

        self._initialized = True

        if self.VERBOSE_OUTPUT:
            print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] phase2 done: "
                  f"error_score={error_score:.3f}, eval_score={eval_score:.3f}")
    
    def _phase1_get_candidates(self, problem: Problem) -> CandidatesArtifact:
        """Build fresh schema and solution candidate lists."""
        artifact = CandidatesArtifact()
        artifact.schema_candidates = self._generate_schema_candidates(problem)
        artifact.solution_candidates = self._generate_solution_candidates(problem)
        return artifact

    def _supplement_candidates(self, problem: Problem) -> bool:
        """Top up cached candidates to configured counts; clears phase-2 fields. Returns True if phase 2 must rerun."""
        if self._candidates_artifact is None:
            return False
        
        required_schemas = self.schema_oracle_config.max_schema_candidates
        required_solutions = self.schema_oracle_config.max_simulation_candidates
        
        current_schemas = len(self._candidates_artifact.schema_candidates)
        current_solutions = len(self._candidates_artifact.solution_candidates)
        
        schema_deficit = max(0, required_schemas - current_schemas)
        solution_deficit = max(0, required_solutions - current_solutions)
        
        if schema_deficit == 0 and solution_deficit == 0:
            return False
        
        if self.VERBOSE_OUTPUT:
            print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] supplement: "
                  f"schema +{schema_deficit} (have {current_schemas}, need {required_schemas}), "
                  f"solution +{solution_deficit} (have {current_solutions}, need {required_solutions})")

        if schema_deficit > 0:
            new_schemas = self._generate_schema_candidates(
                problem, 
                count=schema_deficit, 
                start_index=current_schemas
            )
            self._candidates_artifact.schema_candidates.extend(new_schemas)
            
            self._save_schemas_separately(problem, new_schemas)

            if self.VERBOSE_OUTPUT:
                print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] added {len(new_schemas)} schemas")

        if solution_deficit > 0:
            new_solutions = self._generate_solution_candidates(
                problem, 
                count=solution_deficit, 
                start_index=current_solutions
            )
            self._candidates_artifact.solution_candidates.extend(new_solutions)
            
            self._save_solutions_separately(problem, new_solutions)

            if self.VERBOSE_OUTPUT:
                print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] added {len(new_solutions)} solutions")

        self._candidates_artifact.best_schema = None
        self._candidates_artifact.best_solution = None
        self._candidates_artifact.error_score = 1.0
        self._candidates_artifact.eval_score = 0.0
        
        if self.VERBOSE_OUTPUT:
            print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] supplement done: "
                  f"{len(self._candidates_artifact.schema_candidates)} schemas, "
                  f"{len(self._candidates_artifact.solution_candidates)} solutions")
        
        return True
    
    def _generate_schema_candidates(
        self, 
        problem: Problem,
        count: Optional[int] = None,
        start_index: int = 0
    ) -> List[SchemaCandidate]:
        """LLM schemas; keep those that yield sample data via ``generate_data_from_schema``."""
        max_candidates = count if count is not None else self.schema_oracle_config.max_schema_candidates
        max_retry = self.schema_oracle_config.max_schema_retry
        
        valid_candidates = []
        
        for i in range(max_candidates):
            actual_index = start_index + i
            schema = None
            
            for attempt in range(max_retry):
                try:
                    schema_response = self._call_llm_for_schema(problem)

                    if not schema_response:
                        continue

                    schema_json = self._process_schema_response(schema_response, problem)

                    if not schema_json:
                        continue

                    test_data = generate_data_from_schema(schema_json)

                    if test_data:
                        candidate = SchemaCandidate(
                            schema_json=schema_json,
                            index=actual_index,
                            is_valid=True,
                            metadata={"attempts": attempt + 1}
                        )
                        valid_candidates.append(candidate)
                        print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] schema {actual_index+1} ok (try {attempt+1})")
                        break

                except Exception as e:
                    if attempt == max_retry - 1:
                        print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] schema {actual_index+1} failed: {e}")
                    continue
        
        return valid_candidates
    
    def _call_llm_for_schema(self, problem: Problem) -> Optional[str]:
        """Raw LLM response for schema JSON."""
        full_prompt = LLM_SCHEMA_PROMPT + "\n\n" + problem.question_content
        response = self.engine.generate(full_prompt, temperature=self.sampling_temperature)
        
        return response
    
    def _process_schema_response(self, schema_response: str, problem: Problem) -> Optional[str]:
        """Strip fences, parse JSON, set ``input_format`` from ``func_name``."""
        try:
            schema_str = schema_response.strip()

            if schema_str.startswith('```'):
                schema_str = re.sub(r'^```(?:json)?\s*', '', schema_str)
                schema_str = re.sub(r'\s*```$', '', schema_str)

            schema_str = schema_str.strip()

            schema_dict = json.loads(schema_str)

            func_name = problem.metadata.get("func_name", None)
            if func_name and str(func_name).strip():
                schema_dict["input_format"] = "function_args"
            else:
                schema_dict["input_format"] = "stdin"

            return json.dumps(schema_dict, ensure_ascii=False, indent=2)

        except json.JSONDecodeError as e:
            print(f"[schema_generator:SchemaOracleTestGenerator] schema JSON parse error: {str(e)}")
            return None
        except Exception as e:
            print(f"[schema_generator:SchemaOracleTestGenerator] schema process error: {str(e)}")
            return None
    
    def _generate_solution_candidates(
        self, 
        problem: Problem,
        count: Optional[int] = None,
        start_index: int = 0
    ) -> List[SolutionCandidate]:
        """Brute-force-style solutions (same prompts as BruteforceSolver); uses ``sampling_temperature``."""
        max_candidates = count if count is not None else self.schema_oracle_config.max_simulation_candidates
        max_retry = self.schema_oracle_config.max_schema_retry
        
        valid_candidates = []
        
        for i in range(max_candidates):
            actual_index = start_index + i
            for attempt in range(max_retry):
                try:
                    solution_code = self._call_llm_for_solution(problem)
                    
                    if solution_code and len(solution_code.strip()) > 0:
                        candidate = SolutionCandidate(
                            code=solution_code,
                            index=actual_index,
                            is_valid=True,
                            metadata={"attempts": attempt + 1}
                        )
                        valid_candidates.append(candidate)
                        print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] solution {actual_index+1} ok (try {attempt+1})")
                        break

                except Exception as e:
                    if attempt == max_retry - 1:
                        print(f"[schema_generator:SchemaOracleTestGenerator][{problem.question_id}] solution {actual_index+1} failed: {e}")
                    continue
        
        return valid_candidates
    
    def _call_llm_for_solution(self, problem: Problem) -> Optional[str]:
        """BruteforceSolver-style generation → extracted Python."""
        system_prompt = SYSTEM_PROMPT_FOR_SIMULATION_CODE
        user_prompt = get_simulation_question_template_answer(problem)

        model_output = self.engine.generate(
            user_prompt,
            system_prompt=system_prompt,
            temperature=self.sampling_temperature
        )

        solution_code = self._clean_code_blocks(model_output)

        if len(solution_code) == 0:
            solution_code = extract_code(model_output)
        
        return solution_code
    
    def _clean_code_blocks(self, text: str) -> str:
        """Extract ```python``` body, or strip opening fence if truncated."""
        text = text.strip()

        pattern = r'```(?:python|py)?\s*(.*?)\s*```'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

        if text.startswith('```python'):
            return text[len('```python'):].strip()
        if text.startswith('```py'):
            return text[len('```py'):].strip()
        if text.startswith('```'):
            return text[3:].strip()
        
        return text
    
    def _save_schemas_separately(
        self, 
        problem: Problem, 
        schema_candidates: List[SchemaCandidate]
    ) -> None:
        """Write each schema JSON under ``schemas/`` in cache (debug)."""
        cache_path = self.get_cache_path(problem)
        if cache_path is None:
            return
        
        schemas_dir = cache_path / "schemas"
        schemas_dir.mkdir(parents=True, exist_ok=True)
        
        for candidate in schema_candidates:
            schema_path = schemas_dir / f"schema_{candidate.index}.json"
            try:
                with open(schema_path, 'w', encoding='utf-8') as f:
                    f.write(candidate.schema_json)
            except Exception as e:
                print(f"[schema_generator:SchemaOracleTestGenerator] save schema failed: {e}")
    
    def _save_solutions_separately(
        self, 
        problem: Problem, 
        solution_candidates: List[SolutionCandidate]
    ) -> None:
        """Write each solution under ``solutions/`` in cache (debug)."""
        cache_path = self.get_cache_path(problem)
        if cache_path is None:
            return
        
        solutions_dir = cache_path / "solutions"
        solutions_dir.mkdir(parents=True, exist_ok=True)
        
        for candidate in solution_candidates:
            solution_path = solutions_dir / f"solution_{candidate.index}.py"
            try:
                with open(solution_path, 'w', encoding='utf-8') as f:
                    f.write(candidate.code)
            except Exception as e:
                print(f"[schema_generator:SchemaOracleTestGenerator] save solution failed: {e}")

    def _phase2_select_best_candidates(self) -> Tuple[str, str, float, float, float, str]:
        """Cross-run solutions on schema-sampled inputs; scored majority vote picks best pair."""
        if self._candidates_artifact is None:
            return "", "", 1.0, 0.0, 0.0, "no artifact"
        
        schema_candidates = [sc.schema_json for sc in self._candidates_artifact.schema_candidates if sc.is_valid]
        solution_candidates = [sol.code for sol in self._candidates_artifact.solution_candidates if sol.is_valid]
        
        test_case_count = self.schema_oracle_config.test_case_count
        single_test_timeout = self.schema_oracle_config.single_test_timeout
        total_generation_timeout = self.schema_oracle_config.total_generation_timeout
        allowed_error_ratio = self.schema_oracle_config.allowed_error_ratio
        save_voting_details = self.schema_oracle_config.save_voting_details
        
        if self.VERBOSE_OUTPUT:
            print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}] phase2: "
                  f"{len(schema_candidates)} schemas, {len(solution_candidates)} sols, "
                  f"{test_case_count} tests, mode={self.schema_oracle_config.data_range_mode}")

        self._apply_data_range_config()

        if not schema_candidates or not solution_candidates:
            if self.VERBOSE_OUTPUT:
                print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}] no valid schema/solution")
            return "", "", 1.0, 0.0, 0.0, "no valid schema and solution"
        
        start_time = time.time()
        
        per_schema_timeout = total_generation_timeout / len(schema_candidates)

        if self.VERBOSE_OUTPUT:
            print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}] per-schema timeout: {per_schema_timeout:.1f}s")

        best_solution_for_schema = [None] * len(schema_candidates)
        best_score_for_schema = [0.0] * len(schema_candidates)
        schema_error_ratios = [0.0] * len(schema_candidates)
        schema_test_counts = [0] * len(schema_candidates)

        voting_details: List[SchemaVotingDetail] = []

        fn_name = self._problem.metadata.get('func_name', None) if self._problem else None

        for schema_idx, schema in enumerate(schema_candidates):
            if time.time() - start_time > total_generation_timeout:
                if self.VERBOSE_OUTPUT:
                    print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}] global timeout, stop")
                break

            if self.VERBOSE_OUTPUT:
                print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}] schema {schema_idx+1}/{len(schema_candidates)}")
            
            solution_scores = [0.0] * len(solution_candidates)
            schema_start_time = time.time()
            real_test_cases = 0
            error_test_cases = 0.0
            
            current_voting_detail = SchemaVotingDetail(schema_index=schema_idx)

            for test_idx in range(test_case_count):
                if time.time() - schema_start_time > per_schema_timeout:
                    if self.VERBOSE_OUTPUT:
                        print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}]   schema {schema_idx+1} timeout")
                    break

                real_test_cases += 1

                if self.VERBOSE_OUTPUT and (test_idx == 0 or test_idx == test_case_count - 1 or
                                           (test_case_count > 10 and test_idx % (test_case_count // 10) == 0)):
                    print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}]     progress {test_idx+1}/{test_case_count} ({(test_idx+1)/test_case_count*100:.0f}%)")

                try:
                    test_input = generate_data_from_schema(schema)
                    if not test_input:
                        continue
                except Exception as e:
                    continue

                solution_results = []
                solution_errors = []
                solution_successes = []
                
                for i, solution in enumerate(solution_candidates):
                    success, output, error = self._run_solution_on_input(
                        solution, test_input, fn_name, single_test_timeout
                    )
                    
                    # TLE not counted as run error for error_ratio
                    if success or error == "Time Limit Exceeded":
                        pass
                    else:
                        error_test_cases += 1.0 / len(solution_candidates)
                    
                    solution_results.append(output)
                    solution_errors.append(error)
                    solution_successes.append(success)
                
                if save_voting_details:
                    voting_test_case = VotingTestCase(
                        test_input=test_input,
                        results=[
                            SolutionRunResult(
                                solution_index=i,
                                success=solution_successes[i],
                                output=solution_results[i],
                                error=solution_errors[i]
                            )
                            for i in range(len(solution_candidates))
                        ]
                    )
                    current_voting_detail.test_cases.append(voting_test_case)
                
                hashed_results = [self._make_hashable(r) for r in solution_results]
                result_counter = Counter(hashed_results)
                
                for i, h in enumerate(hashed_results):
                    if solution_results[i] is not None and solution_results[i] != "":
                        solution_scores[i] += result_counter[h] / len(hashed_results)
            
            current_error_ratio = error_test_cases / real_test_cases if real_test_cases > 0 else 0.0
            schema_error_ratios[schema_idx] = current_error_ratio
            schema_test_counts[schema_idx] = real_test_cases
            
            if self.VERBOSE_OUTPUT:
                print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}]   schema {schema_idx+1} done: "
                      f"{real_test_cases} cases, err_ratio {current_error_ratio:.3f}")

            if current_error_ratio > allowed_error_ratio:
                if self.VERBOSE_OUTPUT:
                    print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}]   schema {schema_idx+1} rejected (high error)")
                best_score_for_schema[schema_idx] = -1.0
                best_solution_for_schema[schema_idx] = solution_candidates[0] if solution_candidates else None
                
                current_voting_detail.best_solution_index = 0
                current_voting_detail.best_score = -1.0
                current_voting_detail.error_ratio = current_error_ratio
                current_voting_detail.actual_test_count = real_test_cases
                voting_details.append(current_voting_detail)
                
                if self.VERBOSE_OUTPUT:
                    print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}]   schema {schema_idx+1} best score: -1.000 (rejected)")
                continue

            if real_test_cases > 0:
                solution_scores = [score / real_test_cases for score in solution_scores]
            
            best_idx = 0
            if solution_scores:
                best_score_for_schema[schema_idx] = max(solution_scores)
                best_idx = solution_scores.index(max(solution_scores))
                best_solution_for_schema[schema_idx] = solution_candidates[best_idx]
            
            current_voting_detail.best_solution_index = best_idx
            current_voting_detail.best_score = best_score_for_schema[schema_idx]
            current_voting_detail.error_ratio = current_error_ratio
            current_voting_detail.actual_test_count = real_test_cases
            voting_details.append(current_voting_detail)
            
            if self.VERBOSE_OUTPUT:
                print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}]   schema {schema_idx+1} best score: {best_score_for_schema[schema_idx]:.3f}")

        if save_voting_details and voting_details and self._problem:
            self._save_voting_details(self._problem, voting_details)
            print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}] voting_details/ saved")

        tested_schemas = [c for c in schema_test_counts if c > 0]
        avg_test_cases = sum(tested_schemas) / len(tested_schemas) if tested_schemas else 0.0
        
        if not best_score_for_schema or not schema_candidates:
            if self.VERBOSE_OUTPUT:
                print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}] no valid schema after test")
            return "", "", 1.0, 0.0, avg_test_cases, "no valid schema after testing"
        
        best_schema_idx = best_score_for_schema.index(max(best_score_for_schema))
        best_schema = schema_candidates[best_schema_idx]
        best_solution = best_solution_for_schema[best_schema_idx] or ""
        best_score = best_score_for_schema[best_schema_idx]
        best_error_ratio = schema_error_ratios[best_schema_idx]
        
        if self.VERBOSE_OUTPUT:
            print(f"[schema_generator:SchemaOracleTestGenerator][{self._problem.question_id}] pick schema {best_schema_idx+1}: "
                  f"score {best_score:.3f}, err {best_error_ratio:.3f}, avg_cases {avg_test_cases:.1f}")

        error_score = best_error_ratio
        eval_score = best_score
        error_msg = f"error_ratio: {best_error_ratio:.3f}"
        
        return best_schema, best_solution, error_score, eval_score, avg_test_cases, error_msg
    
    def _run_solution_on_input(
        self,
        solution_code: str,
        test_input: str,
        fn_name: Optional[str],
        timeout: float
    ) -> Tuple[bool, Any, Optional[str]]:
        """``run_code_capture`` wrapper."""
        try:
            success, output, error = run_code_capture(
                fn_name=fn_name,
                test_case=test_input,
                code=solution_code,
                timeout=timeout,
            )
            return success, output, error
        except Exception as e:
            return False, '', str(e)
    
    @staticmethod
    def _make_hashable(obj: Any) -> Any:
        """Stable hash key for outputs (nested containers → tuples)."""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return tuple(SchemaOracleTestGenerator._make_hashable(x) for x in obj)
        elif isinstance(obj, dict):
            return tuple(sorted((k, SchemaOracleTestGenerator._make_hashable(v)) for k, v in obj.items()))
        else:
            return str(obj)
    
    def _save_voting_details(
        self, 
        problem: Problem, 
        voting_details: List[SchemaVotingDetail]
    ) -> None:
        """Persist per-schema vote JSON under ``voting_details/``."""
        cache_path = self.get_cache_path(problem)
        if cache_path is None:
            return
        
        voting_dir = cache_path / "voting_details"
        voting_dir.mkdir(parents=True, exist_ok=True)
        
        for detail in voting_details:
            voting_file = voting_dir / f"schema_{detail.schema_index}.json"
            try:
                with open(voting_file, 'w', encoding='utf-8') as f:
                    json.dump(detail.to_dict(), f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[schema_generator:SchemaOracleTestGenerator] save voting_details failed (schema_{detail.schema_index}): {e}")

    def _load_voting_details(self, problem: Problem) -> List[SchemaVotingDetail]:
        """Load ``voting_details/schema_*.json`` if present."""
        cache_path = self.get_cache_path(problem)
        if cache_path is None:
            return []
        
        voting_dir = cache_path / "voting_details"
        if not voting_dir.exists():
            return []
        
        voting_details = []
        for voting_file in sorted(voting_dir.glob("schema_*.json")):
            try:
                with open(voting_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                detail = SchemaVotingDetail.from_dict(data)
                voting_details.append(detail)
            except Exception as e:
                print(f"[schema_generator:SchemaOracleTestGenerator] load voting_details failed ({voting_file.name}): {e}")

        return voting_details

    def _apply_data_range_config(self) -> None:
        """Push ``data_range_mode`` into ``schema_config_module.default_config`` (global side effect)."""
        mode = self.schema_oracle_config.data_range_mode
        schema_default_config = schema_config_module.default_config

        if mode == SchemaOracleConfig.MODE_LARGE:
            schema_default_config.default_range_limits = self.schema_oracle_config.large_range_limits.copy()
            schema_default_config.variable_default_limits = self.schema_oracle_config.large_variable_limits.copy()
        elif mode == SchemaOracleConfig.MODE_RAND:
            schema_default_config.default_range_limits = self.schema_oracle_config.default_range_limits.copy()
            schema_default_config.variable_default_limits = self.schema_oracle_config.variable_default_limits.copy()
        else:  # MODE_UNLIMITED
            schema_default_config.default_range_limits = {
                'list': (1, 10**6),
                'matrix': (1, 10**4),
                'group': (1, 10**6),
            }
            schema_default_config.variable_default_limits = {
                'int': (-10**18, 10**18),
                'float': (-10**18, 10**18),
                'char': None,
                'string': None,
            }

        schema_default_config.boundary_bias = self.schema_oracle_config.boundary_bias

    def generate(
        self,
        error_score_threshold: Optional[float] = None,
        eval_score_threshold: Optional[float] = None
    ) -> TestCase:
        """Phase 3: one sampled input; oracle output unless ``large`` mode."""
        self._check_initialized()

        error_threshold = error_score_threshold if error_score_threshold is not None else self.schema_oracle_config.error_score_threshold
        eval_threshold = eval_score_threshold if eval_score_threshold is not None else self.schema_oracle_config.eval_score_threshold

        if not self.is_best_combination_valid(error_threshold, eval_threshold):
            raise ValueError(
                f"best pair fails thresholds: "
                f"error_score={self.best_error_score:.3f} (need <= {error_threshold}), "
                f"eval_score={self.best_eval_score:.3f} (need >= {eval_threshold})"
            )

        if not self._candidates_artifact or not self._candidates_artifact.best_schema:
            raise ValueError("no best schema")

        best_schema = self._candidates_artifact.best_schema
        best_solution = self._candidates_artifact.best_solution

        self._apply_data_range_config()

        try:
            test_input = generate_data_from_schema(best_schema)
            if not test_input:
                raise ValueError("schema produced no input")
        except Exception as e:
            raise ValueError(f"input generation failed: {e}")

        if self.schema_oracle_config.should_compute_output():
            if not best_solution:
                raise ValueError("no best solution")
            
            fn_name = self._problem.metadata.get('func_name', None) if self._problem else None
            timeout = self.schema_oracle_config.single_test_timeout
            
            success, output, error = self._run_solution_on_input(
                best_solution, test_input, fn_name, timeout
            )
            
            if not success and error != "Time Limit Exceeded":
                expected_output = ""
            else:
                if isinstance(output, str):
                    expected_output = output
                else:
                    expected_output = json.dumps(output, ensure_ascii=False)
        else:
            expected_output = ""
        
        return TestCase(
            input=test_input,
            expected_output=expected_output,
            metadata={
                "generator": "schema_oracle",
                "schema_index": self._get_best_schema_index(),
                "solution_index": self._get_best_solution_index(),
                "error_score": self.best_error_score,
                "eval_score": self.best_eval_score,
                "data_range_mode": self.schema_oracle_config.data_range_mode,
            }
        )
    
    def _get_best_schema_index(self) -> int:
        if not self._candidates_artifact or not self._candidates_artifact.best_schema:
            return -1
        
        for i, sc in enumerate(self._candidates_artifact.schema_candidates):
            if sc.schema_json == self._candidates_artifact.best_schema:
                return sc.index
        return -1
    
    def _get_best_solution_index(self) -> int:
        if not self._candidates_artifact or not self._candidates_artifact.best_solution:
            return -1
        
        for i, sol in enumerate(self._candidates_artifact.solution_candidates):
            if sol.code == self._candidates_artifact.best_solution:
                return sol.index
        return -1
    
    def is_best_combination_valid(
        self, 
        error_score_threshold: Optional[float] = None,
        eval_score_threshold: Optional[float] = None
    ) -> bool:
        """``error_score <=`` threshold and ``eval_score >=`` threshold."""
        if not self._candidates_artifact:
            return False

        error_threshold = error_score_threshold if error_score_threshold is not None else self.schema_oracle_config.error_score_threshold
        eval_threshold = eval_score_threshold if eval_score_threshold is not None else self.schema_oracle_config.eval_score_threshold

        return (self._candidates_artifact.error_score <= error_threshold and
                self._candidates_artifact.eval_score >= eval_threshold)
    
    @property
    def best_error_score(self) -> float:
        if self._candidates_artifact:
            return self._candidates_artifact.error_score
        return 1.0
    
    @property
    def best_eval_score(self) -> float:
        if self._candidates_artifact:
            return self._candidates_artifact.eval_score
        return 0.0
    
    @property
    def avg_test_cases(self) -> float:
        if self._candidates_artifact:
            return self._candidates_artifact.avg_test_cases
        return 0.0
    
    @property
    def best_schema(self) -> Optional[str]:
        if self._candidates_artifact:
            return self._candidates_artifact.best_schema
        return None
    
    @property
    def best_solution(self) -> Optional[str]:
        if self._candidates_artifact:
            return self._candidates_artifact.best_solution
        return None
    
    @property
    def schema_candidates(self) -> List[SchemaCandidate]:
        if self._candidates_artifact:
            return self._candidates_artifact.schema_candidates
        return []
    
    @property
    def solution_candidates(self) -> List[SolutionCandidate]:
        if self._candidates_artifact:
            return self._candidates_artifact.solution_candidates
        return []
    
    @property
    def valid_schema_count(self) -> int:
        return len([sc for sc in self.schema_candidates if sc.is_valid])
    
    @property
    def solution_count(self) -> int:
        return len(self.solution_candidates)
    
    def get_schema_json(self, index: int = 0) -> Optional[str]:
        if index < len(self.schema_candidates):
            return self.schema_candidates[index].schema_json
        return None
    
    def get_solution_code(self, index: int = 0) -> Optional[str]:
        if index < len(self.solution_candidates):
            return self.solution_candidates[index].code
        return None
