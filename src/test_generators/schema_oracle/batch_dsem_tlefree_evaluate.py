#!/usr/bin/env python3
"""
Dawid-Skene EM + TLEfree batch eval.

1. Pick best code via DS-EM from solution candidates
2. Run TLEfree on that code
3. Write fields on candidates_artifact.json
4. Print confusion vs alpha thresholds

Fields:
- DSEM1218_best_solution, DSEM1218_best_solution_index, DSEM1218_best_schema_index
- DSEM1218_error_score, DSEM1218_eval_score, DSEM1218_tlefree_passed, DSEM1218_alpha

Usage:
    python -m src.test_generators.schema_oracle.batch_dsem_tlefree_evaluate \
        --benchmark LiveCodeBench \
        --model deepseek-chat \
        --start-date 2025-01-01 \
        --end-date 2025-05-31

    Read cache only, print tables:
    python -m src.test_generators.schema_oracle.batch_dsem_tlefree_evaluate \
        --benchmark LiveCodeBench \
        --model deepseek-chat \
        --read-only

    Custom thresholds:
    python -m src.test_generators.schema_oracle.batch_dsem_tlefree_evaluate \
        --benchmark LiveCodeBench \
        --model deepseek-chat \
        --read-only \
        --thresholds 0.99,0.98,0.97,0.96,0.95,0.90,0.85,0.80
"""

import json
import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.problem import Problem
from src.core.benchmark_loader import load_benchmark, create_livecodebench_config
from src.evaluators.tlefree_evaluator import tlefree_evaluate_simulation_code
from src.test_generators.schema_oracle.analysis.algorithms.majority_voting import (
    SchemaVotingDetail
)
from src.test_generators.schema_oracle.analysis.algorithms.Dawid_Skene_1218 import (
    select_best_by_dawid_skene,
)
from src.test_generators.schema_oracle.paths import SCHEMA_ORACLE_CACHE_ROOT


@dataclass
class DSEMTLEfreeResult:
    """One problem DS-EM + TLEfree outcome."""
    question_id: str
    success: bool
    dsem_eval_score: float
    dsem_error_score: float
    dsem_tlefree_passed: bool
    dsem_best_solution_index: int = -1
    dsem_best_schema_index: int = -1
    dsem_alpha: List[float] = field(default_factory=list)
    same_as_mv: bool = False
    error_msg: Optional[str] = None


@dataclass
class ConfusionMatrix:
    """2x2 counts at one alpha threshold."""
    threshold: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    
    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
    
    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0.0


class BatchDSEMTLEfreeEvaluator:
    
    DEFAULT_CACHE_ROOT = SCHEMA_ORACLE_CACHE_ROOT
    DEFAULT_TIMEOUT = 6.0
    
    def __init__(
        self,
        benchmark: str,
        model: str,
        cache_root: Optional[Path] = None,
        timeout: float = DEFAULT_TIMEOUT,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.benchmark = benchmark
        self.model = model
        self.cache_root = cache_root or self.DEFAULT_CACHE_ROOT
        self.timeout = timeout
        self.start_date = start_date
        self.end_date = end_date
        
        self.cache_dir = self.cache_root / benchmark / model
        self.problems: Dict[str, Problem] = {}
        self.results: List[DSEMTLEfreeResult] = []
    
    def load_problems(self, release_version: str = "release_latest") -> int:
        date_filter = ""
        if self.start_date or self.end_date:
            date_filter = f" ({self.start_date or '...'} ~ {self.end_date or '...'})"
        print(f"Loading problems{date_filter}...")
        try:
            config = create_livecodebench_config(
                release_version=release_version,
                start_date=self.start_date,
                end_date=self.end_date
            )
            problems = load_benchmark(config)
            self.problems = {p.question_id: p for p in problems}
            print(f"Loaded {len(self.problems)} problems")
            return len(self.problems)
        except Exception as e:
            print(f"Load failed: {e}")
            return 0
    
    def get_cached_question_ids(self) -> List[str]:
        if not self.cache_dir.exists():
            return []
        
        question_ids = []
        for d in sorted(self.cache_dir.iterdir()):
            if d.is_dir() and (d / "candidates_artifact.json").exists():
                question_ids.append(d.name)
        return question_ids
    
    def load_artifact(self, question_id: str) -> Optional[Dict]:
        artifact_path = self.cache_dir / question_id / "candidates_artifact.json"
        try:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return None
    
    def save_artifact(self, question_id: str, artifact: Dict) -> bool:
        artifact_path = self.cache_dir / question_id / "candidates_artifact.json"
        try:
            with open(artifact_path, 'w', encoding='utf-8') as f:
                json.dump(artifact, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"save artifact failed: {e}")
            return False
    
    def load_voting_details(self, question_id: str) -> List[SchemaVotingDetail]:
        voting_dir = self.cache_dir / question_id / "voting_details"
        voting_details = []
        
        if not voting_dir.exists():
            return voting_details
        
        for voting_file in sorted(voting_dir.glob("schema_*.json")):
            try:
                with open(voting_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                detail = SchemaVotingDetail.from_dict(data)
                voting_details.append(detail)
            except Exception as e:
                print(f"voting detail load fail ({voting_file.name}): {e}")
        
        return voting_details
    
    def evaluate_one(self, question_id: str) -> DSEMTLEfreeResult:
        artifact = self.load_artifact(question_id)
        if artifact is None:
            return DSEMTLEfreeResult(
                question_id=question_id,
                success=False,
                dsem_eval_score=0.0,
                dsem_error_score=1.0,
                dsem_tlefree_passed=False,
                error_msg="cannot load artifact"
            )
        
        voting_details = self.load_voting_details(question_id)
        if not voting_details:
            return DSEMTLEfreeResult(
                question_id=question_id,
                success=False,
                dsem_eval_score=0.0,
                dsem_error_score=1.0,
                dsem_tlefree_passed=False,
                error_msg="no voting_details"
            )
        
        solution_candidates = artifact.get("solution_candidates", [])
        if not solution_candidates:
            return DSEMTLEfreeResult(
                question_id=question_id,
                success=False,
                dsem_eval_score=0.0,
                dsem_error_score=1.0,
                dsem_tlefree_passed=False,
                error_msg="no solution_candidates"
            )
        
        try:
            ds_result = select_best_by_dawid_skene(
                voting_details,
                num_solutions=len(solution_candidates),
                verbose=False
            )
        except Exception as e:
            return DSEMTLEfreeResult(
                question_id=question_id,
                success=False,
                dsem_eval_score=0.0,
                dsem_error_score=1.0,
                dsem_tlefree_passed=False,
                error_msg=f"DS-EM failed: {str(e)}"
            )
        
        best_solution_index = ds_result.best_solution_index
        if best_solution_index < 0 or best_solution_index >= len(solution_candidates):
            return DSEMTLEfreeResult(
                question_id=question_id,
                success=False,
                dsem_eval_score=ds_result.eval_score,
                dsem_error_score=ds_result.error_score,
                dsem_tlefree_passed=False,
                dsem_best_solution_index=best_solution_index,
                dsem_best_schema_index=ds_result.best_schema_index,
                dsem_alpha=ds_result.alpha,
                error_msg=f"bad solution index: {best_solution_index}"
            )
        
        dsem_best_solution = solution_candidates[best_solution_index].get("code", "")
        mv_best_solution = artifact.get("best_solution", "")
        
        same_as_mv = (dsem_best_solution.strip() == mv_best_solution.strip())
        
        if same_as_mv and "tlefree_passed" in artifact:
            dsem_tlefree_passed = artifact["tlefree_passed"]
        else:
            if not dsem_best_solution or not dsem_best_solution.strip():
                dsem_tlefree_passed = False
            elif question_id not in self.problems:
                return DSEMTLEfreeResult(
                    question_id=question_id,
                    success=False,
                    dsem_eval_score=ds_result.eval_score,
                    dsem_error_score=ds_result.error_score,
                    dsem_tlefree_passed=False,
                    dsem_best_solution_index=best_solution_index,
                    dsem_best_schema_index=ds_result.best_schema_index,
                    dsem_alpha=ds_result.alpha,
                    same_as_mv=same_as_mv,
                    error_msg="not in benchmark"
                )
            else:
                problem = self.problems[question_id]
                test_cases = problem.public_test_cases + problem.private_test_cases
                
                if not test_cases:
                    dsem_tlefree_passed = False
                else:
                    fn_name = problem.metadata.get('func_name', None)
                    try:
                        dsem_tlefree_passed = tlefree_evaluate_simulation_code(
                            code=dsem_best_solution,
                            test_cases=test_cases,
                            fn_name=fn_name,
                            timeout=self.timeout
                        )
                    except Exception as e:
                        return DSEMTLEfreeResult(
                            question_id=question_id,
                            success=False,
                            dsem_eval_score=ds_result.eval_score,
                            dsem_error_score=ds_result.error_score,
                            dsem_tlefree_passed=False,
                            dsem_best_solution_index=best_solution_index,
                            dsem_best_schema_index=ds_result.best_schema_index,
                            dsem_alpha=ds_result.alpha,
                            same_as_mv=same_as_mv,
                            error_msg=f"TLEfree error: {str(e)}"
                        )
        
        artifact["DSEM1218_best_solution"] = dsem_best_solution
        artifact["DSEM1218_best_solution_index"] = best_solution_index
        artifact["DSEM1218_best_schema_index"] = ds_result.best_schema_index
        artifact["DSEM1218_error_score"] = ds_result.error_score
        artifact["DSEM1218_eval_score"] = ds_result.eval_score
        artifact["DSEM1218_tlefree_passed"] = dsem_tlefree_passed
        artifact["DSEM1218_alpha"] = ds_result.alpha
        artifact["DSEM1218_same_as_mv"] = same_as_mv
        self.save_artifact(question_id, artifact)
        
        return DSEMTLEfreeResult(
            question_id=question_id,
            success=True,
            dsem_eval_score=ds_result.eval_score,
            dsem_error_score=ds_result.error_score,
            dsem_tlefree_passed=dsem_tlefree_passed,
            dsem_best_solution_index=best_solution_index,
            dsem_best_schema_index=ds_result.best_schema_index,
            dsem_alpha=ds_result.alpha,
            same_as_mv=same_as_mv
        )
    
    def load_results_from_cache(self) -> int:
        question_ids = self.get_cached_question_ids()
        
        if not question_ids:
            print("No cache data")
            return 0
        
        self.results = []
        loaded_count = 0
        
        for qid in question_ids:
            artifact = self.load_artifact(qid)
            if artifact and "DSEM1218_tlefree_passed" in artifact:
                result = DSEMTLEfreeResult(
                    question_id=qid,
                    success=True,
                    dsem_eval_score=artifact.get("DSEM1218_eval_score", 0.0),
                    dsem_error_score=artifact.get("DSEM1218_error_score", 1.0),
                    dsem_tlefree_passed=artifact["DSEM1218_tlefree_passed"],
                    dsem_best_solution_index=artifact.get("DSEM1218_best_solution_index", -1),
                    dsem_best_schema_index=artifact.get("DSEM1218_best_schema_index", -1),
                    dsem_alpha=artifact.get("DSEM1218_alpha", []),
                    same_as_mv=artifact.get("DSEM1218_same_as_mv", False)
                )
                self.results.append(result)
                loaded_count += 1
        
        return loaded_count
    
    def show_confusion_matrix_only(self, thresholds: Optional[List[float]] = None) -> None:
        print("Loading from cache...")
        loaded_count = self.load_results_from_cache()
        
        if loaded_count == 0:
            print("No saved DSEM results")
            return
        
        print(f"Loaded {loaded_count} rows")
        print("=" * 70)
        
        success = sum(1 for r in self.results if r.success)
        print(f"\nBasics:")
        print(f"  total:  {len(self.results)}")
        print(f"  ok:     {success}")
        
        if success == 0:
            print("No success rows; cannot build matrix")
            return
        
        same_as_mv = sum(1 for r in self.results if r.success and r.same_as_mv)
        diff_from_mv = success - same_as_mv
        print(f"\nvs Majority Voting:")
        print(f"  same pick:   {same_as_mv} ({same_as_mv/success*100:.1f}%)")
        print(f"  diff pick:   {diff_from_mv} ({diff_from_mv/success*100:.1f}%)")
        
        tlefree_passed = sum(1 for r in self.results if r.success and r.dsem_tlefree_passed)
        print(f"\nTLEfree:")
        print(f"  pass:  {tlefree_passed} ({tlefree_passed/success*100:.1f}%)")
        print(f"  fail:  {success - tlefree_passed} ({(success - tlefree_passed)/success*100:.1f}%)")
        
        if thresholds is None:
            thresholds = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90]
        
        self.print_alpha_confusion_matrix_table(thresholds=thresholds)
    
    def run(self, skip_evaluated: bool = False) -> None:
        question_ids = self.get_cached_question_ids()
        
        if not question_ids:
            print("No cache data")
            return
        
        valid_question_ids = [qid for qid in question_ids if qid in self.problems]
        skipped_count = len(question_ids) - len(valid_question_ids)
        
        print(f"\n{len(question_ids)} cache, {len(valid_question_ids)} in benchmark")
        if skipped_count > 0:
            print(f"Skip {skipped_count} outside filter / not in benchmark")
        print("=" * 70)
        
        self.results = []
        start_time = time.time()
        
        for i, qid in enumerate(valid_question_ids):
            if skip_evaluated:
                artifact = self.load_artifact(qid)
                if artifact and "DSEM1218_tlefree_passed" in artifact:
                    print(f"[{i+1}/{len(valid_question_ids)}] {qid}: skip (done)")
                    result = DSEMTLEfreeResult(
                        question_id=qid,
                        success=True,
                        dsem_eval_score=artifact.get("DSEM1218_eval_score", 0.0),
                        dsem_error_score=artifact.get("DSEM1218_error_score", 1.0),
                        dsem_tlefree_passed=artifact["DSEM1218_tlefree_passed"],
                        dsem_best_solution_index=artifact.get("DSEM1218_best_solution_index", -1),
                        dsem_best_schema_index=artifact.get("DSEM1218_best_schema_index", -1),
                        dsem_alpha=artifact.get("DSEM1218_alpha", []),
                        same_as_mv=artifact.get("DSEM1218_same_as_mv", False)
                    )
                    self.results.append(result)
                    continue
            
            print(f"[{i+1}/{len(valid_question_ids)}] eval {qid} ... ", end="", flush=True)
            
            result = self.evaluate_one(qid)
            self.results.append(result)
            
            if result.success:
                same_status = "same MV" if result.same_as_mv else "diff MV"
                tlefree_status = "✅ pass" if result.dsem_tlefree_passed else "❌ fail"
                print(f"alpha={result.dsem_eval_score:.4f} | {same_status} | {tlefree_status}")
            else:
                print(f"⚠️ fail: {result.error_msg}")
        
        elapsed = time.time() - start_time
        
        self.print_summary(elapsed)
        
        self.print_alpha_confusion_matrix_table()
    
    def print_summary(self, elapsed: float) -> None:
        print("\n" + "=" * 70)
        print("DS-EM + TLEfree done")
        print("=" * 70)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
        failed = total - success
        
        print(f"\nBasics:")
        print(f"  total:   {total}")
        print(f"  ok:      {success}")
        print(f"  fail:    {failed}")
        print(f"  time:    {elapsed:.1f}s")
        
        if success == 0:
            return
        
        same_as_mv = sum(1 for r in self.results if r.success and r.same_as_mv)
        diff_from_mv = success - same_as_mv
        print(f"\nvs Majority Voting:")
        print(f"  same:    {same_as_mv} ({same_as_mv/success*100:.1f}%)")
        print(f"  diff:    {diff_from_mv} ({diff_from_mv/success*100:.1f}%)")
        
        tlefree_passed = sum(1 for r in self.results if r.success and r.dsem_tlefree_passed)
        print(f"\nTLEfree:")
        print(f"  pass:    {tlefree_passed} ({tlefree_passed/success*100:.1f}%)")
        print(f"  fail:    {success - tlefree_passed} ({(success - tlefree_passed)/success*100:.1f}%)")
    
    def calculate_confusion_matrix(self, threshold: float) -> ConfusionMatrix:
        """predict valid if eval_score >= threshold; actual valid = TLEfree pass."""
        cm = ConfusionMatrix(threshold=threshold)
        
        for r in self.results:
            if not r.success:
                continue
            
            predicted_valid = r.dsem_eval_score >= threshold
            actual_valid = r.dsem_tlefree_passed
            
            if predicted_valid and actual_valid:
                cm.tp += 1
            elif predicted_valid and not actual_valid:
                cm.fp += 1
            elif not predicted_valid and actual_valid:
                cm.fn += 1
            else:
                cm.tn += 1
        
        return cm
    
    def print_alpha_confusion_matrix_table(self, thresholds: Optional[List[float]] = None) -> None:
        print("\n" + "=" * 100)
        print("Alpha threshold vs confusion")
        print("=" * 100)
        print('Rule: eval_score >= t => "predict valid"; TLEfree pass => "actual valid"')
        print()
        
        if thresholds is None:
            thresholds = [0.99, 0.98, 0.97, 0.96, 0.95]
        
        header = f"{'thr':>8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10}"
        print(header)
        print("-" * len(header))
        
        for threshold in thresholds:
            cm = self.calculate_confusion_matrix(threshold)
            row = f"{threshold:>8.2f} {cm.tp:>6} {cm.fp:>6} {cm.fn:>6} {cm.tn:>6} {cm.precision:>10.4f} {cm.recall:>10.4f} {cm.f1:>10.4f} {cm.accuracy:>10.4f}"
            print(row)
        
        print()
        
        print("\nDetail:")
        print(f"{'thr':>8} {'pred+':>10} {'act+':>10} {'pred%':>12} {'act%':>12}")
        print("-" * 60)
        
        success_count = sum(1 for r in self.results if r.success)
        
        for threshold in thresholds:
            cm = self.calculate_confusion_matrix(threshold)
            predicted_valid = cm.tp + cm.fp
            actual_valid = cm.tp + cm.fn
            predicted_rate = predicted_valid / success_count * 100 if success_count > 0 else 0
            actual_rate = actual_valid / success_count * 100 if success_count > 0 else 0
            
            row = f"{threshold:>8.2f} {predicted_valid:>10} {actual_valid:>10} {predicted_rate:>11.1f}% {actual_rate:>11.1f}%"
            print(row)
        
        print("\n" + "=" * 100)
        print("Cells (alpha >= t => predict valid):")
        print("  TP: high conf & TLEfree pass")
        print("  FP: high conf & TLEfree fail (overconfident)")
        print("  FN: low conf & TLEfree pass (missed)")
        print("  TN: low conf & TLEfree fail")
        print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="DS-EM + TLEfree batch eval"
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        default="LiveCodeBench",
        help="Benchmark name"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="deepseek-chat",
        help="Model name"
    )
    
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Cache root"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=6.0,
        help="Per-test timeout (seconds)"
    )
    
    parser.add_argument(
        "--release-version",
        type=str,
        default="release_latest",
        help="LiveCodeBench release"
    )
    
    parser.add_argument(
        "--skip-evaluated",
        action="store_true",
        default=True,
        help="Skip if DSEM1218_tlefree_passed exists (default on)"
    )
    
    parser.add_argument(
        "--force-reeval",
        action="store_true",
        help="Re-eval all (turns off skip-evaluated)"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
        help="Start date YYYY-MM-DD"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-05-31",
        help="End date YYYY-MM-DD"
    )
    
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="Load cache only; print tables, no eval"
    )
    
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Comma thresholds, e.g. 0.99,0.98,..."
    )
    
    args = parser.parse_args()
    
    cache_root = Path(args.cache_root) if args.cache_root else None
    
    evaluator = BatchDSEMTLEfreeEvaluator(
        benchmark=args.benchmark,
        model=args.model,
        cache_root=cache_root,
        timeout=args.timeout,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    
    if args.read_only:
        thresholds = None
        if args.thresholds:
            try:
                thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
                thresholds.sort(reverse=True)
            except ValueError:
                print(f"warn: bad --thresholds '{args.thresholds}', use default")
                thresholds = None
        
        evaluator.show_confusion_matrix_only(thresholds=thresholds)
        return
    
    if evaluator.load_problems(release_version=args.release_version) == 0:
        print("Cannot load problems; exit")
        return
    
    skip_evaluated = args.skip_evaluated and not args.force_reeval
    evaluator.run(skip_evaluated=skip_evaluated)


if __name__ == "__main__":
    main()
