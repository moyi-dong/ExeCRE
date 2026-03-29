#!/usr/bin/env python3
"""
Batch TLEfree eval for Schema Oracle best_solution.

Compares self-judgment (error_score/eval_score) vs ground truth (tlefree_passed).

Binary view:
- Self: valid if error_score <= 0.1 and eval_score >= 0.8 (configurable)
- TLEfree: best_solution passes all problem tests

Usage:
    python -m src.test_generators.schema_oracle.batch_tlefree_evaluate \
        --benchmark LiveCodeBench \
        --model deepseek-chat \
        --start-date 2025-01-01 \
        --end-date 2025-05-31
"""

import json
import argparse
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.problem import Problem
from src.core.benchmark_loader import load_benchmark, create_livecodebench_config
from src.evaluators.tlefree_evaluator import tlefree_evaluate_simulation_code
from src.test_generators.schema_oracle.paths import SCHEMA_ORACLE_CACHE_ROOT


@dataclass
class TLEfreeResult:
    """One problem TLEfree outcome."""
    question_id: str
    success: bool
    self_valid: bool
    tlefree_passed: bool
    error_score: float = 1.0
    eval_score: float = 0.0
    error_msg: Optional[str] = None


class BatchTLEfreeEvaluator:
    
    DEFAULT_CACHE_ROOT = SCHEMA_ORACLE_CACHE_ROOT
    DEFAULT_ERROR_THRESHOLD = 0.1
    DEFAULT_EVAL_THRESHOLD = 0.8
    DEFAULT_TIMEOUT = 6.0
    
    def __init__(
        self,
        benchmark: str,
        model: str,
        cache_root: Optional[Path] = None,
        error_threshold: float = DEFAULT_ERROR_THRESHOLD,
        eval_threshold: float = DEFAULT_EVAL_THRESHOLD,
        timeout: float = DEFAULT_TIMEOUT,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        self.benchmark = benchmark
        self.model = model
        self.cache_root = cache_root or self.DEFAULT_CACHE_ROOT
        self.error_threshold = error_threshold
        self.eval_threshold = eval_threshold
        self.timeout = timeout
        self.start_date = start_date
        self.end_date = end_date
        
        self.cache_dir = self.cache_root / benchmark / model
        self.problems: Dict[str, Problem] = {}
        self.results: List[TLEfreeResult] = []
    
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
    
    def is_self_valid(self, error_score: float, eval_score: float) -> bool:
        eps = 1e-9
        return error_score <= self.error_threshold + eps and eval_score >= self.eval_threshold - eps
    
    def evaluate_one(self, question_id: str) -> TLEfreeResult:
        artifact = self.load_artifact(question_id)
        if artifact is None:
            return TLEfreeResult(
                question_id=question_id,
                success=False,
                self_valid=False,
                tlefree_passed=False,
                error_msg="cannot load artifact"
            )
        
        error_score = artifact.get("error_score", 1.0)
        eval_score = artifact.get("eval_score", 0.0)
        self_valid = self.is_self_valid(error_score, eval_score)
        best_solution = artifact.get("best_solution", "")
        
        if not best_solution or not best_solution.strip():
            result = TLEfreeResult(
                question_id=question_id,
                success=True,
                self_valid=self_valid,
                tlefree_passed=False,
                error_score=error_score,
                eval_score=eval_score,
                error_msg="best_solution empty"
            )
            artifact["tlefree_passed"] = False
            self.save_artifact(question_id, artifact)
            return result
        
        if question_id not in self.problems:
            return TLEfreeResult(
                question_id=question_id,
                success=False,
                self_valid=self_valid,
                tlefree_passed=False,
                error_score=error_score,
                eval_score=eval_score,
                error_msg="not in benchmark"
            )
        
        problem = self.problems[question_id]
        
        test_cases = problem.public_test_cases + problem.private_test_cases
        if not test_cases:
            result = TLEfreeResult(
                question_id=question_id,
                success=True,
                self_valid=self_valid,
                tlefree_passed=False,
                error_score=error_score,
                eval_score=eval_score,
                error_msg="no test cases"
            )
            artifact["tlefree_passed"] = False
            self.save_artifact(question_id, artifact)
            return result
        
        fn_name = problem.metadata.get('func_name', None)
        
        try:
            tlefree_passed = tlefree_evaluate_simulation_code(
                code=best_solution,
                test_cases=test_cases,
                fn_name=fn_name,
                timeout=self.timeout
            )
        except Exception as e:
            result = TLEfreeResult(
                question_id=question_id,
                success=False,
                self_valid=self_valid,
                tlefree_passed=False,
                error_score=error_score,
                eval_score=eval_score,
                error_msg=f"eval error: {str(e)}"
            )
            return result
        
        artifact["tlefree_passed"] = tlefree_passed
        self.save_artifact(question_id, artifact)
        
        return TLEfreeResult(
            question_id=question_id,
            success=True,
            self_valid=self_valid,
            tlefree_passed=tlefree_passed,
            error_score=error_score,
            eval_score=eval_score
        )
    
    def run(self, skip_evaluated: bool = False) -> None:
        question_ids = self.get_cached_question_ids()
        
        if not question_ids:
            print("No cache data")
            return
        
        valid_question_ids = [qid for qid in question_ids if qid in self.problems]
        skipped_count = len(question_ids) - len(valid_question_ids)
        
        print(f"\n{len(question_ids)} cache entries, {len(valid_question_ids)} in benchmark")
        if skipped_count > 0:
            print(f"Skip {skipped_count} outside date filter / not in benchmark")
        print("=" * 70)
        
        self.results = []
        start_time = time.time()
        
        for i, qid in enumerate(valid_question_ids):
            if skip_evaluated:
                artifact = self.load_artifact(qid)
                if artifact and "tlefree_passed" in artifact:
                    print(f"[{i+1}/{len(valid_question_ids)}] {qid}: already done, skip")
                    error_score = artifact.get("error_score", 1.0)
                    eval_score = artifact.get("eval_score", 0.0)
                    result = TLEfreeResult(
                        question_id=qid,
                        success=True,
                        self_valid=self.is_self_valid(error_score, eval_score),
                        tlefree_passed=artifact["tlefree_passed"],
                        error_score=error_score,
                        eval_score=eval_score
                    )
                    self.results.append(result)
                    continue
            
            print(f"[{i+1}/{len(valid_question_ids)}] eval {qid} ... ", end="", flush=True)
            
            result = self.evaluate_one(qid)
            self.results.append(result)
            
            if result.success:
                self_status = "self OK" if result.self_valid else "self bad"
                tlefree_status = "✅ tlefree OK" if result.tlefree_passed else "❌ tlefree fail"
                print(f"{self_status} | {tlefree_status}")
            else:
                print(f"⚠️ fail: {result.error_msg}")
        
        elapsed = time.time() - start_time
        
        self.print_summary(elapsed)
    
    def print_summary(self, elapsed: float) -> None:
        print("\n" + "=" * 70)
        print("TLEfree batch done")
        print("=" * 70)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
        failed = total - success
        
        print(f"\nBasics:")
        print(f"  total:      {total}")
        print(f"  ok:         {success}")
        print(f"  failed:     {failed}")
        print(f"  time:       {elapsed:.1f}s")
        
        if success == 0:
            return
        
        tp = sum(1 for r in self.results if r.success and r.self_valid and r.tlefree_passed)
        fp = sum(1 for r in self.results if r.success and r.self_valid and not r.tlefree_passed)
        fn = sum(1 for r in self.results if r.success and not r.self_valid and r.tlefree_passed)
        tn = sum(1 for r in self.results if r.success and not r.self_valid and not r.tlefree_passed)
        
        print(f"\nConfusion (self vs TLEfree):")
        print(f"                    pass      fail")
        print(f"  self valid        {tp:>8}  {fp:>8}")
        print(f"  self invalid      {fn:>8}  {tn:>8}")
        
        total_valid_self = tp + fp
        total_valid_tlefree = tp + fn
        
        print(f"\nMetrics:")
        print(f"  self-valid count:   {total_valid_self} ({total_valid_self/success*100:.1f}%)")
        print(f"  tlefree-pass count: {total_valid_tlefree} ({total_valid_tlefree/success*100:.1f}%)")
        
        if tp + fp > 0:
            precision = tp / (tp + fp)
            print(f"  Precision: {precision:.4f}  (self-valid & truly pass)")
        
        if tp + fn > 0:
            recall = tp / (tp + fn)
            print(f"  Recall:    {recall:.4f}  (pass & self says valid)")
        
        if tp + fp > 0 and tp + fn > 0:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            print(f"  F1:        {f1:.4f}")
        
        accuracy = (tp + tn) / success
        print(f"  Accuracy:  {accuracy:.4f}  (self agrees with tlefree)")
        
        fp_results = [r for r in self.results if r.success and r.self_valid and not r.tlefree_passed]
        if fp_results:
            print(f"\nSelf-valid but TLEfree fail ({len(fp_results)}):")
            for r in fp_results[:10]:
                print(f"  {r.question_id}: error={r.error_score:.4f}, eval={r.eval_score:.4f}")
            if len(fp_results) > 10:
                print(f"  ... +{len(fp_results) - 10} more")
        
        fn_results = [r for r in self.results if r.success and not r.self_valid and r.tlefree_passed]
        if fn_results:
            print(f"\nSelf-invalid but TLEfree pass ({len(fn_results)}):")
            for r in fn_results[:10]:
                print(f"  {r.question_id}: error={r.error_score:.4f}, eval={r.eval_score:.4f}")
            if len(fn_results) > 10:
                print(f"  ... +{len(fn_results) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Batch TLEfree eval for Schema Oracle"
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
        "--error-threshold",
        type=float,
        default=0.1,
        help="Max error for self-valid"
    )
    
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=0.8,
        help="Min eval for self-valid"
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
        help="Skip if tlefree_passed already in artifact"
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
    
    args = parser.parse_args()
    
    cache_root = Path(args.cache_root) if args.cache_root else None
    
    evaluator = BatchTLEfreeEvaluator(
        benchmark=args.benchmark,
        model=args.model,
        cache_root=cache_root,
        error_threshold=args.error_threshold,
        eval_threshold=args.eval_threshold,
        timeout=args.timeout,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    
    if evaluator.load_problems(release_version=args.release_version) == 0:
        print("Cannot load problems; exit")
        return
    
    evaluator.run(skip_evaluated=args.skip_evaluated)


if __name__ == "__main__":
    main()
