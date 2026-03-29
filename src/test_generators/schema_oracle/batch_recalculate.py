#!/usr/bin/env python3
"""
Batch recalc Schema Oracle phase-2 (Majority Voting).

Usage:
    python -m src.test_generators.schema_oracle.batch_recalculate \
        --benchmark LiveCodeBench \
        --model deepseek-chat \
        --engine deepseek-deepseek-chat

    First N only:
    python -m src.test_generators.schema_oracle.batch_recalculate \
        --benchmark LiveCodeBench \
        --model deepseek-chat \
        --engine deepseek-deepseek-chat \
        --limit 1

    Range 10–20 (1-based, inclusive):
    python -m src.test_generators.schema_oracle.batch_recalculate \
        --benchmark LiveCodeBench \
        --model deepseek-chat \
        --engine deepseek-deepseek-chat \
        --start-index 10 --end-index 20

    From 5 to end:
    python -m src.test_generators.schema_oracle.batch_recalculate \
        --benchmark LiveCodeBench \
        --model deepseek-chat \
        --engine deepseek-deepseek-chat \
        --start-index 5

    Verbose:
    python -m src.test_generators.schema_oracle.batch_recalculate \
        --benchmark LiveCodeBench \
        --model deepseek-chat \
        --engine deepseek-deepseek-chat \
        --limit 1 \
        --verbose

    Recalc using deepseek-chat-direct cache:
    python -m src.test_generators.schema_oracle.batch_recalculate \
    --benchmark LiveCodeBench \
    --model deepseek-chat-direct \
    --engine deepseek-deepseek-chat \
    --verbose
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
from src.test_generators.schema_oracle.generator import (
    SchemaOracleTestGenerator, 
    CandidatesArtifact
)
from src.test_generators.schema_oracle.config import SchemaOracleConfig
from src.test_generators.schema_oracle.paths import SCHEMA_ORACLE_CACHE_ROOT


@dataclass
class RecalculateResult:
    """One problem recalc outcome."""
    question_id: str
    success: bool
    old_error_score: float
    old_eval_score: float
    new_error_score: float
    new_eval_score: float
    is_valid: bool
    avg_test_cases: float = 0.0
    error_msg: Optional[str] = None


class BatchRecalculator:
    """Batch phase-2 recalc."""
    
    DEFAULT_CACHE_ROOT = SCHEMA_ORACLE_CACHE_ROOT
    DEFAULT_ERROR_THRESHOLD = 0.1
    DEFAULT_EVAL_THRESHOLD = 0.8
    
    def __init__(
        self,
        benchmark: str,
        model: str,
        engine_name: str,
        cache_root: Optional[Path] = None,
        error_threshold: float = DEFAULT_ERROR_THRESHOLD,
        eval_threshold: float = DEFAULT_EVAL_THRESHOLD,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_voting_details: bool = True,
        verbose: bool = False,
    ):
        self.benchmark = benchmark
        self.model = model
        self.engine_name = engine_name
        self.cache_root = cache_root or self.DEFAULT_CACHE_ROOT
        self.error_threshold = error_threshold
        self.eval_threshold = eval_threshold
        self.start_date = start_date
        self.end_date = end_date
        self.save_voting_details = save_voting_details
        self.verbose = verbose
        
        self.cache_dir = self.cache_root / benchmark / model
        self.problems: Dict[str, Problem] = {}
        self._engine = None
        self.results: List[RecalculateResult] = []
    
    def get_engine(self):
        if self._engine is None:
            from src.engine import get_engine
            self._engine = get_engine(self.engine_name)
        return self._engine
    
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
    
    def load_cache_info(self, question_id: str) -> tuple:
        """Returns (error_score, eval_score, avg_test_cases)."""
        artifact_path = self.cache_dir / question_id / "candidates_artifact.json"
        try:
            with open(artifact_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("error_score", 1.0), data.get("eval_score", 0.0), data.get("avg_test_cases", 0.0)
        except:
            return 1.0, 0.0, 0.0
    
    def is_valid(self, error_score: float, eval_score: float) -> bool:
        eps = 1e-9
        return error_score <= self.error_threshold + eps and eval_score >= self.eval_threshold - eps
    
    def recalculate_one(self, question_id: str) -> RecalculateResult:
        old_error, old_eval, _ = self.load_cache_info(question_id)
        
        result = RecalculateResult(
            question_id=question_id,
            success=False,
            old_error_score=old_error,
            old_eval_score=old_eval,
            new_error_score=old_error,
            new_eval_score=old_eval,
            is_valid=False
        )
        
        if question_id not in self.problems:
            result.error_msg = "not in benchmark"
            return result
        
        problem = self.problems[question_id]
        
        try:
            engine = self.get_engine()
            generator = SchemaOracleTestGenerator(
                engine=engine,
                schema_oracle_config=SchemaOracleConfig(
                    test_case_count=5000,
                    save_voting_details=self.save_voting_details,
                ),
                cache_dir=self.cache_root,
                raw_cache_dir=True,
            )
            # cache_root / generator.name / cache_key = results/schemas / "{benchmark}/{model}" / question_id
            generator.name = f"{self.benchmark}/{self.model}"
            generator._get_cache_key = lambda p: p.question_id
            
            generator.VERBOSE_OUTPUT = self.verbose
            
            success = generator.initialize(problem, force_recalculate=True)
            
            if success:
                result.success = True
                result.new_error_score = generator.best_error_score
                result.new_eval_score = generator.best_eval_score
                result.avg_test_cases = generator.avg_test_cases
                result.is_valid = self.is_valid(result.new_error_score, result.new_eval_score)
            else:
                result.error_msg = "init failed"
                
        except Exception as e:
            result.error_msg = str(e)
        
        return result
    
    def run(self, skip_valid: bool = False, question_id: str = None, limit: int = None,
            start_index: int = None, end_index: int = None) -> None:
        """Batch run.

        skip_valid: skip problems already valid
        question_id: single id
        limit: cap count after range filter
        start_index, end_index: 1-based inclusive range
        """
        question_ids = self.get_cached_question_ids()
        
        if not question_ids:
            print("No cache data")
            return
        
        total_cached = len(question_ids)
        print(f"{total_cached} cached problems")
        
        if question_id:
            if question_id in question_ids:
                question_ids = [question_id]
            else:
                print(f"{question_id} not in cache")
                return
        else:
            if start_index is not None or end_index is not None:
                start_idx = (start_index - 1) if start_index else 0
                end_idx = end_index if end_index else total_cached
                
                start_idx = max(0, start_idx)
                end_idx = min(total_cached, end_idx)
                
                if start_idx >= end_idx:
                    print(f"Bad range: start_index={start_index}, end_index={end_index}")
                    return
                
                question_ids = question_ids[start_idx:end_idx]
                print(f"Range: items {start_idx + 1}–{end_idx}")
        
        if limit and limit > 0:
            question_ids = question_ids[:limit]
        
        print(f"\nProcessing {len(question_ids)} problems")
        print("=" * 70)
        
        self.results = []
        start_time = time.time()
        
        for i, qid in enumerate(question_ids):
            if skip_valid:
                old_error, old_eval, old_avg_tc = self.load_cache_info(qid)
                if self.is_valid(old_error, old_eval):
                    print(f"[{i+1}/{len(question_ids)}] {qid}: already valid, skip")
                    result = RecalculateResult(
                        question_id=qid,
                        success=True,
                        old_error_score=old_error,
                        old_eval_score=old_eval,
                        new_error_score=old_error,
                        new_eval_score=old_eval,
                        is_valid=True,
                        avg_test_cases=old_avg_tc
                    )
                    self.results.append(result)
                    continue
            
            print(f"[{i+1}/{len(question_ids)}] recalc {qid} ... ", end="", flush=True)
            
            result = self.recalculate_one(qid)
            self.results.append(result)
            
            if result.success:
                status = "✅ valid" if result.is_valid else "⚠️ below thresh"
                print(f"{status} (error={result.new_error_score:.4f}, eval={result.new_eval_score:.4f}, avg_tc={result.avg_test_cases:.1f})")
            else:
                print(f"❌ fail: {result.error_msg}")
        
        elapsed = time.time() - start_time
        
        self.print_summary(elapsed)
    
    def print_summary(self, elapsed: float) -> None:
        print("\n" + "=" * 70)
        print("Batch recalc done")
        print("=" * 70)
        
        total = len(self.results)
        success = sum(1 for r in self.results if r.success)
        valid = sum(1 for r in self.results if r.is_valid)
        failed = total - success
        invalid = success - valid
        
        print(f"\nSummary:")
        print(f"  total:     {total}")
        print(f"  ok:        {success}")
        print(f"  ✅ valid:  {valid}")
        print(f"  ⚠️ below:  {invalid}")
        print(f"  ❌ fail:   {failed}")
        print(f"\nTime: {elapsed:.1f}s")
        
        if success > 0:
            valid_rate = valid / success * 100
            print(f"Valid rate: {valid_rate:.1f}% ({valid}/{success})")
        
        invalid_results = [r for r in self.results if r.success and not r.is_valid]
        if invalid_results:
            print(f"\nBelow threshold ({len(invalid_results)}):")
            for r in invalid_results:
                print(f"  {r.question_id}: error={r.new_error_score:.4f}, eval={r.new_eval_score:.4f}")
        
        failed_results = [r for r in self.results if not r.success]
        if failed_results:
            print(f"\nFailed ({len(failed_results)}):")
            for r in failed_results:
                print(f"  {r.question_id}: {r.error_msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch recalc Schema Oracle phase-2"
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
        "--engine", "-e",
        type=str,
        required=True,
        help="Engine name (required)"
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
        help="Max error for valid"
    )
    
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=0.8,
        help="Min eval for valid"
    )
    
    parser.add_argument(
        "--release-version",
        type=str,
        default="release_latest",
        help="LiveCodeBench release"
    )
    
    parser.add_argument(
        "--skip-valid",
        action="store_true",
        help="Skip problems already valid"
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-01-01",
        help="Problem start date YYYY-MM-DD (default 2025-01-01)"
    )
    
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-05-01",
        help="Problem end date YYYY-MM-DD (default 2025-05-01)"
    )
    
    parser.add_argument(
        "--no-save-voting-details",
        action="store_true",
        help="Do not save voting detail JSON (default: save)"
    )
    
    parser.add_argument(
        "--question-id", "-q",
        type=str,
        default=None,
        help="Single question id"
    )
    
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Max N problems after range"
    )
    
    parser.add_argument(
        "--start-index", "--from",
        type=int,
        default=None,
        dest="start_index",
        help="Start index 1-based (default 1)"
    )
    
    parser.add_argument(
        "--end-index", "--to",
        type=int,
        default=None,
        dest="end_index",
        help="End index 1-based inclusive (default last)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose schema eval logs"
    )
    
    args = parser.parse_args()
    
    cache_root = Path(args.cache_root) if args.cache_root else None
    
    recalculator = BatchRecalculator(
        benchmark=args.benchmark,
        model=args.model,
        engine_name=args.engine,
        cache_root=cache_root,
        error_threshold=args.error_threshold,
        eval_threshold=args.eval_threshold,
        start_date=args.start_date,
        end_date=args.end_date,
        save_voting_details=not args.no_save_voting_details,
        verbose=args.verbose,
    )
    
    if recalculator.load_problems(release_version=args.release_version) == 0:
        print("Cannot load problems; exit")
        return
    
    recalculator.run(
        skip_valid=args.skip_valid,
        question_id=args.question_id,
        limit=args.limit,
        start_index=args.start_index,
        end_index=args.end_index
    )


if __name__ == "__main__":
    main()
