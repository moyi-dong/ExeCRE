#!/usr/bin/env python3
"""
Schema Oracle interactive review tool.

Features:
1. Scan cache dir and load cache info for all problems
2. Show per item: question_id, error_score, eval_score, validity
3. Commands:
   - Enter/n: next
   - p: previous
   - r: recalc phase-2 for current problem
   - g <n>: go to problem n
   - q: quit
   - l: list score summary for all problems
   - h: help

Usage:
    python -m src.test_generators.schema_oracle.interactive_reviewer \\
        --benchmark LiveCodeBench \\
        --model deepseek-chat \\
        --engine deepseek-deepseek-chat
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Project root on path
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
class CacheInfo:
    """Per-problem cache summary."""
    question_id: str
    cache_path: Path
    artifact: Optional[CandidatesArtifact] = None
    error_score: float = 1.0
    eval_score: float = 0.0
    schema_count: int = 0
    solution_count: int = 0
    has_best: bool = False
    load_error: Optional[str] = None


class InteractiveReviewer:
    """Interactive cache reviewer."""
    
    DEFAULT_CACHE_ROOT = SCHEMA_ORACLE_CACHE_ROOT
    
    DEFAULT_ERROR_THRESHOLD = 0.1
    DEFAULT_EVAL_THRESHOLD = 0.8
    
    def __init__(
        self,
        benchmark: str,
        model: str,
        engine_name: Optional[str] = None,
        cache_root: Optional[Path] = None,
        error_threshold: float = DEFAULT_ERROR_THRESHOLD,
        eval_threshold: float = DEFAULT_EVAL_THRESHOLD,
    ):
        """
        Args:
            benchmark: e.g. "LiveCodeBench"
            model: e.g. "deepseek-chat"
            engine_name: for recalc, e.g. "deepseek-deepseek-chat"
            cache_root: cache root dir
            error_threshold: max error for "valid"
            eval_threshold: min eval for "valid"
        """
        self.benchmark = benchmark
        self.model = model
        self.engine_name = engine_name
        self.cache_root = cache_root or self.DEFAULT_CACHE_ROOT
        self.error_threshold = error_threshold
        self.eval_threshold = eval_threshold
        
        self.cache_dir = self.cache_root / benchmark / model
        
        self.cache_infos: List[CacheInfo] = []
        
        self.problems: Dict[str, Problem] = {}
        
        self._engine = None
        
        self.current_index = 0
    
    def scan_cache(self) -> int:
        """Scan cache dir. Returns number of entries."""
        self.cache_infos = []
        
        if not self.cache_dir.exists():
            print(f"Cache dir missing: {self.cache_dir}")
            return 0
        
        question_dirs = sorted(
            [d for d in self.cache_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name
        )
        
        for question_dir in question_dirs:
            artifact_path = question_dir / "candidates_artifact.json"
            
            cache_info = CacheInfo(
                question_id=question_dir.name,
                cache_path=question_dir
            )
            
            if artifact_path.exists():
                try:
                    with open(artifact_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    artifact = CandidatesArtifact.from_dict(data)
                    cache_info.artifact = artifact
                    cache_info.error_score = artifact.error_score
                    cache_info.eval_score = artifact.eval_score
                    cache_info.schema_count = len(artifact.schema_candidates)
                    cache_info.solution_count = len(artifact.solution_candidates)
                    cache_info.has_best = artifact.best_schema is not None
                    
                except Exception as e:
                    cache_info.load_error = str(e)
            else:
                cache_info.load_error = "artifact file missing"
            
            self.cache_infos.append(cache_info)
        
        return len(self.cache_infos)
    
    def load_problems(self, release_version: str = "release_latest") -> int:
        """Load benchmark problems. Returns count."""
        print("Loading problems...")
        
        try:
            config = create_livecodebench_config(release_version=release_version)
            problems = load_benchmark(config)
            
            self.problems = {p.question_id: p for p in problems}
            print(f"Loaded {len(self.problems)} problems")
            
            return len(self.problems)
            
        except Exception as e:
            print(f"Failed to load problems: {e}")
            return 0
    
    def get_engine(self):
        """Lazy-init engine."""
        if self._engine is None:
            if self.engine_name is None:
                raise ValueError("engine_name required for recalc")
            
            from src.engine import get_engine
            self._engine = get_engine(self.engine_name)
        
        return self._engine
    
    def is_valid(self, cache_info: CacheInfo) -> bool:
        """Whether cache passes thresholds."""
        return (
            cache_info.has_best and
            cache_info.error_score <= self.error_threshold and
            cache_info.eval_score >= self.eval_threshold
        )
    
    def format_status(self, cache_info: CacheInfo) -> str:
        """Human-readable status."""
        if cache_info.load_error:
            return f"❌ error: {cache_info.load_error}"
        elif not cache_info.has_best:
            return "⚠️ no phase-2 result"
        elif self.is_valid(cache_info):
            return "✅ valid"
        else:
            return "⚠️ below threshold"
    
    def display_current(self) -> None:
        """Print current problem."""
        if not self.cache_infos:
            print("No cache data")
            return
        
        cache_info = self.cache_infos[self.current_index]
        
        print("\n" + "=" * 60)
        print(f"[{self.current_index + 1}/{len(self.cache_infos)}] problem: {cache_info.question_id}")
        print("=" * 60)
        
        if cache_info.load_error:
            print(f"  status: {self.format_status(cache_info)}")
        else:
            print(f"  schema candidates: {cache_info.schema_count}")
            print(f"  solution candidates: {cache_info.solution_count}")
            print(f"  Error Score: {cache_info.error_score:.4f} (thresh: {self.error_threshold})")
            print(f"  Eval Score:  {cache_info.eval_score:.4f} (thresh: {self.eval_threshold})")
            print(f"  status: {self.format_status(cache_info)}")
        
        print(f"  cache path: {cache_info.cache_path}")
    
    def list_all(self) -> None:
        """Print score table for all problems."""
        print("\n" + "=" * 80)
        print("All problems — score summary")
        print("=" * 80)
        print(f"{'#':>4} | {'Question ID':<20} | {'Error':>8} | {'Eval':>8} | {'Status':<15}")
        print("-" * 80)
        
        valid_count = 0
        invalid_count = 0
        error_count = 0
        
        for i, cache_info in enumerate(self.cache_infos):
            status = self.format_status(cache_info)
            
            if cache_info.load_error:
                error_count += 1
                print(f"{i+1:>4} | {cache_info.question_id:<20} | {'N/A':>8} | {'N/A':>8} | {status}")
            else:
                if self.is_valid(cache_info):
                    valid_count += 1
                else:
                    invalid_count += 1
                print(f"{i+1:>4} | {cache_info.question_id:<20} | {cache_info.error_score:>8.4f} | {cache_info.eval_score:>8.4f} | {status}")
        
        print("-" * 80)
        print(f"total: {len(self.cache_infos)} | valid: {valid_count} | below thresh: {invalid_count} | errors: {error_count}")
    
    def recalculate_current(self) -> bool:
        """Recalc phase-2 for current problem."""
        if not self.cache_infos:
            print("No cache data")
            return False
        
        cache_info = self.cache_infos[self.current_index]
        question_id = cache_info.question_id
        
        if question_id not in self.problems:
            print(f"Problem {question_id} not in loaded set")
            print("hint: load problems with correct --release-version")
            return False
        
        problem = self.problems[question_id]
        
        print(f"\nRecalculating phase-2 for {question_id}...")
        
        try:
            engine = self.get_engine()
            
            generator = SchemaOracleTestGenerator(
                engine=engine,
                schema_oracle_config=SchemaOracleConfig()
            )
            
            success = generator.initialize(problem, force_recalculate=True)
            
            if success:
                cache_info.artifact = generator._candidates_artifact
                cache_info.error_score = generator.best_error_score
                cache_info.eval_score = generator.best_eval_score
                cache_info.has_best = generator.best_schema is not None
                cache_info.load_error = None
                
                print("Recalc done.")
                print(f"  new Error Score: {cache_info.error_score:.4f}")
                print(f"  new Eval Score:  {cache_info.eval_score:.4f}")
                print(f"  status: {self.format_status(cache_info)}")
                return True
            else:
                print("Recalc failed")
                return False
                
        except Exception as e:
            print(f"Recalc error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def show_help(self) -> None:
        """Print command help."""
        print("\n" + "=" * 40)
        print("Commands")
        print("=" * 40)
        print("  Enter/n  - next")
        print("  p        - previous")
        print("  r        - recalc phase-2 for current")
        print("  g <n>    - go to problem n")
        print("  l        - list all scores")
        print("  h        - this help")
        print("  q        - quit")
        print("=" * 40)
    
    def run(self) -> None:
        """REPL."""
        count = self.scan_cache()
        
        if count == 0:
            print("No cache entries")
            return
        
        print(f"Found {count} cache entries")
        
        self.show_help()
        
        self.display_current()
        
        while True:
            try:
                cmd = input("\ncmd (h=help): ").strip().lower()
                
                if cmd == "" or cmd == "n":
                    if self.current_index < len(self.cache_infos) - 1:
                        self.current_index += 1
                        self.display_current()
                    else:
                        print("Already at last")
                
                elif cmd == "p":
                    if self.current_index > 0:
                        self.current_index -= 1
                        self.display_current()
                    else:
                        print("Already at first")
                
                elif cmd == "r":
                    self.recalculate_current()
                
                elif cmd.startswith("g ") or cmd.startswith("g"):
                    try:
                        if cmd.startswith("g "):
                            n = int(cmd[2:].strip())
                        else:
                            n = int(input("Go to index? ").strip())
                        
                        if 1 <= n <= len(self.cache_infos):
                            self.current_index = n - 1
                            self.display_current()
                        else:
                            print(f"Invalid index; use 1-{len(self.cache_infos)}")
                    except ValueError:
                        print("Enter a valid integer")
                
                elif cmd == "l":
                    self.list_all()
                
                elif cmd == "h":
                    self.show_help()
                
                elif cmd == "q":
                    print("bye")
                    break
                
                else:
                    print(f"Unknown: {cmd}; h for help")
            
            except KeyboardInterrupt:
                print("\nbye")
                break
            except EOFError:
                print("\nbye")
                break


def main():
    parser = argparse.ArgumentParser(
        description="Schema Oracle interactive cache reviewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.test_generators.schema_oracle.interactive_reviewer \\
      --benchmark LiveCodeBench --model deepseek-chat

  With recalc (needs --engine):
  python -m src.test_generators.schema_oracle.interactive_reviewer \\
      --benchmark LiveCodeBench --model deepseek-chat \\
      --engine deepseek-deepseek-chat
"""
    )
    
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        default="LiveCodeBench",
        help="Benchmark name (default: LiveCodeBench)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="deepseek-chat",
        help="Model name (default: deepseek-chat)"
    )
    
    parser.add_argument(
        "--engine", "-e",
        type=str,
        default=None,
        help="Engine for recalc, e.g. deepseek-deepseek-chat"
    )
    
    parser.add_argument(
        "--cache-root",
        type=str,
        default=None,
        help="Cache root (default: results/schemas)"
    )
    
    parser.add_argument(
        "--error-threshold",
        type=float,
        default=0.1,
        help="Max error score for valid (default: 0.1)"
    )
    
    parser.add_argument(
        "--eval-threshold",
        type=float,
        default=0.8,
        help="Min eval score for valid (default: 0.8)"
    )
    
    parser.add_argument(
        "--release-version",
        type=str,
        default="release_latest",
        help="LiveCodeBench release (default: release_latest)"
    )
    
    parser.add_argument(
        "--list-only", "-l",
        action="store_true",
        help="Print score summary only; no REPL"
    )
    
    args = parser.parse_args()
    
    cache_root = Path(args.cache_root) if args.cache_root else None
    
    reviewer = InteractiveReviewer(
        benchmark=args.benchmark,
        model=args.model,
        engine_name=args.engine,
        cache_root=cache_root,
        error_threshold=args.error_threshold,
        eval_threshold=args.eval_threshold,
    )
    
    if args.engine:
        reviewer.load_problems(release_version=args.release_version)
    
    if args.list_only:
        reviewer.scan_cache()
        reviewer.list_all()
    else:
        reviewer.run()


if __name__ == "__main__":
    main()
