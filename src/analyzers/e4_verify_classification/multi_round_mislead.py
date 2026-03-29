"""Count multi-round CSV \"mislead\" rows: Passed=True but more rounds, or TLE+False with more rounds.


# ExeCRE deepseek-V3.2
python -m src.analyzers.e4_verify_classification.multi_round_mislead \
    "results/e1/LiveCodeBench/deepseek-deepseek-chat/ExeCRE/0056a1aa/group_1"

# textgrad deepseek-V3.2
python -m src.analyzers.e4_verify_classification.multi_round_mislead \
    "results/e1/LiveCodeBench/deepseek-deepseek-chat/Textgrad/32f981a9/group_1"

# contested deepseek-V3.2
python -m src.analyzers.e4_verify_classification.multi_round_mislead \
    "results/e1/LiveCodeBench/deepseek-deepseek-chat/ConTested/446ce8f7/group_1"

# ExeCRE gpt-5.2
python -m src.analyzers.e4_verify_classification.multi_round_mislead \
    "results/e1/LiveCodeBench/openai-gpt-5.2/ExeCRE/a7a4d987/group_1"

# textgrad gpt-5.2
python -m src.analyzers.e4_verify_classification.multi_round_mislead \
    "results/e1/LiveCodeBench/openai-gpt-5.2/Textgrad/565dc900/group_1"

# contested gpt-5.2
python -m src.analyzers.e4_verify_classification.multi_round_mislead \
    "results/e1/LiveCodeBench/openai-gpt-5.2/ConTested/05231b55/group_1"
"""

import csv
import json
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Any, List, Optional, Tuple


def analyze_multi_round_mislead(result_path: Path) -> Dict[str, Any]:
    """Scan *.csv under result_path; return condition1_only, condition1_and_2, by_question."""
    if not result_path.exists():
        print(f"Path not found: {result_path}")
        return {
            "condition1_only": {
                "questions_with_mislead": 0,
                "total_mislead_count": 0
            },
            "condition1_and_2": {
                "questions_with_mislead": 0,
                "total_mislead_count": 0
            },
            "by_question": {}
        }
    
    if not result_path.is_dir():
        print(f"Not a directory: {result_path}")
        return {
            "condition1_only": {
                "questions_with_mislead": 0,
                "total_mislead_count": 0
            },
            "condition1_and_2": {
                "questions_with_mislead": 0,
                "total_mislead_count": 0
            },
            "by_question": {}
        }
    
    csv_files = list(result_path.glob("*.csv"))

    if not csv_files:
        print(f"No CSV files: {result_path}")
        return {
            "condition1_only": {
                "questions_with_mislead": 0,
                "total_mislead_count": 0
            },
            "condition1_and_2": {
                "questions_with_mislead": 0,
                "total_mislead_count": 0
            },
            "by_question": {}
        }
    
    condition1_only_questions = set()
    condition1_only_total = 0

    condition1_and_2_questions = set()
    condition1_and_2_total = 0

    by_question = {}

    for csv_file in csv_files:
        question_id = csv_file.stem
        
        try:
            with open(csv_file, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    continue

                if len(rows) <= 1:
                    continue

                question_condition1_count = 0
                question_condition2_count = 0

                for i in range(len(rows) - 1):
                    row = rows[i]
                    row_l = {str(k).strip().lower(): v for k, v in row.items()}
                    passed_str = str(row_l.get("passed", "")).strip().upper()
                    result_type = str(
                        row_l.get("result_type", row_l.get("result type", ""))
                    ).strip()

                    if passed_str == 'TRUE':
                        question_condition1_count += 1
                        condition1_only_total += 1
                        condition1_only_questions.add(question_id)
                        condition1_and_2_total += 1
                        condition1_and_2_questions.add(question_id)

                    if passed_str == 'FALSE' and result_type == 'Time Limit Exceeded':
                        question_condition2_count += 1
                        condition1_and_2_total += 1
                        condition1_and_2_questions.add(question_id)

                if question_condition1_count > 0 or question_condition2_count > 0:
                    by_question[question_id] = {
                        "condition1_count": question_condition1_count,
                        "condition2_count": question_condition2_count
                    }
                    
        except Exception as e:
            print(f"    Failed to read CSV {csv_file.name}: {e}")
            continue
    
    return {
        "condition1_only": {
            "questions_with_mislead": len(condition1_only_questions),
            "total_mislead_count": condition1_only_total
        },
        "condition1_and_2": {
            "questions_with_mislead": len(condition1_and_2_questions),
            "total_mislead_count": condition1_and_2_total
        },
        "by_question": by_question
    }


def print_mislead_results(results: Dict[str, Any], title: str = "Multi-round mislead stats") -> None:
    """Pretty-print analyze_multi_round_mislead output."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print("C1: Passed=TRUE but another round follows")
    print("C2: Passed=FALSE and Result_Type=TLE but another round follows")

    condition1_only = results.get("condition1_only", {})
    condition1_and_2 = results.get("condition1_and_2", {})

    print("\n[C1 only]")
    print("-" * 60)
    print(f"Questions with mislead: {condition1_only.get('questions_with_mislead', 0)}")
    print(f"Total mislead rows: {condition1_only.get('total_mislead_count', 0)}")

    print("\n[C1 + C2]")
    print("-" * 60)
    print(f"Questions with mislead: {condition1_and_2.get('questions_with_mislead', 0)}")
    print(f"Total mislead rows: {condition1_and_2.get('total_mislead_count', 0)}")

    by_question = results.get("by_question", {})
    if by_question:
        print("\n" + "-" * 60)
        print("Per-question detail (first 20):")
        print("-" * 60)
        print(f"{'question_id':<30} {'C1':<15} {'C2':<15}")
        print("-" * 60)

        for i, (question_id, stats) in enumerate(sorted(by_question.items())):
            if i >= 20:
                print(f"... {len(by_question)} questions total")
                break
            print(f"{question_id:<30} {stats['condition1_count']:<15} {stats['condition2_count']:<15}")
    
    print("=" * 60)


def _discover_group_dirs(path: Path, max_groups: int = 5) -> List[Path]:
    """
    If `path` is a run directory containing group_1..group_5, return existing group dirs.
    If `path` itself is a group directory, return [path].
    Otherwise return [].
    """
    if not path.exists() or not path.is_dir():
        return []

    if path.name.startswith("group_"):
        return [path]

    groups: List[Path] = []
    for i in range(1, max_groups + 1):
        g = path / f"group_{i}"
        if g.exists() and g.is_dir():
            groups.append(g)
    return groups


def _extract_core_metrics(results: Dict[str, Any]) -> Dict[str, int]:
    c1 = results.get("condition1_only", {}) or {}
    c12 = results.get("condition1_and_2", {}) or {}
    return {
        "c1_questions": int(c1.get("questions_with_mislead", 0) or 0),
        "c1_total": int(c1.get("total_mislead_count", 0) or 0),
        "c12_questions": int(c12.get("questions_with_mislead", 0) or 0),
        "c12_total": int(c12.get("total_mislead_count", 0) or 0),
    }


def analyze_groups_and_aggregate(group_dirs: List[Path]) -> Dict[str, Any]:
    per_group: Dict[str, Dict[str, Any]] = {}
    rows: List[Dict[str, int]] = []

    for g in group_dirs:
        csv_count = len(list(g.glob("*.csv")))
        print(f"  - {g.name}: {csv_count} csv file(s)")
        r = analyze_multi_round_mislead(g)
        per_group[g.name] = r
        rows.append(_extract_core_metrics(r))

    if not rows:
        return {"per_group": per_group, "aggregate": None}

    def agg(key: str) -> Dict[str, float]:
        vals = [row[key] for row in rows]
        if len(vals) == 1:
            return {"mean": float(vals[0]), "std": 0.0, "n": 1}
        # Use population std (pstdev) since we usually have all groups, not a sample.
        return {"mean": float(mean(vals)), "std": float(pstdev(vals)), "n": len(vals)}

    aggregate = {
        "condition1_only": {
            "questions_with_mislead": agg("c1_questions"),
            "total_mislead_count": agg("c1_total"),
        },
        "condition1_and_2": {
            "questions_with_mislead": agg("c12_questions"),
            "total_mislead_count": agg("c12_total"),
        },
    }

    return {"per_group": per_group, "aggregate": aggregate}


def _project_root() -> Path:
    # .../TrustTest/src/analyzers/e4_verify_classification/multi_round_mislead.py
    # parents[0]=e4_verify_classification, [1]=analyzers, [2]=src, [3]=TrustTest
    return Path(__file__).resolve().parents[3]


def _results_root(project_root: Path) -> Path:
    # Most runs are stored alongside the repo, e.g. TrustTest/../results/...
    return project_root.parent / "results"


def _candidate_results_roots(project_root: Path) -> List[Path]:
    """
    Try common layouts:
    - <repo>/../results
    - <repo>/../../results   (e.g. ~/results when repo is ~/MyProject/TrustTest)
    - <repo>/results
    - <cwd>/results and <cwd>/../results
    """
    roots: List[Path] = []

    roots.append(project_root.parent / "results")
    roots.append(project_root.parent.parent / "results")
    roots.append(project_root / "results")

    cwd = Path.cwd()
    roots.append(cwd / "results")
    roots.append(cwd.parent / "results")

    # de-dup while preserving order
    out: List[Path] = []
    seen = set()
    for r in roots:
        key = str(r.resolve()) if r.exists() else str(r)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _tokenize_model_name(model_name: str) -> List[str]:
    raw = re.split(r"[^a-zA-Z0-9]+", (model_name or "").lower())
    toks = []
    for t in raw:
        if len(t) >= 3:
            toks.append(t)
    # De-dup while preserving order
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _score_path(path_str_lower: str, tokens: List[str]) -> int:
    # simple heuristic: count distinct token hits
    hits = 0
    for t in tokens:
        if t in path_str_lower:
            hits += 1
    return hits


def _count_run_csvs(run_dir: Path, max_groups: int = 5) -> Tuple[int, int]:
    """
    Count total csv files across group_1..group_5 under a run directory.
    Returns (group_count_found, total_csv_count).
    """
    group_count = 0
    total_csv = 0
    for i in range(1, max_groups + 1):
        g = run_dir / f"group_{i}"
        if not g.exists() or not g.is_dir():
            continue
        group_count += 1
        try:
            total_csv += len(list(g.glob("*.csv")))
        except Exception:
            pass
    return group_count, total_csv


def _find_best_group_dir(
    *,
    results_root: Path,
    experiment_id: str,
    benchmark: str,
    baseline: str,
    model_name: str,
) -> Optional[Path]:
    base_dir = results_root / experiment_id / benchmark
    if not base_dir.exists():
        return None

    # Look for .../<model_dir>/<baseline>/<run_id>/group_1
    candidates = list(base_dir.glob(f"**/{baseline}/*/group_1"))
    if not candidates:
        return None

    tokens = _tokenize_model_name(model_name)
    if not tokens:
        # fall back to "most complete" run, then newest
        best = None
        best_key = (-1, -1, -1.0)  # total_csv, group_count, mtime
        for c in candidates:
            run_dir = c.parent
            group_count, total_csv = _count_run_csvs(run_dir)
            try:
                mtime = run_dir.stat().st_mtime
            except Exception:
                mtime = 0.0
            key = (total_csv, group_count, mtime)
            if key > best_key:
                best_key = key
                best = c
        return best

    scored: List[Tuple[int, int, int, float, Path]] = []
    for c in candidates:
        s = _score_path(str(c).lower(), tokens)
        run_dir = c.parent
        group_count, total_csv = _count_run_csvs(run_dir)
        try:
            mtime = run_dir.stat().st_mtime
        except Exception:
            mtime = 0.0
        # Prefer: token hits > total csv count > group count > newest
        scored.append((s, total_csv, group_count, mtime, c))

    # Prefer higher token-hit score, break ties by newest
    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    best_score = scored[0][0]
    if best_score == 0:
        # If nothing matches model tokens, pick most complete run anyway.
        best = None
        best_key = (-1, -1, -1.0)  # total_csv, group_count, mtime
        for c in candidates:
            run_dir = c.parent
            group_count, total_csv = _count_run_csvs(run_dir)
            try:
                mtime = run_dir.stat().st_mtime
            except Exception:
                mtime = 0.0
            key = (total_csv, group_count, mtime)
            if key > best_key:
                best_key = key
                best = c
        return best
    return scored[0][4]


def _auto_locate_group_dir_from_config(
    *,
    config_path: Path,
    baseline_override: Optional[str] = None,
) -> Optional[Path]:
    cfg = _load_config(config_path)
    exp = cfg.get("experiment", {}) or {}
    model = cfg.get("model", {}) or {}

    experiment_id = str(exp.get("experiment_id", "e1"))
    benchmark = str(exp.get("benchmark", "LiveCodeBench"))
    baseline = str(baseline_override or exp.get("baseline", "ExeCRE"))
    model_name = str(model.get("model", ""))

    project_root = _project_root()
    for results_root in _candidate_results_roots(project_root):
        guessed = _find_best_group_dir(
            results_root=results_root,
            experiment_id=experiment_id,
            benchmark=benchmark,
            baseline=baseline,
            model_name=model_name,
        )
        if guessed is not None:
            # guessed is .../<baseline>/<run_id>/group_1; return run dir for group aggregation
            return guessed.parent
    return None


def _load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-round mislead stats from result CSVs")
    parser.add_argument("result_path", type=str, nargs="?", default=None, help="Result directory (optional)")
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Optional experiment JSON; auto-locate run dir and aggregate group_1..5.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Override baseline in config (e.g. ExeCRE, Textgrad).",
    )
    
    args = parser.parse_args()

    result_path: Optional[Path] = Path(args.result_path) if args.result_path else None

    if result_path is None and args.config_file:
        config_path = Path(args.config_file)
        if not config_path.exists():
            print(f"Config not found: {config_path}")
            return 1

        guessed = _auto_locate_group_dir_from_config(
            config_path=config_path,
            baseline_override=args.baseline,
        )

        if guessed is None:
            print("Could not auto-locate result_path. Pass it as first arg or put results next to repo.")
            return 1

        result_path = guessed

    if result_path is None:
        print("Provide result_path or --config_file.")
        print('  python -m ... multi_round_mislead "results/.../group_1"')
        print("  python -m ... multi_round_mislead --config_file configs/....json")
        return 1

    if not result_path.exists():
        print(f"Path not found: {result_path}")
        return 1

    group_dirs = _discover_group_dirs(result_path, max_groups=5)
    if not group_dirs:
        print(f"No group_1..group_5 under: {result_path}")
        print("Pass a run dir with group_* or a single group_k directory.")
        return 1

    print(f"Path: {result_path}")
    print(f"Groups: {', '.join([g.name for g in group_dirs])}")

    if len(group_dirs) == 1:
        g = group_dirs[0]
        csvs = sorted(g.glob("*.csv"))
        print(f"  - {g.name}: {len(csvs)} csv file(s)")
        if csvs:
            preview = ", ".join([p.name for p in csvs[:5]])
            print(f"    e.g. {preview}" + (" ..." if len(csvs) > 5 else ""))
        results = analyze_multi_round_mislead(group_dirs[0])
        print_mislead_results(results, title=f"Mislead stats ({group_dirs[0].name})")
        return 0

    bundle = analyze_groups_and_aggregate(group_dirs)
    agg = bundle.get("aggregate")
    if not agg:
        print("No groups to aggregate")
        return 1

    print("\n[Across groups: mean ± std]")
    print("-" * 60)
    for label, key_prefix in [("C1 only", "condition1_only"), ("C1+C2", "condition1_and_2")]:
        q = agg[key_prefix]["questions_with_mislead"]
        t = agg[key_prefix]["total_mislead_count"]
        print(f"{label} (n={int(q['n'])} groups)")
        print(f"  questions with mislead: {q['mean']:.2f} ± {q['std']:.2f}")
        print(f"  total mislead rows: {t['mean']:.2f} ± {t['std']:.2f}")
    
    return 0


if __name__ == "__main__":
    exit(main())
