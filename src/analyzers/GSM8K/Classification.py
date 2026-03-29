"""GSM8K ExeCRE: threshold alpha vs numeric code correctness from result CSVs."""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Iterable, Tuple

from src.config import ExperimentConfig
from src.utils.path_manager import get_group_dir


# region agent log: GSM8K Classification instrumentation
def _agent_debug_log(payload: dict) -> None:
    """Append one NDJSON line to debug-97fe63.log; swallow errors."""
    try:
        base_payload = {
            "sessionId": "97fe63",
            "runId": payload.get("runId", "initial"),
            "hypothesisId": payload.get("hypothesisId", "H_all"),
            "location": payload.get(
                "location", "src/analyzers/GSM8K/Classification.py"
            ),
            "message": payload.get("message", ""),
            "data": payload.get("data", {}),
        }
        base_payload["timestamp"] = payload.get(
            "timestamp", int(__import__("time").time() * 1000)
        )
        with open("debug-97fe63.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(base_payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


# endregion


@dataclass
class QuestionResult:
    """One GSM8K row: alpha, fallback flag, whether code output matches gold."""

    question_id: str
    alpha: float
    fallback: bool
    code_correct: bool


@dataclass
class MetricsAtThreshold:
    """Binary metrics at one alpha cutoff."""

    threshold: float
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0


@dataclass
class AggregatedMetricsAtThreshold:
    """Across groups: mean±std per metric + micro-pooled counts."""

    threshold: float
    n_groups: int

    precision_mean: float = 0.0
    precision_std: float = 0.0
    recall_mean: float = 0.0
    recall_std: float = 0.0
    f1_mean: float = 0.0
    f1_std: float = 0.0
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0

    pooled_tp: int = 0
    pooled_fp: int = 0
    pooled_tn: int = 0
    pooled_fn: int = 0
    pooled_precision: float = 0.0
    pooled_recall: float = 0.0
    pooled_f1: float = 0.0
    pooled_accuracy: float = 0.0


def _to_float(value) -> Optional[float]:
    """Best-effort float parse; None on failure."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        s = s.replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _numeric_equal(
    a,
    b,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3,
    int_tol: float = 1e-6,
) -> bool:
    """Near-equality: integer rounding if both near ints, else abs/rel tolerance."""
    fa = _to_float(a)
    fb = _to_float(b)
    if fa is None or fb is None:
        return False
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return False

    ra = round(fa)
    rb = round(fb)
    if abs(fa - ra) <= int_tol and abs(fb - rb) <= int_tol:
        return ra == rb

    diff = abs(fa - fb)
    if diff <= abs_tol:
        return True
    scale = max(abs(fa), abs(fb))
    if scale == 0:
        return diff <= abs_tol
    return diff <= max(abs_tol, rel_tol * scale)


def _load_question_results_from_group(
    config: ExperimentConfig,
    group_n: int = 1,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3,
) -> List[QuestionResult]:
    """Load gsm8k-*.csv under group dir; gold from Error_Case_Contents."""
    if config.experiment.benchmark in ("GSM8K", "GSM8KExeCRE"):
        benchmark_for_results = "GSM8K"
    else:
        benchmark_for_results = config.experiment.benchmark

    results_dir = get_group_dir(
        experiment_id=config.experiment.experiment_id,
        benchmark=benchmark_for_results,
        model=config.model.model,
        baseline=config.experiment.baseline,
        config_hash=config.get_config_hash(),
        group_n=group_n,
    )

    if not results_dir.exists():
        print(f"Results directory missing: {results_dir}")
        return []

    question_results: List[QuestionResult] = []

    total_rows = 0
    rows_with_gold = 0
    rows_with_executed_answer = 0
    code_correct_true = 0
    code_correct_false = 0
    fallback_true = 0

    csv_files = sorted(results_dir.glob("gsm8k-*.csv"))
    debug_sampled = 0

    for csv_path in csv_files:
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    total_rows += 1

                    qid = (
                        row.get("Question_Id")
                        or row.get("Question_ID")
                        or row.get("question_id")
                        or row.get("QuestionID")
                        or row.get("questionID")
                        or ""
                    )
                    if not qid:
                        for k in row.keys():
                            normalized = (
                                k.strip()
                                .lstrip("\ufeff")
                                .lower()
                                .replace("_", "")
                            )
                            if normalized in ("questionid", "questionid "):
                                candidate = row.get(k) or ""
                                if candidate:
                                    qid = candidate
                                    break
                    if not qid:
                        continue

                    meta_raw = row.get("Metadata", "") or ""
                    try:
                        metadata = json.loads(meta_raw) if meta_raw.strip() else {}
                    except json.JSONDecodeError:
                        metadata = {}

                    alpha = float(metadata.get("alpha", 0.0))
                    fallback = bool(metadata.get("fallback", False))
                    executed_answer = metadata.get("executed_answer", None)
                    if executed_answer is not None:
                        rows_with_executed_answer += 1

                    prediction = None
                    gold = None
                    err = (row.get("Error_Case_Contents") or "").strip()
                    if "prediction=" in err:
                        try:
                            p_parts = err.split("prediction=", 1)[1]
                            p_token = p_parts.split(",", 1)[0].strip()
                            prediction = _to_float(p_token)
                        except Exception:
                            prediction = None
                    if "gold=" in err:
                        rows_with_gold += 1
                        try:
                            parts = err.split("gold=", 1)[1]
                            token = parts.split(",", 1)[0].strip()
                            gold = _to_float(token)
                        except Exception:
                            gold = None
                    if gold is None:
                        continue

                    if debug_sampled < 5:
                        _agent_debug_log(
                            {
                                "runId": "initial",
                                "hypothesisId": "H_schema",
                                "location": "src/analyzers/GSM8K/Classification.py:_load_question_results_from_group",
                                "message": "sample row for schema inspection",
                                "data": {
                                    "question_id": qid,
                                    "alpha": alpha,
                                    "fallback": fallback,
                                    "has_executed_answer": executed_answer is not None,
                                    "error_case_prefix": err[:200],
                                    "row_keys": list(row.keys()),
                                    "passed_field": row.get("Passed", ""),
                                },
                            }
                        )
                        debug_sampled += 1

                    code_output = prediction
                    if code_output is None:
                        code_output = executed_answer

                    code_correct = _numeric_equal(
                        code_output,
                        gold,
                        abs_tol=abs_tol,
                        rel_tol=rel_tol,
                    )

                    if code_correct:
                        code_correct_true += 1
                    else:
                        code_correct_false += 1

                    if fallback:
                        fallback_true += 1

                    question_results.append(
                        QuestionResult(
                            question_id=qid,
                            alpha=alpha,
                            fallback=fallback,
                            code_correct=code_correct,
                        )
                    )
        except Exception as e:
            print(f"Failed to parse CSV {csv_path}: {e}")

    _agent_debug_log(
        {
            "runId": "initial",
            "hypothesisId": "H_all",
            "location": "src/analyzers/GSM8K/Classification.py:_load_question_results_from_group",
            "message": "GSM8KExeCRE row / label statistics",
            "data": {
                "total_rows": total_rows,
                "rows_with_gold": rows_with_gold,
                "rows_with_executed_answer": rows_with_executed_answer,
                "question_results_len": len(question_results),
                "code_correct_true": code_correct_true,
                "code_correct_false": code_correct_false,
                "fallback_true": fallback_true,
            },
        }
    )

    return question_results


def analyze_gsm8k_execre_confidence(
    config: ExperimentConfig,
    thresholds: Optional[Sequence[float]] = None,
    group_n: int = 1,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3,
) -> List[MetricsAtThreshold]:
    """Predict positive if alpha>=thr; label = code_correct; return metrics per threshold."""
    if config.experiment.benchmark not in ("GSM8K", "GSM8KExeCRE"):
        print(f"Note: benchmark={config.experiment.benchmark}; intended for GSM8K / GSM8KExeCRE")

    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]

    question_results = _load_question_results_from_group(
        config=config,
        group_n=group_n,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )
    if not question_results:
        print("No question rows loaded; aborting analysis")
        return []

    metrics_list: List[MetricsAtThreshold] = []

    for thr in thresholds:
        m = MetricsAtThreshold(threshold=float(thr))
        for r in question_results:
            actual_positive = r.code_correct
            predicted_positive = r.alpha >= thr

            if predicted_positive and actual_positive:
                m.tp += 1
            elif predicted_positive and not actual_positive:
                m.fp += 1
            elif (not predicted_positive) and (not actual_positive):
                m.tn += 1
            else:
                m.fn += 1

        total = m.tp + m.fp + m.tn + m.fn
        if total > 0:
            m.accuracy = (m.tp + m.tn) / total
        if (m.tp + m.fp) > 0:
            m.precision = m.tp / (m.tp + m.fp)
        if (m.tp + m.fn) > 0:
            m.recall = m.tp / (m.tp + m.fn)
        if (m.precision + m.recall) > 0:
            m.f1_score = 2 * m.precision * m.recall / (m.precision + m.recall)

        metrics_list.append(m)

    return metrics_list


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """Sample mean and std; n<2 -> std 0."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def analyze_gsm8k_execre_confidence_multi_groups(
    config: ExperimentConfig,
    thresholds: Optional[Sequence[float]] = None,
    group_ns: Optional[Sequence[int]] = None,
    abs_tol: float = 1e-4,
    rel_tol: float = 1e-3,
) -> List[AggregatedMetricsAtThreshold]:
    """Run per-group analyze_gsm8k_execre_confidence then pool + mean±std by threshold index."""
    if thresholds is None:
        thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]

    if group_ns is None:
        n = getattr(getattr(config, "experiment", None), "n", None)
        if isinstance(n, int) and n >= 1:
            group_ns = list(range(1, n + 1))
        else:
            group_ns = [1]

    per_group_metrics: List[List[MetricsAtThreshold]] = []
    for g in group_ns:
        ms = analyze_gsm8k_execre_confidence(
            config=config,
            thresholds=thresholds,
            group_n=int(g),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        )
        if ms:
            per_group_metrics.append(ms)

    if not per_group_metrics:
        return []

    agg: List[AggregatedMetricsAtThreshold] = []
    n_groups = len(per_group_metrics)

    for i, thr in enumerate(thresholds):
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []
        accs: List[float] = []
        pooled_tp = pooled_fp = pooled_tn = pooled_fn = 0

        for ms in per_group_metrics:
            if i >= len(ms):
                continue
            m = ms[i]
            has_pred_pos = (m.tp + m.fp) > 0
            has_actual_pos = (m.tp + m.fn) > 0
            total_m = m.tp + m.fp + m.tn + m.fn
            if has_pred_pos:
                precisions.append(float(m.precision))
            if has_actual_pos:
                recalls.append(float(m.recall))
            if has_pred_pos and has_actual_pos:
                f1s.append(float(m.f1_score))
            if total_m > 0:
                accs.append(float(m.accuracy))
            pooled_tp += int(m.tp)
            pooled_fp += int(m.fp)
            pooled_tn += int(m.tn)
            pooled_fn += int(m.fn)

        precision_mean, precision_std = _mean_std(precisions)
        recall_mean, recall_std = _mean_std(recalls)
        f1_mean, f1_std = _mean_std(f1s)
        accuracy_mean, accuracy_std = _mean_std(accs)

        total = pooled_tp + pooled_fp + pooled_tn + pooled_fn
        pooled_accuracy = (pooled_tp + pooled_tn) / total if total > 0 else 0.0
        pooled_precision = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) > 0 else 0.0
        pooled_recall = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) > 0 else 0.0
        pooled_f1 = (
            2 * pooled_precision * pooled_recall / (pooled_precision + pooled_recall)
            if (pooled_precision + pooled_recall) > 0
            else 0.0
        )

        agg.append(
            AggregatedMetricsAtThreshold(
                threshold=float(thr),
                n_groups=n_groups,
                precision_mean=precision_mean,
                precision_std=precision_std,
                recall_mean=recall_mean,
                recall_std=recall_std,
                f1_mean=f1_mean,
                f1_std=f1_std,
                accuracy_mean=accuracy_mean,
                accuracy_std=accuracy_std,
                pooled_tp=pooled_tp,
                pooled_fp=pooled_fp,
                pooled_tn=pooled_tn,
                pooled_fn=pooled_fn,
                pooled_precision=pooled_precision,
                pooled_recall=pooled_recall,
                pooled_f1=pooled_f1,
                pooled_accuracy=pooled_accuracy,
            )
        )

    return agg


def print_gsm8k_confidence_results_multi_groups(
    aggregated: Sequence[AggregatedMetricsAtThreshold],
) -> None:
    """Print pooled + mean±std table for multi-group run."""
    if not aggregated:
        print("No multi-group GSM8K ExeCRE results to print")
        return

    w_thr = 8
    w_g = 4
    w_ms = 17
    w_cnt = 7
    w_pooled = 10

    header = (
        f"{'thr':>{w_thr}}  {'g':>{w_g}}  "
        f"{'P(mean±std)':>{w_ms}}  {'R(mean±std)':>{w_ms}}  {'F1(mean±std)':>{w_ms}}  {'Acc(mean±std)':>{w_ms}}  "
        f"{'TP':>{w_cnt}}  {'FP':>{w_cnt}}  {'TN':>{w_cnt}}  {'FN':>{w_cnt}}  "
        f"{'P(pool)':>{w_pooled}}  {'R(pool)':>{w_pooled}}  {'F1(pool)':>{w_pooled}}  {'Acc(pool)':>{w_pooled}}"
    )
    line_len = len(header)

    print("\n" + "=" * line_len)
    print("GSM8K ExeCRE — multi-group metrics by alpha")
    print("mean±std: metric per group then average; pooled: sum counts then micro-F1 etc.")
    print("=" * line_len)
    print(header)
    print("-" * line_len)

    for m in aggregated:
        p_ms = f"{m.precision_mean:.4f}±{m.precision_std:.4f}"
        r_ms = f"{m.recall_mean:.4f}±{m.recall_std:.4f}"
        f1_ms = f"{m.f1_mean:.4f}±{m.f1_std:.4f}"
        a_ms = f"{m.accuracy_mean:.4f}±{m.accuracy_std:.4f}"
        print(
            f"{m.threshold:>{w_thr}.3f}  {m.n_groups:>{w_g}d}  "
            f"{p_ms:>{w_ms}}  {r_ms:>{w_ms}}  {f1_ms:>{w_ms}}  {a_ms:>{w_ms}}  "
            f"{m.pooled_tp:>{w_cnt}d}  {m.pooled_fp:>{w_cnt}d}  {m.pooled_tn:>{w_cnt}d}  {m.pooled_fn:>{w_cnt}d}  "
            f"{m.pooled_precision:>{w_pooled}.4f}  {m.pooled_recall:>{w_pooled}.4f}  {m.pooled_f1:>{w_pooled}.4f}  {m.pooled_accuracy:>{w_pooled}.4f}"
        )

    print("=" * line_len)


def print_gsm8k_confidence_results(metrics_list: Sequence[MetricsAtThreshold]) -> None:
    """Print confusion-style table for single-group metrics."""
    if not metrics_list:
        print("No GSM8K ExeCRE results to print")
        return

    print("\n" + "=" * 120)
    print("GSM8K ExeCRE — EM4C confidence vs code correctness (by alpha)")
    print("=" * 120)
    header = (
        f"{'thr':<10} {'TP':<6} {'FP':<6} {'TN':<6} {'FN':<6} "
        f"{'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}"
    )
    print(header)
    print("-" * 120)

    for m in metrics_list:
        print(
            f"{m.threshold:<10.3f} "
            f"{m.tp:<6d} {m.fp:<6d} {m.tn:<6d} {m.fn:<6d} "
            f"{m.precision:<12.4f} {m.recall:<12.4f} {m.f1_score:<12.4f} {m.accuracy:<12.4f}"
        )

    print("=" * 120)
