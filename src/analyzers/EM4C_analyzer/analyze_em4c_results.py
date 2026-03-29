"""Read-only EM4C run analysis: alpha thresholds vs tlefree labels (confusion / F1).

Example:
    python -m src.analyzers.EM4C_analyzer.analyze_em4c_results --config <path>.json
"""

import math
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence
from dataclasses import dataclass
from collections import defaultdict

project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import ExperimentConfig, load_config_from_file
from src.utils.path_manager import get_group_dir


@dataclass
class QuestionResult:
    """Per-question EM4C pick vs brute-force labels."""
    question_id: str
    em_selected_index: int
    em_alpha: float
    ground_truths: Dict[int, bool]
    metadata: Optional[Dict[str, Any]] = None
    difficulty: str = "unknown"


@dataclass
class MetricsAtThreshold:
    """Binary metrics at one alpha threshold."""
    threshold: float
    tp: int = 0  # True Positive
    fp: int = 0  # False Positive
    tn: int = 0  # True Negative
    fn: int = 0  # False Negative
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    accuracy: float = 0.0


@dataclass
class AggregatedMetricsAtThreshold:
    """Across groups: mean±std per metric + pooled counts."""
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


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    return mean, math.sqrt(var)


def format_probability(value: float) -> str:
    """Format probabilities near 1 as 1-epsilon scientific notation."""
    if value >= 0.999:
        epsilon = 1 - value
        if epsilon > 0:
            return f"1-{epsilon:.2e}"
        else:
            return "1.0"
    elif value <= 0.001:
        if value > 0:
            return f"{value:.2e}"
        else:
            return "0.0"
    else:
        return f"{value:.6f}"


class EM4CResultsAnalyzer:
    """Aggregate EM4C metadata + tlefree_evaluation per question."""

    def __init__(
        self,
        config: ExperimentConfig,
        group_n: int = 1,
        difficulty_map: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ):
        self.config = config
        self.group_n = group_n
        self.difficulty_map = difficulty_map or {}
        self.verbose = verbose

        self.results_dir = get_group_dir(
            experiment_id=config.experiment.experiment_id,
            benchmark=config.experiment.benchmark,
            model=config.model.model,
            baseline=config.experiment.baseline,
            config_hash=config.get_config_hash(),
            group_n=group_n
        )
        
        self.results: List[QuestionResult] = []
    
    def load_tlefree_evaluation(self, question_dir: Path) -> Optional[Dict[int, bool]]:
        """Read tlefree_evaluation.json -> {solution_index: bool}; None if missing."""
        tlefree_file = question_dir / "EM4C" / "bruteforce_solutions" / "tlefree_evaluation.json"
        
        if not tlefree_file.exists():
            return None
        
        try:
            with open(tlefree_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            ground_truths = {}
            for key, value in data.items():
                if key.startswith("solution_"):
                    try:
                        solution_index = int(key.split("_")[1])
                        ground_truths[solution_index] = bool(value) if value is not None else False
                    except (ValueError, IndexError):
                        continue
            
            return ground_truths
        except Exception as e:
            if self.verbose:
                print(f"  Failed to read tlefree_evaluation.json {tlefree_file}: {e}")
            return None

    def load_metadata(self, question_dir: Path) -> Optional[Dict[str, Any]]:
        """Read metadata.json or None."""
        metadata_file = question_dir / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            if self.verbose:
                print(f"  Failed to read metadata.json {metadata_file}: {e}")
            return None

    def analyze_one_question(self, question_id: str) -> Optional[QuestionResult]:
        """Build QuestionResult for one question folder or None."""
        question_dir = self.results_dir / question_id

        if not question_dir.exists():
            if self.verbose:
                print(f"  {question_id}: directory missing")
            return None

        ground_truths = self.load_tlefree_evaluation(question_dir)
        if ground_truths is None:
            if self.verbose:
                print(f"  {question_id}: no tlefree_evaluation.json")
            return None

        metadata = self.load_metadata(question_dir)
        if metadata is None:
            if self.verbose:
                print(f"  {question_id}: no metadata.json")
            return None

        selected_solution_index = metadata.get("selected_solution_index", -1)
        alpha = metadata.get("alpha", 0.0)

        if selected_solution_index < 0:
            if self.verbose:
                print(f"  {question_id}: invalid selected_solution_index")
            return None
        
        return QuestionResult(
            question_id=question_id,
            em_selected_index=selected_solution_index,
            em_alpha=alpha,
            ground_truths=ground_truths,
            metadata=metadata,
            difficulty=self.difficulty_map.get(question_id, "unknown"),
        )
    
    def run_analysis(self) -> None:
        """Scan all question subdirs under results_dir."""
        if not self.results_dir.exists():
            if self.verbose:
                print(f"Results directory missing: {self.results_dir}")
            return

        if self.verbose:
            print(f"Results dir: {self.results_dir}")

        question_dirs = [d for d in self.results_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

        if not question_dirs:
            if self.verbose:
                print("No question directories found")
            return

        if self.verbose:
            print(f"\nAnalyzing {len(question_dirs)} questions...")

        for i, question_dir in enumerate(sorted(question_dirs), 1):
            question_id = question_dir.name
            if self.verbose:
                print(f"[{i}/{len(question_dirs)}] {question_id}")

            result = self.analyze_one_question(question_id)
            if result:
                self.results.append(result)
                if self.verbose:
                    print(
                        f"  ok: alpha={format_probability(result.em_alpha)}, "
                        f"selected_solution_{result.em_selected_index}, "
                        f"ground_truth={result.ground_truths.get(result.em_selected_index, False)}"
                    )
            elif self.verbose:
                print(f"  skipped {question_id}")

        if self.verbose:
            print(f"\nDone: {len(self.results)} questions")
    
    def calculate_metrics_at_thresholds(
        self,
        thresholds: List[float]
    ) -> List[MetricsAtThreshold]:
        """Per threshold: predict positive if alpha>=thr on EM-selected solution vs tlefree label."""
        metrics_list = []

        for threshold in thresholds:
            metrics = MetricsAtThreshold(threshold=threshold)

            for result in self.results:
                actual_positive = result.ground_truths.get(result.em_selected_index, False)

                predicted_positive = result.em_alpha >= threshold

                if predicted_positive and actual_positive:
                    metrics.tp += 1
                elif predicted_positive and not actual_positive:
                    metrics.fp += 1
                elif not predicted_positive and not actual_positive:
                    metrics.tn += 1
                else:
                    metrics.fn += 1

            total = metrics.tp + metrics.fp + metrics.tn + metrics.fn
            if total > 0:
                metrics.accuracy = (metrics.tp + metrics.tn) / total
            
            if metrics.tp + metrics.fp > 0:
                metrics.precision = metrics.tp / (metrics.tp + metrics.fp)
            
            if metrics.tp + metrics.fn > 0:
                metrics.recall = metrics.tp / (metrics.tp + metrics.fn)
            
            if metrics.precision + metrics.recall > 0:
                metrics.f1_score = 2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
            
            metrics_list.append(metrics)
        
        return metrics_list
    
    def print_confusion_matrix_table(self, metrics_list: List[MetricsAtThreshold]) -> None:
        """Print confusion counts and metrics per threshold."""
        print("\n" + "=" * 120)
        print("Confusion matrix / metrics by alpha threshold")
        print("=" * 120)
        print(f"{'thr':<30} {'TP':<8} {'FP':<8} {'TN':<8} {'FN':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Accuracy':<12}")
        print("-" * 120)

        for metrics in metrics_list:
            threshold = metrics.threshold
            if threshold >= 0.999:
                epsilon = 1 - threshold
                if epsilon > 0:
                    threshold_str = f"1-{epsilon:.2e}"
                else:
                    threshold_str = "1.0"
            elif threshold >= 0.99:
                threshold_str = f"{threshold:.18f}"
            else:
                threshold_str = f"{threshold:.10f}"
            
            print(
                f"{threshold_str:<30} "
                f"{metrics.tp:<8} "
                f"{metrics.fp:<8} "
                f"{metrics.tn:<8} "
                f"{metrics.fn:<8} "
                f"{metrics.precision:<12.4f} "
                f"{metrics.recall:<12.4f} "
                f"{metrics.f1_score:<12.4f} "
                f"{metrics.accuracy:<12.4f}"
            )
        
        print("=" * 120)
    
    def print_summary(self) -> None:
        """Short dataset summary."""
        if not self.results:
            print("No results")
            return

        total = len(self.results)

        correct_predictions = sum(
            1 for r in self.results
            if r.ground_truths.get(r.em_selected_index, False)
        )

        alphas = [r.em_alpha for r in self.results]

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"Questions: {total}")
        print(f"EM-selected solution label True: {correct_predictions} ({correct_predictions/total*100:.1f}%)")

        if alphas:
            print(f"\nAlpha:")
            print(f"  mean: {sum(alphas)/len(alphas):.4f}")
            print(f"  min: {min(alphas):.4f}")
            print(f"  max: {max(alphas):.4f}")

        solution_truth_counts = {}
        for result in self.results:
            for solution_index, truth_value in result.ground_truths.items():
                if solution_index not in solution_truth_counts:
                    solution_truth_counts[solution_index] = {"true": 0, "false": 0}
                if truth_value:
                    solution_truth_counts[solution_index]["true"] += 1
                else:
                    solution_truth_counts[solution_index]["false"] += 1
        
        print(f"\nPer-solution label counts:")
        for solution_index in sorted(solution_truth_counts.keys()):
            counts = solution_truth_counts[solution_index]
            total_count = counts["true"] + counts["false"]
            if total_count > 0:
                print(f"  solution_{solution_index}: True={counts['true']}, False={counts['false']}, "
                      f"True%={counts['true']/total_count*100:.1f}%")
        
        print("=" * 80)

    def calculate_metrics_by_difficulty(
        self,
        thresholds: List[float],
    ) -> Dict[str, List[MetricsAtThreshold]]:
        """Same as calculate_metrics_at_thresholds but split by difficulty."""
        grouped: Dict[str, List[QuestionResult]] = defaultdict(list)
        for r in self.results:
            grouped[r.difficulty].append(r)

        result: Dict[str, List[MetricsAtThreshold]] = {}
        for difficulty, questions in grouped.items():
            metrics_list: List[MetricsAtThreshold] = []
            for thr in thresholds:
                m = MetricsAtThreshold(threshold=thr)
                for r in questions:
                    actual_positive = r.ground_truths.get(r.em_selected_index, False)
                    predicted_positive = r.em_alpha >= thr
                    if predicted_positive and actual_positive:
                        m.tp += 1
                    elif predicted_positive and not actual_positive:
                        m.fp += 1
                    elif not predicted_positive and not actual_positive:
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
            result[difficulty] = metrics_list
        return result


def print_em4c_difficulty_results(
    by_difficulty: Dict[str, List[MetricsAtThreshold]],
) -> None:
    """Print per-difficulty metric tables."""
    if not by_difficulty:
        print("No per-difficulty data")
        return

    for difficulty in sorted(by_difficulty.keys()):
        metrics_list = by_difficulty[difficulty]
        n_problems = 0
        if metrics_list:
            first = metrics_list[0]
            n_problems = first.tp + first.fp + first.tn + first.fn

        print(f"\n{'=' * 100}")
        print(f"Difficulty: {difficulty}  (n={n_problems})")
        print(f"{'=' * 100}")
        print(
            f"{'thr':>8}  {'TP':>6}  {'FP':>6}  {'TN':>6}  {'FN':>6}  "
            f"{'Prec':>8}  {'Recall':>8}  {'F1':>8}  {'Acc':>8}"
        )
        print("-" * 100)
        for m in metrics_list:
            print(
                f"{m.threshold:>8.3f}  {m.tp:>6d}  {m.fp:>6d}  {m.tn:>6d}  {m.fn:>6d}  "
                f"{m.precision:>8.4f}  {m.recall:>8.4f}  {m.f1_score:>8.4f}  {m.accuracy:>8.4f}"
            )
        print("=" * 100)


def analyze_em4c_multi_groups(
    config: ExperimentConfig,
    thresholds: List[float],
    group_ns: Optional[Sequence[int]] = None,
    difficulty_map: Optional[Dict[str, str]] = None,
    verbose: bool = False,
) -> Tuple[
    List[AggregatedMetricsAtThreshold],
    Dict[str, List[AggregatedMetricsAtThreshold]],
]:
    """Aggregate multiple groups: mean±std + pooled counts per threshold."""
    if group_ns is None:
        n = getattr(getattr(config, "experiment", None), "n", None)
        if isinstance(n, int) and n >= 1:
            group_ns = list(range(1, n + 1))
        else:
            group_ns = [1]

    per_group_overall: List[List[MetricsAtThreshold]] = []
    per_group_by_diff: List[Dict[str, List[MetricsAtThreshold]]] = []
    valid_groups: List[int] = []

    for g in group_ns:
        analyzer = EM4CResultsAnalyzer(
            config, group_n=int(g), difficulty_map=difficulty_map, verbose=verbose
        )
        analyzer.run_analysis()
        if not analyzer.results:
            if verbose:
                print(f"  skip group_{g} (empty)")
            continue
        valid_groups.append(int(g))
        per_group_overall.append(analyzer.calculate_metrics_at_thresholds(thresholds))
        per_group_by_diff.append(analyzer.calculate_metrics_by_difficulty(thresholds))

    if not per_group_overall:
        return [], {}

    overall_agg = _aggregate_metrics(thresholds, per_group_overall)

    all_difficulties: set = set()
    for d in per_group_by_diff:
        all_difficulties.update(d.keys())

    by_difficulty_agg: Dict[str, List[AggregatedMetricsAtThreshold]] = {}
    for diff in sorted(all_difficulties):
        diff_group_metrics = [
            gd[diff] for gd in per_group_by_diff if diff in gd
        ]
        if diff_group_metrics:
            by_difficulty_agg[diff] = _aggregate_metrics(thresholds, diff_group_metrics)

    return overall_agg, by_difficulty_agg


def _aggregate_metrics(
    thresholds: List[float],
    per_group_metrics: List[List[MetricsAtThreshold]],
) -> List[AggregatedMetricsAtThreshold]:
    """Pool MetricsAtThreshold across groups; mean±std only where metric defined (TP+FP>0 etc.)."""
    n_groups = len(per_group_metrics)
    agg: List[AggregatedMetricsAtThreshold] = []

    for i, thr in enumerate(thresholds):
        precisions, recalls, f1s, accs = [], [], [], []
        pooled_tp = pooled_fp = pooled_tn = pooled_fn = 0

        for ms in per_group_metrics:
            if i >= len(ms):
                continue
            m = ms[i]
            has_pred_pos = (m.tp + m.fp) > 0
            has_actual_pos = (m.tp + m.fn) > 0
            total = m.tp + m.fp + m.tn + m.fn
            if has_pred_pos:
                precisions.append(float(m.precision))
            if has_actual_pos:
                recalls.append(float(m.recall))
            if has_pred_pos and has_actual_pos:
                f1s.append(float(m.f1_score))
            if total > 0:
                accs.append(float(m.accuracy))
            pooled_tp += m.tp
            pooled_fp += m.fp
            pooled_tn += m.tn
            pooled_fn += m.fn

        p_m, p_s = _mean_std(precisions)
        r_m, r_s = _mean_std(recalls)
        f_m, f_s = _mean_std(f1s)
        a_m, a_s = _mean_std(accs)

        total = pooled_tp + pooled_fp + pooled_tn + pooled_fn
        pp = pooled_tp / (pooled_tp + pooled_fp) if (pooled_tp + pooled_fp) > 0 else 0.0
        pr = pooled_tp / (pooled_tp + pooled_fn) if (pooled_tp + pooled_fn) > 0 else 0.0
        pf = 2 * pp * pr / (pp + pr) if (pp + pr) > 0 else 0.0
        pa = (pooled_tp + pooled_tn) / total if total > 0 else 0.0

        agg.append(AggregatedMetricsAtThreshold(
            threshold=float(thr), n_groups=n_groups,
            precision_mean=p_m, precision_std=p_s,
            recall_mean=r_m, recall_std=r_s,
            f1_mean=f_m, f1_std=f_s,
            accuracy_mean=a_m, accuracy_std=a_s,
            pooled_tp=pooled_tp, pooled_fp=pooled_fp,
            pooled_tn=pooled_tn, pooled_fn=pooled_fn,
            pooled_precision=pp, pooled_recall=pr,
            pooled_f1=pf, pooled_accuracy=pa,
        ))

    return agg


def print_em4c_aggregated_table(
    aggregated: Sequence[AggregatedMetricsAtThreshold],
    title: str = "overall",
) -> None:
    """Print pooled + mean±std table."""
    if not aggregated:
        print(f"[{title}] no aggregated data")
        return

    w_thr = 8
    w_g = 4
    w_ms = 17
    w_cnt = 7
    w_p = 10

    header = (
        f"{'thr':>{w_thr}}  {'g':>{w_g}}  "
        f"{'P(mean±std)':>{w_ms}}  {'R(mean±std)':>{w_ms}}  {'F1(mean±std)':>{w_ms}}  {'Acc(mean±std)':>{w_ms}}  "
        f"{'TP':>{w_cnt}}  {'FP':>{w_cnt}}  {'TN':>{w_cnt}}  {'FN':>{w_cnt}}  "
        f"{'P(pool)':>{w_p}}  {'R(pool)':>{w_p}}  {'F1(pool)':>{w_p}}  {'Acc(pool)':>{w_p}}"
    )
    line_len = len(header)

    print(f"\n{'=' * line_len}")
    print(f"ExeCRE multi-group aggregate — {title}")
    print(f"{'=' * line_len}")
    print(header)
    print("-" * line_len)

    for m in aggregated:
        p_ms = f"{m.precision_mean:.4f}±{m.precision_std:.4f}"
        r_ms = f"{m.recall_mean:.4f}±{m.recall_std:.4f}"
        f_ms = f"{m.f1_mean:.4f}±{m.f1_std:.4f}"
        a_ms = f"{m.accuracy_mean:.4f}±{m.accuracy_std:.4f}"
        print(
            f"{m.threshold:>{w_thr}.3f}  {m.n_groups:>{w_g}d}  "
            f"{p_ms:>{w_ms}}  {r_ms:>{w_ms}}  {f_ms:>{w_ms}}  {a_ms:>{w_ms}}  "
            f"{m.pooled_tp:>{w_cnt}d}  {m.pooled_fp:>{w_cnt}d}  {m.pooled_tn:>{w_cnt}d}  {m.pooled_fn:>{w_cnt}d}  "
            f"{m.pooled_precision:>{w_p}.4f}  {m.pooled_recall:>{w_p}.4f}  {m.pooled_f1:>{w_p}.4f}  {m.pooled_accuracy:>{w_p}.4f}"
        )

    print("=" * line_len)


def main():
    parser = argparse.ArgumentParser(description="EM4C results analyzer (read-only)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment JSON config"
    )
    parser.add_argument(
        "--group",
        type=int,
        default=1,
        help="Group index (default 1)"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return

    print(f"Config: {config_path}")
    config = load_config_from_file(config_path)

    print("Experiment:")
    print(f"  id: {config.experiment.experiment_id}")
    print(f"  benchmark: {config.experiment.benchmark}")
    print(f"  model: {config.model.model}")
    print(f"  baseline: {config.experiment.baseline}")
    print(f"  config_hash: {config.get_config_hash()}")
    print(f"  group: {args.group}")

    analyzer = EM4CResultsAnalyzer(config, group_n=args.group, verbose=True)

    analyzer.run_analysis()

    if not analyzer.results:
        print("No results, exiting")
        return

    analyzer.print_summary()

    base_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,\
         0.55, 0.6, 0.65, 0.7, 0.75, \
            0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    epsilon_values = [
        0.001, 0.005,
        0.0001, 0.0005,
        # 0.00001, 0.00005,
        # 0.000001, 0.000005,
        # 0.0000001, 0.0000005,
        # 0.00000001, 0.00000005,
        # 0.000000001, 0.000000005,
        # 0.0000000001, 0.0000000005,
        # 0.00000000001, 0.00000000005,
        # 0.000000000001, 0.000000000005,
        # 0.0000000000001, 0.0000000000005,
        # 0.00000000000001, 0.00000000000005,
        # 0.000000000000001, 0.000000000000005,
        # 0.0000000000000001, 0.0000000000000005,
        # 0.00000000000000001, 0.00000000000000005,
    ]
    epsilon_thresholds = [1 - e for e in epsilon_values]
    thresholds = sorted(set(base_thresholds + epsilon_thresholds))
    metrics_list = analyzer.calculate_metrics_at_thresholds(thresholds)

    analyzer.print_confusion_matrix_table(metrics_list)


if __name__ == "__main__":
    main()

