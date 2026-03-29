"""Interactive CLI for post-experiment analyses (pass@k, mislead, EM4C, GSM8K).

Usage:
    python -m src.experiments.analyze_results --config_file configs/e1_base_gpt-4o.json
    python -m src.experiments.analyze_results --experiment_id e1 --benchmark LiveCodeBench --model gpt-4o
"""

from typing import Optional

from src.config import ExperimentConfig
from src.utils.parser import get_config_from_args
from src.utils.config_printer import print_config_summary
from src.utils.path_manager import get_results_dir
from src.analyzers.common.pass_at_k_analyzer import (
    analyze_pass_at_k_by_difficulty,
    print_pass_at_k_by_difficulty_results,
)
from src.analyzers.GSM8K.Classification import (
    analyze_gsm8k_execre_confidence,
    analyze_gsm8k_execre_confidence_multi_groups,
    print_gsm8k_confidence_results,
    print_gsm8k_confidence_results_multi_groups,
)
from src.analyzers.EM4C_analyzer.analyze_em4c_results import (
    EM4CResultsAnalyzer,
    analyze_em4c_multi_groups,
    print_em4c_aggregated_table,
    print_em4c_difficulty_results,
)
from src.analyzers.e4_verify_classification.multi_round_mislead import (
    analyze_groups_and_aggregate,
    analyze_multi_round_mislead,
    print_mislead_results,
    _discover_group_dirs,
)

# LiveCodeBench EM4C-style runs (new name ExeCRE; TrustTestEM kept for legacy configs / paths).
_LCB_EM4C_BASELINES = frozenset({"ExeCRE", "TrustTestEM"})


def show_analysis_menu() -> Optional[str]:
    """Show the analysis menu; return option id or None if user quits."""
    while True:
        print("\n" + "=" * 50)
        print("Choose an analysis:")
        print("=" * 50)
        print("1) Overall pass@k (with difficulty breakdown)")
        print("2) Multi-round mislead (current config results dir)")
        print("3) GSM8K ExeCRE confidence (alpha vs code correctness)")
        print("4) GSM8K ExeCRE confidence (multi-group: mean±std + pooled)")
        print("5) LiveCodeBench ExeCRE binary metrics (alpha vs TLEfree)")
        print("6) LiveCodeBench ExeCRE (multi-group + by difficulty)")
        print("0) Exit")
        print("=" * 50)

        choice = input("Enter option number: ").strip()

        if choice == "0":
            return None
        if choice == "1":
            return "overall_pass_at_k"
        if choice == "2":
            return "multi_round_mislead"
        if choice == "3":
            return "gsm8k_execre_confidence"
        if choice == "4":
            return "gsm8k_execre_confidence_multi_groups"
        if choice == "5":
            return "lcb_em4c_confidence"
        if choice == "6":
            return "lcb_em4c_multi_groups"
        print("Invalid option, try again.")


def run_overall_pass_at_k_analysis(config: ExperimentConfig):
    """Overall pass@k with difficulty breakdown."""
    print("\n" + "=" * 50)
    print("Overall pass@k")
    print("=" * 50)

    try:
        results = analyze_pass_at_k_by_difficulty(
            config=config,
            k_list=[1, 5, 10],
        )

        if results and (results.get("overall") or results.get("by_difficulty")):
            print_pass_at_k_by_difficulty_results(results)
            print("\nOverall pass@k analysis done.")
        else:
            print("\nOverall pass@k finished; no valid data found.")

    except Exception as e:
        print(f"\nOverall pass@k failed: {e}")
        raise


def run_multi_round_mislead_analysis(config: ExperimentConfig):
    """Count mislead events under results/.../<config_hash>/ (group_1..)."""
    print("\n" + "=" * 50)
    print("Multi-round mislead")
    print("=" * 50)

    run_dir = get_results_dir(
        experiment_id=config.experiment.experiment_id,
        benchmark=config.experiment.benchmark,
        model=config.model.model,
        baseline=config.experiment.baseline,
        config_hash=config.get_config_hash(),
    )

    try:
        if not run_dir.exists():
            print(f"No results directory: {run_dir}")
            return

        group_dirs = _discover_group_dirs(run_dir, max_groups=5)
        if not group_dirs:
            print(f"No group_1..group_5 under: {run_dir}")
            return

        print(f"Path: {run_dir}")
        print(f"Groups: {', '.join(g.name for g in group_dirs)}")

        if len(group_dirs) == 1:
            g = group_dirs[0]
            results = analyze_multi_round_mislead(g)
            print_mislead_results(results, title=f"Multi-round mislead stats ({g.name})")
            return

        bundle = analyze_groups_and_aggregate(group_dirs)
        agg = bundle.get("aggregate")
        if not agg:
            print("No aggregate data.")
            return

        print_mislead_results(
            {"condition1_only": {}, "condition1_and_2": {}},
            title="Multi-round mislead stats (aggregated by group)",
        )
        print("\n[Per-group summary (mean ± std)]")
        print("-" * 60)
        for label, key_prefix in [
            ("Condition 1 only", "condition1_only"),
            ("Conditions 1 and 2", "condition1_and_2"),
        ]:
            q = agg[key_prefix]["questions_with_mislead"]
            t = agg[key_prefix]["total_mislead_count"]
            print(f"{label} (n={int(q['n'])} groups)")
            print(f"  Questions with mislead: {q['mean']:.2f} ± {q['std']:.2f}")
            print(f"  Total mislead count: {t['mean']:.2f} ± {t['std']:.2f}")

    except Exception as e:
        print(f"\nMulti-round mislead failed: {e}")
        raise


def run_gsm8k_execre_confidence_analysis(config: ExperimentConfig):
    """GSM8K: EM4C alpha vs code correctness (threshold sweep)."""
    print("\n" + "=" * 50)
    print("GSM8K ExeCRE confidence")
    print("=" * 50)

    try:
        metrics = analyze_gsm8k_execre_confidence(
            config=config,
            group_n=1,
            abs_tol=1e-4,
            rel_tol=1e-3,
        )

        if metrics:
            print_gsm8k_confidence_results(metrics)
            print("\nGSM8K confidence analysis done.")
        else:
            print("\nGSM8K confidence finished; no valid data found.")

    except Exception as e:
        print(f"\nGSM8K confidence analysis failed: {e}")
        raise


def run_gsm8k_execre_confidence_multi_groups_analysis(config: ExperimentConfig):
    """GSM8K: multi-group mean±std and pooled metrics."""
    print("\n" + "=" * 50)
    print("GSM8K ExeCRE confidence (multi-group)")
    print("=" * 50)

    try:
        aggregated = analyze_gsm8k_execre_confidence_multi_groups(
            config=config,
            group_ns=None,
            abs_tol=1e-4,
            rel_tol=1e-3,
        )

        if aggregated:
            print_gsm8k_confidence_results_multi_groups(aggregated)
            print("\nGSM8K multi-group confidence done.")
        else:
            print(
                "\nGSM8K multi-group finished; no data "
                "(maybe only group_1 or missing result dirs)."
            )

    except Exception as e:
        print(f"\nGSM8K multi-group confidence failed: {e}")
        raise


def run_lcb_em4c_confidence_analysis(
    config: ExperimentConfig,
    group_n: int = 1,
):
    """LiveCodeBench ExeCRE: binary metrics (alpha threshold vs TLEfree labels)."""
    if config.experiment.benchmark != "LiveCodeBench":
        print(
            f"Binary analysis supports LiveCodeBench only; "
            f"benchmark={config.experiment.benchmark}"
        )
        return
    if config.experiment.baseline not in _LCB_EM4C_BASELINES:
        print(
            f"Binary analysis supports ExeCRE (or legacy TrustTestEM) only; "
            f"baseline={config.experiment.baseline}"
        )
        return

    print("\n" + "=" * 50)
    print("LiveCodeBench ExeCRE binary metrics")
    print(f"group_{group_n}")
    print("=" * 50)

    try:
        difficulty_map = _build_difficulty_map(config)
        analyzer = EM4CResultsAnalyzer(config, group_n=group_n, difficulty_map=difficulty_map)
        analyzer.run_analysis()

        if not analyzer.results:
            print("No results (check metadata.json / tlefree_evaluation.json).")
            return

        analyzer.print_summary()

        base_thresholds = [
            0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
            0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.81, 0.82, 0.83, 0.84,
            0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94,
            0.95, 0.96, 0.97, 0.98, 0.99,
        ]
        metrics_list = analyzer.calculate_metrics_at_thresholds(base_thresholds)
        analyzer.print_confusion_matrix_table(metrics_list)

        by_diff = analyzer.calculate_metrics_by_difficulty(base_thresholds)
        if by_diff:
            print_em4c_difficulty_results(by_diff)

        print("\nLiveCodeBench ExeCRE binary analysis done.")
    except Exception as e:
        print(f"\nLiveCodeBench ExeCRE binary analysis failed: {e}")
        raise


def _build_difficulty_map(config: ExperimentConfig) -> dict:
    """Build {question_id: difficulty} from the configured test dataset."""
    from src.experiments import initialize_and_load_dataset

    test_dataset = initialize_and_load_dataset(config)
    if not test_dataset:
        return {}
    return {p.question_id: p.difficulty for p in test_dataset}


def run_lcb_em4c_multi_groups_analysis(config: ExperimentConfig):
    """LiveCodeBench ExeCRE: multi-group + per-difficulty aggregates."""
    if config.experiment.benchmark != "LiveCodeBench":
        print(
            f"Binary analysis supports LiveCodeBench only; "
            f"benchmark={config.experiment.benchmark}"
        )
        return
    if config.experiment.baseline not in _LCB_EM4C_BASELINES:
        print(
            f"Binary analysis supports ExeCRE (or legacy TrustTestEM) only; "
            f"baseline={config.experiment.baseline}"
        )
        return

    print("\n" + "=" * 50)
    print("LiveCodeBench ExeCRE multi-group + by difficulty")
    print("=" * 50)

    try:
        print("Loading dataset for difficulty labels...")
        difficulty_map = _build_difficulty_map(config)
        if difficulty_map:
            print(f"   Loaded difficulty for {len(difficulty_map)} problems.")
        else:
            print("   No difficulty map; only overall metrics will be printed.")

        thresholds = [
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
            0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0,
        ]

        overall_agg, by_diff_agg = analyze_em4c_multi_groups(
            config=config,
            thresholds=thresholds,
            group_ns=None,
            difficulty_map=difficulty_map,
        )

        if overall_agg:
            print_em4c_aggregated_table(overall_agg, title="Overall")
        else:
            print("\nNo overall aggregate data.")

        for diff in sorted(by_diff_agg.keys()):
            print_em4c_aggregated_table(by_diff_agg[diff], title=f"Difficulty: {diff}")

        if overall_agg or by_diff_agg:
            print("\nLiveCodeBench ExeCRE multi-group + difficulty done.")
        else:
            print("\nAnalysis finished; no valid data found.")

    except Exception as e:
        print(f"\nLiveCodeBench ExeCRE multi-group analysis failed: {e}")
        raise


def main():
    """Load config, menu loop, dispatch analyses."""
    print("=" * 50)
    print("Result analysis")
    print("=" * 50)

    try:
        config: ExperimentConfig = get_config_from_args()
        print_config_summary(config)
    except Exception as e:
        print(f"Config load failed: {e}")
        return

    while True:
        choice = show_analysis_menu()

        if choice is None:
            print("\nGoodbye.")
            break

        if choice == "overall_pass_at_k":
            run_overall_pass_at_k_analysis(config)
        elif choice == "multi_round_mislead":
            run_multi_round_mislead_analysis(config)
        elif choice == "gsm8k_execre_confidence":
            run_gsm8k_execre_confidence_analysis(config)
        elif choice == "gsm8k_execre_confidence_multi_groups":
            run_gsm8k_execre_confidence_multi_groups_analysis(config)
        elif choice == "lcb_em4c_confidence":
            run_lcb_em4c_confidence_analysis(config, group_n=1)
        elif choice == "lcb_em4c_multi_groups":
            run_lcb_em4c_multi_groups_analysis(config)
        else:
            print("Unknown analysis option.")
            continue

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
