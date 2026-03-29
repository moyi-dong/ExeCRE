"""Per-round accuracy from multi-round CSV logs (ExeCRE / Textgrad-style)."""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from src.config.experiment_config import ExperimentConfig
from src.utils.path_manager import get_group_dir


def parse_local_passed(local_passed_str: str) -> Dict[str, Optional[bool]]:
    """Parse Local_Passed as JSON or 'Key: True/False/None' text."""
    if not local_passed_str or not local_passed_str.strip():
        return {
            "Sample_passed": None,
            "Rand_passed": None,
            "TLE_passed": None
        }

    try:
        data = json.loads(local_passed_str)
        result = {
            "Sample_passed": data.get("Sample_passed"),
            "Rand_passed": data.get("Rand_passed") or data.get("Rand_Passed"),
            "TLE_passed": data.get("TLE_passed") or data.get("TLE_Passed") or data.get("Stress_passed") or data.get("Stress_Passed")
        }
        return result
    except (json.JSONDecodeError, TypeError):
        pass

    result = {
        "Sample_passed": None,
        "Rand_passed": None,
        "TLE_passed": None
    }

    pattern = r'(\w+_?[Pp]assed):\s*(True|False|None)'
    matches = re.findall(pattern, local_passed_str, re.IGNORECASE)

    for key, value in matches:
        key_lower = key.lower()
        bool_value = None
        if value.upper() == "TRUE":
            bool_value = True
        elif value.upper() == "FALSE":
            bool_value = False

        if "sample" in key_lower:
            result["Sample_passed"] = bool_value
        elif "rand" in key_lower:
            result["Rand_passed"] = bool_value
        elif "tle" in key_lower or "stress" in key_lower:
            result["TLE_passed"] = bool_value

    return result


def count_passed_stages(local_passed_dict: Dict[str, Optional[bool]]) -> int:
    """Count how many of Sample/Rand/TLE are explicitly True."""
    count = 0
    if local_passed_dict.get("Sample_passed") is True:
        count += 1
    if local_passed_dict.get("Rand_passed") is True:
        count += 1
    if local_passed_dict.get("TLE_passed") is True:
        count += 1
    return count


def select_best_round_for_trusttest(
    rows: List[Dict[str, str]],
    max_round: Optional[int] = None
) -> Tuple[Optional[int], Optional[bool]]:
    """ExeCRE / legacy TrustTest: prefer Is_Final (scan backward), else max Local_Passed stages."""
    if not rows:
        return None, None

    for row_idx, row in enumerate(reversed(rows)):
        round_index_str = row.get('Round_Index', '')
        try:
            round_index = int(round_index_str) if round_index_str else (len(rows) - 1 - row_idx)
        except (ValueError, TypeError):
            round_index = len(rows) - 1 - row_idx

        if max_round is not None and round_index > max_round:
            continue

        is_final = row.get('Is_Final', '').upper()
        if is_final == 'TRUE' or is_final == 'True':
            passed_str = row.get('Passed', '').upper()
            best_passed = (passed_str == 'TRUE')
            return round_index, best_passed

    best_round_idx = None
    max_pass_count = -1
    best_passed = None

    for row_idx, row in enumerate(rows):
        round_index_str = row.get('Round_Index', '')
        try:
            round_index = int(round_index_str) if round_index_str else row_idx
        except (ValueError, TypeError):
            round_index = row_idx

        if max_round is not None and round_index > max_round:
            continue

        local_passed_str = row.get('Local_Passed', '')
        local_passed_dict = parse_local_passed(local_passed_str)

        pass_count = count_passed_stages(local_passed_dict)

        if pass_count > max_pass_count or (pass_count == max_pass_count and
                                            (best_round_idx is None or round_index < best_round_idx)):
            max_pass_count = pass_count
            best_round_idx = round_index
            passed_str = row.get('Passed', '').upper()
            best_passed = (passed_str == 'TRUE')

    return best_round_idx, best_passed


def get_round_index(row: Dict[str, str], default_idx: int) -> int:
    """Read Round_Index or fall back to default_idx."""
    round_index_str = row.get('Round_Index', '')
    if round_index_str:
        try:
            return int(round_index_str)
        except (ValueError, TypeError):
            pass
    return default_idx


def load_results_by_round_from_csv(
    experiment_id: str,
    benchmark: str,
    model: str,
    baseline: str,
    config_hash: str,
    n: int,
    question_ids: Optional[Set[str]] = None
) -> Tuple[Dict[int, Dict[int, Dict[str, bool]]], Dict[Tuple[int, str], List[Dict[str, str]]]]:
    """Returns (results_by_round, trusttest_all_rows) for TrustTest full-row logic."""
    results_by_round = defaultdict(lambda: defaultdict(dict))
    trusttest_all_rows = {}
    is_trusttest = baseline.lower() in ["trusttest", "trusttestem", "execre"]

    for group_n in range(1, n + 1):
        group_dir = get_group_dir(
            experiment_id=experiment_id,
            benchmark=benchmark,
            model=model,
            baseline=baseline,
            config_hash=config_hash,
            group_n=group_n
        )

        if not group_dir.exists():
            continue

        csv_files = list(group_dir.glob("*.csv"))

        if not csv_files:
            continue

        for csv_file in csv_files:
            question_id = csv_file.stem

            if question_ids is not None and question_id not in question_ids:
                continue

            try:
                with open(csv_file, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)

                    if not rows:
                        continue

                    if is_trusttest:
                        trusttest_all_rows[(group_n, question_id)] = rows
                        for row_idx, row in enumerate(rows):
                            round_index = get_round_index(row, row_idx)
                            passed_str = row.get('Passed', '').upper()
                            passed = (passed_str == 'TRUE')
                            results_by_round[round_index][group_n][question_id] = passed
                    else:
                        for row_idx, row in enumerate(rows):
                            round_index = get_round_index(row, row_idx)
                            passed_str = row.get('Passed', '').upper()
                            passed = (passed_str == 'TRUE')
                            results_by_round[round_index][group_n][question_id] = passed

            except Exception as e:
                print(f"    Failed to read CSV {csv_file.name}: {e}")
                continue

    return dict(results_by_round), trusttest_all_rows


def get_round_result(
    results_by_round: Dict[int, Dict[int, Dict[str, bool]]],
    round_index: int,
    group_n: int,
    question_id: str,
    baseline: str,
    trusttest_all_rows: Optional[Dict[Tuple[int, str], List[Dict[str, str]]]] = None
) -> Optional[bool]:
    """TrustTest: best round in [0, round_index]; else use that round or last available."""
    is_trusttest = baseline.lower() in ["trusttest", "trusttestem", "execre"]

    if is_trusttest and trusttest_all_rows is not None:
        rows = trusttest_all_rows.get((group_n, question_id))
        if rows:
            _, best_passed = select_best_round_for_trusttest(rows, max_round=round_index)
            return best_passed
        return None
    else:
        if round_index in results_by_round:
            if group_n in results_by_round[round_index]:
                if question_id in results_by_round[round_index][group_n]:
                    return results_by_round[round_index][group_n][question_id]

        if results_by_round:
            max_round = max(results_by_round.keys())
            if max_round in results_by_round:
                if group_n in results_by_round[max_round]:
                    return results_by_round[max_round][group_n].get(question_id)

        return None


def analyze_round_accuracy(
    config: ExperimentConfig,
    max_round: Optional[int] = None
) -> Dict[int, Dict[str, Any]]:
    """Per-round {round: {accuracy, total, passed}}."""
    print("Loading results from all groups...")

    results_by_round, trusttest_all_rows = load_results_by_round_from_csv(
        experiment_id=config.experiment.experiment_id,
        benchmark=config.experiment.benchmark,
        model=config.model.model,
        baseline=config.experiment.baseline,
        config_hash=config.get_config_hash(),
        n=config.experiment.n
    )

    is_trusttest = config.experiment.baseline.lower() in [
        "trusttest",
        "trusttestem",
        "execre",
    ]

    if not results_by_round and not trusttest_all_rows:
        print("No result data found")
        return {}

    if is_trusttest and trusttest_all_rows:
        all_rounds = sorted(results_by_round.keys()) if results_by_round else [0]
        if not all_rounds:
            all_rounds = [0]
    else:
        all_rounds = sorted(results_by_round.keys())

    if max_round is not None:
        rounds_to_compute = [r for r in all_rounds if r <= max_round]
        if max_round not in rounds_to_compute and rounds_to_compute:
            rounds_to_compute.append(max(all_rounds))
        elif not rounds_to_compute:
            rounds_to_compute = [max(all_rounds)] if all_rounds else [0]
    else:
        rounds_to_compute = all_rounds

    print(f"Loaded {len(results_by_round)} round keys in index")
    if is_trusttest:
        print(f"TrustTest: {len(trusttest_all_rows)} question raw row sets")
    print(f"Computing accuracy for {len(rounds_to_compute)} round cutoffs...")

    round_metrics = {}

    for round_index in rounds_to_compute:
        if is_trusttest and trusttest_all_rows:
            passed_count = 0
            total_count = len(trusttest_all_rows)

            for (group_n, question_id), rows in trusttest_all_rows.items():
                _, best_passed = select_best_round_for_trusttest(rows, max_round=round_index)
                if best_passed is True:
                    passed_count += 1
        else:
            round_data = results_by_round.get(round_index, {})

            if not round_data and results_by_round:
                max_round_available = max(results_by_round.keys())
                round_data = results_by_round[max_round_available]

            passed_count = 0
            total_count = 0

            for group_n, group_results in round_data.items():
                for question_id, passed in group_results.items():
                    total_count += 1
                    if passed:
                        passed_count += 1

        accuracy = passed_count / total_count if total_count > 0 else 0.0

        round_metrics[round_index] = {
            "accuracy": accuracy,
            "total": total_count,
            "passed": passed_count
        }

    return round_metrics


def print_round_accuracy_results(
    round_metrics: Dict[int, Dict[str, Any]]
) -> None:
    """Print accuracy table."""
    print("\n" + "=" * 50)
    print("Accuracy by round:")
    print("-" * 50)

    for round_index in sorted(round_metrics.keys()):
        metrics = round_metrics[round_index]
        accuracy = metrics["accuracy"]
        total = metrics["total"]
        passed = metrics["passed"]
        print(f"Round {round_index}: {accuracy:.4f} ({passed}/{total})")

    print("=" * 50)


def analyze_round_accuracy_from_config(
    config_path: Path,
    max_round: Optional[int] = None
) -> Dict[int, Dict[str, Any]]:
    """Load config from path and run analyze_round_accuracy."""
    config = ExperimentConfig.from_file(config_path)
    return analyze_round_accuracy(config, max_round=max_round)
