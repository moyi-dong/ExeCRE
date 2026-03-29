"""
Dawid–Skene (crowdkit): infer worker confusion, reliability
Rel(w) = sum_t pi_t * P(o=t|t), pick argmax_w Rel(w).
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
from crowdkit.aggregation import DawidSkene
from .base import ConfidenceCalculator


class DawidSkeneCalculator(ConfidenceCalculator):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def calculate(
        self,
        matrix: List[List[int]],
        codes: List[str],
        schema: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Args:
            matrix: Integer labels [test_case][code].
            codes: Code strings (columns = workers).
            schema: Unused; kept for API compatibility.

        Returns:
            ``(alpha_c, selected_code_index, metadata)`` with worker reliabilities and pi.
        """
        if not codes or not matrix:
            return (0.0, 0, {"worker_reliabilities": {}})

        n = len(matrix)
        m = len(codes)
        print(f"Dawid-Skene: matrix shape ({n} x {m})")

        data = []
        for test_case_idx, row in enumerate(matrix):
            for code_idx, label in enumerate(row):
                data.append(
                    {"task": test_case_idx, "worker": code_idx, "label": label}
                )
        df = pd.DataFrame(data)

        ds = DawidSkene(tol=1, n_iter=3)
        ds.fit(df)

        probas = ds.probas_
        pi_series = probas.mean(axis=0)
        pi = {str(k): float(v) for k, v in pi_series.items()}

        worker_reliabilities: Dict[int, float] = {}
        errors = ds.errors_.copy()

        for worker_id in range(len(codes)):
            if self.verbose:
                print("worker_id:", worker_id)
            worker_errors = errors.xs(worker_id, level="worker", drop_level=True)
            if self.verbose:
                print("worker_confusion_matrix:\n", worker_errors)
            worker_errors = worker_errors.reindex(
                index=[1, 0], columns=[0, 1], fill_value=0
            )

            for col in worker_errors.columns:
                col_sum = worker_errors[col].sum()
                if col_sum > 0:
                    worker_errors[col] = worker_errors[col] / col_sum
                else:
                    worker_errors[col] = 0.5

            if self.verbose:
                print("worker_confusion_matrix:\n", worker_errors)

            reliability = 0.0
            for true_label, pi_t in pi_series.items():
                p_ot_given_t = worker_errors.loc[true_label, true_label]
                if self.verbose:
                    print(
                        "reliability+=pi_t:",
                        pi_t,
                        "*p_ot_given_t:",
                        p_ot_given_t,
                        "=",
                        pi_t * p_ot_given_t,
                    )
                reliability += pi_t * p_ot_given_t
            if self.verbose:
                print("reliability:", reliability)
                print("-" * 50)
            worker_reliabilities[worker_id] = reliability

        best_worker = max(worker_reliabilities, key=worker_reliabilities.get)
        alpha_c = worker_reliabilities[best_worker]

        best_worker_errors = errors.xs(best_worker, level="worker", drop_level=True)
        best_worker_errors = best_worker_errors.reindex(
            index=[1, 0], columns=[0, 1], fill_value=0
        )

        for col in best_worker_errors.columns:
            col_sum = best_worker_errors[col].sum()
            if col_sum > 0:
                best_worker_errors[col] = best_worker_errors[col] / col_sum
            else:
                best_worker_errors[col] = 0.5

        best_worker_confusion_matrix = best_worker_errors.to_dict("index")

        metadata = {
            "worker_reliabilities": worker_reliabilities,
            "pi": pi,
            "best_worker_confusion_matrix": best_worker_confusion_matrix,
        }

        return (alpha_c, best_worker, metadata)
