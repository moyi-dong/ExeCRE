"""
EM4C: execution-matrix aggregation — map outputs to (selected code, confidence alpha_c).

Pipeline sketch: generate codes → schema → test inputs → run → matrix → label processor →
confidence calculator (e.g. Dawid–Skene). This module exposes step 6 given matrix + codes.
"""

from typing import List, Tuple, Any, Optional, Dict
from src.EM4C.label_processors.majority_01 import Majority01Processor
from src.EM4C.confidence_calculators.dawid_skene import DawidSkeneCalculator
from src.EM4C.base import LabelProcessor, ConfidenceCalculator


def step6_calculate_confidence(
    matrix: List[List[Any]],
    codes: List[str],
    label_processor: Optional[LabelProcessor] = None,
    confidence_calculator: Optional[ConfidenceCalculator] = None,
) -> Tuple[str, float, Optional[Dict[str, Any]]]:
    """
    Aggregate execution matrix into confidence and pick one code.

    Uses ``Majority01Processor`` and ``DawidSkeneCalculator`` by default.

    Args:
        matrix: Raw outputs, shape [test_case][code].
        codes: Code strings in column order.
        label_processor: Optional; defaults to Majority01Processor.
        confidence_calculator: Optional; defaults to DawidSkeneCalculator.

    Returns:
        ``(selected_code, alpha_c, metadata)`` with alpha_c in [0, 1].

    Raises:
        ValueError: Empty ``codes`` or row length mismatch vs ``codes``.
    """
    if not codes:
        raise ValueError("codes must be non-empty")

    if not matrix:
        return (codes[0], 0.0, None)

    num_test_cases = len(matrix)
    if num_test_cases == 0:
        return (codes[0], 0.0, None)

    num_codes = len(codes)
    for i, row in enumerate(matrix):
        if len(row) != num_codes:
            raise ValueError(
                f"matrix row {i} has length {len(row)}, expected {num_codes} (len(codes))"
            )

    if label_processor is None:
        label_processor = Majority01Processor()

    if confidence_calculator is None:
        confidence_calculator = DawidSkeneCalculator()

    label_matrix = label_processor.process(matrix)
    schema: Dict[str, Any] = {}
    alpha_c, selected_code_index, metadata = confidence_calculator.calculate(
        label_matrix, codes, schema
    )

    if selected_code_index < 0 or selected_code_index >= len(codes):
        selected_code_index = 0
        if alpha_c < 0 or alpha_c > 1:
            alpha_c = 0.0

    selected_code = codes[selected_code_index]
    return (selected_code, alpha_c, metadata)
