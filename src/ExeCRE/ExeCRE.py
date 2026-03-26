from typing import List, Tuple, Any, Optional, Dict
from src.EM4C.label_processors.majority_01 import Majority01Processor
from src.EM4C.confidence_calculators.dawid_skene import DawidSkeneCalculator
from src.EM4C.base import LabelProcessor, ConfidenceCalculator


def step6_calculate_confidence(
    matrix: List[List[Any]],
    codes: List[str],
    label_processor: Optional[LabelProcessor] = None,
    confidence_calculator: Optional[ConfidenceCalculator] = None
) -> Tuple[str, float, Optional[Dict[str, Any]]]:
    """
    Calculate confidence and select the best code.
    """
    if not codes:
        raise ValueError("codes must not be empty")
    
    if not matrix:
        return (codes[0], 0.0, None)
    
    num_test_cases = len(matrix)
    if num_test_cases == 0:
        return (codes[0], 0.0, None)
    
    num_codes = len(codes)
    for i, row in enumerate(matrix):
        if len(row) != num_codes:
            raise ValueError(
                f"row {i} has {len(row)} columns, expected {num_codes}"
            )
    
    if label_processor is None:
        label_processor = Majority01Processor()
    
    if confidence_calculator is None:
        confidence_calculator = DawidSkeneCalculator()
    
    label_matrix = label_processor.process(matrix)
    
    schema = {}
    alpha_c, selected_code_index, metadata = confidence_calculator.calculate(
        label_matrix, codes, schema
    )
    
    if selected_code_index < 0 or selected_code_index >= len(codes):
        selected_code_index = 0
        if alpha_c < 0 or alpha_c > 1:
            alpha_c = 0.0
    
    selected_code = codes[selected_code_index]
    return (selected_code, alpha_c, metadata)