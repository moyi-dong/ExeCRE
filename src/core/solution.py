"""Unified solution record: one submission per problem; result_type uses 'Accepted' for pass."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union


@dataclass
class Solution:
    """One solution for a problem; use separate instances per self-correction round (see round_index)."""
    # Basic
    code: str  # Solution_Code
    problem_id: str  # Question_Id

    # Execution outcome
    is_normal_end: bool = True  # Is_Normal_End
    passed: bool = False  # Passed
    result_type: Optional[str] = None  # Result_Type: "Accepted" = pass; else e.g. WA/TLE/RE

    # Failing tests
    error_case_indice: Optional[List[int]] = None  # Error_Case_Indice
    error_case_contents: Optional[List[Dict[str, Any]]] = None  # Error_Case_Contents

    # Round metadata (e.g. round-pass@k)
    round_index: Optional[int] = None  # Round_Index
    is_final: bool = True

    # Local run (bool or JSON string from solver)
    local_passed: Optional[Union[bool, str]] = None  # Local_Passed
    local_result_type: Optional[str] = None  # Local_Result_Type

    schema: Optional[str] = None  # Schema
    simulation_code: Optional[str] = None  # Simulation_Code

    execution_results: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.execution_results is None:
            self.execution_results = []
        if self.metadata is None:
            self.metadata = {}
        if self.error_case_indice is None:
            self.error_case_indice = []
        if self.error_case_contents is None:
            self.error_case_contents = []

    def add_execution_result(self, input_data: Any, output: Any,
                            passed: Optional[bool] = None,
                            error_type: Optional[str] = None,
                            error_message: Optional[str] = None,
                            **kwargs) -> None:
        """Append one per-test execution row to execution_results."""
        result = {
            'input': input_data,
            'output': output,
            'passed': passed,
            'error_type': error_type,
            'error_message': error_message,
            **kwargs
        }
        self.execution_results.append(result)

    def has_error(self) -> bool:
        if not self.is_normal_end:
            return True
        if self.result_type and self.result_type != "Accepted":
            return True
        if self.error_case_indice:
            return True
        if self.execution_results:
            return any(r.get('error_type') is not None or not r.get('passed', True) for r in self.execution_results)
        return False

    def is_passed(self) -> bool:
        return self.passed and self.is_normal_end
