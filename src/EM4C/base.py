"""
Abstract bases: ``LabelProcessor`` (raw matrix → int labels) and
``ConfidenceCalculator`` (int matrix + codes → alpha_c and index).
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict


class LabelProcessor(ABC):
    """Map execution outputs to an integer label matrix [test_case][code]."""

    @abstractmethod
    def process(self, matrix: List[List[Any]]) -> List[List[int]]:
        """
        Args:
            matrix: Raw outputs (e.g. list, str, or None on failure).

        Returns:
            Integer label matrix of the same shape.
        """
        pass


class ConfidenceCalculator(ABC):
    """Compute confidence and select a code index from an integer label matrix."""

    @abstractmethod
    def calculate(
        self,
        matrix: List[List[int]],
        codes: List[str],
        schema: Dict[str, Any],
    ) -> Tuple[float, int, Dict[str, Any]]:
        """
        Args:
            matrix: Integer labels [test_case][code].
            codes: Code strings aligned with columns.
            schema: Problem schema (may be unused by some calculators).

        Returns:
            ``(alpha_c, selected_code_index, metadata)``.
        """
        pass
