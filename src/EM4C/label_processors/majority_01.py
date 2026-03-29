"""Per-row majority vote: 1 where cell equals row mode, else 0."""

from typing import List, Any, Hashable
from collections import Counter
from .base import LabelProcessor


def _make_hashable(value: Any) -> Hashable:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return tuple(_make_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            sorted((_make_hashable(k), _make_hashable(v)) for k, v in value.items())
        )
    try:
        return str(value)
    except Exception:
        return repr(value)


class Majority01Processor(LabelProcessor):
    def process(self, matrix: List[List[Any]]) -> List[List[int]]:
        """
        Args:
            matrix: Raw outputs [test_case][code].

        Returns:
            Binary matrix: 1 if value matches row mode, else 0.
        """
        result = []
        for row in matrix:
            hashable_row = [_make_hashable(item) for item in row]
            counter = Counter(hashable_row)
            if len(counter) == 0:
                new_row = [0] * len(row)
            else:
                most_common = counter.most_common(1)[0][0]
                new_row = [1 if _make_hashable(lab) == most_common else 0 for lab in row]
            result.append(new_row)
        return result
