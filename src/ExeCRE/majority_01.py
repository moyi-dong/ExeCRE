"""Majority-vote binary label processor."""

from typing import List, Any, Hashable
from collections import Counter
from .base import LabelProcessor


def _make_hashable(value: Any) -> Hashable:
    """Convert nested values into hashable representations."""
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value
    elif isinstance(value, list):
        return tuple(_make_hashable(item) for item in value)
    elif isinstance(value, dict):
        return tuple(sorted((_make_hashable(k), _make_hashable(v)) for k, v in value.items()))
    else:
        try:
            return str(value)
        except:
            return repr(value)


class Majority01Processor(LabelProcessor):
    """Map each row to 0/1 labels by row majority."""
    
    def process(
        self,
        matrix: List[List[Any]]
    ) -> List[List[int]]:
        """Convert execution outputs to a row-wise majority 0/1 matrix."""
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

