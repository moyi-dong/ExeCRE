"""Label processors: raw execution matrix → integer labels."""

from .base import LabelProcessor
from .majority_01 import Majority01Processor

__all__ = [
    "LabelProcessor",
    "Majority01Processor",
]
