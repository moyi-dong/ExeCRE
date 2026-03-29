"""Confidence calculators: label matrix → alpha_c and selected code index."""

from .base import ConfidenceCalculator
from .dawid_skene import DawidSkeneCalculator

__all__ = [
    "ConfidenceCalculator",
    "DawidSkeneCalculator",
]
