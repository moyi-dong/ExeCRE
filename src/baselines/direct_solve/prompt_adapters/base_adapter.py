"""Abstract adapter: Problem -> chat messages."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from src.core.problem import Problem


class PromptAdapter(ABC):
    def __init__(self, benchmark: str):
        self.benchmark = benchmark

    @abstractmethod
    def format_prompt(self, problem: Problem) -> List[Dict[str, Any]]:
        raise NotImplementedError("subclasses must implement format_prompt")
