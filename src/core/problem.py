"""Unified problem record for benchmark items."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class Problem:
    """Coding problem: statement, tests, and optional contest metadata."""
    question_id: str
    question_title: str
    question_content: str
    difficulty: str

    benchmark: str = ""  # e.g. LiveCodeBench, HumanEval, MBPP

    starter_code: str = ""
    public_test_cases: List[Dict[str, Any]] = field(default_factory=list)
    private_test_cases: List[Dict[str, Any]] = field(default_factory=list)

    platform: str = ""
    contest_id: str = ""
    contest_date: Optional[datetime] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def test_cases(self) -> List[Dict[str, Any]]:
        return self.public_test_cases + self.private_test_cases

    @property
    def total_test_cases(self) -> int:
        return len(self.public_test_cases) + len(self.private_test_cases)

    def __post_init__(self):
        if self.public_test_cases is None:
            self.public_test_cases = []
        if self.private_test_cases is None:
            self.private_test_cases = []
        if self.metadata is None:
            self.metadata = {}
