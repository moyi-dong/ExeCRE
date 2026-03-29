"""Abstract base for test generators: initialize(problem) once, then generate() → TestCase."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.core.problem import Problem

if TYPE_CHECKING:
    from src.config.experiment_config import ExperimentConfig


@dataclass
class TestCase:
    """One synthetic test: stdin/call input string and optional oracle output."""

    input: str
    expected_output: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseTestGenerator(ABC):
    """Lifecycle: __init__ → initialize(problem) → generate() (may be called repeatedly)."""

    def __init__(
        self,
        name: str,
        engine: Any = None,
        config: Optional["ExperimentConfig"] = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.engine = engine
        self.config = config
        self._extra_config = kwargs
        self._initialized = False
        self._problem: Optional[Problem] = None
        self._artifact: Optional[str] = None

    @abstractmethod
    def initialize(self, problem: Problem, **kwargs: Any) -> bool:
        """One-time setup for this problem; set _problem, _artifact, _initialized on success."""
        pass

    @abstractmethod
    def generate(self) -> TestCase:
        """Return one test case; requires successful initialize()."""
        pass

    def generate_batch(self, n: int) -> List[TestCase]:
        """Default: n calls to generate(); override for batching."""
        self._check_initialized()
        return [self.generate() for _ in range(n)]

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def initialization_artifact(self) -> Optional[str]:
        return self._artifact

    @property
    def problem(self) -> Optional[Problem]:
        return self._problem

    @property
    def sampling_temperature(self) -> float:
        if self.config is not None:
            return self.config.model.sampling_temperature
        return 1.0

    def reset(self) -> None:
        self._initialized = False
        self._problem = None
        self._artifact = None

    def _check_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(f"{self.name}: call initialize() first")

    def __repr__(self) -> str:
        st = "ready" if self._initialized else "not initialized"
        return f"{self.__class__.__name__}(name={self.name!r}, {st})"
