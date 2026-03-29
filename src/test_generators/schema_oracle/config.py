"""
Schema Oracle generator settings: candidate counts, validation, timeouts, thresholds,
and data-range modes (unlimited / rand / large — large skips oracle output).
"""

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.experiment_config import ExperimentConfig


class SchemaOracleConfig:
    """All knobs for SchemaOracleTestGenerator; use ``update()`` for overrides.

    ``data_range_mode``: ``unlimited`` | ``rand`` (default, ~small ranges) | ``large`` (big inputs, no output run).
    """

    # data_range_mode values
    MODE_UNLIMITED = "unlimited"
    MODE_RAND = "rand"
    MODE_LARGE = "large"
    
    def __init__(self, **kwargs):
        # --- schema LLM ---
        self.max_schema_candidates: int = kwargs.get('max_schema_candidates', 5)
        self.max_schema_retry: int = kwargs.get('max_schema_retry', 3)

        # --- solution LLM ---
        self.max_simulation_candidates: int = kwargs.get('max_simulation_candidates', 10)
        self.max_simulation_retry: int = kwargs.get('max_simulation_retry', 3)

        # --- phase-2 validation ---
        self.test_case_count: int = kwargs.get('test_case_count', 100000)
        self.single_test_timeout: float = kwargs.get('single_test_timeout', 0.8)
        self.total_generation_timeout: float = kwargs.get('total_generation_timeout', 180)
        self.allowed_error_ratio: float = kwargs.get('allowed_error_ratio', 0.5)
        self.legal_score_threshold: float = kwargs.get('legal_score_threshold', 1.0)
        self.save_voting_details: bool = kwargs.get('save_voting_details', True)

        # --- phase-3 gating ---
        self.error_score_threshold: float = kwargs.get('error_score_threshold', 0.1)
        self.eval_score_threshold: float = kwargs.get('eval_score_threshold', 0.8)

        # --- sampling ranges ---
        self.data_range_mode: str = kwargs.get('data_range_mode', self.MODE_RAND)
        self.boundary_bias: bool = kwargs.get('boundary_bias', False)
        # rand
        self.default_range_limits: dict = kwargs.get('default_range_limits', {
            'list': (1, 20),
            'matrix': (1, 20),
            'group': (1, 20),
        })
        
        self.variable_default_limits: dict = kwargs.get('variable_default_limits', {
            'int': (-20, 20),
            'float': (-20.0, 20.0),
            'char': None,
            'string': None,
        })
        
        # large
        self.large_range_limits: dict = kwargs.get('large_range_limits', {
            'list': (1, 10000),
            'matrix': (1, 1000),
            'group': (1, 10000),
        })
        
        self.large_variable_limits: dict = kwargs.get('large_variable_limits', {
            'int': (-10**9, 10**9),
            'float': (-10**9, 10**9),
            'char': None,
            'string': None,
        })
    
    def update(self, **kwargs) -> None:
        """Apply known attribute updates from ``kwargs``."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_experiment_config(cls, config: "ExperimentConfig", **kwargs) -> "SchemaOracleConfig":
        """Build from ``ExperimentConfig`` (evaluator + experiment), then apply ``kwargs``."""
        evaluator = config.evaluator
        experiment = config.experiment
        
        params = {
            'test_case_count': evaluator.test_case_count,
            'single_test_timeout': evaluator.single_test_timeout,
            'total_generation_timeout': evaluator.total_generation_timeout,
            'boundary_bias': evaluator.boundary_bias,
            'allowed_error_ratio': experiment.allowed_error_ratio,
        }

        params['default_range_limits'] = {
            'list': evaluator.list_range,
            'matrix': evaluator.matrix_range,
            'group': evaluator.group_range,
        }
        
        params['variable_default_limits'] = {
            'int': evaluator.int_range,
            'float': evaluator.float_range,
            'char': None,
            'string': None,
        }
        
        if experiment.error_score is not None:
            params['error_score_threshold'] = experiment.error_score
        if experiment.eval_score is not None:
            params['eval_score_threshold'] = experiment.eval_score
        
        params.update(kwargs)
        
        return cls(**params)
    
    def get_range_limit(self, data_type: str) -> tuple:
        """Range for composite type ``data_type`` under current ``data_range_mode``."""
        if self.data_range_mode == self.MODE_LARGE:
            return self.large_range_limits.get(data_type, (1, 10000))
        elif self.data_range_mode == self.MODE_RAND:
            return self.default_range_limits.get(data_type, (1, 20))
        else:  # unlimited
            return None

    def get_variable_limit(self, var_type: str) -> Optional[tuple]:
        """Scalar range for ``var_type`` under current mode."""
        if self.data_range_mode == self.MODE_LARGE:
            return self.large_variable_limits.get(var_type, None)
        elif self.data_range_mode == self.MODE_RAND:
            return self.variable_default_limits.get(var_type, None)
        else:  # unlimited
            return None

    def should_compute_output(self) -> bool:
        """False in ``large`` mode (inputs only)."""
        return self.data_range_mode != self.MODE_LARGE
    
    def is_valid_mode(self) -> bool:
        """Whether ``data_range_mode`` is a known constant."""
        return self.data_range_mode in (self.MODE_UNLIMITED, self.MODE_RAND, self.MODE_LARGE)


default_config = SchemaOracleConfig()

