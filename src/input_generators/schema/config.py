"""Defaults for schema parsing (ranges, separators) and optional multi-candidate generation."""

class SchemaConfig:
    """Mutable bag of schema-related tuning knobs."""

    def __init__(self, **kwargs):
        self.default_range_limits = kwargs.get(
            "default_range_limits",
            {
                "list": (1, 20),
                "matrix": (1, 20),
                "group": (1, 20),
            },
        )

        self.variable_default_limits = kwargs.get(
            "variable_default_limits",
            {
                "int": (-20, 20),
                "float": (-20.0, 20.0),
                "char": None,
                "string": None,
            },
        )

        self.default_separators = kwargs.get(
            "default_separators",
            {
                "list": " ",
                "group": "\\n",
                "row_split": "\\n",
                "column_split": " ",
            },
        )

        # When True, bias sampling toward the outer ~10% of numeric ranges
        self.boundary_bias = kwargs.get("boundary_bias", False)

        self.max_schema_retry = kwargs.get("max_schema_retry", 3)
        self.max_simulation_retry = kwargs.get("max_simulation_retry", 3)
        self.max_simulation_candidates = kwargs.get("max_simulation_candidates", 10)
        self.max_schema_candidates = kwargs.get("max_schema_candidates", 5)
        self.test_case_count = kwargs.get("test_case_count", 100000)
        self.single_test_timeout = kwargs.get("single_test_timeout", 0.8)
        self.total_generation_timeout = kwargs.get("total_generation_timeout", 60)
        self.allowed_error_ratio = kwargs.get("allowed_error_ratio", 0.2)
        self.legal_score_threshold = kwargs.get("legal_score_threshold", 0.8)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def get_range_limit(self, data_type: str) -> tuple:
        return self.default_range_limits.get(data_type, (-20, 20))

    def get_variable_limit(self, var_type: str) -> tuple:
        return self.variable_default_limits.get(var_type, None)

    def get_default_separator(self, data_type: str) -> str:
        return self.default_separators.get(data_type, " ")


default_config = SchemaConfig()
