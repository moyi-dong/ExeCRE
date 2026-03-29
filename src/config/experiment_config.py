"""Experiment configuration: dataclasses, load/save, and deterministic config hash."""

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse


@dataclass
class ModelConfig:
    """Model / LLM API settings."""
    model: str = "dmx-deepseek-v3-241226"
    temperature: float = 0.0  # default call temperature for stable outputs
    top_p: float = 0.99
    sampling_temperature: float = 1.0  # temperature when drawing multiple samples
    max_tokens: int = 2000
    stop: List[str] = field(default_factory=lambda: ["###"])
    local_model_path: Optional[str] = None
    trust_remote_code: bool = False
    tensor_parallel_size: int = -1
    enable_prefix_caching: bool = False
    dtype: str = "bfloat16"
    openai_timeout: int = 90


@dataclass
class EvaluatorConfig:
    """Evaluator timeouts, parallelism, and random-test ranges."""
    timeout: int = 6
    debug: bool = False
    num_process_evaluate: int = 12
    evaluation_mode: str = "multiprocess"  # multiprocess | single_process | single_process_safe

    test_case_count: int = 100000
    single_test_timeout: float = 0.8
    total_generation_timeout: int = 60
    boundary_bias: bool = False

    list_range: tuple = (1, 20)
    matrix_range: tuple = (1, 20)
    group_range: tuple = (1, 20)
    int_range: tuple = (-20, 20)
    float_range: tuple = (-20.0, 20.0)


@dataclass
class ExperimentParams:
    """Experiment identity, dataset filters, and baseline-related options."""
    experiment_id: str = "e1"  # e1–e7
    benchmark: str = "LiveCodeBench"  # LiveCodeBench, HumanEval, MBPP, …
    baseline: str = "ExeCRE"

    input_generator: str = "schema"  # E7: schema, llm_direct, …

    verifier: str = "TrustTest"  # E4

    eval_score: Optional[float] = None
    error_score: Optional[float] = None
    allowed_error_ratio: float = 0.1

    n: int = 1  # number of experiment groups

    difficulty: str = "all"  # easy | medium | hard | all
    start_date: Optional[str] = "2025-01-01"
    end_date: Optional[str] = "2025-05-01"
    test_count: Optional[int] = None
    specific_question_id: Optional[str] = None
    specific_question_ids: Optional[List[str]] = None

    release_version: str = "release_latest"
    version: str = "trusttest_v001"  # affects artifact paths / hash
    not_fast: bool = False  # use full test set when True
    cot_code_execution: bool = False


@dataclass
class ExperimentConfig:
    """Top-level config: model, evaluator, experiment params, and run controls."""
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    experiment: ExperimentParams = field(default_factory=ExperimentParams)

    multiprocess: int = 10
    max_rounds: int = 10  # max optimization rounds for round-based baselines (e.g. Textgrad)
    continue_existing: bool = False
    continue_existing_with_eval: bool = False
    use_cache: bool = False
    cache_batch_size: int = 100
    force_regenerate: bool = False
    force_reevaluate: bool = False
    skip_generate: bool = False
    skip_evaluate: bool = False
    start_from: int = 1  # 1-based group index to start from
    debug: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict (no runtime timestamp)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save_to_file(self, filepath: Path) -> None:
        """Write JSON; adds `_metadata` with timestamps (not used for hashing)."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        config_dict = self.to_dict()
        payload = {
            **config_dict,
            "_metadata": {
                "run_timestamp": datetime.now().isoformat(),
                "saved_at": datetime.now().isoformat(),
            },
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Rebuild from dict; drops `_metadata`; unknown top-level keys become attributes."""
        data = {k: v for k, v in data.items() if k != '_metadata'}

        if 'model' in data and isinstance(data['model'], dict):
            data['model'] = ModelConfig(**data['model'])
        if 'evaluator' in data and isinstance(data['evaluator'], dict):
            data['evaluator'] = EvaluatorConfig(**data['evaluator'])
        if 'experiment' in data and isinstance(data['experiment'], dict):
            data['experiment'] = ExperimentParams(**data['experiment'])

        known = {
            'model', 'evaluator', 'experiment', 'multiprocess', 'max_rounds',
            'continue_existing', 'continue_existing_with_eval', 'use_cache',
            'cache_batch_size', 'force_regenerate', 'force_reevaluate',
            'skip_generate', 'skip_evaluate', 'start_from', 'debug',
        }
        extra_fields = {k: v for k, v in data.items() if k not in known}

        config = cls(**{k: v for k, v in data.items() if k not in extra_fields})
        for key, value in extra_fields.items():
            setattr(config, key, value)
        return config

    @classmethod
    def from_json(cls, json_str: str) -> 'ExperimentConfig':
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_file(cls, filepath: Path) -> 'ExperimentConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

    def get_config_hash(self) -> str:
        """8-char SHA256 prefix for artifact naming; see `get_config_hash_for_baseline`."""
        return self.get_config_hash_for_baseline(self.experiment.baseline)

    def get_config_hash_for_baseline(self, baseline: str) -> str:
        """Hash of model, method, hyperparams, and key evaluator fields for `baseline`.

        Omits run scope (dates, test_count, difficulty, parallelism, max_rounds, etc.).
        """
        hash_dict = {
            'model': self.model.model,
            'temperature': self.model.temperature,
            'top_p': self.model.top_p,
            'sampling_temperature': self.model.sampling_temperature,
            'max_tokens': self.model.max_tokens,
            'stop': sorted(self.model.stop),
            'benchmark': self.experiment.benchmark,
            'baseline': baseline,
            'input_generator': self.experiment.input_generator,
            'verifier': self.experiment.verifier,
            'version': self.experiment.version,
            'eval_score': self.experiment.eval_score,
            'error_score': self.experiment.error_score,
            'allowed_error_ratio': self.experiment.allowed_error_ratio,
            'timeout': self.evaluator.timeout,
            'test_case_count': self.evaluator.test_case_count,
            'single_test_timeout': self.evaluator.single_test_timeout,
            'total_generation_timeout': self.evaluator.total_generation_timeout,
        }
        json_str = json.dumps(hash_dict, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()[:8]

    def validate(self) -> None:
        valid_experiment_ids = [f"e{i}" for i in range(1, 8)]
        if self.experiment.experiment_id not in valid_experiment_ids:
            raise ValueError(
                f"experiment_id must be one of {valid_experiment_ids}; "
                f"got {self.experiment.experiment_id!r}"
            )

        valid_benchmarks = ["LiveCodeBench", "HumanEval", "MBPP", "GSM8K"]
        if self.experiment.benchmark not in valid_benchmarks:
            raise ValueError(
                f"benchmark must be one of {valid_benchmarks}; "
                f"got {self.experiment.benchmark!r}"
            )

        valid_difficulties = ["easy", "medium", "hard", "all"]
        if self.experiment.difficulty not in valid_difficulties:
            raise ValueError(
                f"difficulty must be one of {valid_difficulties}; "
                f"got {self.experiment.difficulty!r}"
            )

        valid_modes = ["multiprocess", "single_process", "single_process_safe"]
        if self.evaluator.evaluation_mode not in valid_modes:
            raise ValueError(
                f"evaluation_mode must be one of {valid_modes}; "
                f"got {self.evaluator.evaluation_mode!r}"
            )

        if not 0 <= self.model.temperature <= 2:
            raise ValueError(
                f"temperature must be in [0, 2]; got {self.model.temperature}"
            )
        if not 0 < self.model.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1]; got {self.model.top_p}")


def load_config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Build `ExperimentConfig` from an argparse namespace."""
    stop_tokens = args.stop if isinstance(args.stop, list) else args.stop.split(",")

    model_config = ModelConfig(
        model=getattr(args, 'model', 'dmx-deepseek-v3-241226'),
        temperature=getattr(args, 'temperature', 0.0),
        top_p=getattr(args, 'top_p', 0.99),
        sampling_temperature=getattr(args, 'sampling_temperature', 1.0),
        max_tokens=getattr(args, 'max_tokens', 2000),
        stop=stop_tokens,
        local_model_path=getattr(args, 'local_model_path', None),
        trust_remote_code=getattr(args, 'trust_remote_code', False),
        tensor_parallel_size=getattr(args, 'tensor_parallel_size', -1),
        enable_prefix_caching=getattr(args, 'enable_prefix_caching', False),
        dtype=getattr(args, 'dtype', 'bfloat16'),
        openai_timeout=getattr(args, 'openai_timeout', 90),
    )

    evaluator_config = EvaluatorConfig(
        timeout=getattr(args, 'timeout', 6),
        debug=getattr(args, 'debug', False),
        num_process_evaluate=getattr(args, 'num_process_evaluate', 12),
        evaluation_mode=getattr(args, 'evaluation_mode', 'multiprocess'),
        test_case_count=getattr(args, 'test_case_count', 100000),
        single_test_timeout=getattr(args, 'single_test_timeout', 0.8),
        total_generation_timeout=getattr(args, 'total_generation_timeout', 60),
        boundary_bias=getattr(args, 'boundary_bias', False),
    )

    experiment_params = ExperimentParams(
        experiment_id=getattr(args, 'experiment_id', 'e1'),
        benchmark=getattr(args, 'benchmark', 'LiveCodeBench'),
        baseline=getattr(args, 'baseline', 'ExeCRE'),
        input_generator=getattr(args, 'input_generator', 'schema'),
        verifier=getattr(args, 'verifier', 'TrustTest'),
        eval_score=getattr(args, 'eval_score', None),
        error_score=getattr(args, 'error_score', None),
        allowed_error_ratio=getattr(args, 'allowed_error_ratio', 0.1),
        n=getattr(args, 'n', 1),
        difficulty=getattr(args, 'difficulty', 'all'),
        start_date=getattr(args, 'start_date', '2025-01-01'),
        end_date=getattr(args, 'end_date', '2025-05-01'),
        test_count=getattr(args, 'test_count', None),
        specific_question_id=getattr(args, 'specific_question_id', None),
        specific_question_ids=getattr(args, 'specific_question_ids', None),
        release_version=getattr(args, 'release_version', 'release_latest'),
        version=getattr(args, 'version', 'trusttest_v001'),
        not_fast=getattr(args, 'not_fast', False),
        cot_code_execution=getattr(args, 'cot_code_execution', False),
    )

    config = ExperimentConfig(
        model=model_config,
        evaluator=evaluator_config,
        experiment=experiment_params,
        multiprocess=getattr(args, 'multiprocess', 10),
        max_rounds=getattr(args, 'max_rounds', 10),
        continue_existing=getattr(args, 'continue_existing', False),
        continue_existing_with_eval=getattr(args, 'continue_existing_with_eval', False),
        use_cache=getattr(args, 'use_cache', False),
        cache_batch_size=getattr(args, 'cache_batch_size', 100),
        force_regenerate=getattr(args, 'force_regenerate', False),
        force_reevaluate=getattr(args, 'force_reevaluate', False),
        skip_generate=getattr(args, 'skip_generate', False),
        skip_evaluate=getattr(args, 'skip_evaluate', False),
        start_from=getattr(args, 'start_from', 1),
        debug=getattr(args, 'debug', False),
    )

    config.validate()
    return config


def load_config_from_file(filepath: Path) -> ExperimentConfig:
    """Load from `.json` or `.yaml` / `.yml` (YAML requires PyYAML)."""
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    if filepath.suffix == '.json':
        return ExperimentConfig.from_file(filepath)
    if filepath.suffix in ['.yaml', '.yml']:
        try:
            import yaml
        except ImportError:
            raise ImportError("Install pyyaml to load YAML config files") from None
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return ExperimentConfig.from_dict(data)

    raise ValueError(
        f"Unsupported config extension {filepath.suffix!r}; use .json, .yaml, or .yml"
    )
