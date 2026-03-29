"""
Parameter Parsing Module

Parse command line arguments for all experimental parameters.
Based on argparse, compatible with the existing parser.py design.
"""

import os
import sys
import argparse
from typing import Optional
from pathlib import Path

from src.config import ExperimentConfig, load_config_from_args, load_config_from_file


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: The parsed parameter object
    """
    parser = argparse.ArgumentParser(
        description="ExeCRE experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # ==================== Model parameters ====================
    model_group = parser.add_argument_group("Model parameters")
    model_group.add_argument(
        "--model",
        type=str,
        default="dmx-deepseek-v3-241226",
        help="Model name. For DMX API models, use 'dmx-{model_name}' format, "
             "e.g. 'dmx-deepseek-v3-241226', 'dmx-qwen3-235b-a22b-instruct-2507'"
    )
    model_group.add_argument(
        "--local_model_path",
        type=str,
        default=None,
        help="Local model path (if using local model)"
    )
    model_group.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether to trust remote code (for HuggingFace models)"
    )
    model_group.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for model call engine, to make the data reliable"
    )
    model_group.add_argument(
        "--top_p",
        type=float,
        default=0.99,
        help="Top-p sampling parameter"
    )
    model_group.add_argument(
        "--sampling_temperature",
        type=float,
        default=1.0,
        help="Multiple sampling temperature (for multiple sampling scenarios), to ensure the temperature of sampling difference"
    )
    model_group.add_argument(
        "--max_tokens",
        type=int,
        default=2000,
        help="Maximum number of generated tokens"
    )
    model_group.add_argument(
        "--stop",
        type=str,
        default="###",
        help="Stop tokens (multiple tokens separated by commas)"
    )
    model_group.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=-1,
        help="Tensor parallel size (for vllm, -1 means automatic)"
    )
    model_group.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        help="Enable prefix caching (for vllm)"
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Data type (for vllm)"
    )
    model_group.add_argument(
        "--openai_timeout",
        type=int,
        default=90,
        help="OpenAI API timeout (seconds)"
    )
    
    # ==================== Experiment parameters ====================
    experiment_group = parser.add_argument_group("Experiment parameters")
    experiment_group.add_argument(
        "--experiment_id",
        type=str,
        default="e1",
        choices=["e1", "e2", "e3", "e4", "e5", "e6", "e7"],
        help="Experiment identifier: e1(main pass@1), e2(pass@1 extended), e3(brute force gap), "
             "e4(Verify classification), e5(round pass@1), e6(parameter analysis), e7(input analysis)"
    )
    experiment_group.add_argument(
        "--benchmark",
        type=str,
        default="LiveCodeBench",
        choices=["LiveCodeBench", "HumanEval", "MBPP"],
        help="Benchmark name"
    )
    experiment_group.add_argument(
        "--baseline",
        type=str,
        default="ExeCRE",
        choices=[
            "DirectAnswer",
            "Bruteforce",
            "ExeCRE",
            "TrustTestEM",
            "Textgrad",
            "GSM8KPM",
            "GSM8KExeCRE",
        ],
        help="Baseline method name"
    )
    experiment_group.add_argument(
        "--input_generator",
        type=str,
        default="schema",
        choices=["schema", "llm_direct", "llm_suite", "execution_symbolic"],
        help="Input generation method (for experiment E7)"
    )
    experiment_group.add_argument(
        "--verifier",
        type=str,
        default="TrustTest",
        choices=["TrustTest", "Textgrad"],
        help="Verification method (for experiment E4)"
    )
    
    # ==================== TrustTest hyperparameters ====================
    trusttest_group = parser.add_argument_group("TrustTest hyperparameters (for experiment E6)")
    trusttest_group.add_argument(
        "--eval_score",
        type=float,
        default=None,
        help="Eval score threshold (if None, use default value)"
    )
    trusttest_group.add_argument(
        "--error_score",
        type=float,
        default=None,
        help="Error score threshold (if None, use default value)"
    )
    trusttest_group.add_argument(
        "--allowed_error_ratio",
        type=float,
        default=0.1,
        help="Allowed error ratio threshold"
    )
    
    # ==================== Running parameters ====================
    run_group = parser.add_argument_group("Running parameters")
    run_group.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of experiments (for multiple experiments)"
    )
    run_group.add_argument(
        "--multiprocess",
        type=int,
        default=10,
        help="Number of parallel processes (-1 means using CPU cores)"
    )
    
    # ==================== Dataset filtering parameters ====================
    dataset_group = parser.add_argument_group("Dataset filtering parameters")
    dataset_group.add_argument(
        "--difficulty",
        type=str,
        default="all",
        choices=["easy", "medium", "hard", "all"],
        help="Difficulty level filtering"
    )
    dataset_group.add_argument(
        "--start_date",
        type=str,
        default="2025-01-01",
        help="Start date (format: YYYY-MM-DD, default: 2025-01-01)"
    )
    dataset_group.add_argument(
        "--end_date",
        type=str,
        default="2025-05-01",
        help="End date (format: YYYY-MM-DD, default: 2025-05-01)"
    )
    dataset_group.add_argument(
        "--test_count",
        type=int,
        default=None,
        help="Number of test problems (None means all)"
    )
    dataset_group.add_argument(
        "--specific_question_id",
        type=str,
        default=None,
        help="Specify a single problem ID (e.g., '3579' or 'abc374_c')"
    )
    dataset_group.add_argument(
        "--specific_question_ids",
        type=str,
        nargs='+',
        default=None,
        help="Specify multiple problem IDs (e.g., '3579 abc374_c 1234')"
    )
    dataset_group.add_argument(
        "--release_version",
        type=str,
        default="release_latest",
        help="Dataset version"
    )
    
    # ==================== Evaluator configuration ====================
    evaluator_group = parser.add_argument_group("Evaluator configuration")
    evaluator_group.add_argument(
        "--timeout",
        type=int,
        default=6,
        help="Timeout for a single test case (seconds)"
    )
    evaluator_group.add_argument(
        "--num_process_evaluate",
        type=int,
        default=12,
        help="Number of parallel evaluation processes"
    )
    evaluator_group.add_argument(
        "--evaluation_mode",
        type=str,
        default="multiprocess",
        choices=["multiprocess", "single_process", "single_process_safe"],
        help="Evaluation mode: multiprocess(multiple processes), single_process(single process), single_process_safe(safe single process)"
    )
    evaluator_group.add_argument(
        "--test_case_count",
        type=int,
        default=100000,
        help="Number of random test cases (for Diff/TLE evaluator)"
    )
    evaluator_group.add_argument(
        "--single_test_timeout",
        type=float,
        default=0.8,
        help="Timeout for a single random test (seconds)"
    )
    evaluator_group.add_argument(
        "--total_generation_timeout",
        type=int,
        default=60,
        help="Total generation timeout (seconds)"
    )
    evaluator_group.add_argument(
        "--boundary_bias",
        action="store_true",
        help="Whether to enable boundary bias (for random tests)"
    )
    
    # ==================== Running control parameters ====================
    control_group = parser.add_argument_group("Running control parameters")
    control_group.add_argument(
        "--continue_existing",
        action="store_true",
        help="Continue existing experiment (resume from checkpoint): reuse generated code, skip completed problems, re-evaluate unevaluated problems"
    )
    control_group.add_argument(
        "--continue_existing_with_eval",
        action="store_true",
        help="Continue existing experiment and reuse evaluation results: reuse generated code and evaluation results, skip completed problems"
    )
    control_group.add_argument(
        "--use_cache",
        action="store_true",
        help="Use cache"
    )
    control_group.add_argument(
        "--cache_batch_size",
        type=int,
        default=100,
        help="Cache batch size"
    )
    control_group.add_argument(
        "--force_regenerate",
        action="store_true",
        help="Force regenerate all results"
    )
    control_group.add_argument(
        "--force_reevaluate",
        action="store_true",
        help="Force re-evaluate all results"
    )
    control_group.add_argument(
        "--skip_generate",
        action="store_true",
        help="Skip code generation stage"
    )
    control_group.add_argument(
        "--skip_evaluate",
        action="store_true",
        help="Skip evaluation stage"
    )
    control_group.add_argument(
        "--start_from",
        type=int,
        default=1,
        help="Start from which group (skip the generation and evaluation of previous groups), e.g., --start_from 2 starts from the 2nd group"
    )
    control_group.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode"
    )
    
    # ==================== Other parameters ====================
    other_group = parser.add_argument_group("Other parameters")
    other_group.add_argument(
        "--version",
        type=str,
        default="trusttest_v001",
        help="Version identifier, used for result saving path"
    )
    other_group.add_argument(
        "--not_fast",
        action="store_true",
        help="Whether to use the complete test set (slower but more comprehensive)"
    )
    other_group.add_argument(
        "--cot_code_execution",
        action="store_true",
        help="Whether to use CoT in the code execution scenario"
    )
    other_group.add_argument(
        "--config_file",
        "--config",
        type=str,
        dest="config_file",
        default=None,
        help="Configuration file path (JSON or YAML), if specified, will load the configuration from the file and override the command line parameters"
    )
    
    args = parser.parse_args()
    
    # Process stop parameter (convert to list)
    if isinstance(args.stop, str):
        args.stop = [s.strip() for s in args.stop.split(",") if s.strip()]
    
    # Process automatic parameters
    if args.tensor_parallel_size == -1:
        try:
            import torch
            args.tensor_parallel_size = torch.cuda.device_count()
        except (ImportError, RuntimeError):
            args.tensor_parallel_size = 1
    
    if args.multiprocess == -1:
        args.multiprocess = os.cpu_count() or 1
    
    return args


def get_config_from_args(args: Optional[argparse.Namespace] = None) -> ExperimentConfig:
    """
    Get configuration object from command line parameters
    
    Args:
        args: argparse.Namespace object, if None, automatically parse command line parameters
        
    Returns:
        ExperimentConfig: Configuration object
    """
    if args is None:
        args = get_args()
    
    # If a configuration file is specified, load it from the file first
    if hasattr(args, 'config_file') and args.config_file:
        config_path = Path(args.config_file)
        print(f"config_path: {config_path}")
        config = load_config_from_file(config_path)
        print(f"config: {config}")
        # Command line override: --skip_generate / --skip_evaluate / --n can still be accepted when using JSON configuration
        if getattr(args, 'skip_generate', False):
            config.skip_generate = True
        if getattr(args, 'skip_evaluate', False):
            config.skip_evaluate = True
        # Only override when --n is explicitly passed (to avoid using the default value to overwrite n in json)
        if '--n' in sys.argv and hasattr(args, 'n'):
            config.experiment.n = args.n
        if '--start_from' in sys.argv and hasattr(args, 'start_from'):
            config.start_from = args.start_from
        return config
    
    # Create configuration from command line parameters
    return load_config_from_args(args)


if __name__ == "__main__":
    # Test parameter parsing
    args = get_args()
    print("Parsed parameters:")
    print(args)
    
    # Test configuration creation
    config = get_config_from_args(args)
    print("\nConfiguration object:")
    print(config.to_json())
    print(f"\nConfiguration hash: {config.get_config_hash()}")
