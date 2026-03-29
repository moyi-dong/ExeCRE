"""
Configuration Information Printer Module

Print configuration information for debugging and logging.
"""

from src.config import ExperimentConfig


def print_config_summary(config: ExperimentConfig) -> None:
    """
    Print the key basic information of the configuration (simplified format)
    
    Args:
        config: The experiment configuration object
    """
    # Basic information
    print("=" * 80)
    print("Experiment configuration information")
    print("-" * 80)
    print(f"  Configuration hash: {config.get_config_hash()}")
    print(f"  Model:     {config.model.model}")
    print(f"  Experiment ID:   {config.experiment.experiment_id}")
    print(f"  Baseline:     {config.experiment.baseline}")
    print(f"  Dataset:   {config.experiment.benchmark}")
    print(f"  Input generator: {config.experiment.input_generator}")
    print(f"  Date range: {config.experiment.start_date} ~ {config.experiment.end_date}")
    print(f"  Version:     {config.experiment.version}")
    print(f"  Group number:     {config.experiment.n}")
    test_count_str = str(config.experiment.test_count) if config.experiment.test_count is not None else "All"
    print(f"  Test problem number: {test_count_str}")
    print(f"  Process number:   {config.multiprocess}")
    print("-" * 80)
    print("  Control parameters:")
    print(f"    force_regenerate  = {config.force_regenerate}")
    print(f"    force_reevaluate  = {config.force_reevaluate}")
    print(f"    skip_generate     = {config.skip_generate}")
    print(f"    skip_evaluate     = {config.skip_evaluate}")
    print("=" * 80)

