"""
Path Management Module

Provides unified path management functionality to avoid using sys.path.insert in code.
Provides standardized path retrieval functions based on the PATH_DESIGN.md defined path structure.
"""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Determine the project root directory by finding the parent directory containing the src/ and configs/ directories.
    If the current file is inside the project, it will search upwards until the project root directory is found.
    
    Returns:
        Path: The Path object of the project root directory
        
    Raises:
        RuntimeError: If the project root directory cannot be found
    """
    # Start searching from the current file location
    current_file = Path(__file__).resolve()
    
    # Search upwards until the directory containing src/ and configs/ is found
    for parent in [current_file] + list(current_file.parents):
        # Check if the directory contains src/ and configs/
        if (parent / "src").exists() and (parent / "configs").exists():
            return parent
    
    # If not found, try using the current working directory
    cwd = Path.cwd()
    if (cwd / "src").exists() and (cwd / "configs").exists():
        return cwd
    
    raise RuntimeError(
        "The project root directory cannot be found. Please ensure that you are running in the project root directory, "
        "and the project root directory contains the src/ and configs/ directories."
    )


def get_results_dir(
    experiment_id: str,
    benchmark: str,
    model: str,
    baseline: str,
    config_hash: str
) -> Path:
    """
    Get the standard results directory path.
    
    Path format: results/{experiment_id}/{benchmark}/{model}/{baseline}/{config_hash}/
    
    Args:
        experiment_id: Experiment identifier (e.g., e1, e2, e3, etc.)
        benchmark: Benchmark name (e.g., LiveCodeBench, HumanEval, MBPP)
        model: Model name (e.g., dmx-deepseek-v3-241226, openai-gpt-5.1)
        baseline: Baseline name (e.g., ExeCRE, Textgrad, DirectAnswer, ...)
        config_hash: Configuration parameter hash (first 8 hex chars of SHA256)
    
    Returns:
        Path: The Path object of the results directory (the directory may not exist, so you need to call mkdir(parents=True, exist_ok=True) to create it)
    """
    project_root = get_project_root()
    return project_root / "results" / experiment_id / benchmark / model / baseline / config_hash


def get_analysis_dir(
    experiment_id: str,
    benchmark: str,
    model: str
) -> Path:
    """
    Get the analysis results directory path.
    
    Path format: analysis/{experiment_id}/{benchmark}/{model}/
    
    Args:
        experiment_id: Experiment identifier (e.g., e1, e2, e3, etc.)
        benchmark: Benchmark name (e.g., LiveCodeBench, HumanEval, MBPP)
        model: Model name (e.g., dmx-deepseek-v3-241226, openai-gpt-5.1)
    
    Returns:
        Path: The Path object of the analysis results directory (the directory may not exist, so you need to call mkdir(parents=True, exist_ok=True) to create it)
    """
    project_root = get_project_root()
    return project_root / "analysis" / experiment_id / benchmark / model


def get_config_dir() -> Path:
    """
    Get the configuration file directory path.
    
    Path format: configs/
    
    Returns:
        Path: The Path object of the configuration directory
    """
    project_root = get_project_root()
    return project_root / "configs"


def get_src_dir() -> Path:
    """
    Get the source code directory path.
    
    Path format: src/
    
    Returns:
        Path: The Path object of the source code directory
    """
    project_root = get_project_root()
    return project_root / "src"


def get_group_dir(
    experiment_id: str,
    benchmark: str,
    model: str,
    baseline: str,
    config_hash: str,
    group_n: int
) -> Path:
    """
    Get the multi-group experiment directory path.
    
    Path format: results/{experiment_id}/{benchmark}/{model}/{baseline}/{config_hash}/group_{n}/
    
    Args:
        experiment_id: Experiment identifier (e.g., e1, e2, e3, etc.)
        benchmark: Benchmark name (e.g., LiveCodeBench, HumanEval, MBPP)
        model: Model name (e.g., dmx-deepseek-v3-241226, openai-gpt-5.1)
        baseline: Baseline name (e.g., ExeCRE, Textgrad, DirectAnswer, ...)
        config_hash: Configuration parameter hash (first 8 hex chars of SHA256)
        group_n: Group number (1, 2, 3, ...)
    
    Returns:
        Path: The Path object of the group experiment directory (the directory may not exist, so you need to call mkdir(parents=True, exist_ok=True) to create it)
    """
    results_dir = get_results_dir(experiment_id, benchmark, model, baseline, config_hash)
    return results_dir / f"group_{group_n}"


def get_direct_answer_group_dir(
    config: 'ExperimentConfig', 
    group_n: int,
    temperature: Optional[float] = None
) -> Path:
    """
    Get the results directory for the DirectAnswer baseline.
    
    Used for Solvers that need initial code, such as Textgrad, to read the initial code from the DirectAnswer results.
    
    Args:
        config: Experiment configuration object
        group_n: Group number (1, 2, 3, ...)
        temperature: Optional, specify the temperature value. If None, use config.model.temperature (default value).
                     If a temperature value is specified, the configuration hash for the corresponding temperature will be calculated by temporarily modifying the configuration.
    
    Returns:
        Path: The Path object of the group experiment directory for the DirectAnswer baseline
    """
    # If a temperature is specified, temporarily modify the configuration to calculate the correct hash
    original_temperature = None
    if temperature is not None:
        original_temperature = config.model.temperature
        config.model.temperature = temperature
    
    try:
        # Calculate the configuration hash for the DirectAnswer baseline
        direct_answer_hash = config.get_config_hash_for_baseline("DirectAnswer")
    finally:
        # Restore the original temperature value (if previously modified)
        if original_temperature is not None:
            config.model.temperature = original_temperature
    
    return get_group_dir(
        experiment_id=config.experiment.experiment_id,
        benchmark=config.experiment.benchmark,
        model=config.model.model,
        baseline="DirectAnswer",
        config_hash=direct_answer_hash,
        group_n=group_n
    )


def get_Bruteforce_group_dir(
    config: 'ExperimentConfig', 
    group_n: int,
    temperature: Optional[float] = None
) -> Path:
    """
    Get the results directory for the Bruteforce baseline.
    
    Used for Solvers that need to read data from the Bruteforce results.
    
    Args:
        config: Experiment configuration object
        group_n: Group number (1, 2, 3, ...)
        temperature: Optional, specify the temperature value. If None, use config.model.temperature (default value).
                     If a temperature value is specified, the configuration hash for the corresponding temperature will be calculated by temporarily modifying the configuration.
    
    Returns:
        Path: The Path object of the group experiment directory for the Bruteforce baseline
    """
    # If a temperature is specified, temporarily modify the configuration to calculate the correct hash
    original_temperature = None
    if temperature is not None:
        original_temperature = config.model.temperature
        config.model.temperature = temperature
    
    try:
        # Calculate the configuration hash for the Bruteforce baseline
        bruteforce_hash = config.get_config_hash_for_baseline("Bruteforce")
    finally:
        # Restore the original temperature value (if previously modified)
        if original_temperature is not None:
            config.model.temperature = original_temperature
    
    return get_group_dir(
        experiment_id=config.experiment.experiment_id,
        benchmark=config.experiment.benchmark,
        model=config.model.model,
        baseline="Bruteforce",
        config_hash=bruteforce_hash,
        group_n=group_n
    )
