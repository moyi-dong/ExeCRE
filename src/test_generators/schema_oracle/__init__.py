"""
Schema Oracle test generator.

Phases: (1) LLM schema + brute-force solution candidates; (2) validate, majority-vote best pair;
(3) generate tests from that pair.

Cache: ``results/schemas/{benchmark}/{model}/{question_id}/`` → ``candidates_artifact.json``.
"""

from .generator import (
    SchemaOracleTestGenerator,
    SchemaCandidate,
    SolutionCandidate,
    CandidatesArtifact
)
from .config import SchemaOracleConfig, default_config

__all__ = [
    "SchemaOracleTestGenerator",
    "SchemaCandidate",
    "SolutionCandidate",
    "CandidatesArtifact",
    "SchemaOracleConfig",
    "default_config",
]
