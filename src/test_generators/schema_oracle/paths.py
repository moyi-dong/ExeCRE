"""Schema Oracle root of results/schemas/。question directory: {benchmark}/{model}/{question_id}/."""
from pathlib import Path

# schema_oracle/paths.py -> repo root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

SCHEMA_ORACLE_CACHE_ROOT: Path = _PROJECT_ROOT / "results" / "schemas"
