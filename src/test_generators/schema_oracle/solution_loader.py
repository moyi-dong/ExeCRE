"""
Load brute-force solutions from CSVs produced by ``solution_chain.py``.

Layout: ``results/{experiment_id}/{benchmark}/{model}/{baseline}/{group_n}/{question_id}.csv``

Columns include ``Question_Id``, ``Solution_Code``, ``Is_Normal_End``, ``Passed``.
"""

import csv
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .config import SchemaOracleConfig, default_config


@dataclass
class SolutionCandidate:
    """One row turned into a candidate."""

    code: str
    group_n: int
    is_normal_end: bool
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SolutionLoader:
    """Scan ``group_*`` dirs under ``base_dir`` and read per-question CSVs."""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        config: Optional[SchemaOracleConfig] = None
    ):
        self.config = config or default_config
        self._code_col = getattr(self.config, "solution_column_name", "Solution_Code")
        self._normal_col = getattr(self.config, "is_normal_end_column", "Is_Normal_End")
        self._passed_col = getattr(self.config, "passed_column", "Passed")
        self._only_passed = getattr(self.config, "only_passed_solutions", False)
        _sb = getattr(self.config, "solution_base_dir", None)

        if base_dir:
            self.base_dir = Path(base_dir)
        elif _sb:
            self.base_dir = Path(_sb) if not isinstance(_sb, Path) else _sb
        else:
            self.base_dir = None

    def set_base_dir(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)

    def _find_group_dirs(self) -> List[Path]:
        """Numeric subdirs of ``base_dir``, sorted by group id."""
        if self.base_dir is None or not self.base_dir.exists():
            return []

        group_dirs = []
        for item in self.base_dir.iterdir():
            if item.is_dir():
                try:
                    group_n = int(item.name)
                    group_dirs.append((group_n, item))
                except ValueError:
                    continue

        group_dirs.sort(key=lambda x: x[0])
        return [path for _, path in group_dirs]

    def _read_csv_solutions(
        self,
        csv_path: Path,
        group_n: int
    ) -> List[SolutionCandidate]:
        candidates = []

        if not csv_path.exists():
            return candidates

        try:
            with open(csv_path, 'r', encoding='utf-8-sig') as csvfile:
                reader = csv.DictReader(csvfile)

                for row in reader:
                    code = row.get(self._code_col, '')
                    if not code or not code.strip():
                        continue

                    is_normal_end_str = row.get(self._normal_col, 'False')
                    is_normal_end = str(is_normal_end_str).lower() == 'true'

                    if not is_normal_end:
                        continue

                    passed_str = row.get(self._passed_col, '')
                    if passed_str:
                        passed = str(passed_str).upper() == 'TRUE'
                    else:
                        passed = None

                    if self._only_passed and passed is not True:
                        continue

                    metadata = {
                        'question_id': row.get('Question_Id', ''),
                        'question_title': row.get('Question_Title', ''),
                        'round_index': row.get('Round_Index', ''),
                        'result_type': row.get('Result_Type', ''),
                    }

                    candidate = SolutionCandidate(
                        code=code.strip(),
                        group_n=group_n,
                        is_normal_end=is_normal_end,
                        passed=passed,
                        metadata=metadata
                    )
                    candidates.append(candidate)

        except Exception as e:
            print(f"Warning: failed to read CSV {csv_path}: {e}")

        return candidates

    def load_solutions(
        self,
        question_id: str,
        max_candidates: Optional[int] = None
    ) -> List[SolutionCandidate]:
        """Collect candidates across groups until ``max_candidates``."""
        if self.base_dir is None:
            print("Error: solution base_dir not set")
            return []

        if max_candidates is None:
            max_candidates = self.config.max_simulation_candidates

        all_candidates = []
        group_dirs = self._find_group_dirs()

        for group_dir in group_dirs:
            try:
                group_n = int(group_dir.name)
            except ValueError:
                continue

            csv_path = group_dir / f"{question_id}.csv"
            candidates = self._read_csv_solutions(csv_path, group_n)
            all_candidates.extend(candidates)

            if len(all_candidates) >= max_candidates:
                break

        return all_candidates[:max_candidates]

    def load_unique_solutions(
        self,
        question_id: str,
        max_candidates: Optional[int] = None
    ) -> List[SolutionCandidate]:
        """Deduplicate by normalized code, keep first occurrence."""
        all_candidates = self.load_solutions(question_id, max_candidates=None)

        seen_codes = set()
        unique_candidates = []

        for candidate in all_candidates:
            code_key = candidate.code.strip()

            if code_key not in seen_codes:
                seen_codes.add(code_key)
                unique_candidates.append(candidate)

        if max_candidates is None:
            max_candidates = self.config.max_simulation_candidates

        return unique_candidates[:max_candidates]

    def get_solution_codes(
        self,
        question_id: str,
        max_candidates: Optional[int] = None,
        unique: bool = True
    ) -> List[str]:
        """Return code strings only."""
        if unique:
            candidates = self.load_unique_solutions(question_id, max_candidates)
        else:
            candidates = self.load_solutions(question_id, max_candidates)

        return [c.code for c in candidates]
