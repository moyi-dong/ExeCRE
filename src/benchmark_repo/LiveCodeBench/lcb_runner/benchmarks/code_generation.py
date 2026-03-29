import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

from datasets import load_dataset

# datasets>=3 resolves configs explicitly; omitting `name` uses "default" and breaks cache lookup
# for builders keyed as `release_latest-version_tag=...` on the Hub.
_LCB_CODEGEN_LITE_CONFIG = "release_latest"


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        # if self.testtype == TestType.FUNCTIONAL:
        #     self.input = json.loads(self.input)
        #     self.output = json.loads(self.output)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def load_code_generation_dataset(release_version="release_v1", start_date=None, end_date=None) -> list[CodeGenerationProblem]:
    dataset = load_dataset(
        "livecodebench/code_generation_lite",
        _LCB_CODEGEN_LITE_CONFIG,
        split="test",
        version_tag=release_version,
    )
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_not_fast(release_version="release_v1") -> list[CodeGenerationProblem]:
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_streaming(release_version="release_v1", start_date=None, end_date=None, max_count=None):
    """
    Stream LiveCodeBench lite split to reduce peak memory.

    Args:
        release_version: Dataset release tag.
        start_date: Inclusive lower bound (YYYY-MM-DD), or None.
        end_date: Inclusive upper bound (YYYY-MM-DD), or None.
        max_count: Cap on yielded problems; None means no cap.

    Yields:
        CodeGenerationProblem instances passing date filters.
    """
    raw_dataset = load_dataset(
        "livecodebench/code_generation_lite",
        _LCB_CODEGEN_LITE_CONFIG,
        split="test",
        version_tag=release_version,
    )

    p_start_date = None
    p_end_date = None
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    count = 0
    
    for raw_problem in raw_dataset:
        if max_count is not None and count >= max_count:
            break

        if p_start_date is not None or p_end_date is not None:
            contest_date_str = raw_problem.get('contest_date')
            if contest_date_str:
                contest_date = datetime.fromisoformat(contest_date_str)

                if p_start_date is not None and contest_date < p_start_date:
                    continue

                if p_end_date is not None and contest_date > p_end_date:
                    continue

        problem = CodeGenerationProblem(**raw_problem)
        count += 1
        yield problem

    del raw_dataset
    print(f"Streaming loaded {count} problems")


if __name__ == "__main__":
    dataset = load_code_generation_dataset()
