"""LLM-backed schema generation and random input sampling from that schema."""

import json
import re
from typing import Any, Optional, TYPE_CHECKING

from ..base_generator import BaseInputGenerator
from src.core.problem import Problem
from .prompts import LLM_SCHEMA_PROMPT
from .schema import generate_data_from_schema

if TYPE_CHECKING:
    from src.config.experiment_config import ExperimentConfig


class SchemaInputGenerator(BaseInputGenerator):
    """Generate a problem schema with the engine, then sample inputs from it."""

    def __init__(
        self,
        engine: Any = None,
        config: Optional["ExperimentConfig"] = None,
        **kwargs,
    ):
        super().__init__(name="Schema", engine=engine, config=config, **kwargs)

    def initialize(self, problem: Problem) -> bool:
        """Call the LLM to build schema for ``problem``; store result on success."""
        if self.engine is None:
            print("Error: no LLM engine provided")
            return False

        self._problem = problem

        try:
            full_prompt = LLM_SCHEMA_PROMPT + "\n\n" + problem.question_content
            schema_response = self.engine.generate(
                full_prompt, temperature=self.sampling_temperature
            )

            if not schema_response:
                print("Error: empty LLM response")
                return False

            schema = self._process_schema_response(schema_response, problem)
            if schema is None:
                return False

            self._artifact = schema
            self._initialized = True
            return True

        except Exception as e:
            print(f"Schema generation failed: {e}")
            return False

    def _process_schema_response(
        self, schema_response: str, problem: Problem
    ) -> Optional[str]:
        """Strip markdown fences, parse JSON, set ``input_format`` from ``func_name``."""
        try:
            schema_str = schema_response.strip()
            if schema_str.startswith("```"):
                schema_str = re.sub(r"^```(?:json)?\s*", "", schema_str)
                schema_str = re.sub(r"\s*```$", "", schema_str)

            schema_str = schema_str.strip()
            schema_dict = json.loads(schema_str)

            func_name = problem.metadata.get("func_name", None)
            if func_name and str(func_name).strip():
                schema_dict["input_format"] = "function_args"
            else:
                schema_dict["input_format"] = "stdin"

            return json.dumps(schema_dict, ensure_ascii=False, indent=2)

        except json.JSONDecodeError as e:
            print(f"Schema JSON parse error: {e}")
            print(f"Raw response (truncated): {schema_response[:500]}...")
            return None
        except Exception as e:
            print(f"Schema processing failed: {e}")
            return None

    def generate(self) -> str:
        """Sample one input string from the initialized schema."""
        self._check_initialized()
        try:
            return generate_data_from_schema(self._artifact)
        except Exception as e:
            raise RuntimeError(f"Data generation failed: {e}") from e
