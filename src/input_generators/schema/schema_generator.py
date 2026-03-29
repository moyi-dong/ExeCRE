"""Deprecated: use ``SchemaInputGenerator`` in ``generator.py`` for new code."""

import json
import re
import warnings
from typing import Any

from .prompts import LLM_SCHEMA_PROMPT
from src.core.problem import Problem
from src.engine.base import EngineLM


def schema_generator(problem: Problem, engine: EngineLM) -> str:
    """Build schema JSON via LLM; adjust ``input_format`` from ``func_name``.

    .. deprecated::
        Use :class:`~src.input_generators.schema.generator.SchemaInputGenerator`.
    """
    warnings.warn(
        "schema_generator is deprecated; use SchemaInputGenerator instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    full_prompt = LLM_SCHEMA_PROMPT + "\n\n" + problem.question_content
    schema = engine.generate(full_prompt)

    func_name = problem.metadata.get("func_name", None)
    if schema:
        try:
            if isinstance(schema, str):
                if schema.strip().startswith("```"):
                    schema_content = re.sub(r"^```(?:json)?\s*", "", schema.strip())
                    schema_content = re.sub(r"\s*```$", "", schema_content)
                    schema_dict = json.loads(schema_content)
                else:
                    schema_dict = json.loads(schema)
            else:
                schema_dict = schema

            if func_name and str(func_name).strip():
                schema_dict["input_format"] = "function_args"
            else:
                schema_dict["input_format"] = "stdin"

            schema = json.dumps(schema_dict, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"Failed to adjust schema input_format: {e}")

    return schema
