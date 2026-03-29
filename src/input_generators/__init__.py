from typing import Any

from .base_generator import BaseInputGenerator
from .schema.generator import SchemaInputGenerator


_GENERATORS = {
    "schema": SchemaInputGenerator,
}


def create_input_generator(name: str, engine: Any = None, **kwargs) -> BaseInputGenerator:
    if name not in _GENERATORS:
        raise ValueError(f"Unknown generator: {name}. Available: {list(_GENERATORS.keys())}")
    return _GENERATORS[name](engine=engine, **kwargs)


__all__ = [
    "BaseInputGenerator",
    "SchemaInputGenerator",
    "create_input_generator",
]
