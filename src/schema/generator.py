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
    """Generate random inputs from LLM-produced schema."""
    
    def __init__(
        self, 
        engine: Any = None, 
        config: Optional["ExperimentConfig"] = None,
        **kwargs
    ):
        super().__init__(name="Schema", engine=engine, config=config, **kwargs)
    
    def initialize(self, problem: Problem) -> bool:
        if self.engine is None:
            print("Error: LLM engine is required")
            return False
        
        self._problem = problem
        
        try:
            full_prompt = LLM_SCHEMA_PROMPT + "\n\n" + problem.question_content
            
            schema_response = self.engine.generate(full_prompt, temperature=self.sampling_temperature)
            
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
            print(f"Schema generation failed: {str(e)}")
            return False
    
    def _process_schema_response(self, schema_response: str, problem: Problem) -> Optional[str]:
        """Normalize and validate schema response text."""
        try:
            schema_str = schema_response.strip()
            
            if schema_str.startswith('```'):
                schema_str = re.sub(r'^```(?:json)?\s*', '', schema_str)
                schema_str = re.sub(r'\s*```$', '', schema_str)
            
            schema_str = schema_str.strip()
            
            schema_dict = json.loads(schema_str)
            
            func_name = problem.metadata.get("func_name", None)
            if func_name and str(func_name).strip():
                schema_dict["input_format"] = "function_args"
            else:
                schema_dict["input_format"] = "stdin"
            
            return json.dumps(schema_dict, ensure_ascii=False, indent=2)
            
        except json.JSONDecodeError as e:
            print(f"Schema JSON parse failed: {str(e)}")
            print(f"Raw response: {schema_response[:500]}...")
            return None
        except Exception as e:
            print(f"Schema processing failed: {str(e)}")
            return None
    
    def generate(self) -> str:
        self._check_initialized()
        
        try:
            return generate_data_from_schema(self._artifact)
        except Exception as e:
            raise RuntimeError(f"Data generation failed: {str(e)}")
