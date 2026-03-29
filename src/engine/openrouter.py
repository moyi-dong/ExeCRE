try:
    from openai import OpenAI
except ImportError:
    raise ImportError("If you'd like to use OpenRouter models, please install the openai package by running `pip install openai`, and add 'OPENROUTER_API_KEY' to your environment variables.")

import os
import base64
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from typing import List, Union

from .base import EngineLM
from .engine_utils import get_image_type_from_bytes

# Default base URL for OpenRouter API
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'

# Check if the user set the OPENROUTER_BASE_URL environment variable
if os.getenv("OPENROUTER_BASE_URL"):
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")

# Default model for OpenRouter
OPENROUTER_DEFAULT_MODEL = 'meta-llama/llama-3.1-8b-instruct'


class ChatOpenRouter(EngineLM):
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, creative, and smart assistant."

    def __init__(
        self,
        model_string: str = OPENROUTER_DEFAULT_MODEL,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        is_multimodal: bool = False,
        base_url: str = None,
        **kwargs
    ):
        """
        :param model_string: OpenRouter model ID, e.g. meta-llama/llama-3.1-8b-instruct
        :param system_prompt:
        :param base_url: OpenRouter API base URL
        """
        self.system_prompt = system_prompt
        self.base_url = base_url if base_url else OPENROUTER_BASE_URL

        if os.getenv("OPENROUTER_API_KEY") is None:
            raise ValueError("Please set the OPENROUTER_API_KEY environment variable if you'd like to use OpenRouter models.")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

        self.model_string = model_string
        self.is_multimodal = is_multimodal

    @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(5))
    def generate(self, content: Union[str, List[Union[str, bytes]]], system_prompt: str = None, **kwargs):

        if isinstance(content, str):
            return self._generate_from_single_prompt(content, system_prompt=system_prompt, **kwargs)

        elif isinstance(content, list):
            has_multimodal_input = any(isinstance(item, bytes) for item in content)
            if (has_multimodal_input) and (not self.is_multimodal):
                raise NotImplementedError("Multimodal generation is only supported for models that support it.")

            return self._generate_from_multiple_input(content, system_prompt=system_prompt, **kwargs)

    def _generate_from_single_prompt(
        self, prompt: str, system_prompt: str = None, temperature=0, max_tokens=2000, top_p=0.99
    ):

        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": prompt},
            ],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response = response.choices[0].message.content
        return response

    def __call__(self, prompt, **kwargs):
        return self.generate(prompt, **kwargs)

    def _format_content(self, content: List[Union[str, bytes]]) -> List[dict]:
        """Helper function to format a list of strings and bytes into a list of dictionaries to pass as messages to the API.
        """
        formatted_content = []
        for item in content:
            if isinstance(item, bytes):
                image_type = get_image_type_from_bytes(item)
                base64_image = base64.b64encode(item).decode('utf-8')
                formatted_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image_type};base64,{base64_image}"
                    }
                })
            elif isinstance(item, str):
                formatted_content.append({
                    "type": "text",
                    "text": item
                })
            else:
                raise ValueError(f"Unsupported input type: {type(item)}")
        return formatted_content

    def _generate_from_multiple_input(
        self, content: List[Union[str, bytes]], system_prompt=None, temperature=0, max_tokens=2000, top_p=0.99
    ):
        sys_prompt_arg = system_prompt if system_prompt else self.system_prompt
        formatted_content = self._format_content(content)

        response = self.client.chat.completions.create(
            model=self.model_string,
            messages=[
                {"role": "system", "content": sys_prompt_arg},
                {"role": "user", "content": formatted_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )

        response_text = response.choices[0].message.content
        return response_text
