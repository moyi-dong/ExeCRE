"""LLM engine factory."""

from .base import EngineLM

__ENGINE_NAME_SHORTCUTS__ = {}

__MULTIMODAL_ENGINES__ = [
    "openai-gpt-4-turbo",
    "openai-gpt-4o",
    "openai-gpt-4-turbo-2024-04-09",
]


def _check_if_multimodal(engine_name: str) -> bool:
    return any(name == engine_name for name in __MULTIMODAL_ENGINES__)


def validate_multimodal_engine(engine):
    if not _check_if_multimodal(engine.model_string):
        raise ValueError(
            f"The engine provided is not multimodal. Please provide a multimodal engine, one of the following: {__MULTIMODAL_ENGINES__}"
        )


def get_engine(engine_name: str, **kwargs) -> EngineLM:
    """Return an engine for *engine_name*.

    Forms: ``backend-model`` (e.g. ``dmx-gpt-5.1``, ``openai-gpt-4o``,
    ``online-qwen-turbo``, ``deepseek-deepseek-chat``,
    ``openrouter-meta-llama/llama-3.1-8b-instruct``), or a bare ``gpt-*``
    name (OpenAI).
    """
    if engine_name in __ENGINE_NAME_SHORTCUTS__:
        engine_name = __ENGINE_NAME_SHORTCUTS__[engine_name]

    if "-" in engine_name:
        backend_name, model_name = engine_name.split("-", 1)

        if backend_name == "online":
            from .online_openai_style import ChatOnline, ONLINE_BASE_URL
            return ChatOnline(
                model_string=model_name,
                base_url=ONLINE_BASE_URL,
                **kwargs
            )
        elif backend_name == "openai":
            from .openai import ChatOpenAI
            return ChatOpenAI(
                model_string=model_name,
                is_multimodal=_check_if_multimodal(engine_name),
                **kwargs
            )
        elif backend_name == "dmx":
            from .dmx import ChatDMX
            return ChatDMX(model_string=model_name, **kwargs)
        elif backend_name == "deepseek":
            from .deepseek import ChatDeepSeek
            return ChatDeepSeek(
                model_string=model_name,
                is_multimodal=_check_if_multimodal(engine_name),
                **kwargs
            )
        elif backend_name == "openrouter":
            from .openrouter import ChatOpenRouter, OPENROUTER_BASE_URL
            return ChatOpenRouter(
                model_string=model_name,
                base_url=OPENROUTER_BASE_URL,
                **kwargs
            )

    if ("gpt-4" in engine_name) or ("gpt-3.5" in engine_name) or ("gpt-5" in engine_name):
        from .openai import ChatOpenAI
        return ChatOpenAI(
            model_string=engine_name,
            is_multimodal=_check_if_multimodal(engine_name),
            **kwargs
        )
    raise ValueError(
        f"Engine {engine_name} not supported. Supported formats: 'backend-model' "
        "(e.g., 'dmx-gpt-5.1', 'openai-gpt-4o', 'online-qwen-turbo', "
        "'deepseek-deepseek-chat', 'openrouter-meta-llama/llama-3.1-8b-instruct') "
        "or model name starting with 'gpt-'"
    )
