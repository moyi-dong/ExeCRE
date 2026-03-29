from .base import EngineLM, CachedEngine
from textgrad.engine_experimental.litellm import LiteLLMEngine

__ENGINE_NAME_SHORTCUTS__ = {
    "opus": "anthropic-claude-3-opus-20240229",
    "haiku": "anthropic-claude-3-haiku-20240307",
    "sonnet": "anthropic-claude-3-sonnet-20240229",
    "sonnet-3.5": "anthropic-claude-3-5-sonnet-20240620",
    "together-llama-3-70b": "together-meta-llama/Llama-3-70b-chat-hf",
    "vllm-llama-3-8b": "vllm-meta-llama/Meta-Llama-3-8B-Instruct",
}

# Any better way to do this?
__MULTIMODAL_ENGINES__ = ["openai-gpt-4-turbo",
                          "openai-gpt-4o",
                          "anthropic-claude-3-5-sonnet-20240620",
                          "anthropic-claude-3-opus-20240229",
                          "anthropic-claude-3-sonnet-20240229",
                          "anthropic-claude-3-haiku-20240307",
                          "openai-gpt-4-turbo-2024-04-09",
                          ]

def dprint(msg,name="",detail=False,debug=False):
    """Indented debug print when ``debug`` is True (two-space indent per line)."""
    DEBUG_in_engine = debug
    DETAIL_in_engine = detail
    tab_n = 2
    tab_str = " " * tab_n
    if DEBUG_in_engine:
        indented_msg = f"\n{tab_str}".join(str(msg).split("\n"))
        if not DETAIL_in_engine:
            indented_msg = indented_msg.split("\n")[0]
        if indented_msg:
            print(f"D_{name}:\n{tab_str}{indented_msg}")
        else:
            print(f"D_{name}:")
    # print(f"{tab_str}type(msg):{type(msg)}")

def _check_if_multimodal(engine_name: str):
    return any([name == engine_name for name in __MULTIMODAL_ENGINES__])

def validate_multimodal_engine(engine):
    if not _check_if_multimodal(engine.model_string):
        raise ValueError(
            f"The engine provided is not multimodal. Please provide a multimodal engine, one of the following: {__MULTIMODAL_ENGINES__}")

def get_engine(engine_name: str, **kwargs) -> EngineLM:
    
    if engine_name in __ENGINE_NAME_SHORTCUTS__:
        engine_name = __ENGINE_NAME_SHORTCUTS__[engine_name]
    
    if "cache" in kwargs and "experimental" not in engine_name:
        raise ValueError(f"Cache is currently supported only for LiteLLM engines, not {engine_name}")

    
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
            return ChatOpenAI(model_string=model_name, is_multimodal=_check_if_multimodal(model_name), **kwargs)
        elif backend_name == "anthropic":
            from .anthropic import ChatAnthropic
            return ChatAnthropic(model_string=model_name, is_multimodal=_check_if_multimodal(model_name), **kwargs)
        elif backend_name == "gemini":
            from .gemini import ChatGemini
            return ChatGemini(model_string=model_name, **kwargs)
        elif backend_name == "together":
            from .together import ChatTogether
            return ChatTogether(model_string=model_name, **kwargs)
        elif backend_name == "cohere":
            from .cohere import ChatCohere
            return ChatCohere(model_string=model_name, **kwargs)
        elif backend_name == "ollama":
            from .openai import ChatOpenAI, OLLAMA_BASE_URL
            return ChatOpenAI(
                model_string=model_name,
                base_url=OLLAMA_BASE_URL,
                **kwargs
            )
        elif backend_name == "vllm":
            from .vllm import ChatVLLM
            return ChatVLLM(model_string=model_name, **kwargs)
        elif backend_name == "groq":
            from .groq import ChatGroq
            return ChatGroq(model_string=model_name, **kwargs)
        elif backend_name == "dmx":
            from .dmx import ChatDMX
            return ChatDMX(model_string=model_name, **kwargs)
        elif backend_name == "deepseek":
            from .deepseek import ChatDeepSeek
            return ChatDeepSeek(model_string=model_name, **kwargs)
        elif backend_name == "openrouter":
            from .openrouter import ChatOpenRouter
            return ChatOpenRouter(model_string=model_name, **kwargs)
    
    if engine_name in ["command-r-plus", "command-r", "command", "command-light"]:
        from .cohere import ChatCohere
        return ChatCohere(model_string=engine_name, **kwargs)
    elif (("gpt-4" in engine_name) or ("gpt-3.5" in engine_name) or ("gpt-5" in engine_name)):
        from .openai import ChatOpenAI
        return ChatOpenAI(model_string=engine_name, is_multimodal=_check_if_multimodal(engine_name), **kwargs)
    elif "claude" in engine_name:
        from .anthropic import ChatAnthropic
        return ChatAnthropic(model_string=engine_name, is_multimodal=_check_if_multimodal(engine_name), **kwargs)
    elif "gemini" in engine_name:
        from .gemini import ChatGemini
        return ChatGemini(model_string=engine_name, **kwargs)
    elif "together" in engine_name:
        from .together import ChatTogether
        engine_name = engine_name.replace("together-", "")
        return ChatTogether(model_string=engine_name, **kwargs)
    elif engine_name.startswith("ollama"):
        from .openai import ChatOpenAI, OLLAMA_BASE_URL
        model_string = engine_name.replace("ollama-", "")
        return ChatOpenAI(
            model_string=model_string,
            base_url=OLLAMA_BASE_URL,
            **kwargs
        )
    elif "vllm" in engine_name:
        from .vllm import ChatVLLM
        engine_name = engine_name.replace("vllm-", "")
        return ChatVLLM(model_string=engine_name, **kwargs)
    elif "groq" in engine_name:
        from .groq import ChatGroq
        engine_name = engine_name.replace("groq-", "")
        return ChatGroq(model_string=engine_name, **kwargs)
    else:
        raise ValueError(f"Engine {engine_name} not supported")
