# TextGrad Engine

This document describes the TextGrad engine’s inputs, environment variables, and usage.

## Basic usage

The engine supports multiple LLM backends via `get_engine`:

```python
from textgrad.engine import get_engine

engine = get_engine("openai-gpt-4")
response = engine("Briefly introduce yourself.")
```

## API keys

Set keys in your shell (or process environment):

```bash
export ONLINE_API_KEY="your-api-key"
export OPENAI_API_KEY="your-api-key"
export ANTHROPIC_API_KEY="your-api-key"
# Same pattern for other providers
```

Keys are read from the environment; they are not passed directly to `get_engine`.

### Online (OpenAI-compatible) backend

Set `ONLINE_BASE_URL` to point at your endpoint:

```bash
export ONLINE_API_KEY="your-api-key"
export ONLINE_BASE_URL="https://your-api-endpoint.com/v1"
```

If `ONLINE_BASE_URL` is unset, the default is `https://dashscope.aliyuncs.com/compatible-mode/v1`.

## Supported backends and models

| Backend | Supported models (examples) | API key env var | Other requirements | Remote / local |
|--------|-----------------------------|-----------------|--------------------|----------------|
| Online | Any OpenAI-compatible HTTP API | `ONLINE_API_KEY` | `ONLINE_BASE_URL`, model string | Remote |
| OpenAI | gpt-4, gpt-3.5-turbo, gpt-4o, gpt-4-turbo | `OPENAI_API_KEY` | Model string | Remote |
| Anthropic | claude-3-opus, claude-3-sonnet, claude-3-haiku, claude-3-5-sonnet | `ANTHROPIC_API_KEY` | Model string | Remote |
| Google | gemini-pro, gemini-pro-vision | `GOOGLE_API_KEY` | Model string | Remote |
| Cohere | command-r, command-r-plus | `COHERE_API_KEY` | Model string | Remote |
| Together | meta-llama/Llama-3-70b-chat-hf | `TOGETHER_API_KEY` | Model string | Remote |
| Ollama | llama2, mistral, mixtral | (none) | Model string | Local |
| VLLM | meta-llama/Meta-Llama-3-8B-Instruct | (none) | Model string | Local |
| Groq | mixtral-8x7b-32768, llama2-70b-4096 | `GROQ_API_KEY` | Model string | Remote |

## Model string format

Use `[backend]-[model]`, for example:

- `online-qwen-turbo` — Online backend, `qwen-turbo`
- `openai-gpt-4` — OpenAI, GPT-4
- `anthropic-claude-3-opus` — Anthropic, Claude 3 Opus

## Shortcuts

| Shortcut | Resolves to |
|----------|-------------|
| opus | anthropic-claude-3-opus-20240229 |
| haiku | anthropic-claude-3-haiku-20240307 |
| sonnet | anthropic-claude-3-sonnet-20240229 |
| sonnet-3.5 | anthropic-claude-3-5-sonnet-20240620 |
| together-llama-3-70b | together-meta-llama/Llama-3-70b-chat-hf |
| vllm-llama-3-8b | vllm-meta-llama/Meta-Llama-3-8B-Instruct |

## Multimodal

These entries support text + image input:

- openai-gpt-4-turbo
- openai-gpt-4o
- anthropic-claude-3-5-sonnet-20240620
- anthropic-claude-3-opus-20240229
- anthropic-claude-3-sonnet-20240229
- anthropic-claude-3-haiku-20240307
- openai-gpt-4-turbo-2024-04-09

## Caching

Caching is only implemented for the LiteLLM engine path.

## Examples

### Text generation

```python
from textgrad.engine import get_engine

engine = get_engine("online-qwen-turbo")
response = engine("Briefly introduce yourself.")
print(response)
```

### Multimodal

```python
from textgrad.engine import get_engine

engine = get_engine("anthropic-claude-3-opus")

with open("image.jpg", "rb") as f:
    image_data = f.read()

response = engine(["Describe this image.", image_data])
print(response)
```

### Shortcut alias

```python
from textgrad.engine import get_engine

engine = get_engine("opus")
response = engine("Briefly introduce yourself.")
print(response)
```
