# ExeCRE (Anonymous Repository)

This is the anonymous repository of ExeCRE.

## Environment

```bash
conda create -n execre python=3.11 -y
conda activate execre
cd ExeCRE
pip install -e .
```

## Running

Start from a JSON config file (`--config_file` is an alias for `--config`):

```bash
python main.py --config configs/e1_execre_deepseek_chat.json
```

See the files under `configs/` for full options: the `model` block controls sampling, timeouts, etc.; the `experiment` block controls the benchmark, method, date range, and so on.

## LLM configuration

Inference backends live under `src/engine/` and are selected by the **`backend-model`** string in the config file field `model.model`, for example:

| Example `model.model` | Backend | Environment variable |
|------------------------|---------|----------------------|
| `deepseek-deepseek-chat` | DeepSeek | `DEEPSEEK_API_KEY` |
| `openai-gpt-4o` | OpenAI-compatible API | `OPENAI_API_KEY` (bare `gpt-*` names are also supported) |
| `openrouter-meta-llama/llama-3.1-8b-instruct` | OpenRouter | `OPENROUTER_API_KEY` |
| `dmx-gpt-5.1` | DMX | `DMX_API_KEY` |
| `online-qwen-turbo` | Custom OpenAI-style gateway | `ONLINE_API_KEY` (optional `ONLINE_BASE_URL`) |

Example (DeepSeek):

```bash
export DEEPSEEK_API_KEY="sk-xxxx"
python main.py --config configs/e1_execre_deepseek_chat.json
```

For the exact naming rules, see `get_engine` in `src/engine/__init__.py`.
