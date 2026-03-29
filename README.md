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
| `openai-gpt-5.4` | OpenAI-compatible API | `OPENAI_API_KEY` (bare `gpt-*` names are also supported) |
| `deepseek-deepseek-chat` | DeepSeek | `DEEPSEEK_API_KEY` |

Example (GPT):
```bash
export OPENAI_API_KEY="sk-xxxx"
python main.py --config configs/e1_execre_deepseek_chat.json
```

For the exact naming rules, see `get_engine` in `src/engine/__init__.py`.
