# ExeCRE

Official implementation of **ExeCRE: Execution-Consistency Guided Reliability
Estimation for Self-Correcting Code Generation**.

ExeCRE uses execution consistency to estimate the reliability of generated
programs and guide iterative self-correction. The repository contains the
method implementation, experiment configurations, evaluation scripts, and
analysis utilities used in the paper.

## Artifact and results

- Latest archived artifact: [Zenodo, DOI 10.5281/zenodo.19347857](https://doi.org/10.5281/zenodo.19347857)

GitHub is the development and reuse repository. Zenodo provides the immutable,
versioned artifact and the large experimental outputs. See
[`results/README.md`](results/README.md) for download and analysis instructions.

## Environment

```bash
conda create -n execre python=3.11 -y
conda activate execre
cd ExeCRE
pip install -e .
```

Some evaluations execute model-generated code. Run them only in an isolated
environment such as a sandbox, container, or virtual machine.

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

## Repository layout

- `configs/`: experiment configurations
- `src/baselines/`: ExeCRE and baseline implementations
- `src/benchmark_repo/`: vendored benchmark code and data
- `src/evaluators/`: execution-based evaluators
- `src/analyzers/`: scripts for reproducing reported analyses
- `results/`: instructions for obtaining and analyzing archived outputs

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). When reporting
results reproduced from the paper, use the version-specific Zenodo DOI cited
by the paper.

## License

Original ExeCRE material is released under the [MIT License](LICENSE).
Vendored datasets and software retain their original licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
