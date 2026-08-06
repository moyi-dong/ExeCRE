# ExeCRE

Official implementation of **ExeCRE: Execution-Consistency Guided Reliability
Estimation for Self-Correcting Code Generation**. **Accepted (ASE 2026, CCF-A).**
[arXiv:2608.04439](https://arxiv.org/abs/2608.04439)

ExeCRE uses execution consistency to estimate the reliability of generated
programs and guide iterative self-correction. The repository contains the
method implementation, experiment configurations, evaluation scripts, and
analysis utilities used in the paper.

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

## Artifact and results

Artifact at [https://doi.org/10.5281/zenodo.19347857](https://doi.org/10.5281/zenodo.19347857).

GitHub is the development and reuse repository. Zenodo provides the immutable,
versioned artifact and the large experimental outputs. See
[`results/README.md`](results/README.md) for download and analysis instructions.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). If you use
ExeCRE, please cite the ASE 2026 paper and the exact Zenodo artifact version
used for reproduced results.

```bibtex
@inproceedings{dong2026execre,
  author    = {Dong, Yiru and Zhang, Richong and Kong, Fanshuang and Chen, Si},
  title     = {ExeCRE: Execution-Consistency Guided Reliability Estimation
               for Self-Correcting Code Generation},
  booktitle = {Proceedings of the 41st IEEE/ACM International Conference
               on Automated Software Engineering},
  series    = {ASE '26},
  year      = {2026},
  month     = {October},
  location  = {Munich, Germany},
  publisher = {Association for Computing Machinery},
  isbn      = {979-8-4007-2882-2}
}
```

Page or article-number metadata will be added after the final ACM Digital
Library citation becomes available.

## License

- Original ExeCRE source code: [MIT License](LICENSE)
- ExeCRE paper: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- ExeCRE experimental data and outputs archived on Zenodo:
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Vendored datasets and software: their original licenses, as documented in
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)
