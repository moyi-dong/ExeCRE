# `results/` — reading outputs

Run from the repository **root**.

## Experimental data archive

A `results.rar` file may sit **next to this README** (same `results/` directory). It contains the experiment outputs referenced below. **Extract it before** using the paths or commands in this document. The unpacked tree is large (on the order of **~1 GB**), so allow enough disk space.

## Pass@1 on LiveCodeBench (by difficulty) from existing runs

Reuses saved CSVs only (no generation or re-evaluation):

```bash
python -m src.experiments.solution_chain --config_file configs/e1_execre_gpt52.json --skip_generate --skip_evaluate
```

Results layout: `results/e1/LiveCodeBench/openai-gpt-5.2/ExeCRE/<config-hash>/group_1` … (one `group_k` per `experiment.n`).

## Misleading-feedback stats (multi-round correction counts)

- **ExeCRE**

```bash
python -m src.analyzers.e4_verify_classification.multi_round_mislead --config_file configs/e1_execre_gpt52.json
```

- **Textgrad**

```bash
python -m src.analyzers.e4_verify_classification.multi_round_mislead --config_file configs/e1_textgrad_gpt52.json
```

You can also pass a `group_1` path under the run directory; the tool resolves the run folder and aggregates `group_*` the same way.
