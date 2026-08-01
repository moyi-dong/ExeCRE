# Experimental results

Run from the repository **root**.

## Experimental data archive

Large experimental outputs are not stored in Git. They are archived in the
[latest Zenodo artifact](https://doi.org/10.5281/zenodo.19347857).

Download and unpack the artifact package. Locate the split result files
`results.rar.part-00` and `results.rar.part-01`, then reconstruct and extract
the result archive from the directory containing those files:

```bash
cat results.rar.part-* > results.rar
unrar x results.rar
```

Alternatively, use a graphical archive tool that supports split RAR archives.
Place the extracted result tree under this repository's `results/` directory
before running the commands below. The unpacked results require approximately
1 GB of disk space.

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
