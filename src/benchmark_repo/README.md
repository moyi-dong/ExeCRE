# Benchmark setup

This folder holds benchmark data and vendored benchmark code used in our experiments.

## Dataset sources

- **GSM8K** — dataset from:  
  https://codeload.github.com/openai/grade-school-math/zip/refs/heads/master  

- **LiveCodeBench** — dataset from:  
  https://github.com/LiveCodeBench/LiveCodeBench  

## Limitations of the evaluations

For transparency, we highlight limitations of the evaluations.

**Result stability.** Getting stable results from code-oriented tasks is challenging. The code optimization setting is especially volatile: even small edits to a snippet can change measured outcomes. We mitigate this by running multiple trials with different random seeds and reporting averages (or aggregates), but variance cannot be fully removed. This instability is not specific to our method—baselines exhibit it as well.

**Environment and tooling.** The pipeline may use retries (e.g., for flaky I/O or API calls). For reproducibility, **pin your environment** (Python version, dependency versions, OS where relevant). Any **model API calls require valid API keys** (e.g., `API_KEY` or provider-specific variables); configure these according to your config files and documentation.

## Safety: running model-generated code

Similar in spirit to disclaimers for [HumanEval](https://github.com/openai/human-eval)-style evaluation: **this repository is meant to run untrusted, model-generated code.**

- Run evaluations only in an **isolated environment**—e.g., a **sandbox, VM, or container**—not directly on a production host.
- Smaller or less capable models may still emit **destructive or unsafe code**. Treat all generated code as untrusted until reviewed.
- Some scripts or paths may **disable or gate dangerous execution by default**. If you enable execution, do so only after you understand the risk and have isolated the runtime.

Upstream benchmark code (e.g., LiveCodeBench) may include its own notes on execution not being a full security sandbox; see the relevant files under `LiveCodeBench/` for details.

## Third-party code and our changes

- **Benchmark code:** We do not modify the core benchmark logic shipped with GSM8K / LiveCodeBench-style trees beyond what is needed to place data and wire paths.
- **Other repositories** referenced or adapted in this project may have received **environment or interface adjustments** only (paths, imports, API wrappers), not algorithmic changes to the original methods.

Relevant third-party / replication sources include (non-exhaustive):

- https://github.com/DJjjjhao/replication_package  
- https://zenodo.org/records/10390291  
- https://github.com/ZJU-CTAG/B4  
- https://github.com/zkx06111/ALGO  
- https://github.com/zou-group/textgrad  

Portions of the surrounding codebase may be adapted from prior work (e.g., replication or baseline implementations); see file-level comments and git history where applicable.
