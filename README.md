# ExLightNet reproduction

An independent, evidence-based CPU reproduction of the supplied *ExLightNet* paper. The
authoritative brief is preserved in [docs/BRIEF.txt](docs/BRIEF.txt). See
[REPORT.md](REPORT.md) for measured gates, execution status, assumptions and
limitations. Fast results are explicitly separate from full five-seed results.
The included full campaign contains 95 model results across seeds 42--46 and
95 isolated single-thread benchmarks. The delivered notebook was executed in
full mode from these caches; all 14 code cells completed without errors.

## Install and verify

Python 3.11, roughly 2 GB free for dependencies and artifacts, CPU only:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
pytest -q
```

`requirements.txt` pins the direct dependencies, including the official CPU
PyTorch and XGBoost distributions. `requirements-lock.txt`, when included,
records all versions actually used on the execution host. If PyTorch's CDN is
unreachable, the same pinned wheel is available from its official origin:

```bash
python -m pip install 'https://s3.amazonaws.com/pytorch/whl/cpu/torch-2.6.0%2Bcpu-cp311-cp311-linux_x86_64.whl'
```

Put the genuine CSV at `data/RT_IOT2022.csv`. If absent, the loader tries
`ucimlrepo.fetch_ucirepo(id=942)`, then fails clearly if it cannot obtain data.
The dataset is excluded from the archive. Source:
[UCI RT-IoT2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022), CC BY 4.0.

## Run

```bash
python -m exlightnet.run_all --mode fast
python scripts/run_parallel.py
# Equivalent installed CLI:
exlightnet --mode full
```

Fast mode uses seed 42, one epoch per final deep model and per alpha
candidate, the complete real train/validation/test splits, original tree counts,
one-seed ablation demonstrations, and SHAP on 16 train backgrounds/12 random test
flows. It targets a laptop run of about 15 minutes; actual runtime is recorded.
No seed SD can be estimated from fast mode.

Full mode defaults to seeds 42–46 for both tasks, every baseline and every
ablation, with the requested 20/30 epoch limits, patience six, and 200/300 SHAP
sizes. The alpha sweep is capped at eight epochs per candidate, separately per
task, on seed 42 validation only. Selected alpha is frozen across all deep
models and seeds. Expect substantially longer CPU execution.

The preferred full workflow runs one alpha sweep, then five parallel processes
with one thread each, one cache-only aggregation, and a final benchmark-only
process after training and explanations finish:

```bash
python -m exlightnet.run_all --mode full --alpha-only --skip-benchmark
python scripts/run_parallel.py --reuse-alpha
```

`run_parallel.py` records commands, PIDs, timestamps and exit statuses in
`results/full/parallel_execution.json`. It sets OMP, OpenBLAS, MKL and NumExpr
thread limits before importing scientific libraries in each worker. Prepared
data and alpha are read-only in workers; checkpoints and metrics are unique to
each seed. Shared summaries/reports are published only by aggregation.
Seed 42 owns the shared SHAP cache; the other seed workers disable explanations.
Aggregation reads that cache and never starts new SHAP, deletion or proxy work.
The launcher waits for at least 3072 MiB of available RAM before starting all
five workers; `--min-available-ram-mb` makes that resource threshold explicit.
It preserves five-way concurrency and does not silently change the model batch
size or training recipe when the desktop is short on memory.

The individual stages are also available directly:

```bash
# Launch one such process for each seed 42..46, concurrently:
python -m exlightnet.run_all --mode full --seed-worker --seeds 42 --skip-benchmark
# Add --no-explanations for seeds 43..46.
# After all five workers exit:
python -m exlightnet.run_all --mode full --aggregate-only --skip-benchmark
# After aggregation/SHAP and all other experiment processes finish:
python -m exlightnet.run_all --mode full --benchmark-only
```

`--skip-benchmark` suppresses timing. Worker, alpha-only and aggregate-only
stages imply it. Aggregation refuses to train or repeat final test inference for
missing caches. Benchmark-only loads persisted models and training inputs,
checks three consecutive one-second CPU windows at <=5% total busy before each
model, and records idle evidence with the observations. File locks prevent
overlap between seed workers and benchmark/aggregation stages. No benchmark is
performed inside a seed worker.
Locks cover both fast and full modes because preprocessing and report paths are
shared. A failed benchmark refresh removes previous timings from the current
summary and preserves them in the campaign audit files. Metric-specific `_n`
columns record how many observations actually contribute to each mean/SD.

Useful partial runs (clearly marked incomplete in the manifest):

```bash
python -m exlightnet.run_all --mode fast --models exlightnet --no-ablations
python -m exlightnet.run_all --mode full --seeds 42 43 44 --models exlightnet mlp
# Optional stretch, after ordinary full results; adds a class-clustering PROXY:
python -m exlightnet.run_all --mode full --leakage-proxy
```

The optional proxy refits preprocessing on its new training split and compares
three paired seeds. It is **not campaign-disjoint**. KD and true static INT8
compute are not implemented.

## Artifacts and evaluation discipline

- `artifacts/`: class names, ranked features, fitted preprocessing, split
  indices, checksummed array cache; caches stay outside the archive.
- `results/{fast,full}/`: per-task/model/seed checkpoints, epoch CSVs, final
  metrics and predictions, test-evaluation receipts, summary CSV, alpha sweep,
  ablations, quantisation gates, SHAP values/plots, hardware and manifests.
- `summary.csv`: means, sample SDs (ddof=1), seed-42 rows and actual seed counts.
  A missing SD is left blank, never replaced by zero.
- `results/summary.csv` mirrors the latest mode's summary;
  `results/summary_scope.json` records that mode and its actual seeds.
- `PAPER_REPORTED` in `src/exlightnet/paper.py`: quoted references only;
  never used for training, ranking, selection or evaluation.

D1/D2/M1/T1 are hard gates. P1 reports MI agreement without enforcing it. A
failed Q1 stops quantised downstream work while preserving its measured result.
Independent branches continue; `failures.json` lists every stopped branch.

Checkpoints are selected using validation macro-F1. Final test probability
passes are recorded exactly once per checkpoint, separately for FP32 and W8.
Reruns reuse persisted predictions and metrics. If a prediction pass was
interrupted, its receipt blocks silent reevaluation; inspect it and preserve
the original evidence before deliberately starting a new experiment directory.
SHAP/deletion use fixed final-analysis test examples and never select a model.
Do not change the CSV, recipe or class/feature mapping within an existing run.

Quantisation stores per-output-channel INT8 Conv1d/Linear weights, FP32 scales,
BN state and biases. Weights are dequantised once at load and inference computes
in FP32. Sizes include NPZ metadata/serialization. No INT8 compute acceleration
or extrapolated multi-core throughput is claimed.

## Notebook and archive

```bash
python scripts/make_notebook.py
EXLIGHTNET_RUN_MODE=full python scripts/execute_notebook.py
python scripts/package_deliverables.py
```

[notebooks/ExLightNet_reproduction.ipynb](notebooks/ExLightNet_reproduction.ipynb)
contains outputs. Restart-and-Run-All defaults to fast mode; cached test metrics
are reused. Set `RUN_MODE="full"` in its first cell, or set
`EXLIGHTNET_RUN_MODE=full`, to load previously generated full results. This mode
does not train or require the dataset. Notebook execution needs local Jupyter
kernel sockets. The execution script creates its temporary kernel specification
using the active Python interpreter.

`deliverables.zip` includes package, tests, scripts, executed notebook, results,
trained model artifacts, fitted preprocessing, report and requirements. It
excludes every `*.pkl` file, the dataset, cached feature arrays and virtual environment. The archive
has a SHA-256 file manifest; packaging verifies notebook outputs and CRCs.
Local tree checkpoints remain available for the final benchmark pass. They are
excluded only from the archive; an extracted archive can display cached tree
metrics but cannot re-benchmark or verify those absent checkpoints without
restoring the original local files. Benchmark-only never retrains a missing tree.

REPORT.md renders evidence as Markdown tables. The INT8 size tables distinguish
raw weights-only bytes from FP32 scales/rest, metadata, compressed NPZ members
and the complete artifact's actual on-disk size.
DeepExplainer checks additivity on two inputs before attempting the full set;
an unsupported graph uses the prescribed GradientExplainer fallback immediately.
Background size, random explanation sample and fallback sampling settings remain
the declared fast/full values, and measured additivity residuals are reported.

## Raspberry Pi

Copy `scripts/bench_pi.py` and the desired `model_int8.npz` to the Pi; only NumPy
is needed:

```bash
python bench_pi.py model_int8.npz
python bench_pi.py model_int8.npz --input standardized_flows.npy
```

Without an input file, latency uses a zero vector in standardized feature space.
Optional input arrays must already follow the artifact's saved feature order
and training z-scores. The script computes the same dequantised graph as PyTorch;
tests check logit parity. It reports measured mean/median/p95, RSS, hardware and
software. A run on this x86 host does not constitute a Pi measurement.

## Contributing and Git workflow

See [CONTRIBUTING.md](CONTRIBUTING.md) for Conventional Commits, branch names,
verification, pull requests and push commands. GitHub Actions runs the tests
that do not require the local dataset on pull requests and pushes to `main`.
The dataset, prepared feature-array cache and `*.pkl` tree checkpoints stay
local; the recorded metrics, small model artifacts and executed notebook remain
available in Git. Restoring tree checkpoints is necessary to re-benchmark them.
