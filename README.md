# ExLightNet reproduction

An independent, evidence-based CPU reproduction of the supplied *ExLightNet* paper. The
authoritative brief is preserved in [docs/BRIEF.txt](docs/BRIEF.txt). See
[REPORT.md](REPORT.md) for measured gates, execution status, assumptions and
limitations. Fast results are explicitly separate from full five-seed results.
The included full campaign contains 95 model results across seeds 42--46 and
95 isolated single-thread benchmarks. The delivered notebook was executed in
full mode from these caches. Current notebook/statistics verification is stored
under `artifacts/stats_cpu_v1/verification/`; the original verification records
remain historical evidence.

## Cached statistics and supplementary GPU studies

The canonical `cpu_v1` experiment is complete. Do not rerun training or timing
to refresh its presentation. The statistical follow-up reads its existing
probabilities, metrics and epoch logs:

```bash
python scripts/make_stats.py --help
python scripts/aggregate_gpu_v2.py --prepared-dir prepared \
  --output artifacts/stats_gpu_v2 --report REPORT_stats_v2.md
```

`artifacts/stats_cpu_v1/` contains paired t tests, confidence intervals, Holm
adjustment, training dynamics, flow bootstraps and post-hoc sensitivity tables.
Tables with five seeds are labelled **n=5, indicative only**. The original
ablation JSON is preserved for audit and is never read by current reports or
notebooks. New tables replace its obsolete SD-overlap interpretation. Tiny
class supports make multiclass macro-F1 noisy; training seeds do not estimate
split uncertainty. Bootstrap tables distinguish paired flows and describe a
randomly selected training run plus flow uncertainty, not the seed-mean CI.

For user-operated supplementary accuracy studies see
[docs/KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md) and
[notebooks/kaggle_runner.ipynb](notebooks/kaggle_runner.ipynb). The runner uses
separate campaign/device identities, frozen CPU alpha, prepared arrays only,
and resumable epoch checkpoints. CUDA has **NOT RUN locally**. The local smoke
used a separate `gpu_v2_smoke` campaign and cannot populate the canonical
[REPORT_stats_v2.md](REPORT_stats_v2.md), whose unexecuted studies stay PENDING.

No benchmark is part of this follow-up. CPU timing, size and W8 evidence remain
the canonical measured artifacts. Within `gpu_v2` only, CUDA-neural/CPU-tree
accuracy comparisons have explicit device columns and a mixed-device label.
Different campaigns are rejected by statistical aggregators. The CPU/GPU
device comparison is separate, descriptive, and has no p-value or pass/fail.

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

For the dated hosted handoff, refresh all three requested files together using
the recorded validation statements:

```bash
python scripts/package_deliverables.py \
  --evidence-dir artifacts/predata_20261118/verification \
  --hosted-verification artifacts/predata_20261118/verification/hosted_checks.json
```

This additionally writes `delivery/stats_gpu_v2/hosted_handoff.zip` with
`notebooks/kaggle_runner.ipynb`, `prepared/manifest.json`, `hosted_source.zip`
and individual SHA256 / TESTED / NOT TESTED statements. Plain packaging refreshes
the main/component archives only; it does not refresh an existing handoff ZIP.
The hosted runner notebook has never been executed here. Syntax and command
wiring checks do not establish Kaggle or CUDA execution.

[notebooks/ExLightNet_reproduction.ipynb](notebooks/ExLightNet_reproduction.ipynb)
contains outputs. Restart-and-Run-All defaults to full cached analysis and
performs no training or test inference. Existing demonstration results appear
in an explicitly separate historical section. Notebook execution does not
require the dataset. Use a new `--evidence-dir` for each execution; old metadata
cannot be replaced. Notebook execution needs local Jupyter
kernel sockets. The execution script creates its temporary kernel specification
using the active Python interpreter.

`deliverables.zip` includes package, tests, scripts, executed notebook, results,
trained model artifacts, fitted preprocessing, report and requirements. It
excludes every `*.pkl` file, the dataset, cached feature arrays and virtual environment. The archive
has a SHA-256 file manifest; packaging verifies notebook outputs and CRCs.
Separate source, tests, scripts and docs ZIPs, an offline hosted-source bundle,
and `SHA256SUMS` are produced under `delivery/stats_gpu_v2/`. Prepared NPZ data,
temporary recovery optimizer checkpoints, result ZIPs, credentials and process
locks are excluded. `prepared/manifest.json` is included for provenance; the
prepared arrays themselves are supplied separately by the user.
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

The dated stratified final analysis is separate from those original explanation
files: `artifacts/final_analysis/stratified_shap_20261118/analysis.json` binds its
inputs, method, warnings and twelve output checksums. It includes per-class
top-ten rankings, equally weighted class-global rankings, twenty random deletion
orders per class, k=10 comparisons and the earlier random-sample composition.
Reports/notebooks read these files; they do not repeat SHAP. Its additivity
errors are global summaries over all flow/output-logit pairs; separate logit
summaries were not retained. It is approximate final analysis, never selection.

The effective stopping rule in `docs/statistics_protocol.json` is a dated
amendment, alongside the preserved original declaration. Fixed30 LAST versus
paper BEST-validation is primary; fixed30 BEST is secondary. Both use the
predeclared paired uncertainty rules, disagreement is inconclusive and no
causality is claimed. All real supplementary studies remain pending here.

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
