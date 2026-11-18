# Cached CPU statistics and hosted supplementary studies

The full implementation specification is [docs/STATS_GPU_BRIEF.txt](../../STATS_GPU_BRIEF.txt).
It is the user's supplied design, not a request for another reproduction run.

## Intent and fixed approach

Keep the canonical `cpu_v1` evidence byte-for-byte intact. Compute statistical
tables from its cached rows and predictions, and present those tables in the
report and executed notebook. Build a separate, resumable `gpu_v2` accuracy
runner for the user to execute on Kaggle/Colab. No local GPU study or benchmark
is authorized. Local experimental training is limited to one two-seed,
two-epoch CPU smoke experiment.

The brief fixes the approach: standalone cached statistics, locally exported
train-fitted arrays, shared package training/scoring, and independent campaign
roots. In-place re-aggregation is excluded by the stricter immutability rule.

## Evidence boundary

`artifacts/stats_cpu_v1/immutable_before.json` records the hashes of every
pre-existing result/artifact and the raw CSV. Original report/notebook/delivery
versions were copied to a dated folder under `archive/cpu_v1_before_stats/`.
Old result rows, summary and ablation JSON, receipts, checkpoints, benchmarks,
prepared cache and verification records must not be edited. New verification
logs and notebook execution records go under the new statistics directory.

Legacy records have no campaign tag. The statistics loader derives `cpu_v1`
from the canonical full manifest and records the manifest hash; it adds tags
only to new normalized tables, never to old rows.

## Components and interfaces

- `stats.py`: campaign validation; matched-seed t/CI/dz/Holm; training dynamics;
  seed/flow bootstrap; per-class support and sensitivity tables. It never fits
  a model or evaluates a checkpoint.
- `prepared.py`: an offline-safe NPZ loader/content verifier, plus a locally
  imported export path using the existing `FlowPreprocessor`. Canonical array
  keys are `x_{train,val,test}`, `y2_*`, `y12_*`; metadata, feature/class names
  and split indices accompany them. Keep the historical recipe fingerprint
  and add a separate payload digest. Split42 must match cached arrays exactly.
- `device.py`, additions in `train.py`, and `campaign.py`: device configuration,
  shared optimization/sampling, atomic epoch recovery, study plans and durable
  run records. The historical CPU default/recipe remains compatible.
- `scripts/kaggle_campaign.py`: thin CLI; package code owns scientific logic.
- `scripts/aggregate_gpu_v2.py`: local, cache-only zip import/verification and
  per-study analysis. Never concatenate campaigns. Missing studies remain
  `PENDING: study not run`.
- `report.py`, `notebook.py` and notebook scripts: render generated statistical
  files; preserve historical outputs; no writes into old result paths.

## Statistical choices fixed before new study data

Use all matched seeds, sample SD (`ddof=1`), two-sided paired t tests, a 95% t
interval on paired differences, signed Cohen's dz and Holm correction within
each task. Degenerate differences are explicit. No Wilcoxon with fewer than
six pairs. Zero model variance is flagged, not mistaken for zero paired
difference variance.

The 2,000-replicate bootstrap first samples one matched training seed uniformly
and then resamples test flows. These intervals describe a randomly selected
training run plus finite-test-flow uncertainty; they are not confidence
intervals for the five-seed mean. Use cached flow predictions and paired joint
cell counts for exact, efficient paired resampling. If predictions are absent,
use confusion-cell multinomial resampling and label it UNPAIRED.

Post-hoc sensitivity excludes low-support classes from the macro average;
their flows still contribute false positives to retained classes. The mask is
fixed from the original test supports, not recomputed per bootstrap draw.

The pooled convolution-family dynamics diagnostic contains ExLightNet,
no-SE, same-width standard convolution and Std-CNN. Standardize epochs and
macro-F1 separately within each model before pooling; report Spearman as
descriptive, never causal.

## Hosted study contracts

Resolve a single device, default CPU; single CUDA GPU only, FP32, no AMP/TF32.
Record device, GPU/library versions, deterministic settings/warnings,
campaign/study/arm, train/split seeds, full config and data fingerprint in every
run identity. Retain all training tensors on the chosen device. GPU sampling
uses the original per-flow weights and epoch size, replacement and final tail
batch. Keep canonical CPU behavior separate and unchanged.

Save recovery state atomically after each epoch, including current/best model,
optimizer/scheduler, RNG/sampler state and history. Save independent final
evaluation receipts for predeclared best-validation and last-epoch outputs.
Never reset a claimed receipt. Persist int8 class predictions and true labels
for resampling; save logits additionally for determinism diagnostics. Final
model states are retained only for ExLightNet; unfinished recovery states are
allowed until completion. Completed records are immutable and fingerprinted.

B3 paper-rule results reference B1 runs; do not train the reference arm again.
Time budgets stop new launches; interruptions resume at validated checkpoint
boundaries. Zip complete results incrementally and at exit. Probe projections
must identify assumptions (ExLightNet epoch cost does not directly measure
other architectures or tree fits). No SHAP, size, W8 or latency measurements
run on the hosted runner.

## Specification conflicts resolved by the user

The user explicitly resolved all three on 2026-11-17:

1. R1 preserves old ablation JSON while A1/Gate A demand deleting legacy flags
   everywhere. Authorized scope: active code/new tables/report/notebook, with
   immutable historical JSON and archives excluded from that scan.
2. R8 permits only the smoke training, but the required pre-existing full
   pytest suite contains small synthetic training tests. Those tests are
   exempt, with their counts/duration reported separately. Never train on
   real data, execute benchmarks or mutate raw evidence within unit tests.
3. R5 prohibits device mixing, while B1 intentionally trains neural models on
   CUDA and trees on CPU. The user authorized those pairs only inside gpu_v2,
   with explicit mixed-device labels and device columns. Campaigns never pool.
   The cross-campaign descriptive difference/CI view has no p/pass-fail.

A5 is NOT RUN because R4 forbids benchmark reruns. The canonical CPU MLP
compatibility gate uses a read-only validated cache hit; no retraining is
authorized to establish numerical reproduction of that historical run.

## Workspace and publication

Use local branch `feat/cached-statistics-and-hosted-studies` from `main` in the
existing checkout. Preserve the user's newer AGENTS.md bytes separately from
the branch baseline. No commit, push, merge or publication is authorized.
Prepared arrays and result zips stay ignored data artifacts. All delivered
files have checksums; exclude PKL, raw data, prepared arrays, credentials,
virtual environments and process locks.
