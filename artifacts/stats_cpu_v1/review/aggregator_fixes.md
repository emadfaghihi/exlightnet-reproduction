# Aggregator review fixes: R3 and R4

Scope: only `src/exlightnet/study_stats.py` and
`tests/test_study_stats.py` changed in this review response. No training,
inference, benchmark, process action, original-results write, smoke rewrite,
commit or subordinate agent was used. CUDA was NOT EXECUTED.

## R3: replay terminal stopping rules

The import validator now replays the exact validation-history rule used by
`fit_resumable`: strict `val > best` improvement resets stale epochs; ties
increment stale. The next epoch is forbidden once stale reaches patience and
the minimum epoch budget has completed. An imported completed run must end
at its first legal stopping epoch or its configured cap, whichever comes first.
An improvement recorded after that first stopping point cannot legitimize an
overrun. `fixed30` and the three-epoch probe must complete the full cap.
The patience-10/minimum-15 arm accepts the exact epoch-15 boundary when stale
has already reached patience and rejects both epoch-14 termination and an
unnecessary epoch-16 continuation. A separately labelled two-epoch smoke
campaign remains valid at its cap.

Synthetic scored-cache fixtures now use legal seven-epoch patience histories.
The correlation fixture uses lengths seven/eight/nine with best epochs
one/two/three, preserving its paired-seed bootstrap purpose. None of these
fixtures fits a model. A specific binary fixture verifies the sorted JSON
`attack`/`normal` mapping independently of insertion order.

## R4: actual execution-environment identity

The aggregator uses the runner's shared pure helper
`device.environment_identity(device, libraries)`. CPU identity includes device,
NumPy, PyTorch, scikit-learn and XGBoost versions. CUDA identity additionally
includes GPU model, CUDA and cuDNN versions. The reader rejects any identity
change within a task/model seed vector across all imported studies and arms;
this is stricter than checking each arm independently. Missing CUDA hardware
or runtime identity also fails. CPU tree identity ignores unused GPU fields,
while retaining its relevant software identity.

The authorized CUDA-neural/CPU-tree exception remains limited to `gpu_v2`,
with its exact required device label and distinct reference/comparison devices.
Mixed campaigns still fail without an override.

New provenance includes:

- Per-run `environment_id`, canonical `execution_environment` JSON, and explicit
  `gpu_name`, `torch`, `cuda`, `cudnn`, `numpy`, `scikit-learn`, `xgboost` columns.
- Per-model summaries and other model-indexed tables carry the execution
  identity; paired tests and paired-flow bootstrap tables carry
  `reference_environment_id`, `comparison_environment_id`,
  `reference_environment`, and `comparison_environment`.
- `aggregation_manifest.json` records the identity-to-fields mapping as
  `execution_environments` and documents `environment_identity_fields`.

## RED / GREEN evidence

New stopping/runtime/provenance regressions were run before implementation:
exit 1, **13 failed, 33 passed in 3.10 seconds**. These failures covered
premature B1/B2/B4/B5 histories, first-stop overrun, incomplete probe,
different GPU models/runtime versions, CPU-tree relevant-library mismatch,
and missing environment columns.

After the initial fixes: **46 passed in 2.73 seconds**, exit 0.
The additional missing-GPU-identity regression then failed as intended:
**1 failed, 46 passed in 2.73 seconds**, exit 1. It passed after CUDA identity
completeness validation was added.

Final focused verification:

```text
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -m pytest tests/test_stats.py tests/test_study_stats.py -q --junitxml=artifacts/stats_cpu_v1/review/aggregator_fixes_pytest.xml
70 passed in 3.14s
exit status: 0
```

`rtk git diff --check` returned exit 0. The 70 cases comprise 23 Stage A
statistics cases and 47 supplementary aggregator cases. No full-suite success
is claimed here; the controller owns full-suite and cache-only real-smoke ZIP
import validation. Existing canonical tables, report skeleton and real smoke
were not regenerated during this review response.

## Remaining limits

CUDA hardware execution is not validated locally. Imported identity fields
are provenance supplied by the hosted runner and cryptographically bound into
run fingerprints; they are not remote hardware attestation. The prepared
manifest remains the local trust anchor. The canonical CPU campaign lacks
GPU-runtime identity because it ran only on CPU; the dedicated cross-campaign
equivalence view stays descriptive, with differences/CIs and no p-values or
pass/fail claims. No new statistical or scientific requirement was waived.
