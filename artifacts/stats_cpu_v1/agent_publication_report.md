# Publication and status implementation report

Recorded 2026-11-17T12:43:51.928919+00:00.

## Scope and source changes

- src/exlightnet/publication.py adds read-only, manifest-verified CPU statistics
  loading; shared table descriptions/columns; measured rare-class support
  warnings; limitations; effective execution status; stale-caption validation
  covering notebook source and displayed outputs; and an active-scope obsolete
  identifier scanner. The scanner covers src, scripts, tests, REPORT, notebooks
  and new statistics outputs, while results, archive and instruction/brief docs
  remain outside its scope.
- report.py consumes generated CSV tables and never reads historical ablation
  JSON. It renders paired t/CI/dz/Holm, single-random-run/flow bootstrap estimands
  and pairing, class F1 mean/SD with support, seed42, post-hoc sensitivity,
  dynamics/correlations/caps, instability and zero variance. Historical reduced
  evidence has an explicit Fast demonstration heading. Full captions describe
  the cached reproduction. Limitations explain fixed split, n=5, tiny supports,
  descriptive stopping/cosine hypothesis linked to Stage B, std_cnn instability,
  XGBoost zero observed variance, and eager dispatch overhead. Dispatch fraction
  is NOT MEASURED; exported-model benchmarking is NOT RUN.
- build_report(root, output=None, statistics_dir=None, evidence_dir=None) supports
  temporary review targets. New verification is preferred, with old verification
  explicitly labelled historical. The original candidate-accounting wording is
  preserved and the obsolete current-Git limitation was corrected.
- notebook.py uses cached evidence for both scopes and never starts a campaign.
  New helpers show_statistics, show_execution_status, show_limitations and
  show_fast_demonstration share publication definitions. Ablations use the new
  paired CSV; support warnings accompany macro-F1 comparisons.
- run_all.ablation_comparison derives task-wide paired/Holm results from copied
  rows, then selects ablations; no old SD-overlap fields are emitted. Future
  reduced runs are tagged fast_demo separately from cpu_v1. The canonical
  run() rejects device values other than cpu before locks/output; CLI --device
  accepts cpu only. Supplementary devices belong to the independent runner.
- make_notebook.py is import-safe, defaults its generated notebook to full cached
  analysis, has --root/--output, clears generated outputs, validates captions,
  and contains no data preparation or live-training calls.
- execute_notebook.py is import-safe and requires full caches. --evidence-dir
  writes metadata/history only to a fresh directory, refuses historical results
  paths and existing invocation metadata, and preserves failure evidence. Its
  default is a dated path below artifacts/stats_cpu_v1/verification. No metadata
  is written to historical results by the new script. Controller owns actual
  notebook generation/execution and final packaging.

## Verification actually run

Latest focused command (exit 0):

~~~sh
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/exlightnet-mpl .venv/bin/python -m pytest -q tests/test_publication.py tests/test_notebook.py tests/test_report_markdown.py tests/test_execution_modes.py
~~~

Result: 22 passed, 14 warnings, 4.66 seconds. Warnings were existing Matplotlib/
Pyparsing deprecations. The full suite was NOT RUN by this agent. Earlier
publication-only compatibility checks passed 13 tests (2.67 seconds), then
16 tests (2.65 seconds); neither run trained a model.

The first red publication run had 10 failures for missing helpers, old ablation
reads/output, absent output/device parameters and script top-level side effects.
An additional red test exposed an incorrect active-failure count when only the
current benchmark manifest had a failure; status now counts deduplicated active
records across the current failure/manifest sources.

Tests cover mode-aware stale text in captions and rendered outputs, historical
scope exclusions, recovery status and missing recovery evidence, present failures,
paired ablations without mutating input rows, canonical CUDA refusal, report/
notebook old-ablation read prohibition, generated-table checksum and campaign
rejection, n=5 labels, support/dynamics/bootstrap display, safe cached notebooks,
import-safe scripts and fresh-directory execution records. Report and notebook
previews were generated only in pytest temporary directories.

## Synthetic tests versus real smoke

Exactly 1 synthetic fit occurred in this agent's
focused verification: tests/test_execution_modes.py::test_skip_benchmark_suppresses_all_timing_calls. This is the existing timing-suppression
regression, using MLP seed 42, 4 training,
4 validation and 4 test rows,
40 features, and 1 epoch. The
training.json timer is 0.00757394999982352601 seconds. This timer
is model-training time, not total pytest case time. Per-case pytest wall time is
UNAVAILABLE because durations were not requested; the suite wall time was
4.66 seconds. No rerun was performed to obtain timings.

The actual training.json and epochs.csv were preserved byte-for-byte from pytest
/tmp output with source hashes under
[verification/publication_synthetic_training](verification/publication_synthetic_training/manifest.json).
Timing was a failure stub and skip_benchmark=True; no benchmark ran. Prepared-data
and earlier publication-focused runs had zero model fits. This evidence is
separate from the controller's sole real-data smoke; no smoke raw file was read
or changed by this task. There was no real-data training or test inference here.

## Historical status and immutability

Current derived effective status: complete. Raw pipeline
status remains complete_with_stopped_branches; active failure count is
0, archived CPU-idle failure count is
2, and the later benchmark manifest records
95 measured models. These values are
read from persisted manifests/recovery receipts, not patched into old JSON.

All 916 baseline immutable file SHA256s were checked after
this work: 0 changed. Current notebook SHA256 still equals the
archived original: 18b395f832d14cebc7de487a82b46326d19c685e21ab6214619088feaaec2ada. The active source scan has zero obsolete
identifiers. The restored old notebook still has 4 such identifiers;
the controller's authorized regeneration must remove those displayed outputs.
The report and current notebook were not deliberately regenerated by this agent.

## Test-isolation incident and recovery

The initial red test invoked the old notebook generator with --output, assuming
it supported that option. The old script ignored arguments and temporarily
replaced the derived published notebook with an output-cleared version. Importing
the old executor also attempted kernel creation, which the sandbox denied at
socket creation before any cell executed. No model inference/training or old
result/verification write occurred in that attempt. The notebook was immediately
restored byte-for-byte from the pre-existing archive and matched its archived
SHA256 18b395f832d14cebc7de487a82b46326d19c685e21ab6214619088feaaec2ada. Generator/executor tests now operate on script copies
in isolated temporary roots, and both production scripts have __main__ guards.
No subsequent notebook cell execution was attempted by this agent.

## Remaining controller work and limits

Regenerate the current analysis notebook, execute EXLIGHTNET_RUN_MODE=full with
an unused --evidence-dir, regenerate REPORT from the resulting verification,
run the combined suite with its new verification logs, scan active outputs,
verify all immutable hashes again, and package using the new execution record.
Existing metadata remains historical; packaging must choose the new record.
CUDA and exported/frozen-model latency were NOT EXECUTED. No commits/pushes,
benchmark jobs or unrelated process control occurred.

Source SHA256s at handoff:

~~~json
{
  "src/exlightnet/publication.py": "cc66c7f0533608677da871a727dcf03f7c236c3de88edcd5f18ef8684c57ebbb",
  "src/exlightnet/report.py": "cf6cbb23bdca47ac70906a0cfcdcec51d4d56cd0b7d2f46c52cd7daa3410d853",
  "src/exlightnet/notebook.py": "597b8bb19eea8c52d7eeee2a35b4cc01347652cead4dd7cfa12e33cb1e9999a8",
  "src/exlightnet/run_all.py": "40c9733c43b6466f2d6a3e554c663f104def3ba4c9a3822c1cdbc28c529cffa9",
  "scripts/make_notebook.py": "5577504b58a8a9692ffad1c9119cd99405e4be044d008787bf740f5dda073b4b",
  "scripts/execute_notebook.py": "c9144c94a4870e84218ca44a202329d09ee078ea63daee3b14917745cbf5f2ef",
  "tests/test_publication.py": "890b6c3583fb983f61fe79e654ca4ae47b5277121776d786a542599a8865184c"
}
~~~
