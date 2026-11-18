# Cached statistics and hosted supplementary studies Implementation Plan

> **For agentic workers:** Use focused parallel agents for independent package
> components under `superpowers:dispatching-parallel-agents`; the controller owns
> integration, experiment execution and final verification. Steps use checkboxes.

**Goal:** Add honest cache-only CPU statistics and a resumable, offline-capable
hosted GPU study pipeline without altering canonical evidence.

**Architecture:** Independent statistics and prepared-data modules supply CSVs
and verified arrays. A separate campaign runner reuses package model/loss/train
and scoring code. Reports, notebooks and aggregators consume persisted files.

**Tech Stack:** Python3.11; existing NumPy/SciPy/Pandas/sklearn/PyTorch/XGBoost;
stdlib zip/json/hashlib. No new dependencies.

**Spec:** `docs/STATS_GPU_BRIEF.txt` and
`docs/superpowers/specs/2026-11-17-statistics-and-hosted-studies-design.md`.

## Global Constraints

- Do not run the full campaign or any benchmark. Do not touch user processes.
- All original `results/`, existing `artifacts/`, caches and CSV are immutable.
- No local experimental training except centrally controlled two seeds x two
  epochs, CPU. Agents must not run training; synthetic legacy-test exemption is
  pending clarification.
- Preserve M1/T1, preprocessing train-only, frozen CPU alpha, validation-only
  selection, evaluation once per declared checkpoint, and recorded failures.
- No mixed campaigns; keep devices distinct subject to the pending tree/CUDA
  comparison ruling. CUDA is NOT EXECUTED locally.
- New outputs use new paths. Archive old report/notebook before regeneration.
- No commits, pushes, merges, publishing or unrelated changes.
- All shell commands start with `rtk`; use `.venv/bin/python` and one thread.

## Review Focus

- Corrupt/misaligned/missing prediction caches: reject false pairing; label
  confusion-cell fallback UNPAIRED.
- Finished or claimed evaluation interrupted at file boundaries: preserve
  receipts and reject repeat test inference; validate resume fingerprints.
- Exported arrays tampered or split42 differs: fail before training/import.
- Duplicate/conflicting result zips or mixed campaigns/devices: reject conflicts
  and never silently pool observations or count duplicate seeds.
- Zero paired variance, missing seeds/studies, or unstable baselines: honest
  undefined values/counts and PENDING; no superiority/causal overclaim.

### Task 1: Cached CPU statistics (independent agent)

**Files:** create `src/exlightnet/stats.py`, `scripts/make_stats.py`,
`tests/test_stats.py`; new files only under `artifacts/stats_cpu_v1/`.

**Interfaces:** `load_campaign_rows(results_path, manifest_path=None)` normalizes
copies and derives legacy cpu_v1; `paired_tests(rows, reference='exlightnet')`
returns a DataFrame; `write_cpu_statistics(root, output, replicates=2000,
random_seed=42)` writes tables and a statistics manifest and returns paths.
Expose pure helpers for paired summary/CI/Holm, class/flow macro-F1 and paired
bootstrap so the supplementary aggregator can reuse them.

- [x] Add failing SciPy agreement, zero variance, missing pairs, Holm,
  mixed-campaign, paired-flow alignment, sensitivity/support and dynamics tests.
- [x] Implement A1-A3; no inference/model loading/training. Required CSVs plus
  `correlations.csv`, `binary_caps.csv`, `flow_bootstrap.csv`,
  `paired_flow_bootstrap.csv`, `per_class_f1.csv`, `sensitivity.csv`.
- [x] Run only new statistics tests, then compute the actual canonical tables.
- [x] Return API/schema documentation, red/green evidence and output paths.

### Task 2: Prepared-data export and verified offline loader (independent agent)

**Files:** create `src/exlightnet/prepared.py`, `scripts/export_prepared.py`,
`tests/test_prepared.py`; extend `data.make_split` conservatively if needed.

**Interfaces:** `export_prepared(root, output, split_seeds)`; `load_prepared(path)`
returns `{arrays, feature_names, class_names, metadata}`. Arrays have old cache
keys `x_*`, `y2_*`, `y12_*`. Metadata includes `data_fingerprint`, `split_seed`,
`settings`, `payload_sha256`, `frozen_alpha` (per task from cpu_v1 selections).
Write `prepared/manifest.json` with file SHA/fingerprint/size and alpha source.
The loader must not import pandas/preprocessing/run_all; local export imports
them lazily. Preserve the old `digest(settings)` recipe fingerprint exactly.

- [x] Add failing corruption, dtype/shape/class coverage, fingerprint, default
  split preservation and train-only export tests; no model training.
- [x] Implement split42 export via read-only frozen caches and exact comparison.
- [x] Export seeds42--51 serially with original train-fitted preprocessing;
  check all classes in every partition and save validation evidence to a new
  path. Do not upload or package the NPZ data.
- [x] Run new tests and record export checks; return schema and evidence.

### Task 3: Device controls, resumable training, study runner (independent agent)

**Files:** create `device.py`, `campaign.py`, `scripts/kaggle_campaign.py`,
`tests/test_campaign.py`; extend `train.py` and lazy-load presentation-only
dependencies in `metrics.py` if offline imports require it. Do not edit
run_all/report/notebook/prepared/stats modules owned by other tasks.

**Interfaces:** `run_campaign(...)` and a pure `study_runs(study)` plan; thin CLI
implements the full brief's arguments plus explicit smoke controls (one model,
two seeds, two epochs). Shared loader contract is Task2. Each complete run JSON
has schema/version, status, campaign/study/arm/device, model/task/seeds/config,
data fingerprint, run fingerprint, libraries, warnings, history, best and last
metrics and prediction paths/hashes/receipts. Document final record schema for
the aggregator. Make sampler generation an exposed pure helper.

- [x] Add failing device/precision/default CPU compatibility, sampler frequency,
  deterministic run identity, study counts/B3 reuse, budget, safe zip and
  receipt/resume-state tests. Do not execute optimizer/training here.
- [x] Implement FP32 single-CUDA controls and epoch-boundary recovery, preserving
  canonical CPU config/fingerprints and sampling/drop-last invariants.
- [x] Implement all studies/probe/determinism; reuse B1 references, print timing
  assumptions/projections, package incrementally, no test-driven selection.
- [x] Provide an interruption hook for the controller's sole smoke experiment:
  pause after an atomic epoch checkpoint before starting another epoch.
- [x] Run only nontraining tests; return schema and explicit CUDA NOT TESTED.

### Task 4: Local supplementary aggregation and pre-declared report (controller)

**Files:** create `src/exlightnet/study_stats.py`,
`scripts/aggregate_gpu_v2.py`, `tests/test_study_stats.py`,
`docs/statistics_protocol.json`, `REPORT_stats_v2.md`.

- [x] Persist the verbatim a-priori interpretation rules before any smoke data.
- [x] Test mixed campaigns, duplicate/conflicting zips, data/prediction SHA and
  incomplete receipts, missing studies, zero variance and device separation.
- [x] Implement zip streaming (never load untrusted model pickles), expected
  fingerprints, per-study paired statistics/bootstraps/dynamics, B3/B4/B5
  comparisons and separately labelled device-equivalence view. Reuse stats.py.
- [x] Generate canonical `REPORT_stats_v2.md` with PENDING for studies not run;
  smoke results have a different campaign/report, never fill canonical studies.

### Task 5: Active presentation, status and hosted documentation (controller)

**Files:** update `report.py`, `notebook.py`, `run_all.py` ablation presentation,
notebook generation/execution scripts, packaging script, README; create
`docs/KAGGLE_RUNBOOK.md`, `notebooks/kaggle_runner.ipynb` and regression tests.

- [x] Replace active SD-overlap flags with paired columns, subject to the
  immutable-history decision. Read new tables instead of old ablation flags.
- [x] Add mode-aware text tests and a status derivation test accounting for the
  failed initial idle campaigns and successful recovery; do not edit raw status.
- [x] Render n=5 labels, per-class support/F1, tiny-class warnings, instability,
  zero variance, fixed split, descriptive stopping/cosine hypothesis and eager
  dispatch limitation from computed files. Keep original measured timings.
- [x] Add `--evidence-dir` for notebook execution/verification so old result
  logs/manifests remain immutable. Execute full caches only.
- [x] Write beginner runbook and thin offline notebook; preflight dependencies,
  upload prepared data/package only, probe first, priority B1/B3/B2/B4/B5,
  resumability/time budget and download/import instructions.

### Task 6: Controlled smoke, gates, evidence and delivery (controller)

- [x] Verify read-only canonical MLP CPU cache hit equals original fingerprint
  and macro-F1; explicitly distinguish this from an unauthorized retrain.
- [x] Run exactly one CPU smoke study: two seeds x two epochs. Interrupt at a
  checkpoint boundary, restart and confirm no duplicate completed run/test
  receipt; aggregate and zip the smoke campaign separately.
- [x] Run full pytest only after resolving the old synthetic-training tests'
  budget conflict; save new JUnit/logs in new verification paths.
- [x] Rebuild and execute the full notebook; verify all original 916 hashes.
- [x] Fresh whole-change review, fix substantive findings with regression tests.
- [x] Generate Gate A/B tables from persisted checks. Record any untestable or
  excepted gate honestly; CUDA/probe/real studies/A5 are NOT RUN locally.
- [x] Build requested source/test/script/doc component zips plus main delivery,
  checksum everything, exclude prepared arrays/PKL/data/credentials/locks,
  verify CRCs and hashes. No commit/push/merge.

## Recovery ledger

Started on `feat/cached-statistics-and-hosted-studies` from main6f76cb4.
The user's AGENTS.md is preserved byte-for-byte (untracked on the new branch).
916 immutable files are recorded; the existing stage lock was available.
No experiment, training, benchmark, or user-process action has been performed.
Flag scope, legacy synthetic unit-training exemption and cross-device tree
comparisons await user clarification; proceed with independent work.

Final execution closeout: Tasks1--6 implemented and verified. The only local real-data training was the recorded two-seed/two-epoch CPU smoke. GPU/probe/real studies and benchmarks are NOT RUN. User rulings supersede the historical pending questions above; see docs/progress.md and AGENTS.md. No commit/push/merge performed. Delivery integrity is checked by the package builder and final external audit.
