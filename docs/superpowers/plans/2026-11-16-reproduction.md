# ExLightNet Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task by task.

**Goal:** Deliver a reproducible, honest CPU implementation, executed notebook, report and dataset-free archive.

**Architecture:** Focused package modules share train-only transformations and cached results. Gate evidence controls dependent phases. The notebook imports package functions.

**Tech Stack:** Python 3.11, PyTorch CPU, NumPy, pandas, SciPy, sklearn, XGBoost, SHAP, pytest, nbformat/nbconvert.

**Spec:** docs/superpowers/specs/2026-11-16-reproduction-design.md and the supplied authoritative brief.

## Global Constraints

CPU only; fixed split seed 42; training seeds 42..46 in full mode; num_workers=0;
no test-driven selection; no fabricated results; no dataset in archive. Hard gates
stop dependent branches. Commit after each phase only if Git is available (it is not).

## Review Focus

- Unseen categories, infinities and all-missing train features must transform without refitting.
- Restarts must reuse evidence without silently evaluating test checkpoints again.
- Alpha sweeps must never receive test arrays.
- Quantisation exports must retain BN buffers and reproduce torch inference in NumPy.
- Missing data, caches or full results must give clear diagnostics, not fabricated tables.

## Phase tasks (in authoritative order)

1. [x] Data: tests for label normalization, exact counts/sizes/support, missing-data failure; implement `load_data`, `validate_data`, `make_split`; fetch UCI 942 if needed.
2. [x] Preprocessing: no-leak/unseen-category tests; implement `FlowPreprocessor.fit/transform`, MI comparison, JSON/NPZ cache.
3. [x] Model: failing parameter and stage-shape tests; implement exact graph and verify M1.
4. [x] Training: focal CE-equivalence and sampler tests; implement alpha, deterministic sampler/Adam/early stopping/logging.
5. [x] Baselines: verify MLP counts and model shapes; implement BiLSTM, StdCNN, RF and XGBoost.
6. [x] Metrics/runner: tests for binary errors, multiclass AUC and evaluation-once guard; implement per-run persistence and mean/sample-std with seed-42 rows.
7. [x] Ablations: implement five independent variations, full-model reuse and paired seed-std comparisons.
8. [x] Quantisation: per-channel error and NumPy parity tests; export/load NPZ and evaluate Q1 without tuning.
9. [x] Efficiency: single-thread warmup/timing; implement standalone NumPy Pi script; hardware/version/RSS records.
10. [x] Explainability: SHAP output-layout and tiny linear additivity tests; implement fallback, global/local plots, deletion and quantisation stability.
11. [x] Optional leakage PROXY: whole-cluster split, train-refit transforms and three-seed runner, clearly labelled.
12. [x] Integration: generate/import-only notebook; execute fast notebook, run pytest, generate evidence-based REPORT/README, archive excluding data. Review before final verification.

For each phase: write the relevant failing tests, run them, implement, run the
suite, and record evidence in docs/progress.md. CLI: `python -m exlightnet.run_all
--mode fast|full`; full notebook only loads full caches. No Git-specific worktree
or commit operations can run because the supplied .git is an empty read-only mount.

Implementation checklist complete; final fast notebook and pytest verified. Full five-seed experiment execution and optional proxy execution remain unperformed; see REPORT.md.
