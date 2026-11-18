# Agent instructions for ExLightNet

These instructions apply to the entire repository. Read them at the start of
each session and before changing experiments or recorded evidence.

## Purpose and sources of truth

Build and maintain an independent, honest reproduction of ExLightNet.
Scientific correctness, reproducibility and traceable evidence take priority
over matching the paper's numbers. Keep the package, CLI, tests, executed
notebook, report and deliverables consistent.

- `docs/BRIEF.txt` defines the reproduction requirements and resolves paper
  ambiguities. Read its relevant phases before changing scientific behavior.
  Dated amendments appended to it, and the "Amendments" section below, apply.
- `paper/exlightnet.pdf` is the original paper. Read the relevant section when
  interpreting an equation, architecture or quoted result.
- `README.md` documents installation, supported commands and artifact layout.
- `CONTRIBUTING.md` defines branches, Conventional Commits, PRs and publication.
- `REPORT.md` is a generated account of results, assumptions and limitations.
- `docs/progress.md` and `docs/superpowers/plans/` preserve work history.
- Actual manifests, evaluation receipts and measured files under `results/`
  are the evidence for execution claims. Verify them instead of relying only
  on a previous chat, this file or a historical progress entry.

Follow current user instructions within the applicable system/developer rules.
If the user requests a scientific deviation, make it explicit and preserve the
original reproduction. Do not silently reinterpret a requirement.

If this file, `docs/BRIEF.txt` and the current request conflict (for example
two incompatible budgets or constraints), stop that part, state the conflict
and the interpretation you chose in your reply and in `docs/progress.md`. Never
resolve a conflict silently.

## Session startup and recovery

1. Inspect the current branch, working tree and relevant files. Preserve the
   user's edits. Continue an existing task branch after an interruption.
2. Read the task's documentation and implementation before proposing changes.
   Search with `rg` or `rg --files`; avoid dumping large artifacts into context.
3. Inspect the relevant manifests, receipts and logs before starting a job.
   A missing interactive session does not prove a detached process has exited.
4. After a reboot, verify actual process state and locks. Never infer that an
   old PID still identifies the same process. Resume from validated caches.
5. Record substantive experiment changes, recovery decisions, commands,
   validation and outstanding work in `docs/progress.md` or the relevant plan.
   Distinguish historical limitations from the current environment: Git is now
   usable, even though early reproduction entries describe an unavailable Git.

An audited full-mode campaign (`cpu_v1`, seeds 42--46) already exists. Its
counts, dates and status belong in `docs/progress.md`; verify them in
`results/full/cache_audit.json`, `results/full/benchmark_audit.json`,
`results/full/benchmark_manifest.json` and `results/notebook_execution.json`
before relying on them. Completed experiments must not restart just because a
new chat begins.

## Campaigns, devices and amendments

- `cpu_v1` is the canonical, faithful reproduction: CPU-only, split seed 42,
  seeds 42--46. Its raw evidence is immutable (see Scientific rules).
- Any later supplementary campaign (for example `gpu_v2`: more seeds, other
  split seeds, stopping-rule or learning-rate sensitivity) is a separate
  campaign. It never replaces or edits `cpu_v1`, uses its own experiment root,
  caches and provenance, and tags every run record with campaign, study, arm,
  device, library versions and data fingerprint.
- **Amendment A1 (user-authorized, 2026-11-17):** supplementary accuracy and
  statistics studies may train on a single CUDA GPU in FP32 (no AMP, TF32
  disabled, deterministic flags with warnings recorded). This overrides
  "CPU-only" for those campaigns only. It does not apply to `cpu_v1`, latency,
  serialized size, INT8 or memory measurements.
- Never pool results from different campaigns, or multiple devices into one
  model's seed vector. The aggregator must refuse mixed campaigns with a hard
  error and no override. Within `gpu_v2` only, paired CUDA-neural/CPU-tree
  comparisons are authorized: label them `mixed-device within gpu_v2; trees
  are device-independent` and show each family's device in separate columns.
  The `cpu_v1`/`gpu_v2` device-equivalence view is descriptive and side by side:
  show differences and confidence intervals, never p-values or pass/fail.
  GPU runs are not bit-reproducible against CPU runs; do not claim otherwise.
  State clearly when a CUDA path was not executed on a GPU.
- Latency, size, INT8 and memory are measured only on the local CPU with the
  isolated benchmark procedure. For unchanged architectures they do not depend
  on trained weights, so do not re-benchmark merely because new weights exist.
- Supplementary runners on hosted notebooks (Kaggle, Colab) consume only
  prepared arrays exported locally with their fingerprints. Never upload the raw
  dataset, credentials or process locks. Result zips and prepared arrays are
  data artifacts and are not committed.
- Supplementary runs reuse the frozen alpha from `cpu_v1` (per task) unless the
  study is explicitly about alpha. Arms that change `max_epochs` or the stopping
  rule keep alpha frozen.
- Evaluation-once applies per checkpoint. If a study declares in advance that it
  reports both the best-validation checkpoint and the last-epoch checkpoint,
  each is evaluated once, with its own receipt. Which one is headlined is fixed
  before results exist and never depends on test performance.

## Scientific rules

- Never fabricate metrics, timings, successful gates or execution evidence.
  Paper values belong only in `src/exlightnet/paper.py`'s `PAPER_REPORTED` and
  clearly labelled comparisons; they must not drive training or selection.
- Fit medians, category vocabularies, constant pruning, MI ranking and z-scores
  on the training split only. Preserve split indices, class names, feature
  order and data fingerprints with the artifacts.
- Use validation only for epochs, alpha, learning rate and any other model
  selection. Alpha selection uses seed 42, separately per task, then freezes for
  every seed.
- Final test probabilities are computed once per trained checkpoint, and once
  per derived W8 artifact. Reuse cached predictions for metrics and plots.
  SHAP/deletion are declared final analyses, never selection tools.
- Never delete, reset or edit an evaluation receipt to force another test
  pass. A claimed/incomplete receipt requires investigation; stop that branch
  rather than silently repeating inference.
- Raw evidence is immutable: per-run result rows and caches, evaluation
  receipts, checkpoints, manifests, benchmark files and data fingerprints.
  Never overwrite the recorded campaign with a changed dataset, architecture,
  dependency recipe or training configuration. Use a distinct experiment root
  with its own compatible prepared-data/alpha caches and provenance.
- Derived aggregates (`summary.csv`, `ablations.json`, `REPORT.md`, the
  notebook, aggregate manifests) may be regenerated with
  `--aggregate-only --skip-benchmark`. Before regenerating, copy the previous
  versions to a dated archive folder and record their sha256, so every change
  stays auditable.
- Report all requested seeds: mean, sample SD (`ddof=1`), observation count and
  seed 42 separately. Never headline a best seed. Leave missing values and
  unavailable SDs missing; keep `fast` and `full` results separate.
- D1, D2, M1 and T1 are hard gates. P1 is report-only. Failed Q1 stops the
  artifact's downstream branch while preserving evidence. Record failures and
  continue independent work; never weaken a gate to obtain a pass.
- Preserve exact M1 parameter counts: 38,328 for K=12, 35,438 for K=2 and
  23,052 for K=12 without SE. Preserve T1's per-sample tempered sampler; any
  GPU re-implementation must pass an equivalence test against it. Investigate
  mismatches instead of adjusting assertions to hide them.
- `cpu_v1` and all benchmarks stay CPU-only with deterministic settings where
  feasible, `num_workers=0`, pinned dependencies and explicit random seeds.
  Supplementary campaigns follow Amendment A1.
- W8 stores INT8 weights with FP32 scales/rest and uses dequantized FP32
  computation. Do not claim integer-compute acceleration. Distinguish raw
  weights-only bytes from the actual full NPZ file size and its overhead.
- Report SHAP's actual method, fallback reason, sample sizes and additivity
  residuals. Do not describe GradientExplainer as exact. Label leakage stress
  tests as a clustering PROXY, never as campaign-disjoint validation.

## Statistics and reporting rules

- Tables with n=5 seeds are labelled "n=5, indicative only". Compare models with
  paired tests on matched seeds: mean difference, two-sided paired t p-value,
  95% t-interval, effect size, and Holm adjustment within a task family. Do not
  use Wilcoxon below n=6. Do not use SD-overlap flags such as `exceeds_*_std`.
- Historical CPU JSON and archives retain their original obsolete SD-overlap
  fields. Never regenerate those files to remove the fields. Reports and
  notebooks must not read or display the old ablation JSON. Check removal only
  in active source/presentation and newly generated outputs, with immutable
  historical paths explicitly excluded.
- Declare interpretation rules before looking at new data and apply them
  mechanically. Default: "difference supported" only if the 95% CI excludes 0
  and Holm p < 0.05; "practically equivalent" only if the whole CI lies within
  +/-0.01 macro-F1 (multiclass) or +/-0.002 (binary); otherwise "inconclusive".
- Show test support next to every per-class metric. Classes with a handful of
  test flows (for example `nmap_fin_scan`, `metasploit_brute_force_ssh`) make
  macro-F1 noisy; say so wherever macro-F1 is compared.
- Post-hoc analyses (for example macro-F1 without tiny classes) are labelled
  "post-hoc sensitivity" and never replace the pre-declared headline metric.
- Correlations (for example epochs run versus macro-F1) are descriptive. Do not
  state causality without an intervention arm.
- The dated 2026-11-18 stopping-hypothesis amendment is stricter: fixed30 LAST
  versus paper BEST-validation is primary; fixed30 BEST is secondary. Apply the
  predeclared paired mean-CI and SD-ratio/bootstrap criteria, report disagreement
  as inconclusive, and never make a causal claim for this hypothesis.
- B1/B2/B3 execute seed-major without changing run identities. Test every plan
  prefix: at most its final seed can have an incomplete matched block. Exclude
  incomplete predeclared study seed blocks from statistical comparisons and
  explicitly report `incomplete; excluded` in completeness tables. Never fill
  missing seeds through interpolation or substitution.
- State whether a bootstrap is paired (same flows) or unpaired. Flag models with
  zero seed variance (for example XGBoost under the paper configuration).
- Flag unstable baselines (very early best epoch, very large SD). Do not use an
  unstable baseline as evidence that ExLightNet is better.
- PyTorch eager batch-1 latency includes framework dispatch overhead. Never
  present it as pure compute cost or hardware-independent.

## Experiment execution

Expensive experiment commands are for requested experiments or necessary
recovery, not routine code review, documentation edits or session startup.

For a new authorized full campaign, use the existing coordinator:

```bash
.venv/bin/python -m exlightnet.run_all --mode full --alpha-only --skip-benchmark
.venv/bin/python scripts/run_parallel.py --reuse-alpha
```

If compatible frozen alpha selections already exist, reuse them. Do not repeat
the sweep. The coordinator launches seeds 42--46 as five parallel processes,
then one cache-only aggregation, then one isolated benchmark-only process.

- Each worker uses one library thread. Preserve the pre-import OMP, OpenBLAS,
  MKL and NumExpr limits, `n_jobs=1` for trees and `num_workers=0`.
- Keep `--skip-benchmark` during selection, training and aggregation. Workers
  write seed-local artifacts; seed 42 alone owns the shared SHAP cache.
- `--aggregate-only --skip-benchmark` must reuse all seeds' final caches and
  must not train, export fresh models, repeat test inference or start uncached
  SHAP/proxy analysis. Missing evidence is a stopped branch.
- `--benchmark-only` loads persisted models and training inputs, uses one
  process/thread, batch one, 100 warmups and 1,000 timed calls per model.
  Require three consecutive one-second windows at <=5% total CPU busy before
  each model. Respect locks; never overlap it with training or explanations.
- Preserve failed/previous benchmark campaigns. Do not carry stale timings
  into a failed refresh or weaken the idle threshold to finish a run.
- Check RAM before five workers; the coordinator defaults to 3,072 MiB
  available. Do not silently reduce concurrency, batch size or the recipe.
  If resources block progress, explain the actual constraint to the user.
- For long authorized runs, use a durable supervisor with logs, PIDs,
  timestamps and exit statuses. Never launch a duplicate pipeline on resume.
- Do not stop, pause or renice unrelated applications without current
  authorization. If apps are temporarily paused, arrange automatic restoration
  and verify it. Past permission to quiet the desktop was specific to that
  benchmark session and does not carry over.
- Hosted GPU runners must be resumable (assume the session can be killed at any
  moment), write one record per run, stop launching new runs at a time budget,
  and package results incrementally. Run a short timing probe and print the
  projected cost before any long study starts.
- Hosted `--only all/neural/trees` selection changes scheduling only. A trees-only
  session needs neither CUDA nor a neural probe and retains every RF/XGBoost
  seed/recipe. Its records can merge with neural records within gpu_v2 under the
  explicit mixed-device policy; no model seed vector pools different devices.

## Code organization and change discipline

- `src/exlightnet/data.py`, `preprocess.py`: real data, split and fitted caches.
- `model.py`, `losses.py`, `train.py`, `baselines.py`: graphs and model recipes.
- `metrics.py`: scoring, evaluation-once receipts and seed summaries.
- `stats.py` (added by the statistics task): paired tests, bootstrap and
  training-dynamics tables, computed from caches only.
- `run_all.py`, `execution.py`, `scripts/run_parallel.py`: orchestration,
  stage boundaries, locks and recovery.
- `quantize.py`, `bench.py`, `explain.py`, `proxy.py`: measured analysis paths.
- `report.py`, `notebook.py`: presentation from persisted evidence.
- `scripts/make_notebook.py`, `execute_notebook.py`,
  `package_deliverables.py`, `bench_pi.py`: notebook, packaging and Pi tools.
- Supplementary-campaign export and runner scripts live in `scripts/` and call
  package code; they must not duplicate training or scoring logic.
- `tests/`: invariants, leakage prevention, cache recovery and stage isolation.

Keep scientific logic in the package. Notebooks import it; do not duplicate
training or scoring implementations in notebook cells. Extend existing entry
points instead of creating a second orchestration path. Use small, focused
changes; do not add dependencies or refactor unrelated modules by default.

Use Python 3.11 and `.venv/bin/python` with the pinned requirements. Commands in
this file are written without machine-specific wrappers. On the maintainer's
machine, if `~/.codex/RTK.md` exists, read it and prefix shell commands with
`rtk` (`rtk proxy` preserves raw output). Keep such machine-specific notes in
`~/.codex/AGENTS.md` rather than in this repository file, and never make
portable project behavior depend on them.

## Verification

Choose checks according to what changed; report only checks actually run, with
their real output and exit status.

- Documentation-only work: verify referenced files, command syntax and
  whitespace with `git diff --check`. No full training, benchmark or notebook
  execution is needed solely for prose changes.
- Behavior changes: add meaningful regression coverage, run focused tests,
  then the relevant suite. Scientific changes must preserve the invariants
  above. Do not edit measured evidence to make tests pass.
- Before claiming the complete suite passes, run it and inspect its exit code
  and output. A standard local command is:

  ```bash
  env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/exlightnet-mpl .venv/bin/python -m pytest -q
  ```

  Use the documented JUnit/log outputs when refreshing published verification
  evidence. Do not overwrite them with a focused test run.
- Follow `CONTRIBUTING.md` for dataset-dependent tests and CI differences.
  Never launch the full five-seed campaign merely to validate an ordinary fix.
- For notebook changes, regenerate and execute when needed; generation clears
  outputs. Use `EXLIGHTNET_RUN_MODE=full` to read the existing full caches.
  Default fast-mode execution may train if compatible caches are absent.
- For hosted-runner code, run a CPU smoke test (tiny seeds/epochs, including
  resume after an interrupted run). Without a GPU, report the CUDA path as
  "not executed here".
- The two-seed/two-epoch local experiment budget applies to real RT-IoT2022
  training and the Stage B smoke. Tiny synthetic-data unit tests are exempt;
  record their count and duration separately from smoke execution. Tests must
  never train on the real corpus, execute benchmarks or mutate raw evidence.
- Before a refactor or a regeneration of derived files, verify the sha256 of the
  raw evidence is unchanged afterwards.

## Reports and deliverables

Change report generation in `src/exlightnet/report.py`; generated `REPORT.md`
must remain reproducible. All report tables must be valid Markdown. Preserve
the candidate-accounting wording: `consistent (94 - 1 constant = 93)`.
Report measured hardware/runtime, assumptions, deviations and stopped branches.
Every number in a report or notebook must be generated from a persisted file;
write "NOT RUN" or "PENDING" instead of a number for anything that was not run.

When packaged inputs change and a refreshed delivery is part of the task, run
`scripts/package_deliverables.py` after artifacts stop changing. It rebuilds
the report, requires an executed notebook and verifies CRCs and file hashes.
Never hand-edit `deliverables_manifest.json`. Ensure the archive contains the
intended files and excludes every `*.pkl`, the dataset, cached feature arrays,
prepared npz arrays, virtual environments, credentials and process locks. Local
tree checkpoints must stay available for benchmark recovery; packaging exclusion
is not a request to delete them. An extracted archive cannot re-benchmark absent
trees.

Do not claim Raspberry Pi performance from this x86 host. Optional KD, true
integer INT8 and unexecuted proxy experiments remain explicitly separate from
the completed reproduction.

## Git and collaboration

Follow `CONTRIBUTING.md`. Work on a focused task branch from `main`, preserve
unrelated edits, and stage explicit paths only when a commit is requested.
Use Conventional Commits with an English imperative summary. Review diffs and
run whitespace checks before committing. Never force-add ignored data, secrets
or large checkpoints. Never merge or push the historical `master` backup into
`main`; it contains a large checkpoint excluded from the clean history.

Do not commit, push, publish or merge unless the task authorizes it. Do not
force-push or rewrite shared history. Check actual Git availability instead of
repeating the early reproduction's obsolete Git limitation.

Respond to the user in Persian unless they request another language. Keep
code identifiers, repository documentation and commit messages in English to
match the project. Explain technical terms plainly and keep updates concise.
Proceed with authorized, reversible work; ask only when missing information
or a consequential decision blocks it. Before completion, state what changed,
what was verified and what remains, linking the relevant files. Update this
file when an agreed long-term project rule changes; put transient PIDs and
session logs in experiment records instead.

## Amendments

- A1 (2026-11-17, user-authorized): supplementary GPU training campaigns are
  allowed under "Campaigns, devices and amendments". `cpu_v1`, benchmarks and
  every earlier CPU-only requirement remain unchanged.
- A2 (2026-11-17, user-authorized): mixed-device neural/tree paired accuracy
  comparisons are allowed only within `gpu_v2` with explicit device columns and
  label; campaigns remain strictly separate. Cross-campaign device-equivalence
  is descriptive only, with differences/CIs and no p-value or pass/fail.
- A3 (2026-11-17, user-authorized): preserve legacy SD flags in immutable old
  JSON; active code/report/notebook/new outputs use paired tests. Tiny synthetic
  unit training is exempt from the real-data smoke budget under Verification.
- A4 (2026-11-18, user-authorized): dated protocol amendments define the primary
  last-epoch stopping endpoint, quantified paired uncertainty, seed-major
  scheduling/full-block inference, independent tree sessions and a separate
  CPU-only stratified final SHAP analysis. Preserve the original declaration and
  all cpu_v1 explanations; see docs/statistics_protocol.json amendment records.
