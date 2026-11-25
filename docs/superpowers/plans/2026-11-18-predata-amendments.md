# Pre-data amendments and stratified final SHAP Implementation Plan

> **For agentic workers:** Use `superpowers:dispatching-parallel-agents` for
> independent statistics, runner and analysis components. The controller owns
> protocol amendments, actual experiment execution, delivery and commits.

**Goal:** Preserve the original declaration and cpu_v1 evidence while applying
the user's dated pre-data changes and executing only the authorized final CPU
SHAP analysis.

**Architecture:** Append protocol amendments; do not rewrite the old rules.
Compute stopping endpoints and uncertainty from cached matched study records.
Reorder/filter plans without changing any identity. Add a standalone final
analysis path that reuses existing attribution/deletion helpers and reads only
the cached seed42 CPU model, arrays and old background indices.

**Tech Stack:** Existing Python3.11, PyTorch, NumPy, SciPy, pandas, SHAP,
Matplotlib and nbformat. No new dependency or benchmark.

## Global constraints and user clarifications

- The prior completed changes were committed first as `cbbea40`.
- No raw cpu_v1 file, old attribution, metric, receipt or checkpoint changes.
  A new all-existing-evidence SHA snapshot lives under
  `artifacts/predata_20261118/immutable_before.json`.
- No real training, no new smoke, no Stage B run, no CUDA and no benchmark.
- Tiny synthetic unit tests remain exempt from the old real-training budget.
- Original protocol fields/rule text remain verbatim. New effective rules
  reference dated appended amendments and are displayed alongside the original
  declaration, explicitly marking superseded text.
- Every plan prefix has complete matched blocks for all seeds except at most
  the final seed. Test every possible prefix. This interpretation was explicitly
  confirmed by the user on2026-11-18.
- For statistical comparisons, use only complete predeclared study seed blocks
  across the compared tasks/models/arms; mark partial seeds `incomplete; excluded`
  in completeness.csv. Never interpolate, impute or use pairwise availability
  to hide an unfinished final seed. B3 paper references require cached B1 rows.
- Keep neural/tree shards separately executable and merge verified archives
  within gpu_v2 under the already agreed mixed-device rule. Never skip XGB seeds.
- B6 is optional and will be recorded NOT RUN if skipped.
- No push/merge is authorized. Use explicit-path Conventional Commits locally.

## Review focus

- Primary/secondary endpoint disagreement and zero SD/undefined bootstrap ratios
  must be inconclusive rather than a false supported result.
- Missing seeds, arms or tree shards must be explicit and absent from paired
  inference; descriptive partial rows may remain visibly partial.
- Reordering and filtering must preserve original run/config/device identities.
- A trees-only session must never initialize CUDA, require a neural probe, fit
  neural models or drop deterministic XGBoost seeds.
- Stratified analysis must retain all rare-class flows, reuse the exact saved
  background and preserve old attribution bytes. It is final analysis, not a
  selection or repeated final probability-evaluation pass.

## Tasks

- [x] Commit the completed previous work after fresh tests/whitespace review.
  Done first: cbbea40;199tests passed in19.91s. Measured pytest traceback bytes
  retain whitespace through explicit .gitattributes rules rather than edits.
- [x] Append six dated protocol/progress amendments, snapshot original files
  and document endpoint/SHAP definitions before new analysis.
- [x] Statistics agent: replace `_hypothesis` with primary=fixed30 last versus
  paper best, secondary=fixed30 best versus paper best. Require all matched
  seeds42--56; use paired t95% mean CI, sample SD ratio,2000 paired-seed bootstrap
  draws with fixed seed42. Smaller SD requires ratio<=0.75 and upper bootstrap
  ratio CI<1; higher mean requires positive lower mean-difference CI. Report both
  endpoints and disagreement as inconclusive; never claim causality. Add pure
  boundary/degenerate/disagreement tests and show effective dated rules.
- [x] Statistics agent: exclude incomplete predeclared seed blocks from paired
  tests/flow bootstrap/correlations/hypothesis and report completeness explicitly.
  Preserve partial descriptive data and campaign/device validation.
- [x] Runner agent: B1/B2/B3 seed-major loops, exact multiset/fingerprint parity
  tests against previous ordering, every-prefix block test; add --only selection
  with safe CPU-only trees session/no probe and unchanged tree recipes/seeds.
- [x] Analysis agent: implement/test a new stratified final analysis module and
  thin CLI (no actual expensive job by the agent).25 test flows per true class,
  all if fewer; exact old200 training backgrounds; same Deep/Gradient settings.
  Primary per-class feature ranking uses each true class's own logit. Also save
  an all-output-channel ranking for comparability with the old global definition.
  Equal-class global importance averages the12 primary class vectors, not flows.
  Deletion uses each flow's original predicted-class SHAP ordering, as before,
  grouped by true class;20 random permutations and k10 contrasts. Save counts,
  early-deletion values, additivity/method/warnings/indices/model fingerprints,
  plots, CSVs and a completion manifest in a separate new final-analysis path.
  Report old300-random-sample composition from saved indices and cached labels.
- [x] Controller: execute this analysis once in a durable CPU-only supervisor,
  retain logs/checkpoints for recovery, no user-process controls or selection.
- [x] Controller: update report/notebook/runbook with generated final-analysis
  output and amendments; archive previous derived versions first. Kaggle runner
  remains a thin unexecuted hosted notebook. B6 NOT RUN if skipped.
- [x] Fresh tests, independent scoped review, immutable hash check, cached
  notebook execution and requested-artifact tested/not-tested manifest with SHA.
  Deliver notebook+prepared manifest+hosted-source ZIP together, exclude NPZ arrays.
- [x] Refresh component/main archives/checksums, verify CRCs/content and record
  gates. The explicit-path follow-up commit is the final local operation and
  its receipt is Git history; no push/merge.

## Recovery ledger

2026-11-18: current branch feat/cached-statistics-and-hosted-studies; baseline
commit cbbea40. Existing199tests pass, all916 original hashes unchanged, stage
lock available; only real supplementary records are the old gpu_v2_smoke rows.
No real Stage B data exists. The only new costly action authorized here is the
separate stratified CPU final analysis. All agents must leave actual execution
to the controller.

2026-11-18 recovery: one interrupted attribution attempt and one failed
supervisor bootstrap were preserved. The successful retry completed in233.016s
on CPU with259flows and the old200backgrounds. No receipt reset or old output
change. Details and implementation signatures remain in analysis_recovery.json
and analysis_execution_retry.json.

Final independent review: four important findings and one minor handoff issue
were fixed in one bounded RED/GREEN pass. CPU-tree scheduling now uses
run-specific GPU metadata and resolves compatible legacy fingerprints/pending
receipts; session provenance is preserved. Missing final-analysis manifest
members are rejected. Warnings, aggregate residual scope and effective
non-causal stopping rules are visible. No deferred finding or declined scope.

Fresh verification:253passed,923warnings,24.96s,exit0. Four synthetic cases,
fivefits/sevenepochs,1.968s summed pytest case time. No new real smoke or model
training. Cached notebook:17/17cells,zeroerrors,3.367093211s; fresh-kernel
restart:17/17cells,zeroerrors,3.379028596s. Offline hosted-source install/import/
CLI-help checks passed without downloads. Hosted notebook and CUDA NEVER
EXECUTED. All1047 old hashes and12 new output checksums verified. Mandatory
PA1--5 verification is in artifacts/predata_20261118/verification/GATES.md;
optional B6 remains NOT RUN. Final archive checks and commit close out next.

Archive closeout: all main/component CRCs, every disk/archive manifest hash,
the hosted trio and SHA256SUMS passed. REPORT.md has59valid Markdown tables;
REPORT_stats_v2.md has3. No pickle, prepared arrays, raw dataset, virtualenv,
recursive ZIP or process lock was included. The source ZIP matches the copy
installed offline. The three-file handoff includes exact execution limits.
This closeout note triggers a final deterministic refresh after archiving the
previous derived delivery; emitted manifests/audit remain authoritative for
the final archive hash. No source behavior changed after the253-test pass.
