# Reproduction progress

Authoritative brief read; PDF read completely; empty workspace inspected.
No usable Git repository: per-phase commits unavailable.
Plan interfaces: split indices -> preprocessor -> arrays -> all models; class
names and ranked feature names travel with all artifacts. No interface conflicts.

Phase 1: complete. Real UCI 942 downloaded through ucimlrepo. D1/D2 pass;
123117 rows, exact counts, split 86181/18468/18468 and required supports.
Phase 2: complete. Train-only preprocessing cached; P1 overlap 40/40,
rho 0.9994371482176362; 94 candidates before pruning, 93 after.
Data/preprocessing/metrics suite: 11 passed on the actual corpus.
Neural tests written first; initial collection blocked by missing PyTorch.
CPU PyTorch CDN repeatedly times out. Official S3 origin responds; download in progress.
No data gate was bypassed; no trained model has yet touched the test set.

Phases 3--10: implemented; 36 tests pass including exact M1, T1, baseline
shapes/counts, deterministic training, W8 bounds/NumPy parity and tiny SHAP.
PyTorch 2.6.0+cpu installed from official S3 origin; no GPU support or usage.
Fast notebook initially blocked by sandbox kernel sockets, then executed with
approved escalation. Calibration measured ~30 s/epoch; preserved alpha-only
calibration in results/fast_calibration. Fast defaults reduced to one epoch
per alpha candidate and final model to target the specified laptop runtime;
full defaults remain 20/30 epochs and an eight-epoch alpha sweep.

Final independent review: four important recovery/reporting issues and two
presentation/recovery issues found. Added regression tests, observed failures,
and fixed missing-array reconstruction, incomplete checkpoint/receipt rejection,
blocked-gate reports, alpha failure isolation, explicit missing-seed42 display,
and SHAP completion receipt ordering. Model/training recipe did not change.
All review findings addressed; none deferred. No additional reviewer dispatched.
Fast notebook live execution: 14/14 code cells, zero errors, 695.022 s.
Fast results: 19 model/variant rows, one seed each; D1/D2/M1/T1 and both Q1 pass.
P1 report-only: 40/40, rho=0.9994371482. Binary Q1 delta=-0.0002874585;
multiclass delta=-0.0019586271. SHAP DeepExplainer failed additivity and the
prescribed GradientExplainer fallback ran; approximate residuals are reported.
Full five-seed experiments, full-size SHAP and optional proxy are not executed.
Checked paired GradientExplainer randomness against pinned source and identical
models; the same per-class random seed preserves paired draws for shared rows.

Final verification: 49 pytest tests pass; zero failures/errors/skips. All 19
final model/artifact receipts are complete with exactly one test prediction
pass each. Fresh-kernel cached notebook restart: 14/14 cells, no errors, 9.408 s.
Standalone NumPy script executed on this x86 host (not a Pi measurement).
Required top-level summary convenience output added with explicit mode/seed scope.
Archive validated by CRC and SHA-256 file manifest, excludes dataset and arrays.

Follow-up request: isolate timing from parallel full training, freeze alpha once,
exclude PKL files, render report evidence as Markdown, and account for INT8 size.
Implemented --skip-benchmark, --alpha-only, --seed-worker, --aggregate-only and
--benchmark-only plus scripts/run_parallel.py. Workers load prepared caches and
frozen alpha, use one thread, and publish only seed-local status. Seed 42 owns
the shared SHAP cache; aggregation never runs uncached test analysis or proxy
training. Root-level locks cover fast/full shared artifacts and publications.
Final benchmarks clear old timings before refresh, preserve prior observations
in campaign audit files, report failures, and record pre-model CPU-idle checks.
INT8 raw weights/scales/rest/metadata and NPZ/container bytes are computed from
the actual files; report evidence uses Markdown tables. PKLs remain local only.
Independent workflow review findings were reproduced and fixed with regression
tests, including cache-only analysis, cross-mode locks and failed timing refresh.
DeepExplainer preflight detects unsupported graphs before full-set work, with
the prescribed fallback and unchanged sample/background settings.
The full eight-epoch alpha sweep is running once, serially; a waiting pipeline
will reuse its selections, launch seeds42..46 together, aggregate once, then
benchmark alone. No full-model test result has been produced during selection.

Full alpha sweep completed once in 2671.962845894 s. Frozen modes:
binary=effective_num; multiclass=none. Prepared-data fingerprint is unchanged.
Largest-CNN memory probe (batch512, train/validation tail shapes, no test flows):
515552 KiB peak RSS; private-dirty323168 KiB. Launcher requires3072 MiB available
before five concurrent workers. User confirmed "I'll free RAM; keep five workers";
retain five-way concurrency and wait for actual RAM headroom, not elapsed time.
Pipeline tool session96232 is waiting/running, log results/full/logs/pipeline.log.

All65 tests pass (923 retained third-party deprecation warnings),12.17 s.
RAM gate observed3350.140625 MiB and launched five workers together after420.064 s
waiting. Worker process IDs6,7,8,9,10 (same launcher PID namespace) correspond to
seeds42,43,44,45,46. Each worker records Torch threads=1 and interop threads=1.
CPU affinity allows all six host CPUs. Workers are training binary ExLightNet;
all timing is deferred. Full alpha selection will not be repeated.

The first interactive pipeline was interrupted when the session resumed. Its
ten completed binary ExLightNet/MLP results and single-pass receipts were
validated and reused. Interrupted logs/state are preserved under
results/full/interrupted/20261116T112858/. A detached host supervisor (PID569714)
restarted the five workers at2026-11-16T11:29:02Z with --reuse-alpha; worker PIDs
569716--569720 map to seeds42--46. This supervisor survives later user messages.
At2026-11-16T14:23Z, all60 headline results and77/95 total results are cached;
seed45 has completed19/19, while the remaining four workers finish ablations.
All77 existing final-test receipts are complete with exactly one prediction
pass. No worker metric contains benchmark latency; aggregation and the final
idle-CPU benchmark remain queued. The archive and notebook will be refreshed
after those stages, rather than packaging an incomplete full campaign.

Full campaign training and explanations are complete. All five workers exited
successfully; seed44 finished last at 2026-11-16T15:11Z. There are 95 results in
19 groups, each with seeds42--46. All 95 checkpoint fingerprints, prediction
arrays and complete single-pass receipts were verified. Summary means, sample
SDs and seed42 values were checked independently. D1/D2/M1/T1 and all ten full
Q1 checks pass; P1 remains report-only. Cache-only aggregation ran exactly once
in 6.6919 s. Full SHAP used 200 train backgrounds, 300 random test flows and two
local examples, finishing in 799.879 s. DeepExplainer failed additivity and the
prescribed GradientExplainer fallback ran; actual residuals are reported.

The first two benchmark-only campaigns stopped at the CPU-idle prerequisite,
with zero model timings collected. Their manifests remain under
results/full/benchmarks/. The user authorized stopping background apps while
preserving VS Code and the terminal/tunnel. A detached watchdog temporarily
paused JetBrains Toolbox, htop and only VS Code's GPU helper, then restored all
three at 2026-11-17T05:39:01Z. The editor process and terminal/tunnel stayed up.

The final benchmark campaign 20261117T052645467310-1050022 completed all95
measurements in 735.569886336 s, with zero failures/skips/gated models. One
process (PID1050022), one library thread, batch1, 100 warmups and 1000 timed
training-input inferences per model were verified. All285 recorded quiet
windows satisfy <=5% CPU busy for three consecutive one-second windows before
each model. All95 test receipts are unchanged after benchmarking. Actual
latencies and their five-seed summary statistics are consistent across the
per-model, per-campaign and aggregate files (results/full/benchmark_audit.json).
The original parallel_execution.json status preserves the initial stopped
timing branch; benchmark_recovery.json records its successful final recovery.

The regenerated notebook executed in full mode from caches: 14/14 code cells,
zero errors, 3.080653564 s. Seed42 INT8 size evidence: binary weights-only34560 B
versus full NPZ48877 B; multiclass weights-only37440 B versus full NPZ55271 B.
REPORT.md tables distinguish weights, scales, FP32 state, metadata and the ZIP
container.

Final verification: 65 tests pass in 10.10 s; 923 upstream deprecation warnings
are retained. The archive was rebuilt and independently checked: CRCs and every
SHA-256 file-manifest entry match, all38 REPORT.md tables have valid Markdown
headers and consistent column counts, and no PKL, dataset, feature-array cache,
lock or virtual-environment files are included. The archived full notebook has
14 executed code cells and zero error outputs; archived results have95 model
rows,95 isolated benchmark observations and95 unchanged final-test receipts.
All temporarily paused helpers were additionally checked in the host process
table and are running again. Documentation closeout is included in the final
archive refresh. The optional leakage proxy was not run; KD/true integer INT8
remain unimplemented, and no Raspberry Pi hardware measurement was available.

2026-11-17 — cached statistics and hosted supplementary studies started.
Authoritative follow-up: docs/STATS_GPU_BRIEF.txt. Current source branch is
feat/cached-statistics-and-hosted-studies, based on main6f76cb4; no commit,
push, merge or publication is authorized. The user's newer AGENTS.md bytes
were preserved exactly when creating the task branch. The old claim that Git
is unavailable is historical; Git is currently usable.

Before changing evidence/presentation, 916 existing files under results/ and
artifacts/ plus the genuine CSV were SHA-256 recorded in
artifacts/stats_cpu_v1/immutable_before.json. Previous report/notebook/delivery
versions and hashes are in archive/cpu_v1_before_stats/20261117T115530Z/.
The canonical stage lock was available. Actual audits confirm95 results,
19 groups, five seeds,10 Q1 passes,95 complete isolated CPU measurements and
the executed full notebook. No full campaign, benchmark, CUDA run or user
process action has been started in this task.

The supplied specification leaves three conflicts pending clarification:
R1 immutable old JSON versus A1 deleting legacy flags everywhere; R8 sole
smoke training versus the required legacy pytest suite's synthetic training;
and R5 device separation versus B1's CUDA-neural/CPU-tree comparisons.
Dependent destructive/budget/device-comparison operations are stopped; the
proposed interpretations and alternatives were sent to the user. Independent
cached statistics, prepared export and runner code can proceed. A5 is NOT RUN
under the stricter no-benchmark rule. The historical MLP compatibility gate
will validate the unchanged cache, not retrain it beyond the local budget.

Design and plan are in docs/superpowers/specs/2026-11-17-statistics-and-hosted-studies-design.md
and docs/superpowers/plans/2026-11-17-statistics-and-hosted-studies.md.
The verbatim a-priori rules and pending study report were persisted before
any Stage B data. Focused agents own statistics, prepared-data export and
runner package modules; the controller alone owns any actual smoke execution.

User clarifications accepted on2026-11-17: preserve all original summary,
ablation, manifest, receipt and archive bytes. Old ablation JSON still contains
obsolete SD-overlap flags; active reports/notebooks must never read/display
that file. The removal check explicitly excludes historical evidence and
authoritative instructions, and checks active source/presentation/new outputs.
Tiny synthetic unit training is exempt from R8; tests still cannot train on
the real corpus, execute a benchmark, or mutate raw evidence. Record synthetic
test counts/timing separately from the sole real two-seed/two-epoch CPU smoke.
Within gpu_v2 only, matched-seed CUDA-neural/CPU-tree comparisons are permitted
with explicit mixed-device labels and per-family device columns. Mixed
campaigns remain a hard error, with no override. The cpu_v1/gpu_v2 equivalence
view is descriptive difference/CI only, with no p-value or pass/fail. AGENTS.md
now carries these agreed ongoing rules.

Stage A statistics: ten CSVs generated from95 rows/19 groups;17 paired-flow
comparisons use2000 resamples. New pure statistical tests pass and all916
original hashes were checked unchanged. Prepared exports for42--51 finished
serially in266.409349 s; all ten files verify payload/recipe/file SHA and full
class coverage. Split42's nine arrays and three index vectors match the old
cache byte-for-byte, preserving fingerprint
dfba90bd116f9288bc72a837160516984ba773b11904e98561c80cda7dedcf5c.
No real training or inference was performed during either task.

Runner integration review caught a CPU tensor alias in the last-epoch state:
loading best weights could mutate the saved last-state view. A no-training
regression reproduced it, then independent CPU clones fixed it. Safe archive
restore and claimed-receipt guards were also tested before the sole smoke.
CUDA execution remains NOT RUN.

Publication tests initially exercised an old top-level notebook script before
its isolated-output option existed, temporarily clearing the derived notebook.
The archived notebook was restored byte-for-byte (SHA18b395f832d14cebc7de487a82b46326d19c685e21ab6214619088feaaec2ada).
An attempted kernel startup failed at sandbox sockets before any cell ran;
no training, inference, cache or old verification record was changed. Tests
were isolated into temporary copied script roots before further execution.

Stage B's sole real-data smoke finished under its detached controller: MLP,
multiclass, seeds42/43, exactly2epochs each (4total). The first process stopped
after epoch1's atomic checkpoint; its ZIP restored into a fresh directory.
The first history row was preserved, not repeated. Both runs finished, four
best/last evaluation receipts have one prediction pass each, and a cache-only
restart reported zero new runs with unchanged receipts. End-to-end supervisor
wall time10.875827762s, recorded in artifacts/stats_cpu_v1/smoke/control.json.
No other real-data training, GPU execution or benchmark occurred in this task.
The original smoke ZIP and records were not modified during subsequent fixes.

Canonical CPU compatibility was checked by comparing the original CPU function
ASTs and reading the validated multiclass MLP seed42 cache. Its macro-F1 and
fingerprint exactly match the old row; no canonical model was retrained.
This qualification is explicit in cpu_default_compatibility.json and Gate B.

Independent review found four Important boundary problems: completing a tree
after its evaluated ZIP loses the excluded pickle; stopping one branch blocked
later independent plans; import accepted impossible terminal epoch histories;
and a literal cuda label could pool different actual GPU/runtime identities.
All received RED-to-GREEN regressions and fixes. The warning-resume finding
was upgraded from Minor because the brief explicitly requires retaining every
nondeterminism warning; checkpoint warning ledgers and effective precision/
deterministic settings now survive recovery. Prepared-data/provenance failures
stay fatal; run-local stopped branches preserve evidence and continue other
plans. No new real-data run was used to test the fixes. The immutable smoke ZIP
was successfully imported by the stricter final aggregator into a new output
path artifacts/stats_gpu_v2_smoke_verified/. Review and regression evidence are
under artifacts/stats_cpu_v1/review/.

Final full-suite verification:199passed,923 retained upstream warnings,15.12s,
exit0. Its four tiny synthetic-training test cases performed five fits/seven
epochs on4or24synthetic training rows,1.323s combined pytest case wall time.
These counts/time belong to that final suite, not the real smoke. Earlier
full-suite attempts and focused checks remain separately recorded; the first
attempt's failure was the new synthetic optimizer guard missing one existing
four-row fixture, corrected without weakening the no-benchmark rule. An
additional focused publication run's single four-row fit took0.00757395s,
preserved in verification/publication_synthetic_training/. Tests cannot execute
actual benchmark calls or undeclared optimizer steps. No raw evidence changed.

The updated notebook executed16/16cells with zero errors in3.329371227s; a
fresh-kernel Restart-and-Run-All repeated cached analysis in3.431854354s,
16/16cells,zeroerrors. New metadata is under verification/notebook_first/ and
verification/final/notebook_restart/, leaving original results metadata intact.
The final raw-evidence audit verifies all916 hashes unchanged; all seven Gate A
and seven Gate B checks pass in verification/final/GATES.md. B means code/CPU
smoke verification, not executed CUDA. All real gpu_v2 studies remain PENDING.

All approximate numerical claims in the supplied review were reproduced from
the generated tables (reviewer_claims_rederived.json); no mismatching claim was
found. The statement that eager latency is mostly dispatch is not established:
dispatch is included, but its fraction is NOT MEASURED and no exported-model
latency was run. This is documented rather than presented as a measured fact.
An offline no-dependency source installation succeeded using system Python3.11
pip into a temporary target (the task venv has no pip/build backend); import and
prepared-file verification used the task interpreter. No dependencies were
downloaded or added. Actual Kaggle/CUDA execution remains untested here.

The hosted notebook is thin, unexecuted locally, and calls package code only.
The beginner runbook specifies private prepared/code uploads, offline install,
probe first, B1/B3/B2/B4/B5 priority, incremental ZIPs, fresh-session restore,
and local verified aggregation. Canonical study rules were recorded before
the smoke; the real-study report preserves them verbatim with PENDING sections.
Component/delivery ZIP rebuilding and final checksum/CRC verification follow.

Final refresh after the generated verification table was added to REPORT:199tests pass,923warnings,17.15s,exit0. Four synthetic-training cases/fivefits/sevenepochs took1.381s combined pytest case wall time in this invocation; prior verification and hashes were copied to final/previous_suite/. Both gate groups remain allPASS and all916 original file hashes remain unchanged. The task plan is complete; final ZIP/component checksums are generated after this documentation closeout.

2026-11-18 — baseline commit and dated pre-data amendments.
User explicitly requested a local commit first. cbbea40 records the completed
cached-statistics/hosted-runner work, after a fresh199-test pass (19.91s,
923upstream warnings). Diagnostic log/XML whitespace is preserved verbatim via
.gitattributes rather than edited to satisfy whitespace checks. No push/merge.
Only old gpu_v2_smoke data exists; no real Stage B observations were found.
All916 original raw hashes still match and the stage lock was available.
An expanded1047-file existing-evidence snapshot is recorded in
artifacts/predata_20261118/immutable_before.json. Prior protocol/report/notebook/
handoff versions and their SHA are archived under archive/predata_20261118/.

DATED AMENDMENT PA-20261118-1: stopping-hypothesis PRIMARY is fixed30 LAST
versus paper BEST-validation; fixed30 BEST versus paper BEST is SECONDARY.
Report both, use all matched42--56 seeds, require SD ratio<=0.75 and upper95%
paired-seed bootstrap ratio CI<1 (2000draws,seed42), plus a positive paired95%
mean-difference CI. Disagreement/undefined ratios are inconclusive; never claim
causality. Original protocol declaration remains verbatim with this appended
superseding amendment, not a silent rewrite.

DATED AMENDMENT PA-20261118-2: B1/B2/B3 are seed-major with identical run
identities. User explicitly clarified the mathematical prefix edge: every
possible prefix may have at most its last seed incomplete; all earlier seed
blocks contain all tasks/models/arms. Inference excludes incomplete comparison
blocks and completeness.csv states incomplete; excluded. No imputation.

DATED AMENDMENT PA-20261118-3: --only all/neural/trees defaults all. Trees may
run in a CPU-only session of gpu_v2 without GPU or neural probe, retaining all
RF/XGBoost seeds/recipes and actual devices. Verified shards may merge within
gpu_v2 under the existing explicit mixed-device policy; campaigns never pool.

DATED AMENDMENT PA-20261118-4: hand off kaggle_runner.ipynb, prepared/manifest.json
and hosted_source.zip together with per-fileSHA and individual tested/not-tested
statements. Syntax/import/installation checks are not hosted CUDA execution.

DATED AMENDMENT PA-20261118-5: one new CPU-only final analysis is authorized on
the cached seed42 multiclass FP32 model:25 flows per true class or all if rarer,
exact previous200-flow training background, existing Deep/Gradient settings,
measured additivity. Primary class ranking targets the class's own logit;
an auxiliary all-output average preserves comparability with the old definition.
Global ranking gives equal weight to each true class. Deletion keeps the
original predicted-class target and20random permutations, grouped by true
class, with k10 comparisons. Report the earlier300random-flow class composition
from saved indices without the two extra local examples. Old explanation files
are immutable; this is final analysis and cannot select a model or recipe.

DATED AMENDMENT PA-20261118-6: optional B6 leakage PROXY is deferred, NOT RUN.
No B6 code, export or training is claimed. No new real training, smoke or
benchmark is authorized or needed for the mandatory amendments. CUDA remains
NOT RUN locally. Implementation/execution ledger is in the dated plan.

2026-11-18 — completed stratified final analysis and review fix pass.
The CPU-only analysis completed with exit0 in233.015562949s (supervised wall
time),259test flows and the exact prior200 training-background positions.
Five SSH-brute-force and four FIN-scan flows were all retained; the other
ten classes contributed25each. GradientExplainer used200samples after the
same DeepExplainer additivity preflight failed. The preserved warning is
unrecognized nn.Module: AdaptiveMaxPool1d. Absolute logit-additivity residuals
are max3.3370019332892307 and mean0.2966468941682444, aggregated across259x12
flow/output pairs. Separate output-logit summaries were not retained; this is
the same aggregate reporting as before, not an exact SHAP claim. PA-5's
classwise-logit wording describes that computation, not per-output saved
summaries. No new inference was added to reconstruct missing diagnostics.

The first supervised attempt was deliberately interrupted before saving any
attribution checkpoint after a NumPy-metadata serialization bug was found.
Its original claim/logs remain under the separate attempt1_interrupted path;
analysis_recovery.json records the reason and unchanged declared inputs.
The corrected implementation was frozen before the successful retry. A failed
supervisor bootstrap quoting attempt also remains recorded. No receipt was
reset and no earlier explanation or source campaign evidence was edited.

The new analysis is final, never selection. All12 output checksums, all
per-class/global rankings,240random orders, cached k0 values, k10 contrasts
and the old300-flow composition were independently checked without model
queries. The old sample contained232dos_syn_hping flows and zero flows from
ddos_slowloris, SSH-brute-force and FIN-scan; the two extra local examples
are not counted. All1047 expanded preservation hashes remained unchanged.

Independent review found four important issues, now covered by RED/GREEN
regressions: CPU-tree GPU-name provenance could change a fingerprint across
selectors; pending-tree resume compared mutable session metadata; final
presentation could accept an incomplete output-hash set; presentations
omitted warnings/effective stopping endpoints. New CPU-tree fingerprints use
run-specific GPU-name=null; compatible legacy identities retain their opaque
fingerprints, pending provenance and receipts. Ambiguous/mismatching identities
stop rather than create a duplicate. The raw fingerprint function is unchanged,
and all pre-amendment plan/fingerprint multiset tests still apply. Claimed
legacy receipts remain blocked. Tree recovery checks passed46tests in6.11s.
Presentation checks passed21tests in3.00s. A single test initially expected
the word claimed rather than the existing incomplete-receipt message; its
assertion was corrected without changing receipt behavior.

The review's minor handoff issue is fixed too: measured sizes/hashes cannot
be replaced by supplied status metadata, and each rendered statement has an
explicit TESTED or NOT TESTED label. Packaging tests passed6cases in1.97s;
README documents the full handoff refresh command. No review findings remain
deferred. Final suite, notebook, source-install and archive checks follow below.

Final full-suite evidence is under artifacts/predata_20261118/verification/:
253passed,923upstream warnings,24.96s,exit0. The four tiny synthetic-training
cases performed five fits/seven epochs on4or24rows and consumed1.968s combined
pytest case wall time. Real-data smoke in this follow-up: NOT RUN. The earlier
two-seed/two-epoch smoke evidence and its10.875827762s supervisor duration remain
historical, unmodified and separate. No real B1--B5 observations exist.

The regenerated cached-analysis notebook executed17/17cells without errors in
3.367093211s; a second fresh-kernel Restart-and-Run-All took3.379028596s with
17/17cells andzeroerrors. Its new records do not overwrite old metadata.
The thin Kaggle notebook was NEVER EXECUTED locally or on Kaggle; its schema,
syntax and stubbed command wiring were checked. CUDA remains NOT EXECUTED HERE.

All ten prepared splits passed current file/payload/schema checks. Split42's
nine arrays and three split-index vectors equal the canonical cached bytes.
prepared/manifest.json remains unchanged, SHA256
95eb221ecbd74f172d978abd58d3f5c2863ac9708288446a331b6fce980b7b35.
The current hosted source ZIP passed CRC/member checks and a clean offline
no-dependency install; imports and CLI --help used the installed copy. No
download, dependency addition, actual hosted fit or GPU probe occurred.

The requested three-file handoff includes HOSTED_HANDOFF.json/.md with a full
SHA256 and individual TESTED/NOT TESTED statement for each artifact. Prepared
arrays are excluded. Gates PA1--5 pass implementation/evidence verification;
this does not mean real GPU studies ran. Optional B6 is NOT RUN. Final main/
component ZIP checksums and CRC validation are emitted after packaging.

Delivery verification passed: all main/component ZIP CRCs, every packaged file
against both disk and archive manifest hashes, all SHA256SUMS entries and all
three handoff files/status statements. REPORT.md contains59valid Markdown
tables and REPORT_stats_v2.md contains3; the candidate-accounting wording and
INT8 weights-only/full-artifact breakdown remain intact. No pickle, raw data,
prepared arrays, optimizer recovery checkpoint, recursive ZIP, credential or
process lock entered deliverables.zip. Local tree checkpoints were preserved.
The source ZIP equals the offline-installed copy. Old derived delivery bytes
and their hashes were archived again before the final closeout refresh.
No Python behavior changed after the253-test pass; documentation/manifest
closeout and the authorized explicit-path local commit finish this task.
No merge or push. Final archive hashes are external in delivery/stats_gpu_v2/
SHA256SUMS and DELIVERY_AUDIT.json, avoiding a self-referential archive hash.

## 2026-11-21 — Local aggregation of the first gpu_v2 bundle

Copied the supplied `gpu_v2_part1.zip.zip` into `imports/gpu_v2/` byte for byte.
Its SHA-256 is `6c346f558499ffbd1e5e8c9dce906dde36a44c35adad0cd3073bb5bc8467e482`.
Archived the 13 previous statistics files under
`archive/gpu_v2_before_aggregation/20261121T120139+0330/` with their checksums.

```bash
.venv/bin/python scripts/aggregate_gpu_v2.py imports/gpu_v2/*.zip \
  --prepared-dir prepared --output artifacts/stats_gpu_v2 \
  --report REPORT_stats_v2.md --reference-cpu results/full/results.json
```

The command exited 0. The aggregation manifest records 333 verified unique
hosted CUDA runs and 2,188 artifact hashes. ZIP CRC and output SHA-256 checks
passed. All 733 recorded input and CPU evidence checksums matched after the
run. The structure of 11 CSVs and nine Markdown tables was checked. The local
operation used saved predictions and performed no model training or inference.

B3 has all 15 matched seed blocks. B1 has 210 of 330 planned runs and lacks
120 Random Forest and XGBoost runs. All 30 B1 seed blocks are excluded from
paired analyses. B2, B4 and B5 remain pending in the imported evidence.

## 2026-11-23 — Proposed gpu_v2 amendment for two CUDA environments

Preflight read the Colab attachment at `/home/mparsict/Downloads/gpu_v2.zip`
directly. Its size is 12,837,923 bytes and its SHA-256 is
`888fb932dd60c5bec22a681b73de91ffd04672ff6b5f3de4a4e167addc9891ec`.
ZIP CRC and JSON inventory checks passed for both bundles. The Colab archive
contains 111 entries marked complete (one probe, 50 B2 neural, 30 B4 and
30 B5). The existing aggregation manifest records 333 verified Kaggle runs.
Both archives declare `gpu_v2` and a Tesla T4. They differ in NumPy, PyTorch
and cuDNN versions. [Existing manifest](../artifacts/stats_gpu_v2/aggregation_manifest.json)

- The combined import would stop at `Conflicting duplicate completed identity`.
  Both probes share the key `(gpu_v2, probe, probe, multiclass, exlightnet, 42,
  42)` and have identical configs and data identifiers. Their fingerprints
  and library metadata differ. The duplicate key omits both fields.
  [Identity fields](../src/exlightnet/study_stats.py#L33) · [Duplicate check](../src/exlightnet/study_stats.py#L257)
- Resolving the probe collision alone would still leave a hard stop at
  `Mixed actual execution environment for the same model seed vector`. The
  guard groups by task and model across every study and arm. The environment
  identity includes library versions, CUDA, cuDNN and GPU name.
  [Environment fields](../src/exlightnet/device.py#L28) · [Environment guard](../src/exlightnet/study_stats.py#L366)
- B4 is blocked before completeness and pairing. All 30 Colab uncapped60
  entries have Kaggle B1 paper references with matching task, model, seed,
  split and data identifiers. The downstream B4 lookup and pairing code uses
  those keys and checks split and data equality. Bypassing the import guard
  would allow cross-environment pairs without the requested report label.
  [B1 reference reuse](../src/exlightnet/study_stats.py#L392) · [Pairing](../src/exlightnet/study_stats.py#L470)

### Proposed rules for review

1. Keep the hard mixed-campaign error and the existing device policy. Require
   one CUDA environment per study, arm, task and model seed vector. Scope probe
   identities by recorded environment and report each timing probe with its
   own source and environment. Preserve the original record fingerprints and
   receipts. Keep conflicting same-environment duplicates and ambiguous
   planned study slots as hard errors.
2. Use Kaggle for the B1 and B3 neural runs, and Colab for B2 and B5. Resolve
   provenance from the verified archive and recorded environment for each
   output row. Apply complete-block eligibility within each study. Missing
   B1 and B2 tree runs still exclude their incomplete seed blocks from inference.
3. Explicitly authorize B4's Kaggle B1 paper references and Colab uncapped60
   records as the sole cross-environment neural pairing. Require the existing
   matched seed, split and prepared-data checks. Use
   **cross-environment (Kaggle references vs Colab uncapped60)** in the B4
   report and show both arm environments. The paired contrast measures the
   joint change in training recipe and execution environment. It cannot
   isolate the effect of the epoch cap.
4. Add a descriptive comparison of Colab B2 split-42 and B5 `lr_0.0015`
   against Kaggle B1. Match on task, model, seed, split, data identifiers and
   full config. For each matched checkpoint, list exact agreement or divergence
   in saved predictions and macro-F1 separately. Report unmatched entries
   explicitly. Use no hypothesis tests for this comparison.

Review is pending. The attachment was read from Downloads so it stays outside
the existing aggregation command's ZIP glob. The target of 444 verified runs
requires an approved amendment and a fresh aggregation after archiving the
previous outputs. Combined aggregation, the descriptive comparison, training,
benchmarking and B6 are **NOT RUN**.

## 2026-11-23 — PA-20261123-1, gpu_v2 execution environments

The user approved this outcome-independent amendment before inspection of
Colab B2, B4 or B5 statistics. The preceding proposal records the rationale.

1. Require one `environment_id` per inferential vector
   `(study, arm, task, model)`. Stop when a vector contains more than one
   environment. Studies and arms may use different environments. The
   existing one-device rule still applies per `(task, model)`.
2. B3 requires B1 references from the same environment. Stop on a mismatch.
   B4 may reuse binary B1 paper references across environments. For such B4
   results, the summaries, paired tests and paired flow bootstrap rows carry
   `cross_environment=True` and both arm environment IDs. Put
   **cross-environment (Kaggle B1 references vs Colab uncapped60)** next to
   every B4 verdict in the report.
3. Probe and determinism are infrastructure studies. Exclude them from
   eligibility and inference. Scope their identities, tables and determinism
   repeat pairs by recorded environment. Stop on a conflicting duplicate
   within one environment or any duplicate in an inferential study.
4. Resolve environment metadata by `(study, arm, task, model)`. Check the
   grouping used in correlations, learning-rate selection and summaries.
   Preserve the device-equivalence calculation and its B1 split-42,
   seed-42–46 filter.
5. Preserve existing statistical methods, thresholds, seeds and stopping
   rules. Preserve raw records and keep campaigns separate.

The gates run in order. Gate (a) verifies the amendment with synthetic
records. Gate (b) compares every Kaggle-only statistical table and the
report body against the previous outputs and checks the `cpu_v1` hashes.
For byte-identical single-environment output, the infrastructure table
schema uses the campaign manifest's single environment. Multi-environment
outputs add explicit `environment_id` columns to those tables.
Gate (c) records both results and the changed code hashes here before any
real Colab analysis. After the gates, archive the previous outputs and use
verified saved records and predictions for the combined aggregation and the
descriptive B2 and B5 comparison. Training and B6 remain **NOT RUN**.

### PA-20261123-1 verification before Colab aggregation

- Gate (a) passed with exit code 0. The no-training pytest run on
  `tests/test_study_stats.py`, `tests/test_stats.py` and
  `tests/test_predata_protocol.py` reported 84 passed in 8.13 s.
  The synthetic ZIPs cover cross-study environment labels, same-vector
  rejection, probe and determinism isolation, duplicate rejection, B3 and
  B4 reference reuse, and the descriptive comparison.
- Gate (b) passed with exit code 0 for the Kaggle-only aggregator and the
  comparison. The regenerated output verified 333 unique runs. All 11
  CSVs and the entire Markdown report match the previous bytes. The
  `cpu_v1` inventory covers 806 files. Its before and after SHA-256 is
  `6369242b08ea89b9cfb83da8a7b6ddf9397539dca849a52c5b45a4916cea4610`.
  Every individual file hash matches.

| Compared file | SHA-256 of previous and regenerated bytes | Result |
| --- | --- | --- |
| `completeness.csv` | `bf7fa50ddd43daeb347e81e828791f0f840994b531e302305c968d36459fa2eb` | identical |
| `correlations.csv` | `b8113a3dcdee55cd6e18ba373ff01ca207c909263d9d8621a3749dd4072f2815` | identical |
| `determinism.csv` | `76f593e098100d16d3843df30e32a004ea71265533a10c60353dbdf8f286b90a` | identical |
| `device_equivalence.csv` | `0ccb3c1a976f603c90ff98bad94d52c64bd827cc125b93c4c4acb55f0d4c66e8` | identical |
| `early_stopping_hypothesis.csv` | `f6a4af36213e7aab518b01f94e7ecc4a3a3678643c7e667bf14e54d004ae5972` | identical |
| `lr_selection.csv` | `8e9055164ca997186766a25617539874f83dfff575eaa195c651ea365bb616ee` | identical |
| `paired_flow_bootstrap.csv` | `996b023290ff69cbc438842a01a1502a32ff5f1d52b1c357b30e0ff02a1586ed` | identical |
| `paired_tests.csv` | `5e8fabdef0aae9735d96e66796eaf5f3be13922914e941b573f58a683fe275a8` | identical |
| `per_run.csv` | `acf7f57d8320c89615f24320b9bc0790c643ce9ad793f33476e7ea917b951732` | identical |
| `probe.csv` | `a3e1babc6c89b7de5362cd41bc3498d298c762f28f3e0ec595f8ea79973bb07d` | identical |
| `REPORT_stats_v2.md` | `534b92d9b4598c6cf40c598cd570c35050c4846d5fd8de8e8cb6618faa033dc4` | identical |
| `summaries.csv` | `89fb392c78a2e6a86dec4144dcfeaf0c0aec4b37b8ea62d75846ce4e2dadf420` | identical |

The row provenance check covers B1-only correlations, B5 validation means
grouped by arm, and summaries grouped by study, arm, task, model, checkpoint
and environment. The B3 hypothesis takes provenance from its paper
reference. Device-equivalence retains its existing B1 filter and calculation.

Gate (c) records the changed code hashes below. The Colab bundle had yet to
be staged or used for statistics when this verification was recorded.

```text
fefbdea47e52a9369ddb7923b9bbfe0f39516b6f0dc5f68d2629359fde607f3d  src/exlightnet/study_stats.py
a7a91b46de24e19daa93eb9491eb9ce579a6e1148090f759e3f758859da68de6  tests/test_study_stats.py
```

### 2026-11-23 — Combined gpu_v2 aggregation under PA-20261123-1

Archived 13 previous outputs under
`archive/gpu_v2_before_aggregation/20261123_PA-20261123-1/` and verified their
`PREVIOUS_SHA256SUMS`. Then copied the verified 12,837,923-byte Colab ZIP from
Downloads to `imports/gpu_v2/gpu_v2_colab.zip`. The source and copy both have
SHA-256 `888fb932dd60c5bec22a681b73de91ffd04672ff6b5f3de4a4e167addc9891ec`.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -B scripts/aggregate_gpu_v2.py \
  imports/gpu_v2/gpu_v2_part1.zip.zip imports/gpu_v2/gpu_v2_colab.zip \
  --prepared-dir prepared --output artifacts/stats_gpu_v2 \
  --report REPORT_stats_v2.md --reference-cpu results/full/results.json
```

The combined command exited 0. The manifest records 444 complete, unique runs
from two CUDA environments, comprising 333 from Kaggle and 111 from Colab.
It records 2,926 verified artifact hashes and zero incomplete input records.
All 12 manifest output hashes and all 820 input and `cpu_v1` hashes matched.
All 18 Markdown tables passed the column-count check.
[Manifest](../artifacts/stats_gpu_v2/aggregation_manifest.json) ·
[Run log](../archive/gpu_v2_before_aggregation/20261123_PA-20261123-1/aggregation.log)

| Study | Completed new runs | Eligible seed blocks | Reused B1 references | Reason for exclusion |
| --- | --- | --- | --- | --- |
| B1 | 210/330 | 0/30 | — | All 30 seeds lack RF and XGBoost for both tasks (120 runs). |
| B2 | 50/90 | 0/10 | — | All 10 Colab split blocks lack RF and XGBoost for both tasks (40 runs). |
| B3 | 120/120 | 15/15 | 60/60 | None. Both arms and B1 references use the Kaggle environment. |
| B4 | 30/30 | 15/15 | 30/30 | None. B4 requires its two binary B1 model references per seed. |
| B5 | 30/30 | 10/10 | — | None. All three learning-rate arms use Colab. |

Incomplete B1 and B2 blocks are excluded from paired tests and paired flow
bootstrap. The B1 correlations and CPU/GPU device-equivalence tables have no
eligible observations. Both timing probes and the two Kaggle determinism
repeats are counted as verified records and appear outside eligibility and
inference. Colab supplied a probe and no determinism repeats.
[Completeness table](../artifacts/stats_gpu_v2/completeness.csv)

The B4 summary, paired test and flow bootstrap tables each have eight rows
marked `cross_environment=True` with the Kaggle paper and Colab uncapped60
environment IDs. The report places **cross-environment (Kaggle B1 references
vs Colab uncapped60)** next to all eight B4 verdicts. B1 tree gaps do not
affect B4's complete blocks.
[Statistics report](../REPORT_stats_v2.md#b4-binary-capped-versus-uncapped)

The descriptive note was checked independently against the saved prediction
arrays and metric values for all 15 identical-config pairs. In B2 at split
and train seed 42, binary and multiclass ExLightNet and MLP agree exactly in
predictions and macro-F1 at both checkpoints. Multiclass standard_conv
differs in both measures at both checkpoints. In B5 `lr_0.0015`, each of the
ten Std-CNN seeds 42–51 differs in both measures at both checkpoints. The note
uses no hypothesis test. Training, new inference, benchmarking and B6 are
**NOT RUN**.
[Descriptive comparison](../REPORT_stats_v2.md#descriptive-cross-environment-repeats)

## 2026-11-23 — PA-20261123-2, local CPU tree runs for gpu_v2

The user authorized this location-only extension of PA-20261118-3 before
execution. The exception to the local smoke budget covers exactly 120 B1
and 40 B2 RandomForest/XGBoost runs in `gpu_v2` on this machine's CPU. It
preserves the declared plans, recipes, statistical methods, thresholds and
eligibility rules. Neural training, training tests, benchmarking and B6
remain **NOT RUN** for this task. The preceding Colab work was committed
locally as `34f2089`.

All 160 runs and any resume use the existing `.venv/bin/python` (Python
3.11.2) with NumPy 1.26.4, PyTorch 2.6.0+cpu, scikit-learn
1.6.1 and XGBoost 2.1.4. No packages will be installed or changed
during the campaign. The runner and six supporting source files matched
`delivery/stats_gpu_v2/hosted_source.zip` byte for byte. The existing
`prepared/manifest.json` SHA-256 is
`95eb221ecbd74f172d978abd58d3f5c2863ac9708288446a331b6fce980b7b35`.

Both workers use `scripts/kaggle_campaign.py --prepared-dir prepared` with
`--device cpu --only trees --campaign gpu_v2 --time-budget-hours 6 --zip-every 5`.
The environment sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS` and `NUMEXPR_NUM_THREADS` to 1 before import. The workers
have separate output roots and archives. The supervisor runs one worker
at a time because about 2 GiB of RAM is available and swap is nearly full.

| Study | Output directory | Runner ZIP |
| --- | --- | --- |
| B1 | `archive/gpu_v2_local_trees/20261123_PA-20261123-2/B1` | `archive/gpu_v2_local_trees/20261123_PA-20261123-2/B1/gpu_v2.zip` |
| B2 | `archive/gpu_v2_local_trees/20261123_PA-20261123-2/B2` | `archive/gpu_v2_local_trees/20261123_PA-20261123-2/B2/gpu_v2.zip` |

The CLI plan check must complete without fitting any model. The detached
supervisor records commands, processes, timestamps and exit codes. Recovery
uses the same environment, verified local evidence and `--resume-zip` when
restoring an archive. Completion requires both verified ZIPs before copying
them to the import directory. Archive the previous aggregates before running
the existing aggregator on all four bundles. Any aggregator failure requires
a dated proposal for review.

The CLI plan check exited 0 and printed 120 selected B1 trees and 40 selected
B2 trees. In the dry process the budget gate blocked every iteration and
both fitting functions had fail-fast guards. Both temporary ZIPs contained
zero records. The actual output and ZIP paths are recorded with the CLI log.
[Dry check](../archive/gpu_v2_local_trees/20261123_PA-20261123-2/dry_check.log)

The isolated command namespace closed the initial detached launch before
the supervisor created state, locks or study outputs. Inspection confirmed
that neither worker had started. The same supervised plan is launched in
the local user's process namespace for persistence across tool calls.

### 2026-11-23 — PA-20261123-2 execution and verification

The detached supervisor completed both studies sequentially on the local
x86-64 Intel Core i5-8500. Each worker exited 0 with zero stopped runs. The
runner recorded 2,575.12 s for B1 and 862.90 s for B2. Each ZIP passed CRC,
plan identity, library version, prepared-data and artifact digest checks.
All 160 records have status `complete` and device `cpu`. Each tree's best
and last entries share its saved predictions and single complete evaluation
receipt. The source ZIPs were retained and their import copies matched
byte for byte.
[Supervisor state](../archive/gpu_v2_local_trees/20261123_PA-20261123-2/state.json) ·
[ZIP verification](../archive/gpu_v2_local_trees/20261123_PA-20261123-2/verified_imports.json)

| Import ZIP | Records | SHA-256 |
| --- | ---: | --- |
| `gpu_v2_trees_b1.zip` | 120 | `3487c87246d43171cf74b398c4b9a3805df2a4734c1e9dbcab6b5c4b25e99259` |
| `gpu_v2_trees_b2.zip` | 40 | `68fad246b067ce6d69af105d16d405eaf0a11caeb8589426c9277506d722bb8b` |

Archived the 13 outputs from the 444-run aggregation with their SHA-256
values. The four-bundle aggregator exited 0 and verified 604 unique runs,
zero incomplete records and 3,406 artifact hashes. All 12 output hashes,
824 input and `cpu_v1` hashes, and all 13 archived output hashes matched.
All 24 report tables passed their Markdown column checks. Aggregation used
the committed code from `34f2089` and the authorized device label
`mixed-device within gpu_v2; trees are device-independent`.
[Archive checksums](../archive/gpu_v2_before_aggregation/20261123_PA-20261123-2/PREVIOUS_SHA256SUMS) ·
[Aggregation log](../archive/gpu_v2_before_aggregation/20261123_PA-20261123-2/aggregation.log) ·
[Manifest](../artifacts/stats_gpu_v2/aggregation_manifest.json)

| Study | Completed new runs | Eligible blocks | Reused B1 references |
| --- | ---: | ---: | ---: |
| B1 | 330/330 | 30/30 | — |
| B2 | 90/90 | 10/10 | — |
| B3 | 120/120 | 15/15 | 60/60 |
| B4 | 30/30 | 15/15 | 30/30 |
| B5 | 30/30 | 10/10 | — |

Both B1 and B2 now compare ExLightNet with MLP, Random Forest and XGBoost
for binary classification. The multiclass comparisons use MLP,
standard_conv, Random Forest and XGBoost in both studies. B1 also includes
no_se and Std-CNN. Both checkpoints are reported. B1 produced 18 paired
tests and 18 paired flow bootstrap rows over 30 training seeds. B2 produced
14 of each over 10 matched split seeds. XGBoost has zero seed variance in
both B1 tasks.
[Paired tests](../artifacts/stats_gpu_v2/paired_tests.csv) ·
[Paired flow bootstrap](../artifacts/stats_gpu_v2/paired_flow_bootstrap.csv)

Seven B1 epoch-versus-macro-F1 Spearman correlations are defined. They cover
ExLightNet and MLP in both tasks, and multiclass no_se, standard_conv and
Std-CNN. Random Forest and XGBoost have no epoch counts, leaving four
correlation rows undefined. Six binary ExLightNet bootstrap draws had
constant epoch counts and were omitted, leaving 1,994 of 2,000. B2 has no
declared correlation analysis. No study seed blocks were excluded. The two
probes and two determinism repeats are infrastructure and remain outside
inference.
[Correlations](../artifacts/stats_gpu_v2/correlations.csv)

The descriptive `cpu_v1` versus `gpu_v2` view now has 11 rows on matched B1
split-42 seeds 42–46, the five seeds available in `cpu_v1`. It shows seven
CPU/CUDA neural rows and four CPU/CPU tree rows with differences and 95%
confidence intervals. B4 retains its cross-environment label. The only
training performed was the 160 authorized tree runs. Training tests,
benchmarking and B6 are **NOT RUN**.
[Device view](../artifacts/stats_gpu_v2/device_equivalence.csv) ·
[Statistics report](../REPORT_stats_v2.md)
