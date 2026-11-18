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
