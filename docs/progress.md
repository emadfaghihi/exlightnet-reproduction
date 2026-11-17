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
