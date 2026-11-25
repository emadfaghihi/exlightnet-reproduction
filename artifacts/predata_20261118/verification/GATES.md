# Follow-up verification — 2026-11-18

| Gate | Status | Evidence |
| --- | --- | --- |
| PA-1 stopping endpoints | PASS | Original declaration preserved; both endpoint/threshold/degenerate/disagreement tests pass. Real B3 data: PENDING. |
| PA-2 seed-major and completeness | PASS | All 603 B1/B2/B3 plan prefixes and unchanged plan/fingerprint multisets tested; incomplete study blocks excluded in pure regressions. |
| PA-3 independent tree sessions | PASS | CPU-only selector, no CUDA/probe, all RF/XGB seeds, shard merge and legacy resume/claimed-receipt regressions pass. Actual hosted fitting: NOT RUN. |
| PA-4 three-file handoff | PASS | HOSTED_HANDOFF.json/.md bind each SHA and TESTED/NOT TESTED statement; offline install/import/help pass. Hosted notebook/CUDA: NEVER EXECUTED. |
| PA-5 stratified SHAP | PASS | CPU 259 flows, prior 200 backgrounds, 12 checked outputs, 20 random orders per class and k10 contrast; approximate additivity reported. All 1047 prior evidence hashes unchanged. |
| PA-6 leakage PROXY B6 | NOT RUN | Optional runner/export/study deferred. No campaign-disjoint validation claim. |

Full CPU suite: 253 passed, 24.96s, exit0.

Synthetic training: 4 cases, 5 fits, 7 epochs, 1.968s combined pytest case wall time. Real smoke: NOT RUN in this follow-up; historical two-seed/two-epoch evidence is unchanged.

RUN: 253 CPU tests; 4 tiny synthetic-training cases; CPU stratified SHAP/deletion final analysis; 10 prepared split validations; cached report notebook, fresh-kernel restart; offline source install/import/help.

NOT RUN: new real-data smoke/training; benchmark; CUDA; Kaggle notebook execution; real B1-B5; optional B6.

Per-file SHA and execution limits: delivery/stats_gpu_v2/HOSTED_HANDOFF.md.
