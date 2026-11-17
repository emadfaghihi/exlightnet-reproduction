# Parallel full reproduction and isolated benchmarking

The user's execution sequence is authoritative: one validation-only alpha sweep,
five parallel single-thread seed processes (42--46), one cache-only aggregation,
then one benchmark-only process after training/SHAP finish and CPU idle checks.
Keep existing train/test discipline and model recipes unchanged.

- [x] Add --skip-benchmark, --alpha-only, --seed-worker, --aggregate-only and
  --benchmark-only CLI modes. Workers read prepared data/alpha caches, write
  only their seed's artifacts and worker status, and never publish shared
  summaries/reports. Reject duplicate workers and benchmark/training overlap.
- [x] Add cache-only aggregation that refuses to train or run final test
  inference when any checkpoint/metric receipt is missing. Save all seed rows.
- [x] Add an isolated benchmark pass using only persisted models and training
  arrays, one process/thread, idle-CPU evidence, 100 warmups/1000 timings. Save
  benchmark observations separately and merge them into cached metrics.
- [x] Exclude every *.pkl file from the archive; retain local tree checkpoints
  for final benchmarking. Explain archive cache/benchmark limitations in README.
- [x] Compute W8 raw weight/scales/rest/metadata payloads and NPZ compressed
  members/container overhead. Render all report data as Markdown tables; use
  "consistent (94 - 1 constant = 93)" for candidate accounting.
- [x] Verify regression tests, run the full sequence, update the executed
  notebook/report, verify receipts and archive contents, rebuild deliverables.

Git remains unavailable. Record commands, timings, worker PIDs, gates, and any
stopped branches in results/full/ and docs/progress.md. Full experiments will
take sustained CPU time; continue through completion without new approvals.
