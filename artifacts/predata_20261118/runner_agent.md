# Supplementary runner verification

2026-11-18. Scope covers PA-20261118-2 and PA-20261118-3.

- B1, B2 and B3 now use seed, task, model and arm order. The tests visit all
  603 prefixes, including empty and full plans, against frozen plan declarations
  from `cbbea40`. They compare complete fingerprint multisets using the same
  data, libraries, device and config fixtures. B3 paper arms still reference
  B1. B4 and B5 retain exact ordered plans.
- `--only {all,neural,trees}` defaults to `all`. The selector preserves order
  and does not enter run identities. B1 includes 120 tree runs and B2 includes
  40 tree runs across both tasks. Both tree models keep every declared seed.
- Tree sessions require no neural timing probe. The runner forces actual CPU
  for a tree-only request of CUDA or auto and preserves the requested device in
  per-run provenance. Tests prohibit CUDA availability, device-name, seeding
  and initialization calls and prevent model fitting. Empty tree selections
  exit before device configuration, input loading or result packaging.
- Neural and default sessions validate a matching neural probe before long
  studies. Projections count the selected runs and estimate neural time.
  Cache, lock, receipt and archive handling remain on the existing paths.

## Verification

- Initial amendment tests on the prior runner exited 1 with 14 expected
  failures and 2 passes in 2.33 seconds. See `runner_red.log`.
- Final focused verification ran
  `rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLCONFIGDIR=/tmp/exlightnet-mpl .venv/bin/python -m pytest -q tests/test_campaign.py tests/test_plan_amendment.py tests/test_prepared.py`.
  It exited 0 with 61 passed in 7.00 seconds. See `runner_green.log`.
- The CLI help command exited 0 and displayed the selector and default.
  Scoped `git diff --check` on tracked code exited 0. The new test file had
  no whitespace diagnostics.
- The SHA-256 scan of `immutable_before.json` checked 1,047 recorded paths
  and found zero changes at the end of runner verification.

Actual Stage B execution, model fitting, hosted CUDA, smoke and benchmark
execution were not run in this task. The controller owns combined suite
verification and artifact publication.
