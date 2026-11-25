# Hosted runner review fixes

The immutable real-data smoke was not repeated or modified. No real training,
model inference, benchmark, or process control was executed here. Canonical
`TrainConfig` and `fit_model` remain unchanged. CUDA execution remains NOT RUN.

- R1: A tree's identity and record metadata are written before evaluation.
  A restored archive containing a completed, hash-verified evaluation can
  publish its record without the excluded tree pickle, fitting, prediction, or
  a second receipt; best and last refer to the same evidence. The pure fixture
  exercises package and fresh-directory restore and checks receipt bytes.
- R2: Explicit `RunStopped` errors cover receipt failures, invalid cached
  evidence, missing recovery state, and nonfinite supplementary training loss.
  The plan loop writes a durable stopped-branch record and continues independent
  plans; its result lists failures and the CLI exits nonzero after processing.
  Prepared data and environment failures remain fatal. A two-plan fixture puts
  a claimed receipt before a completed cached run and forbids model creation.
- R4: `device.environment_identity` includes CUDA GPU name and runtime versions
  for neural CUDA execution and excludes unused GPU fields for CPU execution.
  A long run requires a probe from the same identity; legacy probes lacking
  environment metadata are rejected. Existing record fingerprint recipes remain
  unchanged, and no recorded smoke evidence was modified.
- R5: epoch warnings carry epoch, phase, category and message into each atomic
  training checkpoint, and are merged when resuming legacy or new checkpoints.
  Device configuration warnings and effective precision/deterministic settings
  are captured in final records. An already-completed checkpoint fixture confirms
  warning merge without optimizer steps. CUDA warning behavior is untested.

RED commands: `pytest -q tests/test_campaign.py -k 'environment_identity or
warning_ledger'` produced 2 missing-API failures; `pytest -q
 tests/test_campaign.py -k 'tree_finalize or run_local_failures'` produced 2
missing-API failures. GREEN command, with OMP, OPENBLAS, MKL and NUMEXPR threads
set to 1: `.venv/bin/python -m pytest -q tests/test_campaign.py` produced
**25 passed in 2.58s**, exit 0. `git diff --check` produced no output, exit 0.
These focused tests used only synthetic files and checkpoint fixtures; they did
not perform an optimizer step. The full suite was not run by this agent.
