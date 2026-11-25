# Hosted runner implementation notes

`study_runs(study)` returns ordered dictionaries containing study, task, model,
seed, split_seed and arm. B1 has 330 runs; B2 has 90; B3 has 180
entries, of which 60 are references to B1 and 120 require new training; B4
has 30 new runs; B5 has 30. Probe has one and determinism has two runs. B4's
paper comparison is supplied by B1. Frozen alpha modes come from each prepared
file. `run_fingerprint(identity)` hashes canonical JSON; identity includes the
campaign, planned run, actual device, full TrainConfig, library/device versions,
historical data fingerprint, prepared payload SHA and split-file SHA. The verified prepared
manifest binds the input file SHA to those values. `budget_allows` stops new
runs after the budget; `projected_neural_hours` uses probe seconds per epoch
and the nominal epoch caps, ignoring tree time, model speed differences and early
stopping. `run_campaign` accepts prepared directory, output directory, study,
device, campaign, budget, zip interval and optional smoke controls.

Each completed record is `<campaign>/<actual-device>/<run-fingerprint>/record.json`
within the output directory; paths in each record are relative to the campaign
root (and zip root). Schema version 1 contains status, campaign, study, arm,
model, task, seed, split_seed, device, requested_device, config,
data_fingerprint, payload_sha256, prepared_sha256, run_fingerprint, libraries, warnings,
best_epoch, best_val_macro_f1, epochs_run, history, headline_checkpoint and
checkpoints. History rows carry epoch, train_loss, val_macro_f1, lr and seconds.
Each of checkpoints.best and checkpoints.last holds metrics, prediction_path,
prediction_sha256, metrics_path, metrics_sha256, receipt_path and receipt_sha256.
The compressed prediction NPZ contains int8 `y` and `predictions`. For
ExLightNet, each checkpoint additionally has state_dict_path and
state_dict_sha256; determinism runs additionally have logits_path and
logits_sha256. Trees have one trained checkpoint and best/last reference the
same evaluation evidence. A receipt transitions from exclusive claimed to
complete only after prediction and metrics exist; a claimed receipt stops the
branch. A complete record is validated against evidence hashes when resumed. Claimed receipts are checked before any resumed training; completed evaluations without their training checkpoint stop the unfinished branch.

The epoch checkpoint records model, best state, optimizer, scheduler, random
states, sampling generator, stale count, best epoch and history with the
identity. It is replaced atomically at epoch boundaries. The optional
`--pause-after-first-checkpoint` hook waits after the first completed checkpoint;
restart without the hook. A nonblocking advisory campaign lock prevents a
concurrent runner from using the same campaign. Safe zips exclude locks,
prepared data and pickle files and are written after every N completed runs and
on normal or exceptional exit; an abrupt kill does not execute final packaging.
Trees use an excluded temporary local pickle for mid-evaluation recovery; that
file is removed after a completed record. Incomplete epoch checkpoints may be
present in the archive to permit recovery; they are trusted only for resuming
training, never loaded by the aggregator.

Verification here used only nontraining tests and CLI help. Initial focused
run failed on the missing campaign module. Archive regressions failed
because arbitrary NPZ files and completed-run epoch checkpoints were included; the implementation was narrowed to
prediction and determinism-logit NPZ files. The final focused command
`env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/test_campaign.py`
reported **20 passed**. These tests cover study counts/reference arms, identity,
budget, sampler distribution against WeightedRandomSampler (absolute class
frequency tolerance 0.01), CPU selection, claimed/completed/corrupted receipts,
archive exclusions, campaign-name safety, prepared manifest binding, safe restore and CPU checkpoint isolation. No optimizer, training, smoke,
GPU, benchmark or full suite was run by this agent. CUDA execution and the
end-to-end interrupted smoke are NOT TESTED here and must be verified by the
controller. The canonical TrainConfig and fit_model implementation were not
changed; the supplementary path is a separate function.

## Integration review follow-up

A no-training fixture restored a completed epoch with distinct best and last
weights. Initially it failed because loading the best state mutated the returned
last state through CPU tensor aliasing. `fit_resumable` now clones the returned
last tensors; that fixture passes. The canonical `fit_model` was not changed.
`restore_campaign_archive(zip, output, campaign)` and repeatable `--resume-zip`
validate member paths and file types, reject symlinks and conflicting existing
files, stage data before installation, and hold the campaign lock. Identical
files are reusable; differing files stop restoration, including differing
in-progress checkpoints. The archive restore tests and the fixture perform no
optimization or model training.

Local preflight revealed the installed distribution is named `xgboost-cpu`;
the library-version lookup now falls back to that distribution when `xgboost`
is absent. The focused no-training regression passes, and the actual local
lookup reports XGBoost 2.1.4.
