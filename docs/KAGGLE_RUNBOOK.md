# Run the supplementary accuracy studies on Kaggle

The local `cpu_v1` reproduction is already complete. These instructions create
a separate `gpu_v2` campaign; they do not replace it. The CUDA path has not been
executed on a GPU here. Local validation is the separately labelled CPU smoke.
There is no SHAP, latency, size, memory or INT8 measurement in this runner.

## Dated pre-data amendment — 2026-11-18

The original protocol remains recorded; PA-20261118-1 through
PA-20261118-6 in `docs/statistics_protocol.json` describe the new effective
rules. The stopping hypothesis compares **fixed30 last epoch** with **paper
best-validation** as its primary endpoint. Fixed30 best-validation is secondary.
The report shows both, applies the declared paired mean-CI and SD-ratio
bootstrap criteria, treats disagreement as inconclusive, and never claims
causality. Neither checkpoint is selected using test performance.

B1/B2/B3 are now seed-major, with the same per-run fingerprints. Every prefix
has complete task/model/arm blocks for all but possibly its final seed. Local
aggregation excludes incomplete predeclared study seed blocks from paired
inference and lists them as `incomplete; excluded` in `completeness.csv`.
Partial descriptive records are retained; no missing observation is filled in.

The separate local stratified SHAP job is a **final analysis**, outside this
hosted training runner. Old explanation files are retained. Optional B6 is
**NOT RUN**; do not describe the unexecuted clustering PROXY as campaign-disjoint
validation.

## 1. Prepare the two uploads locally

Use the checked Python 3.11 environment. The export uses only training rows to
fit each split's imputation, vocabulary, MI selection and scaler:

```bash
.venv/bin/python scripts/export_prepared.py --splits 42-51 --output prepared
```

The delivered workspace already contains the checked exports. Reuse them;
do not repeat an export merely to open a new notebook. `prepared/manifest.json`
records each file's checksum, recipe fingerprint, payload digest and supports.
Split 42 is checked against the original CPU cache. Alpha is frozen from that
CPU campaign for every split and stopping-rule arm.

Create two **private Kaggle datasets**, with these contents:

| Dataset | Upload | Do not upload |
| --- | --- | --- |
| `exlightnet-code` | Contents of `delivery/stats_gpu_v2/hosted_source.zip` | `.venv`, credentials, `.git`, locks, old checkpoints |
| `exlightnet-prepared` | `prepared/split_42.npz` through `split_51.npz`, plus `manifest.json` | Raw CSV, original data directory |

On Kaggle, open **Datasets**, choose **New Dataset**, upload the files and keep
visibility private. Kaggle may unpack a ZIP when creating a dataset. Preserve
`src/exlightnet/`, `scripts/` and `pyproject.toml` together in the code dataset.
Check that `manifest.json` is beside the prepared NPZ files, not in a nested
folder. Change the two dataset path variables in the notebook if your slugs or
folder structure differ. No account credentials are needed inside the code.

## 2. Create the notebook and attach inputs

Create a new notebook and import `notebooks/kaggle_runner.ipynb`. In its **Input**
pane use **Add Input** to attach both private datasets. Under **Settings** or
**Session options**, select an available GPU accelerator. Availability and
quotas vary; the runner uses only logical CUDA device 0 even if the selected
environment offers more than one GPU. Do not select TPU.

Internet may remain disabled. The notebook installs the uploaded source with
`--no-deps --no-build-isolation` into a dedicated working directory and uses the
preinstalled packages. Do **not** run the local `requirements.txt` installation
on Kaggle: its PyTorch pin is the canonical CPU wheel. Assumptions are Python
at least 3.11, setuptools, and preinstalled torch, NumPy, scikit-learn and
XGBoost. Their actual versions and GPU/CUDA/cuDNN identity are recorded; they
are not assumed to match the local CPU environment.

The setup cell checks these dependencies without downloading anything. If a
dependency is absent or the runtime is incompatible, stop and choose a
compatible environment; do not silently change the study recipe.

## 3. Run the probe first

Leave `STUDY = "probe"`, `DEVICE = "cuda"`, `CAMPAIGN = "gpu_v2"` in the
notebook. Run its setup and runner cells. The probe trains the declared
multiclass ExLightNet for three epochs, prints seconds/epoch and prints a rough
cost for each study **before** any long study can start.

Those projections assume the ExLightNet probe cost for neural epochs. They
exclude tree fitting and architecture-specific speed differences and ignore
early stopping; they are planning estimates, not measured study runtimes.
Use the printed projection and your available quota to decide which study to
launch. This repository supplies no invented GPU duration.

Outputs are written beneath `/kaggle/working/gpu_v2/`. The results bundle is
`/kaggle/working/gpu_v2.zip`. It contains scored records, predictions, receipts,
history, ExLightNet final weights and unfinished recovery state where needed;
it contains no prepared arrays or raw CSV.

## 4. Run studies in the declared priority order

Change `STUDY` and rerun the runner cell. Keep the campaign, prepared inputs and
environment fixed. The source of truth for all seeds/models/arms is
`exlightnet.campaign.study_runs`; the statistical protocol is
`docs/statistics_protocol.json`.

| Order | Study | Purpose |
| --- | --- | --- |
| First | `B1` | Training-seed variability on the fixed split |
| Second | `B3` | Fixed-budget and minimum-epoch stopping sensitivity; paper arm references B1 |
| Third | `B2` | Different train/validation/test split seeds |
| Fourth | `B4` | Binary uncapped versus B1's capped recipe |
| Fifth | `B5` | Std-CNN learning-rate sensitivity; selection uses validation only |

For a normal run, the command called by the notebook is:

```bash
python scripts/kaggle_campaign.py \
  --prepared-dir /kaggle/input/exlightnet-prepared \
  --output /kaggle/working --campaign gpu_v2 \
  --study B1 --device cuda --time-budget-hours 8 --zip-every 5
```

The time budget stops launching new runs; the current run may finish after that
limit. Completed identities are skipped. B3 does not retrain the paper-rule
reference: the aggregator looks up the matching B1 results. Missing references
remain visibly incomplete. Trees run on the notebook's CPU using the original
hyperparameters. Their paired accuracy comparisons with CUDA neural models are
labelled **mixed-device within gpu_v2; trees are device-independent**, with
separate device columns. No model's sample vector pools CPU and CUDA runs.

`STUDY = "determinism"` is a separate diagnostic: it trains the same declared
seed twice and records maximum absolute differences in macro-F1 and logits.
It does not establish bitwise equality with the canonical CPU campaign.

### Run neural models and trees in separate sessions

The new `--only` option accepts `all` (default), `neural` or `trees`. It selects
work without changing recipes, cache identities or seeds. `all` preserves the
combined workflow; no XGBoost seed is skipped because its results happen to be
deterministic.

For a GPU session, set `ONLY = "neural"` and `DEVICE = "cuda"` in the notebook.
Run its matching probe before the long neural study:

```bash
python scripts/kaggle_campaign.py --prepared-dir /kaggle/input/exlightnet-prepared \
  --output /kaggle/working --campaign gpu_v2 --study probe --device cuda --only neural
python scripts/kaggle_campaign.py --prepared-dir /kaggle/input/exlightnet-prepared \
  --output /kaggle/working --campaign gpu_v2 --study B1 --device cuda --only neural \
  --time-budget-hours 8
```

For a separate CPU-only session, select no GPU accelerator, set `ONLY = "trees"`
and `DEVICE = "cpu"`, attach the same prepared/code datasets and use:

```bash
python scripts/kaggle_campaign.py --prepared-dir /kaggle/input/exlightnet-prepared \
  --output /kaggle/working --campaign gpu_v2 --study B1 --device cpu --only trees \
  --time-budget-hours 8
```

Trees-only needs neither CUDA nor a neural probe; it runs every declared
RandomForest and XGBoost seed. Use `B2` similarly for its tree split study.
CPU-tree identity uses run-specific library provenance without the session's
GPU name. A compatible inherited tree cache retains its original fingerprint
and receipts; changing the selector or requested device cannot repeat its fit
or final evaluation. Conflicting legacy candidates stop for inspection.
Studies without tree plans must not create fake completed runs. Download the
neural and tree result ZIPs under distinct local filenames, such as
`gpu_v2_neural_B1.zip` and `gpu_v2_trees_B1.zip`, so one download cannot replace
the other. Pass both archives to `aggregate_gpu_v2.py`.

The aggregator merges verified records within the same `gpu_v2` campaign and
preserves per-record devices/environment identities. Comparisons carry the
explicit mixed-device label. A neural-only shard does not yet contain a full
matched study block when the corresponding trees are missing: those seeds
remain `incomplete; excluded` for inference until the other shard is imported.
Separate campaigns are still a hard error. Do not mix different actual GPU
models/runtime identities in one neural model's seed vector.

The local three-file handoff is `delivery/stats_gpu_v2/hosted_handoff.zip`.
`HOSTED_HANDOFF.md` and `HOSTED_HANDOFF.json` state each file's SHA256 and the
exact checks performed. Regenerate it together with the source bundle using
the `--hosted-verification` command in README. The Kaggle notebook and CUDA
path have never been executed locally; packaging/import checks are separate.

## 5. Save outputs and resume an interrupted session

During a live session use the Output/file pane to download `gpu_v2.zip`
regularly. A ZIP is replaced atomically after the configured number of
completed runs and at a normal/handled exit. Every completed epoch has an
atomic recovery checkpoint. A hard process kill cannot execute exit code;
recover from the latest surviving checkpoint or saved ZIP. Complete records
are never rescored. A claimed/incomplete evaluation receipt stops its branch
for investigation and is never reset to force another test pass.

**Save Version → Save & Run All** starts a fresh session, not a continuation of
the interactive working directory. Set the chosen study, restore bundle and
budget before doing that. Keep the environment/image fixed across resumes.
Saved notebook output can be attached as another notebook's input; a new
session cannot assume the previous `/kaggle/working` directory survived.

For a fresh session, upload the latest bundle as another private input dataset,
set `RESUME_ZIP` in the thin notebook, and use:

```bash
python scripts/kaggle_campaign.py \
  --prepared-dir /kaggle/input/exlightnet-prepared \
  --output /kaggle/working --campaign gpu_v2 \
  --study B1 --device cuda --time-budget-hours 8 \
  --resume-zip /kaggle/input/exlightnet-resume/gpu_v2.zip
```

Restore rejects unsafe paths, symlinks and conflicting files. Use an empty new
output directory when restoring an older snapshot that differs from existing
files. Do not delete receipts or edit fingerprints to bypass a conflict.

## 6. Download and aggregate locally

Download all incremental/final bundles you want to preserve and place them in
`imports/gpu_v2/` locally (ZIPs are ignored data artifacts). Identical completed
records deduplicate; conflicting records fail. Use the same local exported
prepared manifest:

```bash
.venv/bin/python scripts/aggregate_gpu_v2.py imports/gpu_v2/*.zip \
  --prepared-dir prepared --output artifacts/stats_gpu_v2 \
  --report REPORT_stats_v2.md --reference-cpu results/full/results.json
```

The aggregator verifies inputs and performs no training, checkpoint loading or
test inference. Its paired tables never combine campaigns. The separate
CPU/GPU equivalence view is descriptive: source values, differences and CIs,
with no p-value or pass/fail. Missing studies say `PENDING: study not run` and
have no filled-in metrics. The CPU smoke is a different campaign and cannot
populate the canonical study report.

Keep the original ZIPs, the import manifest/checksums and the generated tables.
After the new report and notebook are final, refresh deliverables using
`scripts/package_deliverables.py`. Do not upload or commit prepared NPZ arrays,
downloaded result ZIPs, credentials or process locks.

## Platform references

The interface names above follow [Kaggle's official notebook guide](https://www.kaggle.com/docs/notebooks),
including attached datasets, accelerator settings, working outputs and fresh
Save & Run All sessions. Menu locations and available accelerators can change.
Determinism limits follow [PyTorch's reproducibility notes](https://docs.pytorch.org/docs/stable/notes/randomness.html);
seeds and deterministic flags do not guarantee CPU/GPU bit equality.
