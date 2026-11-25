# Data, artifacts and reproduction

## RT-IoT2022

Download RT-IoT2022 from the UCI Machine Learning Repository (dataset ID 942)
and place the genuine CSV at `data/RT_IOT2022.csv`. The training loader also
supports `ucimlrepo.fetch_ucirepo(id=942)` when the file is absent. The dataset
is distributed by UCI under Creative Commons Attribution 4.0 International
(CC BY 4.0). Obtain it from the source and keep the raw CSV out of the repository
and hosted-runner uploads. [UCI dataset and license](https://archive.ics.uci.edu/dataset/942/rt-iot2022) · [Dataset DOI](https://doi.org/10.24432/C5P338)

The canonical CSV fingerprint and preprocessing recipe are in
[`artifacts/cache_manifest.json`](../artifacts/cache_manifest.json). Use that
fingerprint to check compatibility before relying on the recorded split and
frozen alpha.

## Install and review a fresh clone

The CI recipe uses Python 3.11 and the pinned direct CPU dependencies. The
separate [host lock](../requirements-lock.txt) captures the recorded full
dependency set. On the original host, Python was 3.11.2, NumPy 1.26.4,
pandas 2.2.3, SciPy 1.14.1, PyTorch 2.6.0+cpu, scikit-learn 1.6.1,
XGBoost CPU 2.1.4 and SHAP 0.46.0. See the lock for the notebook and test
toolchain as well. [Recorded environment](../results/full/environment.json)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --disable-pip-version-check --retries 10 --timeout 60 -r requirements.txt
python -m pip install --retries 10 --timeout 60 --no-deps -e .

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="$(mktemp -d)"
python -m pytest -q -k 'not test_extracted_archive_can_rebuild_missing_array_cache'
sha256sum -c --ignore-missing docs/evidence/cpu_v1_SHA256SUMS
```

To install the recorded host dependency set, use `requirements-lock.txt` in the
first pip command. The CI test excludes archive recovery with the real CSV.
Dataset-dependent cases with their own guards skip when their inputs are absent.
[CI workflow](../.github/workflows/ci.yml) · [Checksum inventory](evidence/README.md)

Read the published reports and the executed CPU notebook directly from the
clone. To recompute CPU tables from the committed predictions and metrics,
write to a new directory. This reads the saved CPU results and needs no
CSV, array cache, model deserialization or hosted ZIPs.

```bash
review_out="$(mktemp -d)"
python scripts/make_stats.py --root . --output "$review_out/cpu-stats"
```

The full pipeline's `--aggregate-only --skip-benchmark` path is different. It
requires the original `artifacts/cache_arrays.npz`, compatible final-run caches
and the local tree checkpoints. A CPU benchmark-only run also needs the
persisted models and the prepared training inputs to time them. The clone
contains the measured benchmark evidence. [Local-only files](evidence/README.md#local-only-files)

## Supplementary archives

**External download link:** `<EXTERNAL_DOWNLOAD_LINK>`

Place these files under `imports/gpu_v2/`. The double extension in
`gpu_v2_part1.zip.zip` is intentional and preserves the recorded archive
identity. Git carries the final tables and the verification receipts. The input
ZIPs and prepared arrays are distributed separately from Git.

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| `gpu_v2_part1.zip.zip` | 37,330,267 | `6c346f558499ffbd1e5e8c9dce906dde36a44c35adad0cd3073bb5bc8467e482` |
| `gpu_v2_colab.zip` | 12,837,923 | `888fb932dd60c5bec22a681b73de91ffd04672ff6b5f3de4a4e167addc9891ec` |
| `gpu_v2_trees_b1.zip` | 1,330,015 | `3487c87246d43171cf74b398c4b9a3805df2a4734c1e9dbcab6b5c4b25e99259` |
| `gpu_v2_trees_b2.zip` | 442,716 | `68fad246b067ce6d69af105d16d405eaf0a11caeb8589426c9277506d722bb8b` |

Use the original `prepared/manifest.json` and the prepared split arrays
matching those archives. The manifest and the split payloads supply the
fingerprints and test labels the aggregator verifies. Obtain the exact bundle
from the maintainer for a byte-for-byte input match. [604-run provenance and input hashes](../artifacts/stats_gpu_v2/aggregation_manifest.json) · [Local tree receipts](../archive/gpu_v2_local_trees/20261123_PA-20261123-2/verified_imports.json)

With the matching inputs in place, regenerate the supplementary tables into a
new directory. This operation verifies and aggregates saved results on CPU.

```bash
review_out="$(mktemp -d)"
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -B scripts/aggregate_gpu_v2.py \
    imports/gpu_v2/gpu_v2_part1.zip.zip \
    imports/gpu_v2/gpu_v2_colab.zip \
    imports/gpu_v2/gpu_v2_trees_b1.zip \
    imports/gpu_v2/gpu_v2_trees_b2.zip \
    --prepared-dir prepared --output "$review_out/gpu-stats" \
    --report "$review_out/REPORT_stats_v2.md" --reference-cpu results/full/results.json
```

## Reproduce training in separate output directories

These commands are for a new reproduction with the same declared settings.
Use a separate experiment root to preserve the recorded CPU outputs.

```bash
cpu_root="$(mktemp -d)"
mkdir -p "$cpu_root/data"
cp data/RT_IOT2022.csv "$cpu_root/data/RT_IOT2022.csv"
python -m exlightnet.run_all --root "$cpu_root" --mode full --alpha-only --skip-benchmark
python scripts/run_parallel.py --root "$cpu_root" --reuse-alpha
```

The coordinator launches the five single-thread CPU workers after checking RAM,
then runs cache-only aggregation and the isolated CPU benchmark. Budget hours
on hardware comparable to the recorded Intel Core i5-8500. The alpha sweep
took 2,671.96 s. The resumed coordinator invocation took 13,637.78 s with
earlier results reused and the initial timing attempt included. The successful isolated benchmark
took 735.57 s in a separate invocation. These are measured stage timings from a
resumed campaign. [Alpha manifest](../results/full/alpha_sweep_manifest.json) · [Coordinator record](../results/full/parallel_execution.json) · [Benchmark manifest](../results/full/benchmark_manifest.json)

To export prepared arrays from a compatible canonical working copy, the
exporter requires the matching CSV, frozen alpha and split-42 array cache. Use
a fresh output directory and check the resulting fingerprints against the
archived prepared manifest before using existing hosted result ZIPs.

```bash
export_root="$(mktemp -d)"
python scripts/export_prepared.py --root . --splits 42-51 \
  --output "$export_root/prepared" --checks-output "$export_root/export_checks.json"
```

To reproduce the supplementary local CPU tree studies from the matching
prepared arrays, use separate roots and run B1 before B2.

```bash
tree_root="$(mktemp -d)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
for study in B1 B2; do
  python scripts/kaggle_campaign.py --prepared-dir prepared --study "$study" \
    --device cpu --only trees --campaign gpu_v2 --time-budget-hours 6 --zip-every 5 \
    --output "$tree_root/$study" || break
done
```

The recorded local runtimes were **2,575.12 s for B1 trees** and **862.90 s for
B2 trees**, sequentially on the same CPU host. Hosted neural studies use the
prepared arrays and the environment records described in the
[supplementary report](../REPORT_stats_v2.md). See the
[runbook](KAGGLE_RUNBOOK.md) for hosted execution and resume. [PA-20261123-2](progress.md#2026-11-23--pa-20261123-2-execution-and-verification)
