# ExLightNet reproduction

This is an independent reproduction of the ExLightNet intrusion-detection study
on RT-IoT2022. The published paper is open access under the Creative Commons
Attribution (CC BY) license, as stated in the included MDPI PDF.
[Paper DOI](https://doi.org/10.3390/s26217023) · [Paper PDF](paper/exlightnet.pdf)

## Start here

1. This README.
2. The [Persian final report (complete PDF)](docs/FINAL_REPORT_fa.pdf). The
   [Markdown edition](docs/FINAL_REPORT_fa.md) contains the text and tables.
   The chart is in the PDF.
3. The [canonical CPU report](REPORT.md).
4. The [supplementary statistics report](REPORT_stats_v2.md).
5. The [experiment history and amendments](docs/progress.md).

## Findings

**Canonical `cpu_v1`.** ExLightNet's binary macro-F1 was **0.987069 ±
0.000850**, close to the reported **0.986**. Multiclass macro-F1 was **0.846598
± 0.019463**. The paper quotes **0.8621** for its best run and **0.835** as its
seed mean. Values from this reproduction are mean ± sample SD, *n=5,
indicative only*. Two rare classes have just 4 and 5 test flows. W8 passed the
recorded accuracy gates and shrank the serialized model. W8 inference
dequantizes the weights and computes in FP32. [CPU results and limits](REPORT.md#full-cpu-reproduction)

**Supplementary `gpu_v2`.** B1 found practical equivalence with the binary
MLP at the best-validation checkpoint. The two tree baselines led binary
ExLightNet in B1 and B2 and led it in the B2 multiclass split study. The B1
multiclass comparisons with no SE and standard convolution were inconclusive.
The stopping hypothesis was inconclusive, and B4's cap comparison spans two
CUDA environments. The paired CUDA-neural/CPU-tree comparisons belong to
`gpu_v2` only. The two campaigns are never pooled and their device comparison
is descriptive. [Supplementary results](REPORT_stats_v2.md)

**Persian report in brief.** The report explains the partial reproduction,
the differences in model comparisons and SHAP deletion results, and the
limits of using one randomly split dataset. These results leave architecture
superiority, campaign-disjoint generalization, SHAP faithfulness and the
paper's hardware speed ratios unresolved. [Persian report](docs/FINAL_REPORT_fa.pdf)

**NOT RUN.** B6, Raspberry Pi measurements, exported-model latency and real
INT8 compute. Recorded CPU latency uses PyTorch eager inference and includes
framework overhead. [CPU limitations](REPORT.md#limitations) · [Supplementary limits](REPORT_stats_v2.md#optional-b6-leakage-proxy)

## Review and reproduce

The clone includes the published results and the executed CPU notebook.
The data guide covers setup, cached statistics, raw data, hosted archives and
exact reproduction commands. Historical evidence preserves the original
maintainer path `/home/mparsict` for provenance. Run the review commands
from your checkout root. [Data and artifacts](docs/DATA_AND_ARTIFACTS.md) · [Verify the CPU evidence](docs/evidence/README.md)

The Kaggle and Colab notebooks are thin runners for
`scripts/kaggle_campaign.py`. The Colab notebook covers B2, B5 and B4 neural
runs. It was added to Git after those runs and was **NOT re-executed** during
this review. Hosted evidence is in the final aggregation manifest's
environment records. Kaggle used a Tesla T4 with PyTorch 2.11.0+cu130 and
NumPy 2.1.3. Colab used a Tesla T4 with PyTorch 2.14.1+cu130 and NumPy 2.4.6.
[Kaggle runner](notebooks/kaggle_runner.ipynb) · [Colab runner](notebooks/exlightnet_colab.ipynb) · [Execution environments](artifacts/stats_gpu_v2/aggregation_manifest.json)

## Repository map

`src/`, `scripts/` and `tests/` hold the implementation, experiment entry
points and validation. `results/full/` is canonical CPU evidence.
`artifacts/stats_gpu_v2/` is the final supplementary aggregation of **604
runs**. The similarly named `stats_gpu_v2_final/` holds a historical predata
snapshot. [Aggregation manifest](artifacts/stats_gpu_v2/aggregation_manifest.json)

<details>
<summary>Complete directory map</summary>

The map includes ignored directories seen in the maintainer checkout. Directory
names inside smoke outputs are recorded run identities.

- `.agents/` — smoke-or-verification-only — local agent configuration, ignored
- `.aws/` — smoke-or-verification-only — local service configuration, ignored
- `.codex/` — smoke-or-verification-only — local agent configuration, ignored
- `.git/` — canonical — local Git history and checkout metadata
- `.github/` — smoke-or-verification-only — dataset-independent CI workflow
- `.kilo/` — smoke-or-verification-only — local agent tooling, ignored
- `.pytest_cache/` — smoke-or-verification-only — local pytest cache, ignored
- `.venv/` — smoke-or-verification-only — local Python environment, ignored
- `archive/` — logs — historical records and receipts, with selected receipts in Git
- `artifacts/` — canonical, supplementary-final, smoke-or-verification-only, predata — fitted state and analysis for the directories below
- `build/` — smoke-or-verification-only — local build output, ignored
- `data/` — canonical — source CSV for the reproduction, ignored
- `delivery/` — logs — historical packaging and hosted handoff, ignored
- `docs/` — canonical, supplementary-final — reports, protocol, amendments and evidence guide
- `imports/` — supplementary-final — input result archives, ignored
- `notebooks/` — canonical, supplementary-final — executed CPU review notebook and hosted launchers
- `paper/` — canonical — included open-access reference paper
- `prepared/` — supplementary-final — prepared arrays and manifest for hosted studies, ignored
- `results/` — canonical, smoke-or-verification-only, logs — recorded full campaign and separate fast and recovery paths below
- `scripts/` — canonical, supplementary-final — experiment and review entry points
- `src/` — canonical, supplementary-final — shared implementation
- `tests/` — smoke-or-verification-only — invariant, orchestration and presentation coverage

- `artifacts/final_analysis/` — supplementary-final — stratified SHAP follow-up to the CPU campaign
- `artifacts/final_analysis/stratified_shap_20261118/` — supplementary-final — completed stratified CPU SHAP and deletion analysis
- `artifacts/final_analysis/stratified_shap_20261118_attempt1_interrupted/` — logs — interrupted SHAP attempt metadata
- `artifacts/predata_20261118/` — predata — amendments, staged analysis and validation before hosted results
- `artifacts/predata_20261118/pending_studies/` — predata — empty study tables and pending status
- `artifacts/predata_20261118/review/` — smoke-or-verification-only — historical source review and test logs
- `artifacts/predata_20261118/verification/` — smoke-or-verification-only — historical gates, notebook and hosted-runner checks
- `artifacts/predata_20261118/verification/notebook_first/` — smoke-or-verification-only
- `artifacts/predata_20261118/verification/notebook_restart/` — smoke-or-verification-only
- `artifacts/stats_cpu_v1/` — canonical — cached CPU statistical tables and their manifest
- `artifacts/stats_cpu_v1/review/` — smoke-or-verification-only — source/aggregator review records
- `artifacts/stats_cpu_v1/smoke/` — smoke-or-verification-only — local CPU smoke, interruption and resume evidence
- `artifacts/stats_cpu_v1/smoke/first_attempt/` — smoke-or-verification-only — local only
- `artifacts/stats_cpu_v1/smoke/first_attempt/gpu_v2_smoke/` — smoke-or-verification-only — local only
- `artifacts/stats_cpu_v1/smoke/first_attempt/gpu_v2_smoke/cpu/` — smoke-or-verification-only — local only
- `artifacts/stats_cpu_v1/smoke/first_attempt/gpu_v2_smoke/cpu/ca5364c9584721e42f13407f86027e260ecd4b91ed960a1b5c367c9c8469d17a/` — smoke-or-verification-only — local only
- `artifacts/stats_cpu_v1/smoke/resumed/` — smoke-or-verification-only
- `artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/` — smoke-or-verification-only
- `artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/cpu/` — smoke-or-verification-only
- `artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/cpu/36a31c291e1cdb34c4ac0bad7830ad8324f8eb81301a64ce5fc85a3584592081/` — smoke-or-verification-only
- `artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/cpu/36a31c291e1cdb34c4ac0bad7830ad8324f8eb81301a64ce5fc85a3584592081/best/` — smoke-or-verification-only — checkpoint evaluation evidence
- `artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/cpu/36a31c291e1cdb34c4ac0bad7830ad8324f8eb81301a64ce5fc85a3584592081/last/` — smoke-or-verification-only — checkpoint evaluation evidence
- `artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/cpu/ca5364c9584721e42f13407f86027e260ecd4b91ed960a1b5c367c9c8469d17a/` — smoke-or-verification-only
- `artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/cpu/ca5364c9584721e42f13407f86027e260ecd4b91ed960a1b5c367c9c8469d17a/best/` — smoke-or-verification-only — checkpoint evaluation evidence
- `artifacts/stats_cpu_v1/smoke/resumed/gpu_v2_smoke/cpu/ca5364c9584721e42f13407f86027e260ecd4b91ed960a1b5c367c9c8469d17a/last/` — smoke-or-verification-only — checkpoint evaluation evidence
- `artifacts/stats_cpu_v1/verification/` — smoke-or-verification-only — local tests, gates and notebook receipts
- `artifacts/stats_cpu_v1/verification/final/` — smoke-or-verification-only
- `artifacts/stats_cpu_v1/verification/final/notebook_restart/` — smoke-or-verification-only
- `artifacts/stats_cpu_v1/verification/final/previous_suite/` — smoke-or-verification-only
- `artifacts/stats_cpu_v1/verification/notebook_first/` — smoke-or-verification-only
- `artifacts/stats_cpu_v1/verification/publication_synthetic_training/` — smoke-or-verification-only
- `artifacts/stats_gpu_v2/` — supplementary-final — final 604-run aggregation and manuscript tables
- `artifacts/stats_gpu_v2_final/` — predata — historical pending deliverable with zero imported runs
- `artifacts/stats_gpu_v2_smoke/` — smoke-or-verification-only — two local CPU smoke records and their report
- `artifacts/stats_gpu_v2_smoke_verified/` — smoke-or-verification-only — verified aggregation of the same local CPU smoke archive
- `results/.locks/` — logs — local coordinator locks, ignored
- `results/fast/` — smoke-or-verification-only — historical fast demonstration
- `results/fast/alpha/` — smoke-or-verification-only
- `results/fast/alpha/binary/` — smoke-or-verification-only
- `results/fast/alpha/binary/effective_num/` — smoke-or-verification-only
- `results/fast/alpha/binary/inv_freq/` — smoke-or-verification-only
- `results/fast/alpha/binary/none/` — smoke-or-verification-only
- `results/fast/alpha/binary/sqrt_inv_freq/` — smoke-or-verification-only
- `results/fast/alpha/multiclass/` — smoke-or-verification-only
- `results/fast/alpha/multiclass/effective_num/` — smoke-or-verification-only
- `results/fast/alpha/multiclass/inv_freq/` — smoke-or-verification-only
- `results/fast/alpha/multiclass/none/` — smoke-or-verification-only
- `results/fast/alpha/multiclass/sqrt_inv_freq/` — smoke-or-verification-only
- `results/fast/binary/` — smoke-or-verification-only
- `results/fast/binary/bilstm/` — smoke-or-verification-only
- `results/fast/binary/bilstm/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/binary/exlightnet/` — smoke-or-verification-only
- `results/fast/binary/exlightnet/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/binary/exlightnet_w8/` — smoke-or-verification-only
- `results/fast/binary/exlightnet_w8/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/binary/mlp/` — smoke-or-verification-only
- `results/fast/binary/mlp/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/binary/random_forest/` — smoke-or-verification-only
- `results/fast/binary/random_forest/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/binary/std_cnn/` — smoke-or-verification-only
- `results/fast/binary/std_cnn/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/binary/xgboost/` — smoke-or-verification-only
- `results/fast/binary/xgboost/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/explain/` — smoke-or-verification-only
- `results/fast/multiclass/` — smoke-or-verification-only
- `results/fast/multiclass/balanced/` — smoke-or-verification-only
- `results/fast/multiclass/balanced/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/bilstm/` — smoke-or-verification-only
- `results/fast/multiclass/bilstm/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/exlightnet/` — smoke-or-verification-only
- `results/fast/multiclass/exlightnet/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/exlightnet_w8/` — smoke-or-verification-only
- `results/fast/multiclass/exlightnet_w8/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/mlp/` — smoke-or-verification-only
- `results/fast/multiclass/mlp/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/no_sampler/` — smoke-or-verification-only
- `results/fast/multiclass/no_sampler/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/no_se/` — smoke-or-verification-only
- `results/fast/multiclass/no_se/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/random_forest/` — smoke-or-verification-only
- `results/fast/multiclass/random_forest/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/standard_conv/` — smoke-or-verification-only
- `results/fast/multiclass/standard_conv/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/std_cnn/` — smoke-or-verification-only
- `results/fast/multiclass/std_cnn/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/top20/` — smoke-or-verification-only
- `results/fast/multiclass/top20/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/multiclass/xgboost/` — smoke-or-verification-only
- `results/fast/multiclass/xgboost/42/` — smoke-or-verification-only — recorded seed evidence
- `results/fast/plots/` — smoke-or-verification-only
- `results/fast/workers/` — smoke-or-verification-only
- `results/fast/workers/42/` — smoke-or-verification-only
- `results/fast_calibration/` — smoke-or-verification-only — partial fast alpha calibration
- `results/fast_calibration/alpha/` — smoke-or-verification-only
- `results/fast_calibration/alpha/binary/` — smoke-or-verification-only
- `results/fast_calibration/alpha/binary/effective_num/` — smoke-or-verification-only
- `results/fast_calibration/alpha/binary/inv_freq/` — smoke-or-verification-only
- `results/fast_calibration/alpha/binary/sqrt_inv_freq/` — smoke-or-verification-only
- `results/full/` — canonical — completed full CPU campaign
- `results/full/.locks/` — logs — local full-mode pipeline and benchmark locks, ignored
- `results/full/alpha/` — canonical — frozen-alpha validation sweeps
- `results/full/alpha/binary/` — canonical
- `results/full/alpha/binary/effective_num/` — canonical
- `results/full/alpha/binary/inv_freq/` — canonical
- `results/full/alpha/binary/none/` — canonical
- `results/full/alpha/binary/sqrt_inv_freq/` — canonical
- `results/full/alpha/multiclass/` — canonical
- `results/full/alpha/multiclass/effective_num/` — canonical
- `results/full/alpha/multiclass/inv_freq/` — canonical
- `results/full/alpha/multiclass/none/` — canonical
- `results/full/alpha/multiclass/sqrt_inv_freq/` — canonical
- `results/full/benchmarks/` — canonical, logs — measured campaign and the separate idle-gate attempts below
- `results/full/benchmarks/20261116T151119336040-1010228/` — logs — idle-CPU gate stopped this timing attempt before measurement
- `results/full/benchmarks/20261117T051732374195-1034809/` — logs — idle-CPU gate stopped this timing attempt before measurement
- `results/full/benchmarks/20261117T052645467310-1050022/` — canonical — completed isolated CPU measurements
- `results/full/binary/` — canonical — binary per-model and per-seed evidence
- `results/full/binary/bilstm/` — canonical
- `results/full/binary/bilstm/42/` — canonical — recorded seed evidence
- `results/full/binary/bilstm/43/` — canonical — recorded seed evidence
- `results/full/binary/bilstm/44/` — canonical — recorded seed evidence
- `results/full/binary/bilstm/45/` — canonical — recorded seed evidence
- `results/full/binary/bilstm/46/` — canonical — recorded seed evidence
- `results/full/binary/exlightnet/` — canonical
- `results/full/binary/exlightnet/42/` — canonical — recorded seed evidence
- `results/full/binary/exlightnet/43/` — canonical — recorded seed evidence
- `results/full/binary/exlightnet/44/` — canonical — recorded seed evidence
- `results/full/binary/exlightnet/45/` — canonical — recorded seed evidence
- `results/full/binary/exlightnet/46/` — canonical — recorded seed evidence
- `results/full/binary/exlightnet_w8/` — canonical
- `results/full/binary/exlightnet_w8/42/` — canonical — recorded seed evidence
- `results/full/binary/exlightnet_w8/43/` — canonical — recorded seed evidence
- `results/full/binary/exlightnet_w8/44/` — canonical — recorded seed evidence
- `results/full/binary/exlightnet_w8/45/` — canonical — recorded seed evidence
- `results/full/binary/exlightnet_w8/46/` — canonical — recorded seed evidence
- `results/full/binary/mlp/` — canonical
- `results/full/binary/mlp/42/` — canonical — recorded seed evidence
- `results/full/binary/mlp/43/` — canonical — recorded seed evidence
- `results/full/binary/mlp/44/` — canonical — recorded seed evidence
- `results/full/binary/mlp/45/` — canonical — recorded seed evidence
- `results/full/binary/mlp/46/` — canonical — recorded seed evidence
- `results/full/binary/random_forest/` — canonical
- `results/full/binary/random_forest/42/` — canonical — recorded seed evidence
- `results/full/binary/random_forest/43/` — canonical — recorded seed evidence
- `results/full/binary/random_forest/44/` — canonical — recorded seed evidence
- `results/full/binary/random_forest/45/` — canonical — recorded seed evidence
- `results/full/binary/random_forest/46/` — canonical — recorded seed evidence
- `results/full/binary/std_cnn/` — canonical
- `results/full/binary/std_cnn/42/` — canonical — recorded seed evidence
- `results/full/binary/std_cnn/43/` — canonical — recorded seed evidence
- `results/full/binary/std_cnn/44/` — canonical — recorded seed evidence
- `results/full/binary/std_cnn/45/` — canonical — recorded seed evidence
- `results/full/binary/std_cnn/46/` — canonical — recorded seed evidence
- `results/full/binary/xgboost/` — canonical
- `results/full/binary/xgboost/42/` — canonical — recorded seed evidence
- `results/full/binary/xgboost/43/` — canonical — recorded seed evidence
- `results/full/binary/xgboost/44/` — canonical — recorded seed evidence
- `results/full/binary/xgboost/45/` — canonical — recorded seed evidence
- `results/full/binary/xgboost/46/` — canonical — recorded seed evidence
- `results/full/explain/` — canonical — original random-sample SHAP and deletion artifacts
- `results/full/interrupted/` — logs — archived interrupted coordinator and seed-run state
- `results/full/interrupted/20261116T112858/` — logs
- `results/full/interrupted/20261116T112858/binary/` — logs
- `results/full/interrupted/20261116T112858/binary/bilstm/` — logs
- `results/full/interrupted/20261116T112858/binary/bilstm/42/` — logs
- `results/full/interrupted/20261116T112858/binary/bilstm/43/` — logs
- `results/full/interrupted/20261116T112858/binary/bilstm/44/` — logs
- `results/full/interrupted/20261116T112858/binary/bilstm/45/` — logs
- `results/full/interrupted/20261116T112858/binary/bilstm/46/` — logs
- `results/full/logs/` — logs — full-mode worker, aggregation and benchmark transcripts
- `results/full/multiclass/` — canonical — multiclass per-model and per-seed evidence
- `results/full/multiclass/balanced/` — canonical
- `results/full/multiclass/balanced/42/` — canonical — recorded seed evidence
- `results/full/multiclass/balanced/43/` — canonical — recorded seed evidence
- `results/full/multiclass/balanced/44/` — canonical — recorded seed evidence
- `results/full/multiclass/balanced/45/` — canonical — recorded seed evidence
- `results/full/multiclass/balanced/46/` — canonical — recorded seed evidence
- `results/full/multiclass/bilstm/` — canonical
- `results/full/multiclass/bilstm/42/` — canonical — recorded seed evidence
- `results/full/multiclass/bilstm/43/` — canonical — recorded seed evidence
- `results/full/multiclass/bilstm/44/` — canonical — recorded seed evidence
- `results/full/multiclass/bilstm/45/` — canonical — recorded seed evidence
- `results/full/multiclass/bilstm/46/` — canonical — recorded seed evidence
- `results/full/multiclass/exlightnet/` — canonical
- `results/full/multiclass/exlightnet/42/` — canonical — recorded seed evidence
- `results/full/multiclass/exlightnet/43/` — canonical — recorded seed evidence
- `results/full/multiclass/exlightnet/44/` — canonical — recorded seed evidence
- `results/full/multiclass/exlightnet/45/` — canonical — recorded seed evidence
- `results/full/multiclass/exlightnet/46/` — canonical — recorded seed evidence
- `results/full/multiclass/exlightnet_w8/` — canonical
- `results/full/multiclass/exlightnet_w8/42/` — canonical — recorded seed evidence
- `results/full/multiclass/exlightnet_w8/43/` — canonical — recorded seed evidence
- `results/full/multiclass/exlightnet_w8/44/` — canonical — recorded seed evidence
- `results/full/multiclass/exlightnet_w8/45/` — canonical — recorded seed evidence
- `results/full/multiclass/exlightnet_w8/46/` — canonical — recorded seed evidence
- `results/full/multiclass/mlp/` — canonical
- `results/full/multiclass/mlp/42/` — canonical — recorded seed evidence
- `results/full/multiclass/mlp/43/` — canonical — recorded seed evidence
- `results/full/multiclass/mlp/44/` — canonical — recorded seed evidence
- `results/full/multiclass/mlp/45/` — canonical — recorded seed evidence
- `results/full/multiclass/mlp/46/` — canonical — recorded seed evidence
- `results/full/multiclass/no_sampler/` — canonical
- `results/full/multiclass/no_sampler/42/` — canonical — recorded seed evidence
- `results/full/multiclass/no_sampler/43/` — canonical — recorded seed evidence
- `results/full/multiclass/no_sampler/44/` — canonical — recorded seed evidence
- `results/full/multiclass/no_sampler/45/` — canonical — recorded seed evidence
- `results/full/multiclass/no_sampler/46/` — canonical — recorded seed evidence
- `results/full/multiclass/no_se/` — canonical
- `results/full/multiclass/no_se/42/` — canonical — recorded seed evidence
- `results/full/multiclass/no_se/43/` — canonical — recorded seed evidence
- `results/full/multiclass/no_se/44/` — canonical — recorded seed evidence
- `results/full/multiclass/no_se/45/` — canonical — recorded seed evidence
- `results/full/multiclass/no_se/46/` — canonical — recorded seed evidence
- `results/full/multiclass/random_forest/` — canonical
- `results/full/multiclass/random_forest/42/` — canonical — recorded seed evidence
- `results/full/multiclass/random_forest/43/` — canonical — recorded seed evidence
- `results/full/multiclass/random_forest/44/` — canonical — recorded seed evidence
- `results/full/multiclass/random_forest/45/` — canonical — recorded seed evidence
- `results/full/multiclass/random_forest/46/` — canonical — recorded seed evidence
- `results/full/multiclass/standard_conv/` — canonical
- `results/full/multiclass/standard_conv/42/` — canonical — recorded seed evidence
- `results/full/multiclass/standard_conv/43/` — canonical — recorded seed evidence
- `results/full/multiclass/standard_conv/44/` — canonical — recorded seed evidence
- `results/full/multiclass/standard_conv/45/` — canonical — recorded seed evidence
- `results/full/multiclass/standard_conv/46/` — canonical — recorded seed evidence
- `results/full/multiclass/std_cnn/` — canonical
- `results/full/multiclass/std_cnn/42/` — canonical — recorded seed evidence
- `results/full/multiclass/std_cnn/43/` — canonical — recorded seed evidence
- `results/full/multiclass/std_cnn/44/` — canonical — recorded seed evidence
- `results/full/multiclass/std_cnn/45/` — canonical — recorded seed evidence
- `results/full/multiclass/std_cnn/46/` — canonical — recorded seed evidence
- `results/full/multiclass/top20/` — canonical
- `results/full/multiclass/top20/42/` — canonical — recorded seed evidence
- `results/full/multiclass/top20/43/` — canonical — recorded seed evidence
- `results/full/multiclass/top20/44/` — canonical — recorded seed evidence
- `results/full/multiclass/top20/45/` — canonical — recorded seed evidence
- `results/full/multiclass/top20/46/` — canonical — recorded seed evidence
- `results/full/multiclass/xgboost/` — canonical
- `results/full/multiclass/xgboost/42/` — canonical — recorded seed evidence
- `results/full/multiclass/xgboost/43/` — canonical — recorded seed evidence
- `results/full/multiclass/xgboost/44/` — canonical — recorded seed evidence
- `results/full/multiclass/xgboost/45/` — canonical — recorded seed evidence
- `results/full/multiclass/xgboost/46/` — canonical — recorded seed evidence
- `results/full/plots/` — canonical — canonical confusion and training figures
- `results/full/workers/` — canonical — per-seed worker manifests and gates
- `results/full/workers/42/` — canonical
- `results/full/workers/43/` — canonical
- `results/full/workers/44/` — canonical
- `results/full/workers/45/` — canonical
- `results/full/workers/46/` — canonical

</details>

## Open handoff items

A repository license decision and an external link for the supplementary ZIPs
are pending. [Archive inventory and link placeholder](docs/DATA_AND_ARTIFACTS.md#supplementary-archives)
