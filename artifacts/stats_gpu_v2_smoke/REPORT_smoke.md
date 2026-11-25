# Supplementary accuracy studies: gpu_v2_smoke

Source protocol: [`docs/statistics_protocol.json`](docs/statistics_protocol.json). Rules were recorded before Stage B data. Headline: best-validation checkpoint; last epoch is secondary. No test metric selects an epoch, checkpoint, learning rate or arm.

## A-priori interpretation rules

- "Difference supported": the 95% CI of the paired difference excludes 0 AND Holm-adjusted p < 0.05.
- "Practically equivalent": the whole 95% CI lies within +/-0.01 macro-F1 (multiclass) or +/-0.002 macro-F1 (binary). These margins are conventions chosen in advance.
- Anything else: "inconclusive".
- For the early-stopping hypothesis, report the B1 correlation and the B3 variance/mean comparison side by side; do not declare causality unless fixed-budget (arm b) shows both a visibly smaller macro-F1 SD and a higher mean for ExLightNet; otherwise say "not supported".

Each replicate uniformly selects a training seed, then resamples test flows. This describes a randomly selected run plus finite-flow uncertainty, not the CI of the five-seed mean.

Fixed split/training-seed variation and split-seed variation are reported separately. Tiny-class support makes macro-F1 noisy; unstable controls and zero-variance baselines cannot establish superiority. n=5, indicative only. Correlations are descriptive, not causal.

Device label: cpu.

**Separate smoke/supplementary campaign. These observations do not fill canonical gpu_v2 studies.**

## B1: Training-seed study

PARTIAL: missing planned runs or B1 references

| campaign | study | arm | task | model | checkpoint | device | device_label | n | sample_label | macro_f1_mean | macro_f1_sd | macro_f1_seed42 | zero_seed_variance | best_val_macro_f1_mean | epochs_run_mean | n_epoch_observed | n_hit_cap | unstable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu_v2_smoke | B1 | paper | multiclass | mlp | best | cpu | cpu | 2 | n=2 | 0.691394 | 0.0128873 | 0.700507 | False | 0.679474 | 2 | 2 | 2 | True |
| gpu_v2_smoke | B1 | paper | multiclass | mlp | last | cpu | cpu | 2 | n=2 | 0.692219 | 0.0117202 | 0.700507 | False | 0.679474 | 2 | 2 | 2 | True |

Descriptive epochs-versus-F1 correlation and paired-seed bootstrap CI:

| campaign | study | task | model | device | device_label | n | n_epoch_observed | spearman_rho | p_value | ci_low | ci_high | bootstrap_replicates | defined_replicates | bootstrap_unit | sample_label | interpretation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpu_v2_smoke | B1 | multiclass | mlp | cpu | cpu | 2 | 2 | — | — | — | — | 2000 | 0 | paired training-seed (epochs,F1) observations | n=2 | descriptive only, not causal |

## B3: Stopping-rule sensitivity

PENDING: study not run

## B2: Split-seed study

PENDING: study not run

## B4: Binary capped versus uncapped

PENDING: study not run

## B5: Std-CNN validation-selected learning rate

PENDING: study not run

## Device-equivalence view

Explicit side-by-side campaigns on matched seeds 42–46. Difference is gpu_v2 minus cpu_v1. No hypothesis p-values, pass/fail decisions or pooled campaigns.

PENDING: no verified observations

## Probe and determinism studies

PENDING: no verified observations

PENDING: no verified observations

CUDA execution is NOT RUN locally. Cached hosted GPU observations, if imported, do not imply local CUDA validation.

## Supplementary frozen/exported-model latency

NOT RUN: this task prohibits benchmark reruns. Canonical CPU latency/size/INT8/memory evidence stays separate.
