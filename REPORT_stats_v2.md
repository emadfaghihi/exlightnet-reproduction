# Supplementary accuracy studies: gpu_v2

Source protocol: [`docs/statistics_protocol.json`](docs/statistics_protocol.json). Rules were recorded before Stage B data. Headline: best-validation checkpoint; last epoch is secondary. No test metric selects an epoch, checkpoint, learning rate or arm.

## A-priori interpretation rules

- "Difference supported": the 95% CI of the paired difference excludes 0 AND Holm-adjusted p < 0.05.
- "Practically equivalent": the whole 95% CI lies within +/-0.01 macro-F1 (multiclass) or +/-0.002 macro-F1 (binary). These margins are conventions chosen in advance.
- Anything else: "inconclusive".
- For the early-stopping hypothesis, report the B1 correlation and the B3 variance/mean comparison side by side; do not declare causality unless fixed-budget (arm b) shows both a visibly smaller macro-F1 SD and a higher mean for ExLightNet; otherwise say "not supported".

Each replicate uniformly selects a training seed, then resamples test flows. This describes a randomly selected run plus finite-flow uncertainty, not the CI of the five-seed mean.

Fixed split/training-seed variation and split-seed variation are reported separately. Tiny-class support makes macro-F1 noisy; unstable controls and zero-variance baselines cannot establish superiority. n=5, indicative only. Correlations are descriptive, not causal.

Device label: no observations.

## B1: Training-seed study

PENDING: study not run

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
