# Supplementary accuracy studies: gpu_v2

Source protocol: [`docs/statistics_protocol.json`](docs/statistics_protocol.json). Rules were recorded before Stage B data. General study summaries headline the best-validation checkpoint and list the last epoch second. The stopping hypothesis follows its designated primary and secondary checkpoints below. No test metric selects an epoch, checkpoint, learning rate or arm.

## A-priori interpretation rules

- "Difference supported": the 95% CI of the paired difference excludes 0 AND Holm-adjusted p < 0.05.
- "Practically equivalent": the whole 95% CI lies within +/-0.01 macro-F1 (multiclass) or +/-0.002 macro-F1 (binary). These margins are conventions chosen in advance.
- Anything else: "inconclusive".

### PA-20261118-1 (2026-11-18) — Stopping hypothesis

- PRIMARY stopping-hypothesis endpoint: fixed30 LAST-epoch macro-F1 versus the paper arm's best-validation checkpoint. The fixed30 best-validation checkpoint is SECONDARY; report both.
- Visibly smaller SD requires fixed30 sample SD <= 0.75 x paper sample SD AND the upper end of the 95% bootstrap CI of the SD ratio is < 1, using 2000 resamples over matched seeds 42-56 and fixed random seed 42.
- Higher mean requires the paired 95% CI of (fixed30 - paper) macro-F1 to exclude 0 in the positive direction.
- If primary and secondary disagree, report inconclusive. Undefined SD ratios or insufficient matched observations cannot support the hypothesis. Never claim causality.

### PA-20261118-2 (2026-11-18) — Complete seed blocks

Seed-major order: seed, task, model, arm. For every possible prefix all seeds except at most its last seed have complete matched task/model/arm blocks. Run identities and fingerprints remain unchanged.

User confirmed every-prefix testing. Statistical inference excludes seeds incomplete for the predeclared comparison block; completeness.csv explicitly records incomplete; excluded. Never interpolate or substitute missing observations.

### Superseded original stopping-hypothesis rule

For the early-stopping hypothesis, report the B1 correlation and the B3 variance/mean comparison side by side; do not declare causality unless fixed-budget (arm b) shows both a visibly smaller macro-F1 SD and a higher mean for ExLightNet; otherwise say "not supported".

Each replicate uniformly selects a training seed, then resamples test flows. This describes a randomly selected run plus finite-flow uncertainty, not the CI of the five-seed mean.

Fixed split/training-seed variation and split-seed variation are reported separately. Tiny-class support makes macro-F1 noisy; unstable controls and zero-variance baselines cannot establish superiority. n=5, indicative only. Correlations are descriptive, not causal.

Device label: no observations.

## B1: Training-seed study

PENDING: study not run

Complete seed blocks for paired analyses: 0 of 30. Eligible seeds: none. Missing run identities are recorded by seed in `completeness.csv`.

## B3: Stopping-rule sensitivity

PENDING: study not run

Complete seed blocks for paired analyses: 0 of 15. Eligible seeds: none. Missing run identities are recorded by seed in `completeness.csv`.

PENDING: no verified observations

B1 supplies the paper-rule checkpoint for each seed. The mean-difference interval uses the paired Student t distribution. The SD-ratio interval uses paired-seed percentile resampling.

Overall stopping-hypothesis decision — **PENDING**.

| endpoint_role | experimental_arm | experimental_checkpoint | reference_arm | reference_checkpoint | n_pairs | missing_seeds | smaller_sd | higher_mean | endpoint_decision | decision_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PRIMARY | fixed30 | last | paper | best | 0 | 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56 | — | — | PENDING | all declared matched seed blocks are required |
| SECONDARY | fixed30 | best | paper | best | 0 | 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56 | — | — | PENDING | all declared matched seed blocks are required |

| endpoint_role | paper_mean | paper_sd | fixed30_mean | fixed30_sd | mean_difference | mean_difference_ci_low | mean_difference_ci_high | sd_ratio | sd_ratio_ci_low | sd_ratio_ci_high | defined_replicates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PRIMARY | — | — | — | — | — | — | — | — | — | — | 0 |
| SECONDARY | — | — | — | — | — | — | — | — | — | — | 0 |

B1 descriptive epochs-versus-F1 correlation for multiclass ExLightNet and its paired-seed bootstrap interval.

| b1_n | b1_spearman_rho | b1_ci_low | b1_ci_high |
| --- | --- | --- | --- |
| 0 | — | — | — |

## B2: Split-seed study

PENDING: study not run

Complete seed blocks for paired analyses: 0 of 10. Eligible seeds: none. Missing run identities are recorded by seed in `completeness.csv`.

## B4: Binary capped versus uncapped

PENDING: study not run

Complete seed blocks for paired analyses: 0 of 15. Eligible seeds: none. Missing run identities are recorded by seed in `completeness.csv`.

## B5: Std-CNN validation-selected learning rate

PENDING: study not run

Complete seed blocks for paired analyses: 0 of 10. Eligible seeds: none. Missing run identities are recorded by seed in `completeness.csv`.

## Device-equivalence view

Descriptive side-by-side differences and confidence intervals for matched seeds 42–46 with complete gpu_v2 B1 blocks. Differences are gpu_v2 minus cpu_v1.

PENDING: no verified observations

## Probe and determinism studies

PENDING: no verified observations

PENDING: no verified observations

CUDA execution is NOT RUN locally. Cached hosted GPU observations, if imported, do not imply local CUDA validation.

## Supplementary frozen/exported-model latency

NOT RUN: this task prohibits benchmark reruns. Canonical CPU latency/size/INT8/memory evidence stays separate.
