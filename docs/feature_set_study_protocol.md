# Feature-set study protocol

Declared on 2026-11-28 for campaign `feature_sets_v1`. Use the separate
experiment root `experiments/feature_sets_v1/` for prepared inputs,
checkpoints, receipts and aggregates. Campaign execution is **PENDING**.
The Phase 1 validation fits are post-hoc diagnostics. They motivate the
feature sets and remain outside the campaign's seed vectors and estimates.
The same rule applies to `cpu_v1` and `gpu_v2` observations. Reject pooled
campaigns during aggregation. [Phase 1 diagnostic](../artifacts/diagnostics_feature_sets/REPORT.md)

## Question and feature sets

The main contrast is F43 minus F40 for 12-class ExLightNet. It tests what
happens when the three specified flow flags are added after the original
40 columns. The other arms test the full candidate set and two compact
selection methods. An RF importance selector can favour an RF evaluator.
Use the neural comparisons to judge whether the effect transfers across
model families.

Use the canonical RT-IoT2022 split with split seed 42. Verify the recorded
split indices and data fingerprint. Fit medians, vocabularies, constant
pruning, feature rankings and z-scores on train. Use the unscaled float64
encoding for the selectors and apply the train-fitted z-scores to the ordered
columns for model inputs. Freeze one ordered list per set for both tasks and
all model seeds. Retain split indices, class names, fitted statistics and
ordered features with their data and payload fingerprints. [Canonical cache](../artifacts/cache_manifest.json)

| Set | Width | Columns in input order |
| --- | ---: | --- |
| F40 | 40 | Paper recipe using the binary-MI top 40 in descending order from `FlowPreprocessor`. |
| F43 | 43 | F40 followed by `fwd_psh_flag_count`, `flow_fin_flag_count`, `fwd_urg_flag_count`, in that order. |
| F93 | 93 | All train-varying candidates in `FlowPreprocessor.candidate_names_` order. |
| F_RF40 | 40 | First 40 columns in the filtered 12-class RF ranking below. |
| F_RF8 | 8 | First eight columns of that same filtered ranking. |
| F_CA40 | 40 | Class-aware train-only AUC quotas below, in order of selection. |

**RF importance and correlation.** Fit five Random Forests using the baseline
recipe, with the 12-class target and `random_state` values 42, 43, 44, 45 and
46. Fit each forest on the 93 encoded train columns. Average the five
`feature_importances_` values for each feature and rank by descending mean
importance, breaking ties by canonical candidate order. In the prepared-input
receipt, record the rank by mean importance alongside the five per-seed
ranks for each of `fwd_psh_flag_count`, `flow_fin_flag_count` and
`fwd_urg_flag_count`. Compute these ranks before correlation filtering, with
the same tie-break for the per-seed ranks. [Baseline RF recipe](../src/exlightnet/baselines.py)

The Phase 1 RF importance ranking used a single seed (`random_state=42`)
and remains post-hoc. The diagnostic comparison reported ranks 8 and 9 for
`flow_fin_flag_count`, at the unfiltered top-eight cutoff. For F_RF40 and
F_RF8, the campaign applies the filter only to the five-seed ranking by
mean importance. [Phase 1 diagnostic](../artifacts/diagnostics_feature_sets/REPORT.md)

Compute Pearson correlations on the same training matrix. Walk the ranking
and retain a feature if its absolute correlation with every already-retained
feature is at most **0.95**. This fixes the removal rule at `abs(r) > 0.95`
and gives precedence to the earlier-ranked feature. Fix the threshold before
computing the filter, and record every removal and its retained counterpart. Stop
F_RF40 if fewer than 40 candidates remain and F_RF8 if fewer than eight
remain. Membership of the flags follows the fixed rule. PNet-IDS combines RF
importance with correlation filtering and leaves the selection target
unspecified. The 12-class target and cutoff are choices for this protocol.
[Iliyasu et al., IEEE Access (2025), § IV-D.1](https://doi.org/10.1109/ACCESS.2025.3575705)

**Class-aware AUC quotas.** For each 12-class label, compute one-vs-rest train
ROC-AUC for each candidate. Score separation as `max(AUC, 1 - AUC)` so either
direction counts. Rank descending by this score with canonical candidate
order breaking ties. Visit labels in sorted canonical name order for three
rounds. In each round take the highest-ranked remaining feature for each
label. Append four remaining features by descending mean separation across
the 12 labels, breaking ties by candidate order. This gives every class
three opportunities to add a feature and uses a macro average for the
remaining slots. Univariate AUC is sensitive to small supports and can
retain correlated columns.

## Models and checkpoint selection

Fit ExLightNet, MLP, Random Forest and XGBoost for each feature set and task
(binary and 12-class). Use the package architectures and baseline recipes.
Instantiate MLP with the current width as `features`, keeping its hidden
layers fixed. Every ExLightNet uses the same no-padding trunk and dual global
pooling head, with two or 12 logits as appropriate.

Keep the neural optimizer, sampler and stopping recipe fixed across arms.
Use focal loss with gamma 2, the per-sample tempered sampler, and Adam with
learning rate `1.5e-3`, weight decay `1e-4`, batch size 512 and cosine
scheduling. Cap training at 30 epochs for 12-class and 20 for binary with
patience six. The earliest epoch attaining
the highest validation macro-F1 is the best-validation checkpoint. The
last completed epoch is the secondary checkpoint. Freeze the `cpu_v1`
alpha mode for each task, `effective_num` for binary and `none` for 12-class.
Each tree run has one fitted checkpoint. [Binary alpha receipt](../results/full/alpha/binary/selection.json) · [12-class alpha receipt](../results/full/alpha/multiclass/selection.json)

## Endpoints and test use

- 12-class. Macro-F1 and macro-recall are primary at the best-validation
  checkpoint. Report both again at the last epoch as secondary endpoints.
  Show test support alongside every per-class F1. Tiny supports, including
  the four `nmap_fin_scan` and five `metasploit_brute_force_ssh` test flows
  in this split, make 12-class macro-F1 sensitive to a few predictions.
  [D2 gate](../results/data_gates.json)
- Binary. Report attack recall and false-positive rate at the canonical
  argmax rule (threshold 0.5 with ties assigned to benign). Also report
  attack recall at **1% validation-target FPR**, with the realised test FPR
  next to it. For each checkpoint, choose the score threshold on validation
  that maximises attack recall subject to empirical FPR at most 1%. Use
  `p_attack >= threshold` for this endpoint. Candidates are the unique
  validation scores and positive infinity. Break recall ties by choosing
  the highest threshold. Apply this frozen rule to test probabilities.
- Perform test prediction once per checkpoint and persist its receipt and
  probability cache. Give the best and last endpoints separate receipts.
  If they refer to the same checkpoint, both can reference its single
  prediction cache. Compute every report metric from the saved predictions.
  Make all feature, checkpoint and threshold choices on train or validation
  as declared above.

## Matched seeds and inference

Use **15 training seeds, 42–56**, with the same split for every arm. This
sets a feasible matched sample larger than the five-seed diagnostic. Finish
each seed's scheduled runs within a device session before starting the next.
Use complete blocks of feasible arms within each task and model for pairing.
Keep any train-only selector failure visible as a stopped arm and leave its
comparisons missing. Mark partial seed blocks `incomplete; excluded` and
list their available and missing runs in the completeness table. Preserve
the planned seeds across resumes. Report mean, sample SD, count and seed 42
for each arm. Flag zero seed variance and unstable controls explicitly.

For each model, compare each of the five feature arms with F40 using matched
seed differences. Report the mean difference (arm minus F40), its two-sided
paired t p-value, 95% t interval, Cohen's paired `dz` and Holm-adjusted p-value.
Adjust within task and declare the following families:

- Neural best-checkpoint comparisons are primary, across both neural models
  and all five contrasts. The 12-class family covers two endpoints and 20
  planned tests. The binary family covers three endpoints and 30 planned
  tests.
- Last-epoch neural comparisons and the tree comparisons each have their
  own task-wise secondary Holm families. Per-class F1 comparisons are
  descriptive.

Apply the existing decision order. Label a difference `supported` only when
the 95% CI excludes zero and Holm p < 0.05. Otherwise label it `practically
equivalent` when the entire CI lies within ±0.01 for 12-class or ±0.002 for
binary, in absolute metric units. All other comparisons are `inconclusive`.
Mark unavailable statistics as missing. Lower is better for FPR. Each paired
comparison uses the same model, device and matched seeds in this campaign.

Paired tests over training seeds quantify variation from training randomness
alone, with the split and test set held fixed. With only four test flows for
`nmap_fin_scan` and five for `metasploit_brute_force_ssh`, 12-class macro-F1
also depends on which flows are in the test set. The conclusions are therefore
conditional on this split.

## Architecture and efficiency

The CPU architecture check used one synthetic flow for each combination
of width 8, 40, 43 or 93 and head size 12 or 2. All eight outputs had shape
`(1, K)` and finite logits. Global pooling keeps the head input at 288 across
widths. The trainable counts are 38,328 for 12 classes and 35,438 for binary.
[Architecture check](../artifacts/diagnostics_feature_sets/model_lengths.json)

Use the PNet-IDS convolution and dense FLOP convention of two operations
per multiply-accumulate. A grouped Conv1d contributes
`2 * L_out * k * (C_in / groups) * C_out`. A dense layer contributes
`2 * in_features * out_features`. For ExLightNet, the stem and two blocks
use lengths `L - 2`, `L - 4` and `L - 6`. Count the stem, depthwise and
pointwise convolutions, both linear layers in each SE block and the head.
These counts cover the convolution and linear operators. Pooling, batch
normalisation, activations, SE scaling and bias additions are outside the
counting convention. [PNet-IDS, § IV-G.5](https://doi.org/10.1109/ACCESS.2025.3575705) · [ExLightNet, § 3.3](../paper/exlightnet.pdf)

| Set | Width | Trainable params (12 / 2) | Conv and linear FLOPs (12 / 2) | Raw W8 model arrays, bytes (12 / 2) | Full W8 NPZ, bytes (12 / 2) |
| --- | ---: | ---: | ---: | ---: | --- |
| F40 | 40 | 38,328 / 35,438 | 1,349,568 / 1,343,808 | 46,284 / 43,324 | PENDING |
| F43 | 43 | 38,328 / 35,438 | 1,463,616 / 1,457,856 | 46,284 / 43,324 | PENDING |
| F93 | 93 | 38,328 / 35,438 | 3,364,416 / 3,358,656 | 46,284 / 43,324 | PENDING |
| F_RF40 | 40 | 38,328 / 35,438 | 1,349,568 / 1,343,808 | 46,284 / 43,324 | PENDING |
| F_RF8 | 8 | 38,328 / 35,438 | 133,056 / 127,296 | 46,284 / 43,324 | PENDING |
| F_CA40 | 40 | 38,328 / 35,438 | 1,349,568 / 1,343,808 | 46,284 / 43,324 | PENDING |

Raw W8 weights use 37,440 bytes for 12 classes and 34,560 for binary. The
table also counts FP32 scales and state tensors, whose shapes are constant
across widths. For the future efficiency report, record the actual local
CPU NPZ size of the seed-42 best-validation ExLightNet for each arm and
task. Use the existing exporter and include feature order and fitted
preprocessing metadata. The file contains compressed arrays, metadata and
container overhead. W8 inference uses dequantised weights and FP32 compute.
[Package exporter](../src/exlightnet/quantize.py)

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python -B -m scripts.diagnostics.model_lengths
```

## Execution estimate and open questions

Plan to use a single Kaggle T4 for the neural models, with the local CPU for
the trees. Confirm a compatible GPU runtime before starting. If Kaggle is
unavailable, choose a compatible Colab runtime before the first campaign
fit. Use FP32 CUDA, TF32 disabled, no AMP, and deterministic flags with
warnings recorded. Keep each model's hardware and library versions fixed
across resumes. Export only fingerprinted prepared arrays from local
storage for hosted sessions. Record campaign, study, arm, device, versions,
seed and fingerprints for each run. Save resumable progress and results
incrementally. [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)

- Neural budget. Six sets, two neural models and 15 seeds at the two epoch
  caps give at most 9,000 epochs. The historical T4 probes measured about
  1.91–3.21 s per epoch for width-40 ExLightNet, yielding a 4.8–8.0 GPU-hour
  reference at the caps. Allow roughly **6–10 GPU hours** for planning.
  Early stopping, width and the MLP's different compute cost will affect
  this estimate. Take a short timing probe before launching the long study
  and update the scheduling estimate. [Hosted probe records](../artifacts/stats_gpu_v2/probe.csv)
- Tree budget. The prior CPU B1 session completed 120 RF and XGBoost runs
  in 2,575.12 s, about 21.46 s per run. The 360 planned tree runs give a
  reference of 2.15 CPU hours serially. Allow roughly **2–4 CPU hours** for
  width-dependent changes and overhead. [Local B1 timing log](../archive/gpu_v2_local_trees/20261123_PA-20261123-2/B1-attempt-1.log)

The available hosted runtime and the incremental cost of width 93 will be
confirmed before scheduling the campaign. Perfect flag-based separation may
be specific to RT-IoT2022 and its traffic generator. An independent traffic
capture would be needed to establish whether the same flags separate these
classes elsewhere.
