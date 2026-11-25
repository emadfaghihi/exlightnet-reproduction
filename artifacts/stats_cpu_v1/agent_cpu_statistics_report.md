# CPU cached statistics implementation evidence

Scope: Stage A1–A3 only. No training, test inference, benchmark, checkpoint loading, process controls or original evidence writes occurred. The controller owns active report/notebook/status integration and the legacy-flag scope conflict.

## Implementation and reusable API

- `src/exlightnet/stats.py`: cache-only validation, t/CI/dz/Holm, dynamics, exact flow bootstrap, class support and sensitivity. Imports no training/model modules.
- `scripts/make_stats.py`: thin CLI; default output `artifacts/stats_cpu_v1/`, replicates=2000 and random seed=42.
- `tests/test_stats.py`: synthetic scored-cache and pure statistical tests; no optimization or model fitting.

`load_campaign_rows(results_path, manifest_path=None)` reads copies and returns tagged dictionaries. Untagged legacy rows require a full manifest listing seeds 42–46; they become `campaign=cpu_v1`, `device=cpu`, `split_seed=42` only in memory. Mixed tagged/legacy rows, campaigns and duplicate task/model/seed identities are rejected. Devices are separated except for the user-authorized gpu_v2 CUDA-neural/CPU-tree comparison; multiple devices for any one model’s seed vector are always rejected. The timestamped benchmark campaign is not a scientific campaign identifier.

`paired_summary(reference, comparison)` returns `n_pairs`, `mean_difference`, `difference_sd`, `t_statistic`, `p_value`, `ci_low`, `ci_high`, `cohens_dz`, `reference_mean`, `comparison_mean`, `reference_sd`, `comparison_sd`, `reference_zero_seed_variance`, `comparison_zero_seed_variance`, `zero_paired_variance`, `sample_label`. Differences are reference minus comparison. The t interval has df=n−1 and SD uses ddof=1. One pair leaves SD/p/CI/dz unavailable. A constant comparator is a valid one-sample problem; a constant paired difference is separately flagged. Zero constant differences return p=1/dz=0; nonzero constant differences return p=0/dz=±inf and point CIs as explicit degenerate limits. Cancellation noise of at most eight float64 ULPs is treated as a constant difference.

`holm_adjust(p_values)` returns a NumPy vector in input order; NaNs remain missing. `interpretation(ci_low, ci_high, p_holm, task)` applies the predeclared supported/equivalent/inconclusive rules with the difference-supported rule first.

`paired_tests(rows, reference="exlightnet")` returns a DataFrame of summary fields plus `campaign`, `device`, `task`, `reference`, `model`, `seeds`, `missing_reference_seeds`, `missing_comparison_seeds`, `p_holm`, `interpretation`, `reference_device`, `comparison_device`, `device_label`. Pairing is by matched seed; split-seed disagreement is rejected. Unmatched seeds are explicitly listed and excluded from that pair, with its actual n reported. Holm includes every comparison against the reference within a task, including ablations and W8. No Wilcoxon test is used: below six pairs its two-sided minimum p cannot be below .05.

`class_f1_from_confusion(cm)` returns full-axis class F1 with zero_division=0. `macro_f1_from_confusion(cm, labels=None)` averages a fixed label subset without removing any matrix rows/columns. `flow_confusion(y, predictions, n_classes)` returns fixed-axis integer counts and rejects noninteger/out-of-range labels or unaligned arrays.

`bootstrap_macro_f1(confusions, replicates=2000, random_seed=42, labels=None, pairing="UNPAIRED")` returns `macro_f1_mean`, `ci_low`, `ci_high`, `pairing`, `replicates`, `n_seeds`, `sample_label`, `estimand`. A draw first picks a training seed uniformly, then draws N counts from that seed’s empirical confusion-cell multinomial. With verified predictions, these cells are exactly their empirical flow distribution. Without predictions this is explicitly UNPAIRED.

`paired_flow_bootstrap(seed_caches, n_classes, replicates=2000, random_seed=42, labels=None)` returns the same schema with `mean_difference` replacing `macro_f1_mean` and `pairing=PAIRED`. Supply one dictionary per matched seed with aligned integer vectors `y`, `reference`, `prediction`; optional `reference_y`/`prediction_y` must match y, and optional `reference_ids`/`prediction_ids` must match each other and array lengths. The caller must first validate file hashes, campaign and seed identities. A joint (true, pred_reference, pred_comparison) multinomial preserves common-flow covariance exactly.

Both bootstraps produce 95% percentile intervals (NumPy linear quantiles). The estimand is **single-random-run plus flow uncertainty; not CI for the seed mean**. They must not be interpreted as confidence intervals on the five-seed average. The low-support mask, if provided, remains fixed from original support.

`per_class_table(rows)` and `sensitivity_table(rows, min_support=10)` validate consistent original support/class ordering across a task. Sensitivity excludes support<10 labels only from macro averaging, retaining false positives from their flows. It is labelled "post-hoc sensitivity, not a replacement for macro-F1".

`training_dynamics(rows)` returns `(per_run_frame, correlations_frame, binary_caps_frame)`. Missing epoch metadata remains missing, with `n_epoch_observed=0` and unavailable cap counts for trees/W8. LR factor is `0.5*(1+cos(pi*epochs_run/max_epochs))`. Instability is best_epoch<=3 OR per-model sample macro-F1 SD>0.05. Correlations are descriptive, never causal. The multiclass pooled family includes exactly ExLightNet/no-SE/same-width-standard-convolution/Std-CNN; epochs and F1 are separately standardized within model before Spearman.

`write_cpu_statistics(root, output, replicates=2000, random_seed=42)` returns output-name→path mappings. Root may be the repository or full-results directory. Writing within immutable results is refused. It validates complete evaluation receipts (one prediction pass), fingerprint/run identity, probabilities, true-label alignment, confusion matrices and macro-F1 against persisted rows before writing tables. A present claimed receipt is rejected even if predictions are missing. Absent predictions permit only a labelled unpaired fallback, with no paired interval fabricated. All read cache file hashes are checked again after generation.

`campaign_device_label(rows)` validates campaign/device/seed identities and returns the single device or exactly `mixed-device within gpu_v2; trees are device-independent`. Only literal campaign `gpu_v2` permits this exception: every mixed-table neural model must be on one CUDA device, every tree on CPU. The whole table receives the label, including its neural/neural rows. All generic class/dynamics tables also carry device/device_label; paired tables carry separate reference_device/comparison_device. No mixed-campaign override exists.

## Generated schemas and counts

| File | Rows | Columns |
| --- | ---: | --- |
| `per_seed_macro_f1.csv` | 95 | `campaign`, `device`, `task`, `model`, `seed`, `macro_f1`, `accuracy`, `epochs_run`, `best_epoch`, `n_seeds`, `sample_label` |
| `model_summary.csv` | 19 | `campaign`, `task`, `model`, `n_seeds`, `sample_label`, `macro_f1_mean`, `macro_f1_sd`, `macro_f1_seed42`, `zero_seed_variance` |
| `paired_tests.csv` | 17 | `campaign`, `device`, `reference_device`, `comparison_device`, `device_label`, `task`, `reference`, `model`, `seeds`, `missing_reference_seeds`, `missing_comparison_seeds`, `n_pairs`, `mean_difference`, `difference_sd`, `t_statistic`, `p_value`, `ci_low`, `ci_high`, `cohens_dz`, `reference_mean`, `comparison_mean`, `reference_sd`, `comparison_sd`, `reference_zero_seed_variance`, `comparison_zero_seed_variance`, `zero_paired_variance`, `sample_label`, `p_holm`, `interpretation` |
| `training_dynamics.csv` | 95 | `campaign`, `device`, `device_label`, `task`, `model`, `seed`, `macro_f1`, `model_macro_f1_sd`, `epochs_run`, `best_epoch`, `max_epochs`, `hit_cap`, `stopped_early`, `cosine_lr_factor`, `unstable`, `early_best_epoch`, `high_seed_sd`, `n_seeds`, `sample_label` |
| `correlations.csv` | 21 | `campaign`, `device`, `device_label`, `task`, `model`, `n`, `spearman_rho`, `p_value`, `sample_label`, `interpretation`, `members` |
| `binary_caps.csv` | 7 | `campaign`, `device`, `device_label`, `task`, `model`, `n_runs`, `n_epoch_observed`, `n_hit_cap`, `n_best_at_cap`, `sample_label` |
| `flow_bootstrap.csv` | 19 | `campaign`, `task`, `model`, `macro_f1_mean`, `ci_low`, `ci_high`, `pairing`, `replicates`, `n_seeds`, `sample_label`, `estimand` |
| `paired_flow_bootstrap.csv` | 17 | `campaign`, `task`, `reference`, `model`, `mean_difference`, `ci_low`, `ci_high`, `pairing`, `replicates`, `n_seeds`, `sample_label`, `estimand` |
| `per_class_f1.csv` | 158 | `campaign`, `device`, `device_label`, `task`, `model`, `class_name`, `test_support`, `f1_mean`, `f1_sd`, `f1_seed42`, `n_seeds`, `sample_label` |
| `sensitivity.csv` | 19 | `campaign`, `device`, `device_label`, `task`, `model`, `macro_f1_mean`, `macro_f1_sd`, `macro_f1_seed42`, `n_seeds`, `sample_label`, `excluded_classes`, `min_support`, `label` |

`statistics_manifest.json` records 95 source rows, 287 source-file hashes, all output hashes/row counts, software versions, stats source hash, seed/replicate configuration and interpretation scope. All n=5 tables carry `n=5, indicative only`.

## Verification evidence

- RED before implementation: `.venv/bin/python -m pytest tests/test_stats.py -q` exited 2 during collection with `ModuleNotFoundError: No module named exlightnet.stats`.
- GREEN final focused run: `23` tests, `0` failures, `0` errors; exit 0. JUnit: `cpu_statistics_pytest.xml`. Terminal output: `23 passed in 0.94s`.
- R5 amendment RED/GREEN: the two new gpu_v2 device-exception tests first failed (exit 1: 2 failed, 21 passed), then passed after the validator change; they also check that other campaigns, CPU neural comparators and same-model device mixing remain rejected.
- Cases cover SciPy t/CI agreement; constant comparator versus constant differences; one-pair missing values; Holm order/NaNs; missing seed pairs; campaign/device/duplicate rejection; legacy campaign derivation without mutation; support and false-positive-preserving sensitivity; paired identical-model covariance; alignment/ID rejection; fixed-class/reproducible unpaired bootstrap; cosine/cap/instability/four-model pooled dynamics; immutable synthetic scored-cache generation; corrupt probabilities, permuted flows, claimed and missing-prediction claimed receipts; explicit absent-prediction fallback; and single-random-run bootstrap estimand.
- Canonical generation: `rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python scripts/make_stats.py` completed with exit 0.
- Full immutable inventory check: `916` files; unchanged=`true`; differences=`0`. Evidence: `cpu_statistics_immutable_check.json`. `immutable_before.json` itself was preserved.
- `rtk git diff --check` completed with exit 0 at the implementation checkpoint. The controller performs whole-change verification. The full suite was not run by this worker.

## Reviewer observations re-derived from caches

Values below are generated directly from the newly written CSV tables. The reviewer’s approximate statistical observations reproduce; no substantive numerical mismatch was found.

| Task | Comparator | Mean ExLightNet difference | Paired p | Holm p | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| binary | mlp | 0.000169250 | 0.463439077 | 0.926878155 | practically equivalent |
| binary | random_forest | -0.006663488 | 6.89956073e-05 | 0.000413973644 | difference supported |
| binary | xgboost | -0.006301300 | 7.76110468e-05 | 0.000413973644 | difference supported |
| multiclass | mlp | 0.010999316 | 0.158386379 | 0.950318274 | inconclusive |
| multiclass | no_se | 0.019583988 | 0.256678678 | 0.950318274 | inconclusive |
| multiclass | random_forest | -0.013002073 | 0.174622239 | 0.950318274 | inconclusive |
| multiclass | standard_conv | -0.015280745 | 0.0614052981 | 0.491242385 | inconclusive |
| multiclass | xgboost | 0.013934229 | 0.184663468 | 0.950318274 | inconclusive |

Tiny-class support makes multiclass macro-F1 noisy; these are fixed-split training-seed comparisons, not split-variance estimates.

| Multiclass dynamics | n | Spearman rho |
| --- | ---: | ---: |
| exlightnet | 5 | 0.974679434 |
| no_se | 5 | 0.872081599 |
| std_cnn | 5 | 0.900000000 |
| pooled_conv_family | 20 | 0.793376494 |

Binary ExLightNet: 4 of 5 runs hit the epoch cap; 2 best epochs were at the cap. Multiclass Std-CNN: 3 of 5 best epochs were <=3; their macro-F1 range is 0.622700301–0.638552403. The model-level high-SD flag also marks every Std-CNN row unstable.

| Class | Test support | ExLightNet class F1 SD |
| --- | ---: | ---: |
| metasploit_brute_force_ssh | 5 | 0.040655781 |
| nmap_fin_scan | 4 | 0.183870605 |

XGBoost zero training-seed variance is explicitly recorded in `model_summary.csv` and paired comparator diagnostics. The paired-difference variance against ExLightNet remains nonzero.

## Limits and integration concerns

- Canonical prediction NPZs contain true labels and probabilities but no unique test-flow IDs. Their ordered true-label vectors, fixed data/run fingerprints, confusion counts and metrics are verified. A permutation solely within the same true-label class cannot be diagnosed from these legacy fields alone. New hosted predictions should provide IDs/provenance where available; paired helper rejects supplied ID misalignment.
- Existing evaluation receipts have no prediction-file content SHA. This pass records SHA256 for every read prediction and receipt in the new manifest and verifies the current arrays against their metrics; it cannot retroactively prove preexisting bytes against an unavailable historical prediction hash.
- The user resolved the legacy flag scope: immutable historical JSON remains untouched and is excluded from the active scan. Active report/code removal belongs to the controller.
- `REPORT.md`, the notebook, final Gate A/B table, source packaging and whole-suite testing are controller-owned; this worker edited no files outside its ownership.
- No CUDA path, hosted study, supplementary latency or local training was run.
