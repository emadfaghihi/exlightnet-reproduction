# PA-20261118-5 stratified final SHAP

The CLI uses the canonical `cpu_v1` multiclass seed-42 FP32 ExLightNet
checkpoint. It validates the input file hashes against `immutable_before.json`,
checks the prepared-data and checkpoint fingerprints, and requires the original
completed, single-pass evaluation receipt and matching cached labels. It uses
the original cached test predictions for each selected flow's fixed predicted
class and the unmasked deletion probability.

```bash
rtk proxy .venv/bin/python scripts/stratified_shap.py --root .
```

The CLI delegates to `exlightnet.final_analysis.run_stratified_shap(root,
output=None)`. It sets OMP, OpenBLAS, MKL and NumExpr limits before importing
the package and acquires the repository's exclusive `results/.locks/stage.lock`
through `execution_lock`. The controller can keep supervisor logs outside the
analysis output directory and let the child acquire that lock. The default
output is `artifacts/final_analysis/stratified_shap_20261118`. An explicit
`--output` can be an absolute path or relative to `--root`.

`analysis.json` begins as an in-progress claim. Its input signature is the
package's sorted-JSON SHA-256 digest of the protocol settings, package versions,
input evidence and implementation hashes, canonical class and feature names,
and exact test and background positions. Positions are row offsets within
their respective cached splits. On successful completion the same manifest
records `status: complete`, the final-analysis label, stage timings, SHAP
method and fallback metadata, additivity residuals and warnings, per-class
counts, limitations, and SHA-256 for each output. A matching rerun verifies
the hashes and returns the completed manifest. An input-signature mismatch,
altered completed output or unclaimed nonempty directory stops the analysis
for inspection.

The attribution and deletion NPZ stages are atomically published before
plotting. An interruption after the attribution checkpoint can reuse the
saved SHAP tensor. A report failure after both checkpoints can reuse both
scientific stages. Resume with the same command, input evidence, code and
environment. The model loads only for a missing scientific stage.

## Outputs

| File | Contents |
| --- | --- |
| `stratified_shap_values.npz` | Signed SHAP tensor with axes sample, feature and output class, exact selected positions, and canonical mappings |
| `per_class_top10.csv` | Mean absolute SHAP for the true-class logit, ten features per true class |
| `per_class_all_outputs_ranking.csv` | Mean absolute SHAP over all output logits within each true class |
| `balanced_global_ranking.csv` | Equal-class mean of the primary true-class logit vectors |
| `balanced_global_all_outputs_ranking.csv` | Equal-class mean of the auxiliary all-output vectors |
| `deletion_curves.csv` | One row per true class and removal count with SHAP and mean random probabilities |
| `early_deletion_k10.csv` | k=0 and k=10 probabilities, k=10 drops, random-minus-SHAP contrast and normalized full-curve AUC |
| `deletion_replicates.npz` | Signed per-class SHAP curves, all 20 random curves and their exact feature permutations |
| `earlier_random300_class_composition.csv` | Cached class counts, dataset support and both fractions for the original random sample |
| `balanced_global_ranking.png`, `per_class_top10.png`, `deletion_panels.png` | Visualizations generated from the analysis tables |

The deletion baseline uses an independent generator seeded at 42, with one
feature permutation applied to every flow in a true-class group per replicate.
The contrast compares the same flows under both masking orders. Zero masking
probes the model in train-fitted z-space, so a curve can be nonmonotonic and
has a descriptive interpretation. Deletion summaries include support, sampled
counts and correct predictions for each class.

## Verification before supervised execution

The focused synthetic and fixed-model tests passed with exit 0. The
read-only input check verified the expected fingerprints and the 12 original
seed-42/explanation file hashes in its scope. It derived 259 sampled test flows
across 12 classes and verified the 200 saved background positions. Logs live
in `stratified_shap_tests.log` and `stratified_shap_input_check.log`.
The focused test time covers synthetic attribution/deletion and cache recovery.
Wall time and measured SHAP diagnostics for the real checkpoint come from the
supervised run.

## Readiness correction

The first readiness notice followed a passing focused suite whose synthetic
attribution omitted the NumPy `expected_logits` metadata. The controller
started the supervised child on that revision. A subsequent source review
identified that the old stage writer could not serialize the actual NumPy
metadata after SHAP returned. The regression test reproduced the serialization
failure, and the fix reused the repository's NumPy-aware strict JSON encoder.
The fixed focused suite finished with 12 passed in 4.09 seconds.

The initial claim identifies `final_analysis.py` SHA-256
`264f4cdac6f040281e0ae0db584491a3253536f3716e37886141ee03e7f3b9e3`.
The fixed, frozen implementation SHA-256 is
`4ea446d7e41b60d74f16df4ce51a1314479144b82fa12e0787b560c3267b93eb`.
The signature distinguishes these revisions. The controller owns preservation
of the initial attempt and any recovery into a fresh output directory.
