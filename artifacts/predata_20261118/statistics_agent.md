# PA-20261118-1 and PA-20261118-2 statistics verification

`aggregate_studies` retains its callable interface. It reads the two dated
amendments from the supplied protocol. The report presents the active rules
verbatim and labels the original stopping-hypothesis rule as superseded.

`early_stopping_hypothesis.csv` has one row per endpoint, including on empty
input. The primary row compares fixed30 last against paper best. The secondary
row compares fixed30 best against paper best. Both require every declared
matched seed from 42 to 56, and both show their own decision and the overall
decision. The B1 descriptive Spearman result appears alongside the endpoints.
The variance analysis uses sample SD and 2,000 resamples of paired seed indices
with seed 42. An undefined ratio or any undefined bootstrap draw produces an
inconclusive endpoint. A zero fixed-arm SD remains a valid ratio when each
resample's paper SD is positive.

`completeness.csv` keeps one `row_type=study` summary per study and adds one
`row_type=seed` row per declared seed. A seed row includes the split seed,
counts of planned runs and B1 references, and JSON identities of missing runs.
Incomplete seed rows have status `incomplete; excluded`. B3 requires the
four B1 paper references and eight new arm runs for each seed. B4 requires both
models and their B1 references. B5 requires all three LR arms in a seed block
and still requires the entire predeclared study for LR selection. B1 neural
and tree shards contribute together only after archive verification and merge.

`per_run.csv` adds `inference_eligible`. `summaries.csv` adds `n_eligible` and
`observation_scope`. Paired t tests, flow bootstraps, B1 correlations, and the
stopping hypothesis use eligible seed blocks. The descriptive device-equivalence
view uses eligible B1 seed blocks on the supplementary side. Campaign, device,
environment, run recipe and evidence validations still precede eligibility.

## Verification

- Initial focused regression run returned exit 1 with 18 failures and 42
  passes in 6.14 seconds. It exposed absent amended endpoints, complete-block
  filtering, completeness records and report rules.
  See `statistics_tests_red.log` and `statistics_tests_red.xml`.
- Final focused run returned exit 0 with 63 passes in 7.71 seconds.
  See `statistics_tests.log` and `statistics_tests.xml`.
- Cache-only statistics suite returned exit 0 with 86 passes in 7.24 seconds.
  See `statistics_tests_cache_only.log` and `statistics_tests_cache_only.xml`.
- `git diff --check` passed for the tracked statistics changes. The untracked
  test file also produced no whitespace diagnostic under `git diff --no-index
  --check`.
- The tests used synthetic, previously scored evidence fixtures. Full-block
  integration exercised distinct best and last checkpoint receipts for all
  15 seeds. No Stage B experiment, CUDA run, model inference, benchmark, or
  historical result regeneration occurred in this work.
