# Feature-set diagnostics

From the repository root, use the pinned Python environment and the commands
in the [post-hoc diagnostic report](../../artifacts/diagnostics_feature_sets/REPORT.md).
The input flags select the CSV, the recorded split archive and the canonical
cache manifest. Use matching copies of those files. The CSV digest must match
the manifest. Select `--jobs 1` or `--jobs 2` for the CPU RandomForest fits.

`feature_sets` writes a JSON measurement record to stdout and progress to
stderr. It reads the saved train and validation indices, fits on train, and
scores on validation. Predictors use the unscaled encoded columns. Exact
vector matching compares the original float64 values. `render_feature_sets`
regenerates the comparison report from the saved JSON without any dataset
access or model fitting.
