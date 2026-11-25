# Prepared-data export implementation and evidence

Implemented only src/exlightnet/prepared.py, scripts/export_prepared.py,
tests/test_prepared.py, and the split_seed extension in src/exlightnet/data.py.
No original cache, result, checkpoint, receipt, raw CSV or benchmark was changed.
No training, model inference or benchmark was performed. CUDA was NOT EXECUTED.

## API and NPZ schema

- export_prepared(root, output, split_seeds) returns the export manifest. It refuses
  existing exports and canonical evidence/data output directories. It reads the
  local CSV with download=False. Split42 reads frozen arrays and preprocessing;
  all other seeds fit the existing FlowPreprocessor on training rows only.
- load_prepared(path) returns a dict with arrays, feature_names, class_names and
  metadata. Module import and the entire load path use only stdlib and NumPy.
  pandas, sklearn, preprocessing and run_all are absent from the isolated loader
  subprocess's imported modules.
- arrays has x_train/x_val/x_test (float32, N x 40), y2_train/y2_val/y2_test
  (integer binary labels), and y12_train/y12_val/y12_test (integer 12-class labels).
- NPZ additionally contains split_train/split_val/split_test (integer original-row
  indices), feature_names and class_names (Unicode arrays), and metadata (scalar
  Unicode JSON). No pickle is accepted. Indices partition the complete row range,
  with no duplicate or overlapping rows.
- class_names uses the original sorted 12-class mapping. Binary 0 means one of
  thing_speak, mqtt_publish or wipro_bulb; binary 1 means an attack class.
- metadata includes schema_version=1, settings, split_seed, data_fingerprint,
  payload_sha256, frozen_alpha (binary/multiclass mode strings), alpha_source,
  fit_partition=train, full fitted preprocessor state and preprocessor_sha256.
  The frozen modes are {"binary": "effective_num", "multiclass": "none"}.
- data_fingerprint preserves run_all.digest(settings) exactly: SHA256 of
  json.dumps(settings, sort_keys=True). It is the historical recipe identity.
  payload_sha256 independently binds sorted, length-framed array key/dtype/shape
  headers and C-order bytes, then canonical JSON containing feature/class names
  and metadata except payload_sha256 itself. See payload_digest's docstring.
- prepared/manifest.json records each file's SHA256, payload and recipe digests,
  size, elapsed seconds, preprocessor digest, class support and D1/D2 gates.
  File SHA anchors are separate from the recipe and content digests.

## Actual export and independent checks

Command (exit 0):

~~~sh
rtk proxy env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python scripts/export_prepared.py --splits 42-51
~~~

Exported 10 splits serially in 266.409349 seconds;
total NPZ size is 40366861 bytes. Every partition has
all 12 classes and its original 86181/18468/18468 size. All ten files were
independently reopened through the current verified loader and their file and
payload hashes checked against the manifest.

Split42 recipe fingerprint: dfba90bd116f9288bc72a837160516984ba773b11904e98561c80cda7dedcf5c.
Every one of its 9 arrays is byte-identical to the canonical CPU cache, including
dtype and shape. All three split-index arrays are byte-identical. The real-cache
regression forbids FlowPreprocessor.fit during split42 export and hashes the
original source artifacts before and after it. No canonical array rebuild or
prepare_data call occurs.

After export, all 916 immutable baseline
file hashes were checked again: 0 changed. Per-array SHA256s,
all split fingerprints, supports, timings and source provenance are in
[prepared_export_checks.json](prepared_export_checks.json); exported file hashes
are in [manifest.json](../../prepared/manifest.json).

## Tests and review evidence

The first red run had 14 failures: the new module/API and split_seed argument did
not exist. The real split42 regression then exposed object-dtype class names;
export now explicitly writes Unicode strings, so allow_pickle=False succeeds.
Three additional red tests showed noninteger recipe values could pass equality
checks; strict integer recipe validation now rejects them. The CLI red test
failed before the CLI existed and now verifies descending ranges exit 2 before
creating output.

Latest focused command is recorded in prepared_export_checks.json: 27 passed,
exit 0, 7.36 seconds. It includes prepared/data/preprocess tests and exercises
payload corruption, metadata tampering, recipe fingerprint/type mismatches,
non-finite values, float32 requirements, label shapes, class coverage and binary
mapping, overlapping split indices, default split preservation, train-only real
preprocessing, read-only split42 export, overwrite refusal and offline imports.
CLI --help exited 0. git diff --check exited 0; an independent scoped-source
trailing-whitespace check also exited 0. The full suite is controller-owned and
was not run by this agent.

## Limits and integration notes

No GPU path, supplementary training, raw-data upload, archive packaging or model
benchmark was executed. Prepared NPZ files are data artifacts and must be
excluded from source/delivery archives and Git. Runner identities should bind
payload_sha256 as well as the historical recipe fingerprint. Internally stored
hashes detect accidental corruption; externally trusted manifest/file SHA256s
remain the anchor against replacement of an entire self-consistent export.

Exports refuse overwrites. If export is interrupted after publishing some files,
preserve the partial export and choose a fresh directory; this exporter does not
silently rewrite or append to an interrupted manifest. Hosted training has a
separate resume contract and consumes only completed exports.
