# ExLightNet reproduction design

The supplied reproduction brief is the authoritative specification. The local
31-page PDF was extracted with pdftotext and read completely before implementation.
There is no existing application or usable Git repository in this workspace.

Implement a CPU-only Python package with shared, train-fitted preprocessing and a
phase-ordered runner. Persist split indices, preprocessing, checkpoints, epoch logs,
predictions, gate evidence, metrics, and provenance. Validation alone selects alpha
and checkpoints. Each trained checkpoint has one final test prediction pass; plots,
quantisation comparisons and SHAP reuse frozen predictions and fixed examples.
Quantised weights define a separate inference artifact, evaluated once.

Use the specified PyTorch/scikit-learn/XGBoost implementations. A NumPy-only model
reader implements the exported ExLightNet graph for Raspberry Pi benchmarking.
Fast mode is an explicitly reduced single-seed demonstration; full mode uses five
seeds, the specified epoch limits and full SHAP sizes. Never manufacture full-mode
results from fast runs. Hard gate failures stop dependent branches; report-only
MI comparisons and independently runnable model tests remain available.

Assumptions: one-layer BiLSTM (32 units per direction); global average pooling for
the baseline CNN; MI uses sklearn's continuous estimator with explicitly discrete
one-hot indicators, n_neighbors=3, random_state=42; alpha sweep uses at most eight
epochs per candidate on seed 42 validation, separately per task. All choices are
recorded. Optional KD and leakage proxy run last, if requested by CLI.

The user has already specified the approach, libraries, layout, phases and final
verification. Implement directly without another design approval. Alternatives
(paper feature order forced into inputs; synthetic stand-in data; tuning against
paper metrics) do not satisfy the brief.
