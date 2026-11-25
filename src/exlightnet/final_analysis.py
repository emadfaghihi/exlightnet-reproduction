"""Stratified final SHAP and deletion from the immutable cpu_v1 FP32 checkpoint."""

import csv
from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np

from .execution import configure_single_thread, execution_lock, load_prepared_data, utc_now
from .explain import deletion_curve, explain_attributions
from .io import _json_default, read_json, sha256, write_json
from .metrics import read_evaluation_cache
from .run_all import _load_run_model, digest


DEFAULT_OUTPUT = Path("artifacts/final_analysis/stratified_shap_20261118")
RUN_DIRECTORY = Path("results/full/multiclass/exlightnet/42")
ORIGINAL_EXPLANATION = Path("results/full/explain/explanations.json")
SNAPSHOT = Path("artifacts/predata_20261118/immutable_before.json")
SHAP_CACHE = "stratified_shap_values.npz"
DELETION_CACHE = "deletion_replicates.npz"
EARLY_REMOVALS = 10
RANDOM_PERMUTATIONS = 20
RNG_SEED = 42
SAMPLE_CAP = 25
GRADIENT_NSAMPLES = 200
LABEL = "final analysis; no selection use"
AUCDESC = "Trapezoidal probability AUC over fraction of features removed (0..1), unscaled by starting probability"
SETTINGS = {
    "protocol_id": "PA-20261118-5",
    "schema_version": 1,
    "seed": RNG_SEED,
    "sampling": "Without replacement by ascending true-class index; retain all in split order at or below the cap",
    "sample_cap_per_true_class": SAMPLE_CAP,
    "background": "Exact prior FP32 training positions in saved order",
    "background_size": 200,
    "prior_random_test_size": 300,
    "index_basis": "Zero-based row position in the corresponding cached split",
    "deep_additivity_preflight_samples": 2,
    "gradient_nsamples": GRADIENT_NSAMPLES,
    "primary_ranking": "Mean absolute SHAP for the true class logit within each true class",
    "auxiliary_ranking": "Mean absolute SHAP over all output logits within each true class",
    "global_ranking": "Arithmetic mean across classes with equal weight",
    "ranking_tie_break": "Canonical feature order",
    "deletion_group": "True class",
    "deletion_target": "Each flow's fixed original predicted-class probability from the final evaluation cache",
    "deletion_order": "Descending absolute SHAP for the original predicted-class logit, canonical feature tie break",
    "mask_value": "Zero in train-fitted z-space",
    "early_removals": EARLY_REMOVALS,
    "contrast": "Random minus SHAP mean probability at k=10 on the same flows",
    "random_permutations": RANDOM_PERMUTATIONS,
    "random_order": "Independent RNG seeded at 42; one shared feature order per permutation and true class",
    "auc_definition": AUCDESC,
    "device": "CPU",
    "precision": "FP32",
    "threads": 1,
}
TABLES = (
    "per_class_top10",
    "per_class_all_outputs_ranking",
    "balanced_global_ranking",
    "balanced_global_all_outputs_ranking",
    "deletion_curves",
    "early_deletion_k10",
    "earlier_random300_class_composition",
)
PLOTS = ("balanced_global_ranking.png", "per_class_top10.png", "deletion_panels.png")
OUTPUT_FILES = (SHAP_CACHE, DELETION_CACHE, *(f"{name}.csv" for name in TABLES), *PLOTS)


@dataclass
class AnalysisInputs:
    inputs: np.ndarray
    background: np.ndarray
    sample_indices: np.ndarray
    background_indices: np.ndarray
    true_labels: np.ndarray
    predicted: np.ndarray
    original_probabilities: np.ndarray
    class_names: list[str]
    feature_names: list[str]
    support: np.ndarray
    earlier_composition: list[dict]
    provenance: dict


def stratified_indices(labels, classes):
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.dtype.kind not in "iu" or np.any((labels < 0) | (labels >= classes)):
        raise ValueError("Invalid canonical class labels")
    rng = np.random.default_rng(RNG_SEED)
    sampled = []
    for class_index in range(classes):
        indices = np.flatnonzero(labels == class_index)
        if len(indices) == 0:
            raise ValueError(f"No test support for class {class_index}")
        sampled.append(rng.choice(indices, SAMPLE_CAP, replace=False) if len(indices) > SAMPLE_CAP else indices)
    return np.concatenate(sampled)


def _checked_indices(values, size, expected):
    indices = np.asarray(values)
    if (indices.shape != (expected,) or indices.dtype.kind not in "iu"
            or np.any((indices < 0) | (indices >= size)) or len(np.unique(indices)) != expected):
        raise RuntimeError("Saved explanation indices differ from the declared sampling design")
    return indices


def earlier_sample_composition(labels, class_names, old_explanation):
    sample_size = SETTINGS["prior_random_test_size"]
    if old_explanation["random_test_size"] != sample_size:
        raise RuntimeError("Expected the original full explanation's random 300-flow sample")
    positions = _checked_indices(old_explanation["test_indices"], len(labels), sample_size)
    sample_counts = np.bincount(labels[positions], minlength=len(class_names))
    supports = np.bincount(labels, minlength=len(class_names))
    return [
        {"class_index": c, "true_class": name, "test_support": int(supports[c]),
         "sampled_count": int(sample_counts[c]), "test_fraction": float(supports[c] / len(labels)),
         "sample_fraction": float(sample_counts[c] / len(positions))}
        for c, name in enumerate(class_names)
    ]


def _load_analysis_inputs(root):
    source_paths = (
        "artifacts/cache_manifest.json", "artifacts/cache_arrays.npz", "artifacts/preprocessor.json",
        "artifacts/class_names.json", "results/data_gates.json", str(ORIGINAL_EXPLANATION),
        *(str(RUN_DIRECTORY / name) for name in
          ("model.pt", "metrics.json", "predictions.npz", "test_evaluation_receipt.json")),
    )
    hashes = {name: sha256(root / name) for name in source_paths}
    snapshot = read_json(root / SNAPSHOT)["files"]
    for name, sha in hashes.items():
        if snapshot.get(name) != sha:
            raise RuntimeError(f"Original evidence differs from the recorded SHA-256: {name}")

    arrays, class_names, preprocessor, context = load_prepared_data(root)
    prepared_manifest = read_json(root / "artifacts/cache_manifest.json")
    data_fingerprint = context["fingerprint"]
    if digest(prepared_manifest["settings"]) != data_fingerprint:
        raise RuntimeError("Prepared-data fingerprint differs")
    run_dir = root / RUN_DIRECTORY
    metrics = read_evaluation_cache(run_dir)
    receipt = read_json(run_dir / "test_evaluation_receipt.json")
    checkpoint_fingerprint = digest({
        "weights": hashes[str(RUN_DIRECTORY / "model.pt")],
        "data": data_fingerprint,
        "task": "multiclass",
        "recipe": metrics["recipe"],
    })
    if (
        (metrics["task"], metrics["model"], metrics["seed"], metrics["features"]) != ("multiclass", "exlightnet", 42, 40)
        or metrics["data_fingerprint"] != data_fingerprint
        or metrics["fingerprint"] != checkpoint_fingerprint
        or receipt["test_prediction_passes"] != 1
        or any(metrics.get(key) != value for key, value in receipt["metadata"].items())
    ):
        raise RuntimeError("Original checkpoint or completed single-pass evaluation fingerprint differs")

    labels = arrays["y12_test"]
    train, test = arrays["x_train"], arrays["x_test"]
    features = list(preprocessor.selected_names_)
    if (len(class_names) != 12 or len(features) != 40 or train.shape[1:] != (40,)
            or test.shape != (len(labels), 40) or len(set(features)) != 40 or len(set(class_names)) != 12):
        raise RuntimeError("Prepared data differ from the canonical 12-class, 40-feature mapping")
    positions = stratified_indices(labels, len(class_names))
    support = np.bincount(labels, minlength=len(class_names))
    if not np.array_equal(support, [metrics["per_class"][name]["support"] for name in class_names]):
        raise RuntimeError("Prepared labels differ from cached evaluation class supports")
    with np.load(run_dir / "predictions.npz", allow_pickle=False) as cached:
        probabilities = cached["probabilities"]
        if (not np.array_equal(cached["y"], labels) or probabilities.shape != (len(labels), len(class_names))
                or not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
            raise RuntimeError("Original predictions or test labels do not match the prepared split")
        sampled_probabilities = probabilities[positions]
    predicted = sampled_probabilities.argmax(axis=1)

    earlier = read_json(root / ORIGINAL_EXPLANATION)
    if earlier["background_size"] != SETTINGS["background_size"]:
        raise RuntimeError("Expected the original full explanation's 200-flow train background")
    background_positions = _checked_indices(earlier["background_train_indices"], len(train), SETTINGS["background_size"])
    return AnalysisInputs(
        inputs=test[positions], background=train[background_positions], sample_indices=positions,
        background_indices=background_positions, true_labels=labels[positions], predicted=predicted,
        original_probabilities=sampled_probabilities[np.arange(len(positions)), predicted],
        class_names=class_names, feature_names=features, support=support,
        earlier_composition=earlier_sample_composition(labels, class_names, earlier),
        provenance={
            "campaign": "cpu_v1", "task": "multiclass", "model": "exlightnet", "checkpoint_seed": 42,
            "checkpoint_fingerprint": checkpoint_fingerprint, "data_fingerprint": data_fingerprint,
            "source_sha256": hashes, "immutable_snapshot": str(SNAPSHOT),
            "immutable_snapshot_sha256": sha256(root / SNAPSHOT),
        },
    )


def class_importance(phi, labels, classes):
    absolute = np.abs(phi)
    own = []
    all_outputs = []
    for c in range(classes):
        values = absolute[labels == c]
        own.append(values[:, :, c].mean(axis=0))
        all_outputs.append(values.mean(axis=(0, 2)))
    return np.stack(own), np.stack(all_outputs)


def ranked_features(importance, names, limit=None):
    for rank, feature_index in enumerate(np.argsort(-importance, kind="stable")[:limit], start=1):
        yield {"rank": rank, "feature_index": int(feature_index), "feature": names[feature_index],
               "mean_abs_phi": float(importance[feature_index])}


def class_deletion(model, inputs, labels, predicted, original, phi, classes):
    rng = np.random.default_rng(RNG_SEED)
    width = inputs.shape[1]
    predicted_phi = phi[np.arange(len(inputs)), :, predicted]
    orders = np.argsort(-np.abs(predicted_phi), axis=1, kind="stable")
    shap_curves, random_curves, random_orders = [], [], []
    for c in range(classes):
        member = labels == c
        x, p, cached = inputs[member], predicted[member], original[member]
        shap_curves.append(deletion_curve(model, x, p, orders[member], original_probabilities=cached))
        permutations = np.stack([rng.permutation(width) for _ in range(RANDOM_PERMUTATIONS)])
        random_orders.append(permutations)
        random_curves.append([
            deletion_curve(model, x, p, np.broadcast_to(order, x.shape), original_probabilities=cached)
            for order in permutations
        ])
    return np.asarray(shap_curves), np.asarray(random_curves), np.asarray(random_orders)


def deletion_rows(shap_curves, random_curves, class_rows):
    rows, early = [], []
    k10 = EARLY_REMOVALS
    for info, shap, replicates in zip(class_rows, shap_curves, random_curves, strict=True):
        random = replicates.mean(axis=0)
        width = len(shap) - 1
        for k in range(width + 1):
            rows.append({**info, "features_removed": k, "fraction_removed": k / width,
                         "shap_mean_probability": float(shap[k]), "random_mean_probability": float(random[k])})
        early.append({
            **info, "original_mean_probability": float(shap[0]), "shap_k10": float(shap[k10]),
            "random_k10": float(random[k10]), "shap_drop_k10": float(shap[0] - shap[k10]),
            "random_drop_k10": float(shap[0] - random[k10]),
            "random_minus_shap_k10": float(random[k10] - shap[k10]),
            "shap_normalized_auc": float(np.trapz(shap, dx=1 / width)),
            "random_normalized_auc": float(np.trapz(random, dx=1 / width)),
        })
    return rows, early


def _save_stage(path, signature, metadata, **arrays):
    partial = path.with_suffix(".npz.partial")
    encoded_metadata = json.dumps(metadata, default=_json_default, allow_nan=False)
    with partial.open("wb") as stream:
        np.savez_compressed(stream, signature=signature, metadata=encoded_metadata, **arrays)
    partial.replace(path)


def _read_stage(path, signature):
    with np.load(path, allow_pickle=False) as cache:
        if cache["signature"].item() != signature:
            raise RuntimeError(f"Saved stage signature differs: {path}")
        metadata = json.loads(cache["metadata"].item())
        arrays = {name: cache[name] for name in cache.files if name not in {"signature", "metadata"}}
    return arrays, metadata


def _attribution_stage(path, signature, data, load_model):
    if not path.exists():
        print("Computing FP32 SHAP on the stratified test sample", flush=True)
        start = time.perf_counter()
        result = explain_attributions(load_model(), data.background, data.inputs, nsamples=GRADIENT_NSAMPLES)
        phi = result.pop("values")
        result["seconds"] = time.perf_counter() - start
        _save_stage(path, signature, result, phi=phi, sample_indices=data.sample_indices,
                    background_train_indices=data.background_indices, true_labels=data.true_labels,
                    predicted_classes=data.predicted, class_names=data.class_names, feature_names=data.feature_names)
        print(f"Saved attribution checkpoint: {path}", flush=True)
    arrays, metadata = _read_stage(path, signature)
    phi = arrays["phi"]
    if phi.shape != (*data.inputs.shape, len(data.class_names)) or not np.isfinite(phi).all():
        raise RuntimeError("Saved stratified attribution shape or values differ")
    for name, expected in (("sample_indices", data.sample_indices), ("background_train_indices", data.background_indices),
                           ("true_labels", data.true_labels), ("predicted_classes", data.predicted),
                           ("feature_names", data.feature_names), ("class_names", data.class_names)):
        if not np.array_equal(arrays[name], expected):
            raise RuntimeError(f"Saved attribution {name} differs")
    return phi, metadata


def _deletion_stage(path, signature, data, phi, load_model):
    classes = len(data.class_names)
    if not path.exists():
        print("Computing per-class deletion curves and random replicates", flush=True)
        start = time.perf_counter()
        shap, random, orders = class_deletion(
            load_model(), data.inputs, data.true_labels, data.predicted, data.original_probabilities, phi, classes
        )
        _save_stage(path, signature, {"seconds": time.perf_counter() - start},
                    shap_curves=shap, random_curves=random, random_orders=orders)
        print(f"Saved deletion checkpoint: {path}", flush=True)
    arrays, metadata = _read_stage(path, signature)
    shap, random, orders = (arrays[name] for name in ("shap_curves", "random_curves", "random_orders"))
    width = data.inputs.shape[1]
    if (shap.shape != (classes, width + 1) or random.shape != (classes, RANDOM_PERMUTATIONS, width + 1)
            or orders.shape != (classes, RANDOM_PERMUTATIONS, width)
            or not np.isfinite(shap).all() or not np.isfinite(random).all()):
        raise RuntimeError("Saved deletion shapes or values differ")
    return shap, random, metadata


def _write_csv(path, records):
    partial = path.with_suffix(".csv.partial")
    with partial.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    partial.replace(path)


def _write_tables(output, data, phi, shap, random, class_rows):
    primary, auxiliary = class_importance(phi, data.true_labels, len(data.class_names))
    names = data.feature_names
    curves, early = deletion_rows(shap, random, class_rows)
    tables = {
        "per_class_top10": [
            {**info, **row} for info, scores in zip(class_rows, primary, strict=True)
            for row in ranked_features(scores, names, limit=10)
        ],
        "per_class_all_outputs_ranking": [
            {**info, **row} for info, scores in zip(class_rows, auxiliary, strict=True)
            for row in ranked_features(scores, names)
        ],
        "balanced_global_ranking": list(ranked_features(primary.mean(axis=0), names)),
        "balanced_global_all_outputs_ranking": list(ranked_features(auxiliary.mean(axis=0), names)),
        "deletion_curves": curves,
        "early_deletion_k10": early,
        "earlier_random300_class_composition": data.earlier_composition,
    }
    for name, rows in tables.items():
        _write_csv(output / f"{name}.csv", rows)
    return tables


def _write_plots(output, tables, class_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        path = output / name
        partial = path.with_suffix(".png.partial")
        try:
            fig.savefig(partial, dpi=160, format="png")
            partial.replace(path)
        finally:
            plt.close(fig)

    def bars(ax, rows):
        reverse = rows[::-1]
        ax.barh([row["feature"] for row in reverse], [row["mean_abs_phi"] for row in reverse], color="#176b87")

    fig, ax = plt.subplots(figsize=(9, 6))
    bars(ax, tables["balanced_global_ranking"][:15])
    ax.set_xlabel("Equal-class mean of own-logit |SHAP|")
    save(fig, PLOTS[0])

    shape = ((len(class_rows) + 2) // 3, 3)
    fig, axes = plt.subplots(*shape, figsize=(18, 4 * shape[0]), squeeze=False)
    for ax, info in zip(axes.flat, class_rows):
        rows = [r for r in tables["per_class_top10"] if r["class_index"] == info["class_index"]]
        bars(ax, rows)
        ax.set(title=f'{info["true_class"]} ({info["sampled_count"]}/{info["test_support"]})',
               xlabel="Mean own-logit |SHAP|")
        ax.tick_params(axis="y", labelsize=8)
    for ax in list(axes.flat)[len(class_rows):]:
        ax.set_visible(False)
    save(fig, PLOTS[1])

    fig, axes = plt.subplots(*shape, figsize=(14, 3.1 * shape[0]), sharex=True, sharey=True, squeeze=False)
    for ax, info in zip(axes.flat, class_rows):
        rows = [r for r in tables["deletion_curves"] if r["class_index"] == info["class_index"]]
        removed = [r["features_removed"] for r in rows]
        ax.plot(removed, [r["shap_mean_probability"] for r in rows], label="SHAP order")
        ax.plot(removed, [r["random_mean_probability"] for r in rows], label="Random order (20)")
        ax.axvline(EARLY_REMOVALS, color="0.7", linestyle=":", linewidth=1)
        ax.set(title=f'{info["true_class"]} ({info["sampled_count"]}/{info["test_support"]})', ylim=(0, 1.02))
    for ax in list(axes.flat)[len(class_rows):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.supxlabel("Features set to zero in train-fitted z-space")
    fig.supylabel("Mean original predicted-class probability")
    save(fig, PLOTS[2])


def _run_analysis(data, output, load_model):
    claim_path = output / "analysis.json"
    versions = {name: version(name) for name in ("numpy", "shap", "torch", "matplotlib")}
    versions["python"] = platform.python_version()
    implementation = {
        name: sha256(Path(__file__).with_name(name))
        for name in ("final_analysis.py", "explain.py")
    }
    inputs = {
        "settings": SETTINGS, "versions": versions,
        "provenance": {**data.provenance, "implementation_sha256": implementation},
        "feature_names": data.feature_names, "class_names": data.class_names,
        "sample_indices": data.sample_indices.tolist(), "background_train_indices": data.background_indices.tolist(),
    }
    signature = digest(inputs)
    if claim_path.is_file():
        claim = read_json(claim_path)
        if claim.get("input_signature") != signature:
            raise RuntimeError(f"Analysis input signature differs at {output}; use a separate output directory")
        if claim.get("status") == "complete":
            checksums = claim.get("output_sha256", {})
            if checksums.keys() != set(OUTPUT_FILES) or any(
                not (output / name).is_file() or sha256(output / name) != sha for name, sha in checksums.items()
            ):
                raise RuntimeError("Completed analysis has missing or altered outputs; preserving it for inspection")
            print(f"Reusing verified completed analysis: {output}", flush=True)
            return claim
        if claim.get("status") != "in_progress":
            raise RuntimeError(f"Invalid analysis claim status: {claim.get('status')}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        if any(path.name != "analysis.json.partial" for path in output.iterdir()):
            raise RuntimeError(f"Unclaimed output directory contains files: {output}")
        claim = {
            **inputs, "input_signature": signature, "label": LABEL, "status": "in_progress", "started_at": utc_now(),
        }
        write_json(claim_path, claim)

    phi, attribution = _attribution_stage(output / SHAP_CACHE, signature, data, load_model)
    shap, random, deletion = _deletion_stage(output / DELETION_CACHE, signature, data, phi, load_model)
    report_start = time.perf_counter()
    class_rows = [
        {"class_index": c, "true_class": name, "test_support": int(data.support[c]),
         "sampled_count": int(np.count_nonzero(data.true_labels == c)),
         "predicted_correct_count": int(np.count_nonzero((data.true_labels == c) & (data.predicted == c)))}
        for c, name in enumerate(data.class_names)
    ]
    tables = _write_tables(output, data, phi, shap, random, class_rows)
    _write_plots(output, tables, class_rows)
    final = {
        **claim, "status": "complete", "finished_at": utc_now(), "sample_size": len(data.inputs),
        "background_size": len(data.background), "class_summary": class_rows, "attribution": attribution,
        "earlier_random_sample_composition": data.earlier_composition,
        "timing_seconds": {"attribution": attribution["seconds"], "deletion": deletion["seconds"],
                           "tables_and_plots": time.perf_counter() - report_start},
        "limitations": [
            "Classes with few test flows have correspondingly limited evidence.",
            "Zero masking in z-space is a model perturbation. Its probability changes are descriptive and can be nonmonotonic.",
            "Deletion tracks the original predicted class even for misclassified flows.",
        ],
        "output_sha256": {name: sha256(output / name) for name in OUTPUT_FILES},
    }
    write_json(claim_path, final)
    return final


def run_stratified_shap(root, output=None):
    """Run or resume the declared CPU final analysis under the shared stage lock."""
    root = Path(root).resolve()
    output = (root / (DEFAULT_OUTPUT if output is None else Path(output))).resolve()
    protected = (root / "results", root / "data", root / ".git")
    if output in (root, root / "artifacts") or any(output.is_relative_to(path) for path in protected):
        raise ValueError("Choose a separate final-analysis output directory outside original results and data")
    with execution_lock(root, "full"), configure_single_thread():
        data = _load_analysis_inputs(root)
        model = None

        def load_model():
            nonlocal model
            if model is None:
                model = _load_run_model("exlightnet", len(data.class_names), root / RUN_DIRECTORY)
            return model

        return _run_analysis(data, output, load_model)
