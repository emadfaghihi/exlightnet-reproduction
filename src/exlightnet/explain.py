"""FP32-logit SHAP, measured additivity, deletion and W8 stability."""
import copy
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .io import write_json
from .train import predict_proba


def normalize_shap_values(values, samples, features, classes):
    if isinstance(values, list):
        result = np.stack([np.asarray(v).reshape(samples, features) for v in values], axis=-1)
    else:
        result = np.asarray(values)
        if result.shape == (classes, samples, features) and result.shape != (samples, features, classes):
            result = result.transpose(1, 2, 0)
        else:
            result = result.reshape(samples, features, classes)
    if result.shape != (samples, features, classes) or not np.isfinite(result).all():
        raise ValueError(f"Unsupported/nonfinite SHAP shape {result.shape}")
    return result


def explain_attributions(model, background, inputs, nsamples=200):
    import shap
    background = torch.as_tensor(background, dtype=torch.float32)
    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    model = copy.deepcopy(model).eval()
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            assert not layer.training
    np.random.seed(42)
    torch.manual_seed(42)
    with torch.no_grad():
        outputs = model(inputs).numpy()
        expected = model(background).mean(0).numpy()
    fallback = None
    preflight_samples = 2 if len(inputs) > 2 else 0
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        try:
            explainer = shap.DeepExplainer(copy.deepcopy(model), background)
            if preflight_samples:
                # Detect unsupported operators before repeating doomed DeepLIFT
                # passes over the full set. The requested fallback is unchanged.
                explainer.shap_values(inputs[:preflight_samples], check_additivity=True)
            raw = explainer.shap_values(inputs, check_additivity=True)
            method = "DeepExplainer"
        except Exception as exc:
            fallback = f"{type(exc).__name__}: {exc}"
            # Use a clean copy: a failed DeepExplainer may leave module hooks.
            explainer = shap.GradientExplainer(copy.deepcopy(model), background)
            raw = explainer.shap_values(inputs, nsamples=nsamples, rseed=42)
            method = "GradientExplainer"
    values = normalize_shap_values(raw, len(inputs), inputs.shape[-1], outputs.shape[1])
    residual = values.sum(axis=1) + expected[None, :] - outputs
    return {"values": values, "expected_logits": expected, "method": method, "fallback_reason": fallback,
            "deep_additivity_preflight_samples": preflight_samples,
            "warnings": sorted(set(str(w.message) for w in captured)), "nsamples_gradient": nsamples,
            "additivity_max_abs": float(np.abs(residual).max()), "additivity_mean_abs": float(np.abs(residual).mean()),
            "additivity_relative_mean": float(np.mean(np.abs(residual) / (np.abs(outputs - expected) + 1e-6))),
            "output_space": "logits; additivity checked separately for each class"}


def deletion_curve(model, inputs, predicted, order, *, original_probabilities=None):
    n, width = inputs.shape
    perturbed = np.repeat(inputs[:, None, :], width + 1, axis=1)
    for k in range(1, width + 1):
        perturbed[:, k, :] = perturbed[:, k - 1, :]
        perturbed[np.arange(n), k, order[:, k - 1]] = 0.
    start = int(original_probabilities is not None)
    steps = width + 1 - start
    probs = predict_proba(model, perturbed[:, start:].reshape(-1, width)).reshape(n, steps, -1)
    mean = probs[np.arange(n)[:, None], np.arange(steps)[None, :], predicted[:, None]].mean(0)
    if original_probabilities is not None:
        return np.concatenate((np.mean(original_probabilities, dtype=np.float64, keepdims=True), mean))
    return mean


def run_explanations(model, quantized, x_train, x_test, y_test, cached_probabilities, class_names, feature_names, directory, fast=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rng = np.random.default_rng(42)
    bg_indices = rng.choice(len(x_train), min(16 if fast else 200, len(x_train)), replace=False)
    indices = rng.choice(len(x_test), min(12 if fast else 300, len(x_test)), replace=False)
    locals_ = []
    pred_all = cached_probabilities.argmax(1)
    for label in ("dos_syn_hping", "nmap_os_detection"):
        c = class_names.index(label)
        eligible = np.flatnonzero((y_test == c) & (pred_all == c)) if label == "dos_syn_hping" else np.flatnonzero(y_test == c)
        if len(eligible):
            locals_.append((label, int(eligible[0])))
    combined = np.concatenate([indices, np.array([i for _, i in locals_], dtype=int)])
    inputs = x_test[combined]
    fp = explain_attributions(model, x_train[bg_indices], inputs, nsamples=64 if fast else 200)
    quant = explain_attributions(quantized, x_train[bg_indices], x_test[indices], nsamples=64 if fast else 200) if quantized is not None else {"status": "not run: Q1 failed or W8 branch unavailable"}
    values = fp.pop("values")
    qvalues = quant.pop("values") if quantized is not None else np.empty((0, len(feature_names), len(class_names)))
    global_values = np.abs(values[:len(indices)]).mean(axis=(0, 2))
    qglobal = np.abs(qvalues).mean(axis=(0, 2)) if quantized is not None else np.empty(0)
    order = np.argsort(-global_values, kind="stable")
    qorder = np.argsort(-qglobal, kind="stable")
    top15 = order[:15]
    rank_fp = np.argsort(order)
    rank_q = np.argsort(qorder)
    rho = float(spearmanr(rank_fp[top15], rank_q[top15]).statistic) if quantized is not None else None
    pred = pred_all[indices]
    predicted_values = values[np.arange(len(indices)), :, pred]
    sorted_features = np.argsort(-np.abs(predicted_values), axis=1)
    deletion = deletion_curve(model, x_test[indices], pred, sorted_features)
    random_curves = [deletion_curve(model, x_test[indices], pred, np.tile(rng.permutation(len(feature_names)), (len(indices), 1))) for _ in range(20)]
    random_mean = np.mean(random_curves, axis=0)
    width = len(feature_names)
    local_rows = []
    for j, (label, index) in enumerate(locals_):
        p = int(pred_all[index])
        local_values = values[len(indices) + j, :, p]
        top = np.argsort(-np.abs(local_values))[:5]
        local_rows.append({"true_class": label, "predicted_class": class_names[p], "correct": bool(y_test[index] == p), "test_position": index,
                           "probability": float(cached_probabilities[index, p]),
                           "top5": [{"feature": feature_names[k], "signed_logit_phi": float(local_values[k]), "z_value": float(x_test[index, k])} for k in top]})
    result = {"fp32": fp, "dequantized_int8": quant, "background_size": len(bg_indices), "random_test_size": len(indices),
              "test_indices": indices, "background_train_indices": bg_indices, "global": [{"rank": j + 1, "feature": feature_names[k], "mean_abs_phi": float(global_values[k])} for j, k in enumerate(order)],
              "quantization_stability": {"spearman_top15_fp32_feature_set": rho, "top5_overlap": len(set(order[:5]) & set(qorder[:5])) if quantized is not None else None},
              "deletion": {"mean_probability_curve": deletion, "random_mean_probability_curve": random_mean,
                           "probability_after_removals": {str(k): float(deletion[k]) for k in (0, 10, 20)},
                           "normalized_auc": float(np.trapz(deletion, dx=1 / width)), "random_normalized_auc": float(np.trapz(random_mean, dx=1 / width)),
                           "definition": "probability AUC integrated over fraction of all features removed (0..1); not divided by starting probability",
                           "random_permutations": 20, "mask_value": "zero in train-fitted z-space"},
              "local_examples": local_rows, "seconds": time.perf_counter() - start}
    np.savez_compressed(directory / "shap_values.npz", phi=values, quantized_phi=qvalues, global_importance=global_values, quantized_importance=qglobal, sample_indices=combined)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(np.arange(len(top15)), global_values[top15][::-1], color="#176b87")
    ax.set_yticks(np.arange(len(top15)), [feature_names[k] for k in top15[::-1]])
    ax.set_xlabel("Mean |SHAP| over random test flows and classes (logits)")
    fig.tight_layout()
    fig.savefig(directory / "global_shap.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for j, k in enumerate(order[:10][::-1]):
        vals = x_test[indices, k]
        color = (vals - vals.min()) / max(float(np.ptp(vals)), 1e-9)
        points = ax.scatter(predicted_values[:, k], j + rng.uniform(-.2, .2, len(indices)), c=color, cmap="coolwarm", vmin=0, vmax=1, s=16, alpha=.8)
    ax.set_yticks(range(10), [feature_names[k] for k in order[:10][::-1]])
    ax.set_xlabel("SHAP contribution to predicted-class logit")
    fig.colorbar(points, ax=ax, label="Feature value (low → high, within sample)")
    fig.tight_layout()
    fig.savefig(directory / "beeswarm.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(width + 1), deletion, label="SHAP order")
    ax.plot(np.arange(width + 1), random_mean, label="Random (20 permutations)")
    ax.set(xlabel="Features set to zero in z-space", ylabel="Mean original predicted-class probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "deletion.png", dpi=160)
    plt.close(fig)
    # This JSON is the completion receipt; write it only after every plot exists.
    write_json(directory / "explanations.json", result)
    return result
