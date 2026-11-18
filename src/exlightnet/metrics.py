"""Final-test metrics with an on-disk evaluation-once receipt."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .io import read_json, write_json

SCALARS = ("accuracy", "macro_precision", "macro_recall", "weighted_f1", "macro_f1", "auc", "attack_recall", "miss_count", "fp_count", "fpr", "params", "serialized_bytes", "latency_mean_ms", "latency_median_ms", "latency_p95_ms")


def score_predictions(y, probabilities, class_names):
    p = np.asarray(probabilities)
    classes = len(class_names)
    if p.shape != (len(y), classes) or not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid probability matrix")
    pred = p.argmax(1)
    labels = np.arange(classes)
    report = classification_report(y, pred, labels=labels, target_names=class_names, zero_division=0, output_dict=True)
    auc = float(roc_auc_score(y, p[:, 1])) if classes == 2 else float(roc_auc_score(np.eye(classes)[y].ravel(), p.ravel()))
    result = {"accuracy": float(accuracy_score(y, pred)), "macro_precision": float(precision_score(y, pred, labels=labels, average="macro", zero_division=0)),
              "macro_recall": float(recall_score(y, pred, labels=labels, average="macro", zero_division=0)),
              "weighted_f1": float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
              "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
              "auc": auc, "auc_definition": "p_attack" if classes == 2 else "micro OvR",
              "per_class": {n: {"precision": report[n]["precision"], "recall": report[n]["recall"], "f1": report[n]["f1-score"], "support": int(report[n]["support"])} for n in class_names},
              "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist()}
    if classes == 2:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=labels).ravel()
        result.update(attack_recall=float(tp / (tp + fn)), miss_count=int(fn), fp_count=int(fp), fpr=float(fp / (fp + tn)), threshold=.5,
                      decision_rule="argmax of two softmax logits; ties select normal")
    return result


def read_evaluation_cache(directory):
    directory = Path(directory)
    missing = [name for name in ("metrics.json", "predictions.npz", "test_evaluation_receipt.json") if not (directory / name).exists()]
    if missing:
        raise RuntimeError(f"Incomplete cached evaluation, missing evidence/receipt: {missing}")
    result = read_json(directory / "metrics.json")
    receipt = read_json(directory / "test_evaluation_receipt.json")
    if receipt.get("status") != "complete" or receipt.get("metadata", {}).get("fingerprint") != result.get("fingerprint"):
        raise RuntimeError("Incomplete or inconsistent evaluation receipt; test inference will not be repeated")
    return result


def evaluate_once(directory, y, predictor, class_names, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "metrics.json"
    if path.exists():
        result = read_evaluation_cache(directory)
        if result.get("fingerprint") != metadata.get("fingerprint"):
            raise RuntimeError("Cached result fingerprint differs; use a new output directory")
        return result
    claim = directory / "test_evaluation_receipt.json"
    try:
        with claim.open("x") as stream:
            json.dump({"status": "claimed", "metadata": metadata}, stream)
    except FileExistsError as exc:
        raise RuntimeError(f"Test evaluation already claimed at {claim}; interrupted evaluations are not silently repeated") from exc
    p = predictor()
    result = {**metadata, **score_predictions(y, p, class_names)}
    np.savez_compressed(directory / "predictions.npz", y=y, probabilities=p)
    write_json(path, result)
    write_json(claim, {"status": "complete", "metadata": metadata, "test_prediction_passes": 1})
    return result


def summarize(rows):
    import pandas as pd
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["model"]), []).append(row)
    output = []
    for (task, model), group in sorted(groups.items()):
        seeds = sorted(r["seed"] for r in group)
        result = {"task": task, "model": model, "n_seeds": len(group), "seeds": ",".join(map(str, seeds)), "five_seed_complete": seeds == list(range(42, 47))}
        for metric in SCALARS:
            values = [r[metric] for r in group if r.get(metric) is not None]
            result[metric + "_n"] = len(values)
            result[metric + "_mean"] = float(np.mean(values)) if values else None
            result[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            result[metric + "_seed42"] = next((r.get(metric) for r in group if r["seed"] == 42), None)
        output.append(result)
    return pd.DataFrame(output, dtype=object)
