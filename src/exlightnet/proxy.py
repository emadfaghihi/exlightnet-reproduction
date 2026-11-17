"""Optional whole-cluster leakage stress PROXY; never a campaign-disjoint split."""
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from .data import GateFailure, load_data
from .io import read_json, write_json
from .metrics import summarize
from .preprocess import FlowPreprocessor


def cluster_group_split(x_z, labels):
    rng = np.random.default_rng(42)
    partitions = {"train": [], "val": [], "test": []}
    groups = np.empty(len(labels), dtype=np.int64)
    next_group = 0
    class_evidence = {}
    with threadpool_limits(1):
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            clusters = min(10, max(3, len(indices) // 10))
            assignment = MiniBatchKMeans(n_clusters=clusters, random_state=42, batch_size=1024, n_init=3).fit_predict(x_z[indices])
            cluster_ids, sizes = np.unique(assignment, return_counts=True)
            if len(cluster_ids) < 3:
                raise GateFailure("PROXY_GROUPS", {"class": str(label), "distinct_clusters": len(cluster_ids), "required": 3})
            order = sorted(rng.permutation(len(sizes)), key=lambda k: -sizes[k])
            assigned_sizes = np.zeros(3)
            targets = np.array([.70, .15, .15]) * len(indices)
            class_sizes = np.zeros(3, dtype=int)
            for step, k in enumerate(order):
                empty = np.flatnonzero(class_sizes == 0)
                if len(order) - step == len(empty):
                    destination = int(empty[0])
                else:
                    # Minimize squared target error with whole clusters, never move individual flows.
                    errors = [np.sum((assigned_sizes + np.eye(3)[d] * sizes[k] - targets) ** 2) for d in range(3)]
                    destination = int(np.argmin(errors))
                selected = indices[assignment == cluster_ids[k]]
                groups[selected] = next_group
                next_group += 1
                partitions[("train", "val", "test")[destination]].extend(selected.tolist())
                assigned_sizes[destination] += len(selected)
                class_sizes[destination] += len(selected)
            class_evidence[str(label)] = dict(zip(("train", "val", "test"), class_sizes.tolist()))
    split = {k: np.asarray(v, dtype=int) for k, v in partitions.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert set(groups[split[a]]).isdisjoint(groups[split[b]])
    return split, groups, {"label": "PROXY; not campaign-disjoint", "sizes": {k: len(v) for k, v in split.items()}, "class_sizes": class_evidence,
                           "grouping_features": "paper-style train-fitted top-40 z-scores; used only to define groups", "kmeans_seed": 42}


def run_proxy(root, arrays, paper_preprocessor, selections, full_directory):
    from .run_all import _run_one, choose_alpha, digest
    root, full_directory = Path(root), Path(full_directory)
    if not (full_directory / "results.json").exists():
        raise RuntimeError("Run ordinary results first; the optional proxy is last")
    x, labels, binary = load_data(root / "data/RT_IOT2022.csv", download=False)
    class_names, y12 = np.unique(labels, return_inverse=True)
    split, groups, evidence = cluster_group_split(paper_preprocessor.transform(x), y12)
    directory = full_directory / "leakage_proxy"
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "groups_and_split.npz", groups=groups, **split)
    pp = FlowPreprocessor(40).fit(x.iloc[split["train"]], binary[split["train"]])
    pp.save(directory / "preprocessing")
    proxy_arrays = {}
    for key, index in split.items():
        proxy_arrays.update({"x_" + key: pp.transform(x.iloc[index]), "y12_" + key: y12[index]})
    context = {"fingerprint": digest({"split": {k: v.tolist() for k, v in split.items()}, "preprocessing": pp.to_dict()})}
    selection = choose_alpha("multiclass", proxy_arrays["x_train"], proxy_arrays["y12_train"], proxy_arrays["x_val"], proxy_arrays["y12_val"],
                             directory / "alpha", fast=False, fingerprint=context["fingerprint"])
    rows = []
    for name in ("exlightnet", "mlp", "std_cnn", "random_forest", "xgboost"):
        for seed in (42, 43, 44):
            rows.append(_run_one("multiclass", name, seed, proxy_arrays, class_names.tolist(), selection, context, directory / name / str(seed), 30))
    baseline = read_json(full_directory / "results.json")
    comparisons = []
    for name in ("exlightnet", "mlp", "std_cnn", "random_forest", "xgboost"):
        current = {r["seed"]: r["macro_f1"] for r in rows if r["model"] == name}
        reference = {r["seed"]: r["macro_f1"] for r in baseline if r["task"] == "multiclass" and r["model"] == name}
        differences = [current[s] - reference[s] for s in (42, 43, 44) if s in reference]
        comparisons.append({"model": name, "proxy_minus_random_mean_macro_f1": float(np.mean(differences)) if differences else None,
                            "paired_difference_std": float(np.std(differences, ddof=1)) if len(differences) > 1 else None})
    summarize(rows).to_csv(directory / "summary.csv", index=False)
    write_json(directory / "results.json", rows)
    write_json(directory / "proxy_report.json", {**evidence, "comparisons": comparisons, "warning": "Clustering is a PROXY for related flows, not known attack-run/session metadata"})
