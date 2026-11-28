"""Reproduce the post-hoc feature-set diagnostics with validation-only scoring."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from exlightnet.baselines import build_model
from exlightnet.data import load_data, validate_data
from exlightnet.io import read_json, sha256
from exlightnet.preprocess import FlowPreprocessor


PAIR = ("nmap_os_detection", "nmap_xmas_tree_scan")
FLAGS = ("fwd_psh_flag_count", "flow_fin_flag_count", "fwd_urg_flag_count")
BASE_SEEDS = tuple(range(42, 47))
RANKED_SEEDS = tuple(range(42, 45))


def load_partitions(path, n_rows):
    with np.load(path, allow_pickle=False) as split:
        train, validation = split["train"], split["val"]
    for indices in (train, validation):
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or not indices.size
                or indices.min() < 0 or indices.max() >= n_rows
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("Invalid or repeated row indices in the saved split")
    if np.intersect1d(train, validation).size:
        raise ValueError("Train and validation row indices overlap")
    return train, validation


def opposite_label_matches(train_x, train_y, validation_x, validation_y):
    """Find each validation row with an exactly equal opposite-class train row."""
    left, right = np.unique(train_y)
    vectors = {label: set(map(tuple, train_x[train_y == label])) for label in (left, right)}
    return np.array([
        tuple(row) in vectors[right if label == left else left]
        for row, label in zip(validation_x, validation_y, strict=True)
    ], dtype=bool)


def train_forest(x, y, seed, jobs, description):
    print(f"RF {description} seed={seed}", file=sys.stderr, flush=True)
    forest = build_model("random_forest", len(np.unique(y)), seed=seed)
    return forest.set_params(n_jobs=jobs).fit(x, y)


def validation_scores(train, validation, columns, seeds, jobs, description, seed42_forest=None):
    train_x, train_y = train
    validation_x, validation_y = validation
    train_x = train_x[:, columns]
    validation_x = validation_x[:, columns]
    labels = np.unique(train_y)
    rows = []
    for seed in seeds:
        forest = seed42_forest if seed == 42 and seed42_forest is not None else train_forest(
            train_x, train_y, seed, jobs, description
        )
        predicted = forest.predict(validation_x)
        _, _, f1, _ = precision_recall_fscore_support(
            validation_y, predicted, labels=labels, zero_division=0
        )
        rows.append({"seed": seed, "macro_f1": float(f1.mean()), "per_class_f1": f1.tolist()})
    return rows


def pair_diagnostics(train, validation, features, feature_sets, classes, jobs):
    pair_labels = np.searchsorted(classes, PAIR)
    train_x, train_y = train
    validation_x, validation_y = validation
    in_train = np.isin(train_y, pair_labels)
    in_validation = np.isin(validation_y, pair_labels)
    pair_train = train_x[in_train], train_y[in_train]
    pair_validation = validation_x[in_validation], validation_y[in_validation]
    pair_x, pair_y = pair_train

    # Either threshold direction can separate the pair. Fix Xmas as the
    # positive label and retain raw AUCs alongside the direction-free scores.
    positive = pair_y == pair_labels[1]
    auc = {
        features[i]: float(roc_auc_score(positive, pair_x[:, i]))
        for i in [*feature_sets["F40"], *(features.index(flag) for flag in FLAGS)]
    }
    separation = {name: max(value, 1 - value) for name, value in auc.items()}
    best_f40 = max((features[i] for i in feature_sets["F40"]), key=separation.__getitem__)
    return {
        "classes": list(PAIR),
        "train_support": [int(np.count_nonzero(pair_y == label)) for label in pair_labels],
        "validation_support": [int(np.count_nonzero(pair_validation[1] == label)) for label in pair_labels],
        "raw_train_auc": auc,
        "train_separation_auc": separation,
        "best_f40_auc_feature": best_f40,
        "sets": {
            name: {
                "opposite_label_matches": int(opposite_label_matches(
                    pair_x[:, columns], pair_y, pair_validation[0][:, columns], pair_validation[1]
                ).sum()),
                "rows": validation_scores(pair_train, pair_validation, columns, BASE_SEEDS, jobs, f"pair {name}"),
            }
            for name, columns in feature_sets.items()
        },
    }


def measure(data_path, split_path, cache_manifest, jobs):
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    start = perf_counter()
    canonical = read_json(cache_manifest)
    data_sha = sha256(data_path)
    if data_sha != canonical["settings"]["data_sha256"]:
        raise ValueError("CSV SHA-256 differs from the recorded RT-IoT2022 input")
    frame, labels, binary = load_data(data_path, download=False)
    data_gate = validate_data(labels)
    train_indices, validation_indices = load_partitions(split_path, len(frame))
    train_frame = frame.iloc[train_indices]
    validation_frame = frame.iloc[validation_indices]
    train_binary = binary[train_indices]
    classes, train_y = np.unique(labels[train_indices], return_inverse=True)
    validation_y = np.searchsorted(classes, labels[validation_indices])
    if (set(classes) != data_gate["actual"]["counts"].keys()
            or not np.array_equal(np.unique(validation_y), np.arange(len(classes)))):
        raise ValueError("Train and validation must both contain every RT-IoT2022 class")

    print("Fitting canonical binary MI and encoding train and validation", file=sys.stderr, flush=True)
    preprocessor = FlowPreprocessor(top_k=40).fit(train_frame, train_binary)
    features = preprocessor.candidate_names_
    if len(features) != 93:
        raise ValueError(f"Expected 93 train-varying encoded columns, got {len(features)}")
    train_x = preprocessor.encode(train_frame)
    validation_x = preprocessor.encode(validation_frame)
    train, validation = (train_x, train_y), (validation_x, validation_y)
    feature_sets = {"F40": preprocessor.selected_indices_, "F93": list(range(len(features)))}
    print("Fitting 12-class MI with the same preprocessing recipe", file=sys.stderr, flush=True)
    multiclass_mi = FlowPreprocessor(top_k=40).fit(train_frame, train_y).ranking_
    pair = pair_diagnostics(train, validation, features, feature_sets, classes, jobs)

    # This forest has the identical inputs and recipe as the F93 seed-42
    # validation fit, so its trained state serves both diagnostics.
    multiclass_forest = train_forest(train_x, train_y, 42, jobs, "multiclass F93 and importance")
    binary_forest = train_forest(train_x, train_binary, 42, jobs, "binary importance")
    importance = {
        "multiclass": np.argsort(-multiclass_forest.feature_importances_, kind="stable"),
        "binary": np.argsort(-binary_forest.feature_importances_, kind="stable"),
    }
    del binary_forest
    multiclass = {
        name: validation_scores(
            train, validation, columns, BASE_SEEDS, jobs, f"multiclass {name}",
            seed42_forest=multiclass_forest if name == "F93" else None,
        )
        for name, columns in feature_sets.items()
    }
    del multiclass_forest
    ranked = {
        target: {
            width: validation_scores(train, validation, order[:width], RANKED_SEEDS, jobs,
                                     f"multiclass with {target}-ranked top-{width}")
            for width in (40, 20, 8)
        }
        for target, order in importance.items()
    }
    return {
        "status": "post-hoc diagnostic",
        "training_partition": "train",
        "scoring_partition": "validation",
        "started_utc": started_at,
        "wall_seconds": perf_counter() - start,
        "device": "cpu",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "torch")},
        "rf_n_jobs": jobs,
        "data": {
            "csv_sha256": data_sha,
            "cpu_v1_cache_fingerprint": canonical["fingerprint"],
            "split_indices_sha256": sha256(split_path),
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "encoded_dtype": train_x.dtype.name,
            "candidate_count_before_pruning": preprocessor.candidate_count_before_pruning_,
            "dropped_constant": preprocessor.dropped_constant_,
            "candidate_names": features,
        },
        "pair": pair,
        "multiclass": {
            "classes": classes.tolist(),
            "validation_support": np.bincount(validation_y, minlength=len(classes)).tolist(),
            "sets": multiclass,
        },
        "ranking": {
            "binary_mi": [row["feature"] for row in preprocessor.ranking_],
            "multiclass_mi": [row["feature"] for row in multiclass_mi],
            **{target + "_rf": [features[i] for i in order] for target, order in importance.items()},
        },
        "ranked_multiclass": ranked,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/RT_IOT2022.csv"))
    parser.add_argument("--split-indices", type=Path, default=Path("artifacts/split_indices.npz"))
    parser.add_argument("--cache-manifest", type=Path, default=Path("artifacts/cache_manifest.json"))
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    result = measure(args.data, args.split_indices, args.cache_manifest, args.jobs)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
