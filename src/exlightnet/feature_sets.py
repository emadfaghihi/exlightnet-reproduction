"""Train-only selectors for the feature_sets_v1 protocol."""

import logging
from math import lcm

import numpy as np
from scipy.stats import rankdata

from .baselines import build_model


CAMPAIGN = "feature_sets_v1"
FLAGS = ("fwd_psh_flag_count", "flow_fin_flag_count", "fwd_urg_flag_count")
RF_SEEDS = (42, 43, 44, 45, 46)
PEARSON_CUTOFF = 0.95
LOGGER = logging.getLogger(__name__)


def auc_pair_counts(train_x, train_y, label_ids):
    """Return doubled Mann–Whitney U and doubled positive-negative pair counts.

    Divide each class's U row by its pair count for the one-vs-rest ROC-AUC.
    Average ranks count each tied pair as half a win.
    """
    positive = train_y[:, None] == label_ids
    n_positive = positive.sum(axis=0)
    n_negative = len(train_y) - n_positive
    if np.any((n_positive == 0) | (n_negative == 0)):
        raise ValueError("Each one-vs-rest AUC needs positives and negatives on train")
    ranks = rankdata(train_x, method="average", axis=0)
    twice_rank_sums = (2 * (positive.T @ ranks)).astype(np.int64)
    wins = twice_rank_sums - (n_positive * (n_positive + 1))[:, None]
    return wins, 2 * n_positive * n_negative


def class_aware_quotas(train_x, train_y12, candidate_names, class_names):
    """Select three features per class and four more by macro separation."""
    wins, pairs = auc_pair_counts(train_x, train_y12, np.arange(len(class_names)))
    # Take the complement before division so equal separation in either
    # direction ties exactly. Each class ranks by its common denominator.
    separation_wins = np.maximum(wins, pairs[:, None] - wins)
    rankings = np.argsort(-separation_wins, axis=1, kind="stable").tolist()
    label_order = sorted(range(len(class_names)), key=class_names.__getitem__)
    selected = set()
    log = []
    for round_number in range(1, 4):
        for label in label_order:
            column = next(i for i in rankings[label] if i not in selected)
            selected.add(column)
            log.append({
                "round": round_number, "label": class_names[label], "column_index": column,
                "feature": candidate_names[column], "separation": float(separation_wins[label, column] / pairs[label]),
            })

    # A common denominator lets mathematically equal macro scores tie exactly.
    # Its weights can exceed int64, so only this small reduction uses Python ints.
    denominator = lcm(*pairs.tolist())
    weights = np.array([denominator // count for count in pairs.tolist()], dtype=object)
    macro_wins = weights @ separation_wins.astype(object)
    remaining = [i for i in range(len(candidate_names)) if i not in selected]
    for column in sorted(remaining, key=macro_wins.__getitem__, reverse=True)[:4]:
        log.append({
            "round": 4, "label": "macro", "column_index": column,
            "feature": candidate_names[column], "separation": macro_wins[column] / (denominator * len(class_names)),
        })
    return log


def pearson_filter(train_x, ranking, candidate_names):
    """Walk the ranking and record the first retained feature blocking each removal."""
    centered = train_x - train_x.mean(axis=0)
    products = centered.T @ centered
    sums_of_squares = products.diagonal()
    if not np.isfinite(products).all() or np.any(sums_of_squares <= 0):
        raise ValueError("Pearson filtering needs finite, nonconstant train columns")
    abs_correlations = np.abs(products) / np.sqrt(np.outer(sums_of_squares, sums_of_squares))
    retained, removals = [], []
    for column in ranking:
        for earlier_column in retained:
            abs_r = abs_correlations[column, earlier_column]
            if abs_r > PEARSON_CUTOFF:
                removals.append({
                    "feature": candidate_names[column], "retained_counterpart": candidate_names[earlier_column],
                    "abs_r": min(float(abs_r), 1.0),
                })
                break
        else:
            retained.append(column)
    return retained, removals


def select_feature_sets(train_x, train_y12, candidate_names, f40_indices, class_names, *, rf_n_jobs=1):
    """Freeze column order for both tasks from only the encoded training partition."""
    if (train_x.ndim != 2 or train_x.dtype != np.float64 or train_x.shape[1] != 93
            or len(candidate_names) != 93 or len(set(candidate_names)) != 93 or not np.isfinite(train_x).all()):
        raise ValueError("Expected 93 distinct, finite float64 train columns in candidate order")
    if (len(class_names) != 12 or len(set(class_names)) != 12 or train_y12.shape != (len(train_x),)
            or train_y12.dtype.kind not in "iu" or not np.array_equal(np.unique(train_y12), np.arange(12))):
        raise ValueError("Expected canonical labels for all 12 classes on train")
    f40 = list(f40_indices)
    if len(f40) != 40 or len(set(f40)) != 40 or not set(f40).issubset(range(93)):
        raise ValueError("Expected 40 distinct F40 columns from binary MI")
    flag_columns = {name: candidate_names.index(name) for name in FLAGS}
    if set(f40).intersection(flag_columns.values()):
        raise ValueError("The three F43 flags must be outside the canonical F40")
    if rf_n_jobs < 1:
        raise ValueError("rf_n_jobs must be positive")

    importances = {}
    for seed in RF_SEEDS:
        LOGGER.info("Fitting train-only Random Forest for importance (seed %d)", seed)
        forest = build_model("random_forest", len(class_names), seed=seed).set_params(n_jobs=rf_n_jobs)
        forest.fit(train_x, train_y12)
        importances[str(seed)] = forest.feature_importances_
        del forest
    mean = np.mean(list(importances.values()), axis=0)
    mean_ranking = np.argsort(-mean, kind="stable").tolist()
    per_seed_ranking = {seed: np.argsort(-scores, kind="stable").tolist() for seed, scores in importances.items()}
    retained, removals = pearson_filter(train_x, mean_ranking, candidate_names)
    retained_ranks = {column: rank for rank, column in enumerate(retained, start=1)}
    flag_ranks = {
        name: {
            "mean_rank": mean_ranking.index(column) + 1,
            "per_seed_ranks": {seed: ranking.index(column) + 1 for seed, ranking in per_seed_ranking.items()},
            "retained_rank": retained_ranks.get(column),
        }
        for name, column in flag_columns.items()
    }

    LOGGER.info("Computing training ROC-AUC quotas")
    quota_log = class_aware_quotas(train_x, train_y12, candidate_names, class_names)
    stopped_arms = {}
    sets = {
        "F40": f40, "F43": f40 + list(flag_columns.values()), "F93": list(range(93)),
        "F_RF40": retained[:40], "F_RF8": retained[:8],
        "F_CA40": [row["column_index"] for row in quota_log],
    }
    for arm, width in (("F_RF40", 40), ("F_RF8", 8)):
        if len(retained) < width:
            stopped_arms[arm] = f"Only {len(retained)} features retained out of {width} required"
            sets[arm] = []
    return {
        "feature_sets": {
            name: {"status": "STOPPED" if name in stopped_arms else "READY",
                   "column_indices": columns, "feature_names": [candidate_names[i] for i in columns]}
            for name, columns in sets.items()
        },
        "rf": {
            "n_jobs": rf_n_jobs, "seeds": list(RF_SEEDS),
            "mean_importances": mean.tolist(),
            "per_seed_importances": {seed: scores.tolist() for seed, scores in importances.items()},
            "mean_ranking": [candidate_names[i] for i in mean_ranking],
            "pearson_cutoff": PEARSON_CUTOFF,
            "retained": [candidate_names[i] for i in retained],
            "removals": removals,
            "flag_ranks": flag_ranks,
        },
        "class_aware_quota_log": quota_log,
        "stopped_arms": stopped_arms,
    }
