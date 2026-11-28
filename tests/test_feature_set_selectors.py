from collections import Counter
from inspect import signature

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score


@pytest.fixture
def train_inputs():
    names = [f"feature_{i:02}" for i in range(93)]
    names[40:43] = ["fwd_psh_flag_count", "flow_fin_flag_count", "fwd_urg_flag_count"]
    classes = [f"class_{i:02}" for i in range(12)]
    return np.repeat(np.arange(12), 5), names, classes


def test_rank_based_auc_matches_sklearn_with_ties_and_imbalanced_classes():
    from exlightnet.feature_sets import auc_pair_counts

    labels = np.array([0, 2, 0, 1, 0, 2, 1, 0, 0])
    scores = np.array([0, 1, 0, 3, 6, 2, 5, 0, 0], dtype=float)
    x = np.column_stack([scores, -scores, np.ones_like(scores), labels == 0])

    wins, pairs = auc_pair_counts(x, labels, np.arange(3))

    expected = [[roc_auc_score(labels == label, column) for column in x.T] for label in range(3)]
    np.testing.assert_allclose(wins / pairs[:, None], expected, rtol=0, atol=2e-16)


def test_pearson_filter_keeps_exact_cutoff_and_uses_first_retained_counterpart():
    from exlightnet.feature_sets import pearson_filter

    original = np.tile([1.0, -1.0], 40)
    boundary = original.copy()
    boundary[:2] *= -1  # Correlation of 76 / 80 with the original.
    independent = np.tile([1.0, 1.0, -1.0, -1.0], 20)
    x = np.column_stack([original, boundary, -original, original + boundary, independent])

    retained, removals = pearson_filter(
        x, [1, 0, 3, 2, 4], ["original", "boundary", "reverse", "combined", "independent"]
    )

    assert retained == [1, 0, 4]
    assert removals == [
        {"feature": "combined", "retained_counterpart": "boundary", "abs_r": pytest.approx(np.sqrt(0.975))},
        {"feature": "reverse", "retained_counterpart": "original", "abs_r": 1.0},
    ]


def test_quotas_take_36_distinct_class_picks_and_four_macro_picks_in_tie_order(train_inputs):
    from exlightnet.feature_sets import class_aware_quotas

    y, names, classes = train_inputs
    # Opposite score directions have exactly the same separation.
    x = np.tile(np.arange(len(y), dtype=float)[:, None], (1, len(names)))
    x[:, 1::2] *= -1

    log = class_aware_quotas(x, y, names, classes[::-1])

    assert [row["feature"] for row in log] == names[:40]
    assert [row["label"] for row in log] == classes * 3 + ["macro"] * 4
    assert Counter(row["round"] for row in log) == {1: 12, 2: 12, 3: 12, 4: 4}


def test_quotas_take_best_remaining_feature_for_each_class(train_inputs):
    from exlightnet.feature_sets import class_aware_quotas

    y, names, classes = train_inputs
    x = np.tile(np.tile(np.arange(5, dtype=float), 12)[:, None], (1, len(names)))
    x[:, 7] = y == 0
    x[:, 8] = y != 0
    x[:, 4] = y == 1

    log = class_aware_quotas(x, y, names, classes)

    assert [row["column_index"] for row in log[:2]] == [7, 4]
    assert [row["separation"] for row in log[:2]] == [1.0, 1.0]
    assert len({row["feature"] for row in log}) == 40


def test_train_only_selectors_are_deterministic_and_preserve_arm_order(train_inputs):
    from exlightnet.feature_sets import FLAGS, RF_SEEDS, select_feature_sets

    y, names, classes = train_inputs
    x = np.random.default_rng(19).normal(size=(len(y), len(names)))
    f40 = list(range(39, -1, -1))

    first = select_feature_sets(x, y, names, f40, classes, rf_n_jobs=2)
    second = select_feature_sets(x, y, names, f40, classes, rf_n_jobs=2)

    assert list(signature(select_feature_sets).parameters) == [
        "train_x", "train_y12", "candidate_names", "f40_indices", "class_names", "rf_n_jobs"
    ]
    assert first == second
    sets = first["feature_sets"]
    assert {name: len(arm["feature_names"]) for name, arm in sets.items()} == {
        "F40": 40, "F43": 43, "F93": 93, "F_RF40": 40, "F_RF8": 8, "F_CA40": 40
    }
    assert sets["F40"]["column_indices"] == f40
    assert sets["F43"]["feature_names"] == sets["F40"]["feature_names"] + list(FLAGS)
    assert sets["F93"]["feature_names"] == names
    assert sets["F_RF8"]["column_indices"] == sets["F_RF40"]["column_indices"][:8]
    assert first["stopped_arms"] == {}
    rf = first["rf"]
    assert list(rf["per_seed_importances"]) == [str(seed) for seed in RF_SEEDS]
    np.testing.assert_allclose(rf["mean_importances"], np.mean(list(rf["per_seed_importances"].values()), axis=0), rtol=5e-16)
    for flag, ranks in rf["flag_ranks"].items():
        assert ranks["mean_rank"] == rf["mean_ranking"].index(flag) + 1
        for seed, importances in rf["per_seed_importances"].items():
            ranking = sorted(range(len(names)), key=importances.__getitem__, reverse=True)
            assert ranks["per_seed_ranks"][seed] == ranking.index(names.index(flag)) + 1


@pytest.mark.parametrize("unique_columns", [1, 8])
def test_tied_rf_importances_keep_candidate_order_and_stop_only_short_arms(monkeypatch, train_inputs, unique_columns):
    from exlightnet import feature_sets

    y, names, classes = train_inputs
    distinct = np.random.default_rng(10).normal(size=(len(y), unique_columns))
    x = distinct[:, np.arange(len(names)) % unique_columns]
    seeds = []

    class TiedForest:
        def __init__(self, seed):
            # The first three features have identical mean importance. Their
            # summands appear in different orders across the five seeds.
            top = [(.2, .1, .3), (.3, .2, .1), (.1, .3, .2), (.02, .02, .02), (.02, .02, .02)][seed - 42]
            self.feature_importances_ = np.full(len(names), (1 - sum(top)) / (len(names) - 3))
            self.feature_importances_[:3] = top

        def set_params(self, *, n_jobs):
            assert n_jobs == 1
            return self

        def fit(self, train_x, train_y):
            np.testing.assert_array_equal(train_x, x)
            np.testing.assert_array_equal(train_y, y)

    def build_forest(name, class_count, *, seed):
        assert name == "random_forest"
        assert class_count == 12
        seeds.append(seed)
        return TiedForest(seed)

    monkeypatch.setattr(feature_sets, "build_model", build_forest)

    result = feature_sets.select_feature_sets(x, y, names, list(range(40)), classes)

    assert seeds == [42, 43, 44, 45, 46]
    rf = result["rf"]
    assert rf["mean_ranking"] == names
    assert rf["retained"] == names[:unique_columns]
    assert rf["flag_ranks"]["fwd_psh_flag_count"]["mean_rank"] == 41
    assert set(rf["flag_ranks"]["fwd_psh_flag_count"]["per_seed_ranks"].values()) == {41}
    assert rf["flag_ranks"]["fwd_psh_flag_count"]["retained_rank"] is None
    assert result["feature_sets"]["F_RF40"]["status"] == "STOPPED"
    assert result["feature_sets"]["F_RF40"]["column_indices"] == []
    expected_stops = {"F_RF40"}
    if unique_columns < 8:
        expected_stops.add("F_RF8")
    else:
        assert result["feature_sets"]["F_RF8"]["feature_names"] == names[:8]
    assert set(result["stopped_arms"]) == expected_stops
