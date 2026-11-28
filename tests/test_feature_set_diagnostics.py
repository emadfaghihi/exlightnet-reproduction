import numpy as np
import pytest


def test_opposite_label_matches_use_exact_float64_values():
    from scripts.diagnostics.feature_sets import opposite_label_matches

    next_float = np.nextafter(1.0, 2.0)
    train = np.array([[1.0, -0.0], [2.0, 1.0], [3.0, 1.0]])
    validation = np.array([[next_float, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0]])
    matches = opposite_label_matches(train, np.array([0, 1, 1]), validation, np.array([1, 1, 1, 0]))

    np.testing.assert_array_equal(matches, [False, True, False, True])


def test_load_partitions_reads_only_train_and_validation(tmp_path):
    from scripts.diagnostics.feature_sets import load_partitions

    path = tmp_path / "split.npz"
    np.savez(path, train=[3, 0], val=[2], test=np.array([object()]))

    train, validation = load_partitions(path, n_rows=4)

    np.testing.assert_array_equal(train, [3, 0])
    np.testing.assert_array_equal(validation, [2])


def test_load_partitions_rejects_overlap(tmp_path):
    from scripts.diagnostics.feature_sets import load_partitions

    path = tmp_path / "split.npz"
    np.savez(path, train=[0, 1], val=[1, 2])

    with pytest.raises(ValueError, match="overlap"):
        load_partitions(path, n_rows=3)
