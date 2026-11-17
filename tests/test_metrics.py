import numpy as np
import pytest

from exlightnet.metrics import evaluate_once, score_predictions, summarize


def test_binary_operating_point_and_macro_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([[.8, .2], [.4, .6], [.7, .3], [.1, .9]])
    r = score_predictions(y, p, ["normal", "attack"])
    assert r["confusion_matrix"] == [[1, 1], [1, 1]]
    assert r["attack_recall"] == r["fpr"] == .5
    assert r["miss_count"] == r["fp_count"] == 1
    assert r["macro_f1"] == .5
    assert r["auc"] == .75


def test_multiclass_micro_ovr():
    r = score_predictions(np.arange(3), np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]]), list("abc"))
    assert r["auc"] == r["macro_f1"] == 1.
    assert [row["support"] for row in r["per_class"].values()] == [1, 1, 1]


def test_evaluation_is_cached_and_claim_survives_failure(tmp_path):
    calls = []
    def predictor():
        calls.append(1)
        return np.array([[.9, .1], [.1, .9]])
    for _ in range(2):
        evaluate_once(tmp_path / "good", np.array([0, 1]), predictor, ["normal", "attack"], {"fingerprint": "a"})
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="fingerprint"):
        evaluate_once(tmp_path / "good", np.array([0, 1]), predictor, ["normal", "attack"], {"fingerprint": "b"})
    def broken():
        raise ValueError("prediction interrupted")
    with pytest.raises(ValueError):
        evaluate_once(tmp_path / "broken", np.array([0, 1]), broken, ["normal", "attack"], {})
    with pytest.raises(RuntimeError, match="already claimed"):
        evaluate_once(tmp_path / "broken", np.array([0, 1]), predictor, ["normal", "attack"], {})


def test_single_seed_std_is_not_fabricated():
    rows = [{"task": "binary", "model": "mlp", "seed": 42, "macro_f1": .7}]
    result = summarize(rows)
    assert result.iloc[0]["macro_f1_seed42"] == .7
    assert result.iloc[0]["macro_f1_std"] is None
    assert not result.iloc[0]["five_seed_complete"]


def test_receipt_required_when_loading_cached_predictions(tmp_path):
    path = tmp_path / "run"
    evaluate_once(path, np.array([0, 1]), lambda: np.array([[.9, .1], [.1, .9]]), ["normal", "attack"], {"fingerprint": "x"})
    (path / "test_evaluation_receipt.json").unlink()
    with pytest.raises(RuntimeError, match="receipt"):
        evaluate_once(path, np.array([0, 1]), lambda: pytest.fail("must not evaluate twice"), ["normal", "attack"], {"fingerprint": "x"})


def test_summary_reports_actual_observation_count_for_missing_benchmarks():
    rows = [{"task": "binary", "model": "mlp", "seed": 42, "macro_f1": .7, "latency_mean_ms": 1.},
            {"task": "binary", "model": "mlp", "seed": 43, "macro_f1": .8}]
    result = summarize(rows).iloc[0]
    assert result["n_seeds"] == 2
    assert result["macro_f1_n"] == 2 and result["latency_mean_ms_n"] == 1
