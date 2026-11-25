import csv

import numpy as np
import pytest
import torch

from exlightnet import final_analysis as analysis
from exlightnet.io import read_json, sha256


def test_stratified_sample_keeps_all_rare_rows_and_is_reproducible():
    labels = np.repeat([0, 1, 2], [40, 25, 4])
    selected = analysis.stratified_indices(labels, 3)
    np.testing.assert_array_equal(selected, analysis.stratified_indices(labels, 3))
    assert len(set(selected)) == 54
    np.testing.assert_array_equal(np.bincount(labels[selected]), [25, 25, 4])
    np.testing.assert_array_equal(selected[-29:], np.arange(40, 69))


def test_rankings_select_the_true_logit_and_give_rare_classes_equal_weight():
    labels = np.array([0, 0, 0, 1])
    phi = np.empty((4, 2, 2))
    phi[:3] = [[4, 30], [0, 8]]
    phi[3] = [[20, 0], [4, -8]]
    primary, all_outputs = analysis.class_importance(phi, labels, classes=2)
    np.testing.assert_array_equal(primary, [[4, 0], [0, 8]])
    np.testing.assert_array_equal(all_outputs, [[17, 4], [10, 6]])
    np.testing.assert_array_equal(primary.mean(axis=0), [2, 4])
    np.testing.assert_array_equal(np.average(primary, weights=[3, 1], axis=0), [3, 2])


def test_top10_uses_canonical_feature_names_and_stable_ties():
    names = [f"feature_{i}" for i in range(12)]
    importance = np.ones(12)
    importance[-1] = 2
    rows = list(analysis.ranked_features(importance, names, limit=10))
    assert [r["feature"] for r in rows] == [names[-1], *names[:9]]
    assert [r["rank"] for r in rows] == list(range(1, 11))
    assert rows[0]["feature_index"] == 11


def test_old_composition_uses_saved_random300_and_excludes_local_examples():
    labels = np.repeat([0, 1], [299, 3])
    old = {
        "random_test_size": 300,
        "test_indices": list(range(300)),
        "local_examples": [{"test_position": 300}, {"test_position": 301}],
    }
    rows = analysis.earlier_sample_composition(labels, ["common", "rare"], old)
    assert [r["sampled_count"] for r in rows] == [299, 1]
    assert rows[1]["test_support"] == 3
    assert rows[1]["sample_fraction"] == pytest.approx(1 / 300)
    assert rows[1]["test_fraction"] == pytest.approx(3 / 302)


def test_deletion_targets_original_prediction_and_keeps_20_random_replicates():
    torch.set_num_threads(1)
    model = torch.nn.Linear(12, 2, bias=False).eval()
    with torch.no_grad():
        model.weight.zero_()
        model.weight[0, 11] = 2
        model.weight[1, 0] = 4
    inputs = np.ones((2, 12), dtype=np.float32)
    phi = np.zeros((2, 12, 2))
    phi[:, 11, 0] = 5
    phi[:, 0, 1] = 6
    labels = np.array([0, 1])
    predicted = np.array([1, 1])
    original = np.array([0.95, 0.96])
    shap, random, orders = analysis.class_deletion(
        model, inputs, labels, predicted, original, phi, classes=2
    )
    np.testing.assert_array_equal(shap[:, 0], original)
    np.testing.assert_array_equal(random[:, :, 0], np.repeat(original[:, None], 20, axis=1))
    assert shap[0, 1] == pytest.approx(1 / (1 + np.exp(2)))
    assert shap[0, 1] < 0.5
    assert random.shape == (2, 20, 13)
    assert orders.shape == (2, 20, 12)
    np.testing.assert_array_equal(np.sort(orders, axis=-1), np.broadcast_to(np.arange(12), orders.shape))
    _, again, again_orders = analysis.class_deletion(
        model, inputs, labels, predicted, original, phi, classes=2
    )
    np.testing.assert_array_equal(random, again)
    np.testing.assert_array_equal(orders, again_orders)
    _, rows = analysis.deletion_rows(shap, random, [
        {"class_index": 0, "true_class": "a", "predicted_correct_count": 0},
        {"class_index": 1, "true_class": "b", "predicted_correct_count": 1},
    ])
    assert rows[0]["random_minus_shap_k10"] == pytest.approx(random[0, :, 10].mean() - shap[0, 10])
    assert rows[0]["random_minus_shap_k10"] > 0
    assert rows[0]["shap_drop_k10"] == pytest.approx(original[0] - shap[0, 10])


@pytest.fixture
def stage_inputs():
    return analysis.AnalysisInputs(
        inputs=np.ones((2, 12), dtype=np.float32),
        background=np.zeros((2, 12), dtype=np.float32),
        sample_indices=np.array([5, 8]),
        background_indices=np.array([2, 3]),
        true_labels=np.array([0, 1]),
        predicted=np.array([1, 1]),
        original_probabilities=np.array([0.95, 0.96]),
        class_names=["a", "b"],
        feature_names=[f"f{i}" for i in range(12)],
        support=np.array([8, 1]),
        earlier_composition=[
            {"class_index": 0, "true_class": "a", "sampled_count": 6},
            {"class_index": 1, "true_class": "b", "sampled_count": 0},
        ],
        provenance={"source_sha256": {"checkpoint.pt": "synthetic"}},
    )


def test_partial_stages_resume_complete_cache_is_read_only_and_recipe_is_signed(
    stage_inputs, tmp_path, monkeypatch
):
    calls = []

    def fake_shap(_model, background, inputs, nsamples):
        calls.append("shap")
        assert nsamples == 200
        np.testing.assert_array_equal(background, stage_inputs.background)
        return {
            "values": np.ones((len(inputs), 12, 2)), "method": "synthetic",
            "expected_logits": np.array([0., 0.], dtype=np.float32),
        }

    def fake_deletion(*args, **kwargs):
        calls.append("deletion")
        return (
            np.zeros((2, 13)),
            np.zeros((2, 20, 13)),
            np.broadcast_to(np.arange(12), (2, 20, 12)),
        )

    def fail(*args, **kwargs):
        raise OSError("interrupted report")

    monkeypatch.setattr(analysis, "explain_attributions", fake_shap)
    monkeypatch.setattr(analysis, "class_deletion", fake_deletion)
    original_plot = analysis._write_plots
    monkeypatch.setattr(analysis, "_write_plots", fail)
    output = tmp_path / "analysis"
    with pytest.raises(OSError, match="interrupted report"):
        analysis._run_analysis(stage_inputs, output, object)
    assert read_json(output / "analysis.json")["status"] == "in_progress"
    assert (output / "stratified_shap_values.npz").is_file()
    assert (output / "deletion_replicates.npz").is_file()
    monkeypatch.setattr(analysis, "_write_plots", original_plot)
    complete = analysis._run_analysis(stage_inputs, output, object)
    assert complete["status"] == "complete"
    assert complete["attribution"]["expected_logits"] == [0., 0.]
    assert calls == ["shap", "deletion"]
    with (output / "per_class_top10.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 20
    checkpoint_hashes = {p.name: sha256(p) for p in output.iterdir()}
    assert analysis._run_analysis(stage_inputs, output, fail) == complete
    assert checkpoint_hashes == {p.name: sha256(p) for p in output.iterdir()}
    stage_inputs.provenance["source_sha256"]["checkpoint.pt"] = "different"
    with pytest.raises(RuntimeError, match="signature"):
        analysis._run_analysis(stage_inputs, output, fail)
    assert checkpoint_hashes == {p.name: sha256(p) for p in output.iterdir()}


def test_output_cannot_overlap_owned_results(monkeypatch, tmp_path):
    original = tmp_path / "results/full/explain"
    original.mkdir(parents=True)
    evidence = original / "explanations.json"
    evidence.write_bytes(b"old evidence")
    with pytest.raises(ValueError, match="separate"):
        analysis.run_stratified_shap(tmp_path, original)
    assert evidence.read_bytes() == b"old evidence"
