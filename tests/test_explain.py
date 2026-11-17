import numpy as np
import torch

from exlightnet.explain import explain_attributions, normalize_shap_values


def test_shap_shapes_across_versions():
    values = np.arange(24).reshape(3, 4, 2)
    np.testing.assert_array_equal(normalize_shap_values([values[:, :, 0], values[:, :, 1]], 3, 4, 2), values)
    np.testing.assert_array_equal(normalize_shap_values(values[:, None, :, :], 3, 4, 2), values)


def test_shap_additivity_tiny_linear_model():
    model = torch.nn.Linear(4, 2).eval()
    background = np.array([[0., 0., 0., 0.], [1., 2., 3., 4.]], dtype="float32")
    x = np.array([[2., 1., 0., 1.], [0., 1., 2., 3.]], dtype="float32")
    result = explain_attributions(model, background, x)
    assert result["method"] == "DeepExplainer"
    assert result["additivity_max_abs"] < 1e-5
    assert result["values"].shape == (2, 4, 2)


def test_explanation_receipt_is_not_written_before_plots(tmp_path, monkeypatch):
    import matplotlib.figure
    import pytest
    from exlightnet.explain import run_explanations
    torch.set_num_threads(1)
    model = torch.nn.Linear(40, 2).eval()
    x = np.zeros((2, 40), dtype="float32")
    x[1] = 1.
    with torch.no_grad():
        p = model(torch.tensor(x)).softmax(1).numpy()
    def disk_failure(*args, **kwargs):
        raise OSError("simulated plot write interruption")
    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", disk_failure)
    with pytest.raises(OSError):
        run_explanations(model, model, x, x, np.array([0, 1]), p, ["dos_syn_hping", "nmap_os_detection"], [f"f{i}" for i in range(40)], tmp_path, fast=True)
    assert not (tmp_path / "explanations.json").exists()


def test_quantization_stability_uses_paired_gradient_samples(tmp_path):
    from exlightnet.explain import run_explanations
    class Square(torch.nn.Module):
        def forward(self, x):
            return x.square()
    torch.manual_seed(13)
    torch.set_num_threads(1)
    model = torch.nn.Sequential(torch.nn.Linear(40, 2), Square()).eval()
    rng = np.random.default_rng(3)
    background = rng.normal(size=(5, 40)).astype("float32")
    inputs = (rng.normal(size=(2, 40)) * 3).astype("float32")
    with torch.no_grad():
        p = model(torch.tensor(inputs)).softmax(1).numpy()
    result = run_explanations(model, model, background, inputs, np.array([0, 1]), p,
                              ["dos_syn_hping", "nmap_os_detection"], [f"f{i}" for i in range(40)], tmp_path, fast=True)
    assert result["fp32"]["method"] == result["dequantized_int8"]["method"] == "GradientExplainer"
    with np.load(tmp_path / "shap_values.npz") as arrays:
        np.testing.assert_allclose(arrays["global_importance"], arrays["quantized_importance"], atol=1e-12)


def test_deep_failure_is_detected_before_full_explanation_work(monkeypatch):
    import shap
    class Square(torch.nn.Module):
        def forward(self, x):
            return x.square()
    model = torch.nn.Sequential(torch.nn.Linear(4, 2), Square()).eval()
    with torch.no_grad():
        model[0].weight.fill_(1.)
        model[0].bias.zero_()
    observed = []
    original = shap.DeepExplainer.shap_values
    def measured(self, inputs, *args, **kwargs):
        observed.append(len(inputs))
        return original(self, inputs, *args, **kwargs)
    monkeypatch.setattr(shap.DeepExplainer, "shap_values", measured)
    result = explain_attributions(model, np.zeros((3, 4), dtype="float32"), np.ones((8, 4), dtype="float32"), nsamples=16)
    assert result["method"] == "GradientExplainer"
    assert observed == [2]
    assert result["values"].shape == (8, 4, 2)
