import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from exlightnet.model import ExLightNet
from exlightnet.quantize import export_int8, load_int8, quantize_per_channel


def test_quantization_half_scale_bound_and_zero_channel():
    w = torch.tensor([[0., 0., 0.], [1., -.7, .3], [100., 0., -10.]])
    q, s = quantize_per_channel(w)
    assert q.dtype == np.int8
    restored = q * s[:, None]
    assert np.all(np.abs(w.numpy() - restored) <= s[:, None] / 2 + 1e-6)
    assert np.isfinite(s).all() and (s > 0).all()


@pytest.mark.parametrize("classes", [2, 12])
def test_export_roundtrip_and_numpy_inference(classes, tmp_path):
    torch.manual_seed(4)
    torch.set_num_threads(1)
    model = ExLightNet(classes).eval()
    path = tmp_path / "model.npz"
    evidence = export_int8(model, path, {"feature_names": [str(i) for i in range(40)]})
    restored = load_int8(path)
    x = torch.randn(3, 1, 40)
    with torch.no_grad():
        expected = restored(x).numpy()
    spec = importlib.util.spec_from_file_location("pi", Path("scripts/bench_pi.py"))
    pi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pi)
    actual = pi.NumpyExLightNet(path)(x.numpy())
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=2e-6)
    assert evidence["max_abs_weight_error"] > 0
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
            _, s = quantize_per_channel(module.weight)
            err = (module.weight - dict(restored.named_modules())[name].weight).abs().detach().numpy()
            assert np.all(err <= s.reshape((-1,) + (1,) * (err.ndim - 1)) / 2 + 1e-7)
