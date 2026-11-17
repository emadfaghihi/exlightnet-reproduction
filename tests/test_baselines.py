import pytest
import torch

from exlightnet.baselines import build_model
from exlightnet.model import parameter_count


@pytest.mark.parametrize("classes,count", [(2, 13634), (12, 14284)])
def test_mlp_parameter_count(classes, count):
    assert parameter_count(build_model("mlp", classes)) == count


@pytest.mark.parametrize("name", ["mlp", "bilstm", "std_cnn", "no_se", "standard_conv", "top20"])
def test_neural_baseline_forward(name):
    model = build_model(name, 12).eval()
    assert model(torch.randn(2, 20 if name == "top20" else 40)).shape == (2, 12)
