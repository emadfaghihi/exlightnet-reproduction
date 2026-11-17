import pytest
import torch

from exlightnet.model import ExLightNet, parameter_count, model_gate


@pytest.mark.parametrize("classes,se,count", [(12, True, 38328), (2, True, 35438), (12, False, 23052)])
def test_exact_parameter_counts(classes, se, count):
    assert parameter_count(ExLightNet(classes, use_se=se)) == count


def test_stage_lengths_and_biases():
    model = ExLightNet(12).eval()
    x = torch.randn(3, 1, 40)
    shapes = []
    for part in (model.stem, model.block1, model.block2):
        x = part(x)
        shapes.append(x.shape[-1])
    assert shapes == [38, 36, 34]
    assert model(torch.randn(3, 1, 40)).shape == (3, 12)
    assert all(m.bias is None and m.padding == (0,) for m in model.modules() if isinstance(m, torch.nn.Conv1d))
    assert model_gate()["pass"]
