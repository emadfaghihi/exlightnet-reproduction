import numpy as np
import pytest
import torch
from torch.nn import functional as F

from exlightnet.data import EXPECTED_COUNTS, make_split
from exlightnet.losses import FocalLoss, class_alpha
from exlightnet.train import TrainConfig, fit_model, predict_proba, sampler_multiplicity, seed_everything
from exlightnet.model import ExLightNet


def test_focal_loss_gamma_zero_equals_ce():
    logits = torch.tensor([[1., 2., -1.], [-1., 0., 3.]], requires_grad=True)
    target = torch.tensor([0, 2])
    torch.testing.assert_close(FocalLoss(torch.ones(3), gamma=0)(logits, target), F.cross_entropy(logits, target))


def test_sampler_multiplicity():
    labels = np.concatenate([np.repeat(k, n) for k, n in EXPECTED_COUNTS.items()])
    split, _ = make_split(labels)
    _, y = np.unique(labels[split["train"]], return_inverse=True)
    tempered = sampler_multiplicity(y, .5)
    balanced = sampler_multiplicity(y, 1.)
    rare = str(int(np.bincount(y).argmin()))
    assert tempered[rare] == pytest.approx(30., rel=.15)
    assert balanced[rare] == pytest.approx(359., rel=.15)


@pytest.mark.parametrize("mode", ["inv_freq", "sqrt_inv_freq", "effective_num", "none"])
def test_alpha_mean_one(mode):
    alpha = class_alpha(np.array([0, 0, 0, 1]), mode)
    assert alpha.mean().item() == pytest.approx(1.)
    assert torch.isfinite(alpha).all()


def test_deterministic_training_restores_checkpoint(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(24, 40)).astype("float32")
    y = np.arange(24) % 2
    predictions = []
    for i in range(2):
        seed_everything(42)
        model = ExLightNet(2)
        info = fit_model(model, x, y, x[:8], y[:8], TrainConfig(max_epochs=2, batch_size=8), tmp_path / str(i))
        assert info["best_epoch"] in (1, 2)
        assert (tmp_path / str(i) / "epochs.csv").exists()
        predictions.append(predict_proba(model, x))
    np.testing.assert_array_equal(*predictions)
