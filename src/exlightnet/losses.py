"""Focal loss and explicitly documented class-alpha alternatives."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ALPHA_MODES = ("inv_freq", "sqrt_inv_freq", "effective_num", "none")


def class_alpha(y, mode="inv_freq", beta=.999):
    counts = np.bincount(np.asarray(y)).astype(float)
    if (counts == 0).any():
        raise ValueError("Every class must be represented in train")
    if mode == "inv_freq":
        alpha = 1 / counts
    elif mode == "sqrt_inv_freq":
        alpha = 1 / np.sqrt(counts)
    elif mode == "effective_num":
        alpha = (1 - beta) / -np.expm1(counts * np.log(beta))
    elif mode == "none":
        alpha = np.ones_like(counts)
    else:
        raise ValueError(f"Unknown alpha mode {mode}")
    return torch.tensor(alpha / alpha.mean(), dtype=torch.float32)


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.):
        super().__init__()
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1).gather(1, target[:, None]).squeeze(1)
        return (-self.alpha[target] * (1 - logp.exp()).pow(self.gamma) * logp).mean()
