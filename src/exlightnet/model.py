"""Exact no-padding, bias-free-convolution architecture from the authoritative brief."""
import torch
from torch import nn


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channels // 4, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(self.pool(x).squeeze(-1)))))
        return x * scale.unsqueeze(-1)


class Block(nn.Module):
    def __init__(self, incoming, outgoing, use_se=True, standard=False):
        super().__init__()
        if standard:
            self.conv = nn.Conv1d(incoming, outgoing, 3, bias=False)
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(incoming, incoming, 3, groups=incoming, bias=False),
                nn.Conv1d(incoming, outgoing, 1, bias=False))
        self.bn = nn.BatchNorm1d(outgoing)
        self.relu = nn.ReLU()
        self.se = SE(outgoing) if use_se else nn.Identity()

    def forward(self, x):
        return self.se(self.relu(self.bn(self.conv(x))))


class ExLightNet(nn.Module):
    def __init__(self, num_classes=12, use_se=True, standard=False):
        super().__init__()
        self.num_classes = num_classes
        self.use_se = use_se
        self.standard = standard
        self.stem = nn.Sequential(nn.Conv1d(1, 48, 3, bias=False), nn.BatchNorm1d(48), nn.ReLU())
        self.block1 = Block(48, 96, use_se, standard)
        self.block2 = Block(96, 144, use_se, standard)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.maximum = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Linear(288, num_classes)

    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.block2(self.block1(self.stem(x)))
        return self.head(torch.cat([self.avg(x).squeeze(-1), self.maximum(x).squeeze(-1)], dim=1))


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_gate():
    actual = {"multiclass": parameter_count(ExLightNet(12)), "binary": parameter_count(ExLightNet(2)),
              "no_se": parameter_count(ExLightNet(12, use_se=False))}
    expected = {"multiclass": 38328, "binary": 35438, "no_se": 23052}
    assert actual == expected, f"M1 failed; investigate architecture, not target counts: {actual}"
    stage = {name: parameter_count(module) for name, module in ExLightNet(12).named_children() if name in ("stem", "block1", "block2", "head")}
    assert list(stage.values()) == [240, 9672, 24948, 3468], stage
    return {"gate": "M1", "pass": True, "actual": actual, "stages": stage}
