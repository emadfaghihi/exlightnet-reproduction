"""Explicit baseline choices where the paper does not specify a unique graph."""
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight

from .model import ExLightNet

DEEP_MODELS = ("exlightnet", "mlp", "bilstm", "std_cnn")
MODELS = DEEP_MODELS + ("random_forest", "xgboost")
ABLATIONS = ("no_se", "standard_conv", "balanced", "no_sampler", "top20")


class MLP(nn.Module):
    def __init__(self, classes, features=40):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(features, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, classes))

    def forward(self, x):
        return self.layers(x.flatten(1))


class BiLSTM(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=1, bidirectional=True, batch_first=True)
        self.head = nn.Linear(64, classes)

    def forward(self, x):
        x = x.flatten(1).unsqueeze(-1)
        _, (hidden, _) = self.lstm(x)
        return self.head(torch.cat([hidden[-2], hidden[-1]], dim=1))


class StdCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        layers = []
        incoming = 1
        for outgoing in (64, 128, 256):
            layers += [nn.Conv1d(incoming, outgoing, 3, bias=False), nn.BatchNorm1d(outgoing), nn.ReLU()]
            incoming = outgoing
        self.layers = nn.Sequential(*layers, nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(256, classes)

    def forward(self, x):
        return self.head(self.layers(x.flatten(1).unsqueeze(1)).squeeze(-1))


def build_model(name, classes, seed=42):
    if name == "mlp":
        return MLP(classes)
    if name == "bilstm":
        return BiLSTM(classes)
    if name == "std_cnn":
        return StdCNN(classes)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=1)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=8, tree_method="hist", device="cpu", n_jobs=1,
                             random_state=seed, objective="binary:logistic" if classes == 2 else "multi:softprob",
                             eval_metric="logloss" if classes == 2 else "mlogloss", verbosity=0)
    if name not in ("exlightnet", *ABLATIONS):
        raise ValueError(f"Unknown model {name}")
    return ExLightNet(classes, use_se=name != "no_se", standard=name == "standard_conv")


def fit_tree(model, name, x, y):
    if name == "xgboost":
        model.fit(x, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(x, y)
    return model
