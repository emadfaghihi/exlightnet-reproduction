"""Deterministic CPU training. This module never accepts a test partition."""
import csv
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .io import write_json
from .losses import FocalLoss, class_alpha


@dataclass
class TrainConfig:
    seed: int = 42
    max_epochs: int = 30
    batch_size: int = 512
    patience: int = 6
    lr: float = 1.5e-3
    weight_decay: float = 1e-4
    alpha_mode: str = "inv_freq"
    sampler: str = "tempered"
    threads: int = 1


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def sampler_multiplicity(y, exponent=.5):
    counts = np.bincount(y).astype(float)
    denominator = np.power(counts, 1 - exponent).sum()
    return {str(c): float(len(y) * counts[c] ** (-exponent) / denominator) for c in range(len(counts))}


def make_loader(x, y, config):
    generator = torch.Generator().manual_seed(config.seed)
    sampler = None
    if config.sampler != "none":
        exponent = {"tempered": .5, "balanced": 1.}[config.sampler]
        weights = np.bincount(y).astype(float)[y] ** (-exponent)
        sampler = WeightedRandomSampler(torch.from_numpy(weights), len(y), replacement=True, generator=generator)
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.as_tensor(y, dtype=torch.long)),
                      batch_size=config.batch_size, sampler=sampler, shuffle=sampler is None,
                      generator=generator, num_workers=0)


@torch.inference_mode()
def predict_proba(model, x, batch_size=512):
    model.eval()
    outputs = [model(torch.as_tensor(x[i:i + batch_size], dtype=torch.float32)).softmax(1).numpy()
               for i in range(0, len(x), batch_size)]
    return np.concatenate(outputs)


def fit_model(model, x_train, y_train, x_val, y_val, config, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(config.seed, config.threads)
    model.cpu()
    loader = make_loader(x_train, y_train, config)
    loss_fn = FocalLoss(class_alpha(y_train, config.alpha_mode))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .999), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    best, best_epoch, stale = -1., 0, 0
    start = time.perf_counter()
    with (directory / "epochs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_macro_f1", "lr", "elapsed_s"])
        writer.writeheader()
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            total = 0.
            for inputs, targets in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(inputs), targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss; branch stopped")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(targets)
            val = float(f1_score(y_val, predict_proba(model, x_val).argmax(1), average="macro", labels=np.arange(len(np.bincount(y_train))), zero_division=0))
            writer.writerow({"epoch": epoch, "train_loss": total / len(y_train), "val_macro_f1": val,
                             "lr": optimizer.param_groups[0]["lr"], "elapsed_s": time.perf_counter() - start})
            stream.flush()
            if val > best:
                best, best_epoch, stale = val, epoch, 0
                torch.save(model.state_dict(), directory / "model.pt")
            else:
                stale += 1
            scheduler.step()
            if stale >= config.patience:
                break
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    result = {"best_epoch": best_epoch, "best_val_macro_f1": best, "epochs_run": epoch,
              "training_seconds": time.perf_counter() - start, "config": asdict(config)}
    write_json(directory / "training.json", result)
    return result
