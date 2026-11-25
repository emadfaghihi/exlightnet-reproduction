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


def sample_epoch_indices(y, generator, count=None, exponent=.5):
    """Draw per-flow tempered samples, on CPU, using WeightedRandomSampler semantics."""
    y = np.asarray(y, dtype=np.int64)
    weights = np.bincount(y).astype(float)[y] ** (-exponent)
    return torch.multinomial(torch.from_numpy(weights), len(y) if count is None else count,
                             replacement=True, generator=generator)


def checkpoint_warnings(state, current):
    return list(state.get("warnings", [])) + list(current)


def fit_resumable(model, x_train, y_train, x_val, y_val, config, directory, *,
                  identity, device="cpu", min_epochs=0, early_stopping=True,
                  checkpoint_hook=None, warning_ledger=None):
    """Train a supplementary run, atomically checkpointing complete epochs.

    Returns the best and final parameter dictionaries and epoch history. The
    caller owns model initialization, identity construction, and evaluation.
    """
    import copy
    import os
    from .device import configure_device

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    warning_ledger = warning_ledger if warning_ledger is not None else []
    path = directory / "epoch_checkpoint.pt"
    seed_everything(config.seed, config.threads)
    device = configure_device(device)
    model.to(device)
    x = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.long, device=device)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=device)
    loss_fn = FocalLoss(class_alpha(y_train, config.alpha_mode)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .999), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    generator = torch.Generator().manual_seed(config.seed)
    best, best_epoch, stale, history, best_state, start_epoch = -1., 0, 0, [], None, 1
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["identity"] != identity:
            from .campaign import RunStopped
            raise RunStopped("Resume checkpoint identity differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device=0)
        best, best_epoch, stale = state["best"], state["best_epoch"], state["stale"]
        history, best_state, start_epoch = state["history"], state["best_state"], state["epoch"] + 1
        warning_ledger[:] = checkpoint_warnings(state, warning_ledger)
    for epoch in range(start_epoch, config.max_epochs + 1):
        if early_stopping and stale >= config.patience and epoch > min_epochs:
            break
        start = time.perf_counter()
        import warnings
        with warnings.catch_warnings(record=True) as epoch_warnings:
            warnings.simplefilter("always")
            result = _resumable_epoch(model, x, y, xv, y_train, y_val, config, device, generator, optimizer, loss_fn)
            epoch_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
        warning_ledger.extend({"epoch": epoch, "phase": "epoch", "message": str(w.message), "category": w.category.__name__} for w in epoch_warnings)
        total, val = result
        history.append({"epoch": epoch, "train_loss": total / len(y_train), "val_macro_f1": val,
                        "lr": epoch_lr, "seconds": time.perf_counter() - start})
        if val > best:
            best, best_epoch, stale = val, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        state = {"identity": identity, "epoch": epoch, "model": model.state_dict(), "best_state": best_state,
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
                 "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state(device=0) if device == "cuda" else None,
                 "best": best, "best_epoch": best_epoch, "stale": stale, "history": history, "warnings": list(warning_ledger)}
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
        if checkpoint_hook is not None:
            checkpoint_hook(epoch, path)
    if not history:
        raise RuntimeError("No completed epoch")
    return {"best_state": best_state, "last_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "best_epoch": best_epoch, "best_val_macro_f1": best, "history": history, "epochs_run": len(history)}


def _resumable_epoch(model, x, y, xv, y_train, y_val, config, device, generator, optimizer, loss_fn):
        model.train()
        if config.sampler == "none":
            indices = torch.randperm(len(y_train), generator=generator)
        else:
            exponent = {"tempered": .5, "balanced": 1.}[config.sampler]
            indices = sample_epoch_indices(y_train, generator, exponent=exponent)
        total = 0.
        for offset in range(0, len(indices), config.batch_size):
            batch = indices[offset:offset + config.batch_size].to(device)
            targets = y[batch]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x[batch]), targets)
            if not torch.isfinite(loss):
                from .campaign import RunStopped
                raise RunStopped("Nonfinite training loss; branch stopped")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(targets)
        model.eval()
        with torch.inference_mode():
            predictions = torch.cat([model(xv[i:i + config.batch_size]).argmax(1).cpu()
                                     for i in range(0, len(xv), config.batch_size)]).numpy()
        val = float(f1_score(y_val, predictions, average="macro", labels=np.arange(len(np.bincount(y_train))), zero_division=0))
        return total, val
