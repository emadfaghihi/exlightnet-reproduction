"""Plans and execution contracts for separate supplementary campaigns."""
import hashlib
import json
import os

# Must be configured before PyTorch initializes CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

TREE_MODELS = ("random_forest", "xgboost")
RUN_SELECTIONS = ("all", "neural", "trees")


class RunStopped(RuntimeError):
    """A preserved run-local failure that must not stop independent runs."""


def run_independent_plans(plans, work, failure_dir):
    from pathlib import Path
    failures = []
    for plan in plans:
        try:
            work(plan)
        except RunStopped as exc:
            failure = {"plan": plan, "reason": str(exc), "status": "stopped"}
            failures.append(failure)
            path = Path(failure_dir) / (run_fingerprint({"plan": plan}) + ".json")
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                _write_json(path, failure)
    return {"stopped_count": len(failures), "stopped": failures}


def finalize_tree_cache(directory, root, fingerprint):
    from pathlib import Path
    directory, root = Path(directory), Path(root)
    identity = json.loads((directory / "run_identity.json").read_text())
    if identity.get("run_fingerprint") != fingerprint or identity.get("model") not in TREE_MODELS:
        raise RunStopped("Tree identity differs")
    item = _evaluate(directory, "best", None, None, lambda: (_ for _ in ()).throw(RunStopped("Tree evaluation missing")), fingerprint + ":best")
    for key in ("prediction", "metrics", "receipt"):
        item[key + "_path"] = str(Path(item[key + "_path"]).relative_to(root))
    record = {**identity, "status": "complete", "checkpoints": {"best": item, "last": dict(item)}}
    _write_json(directory / "record.json", record)
    return record


def validate_campaign_name(campaign):
    import re
    if not isinstance(campaign, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", campaign) or campaign in {"cpu_v1", "full", "fast"}:
        raise ValueError("Campaign must be a separate, simple supplementary campaign name")


def library_versions():
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "torch", "scikit-learn")}
    try:
        versions["xgboost"] = importlib.metadata.version("xgboost")
    except importlib.metadata.PackageNotFoundError:
        versions["xgboost"] = importlib.metadata.version("xgboost-cpu")
    return versions


def run_fingerprint(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _tree_record_identity(record, plan):
    return {**{key: record[key] for key in ('campaign', 'device', 'config', 'data_fingerprint',
                                            'payload_sha256', 'prepared_sha256', 'libraries')},
            'plan': {key: record[key] for key in plan}}


def _resolve_tree_identity(root, identity):
    """Reuse legacy CPU-tree identities without changing their opaque hashes."""
    matches = {}
    for directory in (root / 'cpu').glob('*'):
        for name in ('record.json', 'run_identity.json'):
            path = directory / name
            if not path.exists():
                continue
            try:
                record = json.loads(path.read_text())
                if record.get('model') not in TREE_MODELS or record.get('campaign') != identity['campaign']:
                    continue
                if any(record.get(key) != value for key, value in identity['plan'].items()):
                    continue
                previous = _tree_record_identity(record, identity['plan'])
                fingerprint = run_fingerprint(previous)
                if fingerprint != record.get('run_fingerprint') or fingerprint != directory.name:
                    raise RunStopped('Existing tree fingerprint differs from its recorded identity')
                comparable = {**previous, 'libraries': {**previous['libraries'], 'gpu_name': None}}
                if comparable != identity:
                    raise RunStopped('Existing tree scientific identity differs; use a separate campaign')
                matches[fingerprint] = previous
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RunStopped('Existing tree identity is corrupt') from exc
    if len(matches) > 1:
        raise RunStopped('Ambiguous existing tree identities; preserve and inspect both branches')
    return next(iter(matches.values()), identity)


def validate_probe(probe, environment):
    if probe.get("environment") != environment:
        raise RuntimeError("Timing probe environment differs or is missing; use a separate campaign and probe again")


def budget_allows(elapsed_seconds, budget_seconds):
    return elapsed_seconds < budget_seconds


def study_runs(study):
    """Return all predeclared runs, including B3 references to B1."""
    rows = []
    def add(task, model, seed, split=42, arm="paper", **extra):
        rows.append(dict(study=study, task=task, model=model, seed=seed, split_seed=split, arm=arm, **extra))
    binary = ("exlightnet", "mlp", "random_forest", "xgboost")
    multi = ("exlightnet", "mlp", "standard_conv", "no_se", "std_cnn", "random_forest", "xgboost")
    if study == "B1":
        for seed in range(42, 72):
            for task, models in (("binary", binary), ("multiclass", multi)):
                for model in models:
                    add(task, model, seed)
    elif study == "B2":
        for seed in range(42, 52):
            for task, models in (("binary", binary), ("multiclass", ("exlightnet", "mlp", "standard_conv", "random_forest", "xgboost"))):
                for model in models:
                    add(task, model, seed, seed)
    elif study == "B3":
        for seed in range(42, 57):
            for model in ("exlightnet", "standard_conv", "mlp", "std_cnn"):
                add("multiclass", model, seed, arm="paper", reuse_study="B1")
                add("multiclass", model, seed, arm="fixed30")
                add("multiclass", model, seed, arm="patience10_min15")
    elif study == "B4":
        for model in ("exlightnet", "mlp"):
            for seed in range(42, 57): add("binary", model, seed, arm="uncapped60")
    elif study == "B5":
        for lr in (.0003, .001, .0015):
            for seed in range(42, 52): add("multiclass", "std_cnn", seed, arm=f"lr_{lr:g}", lr=lr)
    elif study == "probe": add("multiclass", "exlightnet", 42, arm="probe")
    elif study == "determinism":
        for repeat in (1, 2): add("multiclass", "exlightnet", 42, arm=f"repeat{repeat}")
    else: raise ValueError(f"Unknown study: {study}")
    return rows


def select_study_runs(study, only="all"):
    """Filter a study's declared rows by model family while preserving order."""
    if only not in RUN_SELECTIONS:
        raise ValueError("Run selection must be all, neural or trees")
    plans = study_runs(study)
    if only == "all":
        return plans
    return [plan for plan in plans if (plan["model"] in TREE_MODELS) == (only == "trees")]


def projected_neural_hours(study, seconds_per_epoch):
    """Crude estimate at probe speed; excludes trees and model-specific speed."""
    if seconds_per_epoch < 0:
        raise ValueError("Negative epoch duration")
    epochs = 0
    for row in study_runs(study):
        if "reuse_study" in row or row["model"] in TREE_MODELS:
            continue
        epochs += 60 if study == "B4" else 20 if row["task"] == "binary" else 30
    return epochs * seconds_per_epoch / 3600


def _print_neural_projections(seconds_per_epoch, only):
    for study in ("B1", "B3", "B2", "B4", "B5"):
        count = sum("reuse_study" not in plan for plan in select_study_runs(study, only))
        hours = projected_neural_hours(study, seconds_per_epoch)
        print(f"{study}: {count} selected runs; rough neural projection {hours:.2f} hours, trees and model differences excluded", flush=True)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_prepared_manifest(directory, split, metadata):
    from pathlib import Path
    directory = Path(directory)
    try:
        entry = json.loads((directory / "manifest.json").read_text())["splits"][str(split)]
        valid = (entry["sha256"] == _sha256(directory / f"split_{split}.npz")
                 and entry["data_fingerprint"] == metadata["data_fingerprint"]
                 and entry["payload_sha256"] == metadata["payload_sha256"])
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError("Invalid prepared manifest or missing split") from exc
    if not valid:
        raise ValueError("Prepared manifest does not match split and payload")


def _write_json(path, value):
    import os
    from pathlib import Path
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def package_campaign(root, output):
    """Atomically package only campaign results, never prepared arrays or locks."""
    import os
    import zipfile
    from pathlib import Path
    root, output = Path(root), Path(output)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.name == "epoch_checkpoint.pt" and (path.parent / "record.json").is_file():
                continue
            if path.is_file() and path.suffix not in (".tmp", ".lock", ".pkl") and (path.suffix != ".npz" or path.name in ("predictions.npz", "logits.npz")) and path != output and path != tmp:
                archive.write(path, path.relative_to(root))
    os.replace(tmp, output)
    return output


def guard_evaluation_receipts(directory):
    """Stop before training if any prior evaluation was claimed but incomplete."""
    from pathlib import Path
    for label in ("best", "last"):
        folder = Path(directory) / label
        receipt = folder / "receipt.json"
        if receipt.exists():
            try:
                status = json.loads(receipt.read_text()).get("status")
            except (OSError, ValueError, AttributeError) as exc:
                raise RunStopped("Invalid evaluation receipt") from exc
            if status != "complete":
                raise RunStopped("Incomplete evaluation receipt; training and inference stopped")
        elif (folder / "metrics.json").exists() or (folder / "predictions.npz").exists():
            raise RunStopped("Evaluation evidence exists without receipt")


def restore_campaign_archive(archive_path, output, campaign):
    """Restore a campaign zip without following links or replacing evidence."""
    import fcntl
    import shutil
    import stat
    import tempfile
    import zipfile
    from pathlib import Path

    validate_campaign_name(campaign)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = output / campaign
    if root.is_symlink():
        raise ValueError("Campaign root cannot be a symlink")
    root.mkdir(exist_ok=True)
    lock_path = root / "campaign.lock"
    if lock_path.is_symlink():
        raise ValueError("Campaign lock cannot be a symlink")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Campaign is already running") from exc
        with tempfile.TemporaryDirectory(prefix=".restore-", dir=output) as staging:
            stage = Path(staging)
            with zipfile.ZipFile(archive_path) as archive:
                seen = set()
                files = []
                for info in archive.infolist():
                    name = info.filename
                    components = name.rstrip("/").split("/")
                    mode = info.external_attr >> 16
                    if (not name or name.startswith("/") or "\\" in name or
                        any(part in ("", ".", "..") or ":" in part for part in components) or
                        stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR) or
                        name.rstrip("/") in seen or components == ["campaign.lock"]):
                        raise ValueError(f"Unsafe archive member: {name}")
                    seen.add(name.rstrip("/"))
                    if info.is_dir():
                        continue
                    target = root.joinpath(*components)
                    parent = root
                    for part in components:
                        parent = parent / part
                        if parent.is_symlink():
                            raise ValueError(f"Archive destination contains a symlink: {name}")
                    staged = stage.joinpath(*components)
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, staged.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    if target.exists() and (not target.is_file() or _sha256(target) != _sha256(staged)):
                        raise RuntimeError(f"Existing campaign evidence differs: {name}")
                    files.append((staged, target))
                for staged, target in files:
                    if not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(staged, target)
    return root


def _evaluate(directory, checkpoint, y, names, predictor, fingerprint):
    import numpy as np
    from pathlib import Path
    from .metrics import score_predictions
    directory = Path(directory) / checkpoint
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / "receipt.json"
    metrics_path = directory / "metrics.json"
    predictions = directory / "predictions.npz"
    if receipt.exists():
        claim = json.loads(receipt.read_text())
        if claim.get("status") != "complete" or claim.get("fingerprint") != fingerprint:
            raise RunStopped(f"Incomplete or incompatible evaluation receipt: {receipt}")
        if not predictions.is_file() or not metrics_path.is_file() or _sha256(predictions) != claim.get("prediction_sha256") or _sha256(metrics_path) != claim.get("metrics_sha256"):
            raise RunStopped("Evaluation cache is incomplete or corrupted")
        if claim.get("logits_sha256") and (not (directory / "logits.npz").is_file() or _sha256(directory / "logits.npz") != claim["logits_sha256"]):
            raise RunStopped("Evaluation logits cache is incomplete or corrupted")
        try:
            result = json.loads(metrics_path.read_text())
        except (ValueError, OSError) as exc:
            raise RunStopped("Evaluation metrics are corrupt") from exc
    else:
        if predictions.exists() or metrics_path.exists():
            raise RunStopped("Evaluation evidence exists without its receipt")
        with receipt.open("x") as stream:
            json.dump({"status": "claimed", "fingerprint": fingerprint}, stream)
        output = predictor()
        probabilities, logits = output if isinstance(output, tuple) else (output, None)
        if logits is not None:
            np.savez_compressed(directory / "logits.npz", logits=logits)
        result = score_predictions(y, probabilities, names)
        np.savez_compressed(predictions, y=np.asarray(y, dtype=np.int8), predictions=probabilities.argmax(1).astype(np.int8))
        _write_json(metrics_path, result)
        _write_json(receipt, {"status": "complete", "fingerprint": fingerprint, "test_prediction_passes": 1,
                              "prediction_sha256": _sha256(predictions), "metrics_sha256": _sha256(metrics_path),
                              "logits_sha256": _sha256(directory / "logits.npz") if logits is not None else None})
    item = {"metrics": result, "prediction_path": str(predictions), "prediction_sha256": _sha256(predictions),
            "metrics_path": str(metrics_path), "metrics_sha256": _sha256(metrics_path),
            "receipt_path": str(receipt), "receipt_sha256": _sha256(receipt)}
    logits_path = directory / "logits.npz"
    if logits_path.exists():
        item.update(logits_path=str(logits_path), logits_sha256=_sha256(logits_path))
    return item


def run_campaign(prepared_dir, output, study, *, device="cpu", campaign="gpu_v2", time_budget_hours=8, only="all",
                 zip_every=5, smoke_model=None, smoke_task=None, smoke_seeds=None,
                 smoke_epochs=None, checkpoint_hook=None):
    """Run the predeclared study using verified prepared inputs and resumable epochs."""
    import dataclasses
    import fcntl
    import os
    import pickle
    import random
    import time
    import warnings
    from pathlib import Path

    validate_campaign_name(campaign)
    if time_budget_hours <= 0 or zip_every < 1:
        raise ValueError("Positive time budget and zip interval required")
    if any(v is not None for v in (smoke_model, smoke_task, smoke_seeds, smoke_epochs)):
        if not all(v is not None for v in (smoke_model, smoke_task, smoke_seeds, smoke_epochs)):
            raise ValueError("Smoke controls must be specified together")
        if not campaign.endswith("_smoke") or len(smoke_seeds) != 2 or smoke_epochs != 2:
            raise ValueError("Smoke requires a distinct _smoke campaign, two seeds and two epochs")
    plans = [plan for plan in select_study_runs(study, only) if "reuse_study" not in plan]
    if smoke_model is not None:
        plans = [r for r in plans if r["model"] == smoke_model and r["task"] == smoke_task
                 and r["seed"] in smoke_seeds]
        if len(plans) != 2:
            raise ValueError("Smoke selection must resolve to exactly two independent runs")
    output = Path(output)
    root = output / campaign
    if not plans:
        print(f"No runs selected for {study} with --only {only}.", flush=True)
        return {"new_completed": 0, "stopped_count": 0, "stopped": [], "elapsed_seconds": 0.0, "root": str(root)}

    import numpy as np
    import torch
    from .baselines import build_model, fit_tree
    from .device import configure_device, environment_identity, effective_settings
    from .prepared import load_prepared
    from .train import TrainConfig, fit_resumable, seed_everything

    configured_device = "cpu" if only == "trees" and device in ("cuda", "auto") else device
    if configured_device != device:
        print(f"Using CPU for --only trees (requested device: {device}).", flush=True)
    root.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings(record=True) as config_warnings:
        warnings.simplefilter("always")
        actual = configure_device(configured_device)
    initial_warnings = [{"epoch": None, "phase": "configure", "message": str(w.message), "category": w.category.__name__} for w in config_warnings]
    versions = library_versions()
    versions.update(cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    gpu_name=torch.cuda.get_device_name(0) if actual == "cuda" else None)
    probe_file = root / f"probe_{actual}.json"
    has_neural_runs = any(plan["model"] not in TREE_MODELS for plan in plans)
    if only == "trees":
        print(f"{study}: {len(plans)} selected tree runs on CPU; no neural timing probe required.", flush=True)
    if has_neural_runs and study not in ("probe", "determinism") and smoke_model is None:
        if not probe_file.exists():
            raise RuntimeError("Run the three-epoch probe first on this campaign and device")
        probe = json.loads(probe_file.read_text())
        validate_probe(probe, environment_identity(actual, versions))
        _print_neural_projections(probe["seconds_per_epoch"], only)
    lock = (root / "campaign.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Campaign is already running")
    started = time.monotonic()
    complete = 0
    stopped = []
    prepared_cache = {}
    try:
        for plan in plans:
            if not budget_allows(time.monotonic() - started, time_budget_hours * 3600):
                break
            split = plan["split_seed"]
            if split not in prepared_cache:
                prepared_cache[split] = load_prepared(Path(prepared_dir) / f"split_{split}.npz")
                verify_prepared_manifest(prepared_dir, split, prepared_cache[split]["metadata"])
            prepared = prepared_cache[split]
            arrays, meta = prepared["arrays"], prepared["metadata"]
            try:
                task, name, seed = plan["task"], plan["model"], plan["seed"]
                tree = name in TREE_MODELS
                run_device = "cpu" if tree else actual
                max_epochs = smoke_epochs or (3 if study == "probe" else 60 if study == "B4" else 20 if task == "binary" else 30)
                cfg = TrainConfig(seed=seed, max_epochs=max_epochs, alpha_mode=meta["frozen_alpha"][task],
                                  patience=10 if plan["arm"] == "patience10_min15" else 6,
                                  lr=plan.get("lr", .0015))
                run_versions = {**versions, 'gpu_name': None} if tree else versions
                identity = {"campaign": campaign, "plan": plan, "device": run_device, "config": dataclasses.asdict(cfg),
                            "data_fingerprint": meta["data_fingerprint"], "payload_sha256": meta["payload_sha256"],
                            "prepared_sha256": _sha256(Path(prepared_dir) / f"split_{split}.npz"), "libraries": run_versions}
                if tree:
                    identity = _resolve_tree_identity(root, identity)
                    run_versions = identity['libraries']
                fingerprint = run_fingerprint(identity)
                directory = root / run_device / fingerprint
                directory.mkdir(parents=True, exist_ok=True)
                record_path = directory / "record.json"
                if record_path.exists():
                    try:
                        record = json.loads(record_path.read_text())
                    except (ValueError, OSError) as exc:
                        raise RunStopped("Completed run record is corrupt") from exc
                    if record.get("status") != "complete" or record.get("run_fingerprint") != fingerprint:
                        raise RunStopped("Existing run record has incompatible identity")
                    for item in record["checkpoints"].values():
                        for key in ("prediction", "metrics", "receipt", *( ["logits"] if "logits_path" in item else [] ), *( ["state_dict"] if "state_dict_path" in item else [] )):
                            path = root / item[f"{key}_path"]
                            if not path.is_file() or _sha256(path) != item[f"{key}_sha256"]:
                                raise RunStopped("Completed run evidence is incomplete or corrupt")
                    if study == "probe" and not probe_file.exists():
                        mean = float(np.mean([row["seconds"] for row in record["history"]]))
                        _write_json(probe_file, {"seconds_per_epoch": mean, "run_fingerprint": fingerprint, "environment": environment_identity(actual, versions)})
                    continue
                guard_evaluation_receipts(directory)
                if tree and (directory / "run_identity.json").exists() and (directory / "best/receipt.json").exists() and not (directory / "tree_in_progress.pkl").exists():
                    finalize_tree_cache(directory, root, fingerprint)
                    complete += 1
                    continue
                has_receipt = any((directory / label / "receipt.json").exists() for label in ("best", "last"))
                resume_path = directory / ("tree_in_progress.pkl" if tree else "epoch_checkpoint.pt")
                if has_receipt and not resume_path.exists():
                    raise RunStopped("Evaluation already exists but training checkpoint is missing; branch stopped")
                prefix = "y2_" if task == "binary" else "y12_"
                ytrain, yval, ytest = (arrays[prefix + k] for k in ("train", "val", "test"))
                names = ["normal", "attack"] if task == "binary" else prepared["class_names"]
                if only == "trees":
                    # PyTorch's global seeding helper also seeds CUDA generators.
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.default_generator.manual_seed(seed)
                    torch.set_num_threads(cfg.threads)
                    torch.use_deterministic_algorithms(True)
                else:
                    seed_everything(seed)
                with warnings.catch_warnings(record=True) as config_warnings:
                    warnings.simplefilter("always")
                    configure_device(actual)
                warning_ledger = list(initial_warnings) + [{"epoch": None, "phase": "configure", "message": str(w.message), "category": w.category.__name__} for w in config_warnings]
                model = build_model(name, len(names), seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if tree:
                        pending = {"schema_version": 1, "status": "pending", "campaign": campaign, **plan,
                                   "device": run_device, "requested_device": device, "config": dataclasses.asdict(cfg),
                                   "data_fingerprint": meta["data_fingerprint"], "payload_sha256": meta["payload_sha256"],
                                   "prepared_sha256": identity["prepared_sha256"], "run_fingerprint": fingerprint,
                                   "libraries": run_versions, "warnings": warning_ledger, "effective_settings": effective_settings(),
                                   "best_epoch": None, "best_val_macro_f1": None, "epochs_run": None, "history": [], "headline_checkpoint": "best"}
                        identity_path = directory / "run_identity.json"
                        if identity_path.exists():
                            preserved = json.loads(identity_path.read_text())
                            if (preserved.get('status') != 'pending' or preserved.get('run_fingerprint') != fingerprint
                                    or _tree_record_identity(preserved, plan) != identity):
                                raise RunStopped("Tree pending identity differs")
                            pending = preserved
                            warning_ledger = list(pending['warnings'])
                        if not identity_path.exists():
                            _write_json(identity_path, pending)
                    if tree:
                        model_path = directory / "tree_in_progress.pkl"
                        if model_path.exists():
                            with model_path.open("rb") as stream: model = pickle.load(stream)
                        else:
                            fit_tree(model, name, arrays["x_train"], ytrain)
                            temporary = model_path.with_suffix(".tmp")
                            with temporary.open("wb") as stream: pickle.dump(model, stream, protocol=5)
                            os.replace(temporary, model_path)
                        info = {"best_epoch": None, "best_val_macro_f1": None, "epochs_run": None, "history": []}
                    else:
                        info = fit_resumable(model, arrays["x_train"], ytrain, arrays["x_val"], yval, cfg, directory,
                                             identity=identity, device=actual, min_epochs=15 if plan["arm"] == "patience10_min15" else 0,
                                             early_stopping=plan["arm"] != "fixed30", checkpoint_hook=checkpoint_hook, warning_ledger=warning_ledger)
                    checkpoints = {}
                    for checkpoint in ("best", "last"):
                        if tree and checkpoint == "last":
                            # Trees have one fitted checkpoint: reuse its single test pass.
                            checkpoints[checkpoint] = dict(checkpoints["best"])
                            continue
                        if tree:
                            predictor = lambda: model.predict_proba(arrays["x_test"])
                        else:
                            model.load_state_dict(info[checkpoint + "_state"])
                            model.to(actual).eval()
                            def predictor():
                                with torch.inference_mode():
                                    logits = torch.cat([model(torch.as_tensor(arrays["x_test"][i:i + cfg.batch_size], dtype=torch.float32, device=actual)).cpu()
                                                        for i in range(0, len(ytest), cfg.batch_size)])
                                    probabilities = logits.softmax(1).numpy()
                                    return (probabilities, logits.numpy()) if study == "determinism" else probabilities
                        item = _evaluate(directory, checkpoint, ytest, names, predictor, fingerprint + ":" + checkpoint)
                        for key in ("prediction", "metrics", "receipt", *( ["logits"] if "logits_path" in item else [] )):
                            item[f"{key}_path"] = str(Path(item[f"{key}_path"]).relative_to(root))
                        checkpoints[checkpoint] = item
                        if name == "exlightnet":
                            model_path = directory / (checkpoint + "_state_dict.pt")
                            expected_state = info[checkpoint + "_state"]
                            if model_path.exists():
                                stored = torch.load(model_path, map_location="cpu", weights_only=True)
                                if stored.keys() != expected_state.keys() or any(not torch.equal(stored[k], expected_state[k]) for k in stored):
                                    raise RunStopped("Existing final state differs")
                            else:
                                temporary = model_path.with_suffix(".tmp")
                                torch.save(expected_state, temporary)
                                os.replace(temporary, model_path)
                            item["state_dict_path"] = str(model_path.relative_to(root))
                            item["state_dict_sha256"] = _sha256(model_path)
                    record = {"schema_version": 1, "status": "complete", "campaign": campaign, **plan,
                              "device": run_device, "requested_device": pending['requested_device'] if tree else device, "config": dataclasses.asdict(cfg),
                              "data_fingerprint": meta["data_fingerprint"], "payload_sha256": meta["payload_sha256"],
                              "prepared_sha256": identity["prepared_sha256"], "run_fingerprint": fingerprint, "libraries": run_versions,
                              "warnings": warning_ledger + [{"epoch": None, "phase": "run", "message": str(w.message), "category": w.category.__name__} for w in caught], "effective_settings": pending['effective_settings'] if tree else effective_settings(), "best_epoch": info["best_epoch"],
                              "best_val_macro_f1": info["best_val_macro_f1"], "epochs_run": info["epochs_run"],
                              "history": info["history"], "headline_checkpoint": "best", "checkpoints": checkpoints}
                    _write_json(record_path, record)
                    if tree: model_path.unlink()
                complete += 1
                if study == "probe":
                    mean = float(np.mean([row["seconds"] for row in info["history"]]))
                    _write_json(probe_file, {"seconds_per_epoch": mean, "run_fingerprint": fingerprint, "environment": environment_identity(actual, versions)})
                    print(f"Probe: {mean:.3f} seconds/epoch.", flush=True)
                    _print_neural_projections(mean, only)
                if complete % zip_every == 0:
                    package_campaign(root, output / f"{campaign}.zip")
            except RunStopped as exc:
                failure = {"status": "stopped", "campaign": campaign, "plan": plan, "reason": str(exc)}
                failure_path = root / "failures" / (run_fingerprint({"plan": plan}) + ".json")
                failure_path.parent.mkdir(parents=True, exist_ok=True)
                if not failure_path.exists():
                    _write_json(failure_path, failure)
                stopped.append(failure)
        if study == "determinism":
            repeats = []
            for record_path in (root / actual).glob("*/record.json"):
                record = json.loads(record_path.read_text())
                if record.get("study") == "determinism" and record.get("arm") in ("repeat1", "repeat2"):
                    repeats.append(record)
            if len(repeats) == 2 and {r["arm"] for r in repeats} == {"repeat1", "repeat2"}:
                if repeats[0]["data_fingerprint"] != repeats[1]["data_fingerprint"] or repeats[0]["payload_sha256"] != repeats[1]["payload_sha256"]:
                    raise RuntimeError("Determinism repeats used different input data")
                comparison = {"campaign": campaign, "device": actual, "runs": [r["run_fingerprint"] for r in repeats], "checkpoints": {}}
                for cp in ("best", "last"):
                    left, right = (r["checkpoints"][cp] for r in repeats)
                    with np.load(root / left["logits_path"], allow_pickle=False) as a, np.load(root / right["logits_path"], allow_pickle=False) as b:
                        difference = float(np.max(np.abs(a["logits"] - b["logits"])))
                    f1_difference = abs(left["metrics"]["macro_f1"] - right["metrics"]["macro_f1"])
                    comparison["checkpoints"][cp] = {"max_abs_logit_difference": difference, "abs_macro_f1_difference": f1_difference}
                    print(f"Determinism {cp}: max abs logits difference={difference}, macro-F1 difference={f1_difference}", flush=True)
                _write_json(root / f"determinism_{actual}.json", comparison)
    finally:
        try:
            package_campaign(root, output / f"{campaign}.zip")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    return {"new_completed": complete, "stopped_count": len(stopped), "stopped": stopped, "elapsed_seconds": time.monotonic() - started, "root": str(root)}
