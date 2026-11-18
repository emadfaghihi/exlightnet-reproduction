"""Phase-ordered reproduction CLI. Fast results are never promoted to full results."""
import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import ABLATIONS, MODELS, build_model, fit_tree
from .bench import benchmark, environment_info
from .data import GateFailure, load_data, make_split, validate_data
from .io import read_json, sha256, write_json
from .losses import ALPHA_MODES
from .metrics import evaluate_once, read_evaluation_cache, summarize
from .model import model_gate, parameter_count
from .preprocess import FlowPreprocessor
from .quantize import export_int8, load_int8, int8_size_breakdown
from .train import TrainConfig, fit_model, predict_proba, sampler_multiplicity, seed_everything


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_data(root=Path("."), download=True):
    root = Path(root)
    directory = root / "artifacts"
    directory.mkdir(parents=True, exist_ok=True)
    x, labels, binary = load_data(root / "data/RT_IOT2022.csv", download=download)
    d1 = validate_data(labels)
    split, d2 = make_split(labels)
    data_hash = sha256(root / "data/RT_IOT2022.csv")
    manifest = directory / "cache_manifest.json"
    settings = {"data_sha256": data_hash, "preprocessing_format": 1, "split_seed": 42,
                "mi": "binary,3-neighbors,one-hot-discrete,seed42", "top_k": 40}
    expected = digest(settings)
    if not manifest.exists() or read_json(manifest)["fingerprint"] != expected:
        pp = FlowPreprocessor(40).fit(x.iloc[split["train"]], binary[split["train"]])
        pp.save(directory)
        class_names, y12 = np.unique(labels, return_inverse=True)
        arrays = {}
        for name, index in split.items():
            arrays.update({"x_" + name: pp.transform(x.iloc[index]), "y12_" + name: y12[index], "y2_" + name: binary[index]})
        np.savez_compressed(directory / "cache_arrays.npz", **arrays)
        np.savez_compressed(directory / "split_indices.npz", **split)
        write_json(directory / "class_names.json", class_names)
        write_json(manifest, {"fingerprint": expected, "settings": settings, "arrays_sha256": sha256(directory / "cache_arrays.npz")})
    else:
        pp = FlowPreprocessor.load(directory)
        if not (directory / "cache_arrays.npz").exists():
            # The dataset-free archive intentionally omits arrays. Recreate them
            # with frozen train-fitted transforms, never fit against held-out data.
            with np.load(directory / "split_indices.npz", allow_pickle=False) as saved:
                if any(not np.array_equal(saved[k], split[k]) for k in split):
                    raise RuntimeError("Saved split differs from the fixed data partition")
            _, y12 = np.unique(labels, return_inverse=True)
            rebuilt = {}
            for name, index in split.items():
                rebuilt.update({"x_" + name: pp.transform(x.iloc[index]), "y12_" + name: y12[index], "y2_" + name: binary[index]})
            np.savez_compressed(directory / "cache_arrays.npz", **rebuilt)
        if sha256(directory / "cache_arrays.npz") != read_json(manifest)["arrays_sha256"]:
            raise RuntimeError("Cached arrays failed their checksum; preserve evidence and use a fresh directory")
    with np.load(directory / "cache_arrays.npz", allow_pickle=False) as cache:
        arrays = {k: cache[k] for k in cache.files}
    write_json(root / "results/data_gates.json", [d1, d2])
    return arrays, read_json(directory / "class_names.json"), pp, {"fingerprint": expected, "gates": [d1, d2]}


def choose_alpha(task, x_train, y_train, x_val, y_val, directory, fast=False, fingerprint=None):
    """Selection uses ONLY train/validation; no test arguments or metrics."""
    directory = Path(directory)
    path = directory / "selection.json"
    if path.exists():
        cached = read_json(path)
        if cached.get("data_fingerprint") != fingerprint:
            raise RuntimeError("Alpha selection data fingerprint differs")
        return cached
    k = 2 if task == "binary" else 12
    scores = []
    for mode in ALPHA_MODES:
        seed_everything(42)
        model = build_model("exlightnet", k)
        cfg = TrainConfig(max_epochs=1 if fast else 8, alpha_mode=mode)
        info = fit_model(model, x_train, y_train, x_val, y_val, cfg, directory / mode)
        scores.append({"mode": mode, "validation_macro_f1": info["best_val_macro_f1"], "best_epoch": info["best_epoch"]})
    best = max(scores, key=lambda r: r["validation_macro_f1"])
    result = {"chosen": best["mode"], "selection_metric": "seed42 validation macro-F1", "seed": 42, "candidates": scores,
              "sweep_max_epochs": 1 if fast else 8, "tie_break": "first in inv_freq,sqrt_inv_freq,effective_num,none", "test_used": False, "data_fingerprint": fingerprint}
    write_json(path, result)
    return result


def _run_one(task, name, seed, arrays, class_names, selection, context, directory, max_epochs, cache_only=False):
    classes = 2 if task == "binary" else 12
    names = ["normal", "attack"] if classes == 2 else class_names
    prefix = "y2_" if classes == 2 else "y12_"
    width = 20 if name == "top20" else 40
    xtrain, xval, xtest = [arrays["x_" + key][:, :width] for key in ("train", "val", "test")]
    ytrain, yval, ytest = [arrays[prefix + key] for key in ("train", "val", "test")]
    config = TrainConfig(seed=seed, max_epochs=max_epochs, alpha_mode=selection["chosen"] if selection else "inv_freq",
                         sampler="balanced" if name == "balanced" else "none" if name == "no_sampler" else "tempered")
    directory = Path(directory)
    metric_path = directory / "metrics.json"
    recipe = dataclasses.asdict(config) if name not in ("random_forest", "xgboost") else {"trees": 300 if name == "random_forest" else 400, "seed": seed, "max_depth": None if name == "random_forest" else 8, "balanced_weights": True}
    if metric_path.exists():
        checkpoint = directory / ("model.pkl" if name in ("random_forest", "xgboost") else "model.pt")
        if not checkpoint.exists():
            raise RuntimeError(f"Incomplete cached evaluation: missing checkpoint {checkpoint}")
        cached = read_evaluation_cache(directory)
        if cached["data_fingerprint"] != context["fingerprint"] or cached["recipe"] != recipe:
            raise RuntimeError(f"Cached recipe/data differs at {directory}; choose a new output directory")
        expected = digest({"weights": sha256(checkpoint), "data": context["fingerprint"], "task": task, "recipe": recipe})
        if cached.get("fingerprint") != expected or (cached.get("task"), cached.get("model"), cached.get("seed")) != (task, name, seed):
            raise RuntimeError("Cached checkpoint fingerprint/identity differs; test inference will not be repeated")
        return cached
    if cache_only:
        raise RuntimeError(f"Missing cache-only final result at {directory}; aggregation cannot train or evaluate test data")
    seed_everything(seed)
    model = build_model(name, classes, seed)
    directory.mkdir(parents=True, exist_ok=True)
    neural = isinstance(model, torch.nn.Module)
    model_path = directory / ("model.pt" if neural else "model.pkl")
    if neural:
        if (directory / "training.json").exists() and model_path.exists():
            info = read_json(directory / "training.json")
            if info["config"] != dataclasses.asdict(config):
                raise RuntimeError("Existing checkpoint training config differs")
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        else:
            info = fit_model(model, xtrain, ytrain, xval, yval, config, directory)
        params = parameter_count(model)
        predictor = lambda: predict_proba(model, xtest)
    else:
        start = time.perf_counter()
        if model_path.exists():
            with model_path.open("rb") as stream:
                model = pickle.load(stream)
        else:
            fit_tree(model, name, xtrain, ytrain)
            with model_path.open("wb") as stream:
                pickle.dump(model, stream, protocol=5)
        info = {"training_seconds": time.perf_counter() - start, "config": recipe}
        params = None
        predictor = lambda: model.predict_proba(xtest)
    metadata = {"task": task, "model": name, "seed": seed, "params": params, "serialized_bytes": model_path.stat().st_size,
                "training": info, "recipe": recipe, "data_fingerprint": context["fingerprint"], "features": width,
                "fingerprint": digest({"weights": sha256(model_path), "data": context["fingerprint"], "task": task, "recipe": recipe})}
    return evaluate_once(directory, ytest, predictor, names, metadata)


def _load_run_model(name, classes, directory):
    directory = Path(directory)
    if name in ("random_forest", "xgboost"):
        with (directory / "model.pkl").open("rb") as stream:
            return pickle.load(stream)
    model = build_model(name, classes)
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    return model.eval()


def ablation_comparison(rows, campaign="cpu_v1"):
    """Compute future paired ablation output from copies, with task-wide Holm."""
    from .stats import paired_tests
    normalized = [{**row, "campaign": row.get("campaign", campaign),
                   "device": row.get("device", "cpu"), "split_seed": row.get("split_seed", 42)}
                  for row in rows if row["task"] == "multiclass"]
    if not normalized or not any(row["model"] == "exlightnet" for row in normalized):
        return []
    frame = paired_tests(normalized)
    if frame.empty:
        return []
    selected = frame[frame.model.isin(ABLATIONS)]
    return [{**{key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                for key, value in record.items()}, "variant": record["model"]}
            for record in selected.to_dict(orient="records")]


def sampler_gate(y):
    rare = str(int(np.bincount(y).argmin()))
    tempered, balanced = sampler_multiplicity(y, .5), sampler_multiplicity(y, 1.)
    return {"gate": "T1", "pass": bool(abs(tempered[rare] / 30 - 1) <= .15 and abs(balanced[rare] / 359 - 1) <= .15),
            "actual": {"rarest_train_count": int(np.bincount(y).min()), "tempered_multiplicity": tempered[rare], "balanced_multiplicity": balanced[rare]}}


def _blocked_report(root, directory, gates, failures, start):
    manifest = {"status": "blocked", "failures": failures, "elapsed_seconds_this_invocation": time.perf_counter() - start}
    write_json(directory / "gates.json", gates)
    write_json(directory / "manifest.json", manifest)
    from .report import build_report
    build_report(root)
    return manifest


def _run_core(root=Path("."), mode="fast", models=MODELS, tasks=("binary", "multiclass"), seeds=None, epochs=None, ablations=True, explanations=True, leakage_proxy=False,
              skip_benchmark=False, alpha_only=False, seed_worker=False, aggregate_only=False):
    from .execution import load_alpha, load_prepared_data, publish, utc_now
    root = Path(root).resolve()
    fast = mode == "fast"
    seeds = list(seeds) if seeds is not None else [42] if fast else list(range(42, 47))
    if fast and (seeds != [42] or (epochs is not None and not 1 <= epochs <= 8)):
        raise ValueError("Fast mode uses seed 42 and at most eight epochs")
    if not fast and epochs is not None:
        raise ValueError("Full mode fixes the specified 20/30 epoch limits")
    directory = root / "results" / mode
    directory.mkdir(parents=True, exist_ok=True)
    publication = directory / "workers" / str(seeds[0]) if seed_worker else directory
    publication.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    started_at = utc_now()
    failures, rows, gates = [], [], []
    def failure(branch, exc):
        print(f"STOP {branch}: {exc}", flush=True)
        failures.append({"branch": branch, "error_type": type(exc).__name__, "reason": str(exc)})
        write_json(publication / "failures.json", failures)
    try:
        arrays, class_names, pp, context = load_prepared_data(root) if seed_worker or aggregate_only else prepare_data(root)
        gates += context["gates"] + [pp.compare_table4()]
    except (GateFailure, FileNotFoundError) as exc:
        if isinstance(exc, GateFailure):
            gates.append(exc.gate)
        failure("data and all dependent experiments", exc)
        if seed_worker:
            result = {"status": "blocked", "failures": failures, "seed": seeds[0], "pid": os.getpid()}
            write_json(publication / "manifest.json", result)
            return result
        return _blocked_report(root, directory, gates, failures, start)
    neural_ready = True
    try:
        gates.append(model_gate())
    except (AssertionError, GateFailure) as exc:
        neural_ready = False
        gates.append(exc.gate if isinstance(exc, GateFailure) else {"gate": "M1", "pass": False, "actual": {"assertion": str(exc)}})
        failure("ExLightNet architecture and dependent alpha/deep branches", exc)
    t1 = sampler_gate(arrays["y12_train"])
    gates.append(t1)
    if not t1["pass"]:
        neural_ready = False
        failure("tempered-sampler training branches", GateFailure("T1", t1["actual"]))
    seed_everything(42)
    write_json(publication / "environment.json", environment_info())
    selections = {}
    for task in tasks:
        if not neural_ready:
            continue
        print(f"Alpha selection: {task}, validation only", flush=True)
        prefix = "y2_" if task == "binary" else "y12_"
        try:
            selections[task] = load_alpha(task, directory, context["fingerprint"], fast) if seed_worker or aggregate_only else choose_alpha(task, arrays["x_train"], arrays[prefix + "train"], arrays["x_val"], arrays[prefix + "val"], directory / "alpha" / task, fast, context["fingerprint"])
        except Exception as exc:
            failure(f"alpha/{task} and dependent deep-model branches", exc)
    if not seed_worker:
        write_json(directory / "alpha_selection.json", selections)
    if alpha_only:
        result = {"stage": "alpha_only", "seconds": time.perf_counter() - start, "started_at": started_at, "finished_at": utc_now(),
                  "tasks": list(tasks), "seed": 42, "sweep_max_epochs": 1 if fast else 8, "processes": 1, "threads": 1,
                  "test_used": False, "data_fingerprint": context["fingerprint"], "failures": failures}
        write_json(directory / "alpha_sweep_manifest.json", result)
        return result
    # Phases 4--6: all headline models; no test predictions during alpha selection.
    for task in tasks:
        limit = (epochs or 1) if fast else 20 if task == "binary" else 30
        for name in models:
            if name not in ("random_forest", "xgboost") and task not in selections:
                continue
            for seed in seeds:
                print(f"Training/final reporting: {mode}/{task}/{name}/{seed}", flush=True)
                try:
                    row = _run_one(task, name, seed, arrays, class_names, selections.get(task), context, directory / task / name / str(seed), limit, **({"cache_only": True} if aggregate_only else {}))
                    rows.append(row)
                    if not seed_worker:
                        summarize(rows).to_csv(directory / "summary.csv", index=False)
                except Exception as exc:
                    failure(f"{task}/{name}/{seed}", exc)
    # Phase 7: all ablations use the frozen alpha, with five seeds in full mode.
    if ablations and "multiclass" in selections:
        for name in ABLATIONS:
            for seed in seeds:
                print(f"Ablation: {name}/{seed}", flush=True)
                try:
                    rows.append(_run_one("multiclass", name, seed, arrays, class_names, selections["multiclass"], context,
                                         directory / "multiclass" / name / str(seed), (epochs or 1) if fast else 30, **({"cache_only": True} if aggregate_only else {})))
                except Exception as exc:
                    failure(f"ablation/{name}/{seed}", exc)
    if not seed_worker:
        write_json(directory / "ablations.json", ablation_comparison(rows, campaign="cpu_v1" if mode == "full" else "fast_demo"))
    # Phase 8: separate W8 artifact receives exactly one final test prediction pass.
    quant_pass = {}
    quant_rows = []
    for row in list(rows):
        if row["model"] != "exlightnet":
            continue
        task, seed = row["task"], row["seed"]
        run_dir = directory / task / "exlightnet" / str(seed)
        try:
            qpath = run_dir / "model_int8.npz"
            qdir = directory / task / "exlightnet_w8" / str(seed)
            prefix = "y2_" if task == "binary" else "y12_"
            if aggregate_only:
                qrow = read_evaluation_cache(qdir)
                if qrow["fingerprint"] != digest({"weights": sha256(qpath), "data": context["fingerprint"]}):
                    raise RuntimeError("W8 cached checkpoint fingerprint differs")
                evidence = read_json(qdir / "quantization_gate.json")["actual"]
                evidence = {k: v for k, v in evidence.items() if k not in ("fp32_macro_f1", "w8_macro_f1", "delta", "fp32_bytes")}
                evidence["size_breakdown"] = int8_size_breakdown(qpath)
            else:
                model = _load_run_model("exlightnet", 2 if task == "binary" else 12, run_dir)
                evidence = export_int8(model, qpath, {"feature_names": pp.selected_names_, "z_mean": pp.mean_.tolist(), "z_scale": pp.scale_.tolist(), "data_fingerprint": context["fingerprint"]})
                quantized = load_int8(qpath)
                qrow = evaluate_once(qdir, arrays[prefix + "test"], lambda: predict_proba(quantized, arrays["x_test"]),
                                 ["normal", "attack"] if task == "binary" else class_names,
                                 {"task": task, "model": "exlightnet_w8", "seed": seed, "params": row["params"], "serialized_bytes": qpath.stat().st_size,
                                  "fingerprint": digest({"weights": sha256(qpath), "data": context["fingerprint"]}), "quantization": evidence})
            delta = qrow["macro_f1"] - row["macro_f1"]
            passed = delta >= -.01 if task == "multiclass" else abs(delta) <= .001
            gate = {"gate": "Q1", "task": task, "seed": seed, "pass": passed,
                    "actual": {"fp32_macro_f1": row["macro_f1"], "w8_macro_f1": qrow["macro_f1"], "delta": delta,
                               "fp32_bytes": row["serialized_bytes"], **evidence}, "criterion": "drop <=0.01" if task == "multiclass" else "absolute delta <=0.001 (predeclared meaning of approximately zero)"}
            gates.append(gate)
            quant_pass[(task, seed)] = passed
            write_json(qdir / "quantization_gate.json", gate)
            qrow["downstream_stopped"] = not passed
            quant_rows.append(qrow)
            if not passed:
                failure(f"quantized downstream/{task}/{seed}", RuntimeError(f"Q1 failed: {delta}"))
        except Exception as exc:
            failure(f"quantization/{task}/{seed}", exc)
    rows += quant_rows
    # Phase 9: benchmark training flows; accuracy estimates reuse frozen test predictions.
    for row in ([] if skip_benchmark else rows):
        task, name, seed = row["task"], row["model"], row["seed"]
        run_dir = directory / task / name / str(seed)
        if "latency_mean_ms" in row or row.get("downstream_stopped"):
            continue
        try:
            model = load_int8(directory / task / "exlightnet" / str(seed) / "model_int8.npz") if name == "exlightnet_w8" else _load_run_model(name, 2 if task == "binary" else 12, run_dir)
            row.update(benchmark(model, arrays["x_train"][:, :20 if name == "top20" else 40]))
            write_json(run_dir / "metrics.json", row)
        except Exception as exc:
            failure(f"benchmark/{task}/{name}/{seed}", exc)
    # Phase 10: seed 42 owns the shared explanation cache. Aggregation reads only.
    if explanations and not aggregate_only and (not seed_worker or seeds == [42]) and any(r["task"] == "multiclass" and r["model"] == "exlightnet" and r["seed"] == 42 for r in rows):
        try:
            from .explain import run_explanations
            edir = directory / "explain"
            if not (edir / "explanations.json").exists():
                rdir = directory / "multiclass/exlightnet/42"
                model = _load_run_model("exlightnet", 12, rdir)
                quantized = load_int8(rdir / "model_int8.npz") if quant_pass.get(("multiclass", 42)) else None
                with np.load(rdir / "predictions.npz") as cached:
                    probs = cached["probabilities"]
                run_explanations(model, quantized, arrays["x_train"], arrays["x_test"], arrays["y12_test"], probs,
                                 class_names, pp.selected_names_, edir, fast)
        except Exception as exc:
            failure("explainability", exc)
    elif explanations and aggregate_only and not (directory / "explain/explanations.json").exists():
        failure("explainability cache", RuntimeError("Missing explanation cache; aggregate-only never recomputes SHAP or deletion queries"))
    # Phase 11 is an explicitly optional proxy, never a campaign-disjoint estimate.
    if leakage_proxy and aggregate_only:
        if not (directory / "leakage_proxy/proxy_report.json").exists():
            failure("leakage PROXY cache", RuntimeError("Missing proxy cache; aggregate-only never trains a proxy"))
    elif leakage_proxy and not seed_worker:
        try:
            from .proxy import run_proxy
            write_json(directory / "results.json", rows)
            run_proxy(root, arrays, pp, selections, directory)
        except Exception as exc:
            failure("leakage PROXY", exc)
    manifest = {"mode": mode, "seeds": seeds, "tasks": list(tasks), "models": list(models), "fast_max_epochs": epochs or 1 if fast else None,
                "full_five_seed_headlines_complete": not fast and seeds == list(range(42, 47)) and all(len([r for r in rows if r["task"] == task and r["model"] == name]) == 5 for task in ("binary", "multiclass") for name in MODELS),
                "elapsed_seconds_this_invocation": time.perf_counter() - start, "failures": failures,
                "optional_kd": "not implemented (lowest priority)", "leakage_proxy_requested": leakage_proxy,
                "stage": "seed_worker" if seed_worker else "aggregate_only" if aggregate_only else "run",
                "skip_benchmark": skip_benchmark, "pid": os.getpid(), "threads": 1, "started_at": started_at, "finished_at": utc_now(),
                "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}}
    if seed_worker:
        write_json(publication / "results.json", rows)
        write_json(publication / "gates.json", gates)
        write_json(publication / "failures.json", failures)
        write_json(publication / "manifest.json", manifest)
    else:
        publish(root, mode, rows, gates, failures, manifest)
    print(f"Finished {mode}: {len(rows)} results, {len(failures)} stopped branches; {directory}", flush=True)
    return manifest


def run(root=Path("."), mode="fast", models=MODELS, tasks=("binary", "multiclass"), seeds=None, epochs=None, ablations=True, explanations=True, leakage_proxy=False,
        skip_benchmark=False, alpha_only=False, seed_worker=False, aggregate_only=False, benchmark_only=False, device="cpu"):
    if device != "cpu":
        raise ValueError("Canonical run_all is CPU-only; use the separate supplementary campaign runner for CUDA/auto")
    from .execution import configure_single_thread, execution_lock
    if sum((alpha_only, seed_worker, aggregate_only, benchmark_only)) > 1:
        raise ValueError("Choose one execution stage")
    if seed_worker and (seeds is None or len(seeds) != 1):
        raise ValueError("A seed worker requires exactly one seed")
    if benchmark_only and skip_benchmark:
        raise ValueError("--benchmark-only cannot be combined with --skip-benchmark")
    with execution_lock(root, mode, seeds[0] if seed_worker else None), configure_single_thread():
        if benchmark_only:
            from .execution import benchmark_only_pass
            return benchmark_only_pass(root, mode)
        return _run_core(root, mode, models, tasks, seeds, epochs, ablations, explanations, leakage_proxy,
                         skip_benchmark or seed_worker or aggregate_only or alpha_only, alpha_only, seed_worker, aggregate_only)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--device", choices=["cpu"], default="cpu", help="canonical CPU reproduction; supplementary devices use scripts/kaggle_campaign.py")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--tasks", nargs="+", choices=["binary", "multiclass"], default=["binary", "multiclass"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--epochs", type=int, help="fast only, 1..8; default 1")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--no-explanations", action="store_true")
    parser.add_argument("--leakage-proxy", action="store_true", help="optional class-clustering PROXY, not campaign-disjoint")
    parser.add_argument("--skip-benchmark", action="store_true", help="defer all timing to a later isolated pass")
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument("--alpha-only", action="store_true", help="prepare data and select alpha once; no final-model training")
    stages.add_argument("--seed-worker", action="store_true", help="one --seeds value; read frozen alpha and write only this seed's results")
    stages.add_argument("--aggregate-only", action="store_true", help="reuse cached final results; never train or evaluate test data")
    stages.add_argument("--benchmark-only", action="store_true", help="one process, idle CPU, cached models only; refresh timing and summaries")
    args = parser.parse_args()
    result = run(args.root, args.mode, args.models, args.tasks, args.seeds, args.epochs, not args.no_ablations, not args.no_explanations, args.leakage_proxy,
                 args.skip_benchmark, args.alpha_only, args.seed_worker, args.aggregate_only, args.benchmark_only, device=args.device)
    if result.get("status") == "blocked":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
