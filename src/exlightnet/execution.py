"""Process coordination for parallel seeds and one isolated benchmark pass."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from .bench import benchmark, environment_info
from .io import read_json, sha256, write_json
from .preprocess import FlowPreprocessor


def configure_single_thread():
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    return threadpool_limits(limits=1)


@contextmanager
def execution_lock(root, mode, seed=None):
    """Workers share the stage lock; aggregation/benchmarking exclude workers."""
    directory = Path(root) / "results" / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    handles = []
    try:
        for name, kind in [("stage", fcntl.LOCK_SH if seed is not None else fcntl.LOCK_EX)] + ([(f"seed-{seed}", fcntl.LOCK_EX)] if seed is not None else []):
            handle = (directory / (name + ".lock")).open("a+")
            handles.append(handle)
            try:
                fcntl.flock(handle, kind | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("Another training/aggregation/benchmark process holds this execution lock") from exc
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def load_prepared_data(root):
    root = Path(root)
    directory = root / "artifacts"
    required = ["cache_manifest.json", "cache_arrays.npz", "preprocessor.json", "class_names.json"]
    if any(not (directory / name).exists() for name in required):
        raise RuntimeError("Prepared data cache missing; run --alpha-only once before seed workers or aggregation")
    manifest = read_json(directory / "cache_manifest.json")
    if sha256(directory / "cache_arrays.npz") != manifest["arrays_sha256"]:
        raise RuntimeError("Prepared array cache checksum differs")
    csv = root / "data/RT_IOT2022.csv"
    if csv.exists() and sha256(csv) != manifest["settings"]["data_sha256"]:
        raise RuntimeError("Dataset differs from prepared cache")
    with np.load(directory / "cache_arrays.npz", allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    gates = read_json(root / "results/data_gates.json")
    if not all(g["pass"] for g in gates):
        raise RuntimeError("Prepared data gates did not pass")
    return arrays, read_json(directory / "class_names.json"), FlowPreprocessor.load(directory), {"fingerprint": manifest["fingerprint"], "gates": gates}


def load_alpha(task, directory, fingerprint, fast=False):
    path = Path(directory) / "alpha" / task / "selection.json"
    if not path.exists():
        raise RuntimeError(f"Alpha cache missing for {task}; run --alpha-only once")
    result = read_json(path)
    if result.get("data_fingerprint") != fingerprint or result.get("sweep_max_epochs") != (1 if fast else 8):
        raise RuntimeError("Alpha cache fingerprint/recipe differs")
    return result


def read_cpu_ticks():
    path = Path("/proc/stat")
    if not path.exists():
        raise RuntimeError("CPU-idle verification currently requires Linux /proc/stat")
    values = [int(n) for n in path.read_text().splitlines()[0].split()[1:9]]
    total = sum(values)
    return total - values[3] - values[4], total


class IdleCPUUnavailable(RuntimeError):
    pass


def wait_for_idle(max_busy_fraction=.05, consecutive=3, sample_interval=1., timeout=300., reader=read_cpu_ticks, sleeper=time.sleep):
    start = time.perf_counter()
    previous = reader()
    windows = []
    quiet = 0
    while time.perf_counter() - start < timeout:
        sleeper(sample_interval)
        current = reader()
        delta = current[1] - previous[1]
        busy = (current[0] - previous[0]) / delta if delta > 0 else 1.
        windows.append(float(busy))
        quiet = quiet + 1 if busy <= max_busy_fraction else 0
        if quiet >= consecutive:
            return {"max_busy_fraction": max_busy_fraction, "consecutive_windows": consecutive,
                    "sample_interval_s": sample_interval, "quiet_window_busy_fractions": windows[-consecutive:],
                    "wait_seconds": time.perf_counter() - start, "observed_windows": len(windows)}
        previous = current
    raise IdleCPUUnavailable(f"CPU did not remain idle (<= {max_busy_fraction:.0%} busy) for {consecutive} windows within {timeout}s")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def publish(root, mode, rows, gates, failures, manifest, plots=True):
    from .metrics import summarize
    from .report import build_report, make_result_plots
    root = Path(root)
    directory = root / "results" / mode
    summary = summarize(rows)
    summary.to_csv(directory / "summary.csv", index=False)
    summary.to_csv(root / "results/summary.csv", index=False)
    write_json(directory / "results.json", rows)
    write_json(directory / "gates.json", gates)
    write_json(directory / "failures.json", failures)
    write_json(directory / "manifest.json", manifest)
    write_json(root / "results/summary_scope.json", {"mode": mode, "seeds": manifest["seeds"], "source": f"{mode}/summary.csv",
               "full_five_seed_headlines_complete": manifest["full_five_seed_headlines_complete"]})
    if plots:
        make_result_plots(root, mode)
    build_report(root)


def benchmark_only_pass(root, mode):
    """One process, persisted models only; never fit or predict a test split."""
    from .metrics import read_evaluation_cache
    from .run_all import _load_run_model, digest
    from .quantize import load_int8
    root = Path(root).resolve()
    directory = root / "results" / mode
    if not (directory / "results.json").exists():
        raise RuntimeError("Aggregate cached seed results before --benchmark-only")
    arrays, _, _, context = load_prepared_data(root)
    rows = read_json(directory / "results.json")
    if not rows:
        raise RuntimeError("No aggregated results are available to benchmark")
    manifest = read_json(directory / "manifest.json")
    gates = read_json(directory / "gates.json")
    failures = [f for f in read_json(directory / "failures.json") if not f["branch"].startswith("benchmark/")]
    started = utc_now()
    start = time.perf_counter()
    campaign = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + f"-{os.getpid()}"
    output = directory / "benchmarks" / campaign
    output.mkdir(parents=True)
    measured = []
    benchmark_failures = []
    timing_fields = ("latency_mean_ms", "latency_median_ms", "latency_p95_ms", "peak_rss_bytes", "rss_scope", "warmup", "iterations", "batch_size", "threads", "input_partition", "benchmark_context")
    for row in rows:
        previous = {key: row.pop(key) for key in timing_fields if key in row}
        if previous:
            write_json(output / f"{row['task']}_{row['model']}_{row['seed']}_previous.json", previous)
    attempted = 0
    for row in rows:
        if row.get("downstream_stopped"):
            continue
        task, name, seed = row["task"], row["model"], row["seed"]
        run_dir = directory / task / name / str(seed)
        attempted += 1
        try:
            cached = read_evaluation_cache(run_dir)
            if cached["fingerprint"] != row["fingerprint"]:
                raise RuntimeError("Aggregate metrics fingerprint differs from cached evaluation")
            if name == "exlightnet_w8":
                checkpoint = directory / task / "exlightnet" / str(seed) / "model_int8.npz"
                expected = digest({"weights": sha256(checkpoint), "data": context["fingerprint"]})
                model = load_int8(checkpoint)
            else:
                checkpoint = run_dir / ("model.pkl" if name in ("random_forest", "xgboost") else "model.pt")
                if not checkpoint.exists():
                    raise RuntimeError(f"Missing local checkpoint {checkpoint}; *.pkl files are intentionally excluded from the archive")
                expected = digest({"weights": sha256(checkpoint), "data": context["fingerprint"], "task": task, "recipe": row["recipe"]})
                model = _load_run_model(name, 2 if task == "binary" else 12, run_dir)
            if expected != row["fingerprint"]:
                raise RuntimeError("Benchmark checkpoint fingerprint differs from final test artifact")
            print(f"Waiting for idle CPU: {task}/{name}/{seed}", flush=True)
            idle = wait_for_idle()
            values = benchmark(model, arrays["x_train"][:, :20 if name == "top20" else 40])
            values.update({"rss_scope": "benchmark-only process lifetime across preceding benchmarked models; no training",
                           "benchmark_context": {"campaign": campaign, "stage": "benchmark_only", "pid": os.getpid(), "processes": 1,
                                                 "idle_cpu": idle, "timestamp": utc_now(), "model_fingerprint": expected}})
            row.update(values)
            write_json(run_dir / "metrics.json", row)
            record = {"task": task, "model": name, "seed": seed, **values}
            write_json(output / f"{task}_{name}_{seed}.json", record)
            measured.append(record)
            print(f"Measured: {task}/{name}/{seed}, mean={values['latency_mean_ms']:.6f} ms", flush=True)
            del model
        except Exception as exc:
            record = {"branch": f"benchmark/{task}/{name}/{seed}", "error_type": type(exc).__name__, "reason": str(exc), "stage": "benchmark_only", "campaign": campaign}
            benchmark_failures.append(record)
            write_json(run_dir / "metrics.json", row)
            print(f"STOP {record['branch']}: {exc}", flush=True)
            if isinstance(exc, IdleCPUUnavailable):
                # This prerequisite affects the entire remaining benchmark pass.
                break
    result = {"stage": "benchmark_only", "campaign": campaign, "started_at": started, "finished_at": utc_now(),
              "seconds": time.perf_counter() - start, "pid": os.getpid(), "processes": 1, "threads": 1,
              "environment": environment_info(), "measured_models": len(measured), "attempted_models": attempted,
              "failed_models": len(benchmark_failures), "skipped_models": sum(not r.get("downstream_stopped", False) for r in rows) - attempted,
              "gated_out_models": sum(bool(r.get("downstream_stopped")) for r in rows), "failures": benchmark_failures}
    write_json(output / "manifest.json", result)
    write_json(directory / "benchmark_manifest.json", result)
    manifest["benchmark"] = result
    manifest["failures"] = failures + benchmark_failures
    # Keep per-run and aggregate timing views consistent after a failed refresh.
    for row in rows:
        write_json(directory / row["task"] / row["model"] / str(row["seed"]) / "metrics.json", row)
    publish(root, mode, rows, gates, manifest["failures"], manifest, plots=False)
    return result
