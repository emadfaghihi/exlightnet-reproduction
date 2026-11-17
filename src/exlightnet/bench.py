"""Measured batch-one latency, single-thread libraries and process-lifetime RSS."""
import importlib.metadata
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits


def environment_info():
    versions = {}
    for package in ("numpy", "pandas", "scipy", "torch", "scikit-learn", "xgboost-cpu", "shap", "nbformat", "nbconvert", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")), cpu)
    return {"platform": platform.platform(), "cpu": cpu, "python": platform.python_version(), "versions": versions,
            "cuda_used": False, "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "threadpools": [{"api": p["user_api"], "library": p["internal_api"], "threads": p["num_threads"]} for p in threadpool_info()]}


def benchmark(model, x, warmup=100, iterations=1000):
    torch.set_num_threads(1)
    neural = isinstance(model, torch.nn.Module)
    if neural:
        model.eval()
        inputs = torch.as_tensor(x[:min(len(x), iterations)], dtype=torch.float32)
        infer = lambda i: model(inputs[i % len(inputs):i % len(inputs) + 1])
    else:
        model.set_params(n_jobs=1)
        infer = lambda i: model.predict_proba(x[i % len(x):i % len(x) + 1])
    times = []
    with threadpool_limits(limits=1), torch.inference_mode():
        for i in range(warmup):
            infer(i)
        for i in range(iterations):
            start = time.perf_counter_ns()
            infer(i)
            times.append((time.perf_counter_ns() - start) / 1e6)
    return {"latency_mean_ms": float(np.mean(times)), "latency_median_ms": float(np.median(times)),
            "latency_p95_ms": float(np.percentile(times, 95)), "warmup": warmup, "iterations": iterations,
            "batch_size": 1, "threads": 1, "input_partition": "train",
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "rss_scope": "process lifetime including preceding training; not model-exclusive"}
