#!/usr/bin/env python3
"""Standalone NumPy-only ExLightNet W8 reader and floating-point CPU benchmark.

python bench_pi.py model_int8.npz [--input standardized_flows.npy]
Without --input, benchmarks a zero vector in standardized feature space.
This script neither extracts raw network features nor computes SHAP.
"""
import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import json
import platform
import resource
import sys
import time

import numpy as np


class NumpyExLightNet:
    def __init__(self, path):
        self.weights = {}
        with np.load(path, allow_pickle=False) as archive:
            self.meta = json.loads(str(archive["__meta__"]))
            for key in archive.files:
                if key.startswith("q::"):
                    q, s = archive[key], archive["s::" + key[3:]]
                    self.weights[key[3:]] = q.astype(np.float32) * s.reshape((-1,) + (1,) * (q.ndim - 1))
                elif key.startswith("f::"):
                    self.weights[key[3:]] = archive[key].copy()
        if self.meta["standard"] or not self.meta["use_se"]:
            raise ValueError("This standalone reader expects the full ExLightNet artifact")

    def conv(self, x, key, depthwise=False):
        w = self.weights[key + ".weight"]
        windows = np.lib.stride_tricks.sliding_window_view(x, w.shape[-1], axis=-1)
        if depthwise:
            return (windows * w[:, 0, :][None, :, None, :]).sum(axis=-1)
        return np.einsum("bclk,ock->bol", windows, w, optimize=True)

    def bn(self, x, key):
        w = self.weights
        scale = w[key + ".weight"] / np.sqrt(w[key + ".running_var"] + self.meta["bn_eps"])
        bias = w[key + ".bias"] - w[key + ".running_mean"] * scale
        return x * scale[None, :, None] + bias[None, :, None]

    def linear(self, x, key):
        return x @ self.weights[key + ".weight"].T + self.weights[key + ".bias"]

    def block(self, x, name):
        x = self.conv(self.conv(x, name + ".conv.0", True), name + ".conv.1")
        x = np.maximum(self.bn(x, name + ".bn"), 0)
        descriptor = x.mean(axis=-1)
        gate = self.linear(np.maximum(self.linear(descriptor, name + ".se.fc1"), 0), name + ".se.fc2")
        gate = 1 / (1 + np.exp(-np.clip(gate, -80, 80)))
        return x * gate[:, :, None]

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(len(x), 1, -1)
        x = np.maximum(self.bn(self.conv(x, "stem.0"), "stem.1"), 0)
        x = self.block(self.block(x, "block1"), "block2")
        return self.linear(np.concatenate([x.mean(axis=-1), x.max(axis=-1)], axis=1), "head")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    parser.add_argument("--input", help=".npy of already standardized flows in the saved MI order")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    model = NumpyExLightNet(args.artifact)
    width = len(model.meta.get("feature_names", [])) or 40
    x = np.load(args.input, allow_pickle=False).astype("float32") if args.input else np.zeros((1, width), dtype="float32")
    for i in range(args.warmup):
        model(x[i % len(x):i % len(x) + 1])
    times = []
    for i in range(args.iterations):
        row = x[i % len(x):i % len(x) + 1]
        start = time.perf_counter_ns()
        model(row)
        times.append((time.perf_counter_ns() - start) / 1e6)
    print(json.dumps({"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                      "python": platform.python_version(), "numpy": np.__version__, "mean_ms": float(np.mean(times)),
                      "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                      "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
                      "artifact_bytes": os.path.getsize(args.artifact), "warmup": args.warmup, "iterations": args.iterations,
                      "input_source": args.input or "zero vector in z-space", "weight_load": "dequantized once at load", "compute": "float32; no INT8 compute or multi-core throughput claim"}, indent=2))


if __name__ == "__main__":
    main()
