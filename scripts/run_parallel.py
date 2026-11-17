#!/usr/bin/env python3
"""One alpha sweep, five single-thread seeds, cached aggregation, idle benchmarking."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def _validate_alpha(root):
    manifest = root / "artifacts/cache_manifest.json"
    if not manifest.exists():
        raise RuntimeError("Prepared data cache missing; run --alpha-only first")
    fingerprint = json.loads(manifest.read_text())["fingerprint"]
    for task in ("binary", "multiclass"):
        path = root / "results/full/alpha" / task / "selection.json"
        if not path.exists():
            raise RuntimeError(f"Frozen alpha missing for {task}")
        value = json.loads(path.read_text())
        if value.get("data_fingerprint") != fingerprint or value.get("sweep_max_epochs") != 8:
            raise RuntimeError(f"Frozen alpha/data recipe differs for {task}")


def _available_memory_mb():
    lines = Path("/proc/meminfo").read_text().splitlines()
    return next(int(line.split()[1]) / 1024 for line in lines if line.startswith("MemAvailable:"))


def run_pipeline(root, python=sys.executable, reuse_alpha=False, min_available_ram_mb=3072):
    root = Path(root).resolve()
    directory = root / "results/full"
    logs = directory / "logs"
    locks = directory / ".locks"
    logs.mkdir(parents=True, exist_ok=True)
    locks.mkdir(parents=True, exist_ok=True)
    state_path = directory / "parallel_execution.json"
    state = {"status": "running", "started_at": _now(), "seeds": list(range(42, 47)), "threads_per_worker": 1,
             "reuse_alpha": reuse_alpha, "invocations": []}
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MALLOC_ARENA_MAX"):
        env[key] = "1"
    env["PYTHONHASHSEED"] = "42"
    env["MPLCONFIGDIR"] = "/tmp/exlightnet-mpl"
    env["XDG_CACHE_HOME"] = "/tmp/exlightnet-cache"
    base = [str(python), "-u", "-m", "exlightnet.run_all", "--mode", "full", "--root", str(root)]
    start = time.perf_counter()
    children = []
    streams = []
    def launch(stage, flags, logfile, seed=None):
        command = base + flags
        stream = (logs / logfile).open("a")
        streams.append(stream)
        process = subprocess.Popen(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        record = {"stage": stage, "seed": seed, "pid": process.pid, "command": command, "started_at": _now(), "log": str(logs / logfile)}
        state["invocations"].append(record)
        _write(state_path, state)
        return process, record
    def finish(process, record):
        record.update(returncode=process.wait(), finished_at=_now())
        _write(state_path, state)
    with (locks / "pipeline.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another parallel pipeline is already running") from exc
        try:
            if not reuse_alpha:
                process, record = launch("alpha_only", ["--alpha-only", "--skip-benchmark"], "alpha_sweep.log")
                finish(process, record)
                if record["returncode"]:
                    raise RuntimeError("Alpha sweep process failed; see its log")
            _validate_alpha(root)
            memory_start = time.perf_counter()
            available = _available_memory_mb()
            while available < min_available_ram_mb:
                print(f"Waiting for RAM: {available:.0f} MiB available; {min_available_ram_mb} MiB required before five workers", flush=True)
                state.update(waiting_for="memory", available_ram_mb=available, required_ram_mb=min_available_ram_mb)
                _write(state_path, state)
                if time.perf_counter() - memory_start >= 3600:
                    raise RuntimeError("Insufficient available RAM for five workers after one hour; no seed workers launched")
                time.sleep(10)
                available = _available_memory_mb()
            state.pop("waiting_for", None)
            state["worker_launch_memory"] = {"available_mb": available, "minimum_mb": min_available_ram_mb, "wait_seconds": time.perf_counter() - memory_start}
            workers = [launch("seed_worker", ["--seed-worker", "--seeds", str(seed), "--skip-benchmark"] + ([] if seed == 42 else ["--no-explanations"]), f"seed_{seed}.log", seed) for seed in range(42, 47)]
            print("Started single-thread seed processes: " + ", ".join(f"{r['seed']}={p.pid}" for p, r in workers), flush=True)
            next_update = 0.
            while any(p.poll() is None for p, _ in workers):
                if time.perf_counter() >= next_update:
                    status = []
                    for process, record in workers:
                        completed = len(list(directory.glob(f"*/*/{record['seed']}/metrics.json")))
                        status.append(f"{record['seed']}:{'running' if process.poll() is None else 'finished'} ({completed}/19 results)")
                    print("Seed progress: " + "; ".join(status), flush=True)
                    state["progress_at"] = _now()
                    _write(state_path, state)
                    next_update = time.perf_counter() + 30
                time.sleep(1)
            for process, record in workers:
                finish(process, record)
            print("Workers finished; aggregating cached results", flush=True)
            process, record = launch("aggregate_only", ["--aggregate-only", "--skip-benchmark"], "aggregate.log")
            finish(process, record)
            if record["returncode"]:
                raise RuntimeError("Cache-only aggregation failed; see its log")
            print("Aggregation finished; starting one idle-CPU benchmark process", flush=True)
            process, record = launch("benchmark_only", ["--benchmark-only"], "benchmark.log")
            finish(process, record)
            if record["returncode"]:
                raise RuntimeError("Benchmark process failed; see its log")
            manifest = json.loads((directory / "manifest.json").read_text())
            benchmark_manifest = json.loads((directory / "benchmark_manifest.json").read_text())
            stopped = manifest.get("failures", []) or benchmark_manifest.get("failures", [])
            complete = manifest.get("full_five_seed_headlines_complete", False) and all(r["returncode"] == 0 for r in state["invocations"])
            state["status"] = "complete_with_stopped_branches" if complete and stopped else "complete" if complete else "partial"
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", reason=f"{type(exc).__name__}: {exc}")
            for process in children:
                if process.poll() is None:
                    process.terminate()
            for process in children:
                if process.poll() is None:
                    process.wait()
            raise
        finally:
            state.update(finished_at=_now(), seconds=time.perf_counter() - start)
            _write(state_path, state)
            for stream in streams:
                stream.close()
    print(json.dumps({"status": state["status"], "seconds": state["seconds"]}), flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python", default=sys.executable, help="Python interpreter with the pinned CPU dependencies")
    parser.add_argument("--reuse-alpha", action="store_true", help="require an existing complete full alpha sweep instead of running one")
    parser.add_argument("--min-available-ram-mb", type=int, default=3072, help="wait for this RAM headroom before five workers; default 3072 MiB")
    args = parser.parse_args()
    result = run_pipeline(args.root, args.python, args.reuse_alpha, args.min_available_ram_mb)
    if result["status"] not in ("complete", "complete_with_stopped_branches"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
