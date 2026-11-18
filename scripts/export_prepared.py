#!/usr/bin/env python3
"""Export train-fitted NPZ arrays locally; never upload the raw RT-IoT CSV."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

# Set limits before importing NumPy or local preprocessing dependencies.
for _thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"

from exlightnet.prepared import export_prepared


def _split_seeds(value):
    try:
        if "-" in value:
            start, end = map(int, value.split("-"))
            if start < 0 or end < start:
                raise ValueError
            seeds = list(range(start, end + 1))
        else:
            seeds = [int(part) for part in value.split(",")]
        if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 or seed >= 2**32 for seed in seeds):
            raise ValueError
        return seeds
    except ValueError as exc:
        raise argparse.ArgumentTypeError("split range must be ascending (42-51) or distinct seeds (42,43)") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None, help="new directory; default ROOT/prepared")
    parser.add_argument("--splits", type=_split_seeds, default=_split_seeds("42-51"))
    parser.add_argument("--checks-output", type=Path, default=None,
                        help="new evidence JSON; default ROOT/artifacts/stats_cpu_v1/prepared_export_checks.json")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "prepared"
    checks_path = args.checks_output or root / "artifacts/stats_cpu_v1/prepared_export_checks.json"
    if checks_path.exists():
        parser.error(f"checks output already exists; preserve it: {checks_path}")
    print(f"Exporting split seeds {args.splits} serially to {output}; split42 reuses frozen cache", flush=True)
    manifest = export_prepared(root, output, args.splits)
    checks = {"recorded_at": datetime.now(timezone.utc).isoformat(),
              "status": "passed", "source_campaign": "cpu_v1", "splits": args.splits,
              "threads_per_library": 1, "manifest_path": str(output / "manifest.json"),
              "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
              "checks": manifest, "training": "NOT RUN", "inference": "NOT RUN", "benchmarks": "NOT RUN"}
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    with checks_path.open("x") as stream:
        json.dump(checks, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "split_count": len(args.splits), "seconds": manifest["seconds"],
                      "checks_output": str(checks_path)}), flush=True)


if __name__ == "__main__":
    main()
