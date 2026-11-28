#!/usr/bin/env python3
"""Export the local, train-fitted feature_sets_v1 prepared inputs."""

import argparse
import json
import logging
import os
from pathlib import Path

for _thread_key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_key] = "1"

from exlightnet.feature_sets import export_feature_set_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rf-jobs", type=int, default=1, help="CPU workers per Random Forest (default 1)")
    args = parser.parse_args()
    if args.rf_jobs < 1:
        parser.error("--rf-jobs must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        receipt = export_feature_set_inputs(args.root, rf_n_jobs=args.rf_jobs)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Export stopped. {exc}\n")
    print(json.dumps({key: receipt[key] for key in ("status", "npz", "wall_seconds", "stopped_arms")}))


if __name__ == "__main__":
    main()
