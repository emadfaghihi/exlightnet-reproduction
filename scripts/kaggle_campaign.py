#!/usr/bin/env python3
"""Run an offline supplementary campaign from prepared arrays."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path
import threading

from exlightnet.campaign import RUN_SELECTIONS, restore_campaign_archive, run_campaign


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--study", choices=("B1", "B2", "B3", "B4", "B5", "probe", "determinism"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--only", choices=RUN_SELECTIONS, default="all",
                        help="select the model family (default: %(default)s)")
    parser.add_argument("--campaign", default="gpu_v2")
    parser.add_argument("--time-budget-hours", type=float, default=8)
    parser.add_argument("--zip-every", type=int, default=5)
    parser.add_argument("--smoke-model")
    parser.add_argument("--smoke-task", choices=("binary", "multiclass"))
    parser.add_argument("--smoke-seeds", type=int, nargs=2)
    parser.add_argument("--smoke-epochs", type=int)
    parser.add_argument("--pause-after-first-checkpoint", action="store_true")
    parser.add_argument("--resume-zip", type=Path, action="append", default=[])
    args = parser.parse_args()
    for archive in args.resume_zip:
        restore_campaign_archive(archive, args.output, args.campaign)
    def pause(epoch, path):
        if epoch == 1:
            print(f"PAUSED_AFTER_ATOMIC_CHECKPOINT {path}", flush=True)
            threading.Event().wait()
    result = run_campaign(args.prepared_dir, args.output, args.study, device=args.device, campaign=args.campaign,
                          time_budget_hours=args.time_budget_hours, zip_every=args.zip_every, only=args.only,
                          smoke_model=args.smoke_model, smoke_task=args.smoke_task, smoke_seeds=args.smoke_seeds,
                          smoke_epochs=args.smoke_epochs, checkpoint_hook=pause if args.pause_after_first_checkpoint else None)
    print(result, flush=True)
    if result["stopped_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
