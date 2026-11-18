"""Verify downloaded supplementary result ZIPs and aggregate locally on CPU."""
import argparse
from pathlib import Path

from exlightnet.study_stats import aggregate_studies


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives",nargs="*",type=Path)
    parser.add_argument("--prepared-dir",type=Path,default=Path("prepared"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/stats_gpu_v2"))
    parser.add_argument("--report",type=Path,default=Path("REPORT_stats_v2.md"))
    parser.add_argument("--reference-cpu",type=Path,default=Path("results/full/results.json"))
    args=parser.parse_args()
    paths=aggregate_studies(args.archives,args.prepared_dir,args.output,args.report,args.reference_cpu)
    for name,path in paths.items():
        print(f"{name}: {path}")


if __name__=="__main__":
    main()
