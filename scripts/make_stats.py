"""Generate CPU statistics exclusively from immutable canonical caches."""
import argparse
from pathlib import Path

from exlightnet.stats import write_cpu_statistics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--output",type=Path,default=Path("artifacts/stats_cpu_v1"))
    parser.add_argument("--replicates",type=int,default=2000)
    parser.add_argument("--random-seed",type=int,default=42)
    args=parser.parse_args()
    paths=write_cpu_statistics(args.root,args.output,args.replicates,args.random_seed)
    for name,path in paths.items():
        print(f"{name}: {path}")


if __name__=="__main__":
    main()
