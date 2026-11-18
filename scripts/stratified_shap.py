"""Run the separate CPU stratified final analysis from cached cpu_v1 evidence."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, help="Separate output directory, relative to --root by default")
    args = parser.parse_args()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"
    from exlightnet.final_analysis import run_stratified_shap

    result = run_stratified_shap(args.root, args.output)
    print(f'Stratified final analysis {result["status"]}; input signature {result["input_signature"]}')


if __name__ == "__main__":
    main()
