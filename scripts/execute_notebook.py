#!/usr/bin/env python3
"""Execute a cached analysis notebook; write verification only to a new evidence directory."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import nbformat
from nbclient import NotebookClient


def _evidence_directory(root, evidence_dir):
    root, directory = Path(root).resolve(), Path(evidence_dir).resolve()
    protected = root / "results"
    if directory == protected or protected in directory.parents:
        raise ValueError("New verification cannot be written into historical results")
    if (directory / "notebook_execution.json").exists() or (directory / "notebook_execution_history.json").exists():
        raise FileExistsError(f"Preserve prior verification; choose a new evidence directory: {directory}")
    return directory


def write_execution_evidence(root, evidence_dir, evidence):
    """Persist one invocation without replacing existing metadata or old history."""
    directory = _evidence_directory(root, evidence_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (("notebook_execution.json", evidence), ("notebook_execution_history.json", [evidence])):
        with (directory / name).open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return directory / "notebook_execution.json"


def execute_notebook(root, notebook_path, evidence_dir):
    root, path = Path(root).resolve(), Path(notebook_path).resolve()
    directory = _evidence_directory(root, evidence_dir)
    mode = os.environ.get("EXLIGHTNET_RUN_MODE", "full")
    if mode != "full":
        raise ValueError("Verification of this cached analysis notebook requires EXLIGHTNET_RUN_MODE=full")
    from exlightnet.publication import validate_notebook_content
    nb = nbformat.read(path, as_version=4)
    validate_notebook_content(nb, mode)
    cached_at_start = (root / "results" / mode / "manifest.json").exists()
    if not cached_at_start:
        raise FileNotFoundError("Full cached campaign manifest is required before notebook execution")
    start = time.perf_counter()
    error = None
    original_environment = os.environ.copy()
    try:
        with tempfile.TemporaryDirectory(prefix="exlightnet-jupyter-") as temporary:
            temporary = Path(temporary)
            kernel = temporary / "kernels/exlightnet-local"
            kernel.mkdir(parents=True)
            (kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "ExLightNet local", "language": "python"}))
            os.environ["JUPYTER_PATH"] = str(temporary)
            os.environ["JUPYTER_RUNTIME_DIR"] = str(temporary / "runtime")
            os.environ["IPYTHONDIR"] = str(temporary / "ipython")
            os.environ["MPLCONFIGDIR"] = "/tmp/exlightnet-mpl"
            os.environ["EXLIGHTNET_RUN_MODE"] = mode
            for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                os.environ[variable] = "1"
            client = NotebookClient(nb, timeout=1200, kernel_name="exlightnet-local", resources={"metadata": {"path": str(root)}})
            client.execute()
            validate_notebook_content(nb, mode)
    except BaseException as exc:
        error = exc
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
        nb.metadata["kernelspec"] = {"display_name": "Python 3 (ExLightNet)", "language": "python", "name": "python3"}
        nbformat.write(nb, path)
        evidence = {"seconds": time.perf_counter() - start, "mode": mode,
                    "code_cells": sum(c.cell_type == "code" for c in nb.cells),
                    "cached_results_present_at_start": cached_at_start,
                    "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in nb.cells),
                    "error_outputs": sum(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs),
                    "status": "failed" if error is not None else "complete", "notebook": str(path),
                    "recorded_at": datetime.now(timezone.utc).isoformat()}
        if error is not None:
            evidence["execution_error"] = f"{type(error).__name__}: {error}"
        record = write_execution_evidence(root, directory, evidence)
        print(json.dumps({**evidence, "evidence_path": str(record)}), flush=True)
    if error is not None:
        raise error
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--notebook", type=Path, help="default ROOT/notebooks/ExLightNet_reproduction.ipynb")
    parser.add_argument("--evidence-dir", type=Path, help="new verification directory; defaults to a dated directory under artifacts/stats_cpu_v1/verification")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.evidence_dir or args.root / "artifacts/stats_cpu_v1/verification" / stamp
    execute_notebook(args.root, args.notebook or args.root / "notebooks/ExLightNet_reproduction.ipynb", directory)


if __name__ == "__main__":
    main()
