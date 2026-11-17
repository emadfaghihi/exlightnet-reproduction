#!/usr/bin/env python3
"""Execute with this Python interpreter and a temporary Jupyter kernel spec."""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "notebooks/ExLightNet_reproduction.ipynb"
nb = nbformat.read(path, as_version=4)
cached_at_start = (ROOT / "results" / os.environ.get("EXLIGHTNET_RUN_MODE", "fast") / "manifest.json").exists()
start = time.perf_counter()
with tempfile.TemporaryDirectory(prefix="exlightnet-jupyter-") as directory:
    directory = Path(directory)
    kernel = directory / "kernels/exlightnet-local"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text(json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "ExLightNet local", "language": "python"}))
    os.environ["JUPYTER_PATH"] = str(directory)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(directory / "runtime")
    os.environ["IPYTHONDIR"] = str(directory / "ipython")
    os.environ["MPLCONFIGDIR"] = "/tmp/exlightnet-mpl"
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"
    client = NotebookClient(nb, timeout=1200, kernel_name="exlightnet-local", resources={"metadata": {"path": str(ROOT)}})
    try:
        client.execute()
    finally:
        nb.metadata["kernelspec"] = {"display_name": "Python 3 (ExLightNet)", "language": "python", "name": "python3"}
        nbformat.write(nb, path)
duration = time.perf_counter() - start
evidence = {"seconds": duration, "mode": os.environ.get("EXLIGHTNET_RUN_MODE", "fast"), "code_cells": sum(c.cell_type == "code" for c in nb.cells),
            "cached_results_present_at_start": cached_at_start,
            "executed_code_cells": sum(c.cell_type == "code" and c.execution_count is not None for c in nb.cells),
            "error_outputs": sum(o.output_type == "error" for c in nb.cells if c.cell_type == "code" for o in c.outputs)}
record = ROOT / "results/notebook_execution.json"
history_path = ROOT / "results/notebook_execution_history.json"
history = json.loads(history_path.read_text()) if history_path.exists() else [json.loads(record.read_text())] if record.exists() else []
history.append(evidence)
history_path.write_text(json.dumps(history, indent=2) + "\n")
record.write_text(json.dumps(evidence, indent=2) + "\n")
print(json.dumps(evidence))
