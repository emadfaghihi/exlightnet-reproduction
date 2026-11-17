import importlib.util
from pathlib import Path

import nbformat
import pytest


def load_script():
    spec = importlib.util.spec_from_file_location("pack", Path("scripts/package_deliverables.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archive_collection_excludes_dataset_and_caches(tmp_path):
    for name in ("src/exlightnet/model.py", "data/RT_IOT2022.csv", "artifacts/cache_arrays.npz", "artifacts/preprocessor.json", ".venv/secret.py", "results/fast/results.json", "results/fast/binary/random_forest/42/model.pkl"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test fixture")
    files = {str(p.relative_to(tmp_path)) for p in load_script().collect_files(tmp_path)}
    assert "src/exlightnet/model.py" in files
    assert "artifacts/preprocessor.json" in files
    assert "results/fast/results.json" in files
    assert "artifacts/cache_arrays.npz" not in files
    assert not any(f.startswith(("data/", ".venv/")) for f in files)
    assert not any(f.endswith(".pkl") for f in files)


def test_packaging_rejects_unexecuted_or_error_notebook(tmp_path):
    path = tmp_path / "test.ipynb"
    nb = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("1+1")])
    nbformat.write(nb, path)
    with pytest.raises(ValueError, match="unexecuted"):
        load_script().validate_notebook(path)
    nb.cells[0].execution_count = 1
    nb.cells[0].outputs = [nbformat.v4.new_output("error", ename="ValueError", evalue="fixture", traceback=[])]
    nbformat.write(nb, path)
    with pytest.raises(ValueError, match="error"):
        load_script().validate_notebook(path)
