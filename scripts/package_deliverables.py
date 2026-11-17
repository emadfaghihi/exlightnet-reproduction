#!/usr/bin/env python3
"""Create a verified, dataset-free archive after notebook/test execution."""
import hashlib
import json
import zipfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def collect_files(root):
    root = Path(root)
    files = []
    for directory in ("src", "tests", "notebooks", "scripts", "results", "docs", "artifacts"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or "__pycache__" in relative.parts or any(p.endswith(".egg-info") for p in relative.parts):
                continue
            if path.name in ("cache_arrays.npz", "package_verification.log") or "cache" in relative.parts or ".locks" in relative.parts or path.suffix in (".pyc", ".pkl") or path.name.endswith(".partial"):
                continue
            files.append(path)
    for name in ("REPORT.md", "README.md", "CONTRIBUTING.md", "requirements.txt", "requirements-package.txt", "requirements-lock.txt", "pyproject.toml", ".gitignore", ".gitattributes", ".gitmessage"):
        if (root / name).is_file():
            files.append(root / name)
    return sorted(set(files))


def validate_notebook(path):
    nb = nbformat.read(path, as_version=4)
    cells = [c for c in nb.cells if c.cell_type == "code"]
    if not cells or any(c.execution_count is None for c in cells):
        raise ValueError("Refusing to package an unexecuted notebook")
    if any(o.output_type == "error" for c in cells for o in c.outputs):
        raise ValueError("Refusing to package a notebook with error outputs")
    if not any(c.outputs for c in cells):
        raise ValueError("Notebook has no outputs")
    return len(cells)


def main():
    from exlightnet.io import sha256, write_json
    from exlightnet.report import build_report

    cells = validate_notebook(ROOT / "notebooks/ExLightNet_reproduction.ipynb")
    build_report(ROOT)
    paths = collect_files(ROOT)
    manifest = {"notebook_executed_cells": cells, "dataset_included": False,
                "files": {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}}
    write_json(ROOT / "deliverables_manifest.json", manifest)
    path = ROOT / "deliverables.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for source in paths:
            archive.write(source, source.relative_to(ROOT).as_posix())
        archive.write(ROOT / "deliverables_manifest.json", "deliverables_manifest.json")
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert not any(name.startswith(("data/", ".venv/")) or "cache_arrays.npz" in name or name.endswith(".pkl") for name in archive.namelist())
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    print(json.dumps({"archive": str(path), "files": len(paths) + 1, "bytes": path.stat().st_size, "sha256": sha256(path), "notebook_code_cells": cells}))


if __name__ == "__main__":
    main()
