#!/usr/bin/env python3
"""Create a verified, dataset-free archive after notebook/test execution."""
import hashlib
import argparse
import json
import zipfile
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def excluded(relative):
    relative = Path(relative)
    return (any(part in {'__pycache__', '.locks', '.aws', '.codex', '.agents', '.venv', 'cache', 'prepared'} for part in relative.parts)
            or any(part.endswith('.egg-info') or part == '.env' or part.startswith('.env.') for part in relative.parts)
            or relative.suffix in {'.pyc', '.pkl', '.zip', '.lock', '.tmp', '.partial'}
            or relative.name in {'cache_arrays.npz', 'package_verification.log', 'epoch_checkpoint.pt', 'source_review.txt'}
            or (relative.suffix == '.npz' and relative.name.startswith('split_') and relative.name != 'split_indices.npz'))


def collect_files(root):
    root = Path(root)
    files = []
    for directory in ("src", "tests", "notebooks", "scripts", "results", "docs", "artifacts"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if not path.is_file() or path.is_symlink() or excluded(relative):
                continue
            files.append(path)
    for name in ("REPORT.md", "REPORT_stats_v2.md", "AGENTS.md", "README.md", "CONTRIBUTING.md", "requirements.txt", "requirements-package.txt", "requirements-lock.txt", "pyproject.toml", ".gitignore", ".gitattributes", ".gitmessage", "prepared/manifest.json"):
        if (root / name).is_file():
            files.append(root / name)
    return sorted(set(files))


def package_components(root, paths, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups = {name + '.zip': [p for p in paths if p.relative_to(root).parts[0] == name]
              for name in ('src', 'tests', 'scripts', 'docs')}
    groups['hosted_source.zip'] = [p for p in paths if p.relative_to(root).parts[0] in ('src', 'scripts')
                                   or p.name in ('pyproject.toml', 'requirements-package.txt', 'README.md')]
    bundles = {}
    for name, selected in groups.items():
        path = output / name
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for source in selected:
                archive.write(source, source.relative_to(root).as_posix())
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError(f'Invalid component archive: {path}')
            for source in selected:
                if archive.read(source.relative_to(root).as_posix()) != source.read_bytes():
                    raise ValueError(f'Component archive content differs: {source}')
        bundles[name] = path
    return bundles


def write_checksums(root, paths, output):
    root, output = Path(root), Path(output)
    entries = [(p.relative_to(root).as_posix(), hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(paths))]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(f'{digest}  {name}\n' for name, digest in entries))
    return output


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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, help='current verification directory for generated report')
    args = parser.parse_args()
    cells = validate_notebook(ROOT / "notebooks/ExLightNet_reproduction.ipynb")
    build_report(ROOT, evidence_dir=args.evidence_dir)
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
    components = package_components(ROOT, paths, ROOT / 'delivery/stats_gpu_v2')
    checksums = write_checksums(ROOT, [*paths, path, ROOT / 'deliverables_manifest.json', *components.values()],
                               ROOT / 'delivery/stats_gpu_v2/SHA256SUMS')
    print(json.dumps({"archive": str(path), "files": len(paths) + 1, "bytes": path.stat().st_size, "sha256": sha256(path),
                      "notebook_code_cells": cells, "components": {k: str(v) for k,v in components.items()}, "checksums": str(checksums)}))


if __name__ == "__main__":
    main()
