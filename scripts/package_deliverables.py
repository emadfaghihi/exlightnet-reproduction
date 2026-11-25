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
            or relative.name in {'cache_arrays.npz', 'package_verification.log', 'epoch_checkpoint.pt', 'source_review.txt', 'review_package.txt'}
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


def package_hosted_handoff(root, output, verification):
    """Bundle the requested trio with supplied, per-artifact validation evidence."""
    root, output = Path(root), Path(output)
    names = ('notebooks/kaggle_runner.ipynb', 'prepared/manifest.json',
             'delivery/stats_gpu_v2/hosted_source.zip')
    files = {}
    for name in names:
        status = verification.get(name, {})
        if any(not isinstance(status.get(key), str) or not status[key].strip()
               for key in ('tested', 'not_tested')):
            raise ValueError(f'Missing individual verification statements: {name}')
        source = root / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Missing hosted handoff artifact: {name}')
        files[name] = {'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'bytes': source.stat().st_size,
                       'tested': status['tested'], 'not_tested': status['not_tested']}
    output.mkdir(parents=True, exist_ok=True)
    receipt, description = output / 'HOSTED_HANDOFF.json', output / 'HOSTED_HANDOFF.md'
    receipt.write_text(json.dumps({'files': files}, indent=2) + '\n')
    lines = ['# Hosted handoff', '',
             'Validation below does not imply Kaggle or CUDA execution.', '']
    for name, record in files.items():
        lines.extend([f'## {name}', '', f'SHA256: `{record["sha256"]}`', '',
                      'TESTED: ' + record['tested'], '', 'NOT TESTED: ' + record['not_tested'], ''])
    description.write_text('\n'.join(lines))
    bundle = output / 'hosted_handoff.zip'
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in names:
            archive.write(root / name, name)
        for path in (receipt, description):
            archive.write(path, path.name)
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            raise ValueError('Invalid hosted handoff archive')
        for name, record in files.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != record['sha256']:
                raise ValueError(f'Hosted handoff checksum mismatch: {name}')
    return bundle, receipt, description


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
    parser.add_argument('--hosted-verification', type=Path,
                        help='JSON of individual tested/not_tested statements for the hosted handoff trio')
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
    handoff = (package_hosted_handoff(ROOT, ROOT / 'delivery/stats_gpu_v2',
                                    json.loads(args.hosted_verification.read_text()))
               if args.hosted_verification else ())
    checksums = write_checksums(ROOT, [*paths, path, ROOT / 'deliverables_manifest.json', *components.values(), *handoff],
                               ROOT / 'delivery/stats_gpu_v2/SHA256SUMS')
    print(json.dumps({"archive": str(path), "files": len(paths) + 1, "bytes": path.stat().st_size, "sha256": sha256(path),
                      "notebook_code_cells": cells, "components": {k: str(v) for k,v in components.items()},
                      "hosted_handoff": [str(p) for p in handoff], "checksums": str(checksums)}))


if __name__ == "__main__":
    main()
