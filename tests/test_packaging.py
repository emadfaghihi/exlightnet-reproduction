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


def test_delivery_excludes_new_data_secrets_locks_and_recursive_archives(tmp_path):
    included = ("REPORT_stats_v2.md", "AGENTS.md", "prepared/manifest.json", "scripts/export_prepared.py", "artifacts/stats_gpu_v2_smoke/per_run.csv")
    excluded = ("prepared/split_42.npz", "artifacts/nested/prepared/split_43.npz", "artifacts/stats_cpu_v1/smoke/controller.lock",
                "artifacts/stats_cpu_v1/smoke/epoch_checkpoint.pt", "artifacts/stats_cpu_v1/smoke/gpu_v2_smoke.zip",
                "artifacts/.env", "docs/.aws/credentials", "artifacts/private.tmp", "artifacts/stats_cpu_v1/review/source_review.txt")
    for name in included + excluded:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    files = {p.relative_to(tmp_path).as_posix() for p in load_script().collect_files(tmp_path)}
    assert set(included) <= files
    assert not (set(excluded) & files)


def test_component_bundles_and_checksum_file_are_verifiable(tmp_path):
    import hashlib
    import zipfile
    script = load_script()
    for name in ('src/exlightnet/campaign.py', 'tests/test_campaign.py', 'scripts/kaggle_campaign.py',
                 'docs/KAGGLE_RUNBOOK.md', 'pyproject.toml', 'requirements-package.txt', 'README.md'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
    paths = script.collect_files(tmp_path)
    bundles = script.package_components(tmp_path, paths, tmp_path / 'delivery')
    assert set(bundles) == {'src.zip', 'tests.zip', 'scripts.zip', 'docs.zip', 'hosted_source.zip'}
    with zipfile.ZipFile(bundles['hosted_source.zip']) as archive:
        assert 'src/exlightnet/campaign.py' in archive.namelist()
        assert 'scripts/kaggle_campaign.py' in archive.namelist()
        assert 'pyproject.toml' in archive.namelist()
        assert 'tests/test_campaign.py' not in archive.namelist()
        assert archive.testzip() is None
    checksum = script.write_checksums(tmp_path, [*paths, *bundles.values()], tmp_path / 'delivery/SHA256SUMS')
    for line in checksum.read_text().splitlines():
        expected, name = line.split('  ', 1)
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected


def test_hosted_handoff_contains_all_three_artifacts_and_individual_test_status(tmp_path):
    import hashlib
    import json
    import zipfile
    script = load_script()
    names = ('notebooks/kaggle_runner.ipynb', 'prepared/manifest.json',
             'delivery/stats_gpu_v2/hosted_source.zip')
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    verification = {name: {'tested': 'fixture checksum verification',
                           'not_tested': 'hosted execution and CUDA', 'bytes': -1, 'sha256': 'forged'} for name in names}
    output = tmp_path / 'delivery/stats_gpu_v2'
    bundle, receipt, description = script.package_hosted_handoff(tmp_path, output, verification)
    manifest = json.loads(receipt.read_text())
    assert set(manifest['files']) == set(names)
    with zipfile.ZipFile(bundle) as archive:
        assert archive.testzip() is None
        assert set(names) <= set(archive.namelist())
        for name in names:
            assert archive.read(name) == (tmp_path / name).read_bytes()
            record = manifest['files'][name]
            assert record['sha256'] == hashlib.sha256(archive.read(name)).hexdigest()
            assert record['bytes'] == len(archive.read(name))
            assert record['tested'] == verification[name]['tested']
            assert record['not_tested'] == verification[name]['not_tested']
            assert name in description.read_text()
        assert 'TESTED: fixture checksum verification' in description.read_text()
        assert 'NOT TESTED: hosted execution and CUDA' in description.read_text()
        assert archive.read(receipt.name) == receipt.read_bytes()
        assert archive.read(description.name) == description.read_bytes()


def test_hosted_handoff_refuses_missing_artifacts_or_unsubstantiated_status(tmp_path):
    script = load_script()
    with pytest.raises(ValueError, match='verification'):
        script.package_hosted_handoff(tmp_path, tmp_path / 'delivery', {})
