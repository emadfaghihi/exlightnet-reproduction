"""Changing tree scheduling must not repeat a fitted or claimed tree run."""
import json
import pickle

import pytest
import torch

from exlightnet import baselines, campaign, device
from test_plan_amendment import (prepared_b1, baseline_identity, baseline_fingerprint,
                                _cpu_versions, _forbid_cuda, _write_completed)


@pytest.fixture
def one_tree(monkeypatch):
    plan = next(row for row in campaign.study_runs('B1') if row['model'] == 'random_forest')
    monkeypatch.setattr(campaign, 'study_runs', lambda study: [dict(plan)])
    return plan


def test_new_tree_fingerprint_matches_all_cuda_and_trees_only_sessions(
    prepared_b1, one_tree, tmp_path, monkeypatch
):
    prepared, metadata, sha = prepared_b1
    output = tmp_path / 'out'
    root = output / 'gpu_v2'
    monkeypatch.setattr(device, 'configure_device', lambda requested='cpu': requested)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'synthetic GPU name')
    def stop(*args):
        raise campaign.RunStopped('Stop before any fit')
    monkeypatch.setattr(baselines, 'build_model', stop)
    campaign.run_campaign(prepared, output, 'B1', only='all', device='cuda')
    expected = baseline_fingerprint(baseline_identity(one_tree, 'cpu', _cpu_versions(), metadata, sha))
    assert {p.name for p in (root / 'cpu').iterdir()} == {expected}
    _forbid_cuda(monkeypatch)
    campaign.run_campaign(prepared, output, 'B1', only='trees', device='auto')
    assert {p.name for p in (root / 'cpu').iterdir()} == {expected}


def test_tree_resume_retains_opaque_legacy_cuda_session_fingerprint(
    prepared_b1, one_tree, tmp_path, monkeypatch
):
    prepared, metadata, sha = prepared_b1
    output = tmp_path / 'out'
    root = output / 'gpu_v2'
    versions = {**_cpu_versions(), 'gpu_name': 'legacy GPU name'}
    _write_completed(root, [one_tree], metadata, sha, versions)
    before = {p: p.read_bytes() for p in root.rglob('record.json')}
    _forbid_cuda(monkeypatch)
    monkeypatch.setattr(baselines, 'build_model', lambda *args: pytest.fail('Duplicate model creation'))
    result = campaign.run_campaign(prepared, output, 'B1', only='trees', device='auto')
    assert result['new_completed'] == result['stopped_count'] == 0
    assert before == {p: p.read_bytes() for p in root.rglob('record.json')}
    assert len(list((root / 'cpu').iterdir())) == 1


def pending_tree(root, plan, metadata, sha):
    versions = {**_cpu_versions(), 'gpu_name': 'legacy GPU name'}
    identity = baseline_identity(plan, 'cpu', versions, metadata, sha)
    fp = baseline_fingerprint(identity)
    directory = root / 'cpu' / fp
    directory.mkdir(parents=True)
    pending = {**{key: value for key, value in identity.items() if key != 'plan'}, **plan,
               'schema_version': 1, 'status': 'pending', 'run_fingerprint': fp,
               'requested_device': 'cpu', 'warnings': [{'message': 'preserved warning'}],
               'effective_settings': {'preserved': True}, 'best_epoch': None,
               'best_val_macro_f1': None, 'epochs_run': None, 'history': [], 'headline_checkpoint': 'best'}
    path = directory / 'run_identity.json'
    path.write_text(json.dumps(pending))
    return directory, pending


def test_pending_tree_resumes_after_session_changes_without_altering_provenance(
    prepared_b1, one_tree, tmp_path, monkeypatch
):
    prepared, metadata, sha = prepared_b1
    output = tmp_path / 'out'
    root = output / 'gpu_v2'
    directory, pending = pending_tree(root, one_tree, metadata, sha)
    before = (directory / 'run_identity.json').read_bytes()
    (directory / 'tree_in_progress.pkl').write_bytes(pickle.dumps({'synthetic checkpoint': True}))
    _forbid_cuda(monkeypatch)
    monkeypatch.setattr(baselines, 'build_model', lambda *args: object())
    monkeypatch.setattr(baselines, 'fit_tree', lambda *args: pytest.fail('Repeated fit'))
    monkeypatch.setattr(campaign, '_evaluate', lambda path, *args: {
        key + '_path': str(path / 'best' / (key + '.fixture')) for key in ('prediction', 'metrics', 'receipt')})
    result = campaign.run_campaign(prepared, output, 'B1', only='trees', device='auto')
    assert result['new_completed'] == 1 and result['stopped_count'] == 0
    assert (directory / 'run_identity.json').read_bytes() == before
    record = json.loads((directory / 'record.json').read_text())
    for key in ('requested_device', 'warnings', 'effective_settings', 'run_fingerprint', 'libraries'):
        assert record[key] == pending[key]
    assert len(list((root / 'cpu').iterdir())) == 1


def test_legacy_claimed_tree_blocks_new_fingerprint_and_evaluation(
    prepared_b1, one_tree, tmp_path, monkeypatch
):
    prepared, metadata, sha = prepared_b1
    output = tmp_path / 'out'
    root = output / 'gpu_v2'
    directory, _ = pending_tree(root, one_tree, metadata, sha)
    (directory / 'best').mkdir()
    receipt = directory / 'best/receipt.json'
    receipt.write_text('{"status":"claimed"}')
    _forbid_cuda(monkeypatch)
    monkeypatch.setattr(baselines, 'build_model', lambda *args: pytest.fail('Repeated claimed run'))
    result = campaign.run_campaign(prepared, output, 'B1', only='trees')
    assert result['new_completed'] == 0 and result['stopped_count'] == 1
    assert 'incomplete evaluation receipt' in result['stopped'][0]['reason'].lower()
    assert receipt.read_text() == '{"status":"claimed"}'
    assert len(list((root / 'cpu').iterdir())) == 1


def test_ambiguous_legacy_tree_candidates_stop_without_a_fit(
    prepared_b1, one_tree, tmp_path, monkeypatch
):
    prepared, metadata, sha = prepared_b1
    output = tmp_path / 'out'
    root = output / 'gpu_v2'
    _write_completed(root, [one_tree], metadata, sha, _cpu_versions())
    _write_completed(root, [one_tree], metadata, sha, {**_cpu_versions(), 'gpu_name': 'legacy GPU'})
    _forbid_cuda(monkeypatch)
    monkeypatch.setattr(baselines, 'build_model', lambda *args: pytest.fail('Ambiguous fit'))
    result = campaign.run_campaign(prepared, output, 'B1', only='trees')
    assert result['stopped_count'] == 1
    assert 'ambiguous' in result['stopped'][0]['reason'].lower()
