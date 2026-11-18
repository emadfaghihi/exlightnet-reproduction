#!/usr/bin/env python3
"""Verify cached analysis delivery gates without training or inference."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import nbformat


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def check_gates(root, pytest_xml, notebook_evidence, output):
    from exlightnet.publication import active_scope_legacy_flags, load_statistics, validate_notebook_content, validate_publication_text
    root, output = Path(root).resolve(), Path(output).resolve()
    read = lambda name: json.loads((root / name).read_text())
    baseline = read('artifacts/stats_cpu_v1/immutable_before.json')
    changed = [name for name, expected in baseline['files'].items() if not (root / name).is_file() or sha256(root / name) != expected]
    tree = ET.parse(pytest_xml).getroot()
    cases = list(tree.iter('testcase'))
    def passed(name):
        matches = [case for case in cases if case.get('name', '').split('[')[0] == name]
        return bool(matches) and all(not any(case.find(tag) is not None for tag in ('failure', 'error', 'skipped')) for case in matches)
    suite_pass = bool(cases) and not any(case.find(tag) is not None for case in cases for tag in ('failure', 'error'))
    tables = load_statistics(root)
    report = (root / 'REPORT.md').read_text()
    validate_publication_text(report)
    notebook = nbformat.read(root / 'notebooks/ExLightNet_reproduction.ipynb', as_version=4)
    validate_notebook_content(notebook, 'full')
    notebook_record = json.loads(Path(notebook_evidence).read_text())
    flags = active_scope_legacy_flags(root)
    compatibility = read('artifacts/stats_cpu_v1/cpu_default_compatibility.json')
    prepared = read('artifacts/stats_cpu_v1/prepared_export_checks.json')
    smoke = read('artifacts/stats_cpu_v1/smoke/control.json')
    smoke_import_path = 'artifacts/stats_gpu_v2_smoke_verified/aggregation_manifest.json'
    smoke_import = read(smoke_import_path)
    protocol = read('docs/statistics_protocol.json')
    pending_report = (root / 'REPORT_stats_v2.md').read_text()
    gates = []
    def gate(stage, name, ok, evidence):
        gates.append({'stage': stage, 'gate': name, 'status': 'PASS' if ok else 'FAIL', 'evidence': evidence})
    gate('A', 'Immutable evidence', not changed, {'files_checked': len(baseline['files']), 'changed': changed})
    gate('A', 'Paired t/CI agrees with SciPy', passed('test_paired_summary_matches_scipy'), str(pytest_xml))
    gate('A', 'No active legacy SD flags', not flags, {'violations': flags, 'excluded': ['results/', 'archive/', 'authoritative brief/instruction docs']})
    gate('A', 'Mode captions and status recovery', passed('test_mode_validation_rejects_stale_captions_and_rendered_outputs') and passed('test_effective_status_explains_archive_and_recovery_without_mutating'), str(pytest_xml))
    gate('A', 'Generated statistical values trace to checksummed files', len(tables) == 10, {'tables_checked': list(tables), 'manifest': 'artifacts/stats_cpu_v1/statistics_manifest.json', 'method': 'source/table SHA and rendering tests; no freehand new metric values'})
    gate('A', 'Executed cached notebook', notebook_record.get('status') == 'complete' and notebook_record.get('error_outputs') == 0 and notebook_record.get('code_cells') == notebook_record.get('executed_code_cells'), str(notebook_evidence))
    gate('A', 'Full pytest suite', suite_pass, {'tests': len(cases), 'xml': str(pytest_xml)})
    gate('B', 'Canonical CPU default compatibility', compatibility['cache_hit_row_byte_equivalent'] and all(compatibility['canonical_cpu_definitions_unchanged'].values()), {'method': compatibility['method'], 'fingerprint': compatibility['fingerprint'], 'macro_f1': compatibility['macro_f1'], 'retrained': compatibility['retrained_seed42']})
    gate('B', 'Sampler-frequency equivalence', passed('test_sampling_matches_weighted_random_sampler'), str(pytest_xml))
    canonical_fp = read('artifacts/cache_manifest.json')['fingerprint']
    export_manifest = read('prepared/manifest.json')
    gate('B', 'Prepared split42 matches canonical fingerprint', export_manifest['splits']['42']['data_fingerprint'] == canonical_fp and passed('test_real_split42_export_preserves_frozen_cache_without_fitting'), {'fingerprint': canonical_fp, 'evidence': 'artifacts/stats_cpu_v1/prepared_export_checks.json'})
    gate('B', 'CPU smoke, ZIP restore, idempotent resume and aggregation', smoke['status'] == 'complete' and smoke['completed_runs'] == 2 and smoke['completed_epochs'] == 4 and smoke['zip_restore_verified'] and smoke['no_duplicate_runs'] and smoke['receipts_unchanged_on_restart'] and smoke_import['n_unique_runs'] == 2, {'seconds': smoke['seconds'], 'evidence': 'artifacts/stats_cpu_v1/smoke/control.json', 'aggregate': smoke_import_path})
    gate('B', 'Mixed campaigns rejected; within-campaign device exception explicit', passed('test_mixed_campaign_and_conflicting_identity_rejected') and passed('test_gpu_v2_only_neural_cuda_tree_cpu_device_exception') and passed('test_mixed_gpu_neural_cpu_tree_label_and_same_model_device_error'), str(pytest_xml))
    gate('B', 'Real studies remain PENDING; protocol verbatim', pending_report.count('PENDING: study not run') == len(protocol['studies']) and all(rule in pending_report for rule in protocol['rules_verbatim']), 'REPORT_stats_v2.md')
    gate('B', 'Full pytest suite', suite_pass, {'tests': len(cases), 'xml': str(pytest_xml)})
    synthetic = []
    for case in cases:
        properties = {item.get('name'): item.get('value') for item in case.findall('./properties/property')}
        if properties.get('synthetic_training') == 'true':
            synthetic.append({'test': case.get('name'), 'pytest_seconds': float(case.get('time')), **properties})
    output.mkdir(parents=True, exist_ok=True)
    result = {'verified_at': datetime.now(timezone.utc).isoformat(), 'gates': gates,
              'synthetic_unit_tests': {'cases': len(synthetic), 'fits': sum(int(r['fits']) for r in synthetic),
                                       'epochs': sum(int(r['epochs']) for r in synthetic), 'pytest_seconds': sum(r['pytest_seconds'] for r in synthetic),
                                       'scope': 'this full-suite invocation; earlier focused checks retained separately', 'details': synthetic},
              'real_smoke': {'runs': smoke['completed_runs'], 'epochs': smoke['completed_epochs'], 'seconds': smoke['seconds']},
              'not_run': ['CUDA execution', 'real gpu_v2 studies B1-B5', 'GPU probe/determinism', 'all benchmark reruns', 'supplementary frozen-model latency'],
              'immutable_changes': changed}
    (output / 'gates.json').write_text(json.dumps(result, indent=2) + '\n')
    with (output / 'gates.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['stage', 'gate', 'status', 'evidence'])
        writer.writeheader()
        writer.writerows({**row, 'evidence': json.dumps(row['evidence'], ensure_ascii=False)} for row in gates)
    lines = ['# Cached statistics and hosted-runner verification', '', '| Stage | Gate | Status | Evidence |', '| --- | --- | --- | --- |']
    for row in gates:
        evidence = json.dumps(row['evidence'], ensure_ascii=False).replace('|', '\\|')
        lines.append(f'| {row["stage"]} | {row["gate"]} | {row["status"]} | {evidence} |')
    lines += ['', 'Synthetic unit training and real-data smoke are separate:', '', '| Type | Test cases/runs | Fits | Epochs | Seconds |', '| --- | --- | --- | --- | --- |',
              f'| Synthetic unit tests, current full suite | {len(synthetic)} | {sum(int(r["fits"]) for r in synthetic)} | {sum(int(r["epochs"]) for r in synthetic)} | {sum(r["pytest_seconds"] for r in synthetic)} |',
              f'| Real-data CPU smoke, including interruption/ZIP recovery/cache restart | {smoke["completed_runs"]} | {smoke["completed_runs"]} | {smoke["completed_epochs"]} | {smoke["seconds"]} |',
              '', 'Synthetic time is pytest case wall time, including fixture/setup overhead. Smoke time is supervisor end-to-end wall time. They are not comparable per-epoch throughput estimates.',
              '', 'NOT RUN: ' + '; '.join(result['not_run']) + '.',
              '', 'Canonical CPU compatibility is an unchanged source-definition check and validated cache hit. The historical trained MLP was not retrained beyond the local smoke budget.']
    (output / 'GATES.md').write_text('\n'.join(lines) + '\n')
    for row in gates:
        print(f'{row["stage"]}\t{row["status"]}\t{row["gate"]}')
    return all(row['status'] == 'PASS' for row in gates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--pytest-xml', type=Path, required=True)
    parser.add_argument('--notebook-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/stats_cpu_v1/verification'))
    args = parser.parse_args()
    if not check_gates(args.root, args.pytest_xml, args.notebook_evidence, args.output):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
