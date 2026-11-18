import json
from pathlib import Path

import pandas as pd
import pytest

from exlightnet.io import sha256
from exlightnet.final_presentation import FINAL_ANALYSIS_PATH, load_final_analysis
from exlightnet.final_analysis import OUTPUT_FILES


def completed_manifest(directory):
    for name in OUTPUT_FILES:
        (directory / name).write_text('fixture\n1\n' if name.endswith('.csv') else 'fixture')
    return {'status': 'complete', 'label': 'final analysis; no selection use',
            'output_sha256': {name: sha256(directory / name) for name in OUTPUT_FILES}}


def test_final_presentation_is_optional_until_completion(tmp_path):
    assert load_final_analysis(tmp_path) is None
    directory = tmp_path / FINAL_ANALYSIS_PATH
    directory.mkdir(parents=True)
    (directory / 'analysis.json').write_text(json.dumps({'status': 'in_progress'}))
    assert load_final_analysis(tmp_path) is None


def test_final_presentation_checks_saved_tables_without_model_queries(tmp_path):
    directory = tmp_path / FINAL_ANALYSIS_PATH
    directory.mkdir(parents=True)
    table = directory / 'balanced_global_ranking.csv'
    pd.DataFrame([{'rank': 1, 'feature': 'fixture', 'mean_abs_phi': .2}]).to_csv(table, index=False)
    manifest = completed_manifest(directory)
    pd.DataFrame([{'rank': 1, 'feature': 'fixture', 'mean_abs_phi': .2}]).to_csv(table, index=False)
    manifest['output_sha256'][table.name] = sha256(table)
    (directory / 'analysis.json').write_text(json.dumps(manifest))
    actual = load_final_analysis(tmp_path)
    assert actual['tables']['balanced_global_ranking'].iloc[0]['feature'] == 'fixture'
    table.write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        load_final_analysis(tmp_path)


def test_final_presentation_rejects_unsafe_manifest_paths(tmp_path):
    directory = tmp_path / FINAL_ANALYSIS_PATH
    directory.mkdir(parents=True)
    (directory / 'analysis.json').write_text(json.dumps({'status': 'complete', 'label': 'final analysis; no selection use',
                                                       'output_sha256': {'../outside.csv': 'invalid'}}))
    with pytest.raises(ValueError, match='path'):
        load_final_analysis(tmp_path)


@pytest.mark.parametrize('missing', ['deletion_replicates.npz', 'per_class_top10.png'])
def test_final_presentation_rejects_an_omitted_declared_output(tmp_path, missing):
    directory = tmp_path / FINAL_ANALYSIS_PATH
    directory.mkdir(parents=True)
    manifest = completed_manifest(directory)
    manifest['output_sha256'].pop(missing)
    (directory / 'analysis.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='declared'):
        load_final_analysis(tmp_path)
