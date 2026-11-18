"""Read and verify completed final-analysis evidence; never query a model."""
import json
from pathlib import Path, PurePosixPath

import pandas as pd

from .io import sha256


FINAL_ANALYSIS_PATH = Path('artifacts/final_analysis/stratified_shap_20261118')


def load_final_analysis(root):
    from .final_analysis import OUTPUT_FILES
    directory = Path(root) / FINAL_ANALYSIS_PATH
    path = directory / 'analysis.json'
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get('status') != 'complete':
        return None
    if manifest.get('label') != 'final analysis; no selection use':
        raise ValueError('Unexpected final-analysis declaration')
    if set(manifest.get('output_sha256', {})) != set(OUTPUT_FILES):
        raise ValueError('Final-analysis output paths differ from the declared set')
    tables = {}
    for name, expected in manifest['output_sha256'].items():
        member = PurePosixPath(name)
        if member.is_absolute() or len(member.parts) != 1 or '\\' in name or member.name in ('.', '..'):
            raise ValueError('Unsafe final-analysis output path')
        source = directory / name
        if source.is_symlink() or not source.is_file() or sha256(source) != expected:
            raise ValueError(f'Final-analysis checksum differs: {name}')
        if source.suffix == '.csv':
            tables[source.stem] = pd.read_csv(source)
    return {'manifest': manifest, 'tables': tables, 'directory': directory}
