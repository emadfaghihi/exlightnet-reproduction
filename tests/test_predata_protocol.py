import ast
import json
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def test_original_protocol_declaration_is_preserved_with_dated_amendments():
    original = json.loads((ROOT / 'docs/protocol_declarations/2026-11-17-original.json').read_text())
    current = json.loads((ROOT / 'docs/statistics_protocol.json').read_text())
    amendments = current.pop('amendments')
    assert current == original
    assert len(amendments) == 6
    assert len({entry['id'] for entry in amendments}) == 6
    assert all(entry['date'] == '2026-11-18' for entry in amendments)
    stopping = next(entry for entry in amendments if entry['id'] == 'PA-20261118-1')
    assert stopping['supersedes_rule_index'] == 3
    assert stopping['primary_endpoint']['experimental_checkpoint'] == 'last'
    assert stopping['secondary_endpoint']['experimental_checkpoint'] == 'best'
    assert stopping['primary_endpoint']['reference_checkpoint'] == 'best'
    assert stopping['matched_seeds'] == list(range(42, 57))
    assert stopping['sd_ratio_bootstrap']['replicates'] == 2000
    assert stopping['sd_ratio_bootstrap']['random_seed'] == 42


def test_thin_hosted_notebook_wires_selector_without_executing_any_study():
    notebook = nbformat.read(ROOT / 'notebooks/kaggle_runner.ipynb', as_version=4)
    cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
    assert all(cell.execution_count is None and not cell.outputs for cell in cells)
    for cell in cells:
        ast.parse(cell.source)
    cell = next(cell for cell in cells if 'STUDY =' in cell.source)
    calls = []
    class Process:
        @staticmethod
        def run(command, **kwargs):
            calls.append((command, kwargs))
    class Environment:
        environ = {'test': 'fixture'}
    class Interpreter:
        executable = 'fixture-python'
    namespace = {'SOURCE': Path('/fake/code'), 'PREPARED': Path('/fake/prepared'),
                 'subprocess': Process, 'os': Environment, 'sys': Interpreter}
    exec(compile(cell.source, 'kaggle-runner-command-fixture', 'exec'), namespace)
    assert len(calls) == 1
    command = calls[0][0]
    assert command[command.index('--only') + 1] == 'all'
    assert command[command.index('--study') + 1] == 'probe'
    assert command[command.index('--device') + 1] == 'cuda'
    assert not any('fit_model' in node.id for node in ast.walk(ast.parse(cell.source)) if isinstance(node, ast.Name))
