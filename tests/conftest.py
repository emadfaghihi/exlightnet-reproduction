"""Keep unit verification separate from real experiments and measurements."""
import pytest


SYNTHETIC_TRAINING_TESTS = {
    'test_deterministic_training_restores_checkpoint': {'fits': 2, 'epochs': 4, 'train_rows_per_fit': 24},
    'test_changed_checkpoint_cannot_reuse_test_metrics': {'fits': 1, 'epochs': 1, 'train_rows_per_fit': 4},
    'test_skip_benchmark_suppresses_all_timing_calls': {'fits': 1, 'epochs': 1, 'train_rows_per_fit': 4},
    'test_benchmark_pass_only_updates_timing_from_cached_models': {'fits': 1, 'epochs': 1, 'train_rows_per_fit': 4},
}


@pytest.fixture(autouse=True)
def experiment_policy(request, monkeypatch):
    import torch
    from exlightnet import bench, execution, run_all
    specification = SYNTHETIC_TRAINING_TESTS.get(request.node.name)
    if specification:
        request.node.user_properties.extend([('synthetic_training', 'true'),
                                             *[(key, str(value)) for key, value in specification.items()]])
    else:
        def forbidden_step(*args, **kwargs):
            raise AssertionError('Optimizer updates require a declared tiny synthetic-training test')
        monkeypatch.setattr(torch.optim.Adam, 'step', forbidden_step)

    def forbidden_benchmark(*args, **kwargs):
        raise AssertionError('Actual benchmarks are forbidden in unit tests; use a timing stub')
    for module in (bench, execution, run_all):
        monkeypatch.setattr(module, 'benchmark', forbidden_benchmark)
