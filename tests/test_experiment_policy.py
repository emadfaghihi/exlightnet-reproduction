import numpy as np
import pytest
import torch


def test_unmarked_test_cannot_perform_optimizer_updates():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = torch.optim.Adam([parameter])
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(AssertionError, match='synthetic-training'):
        optimizer.step()
    assert parameter.item() == 1.


def test_unit_tests_cannot_collect_actual_latency():
    from exlightnet.bench import benchmark
    with pytest.raises(AssertionError, match='benchmarks are forbidden'):
        benchmark(object(), np.zeros((1, 40), dtype=np.float32))
