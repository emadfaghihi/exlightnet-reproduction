import dataclasses
from pathlib import Path

import numpy as np
import pytest

from exlightnet.io import read_json, write_json
from exlightnet.run_all import _run_one, run
from exlightnet.train import TrainConfig


def small_arrays():
    arrays = {"x_" + p: np.zeros((4, 40), dtype="float32") for p in ("train", "val", "test")}
    for p in ("train", "val", "test"):
        arrays["y2_" + p] = np.array([0, 1, 0, 1])
        arrays["y12_" + p] = np.array([0, 1, 0, 1])
    return arrays


def test_cache_only_refuses_to_train_or_evaluate(tmp_path):
    with pytest.raises(RuntimeError, match="cache-only"):
        _run_one("binary", "exlightnet", 42, small_arrays(), [], {"chosen": "none"}, {"fingerprint": "fixture"}, tmp_path, 20, cache_only=True)
    assert not (tmp_path / "model.pt").exists()
    assert not (tmp_path / "test_evaluation_receipt.json").exists()


def test_skip_benchmark_suppresses_all_timing_calls(tmp_path, monkeypatch):
    import exlightnet.run_all as runner
    class PP:
        def compare_table4(self):
            return {"gate": "P1", "report_only": True}
    monkeypatch.setattr(runner, "prepare_data", lambda root: (small_arrays(), ["normal", "attack"], PP(), {"fingerprint": "fixture", "gates": []}))
    monkeypatch.setattr(runner, "sampler_gate", lambda y: {"gate": "T1", "pass": True})
    monkeypatch.setattr(runner, "choose_alpha", lambda *args: {"chosen": "none", "sweep_max_epochs": 1, "candidates": []})
    def no_benchmark(*args, **kwargs):
        pytest.fail("--skip-benchmark must not time models")
    monkeypatch.setattr(runner, "benchmark", no_benchmark)
    run(tmp_path, models=("mlp",), tasks=("binary",), epochs=1, ablations=False, explanations=False, skip_benchmark=True)
    rows = read_json(tmp_path / "results/fast/results.json")
    assert len(rows) == 1 and "latency_mean_ms" not in rows[0]


def test_seed_worker_requires_precomputed_alpha_and_single_seed(tmp_path):
    with pytest.raises(ValueError, match="one seed"):
        run(tmp_path, mode="full", seeds=[42, 43], seed_worker=True, skip_benchmark=True)
    with pytest.raises((RuntimeError, FileNotFoundError), match="[Pp]repared|[Aa]lpha|cache"):
        run(tmp_path, mode="full", seeds=[42], seed_worker=True, skip_benchmark=True)
    assert not (tmp_path / "results/full/manifest.json").exists()
    assert not (tmp_path / "REPORT.md").exists()


def test_alpha_only_does_not_train_final_models(tmp_path, monkeypatch):
    import exlightnet.run_all as runner
    class PP:
        def compare_table4(self):
            return {"gate": "P1", "report_only": True}
    monkeypatch.setattr(runner, "prepare_data", lambda root: (small_arrays(), [], PP(), {"fingerprint": "fixture", "gates": []}))
    monkeypatch.setattr(runner, "sampler_gate", lambda y: {"gate": "T1", "pass": True})
    seen = []
    def selection(task, *args):
        seen.append(task)
        return {"chosen": "none", "sweep_max_epochs": 8, "candidates": []}
    monkeypatch.setattr(runner, "choose_alpha", selection)
    monkeypatch.setattr(runner, "_run_one", lambda *args, **kwargs: pytest.fail("alpha-only cannot evaluate final models"))
    result = run(tmp_path, mode="full", alpha_only=True, skip_benchmark=True)
    assert seen == ["binary", "multiclass"]
    assert result["stage"] == "alpha_only"
    assert not (tmp_path / "results/full/results.json").exists()


def test_aggregation_never_runs_uncached_analysis_or_proxy_training(tmp_path, monkeypatch):
    import exlightnet.run_all as runner
    import exlightnet.execution as execution
    import exlightnet.explain as explain
    import exlightnet.proxy as proxy
    class PP:
        def compare_table4(self):
            return {"gate": "P1", "report_only": True}
    monkeypatch.setattr(execution, "load_prepared_data", lambda root: (small_arrays(), [], PP(), {"fingerprint": "fixture", "gates": []}))
    monkeypatch.setattr(execution, "load_alpha", lambda *args: {"chosen": "none", "sweep_max_epochs": 8, "candidates": []})
    monkeypatch.setattr(runner, "sampler_gate", lambda y: {"gate": "T1", "pass": True})
    def cached_row(task, name, seed, *args, **kwargs):
        assert kwargs["cache_only"]
        return {"task": task, "model": name, "seed": seed, "macro_f1": .7, "params": 38328, "serialized_bytes": 1}
    monkeypatch.setattr(runner, "_run_one", cached_row)
    monkeypatch.setattr(runner, "_load_run_model", lambda *args: None)
    monkeypatch.setattr(runner, "load_int8", lambda *args: None)
    monkeypatch.setattr(explain, "run_explanations", lambda *args: pytest.fail("aggregation cannot explain uncached test inputs"))
    monkeypatch.setattr(proxy, "run_proxy", lambda *args: pytest.fail("aggregation cannot train a proxy"))
    path = tmp_path / "results/full/multiclass/exlightnet/42"
    path.mkdir(parents=True)
    np.savez_compressed(path / "predictions.npz", probabilities=np.ones((4, 12)) / 12)
    run(tmp_path, mode="full", models=("exlightnet",), tasks=("multiclass",), seeds=[42], ablations=False, aggregate_only=True, leakage_proxy=True)
    assert not (tmp_path / "results/full/explain/explanations.json").exists()
