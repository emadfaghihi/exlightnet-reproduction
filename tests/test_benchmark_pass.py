import numpy as np
import pytest

from exlightnet.execution import execution_lock, wait_for_idle


def test_idle_check_requires_consecutive_quiet_windows():
    values = iter([(0, 0), (30, 100), (31, 200), (33, 300), (34, 400)])
    result = wait_for_idle(consecutive=3, reader=lambda: next(values), sleeper=lambda _: None)
    assert result["observed_windows"] == 4
    assert result["quiet_window_busy_fractions"] == [.01, .02, .01]


def test_exclusive_benchmark_lock_refuses_active_worker(tmp_path):
    with execution_lock(tmp_path, "full", seed=42):
        with pytest.raises(RuntimeError, match="process"):
            with execution_lock(tmp_path, "full"):
                pytest.fail("cannot benchmark while seed worker runs")


def test_benchmark_lock_is_shared_across_fast_and_full(tmp_path):
    with execution_lock(tmp_path, "fast", seed=42):
        with pytest.raises(RuntimeError, match="process"):
            with execution_lock(tmp_path, "full"):
                pytest.fail("full benchmarking cannot overlap a fast worker")


def test_benchmark_pass_only_updates_timing_from_cached_models(tmp_path, monkeypatch):
    import exlightnet.execution as execution
    import exlightnet.run_all as runner
    from exlightnet.io import read_json, write_json
    from exlightnet.metrics import evaluate_once
    from exlightnet.train import seed_everything
    seed_everything(42)
    x = np.zeros((4, 40), dtype="float32")
    y = np.array([0, 1, 0, 1])
    arrays = {"x_" + p: x for p in ("train", "val", "test")}
    arrays.update({"y2_" + p: y for p in ("train", "val", "test")})
    directory = tmp_path / "results/full"
    row = runner._run_one("binary", "mlp", 42, arrays, [], {"chosen": "none"}, {"fingerprint": "fixture"}, directory / "binary/mlp/42", 1)
    write_json(directory / "results.json", [row])
    write_json(directory / "gates.json", [])
    write_json(directory / "failures.json", [])
    write_json(directory / "manifest.json", {"mode": "full", "seeds": [42], "full_five_seed_headlines_complete": False})
    monkeypatch.setattr(execution, "load_prepared_data", lambda root: ({"x_train": x}, [], None, {"fingerprint": "fixture"}))
    monkeypatch.setattr(execution, "wait_for_idle", lambda: {"quiet_window_busy_fractions": [.01, .01, .01]})
    observed = []
    def measured(model, x):
        observed.append(len(x))
        return {"latency_mean_ms": 1., "latency_median_ms": .9, "latency_p95_ms": 1.2}
    monkeypatch.setattr(execution, "benchmark", measured)
    def publish_fixture(root, mode, rows, gates, failures, manifest, **kwargs):
        write_json(directory / "results.json", rows)
        write_json(directory / "manifest.json", manifest)
        write_json(directory / "failures.json", failures)
    monkeypatch.setattr(execution, "publish", publish_fixture)
    monkeypatch.setattr(runner, "_run_one", lambda *args, **kwargs: pytest.fail("benchmark-only cannot train/test"))
    receipt = (directory / "binary/mlp/42/test_evaluation_receipt.json").read_bytes()
    result = runner.run(tmp_path, mode="full", benchmark_only=True)
    assert observed == [4]
    assert result["stage"] == "benchmark_only" and result["processes"] == 1
    updated = read_json(directory / "results.json")[0]
    assert updated["macro_f1"] == row["macro_f1"]
    assert updated["latency_mean_ms"] == 1.
    assert receipt == (directory / "binary/mlp/42/test_evaluation_receipt.json").read_bytes()
    def busy_cpu():
        raise RuntimeError("idle check fixture failed")
    monkeypatch.setattr(execution, "wait_for_idle", busy_cpu)
    failed = runner.run(tmp_path, mode="full", benchmark_only=True)
    stale = read_json(directory / "results.json")[0]
    assert "latency_mean_ms" not in stale and "peak_rss_bytes" not in stale and "benchmark_context" not in stale
    assert failed["failed_models"] == 1 and failed["measured_models"] == 0
    assert any("idle check fixture failed" in f["reason"] for f in read_json(directory / "manifest.json")["failures"])
