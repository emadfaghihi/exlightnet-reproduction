import dataclasses
import shutil
from pathlib import Path

import numpy as np
import pytest

from exlightnet.io import read_json, write_json
from exlightnet.report import build_report
from exlightnet.run_all import _run_one, prepare_data, run
from exlightnet.train import TrainConfig


def test_extracted_archive_can_rebuild_missing_array_cache(tmp_path):
    if not Path("artifacts/cache_manifest.json").exists():
        pytest.skip("requires a prepared real corpus")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/RT_IOT2022.csv").symlink_to(Path("data/RT_IOT2022.csv").resolve())
    (tmp_path / "artifacts").mkdir()
    for name in ("cache_manifest.json", "preprocessor.json", "mi_ranking.json", "mi_comparison.json", "split_indices.npz", "class_names.json"):
        shutil.copy(Path("artifacts") / name, tmp_path / "artifacts" / name)
    expected = read_json(tmp_path / "artifacts/cache_manifest.json")["arrays_sha256"]
    arrays, _, _, _ = prepare_data(tmp_path, download=False)
    assert arrays["x_train"].shape == (86181, 40)
    from exlightnet.io import sha256
    assert sha256(tmp_path / "artifacts/cache_arrays.npz") == expected


def test_cached_metrics_cannot_bypass_missing_evidence(tmp_path):
    config = dataclasses.asdict(TrainConfig(max_epochs=1, alpha_mode="none"))
    write_json(tmp_path / "metrics.json", {"recipe": config, "data_fingerprint": "fixture"})
    arrays = {"x_" + p: np.zeros((4, 40), dtype="float32") for p in ("train", "val", "test")}
    arrays.update({"y2_" + p: np.array([0, 1, 0, 1]) for p in ("train", "val", "test")})
    with pytest.raises(RuntimeError, match="Incomplete"):
        _run_one("binary", "exlightnet", 42, arrays, [], {"chosen": "none"}, {"fingerprint": "fixture"}, tmp_path, 1)


def test_changed_checkpoint_cannot_reuse_test_metrics(tmp_path):
    from exlightnet.train import seed_everything
    seed_everything(42)
    arrays = {"x_" + p: np.zeros((4, 40), dtype="float32") for p in ("train", "val", "test")}
    arrays.update({"y2_" + p: np.array([0, 1, 0, 1]) for p in ("train", "val", "test")})
    _run_one("binary", "exlightnet", 42, arrays, [], {"chosen": "none"}, {"fingerprint": "fixture"}, tmp_path, 1)
    with (tmp_path / "model.pt").open("ab") as stream:
        stream.write(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="fingerprint"):
        _run_one("binary", "exlightnet", 42, arrays, [], {"chosen": "none"}, {"fingerprint": "fixture"}, tmp_path, 1)


def test_blocked_run_report_retains_actual_gate_evidence(tmp_path):
    directory = tmp_path / "results/fast"
    write_json(directory / "gates.json", [{"gate": "D1", "pass": False, "actual": {"rows": 17}}])
    write_json(directory / "failures.json", [{"branch": "data", "error_type": "GateFailure", "reason": "D1 fixture has 17 rows"}])
    build_report(tmp_path)
    report = (tmp_path / "REPORT.md").read_text()
    assert "D1" in report and "17" in report and "FAIL" in report


@pytest.mark.parametrize("failing", ["alpha", "M1", "T1"])
def test_failed_deep_prerequisite_continues_independent_tree_branch(tmp_path, monkeypatch, failing):
    import exlightnet.run_all as runner
    arrays = {"x_" + p: np.zeros((4, 40), dtype="float32") for p in ("train", "val", "test")}
    arrays.update({"y12_" + p: np.array([0, 1, 0, 1]) for p in ("train", "val", "test")})
    class Preprocessor:
        def compare_table4(self):
            return {"gate": "P1", "report_only": True}
    # External boundaries: fixture data and deliberate selection failure. Real
    # runner dispatch/finalization operates; _run_one records which job survives.
    monkeypatch.setattr(runner, "prepare_data", lambda root: (arrays, ["a", "b"], Preprocessor(), {"fingerprint": "fixture", "gates": []}))
    monkeypatch.setattr(runner, "sampler_gate", lambda y: {"gate": "T1", "pass": True})
    def failed_selection(*args, **kwargs):
        raise ValueError("alpha fixture failure")
    monkeypatch.setattr(runner, "choose_alpha", failed_selection)
    if failing == "M1":
        def failed_model_gate():
            raise AssertionError("M1 fixture count 38329")
        monkeypatch.setattr(runner, "model_gate", failed_model_gate)
    if failing == "T1":
        monkeypatch.setattr(runner, "sampler_gate", lambda y: {"gate": "T1", "pass": False, "actual": {"tempered_multiplicity": 999}})
    observed = []
    def record_run(task, name, *args):
        observed.append(name)
        raise RuntimeError("stop before fixture tree training")
    monkeypatch.setattr(runner, "_run_one", record_run)
    result = run(tmp_path, models=("exlightnet", "random_forest"), tasks=("multiclass",), ablations=False, explanations=False)
    assert observed == ["random_forest"]
    assert any(("alpha" if failing in ("alpha", "M1") else "sampler") in f["branch"] for f in result["failures"])
    assert (tmp_path / "REPORT.md").exists()
    assert (tmp_path / "results/summary.csv").exists()
    assert read_json(tmp_path / "results/summary_scope.json")["mode"] == "fast"
