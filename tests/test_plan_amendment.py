"""Pre-data scheduling and family selection without training or GPU access."""
from collections import Counter
from dataclasses import asdict
import hashlib
from itertools import groupby
import json
from pathlib import Path
import runpy
import sys

import pytest
import torch

from exlightnet import baselines, campaign, device, train
from test_prepared import recipe_digest, save_fixture


TREE_MODELS = {"random_forest", "xgboost"}
ARMS = ("paper", "fixed30", "patience10_min15")
BINARY = ("exlightnet", "mlp", "random_forest", "xgboost")
MULTI_B1 = ("exlightnet", "mlp", "standard_conv", "no_se", "std_cnn", "random_forest", "xgboost")
MULTI_B2 = ("exlightnet", "mlp", "standard_conv", "random_forest", "xgboost")
MULTI_B3 = ("exlightnet", "standard_conv", "mlp", "std_cnn")


def baseline_plans(study):
    """Frozen pure plans in the cbbea40 order."""
    def row(task, model, seed, split=42, arm="paper", **extra):
        return dict(study=study, task=task, model=model, seed=seed, split_seed=split, arm=arm, **extra)

    if study == "B1":
        return [row(task, model, seed)
                for task, models in (("binary", BINARY), ("multiclass", MULTI_B1))
                for model in models for seed in range(42, 72)]
    if study == "B2":
        return [row(task, model, seed, seed)
                for task, models in (("binary", BINARY), ("multiclass", MULTI_B2))
                for model in models for seed in range(42, 52)]
    if study == "B3":
        return [row("multiclass", model, seed, arm=arm, **({"reuse_study": "B1"} if arm == "paper" else {}))
                for model in MULTI_B3 for seed in range(42, 57) for arm in ARMS]
    if study == "B4":
        return [row("binary", model, seed, arm="uncapped60")
                for model in ("exlightnet", "mlp") for seed in range(42, 57)]
    if study == "B5":
        return [row("multiclass", "std_cnn", seed, arm=f"lr_{lr:g}", lr=lr)
                for lr in (.0003, .001, .0015) for seed in range(42, 52)]
    raise ValueError(study)


def baseline_identity(plan, gpu_device, versions, metadata, prepared_sha256):
    """Construct the runner's cbbea40 identity using fixed non-training inputs."""
    task = plan["task"]
    max_epochs = 60 if plan["study"] == "B4" else 20 if task == "binary" else 30
    config = train.TrainConfig(seed=plan["seed"], max_epochs=max_epochs, alpha_mode=metadata["frozen_alpha"][task],
                               patience=10 if plan["arm"] == "patience10_min15" else 6, lr=plan.get("lr", .0015))
    return {"campaign": "gpu_v2", "plan": plan, "device": "cpu" if plan["model"] in TREE_MODELS else gpu_device,
            "config": asdict(config), "data_fingerprint": metadata["data_fingerprint"],
            "payload_sha256": metadata["payload_sha256"], "prepared_sha256": prepared_sha256,
            "libraries": versions}


def baseline_fingerprint(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def test_seed_major_fingerprints_preserve_exact_pre_amendment_multisets():
    for study in ("B1", "B2", "B3", "B4", "B5"):
        old = baseline_plans(study)
        new = campaign.study_runs(study)
        assert Counter(map(campaign.run_fingerprint, new)) == Counter(map(baseline_fingerprint, old))
        if study in ("B4", "B5"):
            assert new == old
        versions = {"numpy": "n", "torch": "t", "scikit-learn": "s", "xgboost": "x",
                    "cuda": "12", "cudnn": 9, "gpu_name": "probe fixture"}
        metadata = {"frozen_alpha": {"binary": "effective_num", "multiclass": "none"},
                    "data_fingerprint": "fixture recipe", "payload_sha256": "fixture payload"}
        def identity(row):
            return baseline_identity(row, "cuda", versions, metadata, f"fixture split {row['split_seed']}")
        assert Counter(campaign.run_fingerprint(identity(row)) for row in new) == Counter(
            baseline_fingerprint(identity(row)) for row in old
        )


@pytest.mark.parametrize("study", ("B1", "B2", "B3"))
def test_every_plan_prefix_has_complete_seed_blocks_except_the_last(study):
    key = lambda row: (row["task"], row["model"], row["arm"])
    seed_key = lambda row: row["seed"]
    plans = campaign.study_runs(study)
    reference = baseline_plans(study)
    block = [key(row) for row in reference if row["seed"] == 42]
    for end in range(len(plans) + 1):
        complete, partial = divmod(end, len(block))
        expected = [(seed, block) for seed in range(42, 42 + complete)]
        if partial:
            expected.append((42 + complete, block[:partial]))
        actual = [(seed, list(map(key, rows))) for seed, rows in groupby(plans[:end], seed_key)]
        assert actual == expected, (study, end)


@pytest.mark.parametrize("study,stop", [("B1", 72), ("B2", 52)])
def test_tree_selection_keeps_every_seed_of_both_models(study, stop):
    rows = campaign.select_study_runs(study, "trees")
    assert Counter((p["task"], p["model"], p["seed"]) for p in rows) == Counter(
        (task, model, seed) for task in ("binary", "multiclass") for model in TREE_MODELS for seed in range(42, stop)
    )
    assert campaign.select_study_runs(study) == campaign.study_runs(study)
    assert campaign.select_study_runs(study, "neural") == [p for p in campaign.study_runs(study) if p not in rows]


def _forbid_cuda(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("CUDA called during a trees-only session")

    for name in ("is_available", "get_device_name", "manual_seed_all", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, unexpected)


@pytest.fixture
def prepared_b1(tmp_path):
    directory = tmp_path / "prepared"
    directory.mkdir()
    path = directory / "split_42.npz"

    def split_42(arrays, features, classes, metadata):
        metadata["split_seed"] = metadata["settings"]["split_seed"] = 42
        metadata["data_fingerprint"] = recipe_digest(metadata["settings"])

    _, metadata = save_fixture(path, split_42)
    sha256 = campaign._sha256(path)
    (directory / "manifest.json").write_text(json.dumps({"splits": {"42": {
        "sha256": sha256, "data_fingerprint": metadata["data_fingerprint"], "payload_sha256": metadata["payload_sha256"]
    }}}))
    return directory, metadata, sha256


def _write_completed(root, rows, metadata, sha256, versions):
    for plan in rows:
        identity = baseline_identity(plan, "cpu", versions, metadata, sha256)
        fingerprint = baseline_fingerprint(identity)
        path = root / "cpu" / fingerprint / "record.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({**{key: value for key, value in identity.items() if key != 'plan'}, **plan,
                                    "status": "complete", "run_fingerprint": fingerprint,
                                    "checkpoints": {}}))


def _cpu_versions():
    return {**campaign.library_versions(), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu_name": None}


def test_trees_only_uses_cpu_without_probe_and_resumes_cached_runs(prepared_b1, tmp_path, monkeypatch, capsys):
    prepared, metadata, sha256 = prepared_b1
    output = tmp_path / "out"
    root = output / "gpu_v2"
    rows = [p for p in baseline_plans("B1") if p["model"] in TREE_MODELS]
    versions = _cpu_versions()
    _write_completed(root, rows[:-1], metadata, sha256, versions)
    _forbid_cuda(monkeypatch)

    def stop_before_fit(name, classes, seed):
        assert (name, classes, seed) == ("xgboost", 12, 71)
        raise campaign.RunStopped("Synthetic stop before fitting")

    monkeypatch.setattr(baselines, "build_model", stop_before_fit)
    monkeypatch.setattr(baselines, "fit_tree", lambda *args: pytest.fail("Tree fitting reached"))
    result = campaign.run_campaign(prepared, output, "B1", device="cuda", only="trees")
    assert result["new_completed"] == 0
    assert result["stopped_count"] == 1
    assert result["stopped"][0]["plan"] == rows[-1]
    assert len(list((root / "cpu").glob("*/record.json"))) == 119
    assert not (root / "cuda").exists()
    assert (output / "gpu_v2.zip").is_file()
    assert "120" in capsys.readouterr().out


@pytest.mark.parametrize("only,expected_count", [("all", 330), ("neural", 210)])
def test_all_and_neural_reuse_baseline_fingerprints_with_matching_probe(
    only, expected_count, prepared_b1, tmp_path, monkeypatch, capsys
):
    prepared, metadata, sha256 = prepared_b1
    output = tmp_path / "out"
    root = output / "gpu_v2"
    versions = _cpu_versions()
    rows = [p for p in baseline_plans("B1") if only == "all" or p["model"] not in TREE_MODELS]
    _write_completed(root, rows, metadata, sha256, versions)
    monkeypatch.setattr(baselines, "build_model", lambda *args: pytest.fail("Model creation reached"))
    kwargs = {"only": only} if only != "all" else {}
    with pytest.raises(RuntimeError, match="three-epoch probe"):
        campaign.run_campaign(prepared, output, "B1", **kwargs)
    probe = root / "probe_cpu.json"
    probe.write_text(json.dumps({"seconds_per_epoch": 1, "environment": {"device": "cuda"}}))
    with pytest.raises(RuntimeError, match="Timing probe environment differs"):
        campaign.run_campaign(prepared, output, "B1", **kwargs)
    probe.write_text(json.dumps({"seconds_per_epoch": 1, "environment": device.environment_identity("cpu", versions)}))
    result = campaign.run_campaign(prepared, output, "B1", **kwargs)
    assert result["new_completed"] == 0
    assert result["stopped_count"] == 0
    assert len(list((root / "cpu").glob("*"))) == expected_count
    assert f"B1: {expected_count} " in capsys.readouterr().out


@pytest.mark.parametrize("study", ("B3", "B4", "B5", "probe", "determinism"))
def test_empty_tree_selection_skips_inputs_cuda_and_outputs(study, tmp_path, monkeypatch, capsys):
    _forbid_cuda(monkeypatch)
    output = tmp_path / "out"
    result = campaign.run_campaign(tmp_path / "missing", output, study, device="auto", only="trees")
    assert result["new_completed"] == result["stopped_count"] == 0
    assert "No runs selected" in capsys.readouterr().out
    assert not output.exists()


def test_runner_help_reports_only_choices_and_default(monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_campaign.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(script), run_name="__main__")
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--only {all,neural,trees}" in help_text
    assert "default: all" in help_text


def test_runner_cli_forwards_trees_selection(tmp_path, monkeypatch, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "kaggle_campaign.py"
    output = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [str(script), "--study", "probe", "--device", "cuda", "--only", "trees",
                                      "--prepared-dir", str(tmp_path / "missing"), "--output", str(output)])
    _forbid_cuda(monkeypatch)
    runpy.run_path(str(script), run_name="__main__")
    assert "No runs selected" in capsys.readouterr().out
    assert not output.exists()
