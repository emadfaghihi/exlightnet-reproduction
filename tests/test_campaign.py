import numpy as np
import pytest
import torch
from torch.utils.data import WeightedRandomSampler

from exlightnet.campaign import study_runs, run_fingerprint, budget_allows
from exlightnet.train import sample_epoch_indices


def test_study_plan_and_reuse():
    assert len(study_runs("B1")) == 330
    b3 = study_runs("B3")
    assert len(b3) == 180
    assert sum(row.get("reuse_study") == "B1" for row in b3) == 60
    assert len(study_runs("B2")) == 90
    assert len(study_runs("B4")) == 30
    assert len(study_runs("B5")) == 30


def test_run_fingerprint_depends_on_device_and_campaign():
    base = {"campaign": "gpu_v2", "device": "cpu", "seed": 42}
    assert run_fingerprint(base) == run_fingerprint(dict(reversed(list(base.items()))))
    assert run_fingerprint(base) != run_fingerprint({**base, "device": "cuda"})
    assert run_fingerprint(base) != run_fingerprint({**base, "campaign": "other"})


def test_budget_boundary():
    assert budget_allows(0, 3600)
    assert not budget_allows(3600, 3600)


def test_sampling_matches_weighted_random_sampler():
    y = np.repeat(np.arange(3), [10, 30, 60])
    weights = np.bincount(y)[y].astype(float) ** -.5
    reference = np.array(list(WeightedRandomSampler(torch.from_numpy(weights), 100000, True, generator=torch.Generator().manual_seed(1))))
    actual = sample_epoch_indices(y, torch.Generator().manual_seed(2), count=100000).numpy()
    assert np.max(np.abs(np.bincount(y[actual], minlength=3) / len(actual) - np.bincount(y[reference], minlength=3) / len(reference))) < .01


def test_device_cpu_default():
    from exlightnet.device import configure_device
    assert configure_device() == "cpu"
    with pytest.raises(ValueError):
        configure_device("tpu")


def test_claimed_receipt_is_never_repeated(tmp_path):
    import json
    from exlightnet.campaign import _evaluate
    folder = tmp_path / "best"
    folder.mkdir()
    (folder / "receipt.json").write_text(json.dumps({"status": "claimed", "fingerprint": "x"}))
    with pytest.raises(RuntimeError, match="Incomplete"):
        _evaluate(tmp_path, "best", np.array([0, 1]), ["a", "b"], lambda: pytest.fail("inference repeated"), "x")


def test_archive_excludes_prepared_and_lock_files(tmp_path):
    import zipfile
    from exlightnet.campaign import package_campaign
    root = tmp_path / "campaign"
    root.mkdir()
    (root / "record.json").write_text("{}")
    (root / "prepared.npz").write_bytes(b"sensitive")
    (root / "job.lock").write_text("")
    target = package_campaign(root, tmp_path / "out.zip")
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ["record.json"]


def test_completed_evaluation_reuses_evidence_and_detects_corruption(tmp_path):
    from exlightnet.campaign import _evaluate
    y = np.array([0, 1], dtype=np.int8)
    p = np.array([[.8, .2], [.1, .9]])
    first = _evaluate(tmp_path, "best", y, ["a", "b"], lambda: p, "run:best")
    cached = _evaluate(tmp_path, "best", y, ["a", "b"], lambda: pytest.fail("inference repeated"), "run:best")
    assert first == cached
    with np.load(first["prediction_path"], allow_pickle=False) as data:
        assert data["predictions"].dtype == np.int8
        assert data["y"].dtype == np.int8
    (tmp_path / "best" / "predictions.npz").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupted"):
        _evaluate(tmp_path, "best", y, ["a", "b"], lambda: pytest.fail("inference repeated"), "run:best")


def test_prepared_manifest_binds_file_and_payload(tmp_path):
    import hashlib
    import json
    from exlightnet.campaign import verify_prepared_manifest
    data = tmp_path / "split_42.npz"
    data.write_bytes(b"fixture")
    fingerprint = hashlib.sha256(b"fixture").hexdigest()
    metadata = {"data_fingerprint": "recipe", "payload_sha256": "payload"}
    manifest = {"splits": {"42": {"sha256": fingerprint, "data_fingerprint": "recipe", "payload_sha256": "payload"}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    verify_prepared_manifest(tmp_path, 42, metadata)
    data.write_bytes(b"different")
    with pytest.raises(ValueError, match="manifest"):
        verify_prepared_manifest(tmp_path, 42, metadata)


def test_archive_excludes_completed_training_checkpoint(tmp_path):
    import zipfile
    from exlightnet.campaign import package_campaign
    root = tmp_path / "campaign"
    run = root / "cpu" / "run"
    run.mkdir(parents=True)
    (run / "record.json").write_text('{"status":"complete"}')
    (run / "epoch_checkpoint.pt").write_bytes(b"optimizer and model")
    target = package_campaign(root, tmp_path / "out.zip")
    with zipfile.ZipFile(target) as archive:
        assert archive.namelist() == ["cpu/run/record.json"]


def test_campaign_name_cannot_target_canonical_or_traverse():
    from exlightnet.campaign import validate_campaign_name
    for name in ("cpu_v1", "../full", "/tmp/run", "", "."):
        with pytest.raises(ValueError):
            validate_campaign_name(name)
    validate_campaign_name("gpu_v2_smoke")


def test_claimed_receipt_blocks_before_training(tmp_path):
    import json
    from exlightnet.campaign import guard_evaluation_receipts
    best = tmp_path / "best"
    best.mkdir()
    (best / "receipt.json").write_text(json.dumps({"status": "claimed"}))
    with pytest.raises(RuntimeError, match="receipt"):
        guard_evaluation_receipts(tmp_path)


def test_resumed_last_state_does_not_alias_model(tmp_path):
    import random
    from exlightnet.train import TrainConfig, fit_resumable
    model = torch.nn.Linear(2, 2)
    config = TrainConfig(seed=42, max_epochs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .999), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1)
    last = {key: torch.full_like(value, 2) for key, value in model.state_dict().items()}
    best = {key: torch.full_like(value, 1) for key, value in model.state_dict().items()}
    state = {"identity": {"id": "fixture"}, "epoch": 1, "model": last, "best_state": best,
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "generator": torch.Generator().manual_seed(42).get_state(),
             "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
             "torch_rng": torch.get_rng_state(), "cuda_rng": None,
             "best": .5, "best_epoch": 1, "stale": 0,
             "history": [{"epoch": 1}], "warnings": [{"epoch": 1, "phase": "epoch", "message": "prior warning"}]}
    torch.save(state, tmp_path / "epoch_checkpoint.pt")
    x = np.zeros((2, 2), dtype=np.float32)
    y = np.array([0, 1])
    ledger = [{"epoch": None, "phase": "configure", "message": "current warning"}]
    result = fit_resumable(model, x, y, x, y, config, tmp_path, identity={"id": "fixture"}, warning_ledger=ledger)
    assert [warning["message"] for warning in ledger] == ["prior warning", "current warning"]
    model.load_state_dict(result["best_state"])
    for value in result["last_state"].values():
        assert torch.all(value == 2)


def test_restore_archive_is_safe_and_idempotent(tmp_path):
    import zipfile
    from exlightnet.campaign import restore_campaign_archive
    archive = tmp_path / "results.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("cpu/abc/record.json", '{"status":"complete"}')
        z.writestr("cpu/abc/best/predictions.npz", b"predictions")
    output = tmp_path / "out"
    restore_campaign_archive(archive, output, "gpu_v2")
    restore_campaign_archive(archive, output, "gpu_v2")
    assert (output / "gpu_v2/cpu/abc/record.json").read_text() == '{"status":"complete"}'
    other = tmp_path / "other.zip"
    with zipfile.ZipFile(other, "w") as z:
        z.writestr("cpu/abc/record.json", "different")
    with pytest.raises(RuntimeError, match="differ"):
        restore_campaign_archive(other, output, "gpu_v2")
    assert (output / "gpu_v2/cpu/abc/record.json").read_text() == '{"status":"complete"}'


@pytest.mark.parametrize("name,symlink", [("../escape", False), ("/absolute", False), ("cpu/link", True), ("cpu/../../escape", False)])
def test_restore_rejects_unsafe_members(tmp_path, name, symlink):
    import stat
    import zipfile
    from exlightnet.campaign import restore_campaign_archive
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        info = zipfile.ZipInfo(name)
        if symlink:
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
        z.writestr(info, "target")
    with pytest.raises(ValueError):
        restore_campaign_archive(archive, tmp_path / "out", "gpu_v2")
    assert not (tmp_path / "escape").exists()


def test_restore_rejects_existing_symlink_destination(tmp_path):
    import zipfile
    from exlightnet.campaign import restore_campaign_archive
    root = tmp_path / "out" / "gpu_v2"
    root.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "cpu").symlink_to(elsewhere, target_is_directory=True)
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("cpu/record.json", "{}")
    with pytest.raises(ValueError, match="symlink"):
        restore_campaign_archive(archive, tmp_path / "out", "gpu_v2")
    assert not (elsewhere / "record.json").exists()


def test_xgboost_version_accepts_cpu_distribution(monkeypatch):
    import importlib.metadata
    from exlightnet.campaign import library_versions
    real_version = importlib.metadata.version
    def version(name):
        if name == "xgboost":
            raise importlib.metadata.PackageNotFoundError(name)
        if name == "xgboost-cpu":
            return "2.1.4"
        return real_version(name)
    monkeypatch.setattr(importlib.metadata, "version", version)
    assert library_versions()["xgboost"] == "2.1.4"


def test_environment_identity_and_probe_validation():
    from exlightnet.device import environment_identity
    from exlightnet.campaign import validate_probe
    libs = {"numpy": "1", "torch": "2", "scikit-learn": "3", "xgboost": "4", "cuda": "12", "cudnn": 9, "gpu_name": "A"}
    env = environment_identity("cuda", libs)
    assert env != environment_identity("cuda", {**libs, "gpu_name": "B"})
    assert environment_identity("cpu", libs) == environment_identity("cpu", {**libs, "gpu_name": "B"})
    validate_probe({"environment": env}, env)
    with pytest.raises(RuntimeError, match="probe"):
        validate_probe({}, env)
    with pytest.raises(RuntimeError, match="probe"):
        validate_probe({"environment": env}, environment_identity("cuda", {**libs, "gpu_name": "B"}))


def test_warning_ledger_from_previous_epoch_checkpoint(tmp_path):
    from exlightnet.train import checkpoint_warnings
    previous = [{"epoch": 1, "phase": "train", "message": "nondeterministic"}]
    assert checkpoint_warnings({"warnings": previous}, [{"epoch": 2, "phase": "train", "message": "warning"}]) == previous + [{"epoch": 2, "phase": "train", "message": "warning"}]
    assert checkpoint_warnings({}, []) == []


def test_tree_finalize_after_zip_restore_without_model(tmp_path):
    import json
    from exlightnet.campaign import _evaluate, finalize_tree_cache, package_campaign, restore_campaign_archive
    root = tmp_path / "source" / "gpu_v2"
    run = root / "cpu" / "abc"
    run.mkdir(parents=True)
    record = {"run_fingerprint": "abc", "status": "pending", "model": "random_forest"}
    (run / "run_identity.json").write_text(json.dumps(record))
    y = np.array([0, 1]); p = np.array([[.9, .1], [.2, .8]])
    _evaluate(run, "best", y, ["a", "b"], lambda: p, "abc:best")
    before = (run / "best/receipt.json").read_bytes()
    zip_path = package_campaign(root, tmp_path / "backup.zip")
    restored = restore_campaign_archive(zip_path, tmp_path / "out", "gpu_v2")
    result = finalize_tree_cache(restored / "cpu/abc", restored, "abc")
    assert result["status"] == "complete"
    assert result["checkpoints"]["best"] == result["checkpoints"]["last"]
    assert (restored / "cpu/abc/best/receipt.json").read_bytes() == before


def test_run_local_failures_do_not_stop_next_plan(tmp_path):
    from exlightnet.campaign import run_independent_plans, RunStopped
    seen = []
    def work(plan):
        seen.append(plan)
        if plan == 1:
            raise RunStopped("claimed receipt")
    result = run_independent_plans([1, 2], work, tmp_path)
    assert seen == [1, 2]
    assert result["stopped_count"] == 1
    assert "claimed receipt" in result["stopped"][0]["reason"]


def test_campaign_continues_after_claimed_receipt_without_training(tmp_path, monkeypatch):
    import dataclasses
    import json
    import exlightnet.campaign as campaign
    import exlightnet.device as device
    import exlightnet.prepared as prepared
    import exlightnet.baselines as baselines
    from exlightnet.train import TrainConfig
    plans = [{"study": "B1", "task": "binary", "model": "random_forest", "seed": n, "split_seed": 42, "arm": "paper"} for n in (42, 43)]
    libs = {"numpy": "n", "torch": "t", "scikit-learn": "s", "xgboost": "x"}
    meta = {"frozen_alpha": {"binary": "none"}, "data_fingerprint": "data", "payload_sha256": "payload"}
    monkeypatch.setattr(campaign, "study_runs", lambda study: plans)
    monkeypatch.setattr(campaign, "library_versions", lambda: dict(libs))
    monkeypatch.setattr(device, "configure_device", lambda value="cpu": "cpu")
    monkeypatch.setattr(prepared, "load_prepared", lambda path: {"arrays": {}, "metadata": meta})
    monkeypatch.setattr(campaign, "verify_prepared_manifest", lambda *args: None)
    monkeypatch.setattr(baselines, "build_model", lambda *a: pytest.fail("training or fitting reached"))
    prepared_dir = tmp_path / "prepared"; prepared_dir.mkdir()
    (prepared_dir / "split_42.npz").write_bytes(b"file")
    root = tmp_path / "out/gpu_v2"; root.mkdir(parents=True)
    versions = {**libs, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu_name": None}
    (root / "probe_cpu.json").write_text(json.dumps({"seconds_per_epoch": 1, "environment": device.environment_identity("cpu", versions)}))
    for plan in plans:
        cfg = TrainConfig(seed=plan["seed"], max_epochs=20, alpha_mode="none")
        identity = {"campaign": "gpu_v2", "plan": plan, "device": "cpu", "config": dataclasses.asdict(cfg), "data_fingerprint": "data", "payload_sha256": "payload", "prepared_sha256": campaign._sha256(prepared_dir / "split_42.npz"), "libraries": versions}
        fp = campaign.run_fingerprint(identity)
        folder = root / "cpu" / fp; folder.mkdir(parents=True)
        if plan["seed"] == 42:
            (folder / "best").mkdir()
            (folder / "best/receipt.json").write_text('{"status":"claimed"}')
        else:
            (folder / "record.json").write_text(json.dumps({"status": "complete", "run_fingerprint": fp, "checkpoints": {}}))
    result = campaign.run_campaign(prepared_dir, tmp_path / "out", "B1")
    assert result["stopped_count"] == 1
    assert result["new_completed"] == 0
    assert list((root / "failures").glob("*.json"))
