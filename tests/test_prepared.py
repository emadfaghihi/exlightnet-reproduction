"""Prepared exports reject tampering and preserve the original train-only recipe."""
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.model_selection import train_test_split


def prepared_module():
    return importlib.import_module("exlightnet.prepared")


def recipe_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def fixture_payload():
    from exlightnet.data import EXPECTED_COUNTS
    names = sorted(EXPECTED_COUNTS)
    arrays = {}
    for i, part in enumerate(("train", "val", "test")):
        arrays["x_" + part] = np.arange(480, dtype=np.float32).reshape(12, 40) + i
        arrays["y12_" + part] = np.arange(12, dtype=np.int64)
        arrays["y2_" + part] = (~np.isin(names, ["thing_speak", "mqtt_publish", "wipro_bulb"])).astype(np.int64)
        arrays["split_" + part] = np.arange(i * 12, (i + 1) * 12, dtype=np.int64)
    settings = {"data_sha256": "0" * 64, "preprocessing_format": 1, "split_seed": 43,
                "mi": "binary,3-neighbors,one-hot-discrete,seed42", "top_k": 40}
    metadata = {"schema_version": 1, "data_fingerprint": recipe_digest(settings), "split_seed": 43,
                "settings": settings, "frozen_alpha": {"binary": "effective_num", "multiclass": "none"}}
    return arrays, ["f" + str(i) for i in range(40)], names, metadata


def save_fixture(path, mutate=None, rehash=True):
    module = prepared_module()
    arrays, features, classes, metadata = fixture_payload()
    if mutate:
        mutate(arrays, features, classes, metadata)
    metadata["payload_sha256"] = module.payload_digest(arrays, features, classes, metadata) if rehash else "0" * 64
    np.savez_compressed(path, **arrays, feature_names=np.asarray(features), class_names=np.asarray(classes),
                        metadata=np.asarray(json.dumps(metadata)))
    return arrays, metadata


def test_load_prepared_roundtrip_and_offline_import(tmp_path):
    path = tmp_path / "split_43.npz"
    arrays, metadata = save_fixture(path)
    result = prepared_module().load_prepared(path)
    assert set(result) == {"arrays", "feature_names", "class_names", "metadata"}
    assert result["metadata"] == metadata
    assert set(result["arrays"]) == {k for k in arrays if not k.startswith("split_")}
    for key, value in result["arrays"].items():
        np.testing.assert_array_equal(value, arrays[key])
    code = "import sys; import exlightnet.prepared; assert not any(k in sys.modules for k in ('pandas','sklearn','torch','exlightnet.preprocess','exlightnet.run_all'))"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("mutation,match", [
    (lambda a,f,c,m: a.__setitem__("x_train", a["x_train"].astype(np.float64)), "float32"),
    (lambda a,f,c,m: a["x_val"].__setitem__((0,0), np.nan), "finite"),
    (lambda a,f,c,m: a.__setitem__("y12_test", a["y12_test"][:-1]), "shape"),
    (lambda a,f,c,m: a["y12_train"].__setitem__(0, 1), "coverage"),
    (lambda a,f,c,m: a["y2_val"].__setitem__(0, 0), "mapping"),
    (lambda a,f,c,m: a["split_val"].__setitem__(0, 0), "partition"),
    (lambda a,f,c,m: c.__setitem__(0, "unknown_class"), "class"),
    (lambda a,f,c,m: m["settings"].__setitem__("split_seed", 44), "fingerprint"),
    (lambda a,f,c,m: m.__setitem__("split_seed", 44), "split_seed"),
    (lambda a,f,c,m: m["settings"].__setitem__("top_k", 20), "recipe"),
])
def test_load_rejects_invalid_content_even_with_fresh_payload_hash(tmp_path, mutation, match):
    path = tmp_path / "bad.npz"
    save_fixture(path, mutation)
    with pytest.raises(ValueError, match=match):
        prepared_module().load_prepared(path)


def test_load_rejects_payload_corruption_and_non_npz(tmp_path):
    path = tmp_path / "bad.npz"
    save_fixture(path, rehash=False)
    with pytest.raises(ValueError, match="payload"):
        prepared_module().load_prepared(path)
    path.write_bytes(b"not an npz")
    with pytest.raises(ValueError):
        prepared_module().load_prepared(path)


def test_default_split_preserves_original_indices_and_general_seeds():
    from exlightnet.data import EXPECTED_COUNTS, make_split
    labels = np.concatenate([np.repeat(k, n) for k, n in EXPECTED_COUNTS.items()])
    indices = np.arange(len(labels))
    train, rest = train_test_split(indices, test_size=.30, random_state=42, stratify=labels)
    val, test = train_test_split(rest, test_size=.50, random_state=42, stratify=labels[rest])
    original = dict(train=train, val=val, test=test)
    default, _ = make_split(labels)
    explicit, _ = make_split(labels, split_seed=42)
    for name in original:
        np.testing.assert_array_equal(default[name], original[name])
        np.testing.assert_array_equal(explicit[name], original[name])
    changed, gate = make_split(labels, split_seed=43)
    assert gate["pass"] and not np.array_equal(changed["train"], train)
    assert [len(changed[k]) for k in ("train", "val", "test")] == [86181, 18468, 18468]
    assert all(set(labels[v]) == set(EXPECTED_COUNTS) for v in changed.values())


def test_export_uses_only_training_rows_and_refuses_overwrite(tmp_path, monkeypatch):
    import pandas as pd
    from exlightnet import data
    from exlightnet.preprocess import FlowPreprocessor
    module = prepared_module()
    names = sorted(data.EXPECTED_COUNTS)
    labels = np.tile(names, 16)
    binary = (~np.isin(labels, ["thing_speak", "mqtt_publish", "wipro_bulb"])).astype(np.int64)
    split = {"train": np.arange(120), "val": np.arange(120,156), "test": np.arange(156,192)}
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(192,41)), columns=[f"f{i}" for i in range(41)])
    frame.iloc[120:] += 1000
    (tmp_path / "data").mkdir()
    (tmp_path / "data/RT_IOT2022.csv").write_text("test fixture, not raw data")
    (tmp_path / "artifacts").mkdir()
    settings = fixture_payload()[3]["settings"].copy()
    settings["split_seed"] = 42
    settings["data_sha256"] = hashlib.sha256((tmp_path / "data/RT_IOT2022.csv").read_bytes()).hexdigest()
    fp = recipe_digest(settings)
    (tmp_path / "artifacts/cache_manifest.json").write_text(json.dumps({"fingerprint":fp,"settings":settings}))
    (tmp_path / "results/full").mkdir(parents=True)
    selections = {t:{"chosen":v,"data_fingerprint":fp,"test_used":False,"sweep_max_epochs":8,"seed":42} for t,v in [("binary","effective_num"),("multiclass","none")]}
    (tmp_path / "results/full/alpha_selection.json").write_text(json.dumps(selections))
    monkeypatch.setattr(data, "load_data", lambda *a, **kw: (frame,labels,binary))
    monkeypatch.setattr(data, "validate_data", lambda y: {"pass":True})
    monkeypatch.setattr(data, "make_split", lambda labels,split_seed=42: (split,{"pass":True}))
    expected = FlowPreprocessor(40).fit(frame.iloc[:120], binary[:120])
    manifest = module.export_prepared(tmp_path, tmp_path / "prepared", [43])
    actual = module.load_prepared(tmp_path / "prepared/split_43.npz")
    assert manifest["splits"]["43"]["data_fingerprint"] == actual["metadata"]["data_fingerprint"]
    assert actual["feature_names"] == expected.selected_names_
    np.testing.assert_array_equal(actual["arrays"]["x_train"], expected.transform(frame.iloc[:120]))
    np.testing.assert_array_equal(actual["arrays"]["x_test"], expected.transform(frame.iloc[156:]))
    before = (tmp_path / "prepared/split_43.npz").read_bytes()
    with pytest.raises(FileExistsError):
        module.export_prepared(tmp_path, tmp_path / "prepared", [43])
    assert (tmp_path / "prepared/split_43.npz").read_bytes() == before


def test_real_split42_export_preserves_frozen_cache_without_fitting(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    if not (root / "artifacts/cache_arrays.npz").exists() or not (root / "data/RT_IOT2022.csv").exists():
        pytest.skip("requires the canonical local prepared cache and raw corpus")
    from exlightnet.preprocess import FlowPreprocessor
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in
              [*list((root / "artifacts").glob("*.*")), root / "results/data_gates.json"] if path.is_file()}
    monkeypatch.setattr(FlowPreprocessor, "fit", lambda *a, **kw: pytest.fail("split42 must never refit"))
    manifest = prepared_module().export_prepared(root, tmp_path / "prepared", [42])
    record = manifest["splits"]["42"]
    assert record["data_fingerprint"] == "dfba90bd116f9288bc72a837160516984ba773b11904e98561c80cda7dedcf5c"
    assert all(record["canonical_arrays_identical"].values())
    assert record["preprocessor_fitted"] is False
    loaded = prepared_module().load_prepared(tmp_path / "prepared/split_42.npz")
    with np.load(root / "artifacts/cache_arrays.npz", allow_pickle=False) as old:
        for key, value in loaded["arrays"].items():
            assert value.dtype == old[key].dtype
            assert value.tobytes() == old[key].tobytes()
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}


def test_load_binds_metadata_and_names_to_payload(tmp_path):
    path = tmp_path / "tampered.npz"
    save_fixture(path)
    with np.load(path, allow_pickle=False) as saved:
        values = {key:saved[key] for key in saved.files}
    metadata = json.loads(values["metadata"].item())
    metadata["frozen_alpha"]["multiclass"] = "effective_num"
    values["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="payload"):
        prepared_module().load_prepared(path)


def test_export_cli_rejects_descending_split_range_without_writing(tmp_path):
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts/export_prepared.py"), "--splits", "51-42", "--output", str(tmp_path / "prepared")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 2
    assert "split range" in result.stderr
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("field,value", [("split_seed",43.0), ("preprocessing_format",True), ("top_k",40.0)])
def test_load_rejects_noninteger_recipe_even_if_fingerprints_recomputed(tmp_path, field, value):
    def mutation(arrays, features, classes, metadata):
        metadata["settings"][field] = value
        metadata["data_fingerprint"] = recipe_digest(metadata["settings"])
    path = tmp_path / "bad_recipe.npz"
    save_fixture(path, mutation)
    with pytest.raises(ValueError, match="recipe|split_seed"):
        prepared_module().load_prepared(path)
