import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from exlightnet.data import EXPECTED_COUNTS, load_data
from exlightnet.io import read_json, sha256, write_json
from exlightnet.prepared import PARTITIONS, payload_digest
from exlightnet.preprocess import FlowPreprocessor


@pytest.fixture
def canonical_inputs(tmp_path, monkeypatch):
    from exlightnet import feature_sets

    classes = sorted(EXPECTED_COUNTS)
    labels = np.tile(classes, 9)
    is_attack = ~np.isin(labels, ["thing_speak", "mqtt_publish", "wipro_bulb"])
    split = {part: indices for part, indices in zip(PARTITIONS, [np.arange(60), np.arange(60, 84), np.arange(84, 108)])}
    rng = np.random.default_rng(34)
    x = rng.normal(0, 0.1, (len(labels), 93))
    x[:, :90] += 3 * is_attack[:, None]
    x[60:, :] += 20  # A fit on the held-out rows would change the train statistics.
    columns = [f"feature_{i:02}" for i in range(90)] + list(feature_sets.FLAGS)
    csv = tmp_path / "data/RT_IOT2022.csv"
    csv.parent.mkdir()
    pd.DataFrame(x, columns=columns).assign(Attack_type=labels).to_csv(csv, index=False)
    frame, labels, binary = load_data(csv, download=False)
    preprocessor = FlowPreprocessor(40).fit(frame.iloc[split["train"]], binary[split["train"]])
    y12 = np.searchsorted(classes, labels)
    arrays = {}
    for part, indices in split.items():
        arrays.update({f"x_{part}": preprocessor.transform(frame.iloc[indices]),
                       f"y2_{part}": binary[indices], f"y12_{part}": y12[indices]})
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    np.savez_compressed(artifacts / "cache_arrays.npz", **arrays)
    np.savez_compressed(artifacts / "split_indices.npz", **split)
    write_json(artifacts / "preprocessor.json", preprocessor.to_dict())
    write_json(artifacts / "class_names.json", classes)
    settings = {"data_sha256": sha256(csv), "preprocessing_format": 1, "split_seed": 42,
                "mi": "binary,3-neighbors,one-hot-discrete,seed42", "top_k": 40}
    write_json(artifacts / "cache_manifest.json", {
        "settings": settings,
        "fingerprint": hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        "arrays_sha256": sha256(artifacts / "cache_arrays.npz"),
    })

    def make_synthetic_split(actual_labels, *, split_seed):
        assert split_seed == 42
        np.testing.assert_array_equal(actual_labels, labels)
        return split, {"gate": "D2", "pass": True}

    monkeypatch.setattr(feature_sets, "make_split", make_synthetic_split)
    monkeypatch.setattr(feature_sets, "validate_data", lambda labels: {"gate": "D1", "pass": True})
    return frame, y12, split, preprocessor, arrays


def test_export_roundtrips_93_columns_and_passes_only_train_to_selector(tmp_path, monkeypatch, canonical_inputs):
    from exlightnet import feature_sets

    frame, y12, split, preprocessor, canonical_arrays = canonical_inputs
    original_selector = feature_sets.select_feature_sets
    train = preprocessor.encode(frame.iloc[split["train"]])
    calls = []

    def select_train_only(train_x, train_y12, *metadata, **options):
        calls.append(len(train_x))
        assert train_x.dtype == np.float64
        np.testing.assert_array_equal(train_x, train)
        np.testing.assert_array_equal(train_y12, y12[split["train"]])
        return original_selector(train_x, train_y12, *metadata, **options)

    monkeypatch.setattr(feature_sets, "select_feature_sets", select_train_only)

    receipt = feature_sets.export_feature_set_inputs(tmp_path)

    directory = tmp_path / "experiments/feature_sets_v1/prepared"
    manifest = read_json(directory / "manifest.json")
    metadata = manifest["metadata"]
    npz = directory / "split_42.npz"
    assert set(path.name for path in directory.iterdir()) == {"split_42.npz", "manifest.json", "receipt.json"}
    assert manifest["npz"] == {"file": npz.name, "sha256": sha256(npz), "size_bytes": npz.stat().st_size}
    assert receipt == read_json(directory / "receipt.json")
    assert receipt["manifest_sha256"] == sha256(directory / "manifest.json")
    assert receipt["status"] == "complete"
    assert receipt["feature_sets"] == metadata["feature_sets"]
    with np.load(npz, allow_pickle=False) as saved:
        arrays = dict(saved)
    assert set(arrays) == {f"{prefix}_{part}" for prefix in ("x", "y2", "y12", "split") for part in PARTITIONS}
    assert manifest["payload_sha256"] == payload_digest(arrays, metadata["candidate_names"], metadata["class_names"], metadata)
    mean, scale = train.mean(axis=0), train.std(axis=0)
    assert metadata["standardization"] == {"mean": mean.tolist(), "scale": scale.tolist()}
    for part, indices in split.items():
        key = f"x_{part}"
        np.testing.assert_array_equal(arrays[f"split_{part}"], indices)
        for target in ("y2", "y12"):
            np.testing.assert_array_equal(arrays[f"{target}_{part}"], canonical_arrays[f"{target}_{part}"])
        assert arrays[key].dtype == np.float32
        assert arrays[key].shape == (len(indices), 93)
        expected = ((preprocessor.encode(frame.iloc[indices]) - mean) / scale).astype(np.float32)
        np.testing.assert_array_equal(arrays[key], expected)
        assert arrays[key][:, preprocessor.selected_indices_].tobytes() == canonical_arrays[key].tobytes()
        assert receipt["equality_guard"][key] == {"equal": True, "max_abs_difference": 0.0}
    for arm in metadata["feature_sets"].values():
        assert arm["feature_names"] == [metadata["candidate_names"][i] for i in arm["column_indices"]]
    with pytest.raises(FileExistsError, match="prepared output already exists"):
        feature_sets.export_feature_set_inputs(tmp_path)
    assert calls == [len(train)]
    assert receipt == read_json(directory / "receipt.json")


def test_f40_mismatch_stops_before_rf_or_payload_and_records_all_differences(tmp_path, monkeypatch, canonical_inputs):
    from exlightnet import feature_sets

    _, _, _, _, arrays = canonical_inputs
    arrays["x_val"][0, 0] += 8
    arrays["x_test"][0, 0] += 4
    path = tmp_path / "artifacts/cache_arrays.npz"
    np.savez_compressed(path, **arrays)
    manifest_path = tmp_path / "artifacts/cache_manifest.json"
    write_json(manifest_path, {**read_json(manifest_path), "arrays_sha256": sha256(path)})
    monkeypatch.setattr(feature_sets, "select_feature_sets", lambda *args, **kwargs: pytest.fail("Selector ran after guard failure"))

    with pytest.raises(ValueError, match="F40 exact equality failed"):
        feature_sets.export_feature_set_inputs(tmp_path)

    directory = tmp_path / "experiments/feature_sets_v1/prepared"
    assert sorted(path.name for path in directory.iterdir()) == ["receipt.json"]
    receipt = read_json(directory / "receipt.json")
    assert receipt["status"] == "STOPPED"
    guard = receipt["equality_guard"]
    assert guard["x_train"] == {"equal": True, "max_abs_difference": 0.0}
    for part, difference in (("val", 8), ("test", 4)):
        assert guard[f"x_{part}"]["equal"] is False
        assert guard[f"x_{part}"]["max_abs_difference"] == pytest.approx(difference, abs=5e-6)
