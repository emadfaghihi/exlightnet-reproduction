"""Verified prepared arrays; loading needs only Python's stdlib and NumPy.

The historical data fingerprint describes the recipe, not the array bytes.
``payload_sha256`` additionally binds arrays, split indices, names and metadata.
Prepared data are local export artifacts and must not enter source deliveries.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np

PARTITIONS = ("train", "val", "test")
CLASS_NAMES = sorted(("dos_syn_hping", "thing_speak", "arp_poisioning", "mqtt_publish",
                      "nmap_udp_scan", "nmap_xmas_tree_scan", "nmap_os_detection", "nmap_tcp_scan",
                      "ddos_slowloris", "wipro_bulb", "metasploit_brute_force_ssh", "nmap_fin_scan"))
BENIGN = ("thing_speak", "mqtt_publish", "wipro_bulb")
ARRAY_KEYS = {prefix + part for prefix in ("x_", "y2_", "y12_") for part in PARTITIONS}
SPLIT_KEYS = {"split_" + part for part in PARTITIONS}


def _digest(value):
    # Preserve run_all.digest(settings), including its default JSON separators.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def payload_digest(arrays, feature_names, class_names, metadata):
    """Hash framed array bytes and JSON, independently of ZIP encoding/timestamps.

    All array members (including ``split_*``) participate, in sorted key order.
    Each member has a length-prefixed JSON [key, dtype.str, shape] header and
    length-prefixed C-order bytes. The final framed JSON contains feature names,
    class names and metadata except the self-referential payload_sha256 field.
    """
    value = hashlib.sha256()

    def frame(block):
        value.update(len(block).to_bytes(8, "big"))
        value.update(block)

    for key in sorted(arrays):
        array = np.asarray(arrays[key])
        frame(json.dumps([key, array.dtype.str, list(array.shape)], separators=(",", ":")).encode())
        frame(np.ascontiguousarray(array).tobytes())
    document = {"feature_names": list(feature_names), "class_names": list(class_names),
                "metadata": {k: v for k, v in metadata.items() if k != "payload_sha256"}}
    frame(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return value.hexdigest()


def _validate(arrays, features, classes, metadata):
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("Unsupported prepared metadata schema")
    settings = metadata.get("settings")
    if not isinstance(settings, dict) or set(settings) != {"data_sha256", "preprocessing_format", "split_seed", "mi", "top_k"}:
        raise ValueError("Invalid prepared recipe settings")
    if (any(type(settings[key]) is not int for key in ("preprocessing_format", "top_k", "split_seed"))
            or settings["preprocessing_format"] != 1 or settings["top_k"] != 40
            or settings["mi"] != "binary,3-neighbors,one-hot-discrete,seed42"):
        raise ValueError("Unsupported prepared preprocessing recipe")
    if not isinstance(settings["data_sha256"], str) or len(settings["data_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in settings["data_sha256"]):
        raise ValueError("Invalid raw data fingerprint")
    if metadata.get("data_fingerprint") != _digest(settings):
        raise ValueError("Prepared recipe fingerprint mismatch")
    if type(metadata.get("split_seed")) is not int or metadata["split_seed"] != settings["split_seed"] or not 0 <= metadata["split_seed"] < 2**32:
        raise ValueError("Prepared split_seed differs from recipe")
    alpha = metadata.get("frozen_alpha")
    if not isinstance(alpha, dict) or set(alpha) != {"binary", "multiclass"} or any(v not in ("inv_freq", "sqrt_inv_freq", "effective_num", "none") for v in alpha.values()):
        raise ValueError("Invalid frozen alpha selections")
    if classes != CLASS_NAMES:
        raise ValueError("Prepared class names/order differ from canonical class mapping")
    if len(features) != 40 or len(set(features)) != 40 or any(not isinstance(f, str) or not f for f in features):
        raise ValueError("Invalid prepared feature names")
    if set(arrays) != ARRAY_KEYS | SPLIT_KEYS:
        raise ValueError("Prepared arrays/partition indices missing or unexpected")
    indices = []
    binary_mapping = (~np.isin(classes, BENIGN)).astype(np.int64)
    for part in PARTITIONS:
        x, y2, y12, index = [arrays[prefix + part] for prefix in ("x_", "y2_", "y12_", "split_")]
        if x.dtype != np.dtype("float32"):
            raise ValueError(f"{part}: x must have float32 dtype")
        if x.ndim != 2 or x.shape[1] != 40 or x.shape[0] == 0:
            raise ValueError(f"{part}: invalid x shape")
        if not np.isfinite(x).all():
            raise ValueError(f"{part}: x must be finite")
        for name, y in (("y2", y2), ("y12", y12), ("partition index", index)):
            if y.dtype.kind not in "iu" or y.shape != (len(x),):
                raise ValueError(f"{part}: invalid {name} integer dtype or shape")
        if not np.array_equal(np.unique(y12), np.arange(12)) or not np.array_equal(np.unique(y2), [0, 1]):
            raise ValueError(f"{part}: every class must have coverage")
        if not np.array_equal(y2, binary_mapping[y12]):
            raise ValueError(f"{part}: binary/multiclass mapping differs")
        indices.append(index)
    all_indices = np.concatenate(indices)
    if not np.array_equal(np.sort(all_indices), np.arange(len(all_indices))):
        raise ValueError("Prepared partition indices overlap or fail complete coverage")
    if metadata.get("payload_sha256") != payload_digest(arrays, features, classes, metadata):
        raise ValueError("Prepared payload checksum mismatch")


def load_prepared(path):
    """Load and verify one NPZ without pandas, sklearn, torch or pickle.

    Return canonical arrays only; raw-row split indices remain available as
    ``split_train``, ``split_val`` and ``split_test`` in the NPZ for traceability.
    File-level SHA256s are recorded separately in the export manifest.
    """
    try:
        with np.load(Path(path), allow_pickle=False) as saved:
            expected = ARRAY_KEYS | SPLIT_KEYS | {"metadata", "feature_names", "class_names"}
            if len(saved.files) != len(set(saved.files)) or set(saved.files) != expected:
                raise ValueError("Prepared NPZ members missing, duplicated or unexpected")
            arrays = {key: saved[key] for key in ARRAY_KEYS | SPLIT_KEYS}
            for name in ("feature_names", "class_names"):
                if saved[name].ndim != 1 or saved[name].dtype.kind != "U":
                    raise ValueError(f"Invalid {name} string array")
            features, classes = saved["feature_names"].tolist(), saved["class_names"].tolist()
            if saved["metadata"].shape != () or saved["metadata"].dtype.kind != "U":
                raise ValueError("Prepared metadata must be scalar JSON text")
            metadata = json.loads(saved["metadata"].item())
        _validate(arrays, features, classes, metadata)
    except (OSError, KeyError, TypeError, zipfile.BadZipFile, EOFError) as exc:
        raise ValueError(f"Invalid prepared NPZ: {exc}") from exc
    return {"arrays": {key: arrays[key] for key in ARRAY_KEYS}, "feature_names": features,
            "class_names": classes, "metadata": metadata}


def _write_new_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def export_prepared(root, output, split_seeds):
    """Export new local arrays serially; never refit or rewrite canonical caches.

    Split42 reads and checks the frozen CPU cache. Other split seeds call the
    existing FlowPreprocessor.fit only on the respective training rows.
    Existing exports are refused; use a fresh output directory for a new export.
    """
    from .data import load_data, make_split, validate_data
    from .preprocess import FlowPreprocessor

    root, output = Path(root).resolve(), Path(output).resolve()
    seeds = list(split_seeds)
    if not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("split_seeds must be distinct nonnegative integer seeds")
    for protected in (root / "artifacts", root / "results", root / "data"):
        if output == protected or protected in output.parents:
            raise ValueError("Export output must be outside canonical evidence/data directories")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() or any((output / f"split_{seed}.npz").exists() for seed in seeds):
        raise FileExistsError("Prepared export already exists; preserve it and choose a new directory")
    started = time.perf_counter()
    directory = root / "artifacts"
    canonical = json.loads((directory / "cache_manifest.json").read_text())
    settings42 = canonical["settings"]
    if canonical["fingerprint"] != _digest(settings42) or settings42["split_seed"] != 42:
        raise ValueError("Canonical cache recipe fingerprint mismatch")
    raw_path = root / "data/RT_IOT2022.csv"
    if _sha256(raw_path) != settings42["data_sha256"]:
        raise ValueError("Raw data fingerprint differs from canonical data")
    alpha_path = root / "results/full/alpha_selection.json"
    selections = json.loads(alpha_path.read_text())
    for task in ("binary", "multiclass"):
        selection = selections[task]
        if selection["data_fingerprint"] != canonical["fingerprint"] or selection["test_used"] is not False or selection["sweep_max_epochs"] != 8 or selection["seed"] != 42:
            raise ValueError("Frozen alpha source is incompatible with cpu_v1")
    alpha = {task: selections[task]["chosen"] for task in ("binary", "multiclass")}
    frame, labels, binary = load_data(raw_path, download=False)
    d1 = validate_data(labels)
    classes, y12 = np.unique(labels, return_inverse=True)
    manifest = {"schema_version": 1, "source_campaign": "cpu_v1", "alpha_source": {
        "path": str(alpha_path.relative_to(root)), "sha256": _sha256(alpha_path),
        "data_fingerprint": canonical["fingerprint"]}, "frozen_alpha": alpha, "splits": {}}
    for seed in seeds:
        split_started = time.perf_counter()
        split, d2 = make_split(labels, split_seed=seed)
        equality = None
        if seed == 42:
            cache_path = directory / "cache_arrays.npz"
            if _sha256(cache_path) != canonical["arrays_sha256"]:
                raise ValueError("Canonical cached arrays checksum differs; export stopped")
            with np.load(directory / "split_indices.npz", allow_pickle=False) as saved:
                if any(not np.array_equal(split[p], saved[p]) for p in PARTITIONS):
                    raise ValueError("Canonical split42 indices differ")
            with np.load(cache_path, allow_pickle=False) as saved:
                if set(saved.files) != ARRAY_KEYS:
                    raise ValueError("Canonical array keys differ")
                arrays = {key: saved[key] for key in saved.files}
            state = json.loads((directory / "preprocessor.json").read_text())
            if classes.tolist() != json.loads((directory / "class_names.json").read_text()):
                raise ValueError("Canonical class mapping differs")
            features = state["selected_names_"]
            equality = {}
            for part in PARTITIONS:
                for prefix, y in (("y12_", y12), ("y2_", binary)):
                    if not np.array_equal(arrays[prefix + part], y[split[part]]):
                        raise ValueError("Canonical cached target alignment differs")
                equality.update({key: True for key in ARRAY_KEYS if key.endswith("_" + part)})
        else:
            pp = FlowPreprocessor(40).fit(frame.iloc[split["train"]], binary[split["train"]])
            state, features = pp.to_dict(), pp.selected_names_
            arrays = {}
            for part in PARTITIONS:
                index = split[part]
                arrays.update({"x_" + part: pp.transform(frame.iloc[index]), "y2_" + part: binary[index], "y12_" + part: y12[index]})
        settings = {**settings42, "split_seed": seed}
        metadata = {"schema_version": 1, "split_seed": seed, "settings": settings,
                    "data_fingerprint": _digest(settings), "frozen_alpha": alpha,
                    "alpha_source": manifest["alpha_source"], "fit_partition": "train",
                    "preprocessor": state, "preprocessor_sha256": _digest(state)}
        payload = {**arrays, **{"split_" + part: split[part] for part in PARTITIONS}}
        metadata["payload_sha256"] = payload_digest(payload, features, classes.tolist(), metadata)
        _validate(payload, features, classes.tolist(), metadata)
        path = output / f"split_{seed}.npz"
        temporary = path.with_suffix(".npz.partial")
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **payload, feature_names=np.asarray(features, dtype=str), class_names=np.asarray(classes, dtype=str),
                                metadata=np.asarray(json.dumps(metadata, allow_nan=False)))
        # Hard-link publishes atomically while refusing to overwrite an existing file.
        os.link(temporary, path)
        temporary.unlink()
        checked = load_prepared(path)
        if seed == 42:
            equality = {key: bool(checked["arrays"][key].dtype == arrays[key].dtype
                                and checked["arrays"][key].shape == arrays[key].shape
                                and checked["arrays"][key].tobytes() == arrays[key].tobytes()) for key in ARRAY_KEYS}
            if not all(equality.values()) or metadata["data_fingerprint"] != canonical["fingerprint"]:
                raise ValueError("Exported split42 differs from frozen CPU arrays")
        manifest["splits"][str(seed)] = {"path": path.name, "sha256": _sha256(path),
            "size_bytes": path.stat().st_size, "data_fingerprint": metadata["data_fingerprint"],
            "payload_sha256": metadata["payload_sha256"], "preprocessor_sha256": metadata["preprocessor_sha256"],
            "class_support": {part: np.bincount(arrays["y12_" + part], minlength=12).tolist() for part in PARTITIONS},
            "partition_sizes": {part: len(split[part]) for part in PARTITIONS},
            "class_names": classes.tolist(), "all_classes_present": True,
            "canonical_arrays_identical": equality, "preprocessor_fitted": seed != 42,
            "data_gates": [d1, d2], "seconds": time.perf_counter() - split_started}
    manifest["seconds"] = time.perf_counter() - started
    _write_new_json(output / "manifest.json", manifest)
    return manifest
