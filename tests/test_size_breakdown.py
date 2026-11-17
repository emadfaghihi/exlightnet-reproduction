import zipfile

import numpy as np

from exlightnet.quantize import int8_size_breakdown


def test_int8_breakdown_separates_raw_weights_from_full_file(tmp_path):
    path = tmp_path / "model.npz"
    np.savez_compressed(path, **{"q::a": np.zeros((3, 4), dtype=np.int8), "s::a": np.ones(3, dtype=np.float32),
                                 "f::bias": np.zeros(3, dtype=np.float32), "__meta__": np.array("{}")})
    r = int8_size_breakdown(path)
    assert r["weights_only_int8_bytes"] == 12
    assert r["scales_fp32_bytes"] == 12
    assert r["rest_fp32_bytes"] == 12
    assert r["full_artifact_bytes"] == path.stat().st_size
    assert r["full_artifact_bytes"] == sum(r["compressed_members_bytes"].values()) + r["zip_container_bytes"]
