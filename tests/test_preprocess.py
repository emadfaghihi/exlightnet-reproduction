import copy

import numpy as np
import pandas as pd

from exlightnet.preprocess import FlowPreprocessor


def fixture():
    return pd.DataFrame({"value": [1., 3., np.inf, 7., 9., 11., 13., 15.], "other": [7., 8., 9., 10., 11., 12., 13., 14.], "bwd_urg_flag_count": [0.] * 8, "empty": [np.nan] * 8, "proto": ["tcp", "udp"] * 4, "service": ["http", "mqtt"] * 4})


def test_train_only_imputation_scaling_and_unseen_categories(tmp_path):
    x = fixture()
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    pp = FlowPreprocessor(top_k=3).fit(x, y)
    before = copy.deepcopy(pp.to_dict())
    assert pp.medians_["value"] == 9.
    assert "bwd_urg_flag_count" in pp.dropped_constant_
    assert "empty" in pp.dropped_constant_
    assert np.allclose(pp.transform(x).mean(axis=0), 0, atol=1e-6)
    assert np.allclose(pp.transform(x).std(axis=0), 1, atol=1e-6)
    heldout = x.copy()
    heldout["value"] = -999999.
    heldout["proto"] = "unseen"
    heldout["service"] = "new-service"
    encoded = pp.encode(heldout)
    names = pp.candidate_names_
    assert np.all(encoded[:, [i for i, n in enumerate(names) if n.startswith("proto_") or n.startswith("service_")]] == 0)
    assert np.isfinite(pp.transform(heldout)).all()
    assert pp.to_dict() == before
    pp.save(tmp_path)
    restored = FlowPreprocessor.load(tmp_path)
    np.testing.assert_allclose(restored.transform(x), pp.transform(x))
    assert pp.mi_scores_ == sorted(pp.mi_scores_, reverse=True)


def test_heldout_values_cannot_change_fitted_statistics():
    x = fixture()
    pp = FlowPreprocessor(top_k=3).fit(x, np.arange(8) % 2)
    candidate = pp.encode(x)
    np.testing.assert_allclose(pp.mean_, candidate[:, pp.selected_indices_].mean(axis=0))
    np.testing.assert_allclose(pp.scale_, candidate[:, pp.selected_indices_].std(axis=0))
    mean = pp.mean_.copy()
    for value in (-1e10, 1e10):
        heldout = x.copy()
        heldout["value"] = value
        pp.transform(heldout)
        np.testing.assert_array_equal(pp.mean_, mean)
