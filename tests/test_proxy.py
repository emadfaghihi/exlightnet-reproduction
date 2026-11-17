import numpy as np
import pytest

from exlightnet.proxy import cluster_group_split


def test_proxy_keeps_clusters_whole_and_classes_present():
    rng = np.random.default_rng(42)
    x, y = [], []
    for c in range(2):
        for k in range(8):
            x.extend(rng.normal(k * 10, .01, size=(20, 3)))
            y.extend([c] * 20)
    split, groups, info = cluster_group_split(np.asarray(x), np.asarray(y))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        assert set(groups[split[a]]).isdisjoint(groups[split[b]])
    assert sorted(np.concatenate(list(split.values())).tolist()) == list(range(320))
    for idx in split.values():
        assert set(np.asarray(y)[idx]) == {0, 1}
    assert info["label"] == "PROXY; not campaign-disjoint"
