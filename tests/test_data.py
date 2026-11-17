from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exlightnet.data import EXPECTED_COUNTS, GateFailure, load_data, normalize_frame, make_split, validate_data


def test_labels_and_index_are_normalized():
    frame = pd.DataFrame({"Unnamed: 0": [0, 1, 2], "Attack_type": ["Thing_Speak", "Wipro_bulb_Dataset", "DOS_SYN_Hping"], "proto": ["TCP", "udp", "TCP"]})
    x, labels, binary = normalize_frame(frame)
    assert "unnamed: 0" not in x
    assert labels.tolist() == ["thing_speak", "wipro_bulb", "dos_syn_hping"]
    assert binary.tolist() == [0, 0, 1]


def test_wrong_corpus_fails_hard():
    with pytest.raises(GateFailure, match="D1"):
        validate_data(np.array(["thing_speak"]))


def test_exact_split():
    labels = np.concatenate([np.repeat(k, n) for k, n in EXPECTED_COUNTS.items()])
    split, gate = make_split(labels)
    assert [len(split[k]) for k in ("train", "val", "test")] == [86181, 18468, 18468]
    assert set(split["train"]).isdisjoint(split["test"])
    assert gate["pass"]


def test_real_split_if_present():
    if not Path("data/RT_IOT2022.csv").exists():
        pytest.skip("RT-IoT2022 not downloaded")
    _, labels, _ = load_data(download=False)
    assert validate_data(labels)["pass"]
    assert make_split(labels)[1]["pass"]


def test_missing_data_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="942"):
        load_data(tmp_path / "missing.csv", download=False)
