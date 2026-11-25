"""RT-IoT2022 loading, canonical labels and hard data gates."""
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

EXPECTED_COUNTS = {
    "dos_syn_hping": 94659, "thing_speak": 8108, "arp_poisioning": 7750,
    "mqtt_publish": 4146, "nmap_udp_scan": 2590, "nmap_xmas_tree_scan": 2010,
    "nmap_os_detection": 2000, "nmap_tcp_scan": 1002, "ddos_slowloris": 534,
    "wipro_bulb": 253, "metasploit_brute_force_ssh": 37, "nmap_fin_scan": 28,
}


class GateFailure(RuntimeError):
    def __init__(self, gate, details):
        self.gate = {"gate": gate, "pass": False, "actual": details}
        super().__init__(f"Gate {gate} failed: {details}. Dependent branch stopped.")


def normalize_frame(frame):
    frame = frame.copy()
    frame.columns = frame.columns.str.strip().str.lower()
    if "attack_type" not in frame:
        raise ValueError(f"Missing Attack_type label; real columns: {frame.columns.tolist()}")
    if frame["attack_type"].isna().any():
        raise GateFailure("D1", "missing labels")
    labels = frame.pop("attack_type").astype(str).str.strip().str.lower()
    labels = labels.where(~labels.str.startswith("wipro_bulb"), "wipro_bulb").to_numpy()
    frame = frame.drop(columns=[c for c in frame if c.startswith("unnamed:") or c == "index"])
    for c in ("proto", "service"):
        if c in frame:
            frame[c] = frame[c].astype("string").str.strip().str.lower().fillna("__missing__")
    benign = np.isin(labels, ["thing_speak", "mqtt_publish", "wipro_bulb"])
    return frame.reset_index(drop=True), labels, (~benign).astype(np.int64)


def load_data(path="data/RT_IOT2022.csv", download=True):
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}. Supply RT_IOT2022.csv or enable UCI ucimlrepo id 942 download.")
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=942)
            frame = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
        except Exception as exc:
            raise FileNotFoundError(
                f"Missing {path}; ucimlrepo id 942 download failed ({type(exc).__name__}: {exc}). "
                "Place the real RT_IOT2022.csv at that path. No synthetic substitute is used."
            ) from exc
    return normalize_frame(pd.read_csv(path, low_memory=False))


def validate_data(labels):
    counts = dict(Counter(labels))
    normal = int(sum(counts.get(k, 0) for k in ("thing_speak", "mqtt_publish", "wipro_bulb")))
    actual = {"rows": len(labels), "counts": counts, "normal": normal, "attack": len(labels) - normal}
    if counts != EXPECTED_COUNTS or normal != 12507 or len(labels) != 123117:
        raise GateFailure("D1", actual)
    return {"gate": "D1", "pass": True, "actual": actual}


def make_split(labels, split_seed=42):
    """Stratify 70/15/15; preserve the canonical D2 supports for seed42 only."""
    indices = np.arange(len(labels))
    train, rest = train_test_split(indices, test_size=.30, random_state=split_seed, stratify=labels)
    val, test = train_test_split(rest, test_size=.50, random_state=split_seed, stratify=labels[rest])
    split = {"train": train, "val": val, "test": test}
    counts = {k: dict(Counter(labels[v])) for k, v in split.items()}
    actual = {"sizes": {k: len(v) for k, v in split.items()}, "supports": counts}
    target = {"dos_syn_hping": 14199, "metasploit_brute_force_ssh": 5, "nmap_fin_scan": 4}
    if (list(actual["sizes"].values()) != [86181, 18468, 18468]
            or any(set(part_counts) != set(EXPECTED_COUNTS) for part_counts in counts.values())
            or (split_seed == 42 and any(counts["test"].get(k) != v for k, v in target.items()))):
        raise GateFailure("D2", actual)
    return split, {"gate": "D2", "pass": True, "actual": actual}
