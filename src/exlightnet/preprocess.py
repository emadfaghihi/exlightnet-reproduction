"""Every learned statistic is fitted exclusively on the training partition."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif

from .io import read_json, write_json

TABLE4 = """fwd_pkts_payload.avg fwd_pkts_payload.max fwd_subflow_bytes flow_pkts_payload.tot flow_pkts_payload.std fwd_pkts_payload.tot flow_pkts_payload.avg flow_pkts_payload.max flow_iat.min bwd_subflow_bytes bwd_pkts_payload.tot bwd_pkts_payload.avg bwd_pkts_payload.max flow_iat.tot flow_iat.max flow_duration active.max active.tot active.avg fwd_iat.min payload_bytes_per_second active.min fwd_pkts_per_sec flow_pkts_per_sec fwd_iat.max bwd_pkts_per_sec fwd_iat.tot bwd_pkts_payload.std flow_iat.avg fwd_iat.avg bwd_iat.tot bwd_iat.max fwd_header_size_tot bwd_header_size_tot bwd_iat.avg flow_iat.std id.resp_p bwd_header_size_max bwd_header_size_min bwd_data_pkts_tot""".split()


class FlowPreprocessor:
    def __init__(self, top_k=40):
        self.top_k = top_k

    @staticmethod
    def _category(series):
        return series.astype("string").str.strip().str.lower().fillna("__missing__")

    def _raw_encode(self, frame):
        missing = set(self.numeric_columns_ + self.categorical_columns_) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing input features: {sorted(missing)}")
        numeric = frame[self.numeric_columns_].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
        values = numeric.fillna(self.medians_).to_numpy(dtype=np.float64)
        blocks = [values]
        for col in self.categorical_columns_:
            v = self._category(frame[col]).to_numpy()
            blocks.append(np.column_stack([(v == k).astype(float) for k in self.vocabulary_[col]]))
        return np.column_stack(blocks)

    def fit(self, train, y_binary):
        self.categorical_columns_ = [c for c in ("proto", "service") if c in train]
        self.numeric_columns_ = [c for c in train if c not in self.categorical_columns_]
        numeric = train[self.numeric_columns_].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
        self.all_missing_ = numeric.columns[numeric.isna().all()].tolist()
        self.medians_ = numeric.median().fillna(0).to_dict()
        self.vocabulary_ = {c: sorted(self._category(train[c]).unique().tolist()) for c in self.categorical_columns_}
        names = self.numeric_columns_ + [f"{c}_{v}" for c in self.categorical_columns_ for v in self.vocabulary_[c]]
        encoded = self._raw_encode(train)
        self.candidate_count_before_pruning_ = len(names)
        self.keep_indices_ = np.flatnonzero(np.ptp(encoded, axis=0) > 0).tolist()
        self.dropped_constant_ = [n for i, n in enumerate(names) if i not in self.keep_indices_]
        self.candidate_names_ = [names[i] for i in self.keep_indices_]
        if not 1 <= self.top_k <= len(self.candidate_names_):
            raise ValueError(f"top_k={self.top_k}, but only {len(self.candidate_names_)} nonconstant candidates")
        encoded = encoded[:, self.keep_indices_]
        discrete = np.array([i >= len(self.numeric_columns_) for i in self.keep_indices_])
        scores = mutual_info_classif(encoded, y_binary, discrete_features=discrete, n_neighbors=3, random_state=42, n_jobs=1)
        order = np.argsort(-scores, kind="stable")
        self.ranking_ = [{"rank": rank + 1, "feature": self.candidate_names_[i], "mi": float(scores[i])} for rank, i in enumerate(order)]
        self.selected_indices_ = order[:self.top_k].tolist()
        self.selected_names_ = [self.candidate_names_[i] for i in self.selected_indices_]
        self.mi_scores_ = scores[order].tolist()
        selected = encoded[:, self.selected_indices_]
        self.mean_ = selected.mean(axis=0)
        self.scale_ = selected.std(axis=0, ddof=0)
        self.scale_[self.scale_ == 0] = 1.
        return self

    def encode(self, frame):
        return self._raw_encode(frame)[:, self.keep_indices_]

    def transform(self, frame):
        result = (self.encode(frame)[:, self.selected_indices_] - self.mean_) / self.scale_
        if not np.isfinite(result).all():
            raise ValueError("Non-finite transformed inputs")
        return result.astype(np.float32)

    def to_dict(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.__dict__.items()}

    def save(self, directory):
        directory = Path(directory)
        write_json(directory / "preprocessor.json", self.to_dict())
        write_json(directory / "mi_ranking.json", self.ranking_)
        write_json(directory / "mi_comparison.json", self.compare_table4())

    @classmethod
    def load(cls, directory):
        obj = cls()
        obj.__dict__.update(read_json(Path(directory) / "preprocessor.json"))
        obj.mean_ = np.asarray(obj.mean_)
        obj.scale_ = np.asarray(obj.scale_)
        return obj

    def compare_table4(self):
        shared = [n for n in TABLE4 if n in self.selected_names_]
        rho = float(spearmanr([TABLE4.index(n) for n in shared], [self.selected_names_.index(n) for n in shared]).statistic) if len(shared) > 1 else None
        return {"gate": "P1", "report_only": True, "overlap_count": len(shared), "budget": self.top_k,
                "spearman_shared_ranks": rho, "shared_features": shared,
                "candidate_count_before_pruning": self.candidate_count_before_pruning_,
                "candidate_count_after_pruning": len(self.candidate_names_), "dropped_constant": self.dropped_constant_,
                "mi_settings": {"target": "binary", "n_neighbors": 3, "random_state": 42, "discrete_features": "one-hot only", "ties": "stable candidate order"}}
