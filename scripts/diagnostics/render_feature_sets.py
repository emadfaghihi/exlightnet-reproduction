"""Render the post-hoc comparison from saved measurements without any fitting."""

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

from .feature_sets import FLAGS


def compare(result):
    rows = []

    def add(item, metric, reference, actual, tolerance, digits=4, unit="", scores=None):
        delta = actual - reference
        suffix = " pp" if unit == "%" else ""
        observed = f"{actual:.{digits}f}{unit}"
        status = "Within" if abs(delta) <= tolerance + 1e-12 else "**Outside**"
        seed_summary = ("—", "—", "—")
        if scores is not None:
            seed_summary = (str(len(scores)), f"{stdev(scores.values()):.4f}", f"{scores[42]:.4f}")
        rows.append((str(item), metric, f"{reference:.{digits}f}{unit}", observed,
                     f"{delta:+.{digits}f}{suffix}", f"±{tolerance:g}{suffix}", *seed_summary, status))

    def add_f1(item, metric, reference, runs, class_index=None):
        scores = {
            r["seed"]: r["macro_f1"] if class_index is None else r["per_class_f1"][class_index]
            for r in runs
        }
        add(item, metric, reference, mean(scores.values()), 0.01, scores=scores)

    pair = result["pair"]
    support = sum(pair["validation_support"])
    for name, f1, overlap in (("F40", 0.467, 15.0), ("F93", 1.000, 0.0)):
        add_f1(1, f"Pair {name} validation macro-F1", f1, pair["sets"][name]["rows"])
        actual_pct = 100 * pair["sets"][name]["opposite_label_matches"] / support
        add(1, f"Pair {name} opposite-class vector overlap", overlap, actual_pct, 1, digits=2, unit="%")

    ranks = result["ranking"]
    auc = pair["train_separation_auc"]
    for flag, reference_auc, reference_rank in zip(FLAGS, (1.000, 1.000, 0.999), (60, 71, 82), strict=True):
        add(2, f"Train pair AUC, `{flag}`", reference_auc, auc[flag], 0.01)
        add(2, f"Binary-MI rank, `{flag}`", reference_rank, ranks["binary_mi"].index(flag) + 1, 3, digits=0)
    best = pair["best_f40_auc_feature"]
    add(2, f"Best train pair AUC in F40 (`{best}`)", 0.582, auc[best], 0.01)

    multi = result["multiclass"]
    for name, f1, sd in (("F40", 0.8725, 0.0031), ("F93", 0.9755, 0.0002)):
        runs = multi["sets"][name]
        add_f1(3, f"12-class {name} validation macro-F1", f1, runs)
        add(3, f"12-class {name} macro-F1 sample SD", sd, stdev(r["macro_f1"] for r in runs), 0.01)
    for name, f40, f93 in (
        ("nmap_os_detection", 0.470, 1.000),
        ("nmap_xmas_tree_scan", 0.467, 1.000),
        ("ddos_slowloris", 0.905, 0.973),
        ("wipro_bulb", 0.910, 0.946),
        ("metasploit_brute_force_ssh", 0.884, 0.923),
    ):
        index = multi["classes"].index(name)
        support = multi["validation_support"][index]
        for feature_set, reference in (("F40", f40), ("F93", f93)):
            add_f1(3, f"`{name}` {feature_set} F1 ({support} val)", reference, multi["sets"][feature_set], index)

    for flag, reference_rank in zip(FLAGS, (57, 67, 75), strict=True):
        add(4, f"12-class MI rank, `{flag}`", reference_rank, ranks["multiclass_mi"].index(flag) + 1, 3, digits=0)
    overlap = set(ranks["binary_mi"][:40]) & set(ranks["multiclass_mi"][:40])
    add(4, "Shared features in binary and 12-class MI top-40", 31, len(overlap), 2, digits=0)

    for target, reference_ranks, reference_f1 in (
        ("multiclass", (5, 8, 2), (0.9776, 0.9673, 0.9573)),
        ("binary", (66, 78, 91), (0.9381, 0.5940, 0.5748)),
    ):
        for flag, reference_rank in zip(FLAGS, reference_ranks, strict=True):
            add(5, f"{target} RF-importance rank, `{flag}`", reference_rank,
                ranks[f"{target}_rf"].index(flag) + 1, 3, digits=0)
        for width, reference in zip((40, 20, 8), reference_f1, strict=True):
            runs = result["ranked_multiclass"][target][str(width)]
            count = sum(flag in ranks[f"{target}_rf"][:width] for flag in FLAGS)
            add_f1(5, f"{target} RF top-{width}, 12-class macro-F1 ({count}/3 flags)", reference, runs)
    top8_count = sum(flag in ranks["multiclass_rf"][:8] for flag in FLAGS)
    add(5, "Flags in the 12-class RF top-8", 3, top8_count, 0, digits=0)
    return rows


def qualitative_checks(result):
    pair = result["pair"]
    auc = pair["train_separation_auc"]
    ranks = result["ranking"]
    f1_gains = [
        mean(r["macro_f1"] for r in sets["F93"]) - mean(r["macro_f1"] for r in sets["F40"])
        for sets in ({name: entry["rows"] for name, entry in pair["sets"].items()}, result["multiclass"]["sets"])
    ]
    return {
        "The three flags have train pair AUC ≥ 0.989 and exceed the best F40 AUC": all(
            auc[flag] >= 0.989 and auc[flag] > auc[pair["best_f40_auc_feature"]]
            for flag in FLAGS
        ) and set(FLAGS).isdisjoint(ranks["binary_mi"][:40]),
        "F93 raises pair and 12-class macro-F1 by more than 0.01 versus F40": min(f1_gains) > 0.01,
        "The 12-class RF top-40 contains all three flags and the binary RF top-40 contains zero": (
            set(FLAGS).issubset(ranks["multiclass_rf"][:40]) and set(FLAGS).isdisjoint(ranks["binary_rf"][:40])
        ),
    }


def render(result):
    data = result["data"]
    pair = result["pair"]
    checks = qualitative_checks(result)
    rows = compare(result)
    libraries = ", ".join(f"{name} {value}" for name, value in result["libraries"].items())
    supports = "\n".join(f"| `{name}` | {train} | {validation} |" for name, train, validation in zip(
        pair["classes"], pair["train_support"], pair["validation_support"], strict=True
    ))
    table = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    check_lines = "\n".join(f"- {'Holds' if passed else 'Fails'} — {text}." for text, passed in checks.items())
    overlap_counts = ", ".join(f"{name} {item['opposite_label_matches']}/{sum(pair['validation_support'])}"
                               for name, item in pair["sets"].items())
    phase2 = "Proceed to Phase 2." if all(checks.values()) else "Stop before Phase 2."
    return f"""# Feature-set post-hoc diagnostic

Exploratory comparison for hypothesis generation. All model and preprocessing
fits used train rows, and scores used the saved validation partition. The
RF validation study has five seeds (42–46, n=5, indicative only) in items 1 and
3 and three seeds (42–44) in item 5. Small validation classes, including
`nmap_fin_scan` and `metasploit_brute_force_ssh`, make macro-F1 noisy.

## Run record

- Started at `{result['started_utc']}`. Diagnostic wall time was
  {result['wall_seconds']:.2f} s using the CPU on `{result['platform']}`.
- Python {result['python']}, {libraries}. RF used `n_jobs={result['rf_n_jobs']}`.
- Canonical data fingerprint `{data['cpu_v1_cache_fingerprint']}`.
- CSV SHA-256 `{data['csv_sha256']}`.
- Split archive SHA-256 `{data['split_indices_sha256']}`.
- {data['train_rows']:,} train rows and {data['validation_rows']:,} validation rows overall.
- Candidate accounting is consistent ({data['candidate_count_before_pruning']} - {len(data['dropped_constant'])} constant = {len(data['candidate_names'])}).
  `FlowPreprocessor.encode` supplies the unscaled `{data['encoded_dtype']}` candidates.

| Pair class | Train support | Validation support |
| --- | ---: | ---: |
{supports}
| **Total** | **{sum(pair['train_support'])}** | **{sum(pair['validation_support'])}** |

An overlap counts a validation flow once when its entire vector equals any
opposite-class train vector. Observed counts were {overlap_counts}. AUC uses
Xmas as positive and reports `max(AUC, 1 - AUC)` to allow either threshold
direction. Both RF-importance lists use one train fit with seed 42 on F93.
The multiclass ranking fit also supplies the matching F93 seed-42 score.
Item 5 evaluates both sets of importance prefixes on the 12-class task. The
rankings used for item 5 are unfiltered.

## My number vs yours

The reference values come from the user's exploratory run with NumPy 2.5.3,
pandas 3.0.5 and scikit-learn 1.9.1. Deltas are yours minus mine. F1 summaries
use the mean and sample SD across the listed seeds.

| Item | Metric | My number | Yours | Delta | Tolerance | n seeds | Sample SD | Seed 42 | Check |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{table}

## Qualitative checks

{check_lines}

{phase2} F93 adds many columns alongside the three flags, so this
comparison measures the joint feature-set difference. Perfect separation by
the flags, if observed with F43, may be specific to this dataset and its
traffic generator.

## Commands

Run from the repository root with the pinned environment. The fitter emits
the [measurement record](measurements.json) to stdout and progress to stderr.
The renderer reads only that record.

```bash
diag_dir=$(mktemp -d)
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
  .venv/bin/python -B -u -m scripts.diagnostics.feature_sets \\
  --data data/RT_IOT2022.csv --split-indices artifacts/split_indices.npz \\
  --cache-manifest artifacts/cache_manifest.json --jobs {result['rf_n_jobs']} \\
  > "$diag_dir/measurements.json" 2> "$diag_dir/progress.log"
.venv/bin/python -B -m scripts.diagnostics.render_feature_sets \\
  "$diag_dir/measurements.json" > "$diag_dir/REPORT.md"
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    args = parser.parse_args()
    print(render(json.loads(args.measurements.read_text(encoding="utf-8"))), end="")


if __name__ == "__main__":
    main()
