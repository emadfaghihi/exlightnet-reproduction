"""Read-only publication contracts for cached statistics and execution status."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .io import read_json

STATISTICS_PATH = Path("artifacts/stats_cpu_v1")
TABLE_SPECS = (
    ("model_summary", "Seed summaries", ["task", "model", "n_seeds", "sample_label", "macro_f1_mean", "macro_f1_sd", "macro_f1_seed42", "zero_seed_variance"],
     "All requested seeds contribute; sample SD uses ddof=1. Seed 42 is separate. Zero observed seed variance is flagged."),
    ("paired_tests", "Paired comparisons", ["task", "reference", "model", "reference_device", "comparison_device", "device_label", "n_pairs", "sample_label", "mean_difference", "difference_sd", "ci_low", "ci_high", "p_value", "cohens_dz", "p_holm", "interpretation"],
     "Differences are ExLightNet minus the comparison on matched seeds. Two-sided paired t tests, 95% t intervals and signed Cohen's dz use the paired differences. Holm adjustment includes all comparisons within each task, including ablations and W8. Wilcoxon is not used below six pairs. A constant model vector reduces the paired test to a one-sample test against that constant."),
    ("flow_bootstrap", "Flow bootstrap uncertainty", ["task", "model", "macro_f1_mean", "ci_low", "ci_high", "pairing", "replicates", "n_seeds", "sample_label", "estimand"],
     "Each replicate selects a training seed uniformly and resamples test flows. The estimand is single-random-run plus flow uncertainty; this is not a confidence interval for the seed mean. The pairing/method column identifies cached-prediction or UNPAIRED confusion-cell resampling."),
    ("paired_flow_bootstrap", "Paired flow bootstrap differences", ["task", "reference", "model", "mean_difference", "ci_low", "ci_high", "pairing", "replicates", "n_seeds", "sample_label", "estimand"],
     "PAIRED means both models use identical resampled flows from the same matched training seed; an UNPAIRED fallback cannot support a paired flow comparison. The estimand remains a randomly selected training run plus finite-flow uncertainty."),
    ("per_class_f1", "Per-class F1 and test support", ["task", "model", "class_name", "test_support", "f1_mean", "f1_sd", "f1_seed42", "n_seeds", "sample_label"],
     "Every class metric is accompanied by its fixed test support. Mean/sample-SD and seed-42 F1 are computed from cached predictions/confusion matrices."),
    ("sensitivity", "Post-hoc sensitivity", ["task", "model", "macro_f1_mean", "macro_f1_sd", "macro_f1_seed42", "n_seeds", "sample_label", "excluded_classes", "min_support", "label"],
     "post-hoc sensitivity, not a replacement for macro-F1. Low-support classes are excluded from the macro average only; their flows still contribute false positives to retained classes."),
    ("training_dynamics", "Training dynamics", ["task", "model", "seed", "n_seeds", "sample_label", "macro_f1", "epochs_run", "best_epoch", "max_epochs", "hit_cap", "stopped_early", "cosine_lr_factor", "unstable", "early_best_epoch", "high_seed_sd"],
     "Persisted epoch counts and best-validation epochs define cap/stopping indicators. The cosine factor is evaluated at the last epoch using the recorded maximum epoch budget; instability is flagged rather than used to claim superiority."),
    ("correlations", "Epoch/F1 correlations", ["task", "model", "n", "sample_label", "spearman_rho", "p_value", "interpretation", "members"],
     "Spearman correlations, including the pooled within-model z-scored convolutional family, are descriptive only, not causal. The stopping/cosine explanation remains a hypothesis for the intervention study in Stage B."),
    ("binary_caps", "Binary epoch caps", ["task", "model", "n_runs", "n_epoch_observed", "n_hit_cap", "n_best_at_cap", "sample_label"],
     "Counts describe the recorded binary epoch cap and best-validation epoch; no uncapped local experiment was run."),
    ("per_seed_macro_f1", "Per-seed observations", ["task", "model", "seed", "n_seeds", "sample_label", "macro_f1", "accuracy", "epochs_run", "best_epoch"],
     "All persisted seed observations are shown; missing values remain unavailable and no best seed is headlined."),
)


def load_statistics(root, statistics_dir=None):
    """Read generated CPU tables, verifying their manifest and source identity."""
    root = Path(root)
    directory = Path(statistics_dir) if statistics_dir is not None else root / STATISTICS_PATH
    manifest_path = directory / "statistics_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = read_json(manifest_path)
    if (manifest.get("campaign"), manifest.get("device")) != ("cpu_v1", "cpu"):
        raise ValueError("Canonical presentation requires cpu_v1 CPU statistics")
    for basename in ("results.json", "manifest.json"):
        current = root / "results/full" / basename
        recorded = [sha for source, sha in manifest.get("source_sha256", {}).items()
                    if str(source).endswith("/results/full/" + basename)]
        if current.exists() and recorded and hashlib.sha256(current.read_bytes()).hexdigest() not in recorded:
            raise ValueError("Statistics source fingerprint differs from the canonical evidence")
    tables = {}
    for name, _, columns, _ in TABLE_SPECS:
        entry = manifest.get("tables", {}).get(name)
        if not entry or Path(entry["path"]).name != entry["path"]:
            raise ValueError(f"Missing/unsafe statistics manifest entry: {name}")
        path = directory / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Statistics table checksum mismatch: {name}")
        frame = pd.read_csv(path)
        if any(col not in frame for col in columns):
            raise ValueError(f"Statistics table columns incomplete: {name}")
        if not frame.empty and (set(frame.campaign) != {"cpu_v1"} or ("device" in frame and set(frame.device) != {"cpu"})):
            raise ValueError("Mixed campaigns/devices cannot enter a CPU publication")
        tables[name] = frame
    return tables


def rare_class_warning(root, rows=None, tables=None):
    """Derive rare-class names and supports from persisted evidence."""
    tables = load_statistics(root) if tables is None else tables
    supports = {}
    if "per_class_f1" in tables:
        frame = tables["per_class_f1"]
        for item in frame[(frame.task == "multiclass") & (frame.test_support < 10)].itertuples():
            supports[item.class_name] = int(item.test_support)
    else:
        for row in rows or []:
            if row.get("task") == "multiclass":
                for name, metric in row.get("per_class", {}).items():
                    if metric.get("support", 10) < 10:
                        supports[name] = int(metric["support"])
    if not supports:
        return "Multiclass macro-F1 is sensitive to rare classes; inspect the test-support column before interpreting comparisons."
    detail = ", ".join(f"{name} (test n={count})" for name, count in sorted(supports.items()))
    return f"Tiny multiclass test supports make macro-F1 comparisons noisy: {detail}. Training seeds do not measure split or campaign variability."


def statistic_sections(root, tables=None):
    tables = load_statistics(root) if tables is None else tables
    return [{"name": name, "title": title, "description": description,
             "source": str(STATISTICS_PATH / (name + ".csv")), "frame": tables[name][columns].copy()}
            for name, title, columns, description in TABLE_SPECS if name in tables]


def execution_status(root, mode="full"):
    """Explain historical pipeline flags using current and archived receipts."""
    directory = Path(root) / "results" / mode
    def saved(name, default):
        path = directory / name
        return read_json(path) if path.exists() else default
    manifest = saved("manifest.json", {})
    pipeline = saved("parallel_execution.json", {})
    failures = saved("failures.json", manifest.get("failures", []))
    benchmark = saved("benchmark_manifest.json", {})
    recovery = saved("benchmark_recovery.json", {})
    archived = []
    for path in sorted((directory / "benchmarks").glob("*/manifest.json")):
        record = read_json(path)
        if record.get("campaign") == benchmark.get("campaign"):
            continue
        for failure in record.get("failures", []):
            if failure.get("error_type") == "IdleCPUUnavailable":
                archived.append({"campaign": record.get("campaign"), "source": str(path.relative_to(Path(root))), **failure})
    current_records = failures + manifest.get("failures", []) + benchmark.get("failures", [])
    active = len({json.dumps(record, sort_keys=True) for record in current_records})
    all_failures = bool(current_records)
    complete_benchmark = (benchmark.get("attempted_models", 0) > 0
        and benchmark.get("measured_models") == benchmark.get("attempted_models")
        and all(benchmark.get(key, 0) == 0 for key in ("failed_models", "skipped_models", "gated_out_models")))
    complete_training = bool(manifest.get("full_five_seed_headlines_complete"))
    raw = pipeline.get("status", "not recorded")
    if all_failures:
        effective, explanation = "complete_with_stopped_branches", "Current persisted failures remain active; inspect the branch records."
    elif complete_training and complete_benchmark:
        if raw == "complete_with_stopped_branches":
            try:
                later = datetime.fromisoformat(benchmark["finished_at"]) > datetime.fromisoformat(pipeline["finished_at"])
            except (KeyError, ValueError, TypeError):
                later = False
            recovered = (archived and later and recovery.get("status") == "complete"
                and recovery.get("campaign") == benchmark.get("campaign") and recovery.get("returncode") == 0
                and not recovery.get("failures"))
            if recovered:
                effective, explanation = "complete", "The raw pipeline flag preserves its initial CPU-idle timeout. Archived idle failures remain intact; a later successful benchmark recovery completed all current measurements with no active failures."
            else:
                effective, explanation = "requires_review", "The historical pipeline flag lacks sufficient archived idle-failure and later successful recovery evidence; completion is not inferred."
        else:
            effective, explanation = "complete", "Current training and isolated benchmark manifests are complete with no active failures."
    else:
        effective = "not_run" if not manifest else "incomplete"
        explanation = "Required completed training/benchmark evidence is missing; no completion is inferred."
    return {"effective_status": effective, "raw_pipeline_status": raw, "active_failure_count": active,
            "archived_idle_failure_count": len(archived), "latest_benchmark_campaign": benchmark.get("campaign"),
            "latest_benchmark_measured_models": benchmark.get("measured_models"), "explanation": explanation,
            "archived_idle_failures": archived}


def limitation_paragraphs(root, rows=None, tables=None):
    tables = load_statistics(root) if tables is None else tables
    return [
        "The class-stratified flow split is fixed. The five canonical seeds vary initialization and sampling, not the split; n=5, indicative only. Related attack-run flows can cross partitions because session/campaign identifiers are absent. A clustering PROXY is not campaign-disjoint validation.",
        rare_class_warning(root, rows, tables),
        "The relation between epochs run and macro-F1 is descriptive. The dated stopping-rule hypothesis uses predeclared primary and secondary endpoints; even a supported result never establishes causality. See the effective declaration and Stage B report.",
        "The persisted training-dynamics table flags std_cnn instability (early best-validation epochs or large seed SD). This baseline cannot establish that ExLightNet is superior. Zero observed seed variance, including XGBoost under the paper configuration, is flagged in the generated summary; its paired test remains a test of the paired differences.",
        "PyTorch eager batch-one latency includes framework dispatch overhead. It is not pure compute cost or hardware-independent performance. The dispatch fraction is NOT MEASURED. Exported/frozen-model latency is NOT RUN; no such benchmark is inferred from the eager timings.",
        "Supplementary GPU accuracy/stopping/split studies are separate from cpu_v1; CUDA was NOT EXECUTED here. CPU/GPU device-equivalence views are descriptive and cannot pool campaigns or assert a pass/fail equivalence gate.",
    ]


def stopping_hypothesis_declaration(root):
    """Load the dated effective text; never reinterpret it from measured scores."""
    path = Path(root) / 'docs/statistics_protocol.json'
    if not path.exists():
        return None
    protocol = read_json(path)
    amendment = next((item for item in protocol.get('amendments', [])
                      if item['id'] == 'PA-20261118-1'), None)
    if amendment is None:
        return None
    return {'id': amendment['id'], 'date': amendment['date'], 'rules': amendment['rules_verbatim'],
            'original_source': 'docs/protocol_declarations/2026-11-17-original.json',
            'effective_source': 'docs/statistics_protocol.json', 'results_source': 'REPORT_stats_v2.md'}


def validate_publication_text(text):
    """Reject stale reduced-run captions outside explicitly labelled sections."""
    stack = []
    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)", line)
        if heading:
            level, title = len(heading[1]), heading[2]
            stack = [(n, fast) for n, fast in stack if n < level]
            stack.append((level, "fast demonstration" in title.lower()))
        if re.search(r"\bfast[ -]mode\b", line, re.IGNORECASE) and not any(fast for _, fast in stack):
            raise ValueError("Stale caption outside an explicitly labelled Fast demonstration section")


def validate_notebook_content(notebook, mode="full"):
    if mode != "full":
        return
    text = []
    for cell in notebook.cells:
        text.append(cell.source)
        for output in cell.get("outputs", []):
            if output.get("text"):
                text.append(str(output["text"]))
            for key in ("text/plain", "text/html", "text/markdown"):
                value = output.get("data", {}).get(key)
                if value:
                    text.append("".join(value) if isinstance(value, list) else str(value))
    validate_publication_text("\n".join(text))


def active_scope_legacy_flags(root):
    """Scan active sources/presentations/new statistics, excluding immutable history."""
    root = Path(root)
    candidates = []
    names = ["src", "scripts", "tests", "notebooks", "artifacts/final_analysis"]
    names += [str(path.relative_to(root)) for path in (root / "artifacts").glob("stats_*") if path.is_dir()]
    names += [str(path.relative_to(root)) for path in (root / "artifacts").glob("predata_*/pending_studies") if path.is_dir()]
    for name in names:
        candidates.extend(path for path in (root / name).rglob("*") if path.is_file()
                          and path.suffix in (".py", ".md", ".ipynb", ".json", ".csv"))
    if (root / "REPORT.md").exists():
        candidates.append(root / "REPORT.md")
    if (root / "REPORT_stats_v2.md").exists():
        candidates.append(root / "REPORT_stats_v2.md")
    pattern = re.compile(r"\b" + "exceeds" + r"_[A-Za-z0-9_]*_std\b")
    found = []
    for path in sorted(set(candidates)):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            for match in pattern.finditer(line):
                found.append({"path": str(path.relative_to(root)), "line": number, "identifier": match[0]})
    return found
