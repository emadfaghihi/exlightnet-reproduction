"""Notebook presentation helpers; experiments and metrics stay in package modules."""
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from .io import read_json
from .metrics import summarize
from .report import paper_comparison


def experiment(root, mode):
    from .run_all import run
    if mode == "fast":
        return run(root, mode="fast")
    if mode != "full":
        raise ValueError("RUN_MODE must be fast or full")
    path = Path(root) / "results/full/manifest.json"
    if not path.exists():
        raise FileNotFoundError("Full results absent. Run python -m exlightnet.run_all --mode full first; the full notebook only loads caches.")
    return read_json(path)


def show_task(root, mode, task):
    directory = Path(root) / "results" / mode
    rows = [r for r in read_json(directory / "results.json") if r["task"] == task and r["model"] == "exlightnet"]
    if not rows:
        print(f"No completed {task} ExLightNet runs; inspect failures.json.")
        return
    summary = summarize(rows)
    display(summary[["model", "n_seeds", "macro_f1_mean", "macro_f1_std", "macro_f1_seed42", "accuracy_mean", "macro_recall_mean", "auc_mean"]])
    row = next((r for r in rows if r["seed"] == 42), rows[0])
    print(f"Per-class table and operating point below use seed {row['seed']}.")
    display(pd.DataFrame(row["per_class"]).T)
    if task == "binary":
        display(pd.DataFrame([{k: row[k] for k in ("attack_recall", "miss_count", "fp_count", "fpr", "threshold")}]))
    path = directory / "plots" / f"{task}_confusion.png"
    if path.exists():
        display(Image(filename=str(path)))
    seed42 = summary.iloc[0].macro_f1_seed42
    seed42_text = "seed 42 unavailable" if seed42 is None else f"seed 42 is {seed42:.6f}"
    print(f"Across {len(rows)} seed(s), mean macro-F1 is {summary.iloc[0].macro_f1_mean:.6f}; {seed42_text}. "
          + ("A single seed cannot estimate variation." if len(rows) == 1 else f"Sample SD is {summary.iloc[0].macro_f1_std:.6f}."))


def show_baselines(root, mode):
    from .baselines import MODELS
    rows = [r for r in read_json(Path(root) / "results" / mode / "results.json") if r["model"] in MODELS]
    summary = summarize(rows)
    display(summary[["task", "model", "n_seeds", "macro_f1_mean", "macro_f1_std", "macro_f1_seed42", "five_seed_complete"]])
    print("Each model uses the same split and train-fitted forty features. " + ("These are single-seed fast demonstrations; multi-seed uncertainty is unavailable." if mode == "fast" else "Mean/sample-SD and seed-42 values use all completed runs; missing runs are flagged."))


def show_ablations(root, mode):
    records = read_json(Path(root) / "results" / mode / "ablations.json")
    display(pd.DataFrame(records))
    print("Deltas are variant minus full-model macro-F1 on paired seeds. " + ("One seed cannot establish whether differences exceed seed SD." if mode == "fast" else "The SD comparisons describe training variation and are not significance tests."))


def show_quantization(root, mode):
    gates = [g for g in read_json(Path(root) / "results" / mode / "gates.json") if g["gate"] == "Q1"]
    display(pd.DataFrame([{**{k: g[k] for k in ("task", "seed", "pass")}, **g["actual"]} for g in gates]))
    print("Artifact sizes and metric changes are measured. Weight-only quantisation retains floating-point compute and does not establish an INT8 compute speedup.")


def show_latency(root, mode):
    rows = read_json(Path(root) / "results" / mode / "results.json")
    display(pd.DataFrame([{k: r.get(k) for k in ("task", "model", "seed", "params", "serialized_bytes", "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "peak_rss_bytes")} for r in rows]))
    print("Measurements use one thread and batch one after 100 warmups, with 1000 timed inferences. Peak RSS is the whole process peak; no Raspberry Pi or multi-core throughput is inferred.")


def show_shap(root, mode):
    directory = Path(root) / "results" / mode / "explain"
    if not (directory / "explanations.json").exists():
        print("SHAP branch did not complete; inspect failures.json.")
        return
    result = read_json(directory / "explanations.json")
    display(pd.DataFrame([{k: result["fp32"][k] for k in ("method", "fallback_reason", "additivity_max_abs", "additivity_mean_abs")}]))
    for name in ("global_shap.png", "beeswarm.png", "deletion.png"):
        display(Image(filename=str(directory / name)))
    display(pd.DataFrame([result["quantization_stability"]]))
    display(pd.DataFrame(result["local_examples"]))
    print(f"SHAP explains FP32 logits using {result['background_size']} train backgrounds and {result['random_test_size']} random test flows. "
          f"Deletion-AUC is {result['deletion']['normalized_auc']:.6f}, versus {result['deletion']['random_normalized_auc']:.6f} for random ordering; lower means less retained original-class probability under this masking protocol.")
    if result["fp32"]["method"] == "GradientExplainer":
        print("DeepExplainer failed its additivity check. The fallback's displayed residuals limit additive-efficiency claims; these attributions are approximate diagnostics.")


def show_comparison(root, mode):
    display(paper_comparison(read_json(Path(root) / "results" / mode / "results.json")))
    print("Paper columns are labelled quotations from PAPER_REPORTED. Agreement with a paper number is not a training objective, and fast mode does not estimate full-recipe performance.")
