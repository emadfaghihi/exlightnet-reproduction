"""Notebook presentation helpers; experiments and metrics stay in package modules."""
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from .io import read_json
from .metrics import summarize
from .report import paper_comparison
from .publication import (execution_status, limitation_paragraphs, load_statistics,
                          rare_class_warning, statistic_sections, stopping_hypothesis_declaration)


def experiment(root, mode):
    """Read cached execution evidence; notebooks never launch a campaign."""
    if mode not in ("fast", "full"):
        raise ValueError("RUN_MODE must be fast or full")
    path = Path(root) / "results" / mode / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Cached {mode} results absent; the analysis notebook only reads completed evidence.")
    return read_json(path)


def _sample_label(count):
    return "n=5, indicative only" if count == 5 else f"n={count}"


def show_task(root, mode, task):
    directory = Path(root) / "results" / mode
    rows = [r for r in read_json(directory / "results.json") if r["task"] == task and r["model"] == "exlightnet"]
    if not rows:
        print(f"No completed {task} ExLightNet runs; inspect failures.json.")
        return
    summary = summarize(rows)
    print(_sample_label(len(rows)))
    print(rare_class_warning(root, rows))
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
    seed42_text = "seed 42 unavailable" if pd.isna(seed42) else f"seed 42 is {seed42:.6f}"
    print(f"Across {len(rows)} seed(s), mean macro-F1 is {summary.iloc[0].macro_f1_mean:.6f}; {seed42_text}. "
          + ("A single seed cannot estimate variation." if len(rows) == 1 else f"Sample SD is {summary.iloc[0].macro_f1_std:.6f}."))


def show_baselines(root, mode):
    from .baselines import MODELS
    rows = [r for r in read_json(Path(root) / "results" / mode / "results.json") if r["model"] in MODELS]
    summary = summarize(rows)
    display(summary[["task", "model", "n_seeds", "macro_f1_mean", "macro_f1_std", "macro_f1_seed42", "five_seed_complete"]])
    print("; ".join(sorted({_sample_label(int(n)) for n in summary.n_seeds})))
    print(rare_class_warning(root, rows))
    print("Each model uses the same split and train-fitted forty features. " + ("These are single-seed fast demonstrations; multi-seed uncertainty is unavailable." if mode == "fast" else "Mean/sample-SD and seed-42 values use all completed runs; missing runs are flagged."))


def show_ablations(root, mode):
    if mode != "full":
        print("NOT RUN: multi-seed paired ablations are unavailable for the cached demonstration.")
        return
    tables = load_statistics(root)
    if "paired_tests" not in tables:
        print("NOT RUN: generated paired statistics are unavailable.")
        return
    from .baselines import ABLATIONS
    frame = tables["paired_tests"]
    selected = frame[(frame.task == "multiclass") & frame.model.isin(ABLATIONS)]
    columns = ["reference", "model", "n_pairs", "sample_label", "mean_difference", "difference_sd", "ci_low", "ci_high", "p_value", "cohens_dz", "p_holm", "interpretation"]
    display(selected[columns])
    print("n=5, indicative only. ExLightNet minus variant on matched seeds; Holm correction uses the entire task family. Source: artifacts/stats_cpu_v1/paired_tests.csv.")
    print(rare_class_warning(root, tables=tables))


def show_statistics(root, mode):
    if mode != "full":
        print("NOT RUN: canonical CPU statistics apply to the full reproduction only.")
        return
    tables = load_statistics(root)
    if not tables:
        print("NOT RUN: generated cached statistics are unavailable.")
        return
    print("n=5, indicative only. All tables are read from generated CPU statistics files.")
    warning = rare_class_warning(root, tables=tables)
    for section in statistic_sections(root, tables):
        print(section["title"] + " — " + section["source"])
        print(section["description"])
        print(warning)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(section["frame"])


def show_execution_status(root, mode):
    status = execution_status(root, mode)
    display(pd.DataFrame([{key: value for key, value in status.items() if key != "archived_idle_failures"}]))
    if status["archived_idle_failures"]:
        display(pd.DataFrame(status["archived_idle_failures"]))


def show_limitations(root):
    for paragraph in limitation_paragraphs(root):
        print(paragraph)


def show_fast_demonstration(root):
    """Display optional historical reduced-run evidence in its labelled section."""
    path = Path(root) / "results/fast/results.json"
    if not path.exists():
        print("NOT RUN: no cached historical demonstration is available.")
        return
    rows = read_json(path)
    display(summarize(rows)[["task", "model", "n_seeds", "macro_f1_mean", "macro_f1_std", "macro_f1_seed42"]])
    print("Historical reduced training and SHAP evidence. A single seed cannot estimate seed variation; this scope remains separate from the full reproduction.")
    print(rare_class_warning(root, rows, tables={}))


def show_quantization(root, mode):
    gates = [g for g in read_json(Path(root) / "results" / mode / "gates.json") if g["gate"] == "Q1"]
    display(pd.DataFrame([{**{k: g[k] for k in ("task", "seed", "pass")}, **g["actual"]} for g in gates]))
    print(rare_class_warning(root))
    print("Artifact sizes and metric changes are measured. Weight-only quantisation retains floating-point compute and does not establish an INT8 compute speedup.")


def show_latency(root, mode):
    rows = read_json(Path(root) / "results" / mode / "results.json")
    display(pd.DataFrame([{k: r.get(k) for k in ("task", "model", "seed", "params", "serialized_bytes", "latency_mean_ms", "latency_median_ms", "latency_p95_ms", "peak_rss_bytes")} for r in rows]))
    print("Measurements use one thread and batch one after 100 warmups, with 1000 timed inferences. Peak RSS is the whole process peak; no Raspberry Pi or multi-core throughput is inferred. Eager batch-one latency includes framework dispatch overhead; it is not pure compute or hardware-independent. The dispatch fraction is NOT MEASURED.")


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


def show_final_analysis(root):
    from .final_presentation import load_final_analysis
    result = load_final_analysis(root)
    if result is None:
        print('NOT RUN/PENDING: stratified final analysis has no completed, verified evidence.')
        return
    manifest = result['manifest']
    attribution = manifest['attribution']
    print(manifest['label'])
    print('Per-class rankings target each true class’s own logit; the primary global ranking weights classes equally. '
          'The auxiliary all-output ranking preserves the earlier output-averaging definition. '
          'Deletion follows the original predicted class even when it is incorrect; this is final analysis, never selection.')
    display(pd.DataFrame([{'sample_size': manifest['sample_size'], 'background_size': manifest['background_size'],
                           **{key: attribution.get(key) for key in ('method', 'fallback_reason', 'warnings', 'additivity_max_abs', 'additivity_mean_abs', 'nsamples_gradient')}}]))
    print(f"Additivity residuals summarize {manifest['sample_size']} flows by {len(manifest['class_names'])} output logits. "
          'Separate output-logit summaries were not retained. These are approximate attributions, not exact SHAP.')
    print('Recorded warnings: ' + '; '.join(attribution.get('warnings', [])))
    for name in ('earlier_random300_class_composition', 'per_class_top10', 'balanced_global_ranking',
                 'balanced_global_all_outputs_ranking', 'early_deletion_k10'):
        print(name + '.csv')
        display(result['tables'][name])
    for name in ('balanced_global_ranking.png', 'per_class_top10.png', 'deletion_panels.png'):
        display(Image(filename=str(result['directory'] / name)))
    print('Random-minus-SHAP at k=10 is a descriptive probability contrast on the same flows. '
          'Small class supports and measured additivity residuals limit interpretation; '
          'no monotonicity, significance, causal effect or model-selection conclusion is implied.')


def show_stopping_hypothesis_declaration(root):
    declaration = stopping_hypothesis_declaration(root)
    if declaration is None:
        print('NOT DECLARED: dated stopping-hypothesis rules are unavailable.')
        return
    print(declaration['id'] + ' (' + declaration['date'] + ')')
    for rule in declaration['rules']:
        print(rule)
    print('Original declaration: ' + declaration['original_source'])
    print('Effective amendment: ' + declaration['effective_source'])
    print('Execution/completeness and both endpoints: ' + declaration['results_source'] +
          '; a declaration alone is not execution evidence.')


def show_comparison(root, mode):
    frame = paper_comparison(read_json(Path(root) / "results" / mode / "results.json"))
    frame["sample_label"] = frame.n_seeds.map(_sample_label)
    display(frame)
    print(rare_class_warning(root))
    print("Paper columns are labelled quotations from PAPER_REPORTED. Agreement with a paper number is not a training objective. Single-seed demonstrations do not estimate full-recipe performance.")
