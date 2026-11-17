"""Generate tables and plots from actual persisted evidence, never paper targets."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .io import read_json
from .metrics import summarize
from .paper import PAPER_REPORTED


def md_table(headers, rows):
    def clean(value):
        if value is None:
            return "—"
        if isinstance(value, dict):
            value = "; ".join(f"{k}: {clean(v)}" for k, v in value.items())
        elif isinstance(value, (list, tuple)):
            value = ", ".join(clean(v) for v in value)
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(clean(v) for v in row) + " |" for row in rows])


def key_values(mapping, prefix=""):
    rows = []
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            rows.extend(key_values(value, name))
        else:
            rows.append([name, value])
    return rows


def _failure_table(records):
    return md_table(["Branch", "Type", "Reason"], [[r["branch"], r["error_type"], r["reason"]] for r in records])


def _int8_tables(root, directory, rows):
    from .quantize import int8_size_breakdown
    output = ["### INT8 size breakdown", "",
              "Weights-only means the raw INT8 Conv1d/Linear tensor payload. The full NPZ also contains per-channel FP32 scales, FP32 BN/bias state, metadata, NPY headers and the ZIP container; only the full-artifact column is an on-disk file size.", ""]
    records = []
    components = []
    for row in rows:
        if row["model"] != "exlightnet":
            continue
        path = directory / row["task"] / "exlightnet" / str(row["seed"]) / "model_int8.npz"
        if not path.exists():
            continue
        size = int8_size_breakdown(path)
        records.append([row["task"], row["seed"], row["serialized_bytes"], size["weights_only_int8_bytes"], size["scales_fp32_bytes"], size["rest_fp32_bytes"], size["full_artifact_bytes"]])
        if row["seed"] == 42:
            for label, raw_key, compressed_key in (("INT8 weights", "weights_only_int8_bytes", "weights"), ("FP32 scales", "scales_fp32_bytes", "scales"),
                                                    ("FP32 rest", "rest_fp32_bytes", "rest"), ("Metadata", "metadata_array_bytes", "metadata")):
                components.append([row["task"], label, size[raw_key], size["compressed_members_bytes"][compressed_key]])
            components.append([row["task"], "ZIP container", None, size["zip_container_bytes"]])
            components.append([row["task"], "Full artifact", size["total_array_payload_bytes"], size["full_artifact_bytes"]])
    output += [md_table(["Task", "Seed", "FP32 file bytes", "INT8 weights-only bytes", "FP32 scale bytes", "FP32 rest bytes", "Full INT8 artifact bytes"], records), "",
               "Seed-42 component accounting (compressed member sizes include each NPY header; the full-artifact total also includes the ZIP container):", "",
               md_table(["Task", "Component", "Raw array payload bytes", "Compressed members/container bytes"], components), ""]
    return output


def paper_comparison(rows):
    summary = summarize(rows)
    records = []
    for _, row in summary.iterrows():
        quoted = PAPER_REPORTED.get(row.task, {}).get(row.model, {})
        records.append({"task": row.task, "model": row.model, "n_seeds": row.n_seeds,
                        "ours_mean_macro_f1": row.macro_f1_mean, "ours_sample_std": row.macro_f1_std,
                        "ours_seed42_macro_f1": row.macro_f1_seed42, "paper_quoted_macro_f1": quoted.get("macro_f1")})
    return pd.DataFrame(records)


def make_result_plots(root, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(root) / "results" / mode
    plotdir = directory / "plots"
    plotdir.mkdir(parents=True, exist_ok=True)
    for task in ("binary", "multiclass"):
        run_dir = directory / task / "exlightnet/42"
        if not (run_dir / "metrics.json").exists():
            continue
        result = read_json(run_dir / "metrics.json")
        cm = np.array(result["confusion_matrix"])
        names = list(result["per_class"])
        normalized = cm / cm.sum(1, keepdims=True)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, values, title in zip(axes, (cm, normalized), ("Raw counts", "Recall by true class")):
            ax.imshow(values, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
            ax.set_yticks(range(len(names)), names, fontsize=7)
            ax.set(xlabel="Predicted class", ylabel="True class", title=title)
            if len(names) == 2:
                for (i, j), v in np.ndenumerate(values):
                    ax.text(j, i, str(int(v)) if title == "Raw counts" else f"{v:.3f}", ha="center", va="center")
        fig.suptitle(f"{mode}: {task}, seed 42")
        fig.tight_layout()
        fig.savefig(plotdir / f"{task}_confusion.png", dpi=150)
        plt.close(fig)
        if (run_dir / "epochs.csv").exists():
            history = pd.read_csv(run_dir / "epochs.csv")
            fig, axes = plt.subplots(1, 2, figsize=(9, 3))
            axes[0].plot(history.epoch, history.train_loss)
            axes[0].set(xlabel="Epoch", ylabel="Training focal loss")
            axes[1].plot(history.epoch, history.val_macro_f1)
            axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1")
            fig.tight_layout()
            fig.savefig(plotdir / f"{task}_training.png", dpi=150)
            plt.close(fig)


def build_report(root=Path(".")):
    root = Path(root)
    lines = ["# ExLightNet reproduction report", "", f"Generated from saved evidence: {datetime.now(ZoneInfo('Asia/Tehran')).isoformat()}", "",
             "The supplied 31-page PDF was fully extracted with `pdftotext -layout` and read before implementation. The supplied brief is authoritative where it conflicts with the paper. All measured results below come from code. Quoted reference metrics appear only in `src/exlightnet/paper.py::PAPER_REPORTED` and comparison tables.", "",
             "## Execution status", ""]
    if not (root / "results/full/manifest.json").exists():
        lines += ["Only the reduced fast experiment has been executed. Full five-seed training, multi-seed ablation evidence, the full-size SHAP analysis and the optional leakage PROXY remain unexecuted; no full-recipe performance is claimed.", ""]
    if (root / "results/pytest.xml").exists():
        import xml.etree.ElementTree as ET
        suites = ET.parse(root / "results/pytest.xml").getroot().iter("testsuite")
        totals = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        for suite in suites:
            for key in totals:
                totals[key] += int(suite.get(key, 0))
        lines += ["Pytest verification from results/pytest.xml. Third-party SHAP/Matplotlib deprecation warnings are retained in results/pytest.log.", "",
                  md_table(["Tests", "Failures", "Errors", "Skipped"], [[totals[k] for k in ("tests", "failures", "errors", "skipped")]]), ""]
    history = root / "results/notebook_execution_history.json"
    if history.exists():
        lines += ["Notebook verification history (wall-clock time includes initial live training; later invocations may reuse caches):", "",
                  md_table(["Mode", "Seconds", "Executed cells", "Error outputs", "Caches present at start"],
                           [[r["mode"], f"{r['seconds']:.3f}", r["executed_code_cells"], r["error_outputs"], r.get("cached_results_present_at_start", "initial live run")] for r in read_json(history)]), ""]
    for mode in ("fast", "full"):
        directory = root / "results" / mode
        if not (directory / "manifest.json").exists() or read_json(directory / "manifest.json").get("status") == "blocked":
            lines += [f"**{mode}: not executed to completion.** No missing results or seed standard deviations are filled in.", ""]
            if (directory / "gates.json").exists():
                evidence = read_json(directory / "gates.json")
                lines += [md_table(["Gate", "Status", "Actual evidence"], [[g["gate"], "REPORT ONLY" if g.get("report_only") else "PASS" if g.get("pass") else "FAIL", g.get("actual", g)] for g in evidence]), ""]
            if (directory / "failures.json").exists():
                lines += ["Stopped branches:", "", _failure_table(read_json(directory / "failures.json")), ""]
            if (directory / "alpha_selection.json").exists():
                selection = read_json(directory / "alpha_selection.json")
                lines += ["Completed validation-only alpha selections (full model seeds have not yet been aggregated):", "",
                          md_table(["Task", "Chosen mode", "Sweep epochs"], [[task, value["chosen"], value["sweep_max_epochs"]] for task, value in selection.items()]), ""]
            if (directory / "alpha_sweep_manifest.json").exists():
                sweep = read_json(directory / "alpha_sweep_manifest.json")
                lines += [md_table(["Alpha sweep wall seconds", "Processes", "Threads", "Test used"], [[sweep["seconds"], sweep["processes"], sweep["threads"], sweep["test_used"]]]), ""]
            if (directory / "parallel_execution.json").exists():
                state = read_json(directory / "parallel_execution.json")
                fields = {k: state[k] for k in ("status", "waiting_for", "available_ram_mb", "required_ram_mb", "started_at", "finished_at", "reason") if k in state}
                lines += ["Current parallel execution status (last saved observation):", "", md_table(["Field", "Value"], key_values(fields)), ""]
            continue
        manifest = read_json(directory / "manifest.json")
        lines += [f"**{mode}:** seeds {manifest['seeds']}; {manifest['elapsed_seconds_this_invocation']:.2f} seconds for the recorded invocation. Full five-seed headline matrix complete: {manifest['full_five_seed_headlines_complete']}. Stopped branches: {len(manifest['failures'])}.", ""]
        gates = read_json(directory / "gates.json")
        lines += ["### Gates", "", md_table(["Gate", "Task/seed", "Status", "Actual evidence"], [
            [g["gate"], f"{g.get('task', '')} {g.get('seed', '')}", "REPORT ONLY" if g.get("report_only") else "PASS" if g.get("pass") else "FAIL",
             g.get("actual", {k: v for k, v in g.items() if k not in ("shared_features", "gate")})] for g in gates]), ""]
        rows = read_json(directory / "results.json")
        table = paper_comparison(rows)
        lines += ["### Measured macro-F1 and quoted paper comparison", "",
                  "Standard deviation is the sample SD (ddof=1); a single seed has no estimated SD. Fast mode is a demonstration with reduced training and SHAP, not a five-seed reproduction. Seed 42 is always shown separately; no best seed is selected.", "",
                  md_table(["Task", "Model", "Seeds", "Our mean", "Our SD", "Our seed 42", "Paper (quoted)"],
                           [[r.task, r.model, r.n_seeds, f"{r.ours_mean_macro_f1:.6f}", None if pd.isna(r.ours_sample_std) else f"{r.ours_sample_std:.6f}",
                             None if pd.isna(r.ours_seed42_macro_f1) else f"{r.ours_seed42_macro_f1:.6f}", None if pd.isna(r.paper_quoted_macro_f1) else r.paper_quoted_macro_f1] for r in table.itertuples()]), "",
                  "All accuracy, macro precision/recall, weighted F1, macro F1, AUC, per-class support, confusion matrices, parameter counts, sizes and latency observations are in the per-run JSON files and `summary.csv`. Binary JSON additionally records attack recall, misses, false positives and FPR at the fixed argmax threshold.", ""]
        selections = read_json(directory / "alpha_selection.json")
        lines += ["### Measured size and latency (seed 42)", "",
                  md_table(["Task", "Model", "Parameters", "Serialized bytes", "Mean ms", "Median ms", "p95 ms"],
                           [[r["task"], r["model"], r.get("params"), r.get("serialized_bytes"), r.get("latency_mean_ms"), r.get("latency_median_ms"), r.get("latency_p95_ms")] for r in rows if r["seed"] == 42]), ""]
        lines += _int8_tables(root, directory, rows)
        lines += ["### Alpha selection", "", "Selection uses seed 42 validation macro-F1 only, separately per task, then freezes the selected mode for every deep model, seed and ablation. The small sweep is intentionally capped and does not optimize paper agreement.", "",
                  md_table(["Task", "Chosen", "Sweep epochs"], [[t, v["chosen"], v["sweep_max_epochs"]] for t, v in selections.items()]), "",
                  md_table(["Task", "Alpha mode", "Validation macro-F1", "Best epoch", "Selected"],
                           [[t, c["mode"], c["validation_macro_f1"], c["best_epoch"], c["mode"] == v["chosen"]] for t, v in selections.items() for c in v["candidates"]]), ""]
        ablations = read_json(directory / "ablations.json")
        lines += ["### Ablations", "",
                  md_table(["Variant", "Paired seeds", "Delta macro-F1", "Paired delta SD", "Full-model SD", "Exceeds full SD", "Exceeds paired SD"],
                           [[r[k] for k in ("variant", "paired_seeds", "delta_macro_f1", "paired_delta_std", "full_seed_std", "exceeds_full_seed_std", "exceeds_paired_seed_std")] for r in ablations]), "",
                  "Differences are compared with both full-model seed SD and paired difference SD when multiple seeds exist. This is descriptive variation, not a statistical significance claim. Full mode uses five seeds; fast mode uses one and cannot establish whether a difference exceeds seed variation.", ""]
        edir = directory / "explain/explanations.json"
        if edir.exists():
            explanation = read_json(edir)
            methods = [("FP32", explanation["fp32"], len(explanation["local_examples"])), ("Dequantised INT8", explanation["dequantized_int8"], 0)]
            lines += ["### Explainability evidence", "", "Background and random test examples use a fixed seed. FP32 additionally includes the local examples listed below.", "",
                      md_table(["Artifact", "Method/status", "Train backgrounds", "Random test flows", "Extra local flows", "Max abs. additivity error", "Mean abs. additivity error"],
                               [[label, e.get("method", e.get("status")), explanation["background_size"], explanation["random_test_size"], extras, e.get("additivity_max_abs"), e.get("additivity_mean_abs")] for label, e, extras in methods]), "",
                      md_table(["Artifact", "Fallback reason"], [[label, e.get("fallback_reason")] for label, e, _ in methods]), "",
                      "Deletion probabilities:", "", md_table(["Removed features", "Mean original predicted-class probability"], [[k, v] for k, v in explanation["deletion"]["probability_after_removals"].items()]), "",
                      md_table(["SHAP deletion-AUC", "Random deletion-AUC", "Random permutations"], [[explanation["deletion"]["normalized_auc"], explanation["deletion"]["random_normalized_auc"], explanation["deletion"]["random_permutations"]]]), "",
                      md_table(["Quantisation stability measure", "Value"], key_values(explanation["quantization_stability"])), "",
                      "Local examples (signed logit contributions, with correctness stated):", "",
                      md_table(["True class", "Predicted class", "Correct", "Probability", "Test position"], [[r[k] for k in ("true_class", "predicted_class", "correct", "probability", "test_position")] for r in explanation["local_examples"]]), "",
                      md_table(["Example true class", "Rank", "Feature", "Signed logit contribution", "z-value"], [[r["true_class"], i + 1, c["feature"], c["signed_logit_phi"], c["z_value"]] for r in explanation["local_examples"] for i, c in enumerate(r["top5"])]), "",
                      "Attributions explain logits; deletion measures the probability of the original predicted class. Additivity residuals are measured without correction. Deletion curves need not be monotone; neither causality nor guaranteed faithfulness is assumed. W8 uses identical background/test examples and GradientExplainer randomness for a controlled ranking probe.", ""]
            if explanation["fp32"]["method"] == "GradientExplainer":
                lines += ["DeepExplainer failed its additivity check and the prescribed GradientExplainer fallback was used. The residuals above show that these reduced-sample attributions do not establish exact additive efficiency; treat the fast explanation figures as approximate diagnostics.", ""]
        environment = read_json(directory / "environment.json")
        lines += ["### Hardware and runtime", "", md_table(["Field", "Value"], key_values(environment)), "",
                  "Latency uses one library thread, batch one, 100 warmups and 1000 timed inferences on training examples. Mean, median, p95 and peak RSS are saved per run. Inline measurements include preceding training in their process-lifetime RSS; isolated benchmark-only measurements include preceding benchmarks, with no training in that process. Neither is incremental model memory. No multi-core throughput extrapolation is reported. Raspberry Pi measurements require running `scripts/bench_pi.py` on the device; this host is not a Pi.", ""]
        runtime = []
        if (directory / "alpha_sweep_manifest.json").exists():
            alpha = read_json(directory / "alpha_sweep_manifest.json")
            runtime.append(["Alpha sweep", alpha["seconds"], 1, 1, alpha.get("started_at"), alpha.get("finished_at")])
        for worker in sorted((directory / "workers").glob("*/manifest.json")):
            w = read_json(worker)
            runtime.append([f"Seed {w['seeds'][0]} worker", w["elapsed_seconds_this_invocation"], 1, w.get("threads"), w.get("started_at"), w.get("finished_at")])
        if (directory / "parallel_execution.json").exists():
            pipeline = read_json(directory / "parallel_execution.json")
            runtime.append(["Recorded parallel pipeline", pipeline.get("seconds"), "up to 5 workers", 1, pipeline.get("started_at"), pipeline.get("finished_at")])
            for invocation in pipeline.get("invocations", []):
                if invocation["stage"] == "aggregate_only" and invocation.get("finished_at"):
                    elapsed = (datetime.fromisoformat(invocation["finished_at"]) - datetime.fromisoformat(invocation["started_at"])).total_seconds()
                    runtime.append(["Cache-only aggregation", elapsed, 1, 1, invocation["started_at"], invocation["finished_at"]])
        if (directory / "interrupted").exists() and (directory / "supervisor.json").exists():
            lines += ["The recorded parallel pipeline resumed after an interrupted interactive invocation, reusing ten completed results and the frozen alpha selections. Its wall time covers the detached resumed invocation. Earlier logs and state are preserved under `results/full/interrupted/`; the alpha sweep was not repeated.", ""]
        if (directory / "benchmark_manifest.json").exists():
            b = read_json(directory / "benchmark_manifest.json")
            runtime.append(["Final isolated benchmark", b["seconds"], b["processes"], b["threads"], b["started_at"], b["finished_at"]])
            lines += ["The benchmark-only process runs separately from training and explanations. Successful measurements have consecutive CPU-idle checks recorded in their benchmark context. RSS is the benchmark process lifetime peak across previously loaded models, with no training in that process.", ""]
            lines += [md_table(["Benchmark attempted", "Successful", "Failed", "Skipped after prerequisite failure", "Gated out"],
                               [[b.get("attempted_models"), b["measured_models"], b.get("failed_models"), b.get("skipped_models"), b.get("gated_out_models")]]), ""]
        if (directory / "benchmark_recovery.json").exists():
            recovery = read_json(directory / "benchmark_recovery.json")
            lines += ["Benchmark recovery after the initial CPU-idle timeout (the stopped campaign remains in `results/full/benchmarks/`):", "",
                      md_table(["Status", "Waiting started (UTC)", "Benchmark started (UTC)", "Finished (UTC)", "Last CPU busy fraction"],
                               [[recovery.get(k) for k in ("status", "started_at", "benchmark_started_at", "finished_at", "last_cpu_busy_fraction")]]), ""]
        if runtime:
            lines += [md_table(["Stage", "Wall seconds", "Processes per stage/worker", "Threads per process", "Started (UTC)", "Finished (UTC)"], runtime), "",
                      "Seed workers overlap in wall time, so their elapsed times must not be summed as end-to-end runtime. Aggregate-only publication reuses final probabilities and test receipts.", ""]
        if manifest["failures"]:
            lines += ["### Stopped branches", "", md_table(["Branch", "Type", "Reason"], [[r["branch"], r["error_type"], r["reason"]] for r in manifest["failures"]]), ""]
    lines += ["## Implementation and assumptions", "",
              "Implemented: real-data loading and hard gates; fixed class-stratified partition; train-fitted preprocessing/MI; exact ExLightNet; four alpha alternatives and validation-only sweep; shared deterministic training; five external baselines; metrics and evaluation receipts; five architectural/training/feature ablations; per-channel W8 exports and float evaluation; single-thread timing; standalone NumPy inference; SHAP with measured fallback/additivity/deletion/local/stability checks; notebook generation and evidence-based reporting. Full-mode code defaults to five seeds (42–46) for both tasks and all ablations. Optional class-clustering leakage proxy is implemented behind `--leakage-proxy`; it is never labelled campaign-disjoint.", "",
              "Not implemented: optional knowledge distillation, true INT8 arithmetic/ONNX Runtime, packet capture/Zeek extraction, or multi-process throughput measurement. Full-mode execution status above is separate from implementation status. No Pi hardware was available.", "",
              "Preprocessing: lower-case labels/columns and category strings; benign Wipro prefixes collapse to wipro_bulb; raw export index is removed. Numeric infinities become NaN, training medians fill them, and an entirely missing training column uses zero then is dropped as constant. proto/service one-hot vocabularies come from train only; unseen categories produce all-zero blocks. Constant pruning precedes binary-target MI. MI uses sklearn n_neighbors=3, random_state=42, discrete one-hot columns, continuous numeric columns and stable tie ordering. Selected features are sorted by descending computed MI; z-scores use population (ddof=0) training SD. The top-20 ablation takes the first 20 of the same ranking/scaler.", "",
              "Alpha formulas: 1/n_c; 1/sqrt(n_c); (1-beta)/(1-beta**n_c), beta=0.999; and all ones. Each vector is normalized to arithmetic mean one, then focal loss averages per-example alpha*(1-p_y)**2*(-log p_y), with no weighted-loss denominator. Sampling uses per-sample n_c**(-1/2); the fully balanced ablation uses n_c**(-1). Counts and frequencies differ by a common factor that cancels in both normalized schemes.", "",
              "Baselines: MLP 40–128–64–K, ReLU, biased Linear; one-layer bidirectional LSTM, input size 1, 32 hidden units/direction, final directional hidden states concatenated into a 64-wide head; Std-CNN widths 64/128/256, k=3, no padding, bias-free convolutions, BN/ReLU after each, global average pooling then biased Linear. Standard-convolution ablation replaces each depthwise+pointwise pair by a single no-padding k=3 convolution with the same channel widths and retains SE. Counts are measured, not fitted to paper totals. RF uses 300 balanced trees; XGBoost uses 400 depth-8 hist trees, balanced per-sample weights, CPU, one thread, otherwise pinned-library defaults (learning_rate=0.3). Tree parameter count is null because trainable neural parameters are not a meaningful equivalent; serialized bytes are measured. Official xgboost-cpu supplies the xgboost API without GPU libraries.", "",
              "Quantisation: Conv1d/Linear weights only, per-output-channel scale max(abs(W))/127 clamped at 1e-12, nearest-even rounding, clipping [-127,127]. BN state and all biases stay FP32; the non-inference BN counter is stored FP32 and restored to its integer dtype. Scales/rest/metadata contribute to the measured compressed NPZ size. Dequantisation occurs once at load; compute remains FP32 and no INT8 speedup is claimed. Q1 binary approximately-zero tolerance is declared as absolute macro-F1 delta <=0.001; multiclass drop <=0.01. Failed Q1 artifacts and metrics remain reviewable but their downstream deployment/stability branch stops.", "",
              "The test set receives one final probability pass per trained checkpoint (and one per derived W8 artifact). Receipts prevent silent reevaluation after interruption. Confusion matrices, tables and plots reuse cached predictions. Required SHAP and deletion queries are final interpretability analysis, with no model/epoch/alpha/threshold selection. DeepExplainer first checks additivity on two inputs to detect unsupported operators before full-set work; failure invokes the same declared GradientExplainer fallback, without changing background, explanation sample or GradientExplainer settings. Separate fast and full checkpoints have separate receipts.", "",
              "## Paper inconsistencies and limits", "",
              "The literal per-class sampler formula conflicts with the per-flow repetition narrative; the brief's per-sample interpretation is used. The sampler-disabled paper ablation does not show an accuracy gain from sampling. Figure 3 stage counts do not sum to its stated total. The Std-CNN descriptions conflict on widths and parameter counts. The same-width standard-convolution ablation count is reported from our graph, not adjusted to match the paper. One-hot candidate accounting is consistent (94 - 1 constant = 93). The prominent SHAP rate/header features occur in the lower half of the MI ranking: marginal binary MI and model-specific multi-class attribution answer different questions, and the actual overlap/ranks are reported rather than forced.", "",
              f"The paper's multiclass headline is its best of three seeds; its quoted three-seed mean/SD is {PAPER_REPORTED['exlightnet_multiclass_three_seed']['mean_macro_f1']} +/- {PAPER_REPORTED['exlightnet_multiclass_three_seed']['std_macro_f1']}. Our summaries use all requested seeds and never select a best seed. Some paper precision/recall columns track weighted values/accuracy; our precision/recall are explicitly macro-averaged as requested. SHAP efficiency is classwise in logit space, not a sum across class outputs. Weight-only quantisation does not establish integer-compute acceleration. The paper's four-core Pi throughput is an extrapolation, and its cache explanation confuses instruction/data caches; neither assertion is reproduced as measured evidence here.", "",
              "The flat corpus has no campaign/session IDs. Random flow splitting can put related flows in multiple partitions and optimistically bias metrics and relative rankings. A class-clustering PROXY cannot establish campaign independence. Rare-class supports (four FIN scans, five SSH brute-force examples in test) impose substantial uncertainty; five training seeds do not estimate dataset/campaign variability. No clinical, encrypted-traffic, adversarial or cross-dataset generalisation claim follows from these runs.", "",
              "## Reproduction commands", "", "```bash", "python3 -m venv .venv", "source .venv/bin/activate", "python -m pip install -r requirements.txt", "python -m pip install --no-deps -e .", "pytest -q", "python -m exlightnet.run_all --mode fast", "python -m exlightnet.run_all --mode full --alpha-only --skip-benchmark", "python scripts/run_parallel.py --reuse-alpha", "# The pipeline runs five single-thread workers, then exactly these final stages:", "# python -m exlightnet.run_all --mode full --aggregate-only --skip-benchmark", "# python -m exlightnet.run_all --mode full --benchmark-only", "python scripts/make_notebook.py", "EXLIGHTNET_RUN_MODE=full python scripts/execute_notebook.py", "python scripts/package_deliverables.py", "# Optional, separately, after full results: python -m exlightnet.run_all --mode full --leakage-proxy --skip-benchmark", "# On a Raspberry Pi with NumPy only:", "python scripts/bench_pi.py results/full/multiclass/exlightnet/42/model_int8.npz", "```", "",
              "This workspace contains an empty read-only .git mount rather than a usable Git repository. Per-phase commits were unavailable; phase evidence is recorded in docs/progress.md. Network setup initially required escalation and retries; the CPU PyTorch wheel was obtained from the official S3 origin when the CDN timed out. Source data came from UCI ucimlrepo id 942. The archive excludes the dataset, virtual environment, cached feature arrays and every *.pkl tree checkpoint. Tree checkpoints remain local for benchmarking; an extracted archive displays their cached metrics but cannot re-benchmark absent model files.", "",
              "Primary implementation references: [UCI RT-IoT2022](https://archive.ics.uci.edu/dataset/942/rt-iot2022), [SHAP DeepExplainer](https://shap.readthedocs.io/en/stable/generated/shap.DeepExplainer.html), [official XGBoost CPU package](https://pypi.org/project/xgboost-cpu/2.1.4/).", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
