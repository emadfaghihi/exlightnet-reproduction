"""Cache-only campaign statistics and exact multinomial flow bootstraps.

``paired_summary(a, b)`` uses a-b, sample SD, two-sided Student t inference
and signed Cohen's dz. A constant comparator remains a valid one-sample
problem; a constant *difference* is a separate, explicitly flagged case.
``holm_adjust`` preserves input order and leaves unavailable p-values missing.
No Wilcoxon inference is used (five pairs cannot attain a two-sided p < .05).

Bootstrap helpers uniformly choose ONE training seed per replicate, then draw
N test flows with replacement. Their percentile intervals describe a single
random training run plus test-flow uncertainty, NOT the seed-average mean.
Multinomial confusion/joint-cell draws are exactly equivalent to resampling
the empirical flows, while avoiding repeated large prediction arrays.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


ESTIMAND = "single-random-run plus flow uncertainty; not CI for the seed mean"
CONV_FAMILY = ("exlightnet", "no_se", "standard_conv", "std_cnn")
TREE_MODELS = frozenset(("random_forest", "xgboost"))
MIXED_DEVICE_LABEL = "mixed-device within gpu_v2; trees are device-independent"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sample_label(n):
    return "n=5, indicative only" if n == 5 else f"n={n}"


def _validate_rows(rows):
    campaigns = {r.get("campaign") for r in rows}
    if len(campaigns) != 1 or None in campaigns:
        raise ValueError("Mixed or missing campaigns are forbidden")
    devices = {r.get("device", "cpu") for r in rows}
    model_devices = {}
    for row in rows:
        model_devices.setdefault((row["task"], row["model"]), set()).add(row.get("device", "cpu"))
    if any(len(values) != 1 for values in model_devices.values()):
        raise ValueError("Multiple devices for the same model's seed vector are forbidden")
    if len(devices) != 1:
        # Explicit user amendment: only gpu_v2 may compare unchanged CPU
        # trees with CUDA neural models; no general device-mixing override.
        allowed = (campaigns == {"gpu_v2"} and len(devices) == 2 and "cpu" in devices
                   and all(r.get("device", "cpu") == "cpu" if r["model"] in TREE_MODELS
                           else r.get("device", "cpu").startswith("cuda") for r in rows))
        if not allowed:
            raise ValueError("Mixed devices are forbidden except CUDA neural/CPU trees within gpu_v2")
    identities = [(r["task"], r["model"], r["seed"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate task/model/seed identity")
    for row in rows:
        if not np.isfinite(row["macro_f1"]):
            raise ValueError("Non-finite macro-F1")


def campaign_device_label(rows):
    """Validate a table and return its required device label, including R5 exception."""
    _validate_rows(rows)
    devices = {r.get("device", "cpu") for r in rows}
    return MIXED_DEVICE_LABEL if len(devices) > 1 else next(iter(devices))


def load_campaign_rows(results_path, manifest_path=None):
    """Read fresh copies; derive legacy cpu_v1 only from a full-run manifest.

    A legacy benchmark's timestamped ``benchmark.campaign`` is a timing-run
    identifier, not a scientific campaign. It is deliberately not used here.
    Existing JSON files and dictionaries are never mutated.
    """
    results_path = Path(results_path)
    rows = copy.deepcopy(json.loads(results_path.read_text()))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty list of per-seed results")
    legacy = any("campaign" not in r for r in rows)
    if legacy:
        if any("campaign" in r for r in rows):
            raise ValueError("Mixed tagged and legacy campaigns")
        manifest_path = Path(manifest_path or results_path.with_name("manifest.json"))
        if not manifest_path.exists():
            raise ValueError("Legacy cpu_v1 requires its full manifest")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("mode") != "full" or manifest.get("seeds") != list(range(42, 47)):
            raise ValueError("Legacy cpu_v1 requires the canonical full five-seed manifest")
        if manifest.get("benchmark", {}).get("environment", {}).get("cuda_used", False):
            raise ValueError("Legacy cpu_v1 manifest indicates CUDA")
        if any(r["seed"] not in manifest["seeds"] for r in rows):
            raise ValueError("Seed is absent from canonical manifest")
        for row in rows:
            row.update(campaign="cpu_v1", device="cpu", split_seed=42)
    _validate_rows(rows)
    return rows


def paired_summary(reference, comparison):
    """Pure matched-vector summary; differences are reference minus comparison.

    One observation has undefined SD, CI, p and dz. Constant zero differences
    have p=1/dz=0, constant nonzero differences p=0/dz=+/-inf and point CIs.
    These are degenerate limits, not evidence of broad population certainty.
    """
    a, b = np.asarray(reference, dtype=float), np.asarray(comparison, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a) or not np.isfinite([a, b]).all():
        raise ValueError("Paired finite, nonempty one-dimensional vectors required")
    d = a-b
    n = len(d)
    difference = float(d.mean())
    sa = float(a.std(ddof=1)) if n > 1 else np.nan
    sb = float(b.std(ddof=1)) if n > 1 else np.nan
    sd = float(d.std(ddof=1)) if n > 1 else np.nan
    # Tolerate only cancellation noise of a few float64 ULPs.
    zero = n > 1 and np.ptp(d) <= np.finfo(float).eps * 8 * max(1., np.abs(d).max())
    p = low = high = t = dz = np.nan
    if n > 1:
        if zero:
            sd = 0.
            low = high = difference
            t = dz = 0. if difference == 0 else float(np.copysign(np.inf, difference))
            p = 1. if difference == 0 else 0.
        else:
            se = sd / np.sqrt(n)
            t = difference/se
            p = float(2*scipy_stats.t.sf(abs(t), df=n-1))
            margin = float(scipy_stats.t.ppf(.975, n-1)*se)
            low, high = difference-margin, difference+margin
            dz = difference/sd
    return dict(n_pairs=n, mean_difference=difference, difference_sd=sd,
                t_statistic=t, p_value=p, ci_low=low, ci_high=high, cohens_dz=dz,
                reference_mean=float(a.mean()), comparison_mean=float(b.mean()),
                reference_sd=sa, comparison_sd=sb,
                reference_zero_seed_variance=bool(n > 1 and np.ptp(a) == 0),
                comparison_zero_seed_variance=bool(n > 1 and np.ptp(b) == 0),
                zero_paired_variance=bool(zero), sample_label=_sample_label(n))


def holm_adjust(p_values):
    """Holm step-down p-values in input order; NaNs remain NaN."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or np.any((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)) or np.isinf(p).any():
        raise ValueError("p-values must lie in [0,1] or be NaN")
    out = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    order = valid[np.argsort(p[valid], kind="stable")]
    out[order] = np.minimum(1., np.maximum.accumulate(p[order]*np.arange(len(order), 0, -1)))
    return out


def interpretation(ci_low, ci_high, p_holm, task):
    """Mechanical a-priori rules; difference-supported has first precedence."""
    if not np.isfinite([ci_low, ci_high, p_holm]).all():
        return "inconclusive"
    if (ci_low > 0 or ci_high < 0) and p_holm < .05:
        return "difference supported"
    margin = .002 if task == "binary" else .01
    if ci_low >= -margin and ci_high <= margin:
        return "practically equivalent"
    return "inconclusive"


def paired_tests(rows, reference="exlightnet"):
    """Pair by seed within task, reporting unmatched seeds rather than hiding n.

    Holm family includes every available comparison against the reference in
    that task (including ablations and W8). Mixed campaigns, duplicate seed
    identities and mismatched split seeds are hard errors. The only permitted
    device mixture is labelled CUDA neural/CPU tree comparisons within gpu_v2.
    """
    _validate_rows(rows)
    device_label = campaign_device_label(rows)
    output = []
    for task in sorted({r["task"] for r in rows}):
        models = sorted({r["model"] for r in rows if r["task"] == task})
        ref = {r["seed"]: r for r in rows if r["task"] == task and r["model"] == reference}
        if not ref:
            continue
        for model in models:
            if model == reference:
                continue
            other = {r["seed"]: r for r in rows if r["task"] == task and r["model"] == model}
            seeds = sorted(ref.keys() & other.keys())
            if not seeds:
                raise ValueError(f"No matched seeds for {task}/{model}")
            if any(ref[s].get("split_seed") != other[s].get("split_seed") for s in seeds):
                raise ValueError("Mismatched split seeds")
            summary = paired_summary([ref[s]["macro_f1"] for s in seeds], [other[s]["macro_f1"] for s in seeds])
            ref_device = ref[seeds[0]].get("device", "cpu")
            comparison_device = other[seeds[0]].get("device", "cpu")
            output.append(dict(campaign=rows[0]["campaign"], device="mixed" if device_label == MIXED_DEVICE_LABEL else ref_device,
                reference_device=ref_device, comparison_device=comparison_device, device_label=device_label,
                task=task, reference=reference, model=model, seeds=",".join(map(str, seeds)),
                missing_reference_seeds=",".join(map(str, sorted(other.keys()-ref.keys()))),
                missing_comparison_seeds=",".join(map(str, sorted(ref.keys()-other.keys()))),
                **summary))
    frame = pd.DataFrame(output)
    if not frame.empty:
        frame["p_holm"] = np.nan
        for _, group in frame.groupby("task"):
            frame.loc[group.index, "p_holm"] = holm_adjust(group.p_value)
        frame["interpretation"] = [interpretation(r.ci_low, r.ci_high, r.p_holm, r.task) for r in frame.itertuples()]
    return frame


def _confusion(cm):
    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or not cm.size or not np.isfinite(cm).all() or np.any(cm < 0) or np.any(cm != np.floor(cm)):
        raise ValueError("Expected a square nonnegative integer confusion matrix")
    return cm.astype(np.int64)


def class_f1_from_confusion(cm):
    """Per-class F1 on the full confusion matrix, zero_division=0."""
    cm = _confusion(cm)
    numerator = 2*np.diag(cm).astype(float)
    denominator = cm.sum(0)+cm.sum(1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)


def macro_f1_from_confusion(cm, labels=None):
    """Average selected F1 labels; all flows still contribute false positives."""
    values = class_f1_from_confusion(cm)
    if labels is not None:
        values = values[np.asarray(labels, dtype=int)]
    return float(values.mean()) if len(values) else np.nan


def flow_confusion(y, predictions, n_classes):
    """Pure fixed-class-axis counts from integer class labels."""
    y, pred = np.asarray(y), np.asarray(predictions)
    if y.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError("Flow arrays must align and be nonempty")
    for arr in (y, pred):
        if not np.issubdtype(arr.dtype, np.integer) or np.any(arr < 0) or np.any(arr >= n_classes):
            raise ValueError("Class predictions must be integer labels within class axis")
    return np.bincount(y.astype(np.int64)*n_classes+pred, minlength=n_classes**2).reshape(n_classes, n_classes)


def _bootstrap_summary(values, pairing, replicates, n_seeds, difference=False):
    low, high = np.quantile(values, [.025, .975])
    return {"mean_difference" if difference else "macro_f1_mean": float(np.mean(values)),
            "ci_low": float(low), "ci_high": float(high), "pairing": pairing,
            "replicates": replicates, "n_seeds": n_seeds,
            "sample_label": _sample_label(n_seeds), "estimand": ESTIMAND}


def bootstrap_macro_f1(confusions, replicates=2000, random_seed=42, labels=None, pairing="UNPAIRED"):
    """Seed-then-flow bootstrap from confusion cells; labels mask is fixed.

    ``pairing`` describes available provenance. With no predictions the result
    MUST use the default UNPAIRED; this helper never pairs two models.
    """
    matrices = [_confusion(c) for c in confusions]
    if not matrices or replicates < 1 or any(c.shape != matrices[0].shape or c.sum() < 1 for c in matrices):
        raise ValueError("Compatible nonempty confusion matrices and positive replicates required")
    rng = np.random.default_rng(random_seed)
    counts = [(int(c.sum()), c.ravel()/c.sum()) for c in matrices]
    values = []
    for index in rng.integers(0, len(matrices), size=replicates):
        n, p = counts[index]
        cm = rng.multinomial(n, p).reshape(matrices[0].shape)
        values.append(macro_f1_from_confusion(cm, labels))
    return _bootstrap_summary(values, pairing, replicates, len(matrices))


def paired_flow_bootstrap(seed_caches, n_classes, replicates=2000, random_seed=42, labels=None):
    """Exact paired bootstrap of reference minus comparison macro-F1.

    ``seed_caches`` is a sequence of dictionaries, one per matched seed, with
    aligned integer arrays ``y``, ``reference``, ``prediction``. Optional
    ``reference_y``/``prediction_y`` and ``reference_ids``/``prediction_ids``
    must match y/each other. Callers verify persisted hashes and seed identity
    before building this pure representation. Joint (y,pred_a,pred_b) cells
    preserve covariance under every common-flow multinomial draw.
    """
    if not seed_caches or replicates < 1:
        raise ValueError("Matched caches and positive replicates required")
    prepared = []
    for cache in seed_caches:
        y, a, b = [np.asarray(cache[k]) for k in ("y", "reference", "prediction")]
        flow_confusion(y, a, n_classes)
        flow_confusion(y, b, n_classes)
        for name in ("reference_y", "prediction_y"):
            if name in cache and not np.array_equal(y, cache[name]):
                raise ValueError("True-label arrays do not align")
        ids = (cache.get("reference_ids"), cache.get("prediction_ids"))
        if any(x is not None for x in ids) and (any(x is None for x in ids) or not np.array_equal(*ids) or len(ids[0]) != len(y)):
            raise ValueError("Flow IDs do not align")
        codes = (y.astype(np.int64)*n_classes+a)*n_classes+b
        cells, counts = np.unique(codes, return_counts=True)
        prepared.append((len(y), cells, counts/counts.sum()))
    rng = np.random.default_rng(random_seed)
    values = []
    for index in rng.integers(0, len(prepared), size=replicates):
        n, codes, probabilities = prepared[index]
        counts = rng.multinomial(n, probabilities)
        ya, b = codes//n_classes, codes % n_classes
        y = ya//n_classes
        cm_a = np.bincount(ya, weights=counts, minlength=n_classes**2).reshape(n_classes, n_classes)
        cm_b = np.bincount(y*n_classes+b, weights=counts, minlength=n_classes**2).reshape(n_classes, n_classes)
        values.append(macro_f1_from_confusion(cm_a, labels)-macro_f1_from_confusion(cm_b, labels))
    return _bootstrap_summary(values, "PAIRED", replicates, len(prepared), difference=True)


def _groups(rows):
    _validate_rows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["model"]), []).append(row)
    return sorted(groups.items())


def _validate_class_supports(rows):
    supports = {}
    for row in rows:
        cm = _confusion(row["confusion_matrix"])
        names = tuple(row["per_class"])
        support = tuple(cm.sum(1))
        if len(names) != len(cm) or any(row["per_class"][name]["support"] != n for name, n in zip(names, support)):
            raise ValueError("Per-class support and confusion matrix disagree")
        expected = supports.setdefault(row["task"], (names, support))
        if expected != (names, support):
            raise ValueError("Class order or immutable test support differs across runs")
    return supports


def per_class_table(rows):
    """Mean/sample SD per-class F1 with immutable original test support."""
    _validate_class_supports(rows)
    device_label = campaign_device_label(rows)
    output = []
    for (task, model), group in _groups(rows):
        for i, name in enumerate(group[0]["per_class"]):
            values = [class_f1_from_confusion(r["confusion_matrix"])[i] for r in group]
            output.append(dict(campaign=group[0]["campaign"], device=group[0].get("device", "cpu"),
                device_label=device_label, task=task, model=model,
                class_name=name, test_support=group[0]["per_class"][name]["support"],
                f1_mean=float(np.mean(values)), f1_sd=float(np.std(values, ddof=1)) if len(values)>1 else np.nan,
                f1_seed42=next((values[i] for i, r in enumerate(group) if r["seed"]==42), np.nan),
                n_seeds=len(group), sample_label=_sample_label(len(group))))
    return pd.DataFrame(output)


def sensitivity_table(rows, min_support=10):
    """Post-hoc retained-label mean, keeping full-matrix FP contributions."""
    supports = _validate_class_supports(rows)
    device_label = campaign_device_label(rows)
    output = []
    for (task, model), group in _groups(rows):
        names, support = supports[task]
        labels = np.flatnonzero(np.asarray(support) >= min_support)
        values = [macro_f1_from_confusion(r["confusion_matrix"], labels) for r in group]
        output.append(dict(campaign=group[0]["campaign"], device=group[0].get("device", "cpu"),
            device_label=device_label, task=task, model=model,
            macro_f1_mean=float(np.mean(values)), macro_f1_sd=float(np.std(values, ddof=1)) if len(values)>1 else np.nan,
            macro_f1_seed42=next((values[i] for i,r in enumerate(group) if r["seed"]==42),np.nan),
            n_seeds=len(group), sample_label=_sample_label(len(group)),
            excluded_classes=",".join(name for name, n in zip(names,support) if n<min_support),
            min_support=min_support, label="post-hoc sensitivity, not a replacement for macro-F1"))
    return pd.DataFrame(output)


def training_dynamics(rows):
    """Return (per-run dynamics, descriptive correlations, binary cap counts).

    Missing epoch data (trees/derived W8) stays missing, including cap flags.
    Pooled conv family standardizes epochs/F1 within each of four models.
    """
    output, correlations, caps = [], [], []
    device_label = campaign_device_label(rows)
    for (task, model), group in _groups(rows):
        sd = np.std([r["macro_f1"] for r in group], ddof=1) if len(group)>1 else np.nan
        for row in sorted(group, key=lambda r:r["seed"]):
            info = row.get("training", {})
            epochs, best = info.get("epochs_run"), info.get("best_epoch")
            max_epochs = info.get("config", {}).get("max_epochs")
            observed = epochs is not None and max_epochs is not None
            output.append(dict(campaign=row["campaign"], device=row.get("device", "cpu"), device_label=device_label,
                task=task, model=model, seed=row["seed"],
                macro_f1=row["macro_f1"], model_macro_f1_sd=sd, epochs_run=epochs, best_epoch=best,
                max_epochs=max_epochs, hit_cap=epochs>=max_epochs if observed else None,
                stopped_early=epochs<max_epochs if observed else None,
                cosine_lr_factor=.5*(1+np.cos(np.pi*epochs/max_epochs)) if observed else np.nan,
                unstable=bool((best is not None and best<=3) or (np.isfinite(sd) and sd>.05)),
                early_best_epoch=bool(best is not None and best<=3),
                high_seed_sd=bool(np.isfinite(sd) and sd>.05), n_seeds=len(group), sample_label=_sample_label(len(group))))
    frame=pd.DataFrame(output)
    for (task, model), group in frame.groupby(["task","model"], sort=True):
        complete=group.dropna(subset=["epochs_run", "macro_f1"])
        defined=len(complete)>1 and complete.epochs_run.nunique()>1 and complete.macro_f1.nunique()>1
        rho, p = scipy_stats.spearmanr(complete.epochs_run, complete.macro_f1) if defined else (np.nan,np.nan)
        correlations.append(dict(campaign=rows[0]["campaign"],device=group.iloc[0].device,
            device_label=device_label,task=task,model=model,n=len(complete),
            spearman_rho=rho,p_value=p,sample_label=_sample_label(len(complete)),
            interpretation="descriptive only, not causal", members=model))
        if task=="binary":
            observed=group.dropna(subset=["hit_cap"])
            caps.append(dict(campaign=rows[0]["campaign"],device=group.iloc[0].device,
                device_label=device_label,task=task,model=model,n_runs=len(group),
                n_epoch_observed=len(observed),n_hit_cap=int(observed.hit_cap.sum()) if len(observed) else np.nan,
                n_best_at_cap=int((observed.best_epoch==observed.max_epochs).sum()) if len(observed) else np.nan,
                sample_label=_sample_label(len(group))))
    for task in sorted(frame.task.unique()):
        xs,ys,members=[],[],[]
        for model in CONV_FAMILY:
            group=frame[(frame.task==task)&(frame.model==model)].dropna(subset=["epochs_run"])
            if len(group)>1 and group.epochs_run.nunique()>1 and group.macro_f1.nunique()>1:
                xs.extend((group.epochs_run-group.epochs_run.mean())/group.epochs_run.std(ddof=1))
                ys.extend((group.macro_f1-group.macro_f1.mean())/group.macro_f1.std(ddof=1))
                members.append(model)
        rho,p=scipy_stats.spearmanr(xs,ys) if len(xs)>1 else (np.nan,np.nan)
        family_devices=set(frame[(frame.task==task)&frame.model.isin(members)].device)
        correlations.append(dict(campaign=rows[0]["campaign"], device=next(iter(family_devices)) if family_devices else None,
            device_label=device_label, task=task,model="pooled_conv_family", n=len(xs),
            spearman_rho=rho,p_value=p, sample_label=f"n={len(xs)}, pooled descriptive only",
            interpretation="descriptive only, not causal; within-model z scores",members=",".join(members)))
    return frame,pd.DataFrame(correlations),pd.DataFrame(caps)


def _load_predictions(row, directory, source_hashes):
    """Verify persisted labels/probabilities against receipts and result rows."""
    path=directory/row["task"]/row["model"]/str(row["seed"])/"predictions.npz"
    metric_path=path.with_name("metrics.json")
    receipt_path=path.with_name("test_evaluation_receipt.json")
    if not path.exists() and not receipt_path.exists():
        # An absent legacy prediction/receipt cache permits only the explicitly
        # UNPAIRED confusion fallback. A present claimed receipt never does.
        return None
    for required in (metric_path,receipt_path):
        if not required.exists():
            raise ValueError(f"Incomplete prediction cache: {required}")
    metrics=json.loads(metric_path.read_text())
    receipt=json.loads(receipt_path.read_text())
    if receipt.get("status")!="complete" or receipt.get("test_prediction_passes")!=1 or receipt.get("metadata",{}).get("fingerprint")!=row.get("fingerprint") or metrics.get("fingerprint")!=row.get("fingerprint"):
        raise ValueError(f"Prediction receipt/fingerprint mismatch: {path}")
    if (metrics.get("task"), metrics.get("model"), metrics.get("seed")) != (row["task"],row["model"],row["seed"]):
        raise ValueError(f"Prediction identity mismatch: {path}")
    for source in (metric_path,receipt_path):
        source_hashes[str(source)]=_sha256(source)
    if not path.exists():
        return None
    k=len(row["confusion_matrix"])
    with np.load(path,allow_pickle=False) as arrays:
        y=arrays["y"].copy()
        probs=arrays["probabilities"]
        if probs.shape!=(len(y),k) or not np.isfinite(probs).all() or np.any(probs<0) or not np.allclose(probs.sum(1),1,atol=1e-5):
            raise ValueError(f"Corrupt cached probabilities: {path}")
        pred=probs.argmax(1)
    cm=flow_confusion(y,pred,k)
    if not np.array_equal(cm,row["confusion_matrix"]) or not np.isclose(macro_f1_from_confusion(cm),row["macro_f1"],rtol=0,atol=1e-12):
        raise ValueError(f"Predictions and cached metrics disagree: {path}")
    source_hashes[str(path)]=_sha256(path)
    return {"y":y,"prediction":pred}


def write_cpu_statistics(root, output, replicates=2000, random_seed=42):
    """Read immutable full caches, write new CSVs/manifest, return path mapping.

    ``root`` is the repository root, or the full-results directory. ``output``
    must not be within results. No model/pickle loading, fitting or inference.
    Canonical output uses 2,000 replicates; smaller counts aid pure unit tests.
    """
    root, output=Path(root).resolve(),Path(output).resolve()
    directory=root/"results"/"full" if (root/"results"/"full"/"results.json").exists() else root
    immutable_results=directory.parent if directory.name=="full" and directory.parent.name=="results" else directory
    if output==immutable_results or immutable_results in output.parents:
        raise ValueError("Statistics output must be outside immutable results")
    result_path,manifest_path=directory/"results.json",directory/"manifest.json"
    rows=load_campaign_rows(result_path,manifest_path)
    if {r["campaign"] for r in rows}!={"cpu_v1"}:
        raise ValueError("write_cpu_statistics accepts only cpu_v1")
    _validate_class_supports(rows)
    hashes={str(p):_sha256(p) for p in (result_path,manifest_path)}
    caches={}
    task_y={}
    for row in rows:
        key=(row["task"],row["model"],row["seed"])
        cached=_load_predictions(row,directory,hashes)
        if cached is not None:
            expected=task_y.setdefault(row["task"],cached["y"])
            if not np.array_equal(expected,cached["y"]):
                raise ValueError(f"Cached true-label arrays do not align: {key}")
        caches[key]=cached
    dynamics,correlations,caps=training_dynamics(rows)
    per_seed=[]
    summaries=[]
    bootstraps=[]
    paired_bootstraps=[]
    for (task,model),group in _groups(rows):
        values=np.asarray([r["macro_f1"] for r in group])
        for row in sorted(group,key=lambda r:r["seed"]):
            info=row.get("training",{})
            per_seed.append(dict(campaign=row["campaign"],device=row["device"],task=task,model=model,
                seed=row["seed"],macro_f1=row["macro_f1"],accuracy=row["accuracy"],
                epochs_run=info.get("epochs_run"),best_epoch=info.get("best_epoch"),
                n_seeds=len(group),sample_label=_sample_label(len(group))))
        summaries.append(dict(campaign=group[0]["campaign"],task=task,model=model,n_seeds=len(group),
            sample_label=_sample_label(len(group)),macro_f1_mean=float(values.mean()),
            macro_f1_sd=float(values.std(ddof=1)) if len(values)>1 else np.nan,
            macro_f1_seed42=next((r["macro_f1"] for r in group if r["seed"]==42),np.nan),
            zero_seed_variance=bool(len(values)>1 and np.ptp(values)==0)))
        present=all(caches[(task,model,r["seed"])] is not None for r in group)
        bootstrap=bootstrap_macro_f1([r["confusion_matrix"] for r in group],replicates,random_seed,
            pairing="cached predictions; single-model flow bootstrap" if present else "UNPAIRED")
        bootstraps.append(dict(campaign=group[0]["campaign"],task=task,model=model,**bootstrap))
        if model=="exlightnet":
            continue
        reference={r["seed"]:r for r in rows if r["task"]==task and r["model"]=="exlightnet"}
        matched=sorted(reference.keys()&{r["seed"] for r in group})
        seed_caches=[]
        for seed in matched:
            a,b=caches[(task,"exlightnet",seed)],caches[(task,model,seed)]
            if a is None or b is None:
                seed_caches=[]
                break
            seed_caches.append(dict(y=a["y"],reference=a["prediction"],prediction=b["prediction"],prediction_y=b["y"]))
        if seed_caches:
            summary=paired_flow_bootstrap(seed_caches,len(group[0]["confusion_matrix"]),replicates,random_seed)
        else:
            summary=dict(mean_difference=np.nan,ci_low=np.nan,ci_high=np.nan,pairing="UNPAIRED; missing predictions, no paired interval",
                replicates=0,n_seeds=len(matched),sample_label=_sample_label(len(matched)),estimand=ESTIMAND)
        paired_bootstraps.append(dict(campaign=group[0]["campaign"],task=task,reference="exlightnet",model=model,**summary))
    tables={"per_seed_macro_f1":pd.DataFrame(per_seed),"model_summary":pd.DataFrame(summaries),
        "paired_tests":paired_tests(rows),"training_dynamics":dynamics,"correlations":correlations,
        "binary_caps":caps,"flow_bootstrap":pd.DataFrame(bootstraps),
        "paired_flow_bootstrap":pd.DataFrame(paired_bootstraps),
        "per_class_f1":per_class_table(rows),"sensitivity":sensitivity_table(rows)}
    output.mkdir(parents=True,exist_ok=True)
    paths={}
    for name,frame in tables.items():
        path=output/f"{name}.csv"
        frame.to_csv(path,index=False)
        paths[name]=str(path)
    changed=[p for p,sha in hashes.items() if _sha256(p)!=sha]
    if changed:
        raise RuntimeError(f"Immutable source changed during statistics: {changed}")
    manifest=dict(schema_version=1,campaign="cpu_v1",device="cpu",n_rows=len(rows),
        seed_values=sorted({r["seed"] for r in rows}),sample_label="n=5, indicative only",
        replicates=replicates,random_seed=random_seed,bootstrap_estimand=ESTIMAND,
        bootstrap_method="uniform training seed, exact multinomial empirical flow/joint-cell draws",
        wilcoxon="not used; fewer than six matched pairs cannot attain two-sided p<0.05",
        holm_family="all comparisons versus ExLightNet within each task, including W8 and ablations",
        sensitivity="post-hoc sensitivity, not a replacement for macro-F1; support<10 excluded from averaging only",
        libraries={name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy")},
        statistics_source_sha256=_sha256(__file__),
        source_sha256=hashes,sources_unchanged=True,
        tables={name:dict(path=Path(path).name,sha256=_sha256(path),n_rows=len(tables[name])) for name,path in paths.items()})
    manifest_file=output/"statistics_manifest.json"
    manifest_file.write_text(json.dumps(manifest,indent=2)+"\n")
    paths["manifest"]=str(manifest_file)
    return paths
