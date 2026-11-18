"""Cache-only statistics: no model fitting or final inference."""
import json

import numpy as np
import pytest
from scipy import stats as scipy_stats

from exlightnet.stats import (
    bootstrap_macro_f1, flow_confusion, holm_adjust, load_campaign_rows,
    macro_f1_from_confusion, paired_flow_bootstrap, paired_summary, paired_tests,
    per_class_table, sensitivity_table, training_dynamics, write_cpu_statistics,
)


def row(model, seed, value, **extra):
    return dict(campaign="test", device="cpu", task="multiclass", model=model,
                seed=seed, macro_f1=value, accuracy=value, **extra)


def test_paired_summary_matches_scipy():
    a = np.array([.81, .79, .9, .88, .82])
    b = np.array([.80, .77, .84, .89, .81])
    result = paired_summary(a, b)
    expected = scipy_stats.ttest_rel(a, b)
    ci = expected.confidence_interval()
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["t_statistic"] == pytest.approx(expected.statistic)
    assert result["ci_low"] == pytest.approx(ci.low)
    assert result["ci_high"] == pytest.approx(ci.high)
    assert result["cohens_dz"] == pytest.approx(np.mean(a-b)/np.std(a-b, ddof=1))


def test_constant_model_is_not_constant_paired_difference():
    result = paired_summary([.7, .8, .9, .75, .85], [.8]*5)
    assert result["comparison_zero_seed_variance"]
    assert not result["zero_paired_variance"]
    assert result["p_value"] == pytest.approx(1)


@pytest.mark.parametrize("a,b,p,dz", [([.8]*5, [.8]*5, 1, 0),
                                    ([.9]*5, [.8]*5, 0, np.inf)])
def test_degenerate_paired_difference(a, b, p, dz):
    result = paired_summary(a, b)
    assert result["zero_paired_variance"]
    assert result["p_value"] == p
    assert result["cohens_dz"] == dz
    assert result["ci_low"] == pytest.approx(np.mean(np.array(a)-b))


def test_single_pair_does_not_invent_sd_or_ci():
    result = paired_summary([.8], [.7])
    assert result["n_pairs"] == 1
    assert np.isnan(result["p_value"])
    assert np.isnan(result["ci_low"])


def test_holm_keeps_order_and_missing_values():
    np.testing.assert_allclose(holm_adjust([.04, .001, .03, np.nan]),
                               [.06, .003, .06, np.nan], equal_nan=True)


def test_pairs_match_seed_and_show_missing():
    rows = [row("exlightnet", s, .6+s/100) for s in range(5)]
    rows += [row("mlp", s, .5+s/100) for s in [4, 1, 2]]
    result = paired_tests(rows).iloc[0]
    assert result.n_pairs == 3
    assert result.mean_difference == pytest.approx(.1)
    assert result.missing_comparison_seeds == "0,3"


def test_reject_mixed_campaign_duplicate_seed_and_device():
    rows = [row("exlightnet", 42, .8), row("mlp", 42, .7)]
    for bad in [dict(rows[1], campaign="other"), dict(rows[1], device="cuda")]:
        with pytest.raises(ValueError):
            paired_tests([rows[0], bad])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_tests(rows + [rows[0]])


def test_gpu_v2_only_neural_cuda_tree_cpu_device_exception():
    rows=[dict(row("exlightnet",seed,.8+seed/10000),campaign="gpu_v2",device="cuda")
          for seed in [42,43]]
    rows += [dict(row("random_forest",seed,.7+seed/10000),campaign="gpu_v2",device="cpu")
             for seed in [42,43]]
    result=paired_tests(rows).iloc[0]
    assert result.reference_device == "cuda"
    assert result.comparison_device == "cpu"
    assert result.device_label == "mixed-device within gpu_v2; trees are device-independent"
    with pytest.raises(ValueError):
        paired_tests([dict(r,campaign="gpu_v3") for r in rows])
    with pytest.raises(ValueError):
        paired_tests([dict(r,model="mlp") if r["model"]=="random_forest" else r for r in rows])
    with pytest.raises(ValueError):
        paired_tests([dict(r,device="cpu") if r["model"]=="exlightnet" and r["seed"]==42 else r for r in rows])


def test_mixed_device_label_applies_to_whole_gpu_table():
    rows=[dict(row("exlightnet",42,.8),campaign="gpu_v2",device="cuda"),
          dict(row("mlp",42,.7),campaign="gpu_v2",device="cuda"),
          dict(row("xgboost",42,.9),campaign="gpu_v2",device="cpu")]
    table=paired_tests(rows)
    assert set(table.device_label)=={"mixed-device within gpu_v2; trees are device-independent"}
    assert table.set_index("model").loc["mlp","comparison_device"]=="cuda"


def test_legacy_campaign_requires_full_manifest_without_mutation(tmp_path):
    source = [{"task":"binary", "model":"mlp", "seed":42, "macro_f1":.8}]
    results = tmp_path / "results.json"
    results.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="manifest"):
        load_campaign_rows(results)
    (tmp_path / "manifest.json").write_text(json.dumps({"mode":"full", "seeds":list(range(42,47))}))
    loaded = load_campaign_rows(results)
    assert loaded[0]["campaign"] == "cpu_v1"
    assert loaded[0]["device"] == "cpu"
    assert json.loads(results.read_text()) == source


def test_sensitivity_retains_false_positives_from_excluded_class():
    cm = np.array([[10, 0], [4, 0]])
    assert macro_f1_from_confusion(cm, labels=[0]) == pytest.approx(20/24)
    rows = [row("exlightnet", s, .5, confusion_matrix=cm.tolist(),
                per_class={"large":{"f1":20/24,"support":10},
                           "tiny":{"f1":0.,"support":4}}) for s in range(42,47)]
    assert sensitivity_table(rows).iloc[0].macro_f1_mean == pytest.approx(20/24)
    classes = per_class_table(rows)
    assert classes.set_index("class_name").loc["tiny", "test_support"] == 4
    with pytest.raises(ValueError, match="support"):
        per_class_table(rows + [row("mlp", 42, .5, confusion_matrix=[[9,0],[4,1]],
            per_class={"large":{"f1":1,"support":9},"tiny":{"f1":0,"support":5}})])


def test_paired_flow_bootstrap_identical_models_zero_difference():
    y=np.array([0]*30+[1]*20)
    pred=np.array([0]*27+[1]*3+[0]*2+[1]*18)
    caches=[{"y":y,"reference":pred,"prediction":pred}]
    result=paired_flow_bootstrap(caches, 2, replicates=100)
    assert result["mean_difference"] == 0
    assert result["ci_low"] == result["ci_high"] == 0
    assert result["pairing"] == "PAIRED"
    assert "single-random-run" in result["estimand"]


def test_flow_alignment_rejects_misaligned_ids_and_lengths():
    with pytest.raises(ValueError, match="align"):
        paired_flow_bootstrap([{"y":[0,1],"reference":[0,1],"prediction":[0]}],2)
    with pytest.raises(ValueError, match="align"):
        paired_flow_bootstrap([{"y":[0,1],"reference":[0,1],"prediction":[0,1],
                               "reference_ids":[1,2],"prediction_ids":[2,1]}],2)


def test_unpaired_bootstrap_uses_fixed_class_axis_and_is_reproducible():
    cm=np.array([[9,1],[2,8]])
    a=bootstrap_macro_f1([cm], replicates=100, random_seed=3)
    assert a == bootstrap_macro_f1([cm], replicates=100, random_seed=3)
    assert a["pairing"] == "UNPAIRED"
    assert 0 <= a["ci_low"] <= a["ci_high"] <= 1
    np.testing.assert_array_equal(flow_confusion([0,1],[0,1],3),np.diag([1,1,0]))


def test_dynamics_cosine_cap_instability():
    rows=[]
    for i, model in enumerate(["exlightnet","no_se","standard_conv","std_cnn"]):
        for s in range(5):
            rows.append(row(model,42+s,.7+.01*s, training={"epochs_run":10+s,
                "best_epoch":2 if s==0 else 8,"config":{"max_epochs":14}}))
    dynamics, correlations, caps=training_dynamics(rows)
    assert dynamics.iloc[0].cosine_lr_factor == pytest.approx(.5*(1+np.cos(np.pi*10/14)))
    assert dynamics.iloc[0].unstable
    assert not dynamics.iloc[1].unstable
    assert dynamics.iloc[4].hit_cap
    pooled=correlations[correlations.model=="pooled_conv_family"].iloc[0]
    assert pooled.n == 20
    assert pooled.spearman_rho == pytest.approx(1)
    assert caps.empty


def cached_campaign(tmp_path):
    """Write synthetic *already scored* evidence, never fit a model."""
    y = np.array([0]*10+[1]*10)
    predictions = {"exlightnet": y, "mlp": np.array([1]+[0]*9+[1]*10)}
    rows=[]
    root=tmp_path/"results"/"full"
    for model, pred in predictions.items():
        for seed in [42,43]:
            cm=flow_confusion(y,pred,2)
            fingerprint=f"{model}-{seed}"
            result=dict(task="binary",model=model,seed=seed,accuracy=float(np.mean(y==pred)),
                macro_f1=macro_f1_from_confusion(cm),fingerprint=fingerprint,
                confusion_matrix=cm.tolist(),per_class={name:{"support":10,"f1":float(value)}
                    for name,value in zip(["a","b"], np.diag(cm)*2/(cm.sum(0)+cm.sum(1)))})
            rows.append(result)
            directory=root/"binary"/model/str(seed)
            directory.mkdir(parents=True)
            (directory/"metrics.json").write_text(json.dumps(result))
            (directory/"test_evaluation_receipt.json").write_text(json.dumps(
                {"status":"complete","test_prediction_passes":1,"metadata":{"fingerprint":fingerprint}}))
            np.savez_compressed(directory/"predictions.npz",y=y,probabilities=np.eye(2)[pred])
    (root/"results.json").write_text(json.dumps(rows))
    (root/"manifest.json").write_text(json.dumps({"mode":"full","seeds":list(range(42,47))}))
    return root


def test_writer_uses_predictions_without_mutating_sources(tmp_path):
    root=cached_campaign(tmp_path)
    before={p:p.read_bytes() for p in root.rglob("*") if p.is_file()}
    paths=write_cpu_statistics(tmp_path,tmp_path/"output",replicates=25)
    manifest=json.loads(open(paths["manifest"]).read())
    assert manifest["n_rows"] == 4
    assert manifest["replicates"] == 25
    assert manifest["sources_unchanged"]
    assert len(manifest["tables"]) == 10
    for path,contents in before.items():
        assert path.read_bytes() == contents


@pytest.mark.parametrize("corruption", ["permuted_y","corrupt_probability","claimed_receipt","missing_predictions_claimed_receipt"])
def test_writer_rejects_corrupt_or_misaligned_predictions(tmp_path,corruption):
    root=cached_campaign(tmp_path)
    directory=root/"binary"/"mlp"/"42"
    if "claimed_receipt" in corruption:
        (directory/"test_evaluation_receipt.json").write_text(json.dumps({"status":"claimed"}))
        if corruption=="missing_predictions_claimed_receipt":
            (directory/"predictions.npz").unlink()
    else:
        with np.load(directory/"predictions.npz") as data:
            y,p=data["y"].copy(),data["probabilities"].copy()
        if corruption=="permuted_y":
            y=y[::-1]
            p=p[::-1]
        else:
            p[0,0]=np.nan
        np.savez_compressed(directory/"predictions.npz",y=y,probabilities=p)
    with pytest.raises(ValueError):
        write_cpu_statistics(tmp_path,tmp_path/"output",replicates=25)
    assert not (tmp_path/"output").exists()


def test_missing_predictions_fallback_is_explicitly_unpaired(tmp_path):
    root=cached_campaign(tmp_path)
    (root/"binary"/"mlp"/"42"/"predictions.npz").unlink()
    paths=write_cpu_statistics(tmp_path,tmp_path/"output",replicates=25)
    import pandas as pd
    flow=pd.read_csv(paths["flow_bootstrap"])
    assert flow.set_index("model").loc["mlp","pairing"] == "UNPAIRED"
    paired=pd.read_csv(paths["paired_flow_bootstrap"]).iloc[0]
    assert paired.pairing.startswith("UNPAIRED")
    assert np.isnan(paired.ci_low)


def test_seed_flow_bootstrap_is_single_random_run_not_average():
    # Two deterministic runs with opposite outcomes. Sampling the training
    # seed preserves both outcomes; averaging seed draws would narrow this CI.
    result=bootstrap_macro_f1([np.diag([20,20]),np.array([[0,20],[20,0]])],
                             replicates=2000,random_seed=42)
    assert result["ci_low"] == 0
    assert result["ci_high"] == 1
    assert result["macro_f1_mean"] == pytest.approx(.5,abs=.04)
    assert result["estimand"] == "single-random-run plus flow uncertainty; not CI for the seed mean"
