"""Pure synthetic scored-evidence fixtures; no fitting or final inference."""
import copy
import hashlib
import io
import json
import stat
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exlightnet.campaign import run_fingerprint, study_runs
from exlightnet.stats import class_f1_from_confusion, flow_confusion, macro_f1_from_confusion
from exlightnet.study_stats import aggregate_studies, read_study_archives, _validate_history


PROTOCOL = Path(__file__).resolve().parents[1]/"docs/statistics_protocol.json"


def encoded(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def npz(**arrays):
    out=io.BytesIO()
    np.savez_compressed(out,**arrays)
    return out.getvalue()


def prepared(tmp_path, split=42):
    directory=tmp_path/"prepared"
    directory.mkdir(exist_ok=True)
    data=npz(y2_test=np.array([0]*12+[1]*12,dtype=np.int8),
             y12_test=np.array([0]*12+[1]*12,dtype=np.int8))
    path=directory/f"split_{split}.npz"; path.write_bytes(data)
    manifest_path=directory/"manifest.json"
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {"frozen_alpha":{"binary":"none","multiclass":"none"},"splits":{}}
    manifest["splits"][str(split)]={"path":path.name,"sha256":sha(data),"data_fingerprint":"a"*64,
         "payload_sha256":"b"*64,"class_names":["normal","attack"]}
    manifest_path.write_bytes(encoded(manifest))
    return directory


def evidence(seed=42, model="exlightnet", *, campaign="gpu_v2", study="B1", arm="paper", device="cuda", split=42, wrong=1, lr=.0015, task=None, last_wrong=None):
    task=task or ("multiclass" if study in ("B3","B5","probe","determinism") else "binary")
    config=dict(seed=seed,max_epochs=20,batch_size=512,patience=6,lr=lr,
                weight_decay=.0001,alpha_mode="none",sampler="tempered",threads=1)
    if task=="multiclass":config["max_epochs"]=30
    if study=="probe":config["max_epochs"]=3
    if arm=="uncapped60":config["max_epochs"]=60
    if arm=="patience10_min15":config["patience"]=10
    plan=dict(study=study,task=task,model=model,seed=seed,split_seed=split,arm=arm)
    if study=="B5":plan["lr"]=lr
    y=np.array([0]*12+[1]*12,dtype=np.int8);pred=y.copy();pred[:wrong]=1
    length=config["max_epochs"] if arm=="fixed30" or study=="probe" else 15 if arm=="patience10_min15" else 7
    payload={};record=dict(schema_version=1,status="complete",campaign=campaign,**plan,device=device,
        config=config,data_fingerprint="a"*64,payload_sha256="b"*64,prepared_sha256=None,
        libraries={"numpy":"fixture","torch":"fixture","scikit-learn":"fixture","xgboost":"fixture",
                   "cuda":"fixture","cudnn":"fixture","gpu_name":"GPU A"},warnings=[],epochs_run=length,best_epoch=1,best_val_macro_f1=.8,
        history=[dict(epoch=epoch,train_loss=.3,val_macro_f1=.8 if epoch==1 else .7,lr=lr,seconds=.1) for epoch in range(1,length+1)],
        headline_checkpoint="best",checkpoints={})
    if model in ("random_forest","xgboost"):
        record.update(history=[],epochs_run=None,best_epoch=None,best_val_macro_f1=None)
    record["_prediction"]=(y,pred)
    if last_wrong is not None:
        last=y.copy();last[:last_wrong]=1
        record["_last_prediction"]=last
    return record


def seed_block(study,seed):
    return [evidence(seed,row["model"],study=row.get("reuse_study",study),arm=row["arm"],
                     split=row["split_seed"],task=row["task"],
                     device="cpu" if row["model"] in ("random_forest","xgboost") else "cuda")
            for row in study_runs(study) if row["seed"]==seed]


def archive(tmp_path, records, name="results.zip", transform=None):
    directory=prepared(tmp_path)
    manifest=json.loads((directory/"manifest.json").read_text())
    members={}
    for original in records:
        record=copy.deepcopy(original);y,best=record.pop("_prediction")
        last=record.pop("_last_prediction",best)
        if str(record["split_seed"]) not in manifest["splits"]:
            prepared(tmp_path,record["split_seed"])
            manifest=json.loads((directory/"manifest.json").read_text())
        entry=manifest["splits"][str(record["split_seed"])]
        record["prepared_sha256"]=entry["sha256"]
        plan={k:record[k] for k in ("study","task","model","seed","split_seed","arm")}
        if record["study"]=="B5":plan["lr"]=record["lr"]
        identity=dict(campaign=record["campaign"],plan=plan,device=record["device"],config=record["config"],
            data_fingerprint=record["data_fingerprint"],payload_sha256=record["payload_sha256"],prepared_sha256=record["prepared_sha256"],libraries=record["libraries"])
        fp=run_fingerprint(identity);record["run_fingerprint"]=fp
        base=f'{record["device"]}/{fp}'
        for checkpoint,pred in (("best",best),("last",last)):
            if record["model"] in ("random_forest","xgboost") and checkpoint=="last":
                record["checkpoints"][checkpoint]=copy.deepcopy(record["checkpoints"]["best"])
                continue
            cm=flow_confusion(y,pred,2)
            metrics=dict(accuracy=float(np.mean(y==pred)),macro_f1=macro_f1_from_confusion(cm),confusion_matrix=cm.tolist(),
                per_class={k:{"f1":float(v),"support":12} for k,v in zip(["normal","attack"],class_f1_from_confusion(cm))})
            prefix=f"{base}/{checkpoint}"
            predictions=npz(y=y,predictions=pred)
            metricbytes=encoded(metrics)
            receipt_value=dict(status="complete",fingerprint=f'{fp}:{checkpoint}',test_prediction_passes=1,
                prediction_sha256=sha(predictions),metrics_sha256=sha(metricbytes))
            item={"metrics":metrics}
            if record["study"]=="determinism":
                logits=npz(logits=np.eye(2,dtype=np.float32)[pred])
                logitpath=f"{prefix}/logits.npz";members[logitpath]=logits
                item.update(logits_path=logitpath,logits_sha256=sha(logits))
                receipt_value["logits_sha256"]=sha(logits)
            receipt=encoded(receipt_value)
            for label,filename,data in [("prediction","predictions.npz",predictions),("metrics","metrics.json",metricbytes),("receipt","receipt.json",receipt)]:
                path=f"{prefix}/{filename}";members[path]=data;item[f"{label}_path"]=path;item[f"{label}_sha256"]=sha(data)
            record["checkpoints"][checkpoint]=item
        members[f"{base}/record.json"]=encoded(record)
    if transform:transform(members)
    path=tmp_path/name
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for member,data in members.items():z.writestr(member,data)
    return path,directory


def aggregate(tmp_path, paths, directory):
    return aggregate_studies(paths,directory,tmp_path/"out",tmp_path/"REPORT_test.md",protocol_path=PROTOCOL,replicates=50)


def test_no_archives_pending_and_exact_protocol(tmp_path):
    paths=aggregate(tmp_path,[],tmp_path/"not_present")
    text=Path(paths["report"]).read_text()
    assert text.count("PENDING: study not run")==5
    protocol=json.loads(PROTOCOL.read_text())
    original=protocol["rules_verbatim"]
    assert "unless fixed-budget (arm b) shows both" in original[-1]
    active,history=text.split("### Superseded original stopping-hypothesis rule")
    assert original[-1] in history and original[-1] not in active
    for rule in original[:3]+protocol["amendments"][0]["rules_verbatim"]:
        assert rule in active
    assert protocol["amendments"][1]["clarification"] in active
    assert "PA-20261118-1 (2026-11-18)" in active
    stopping=pd.read_csv(paths["early_stopping_hypothesis"])
    assert list(stopping.endpoint_role)==["PRIMARY","SECONDARY"]
    assert set(stopping.endpoint_decision)=={"PENDING"}
    assert stopping.mean_difference.isna().all()
    for name,path in paths.items():
        if str(path).endswith(".csv"):
            assert len(pd.read_csv(path).columns)>0


def test_valid_dedup_zip_and_hash_manifest(tmp_path):
    zip_path,directory=archive(tmp_path,seed_block("B1",42))
    paths=aggregate(tmp_path,[zip_path,zip_path],directory)
    manifest=json.loads(Path(paths["manifest"]).read_text())
    assert manifest["n_unique_runs"]==11
    assert manifest["input_archives"][str(zip_path)]["sha256"]==sha(zip_path.read_bytes())
    paired=pd.read_csv(paths["paired_tests"])
    assert paired.iloc[0].n_pairs==1
    assert set(paired.checkpoint)=={"best","last"}
    assert "PARTIAL" in Path(paths["report"]).read_text()


def test_mixed_campaign_and_conflicting_identity_rejected(tmp_path):
    first,directory=archive(tmp_path,[evidence()],"one.zip")
    other,_=archive(tmp_path,[evidence(campaign="gpu_v2_smoke")],"two.zip")
    with pytest.raises(ValueError,match="campaign"):
        aggregate(tmp_path,[first,other],directory)
    conflict,_=archive(tmp_path,[evidence(wrong=3)],"conflict.zip")
    with pytest.raises(ValueError,match="Conflicting"):
        aggregate(tmp_path,[first,conflict],directory)


@pytest.mark.parametrize("bad",["../outside","/absolute","a/../../escape","a\\evil"])
def test_archive_traversal_rejected_even_unreferenced(tmp_path,bad):
    z,d=archive(tmp_path,[evidence()],transform=lambda members:members.update({bad:b"x"}))
    with pytest.raises(ValueError,match="path"):
        aggregate(tmp_path,[z],d)


def test_archive_symlink_rejected(tmp_path):
    z,d=archive(tmp_path,[evidence()])
    with zipfile.ZipFile(z,"a") as archivefile:
        info=zipfile.ZipInfo("link"); info.create_system=3; info.external_attr=(stat.S_IFLNK|0o777)<<16
        archivefile.writestr(info,b"/tmp/target")
    with pytest.raises(ValueError,match="symlink"):
        aggregate(tmp_path,[z],d)


@pytest.mark.parametrize("corruption",["predictions","receipt","inline","fingerprint","prepared"])
def test_evidence_corruption_rejected(tmp_path,corruption):
    def corrupt(members):
        path=next(p for p in members if p.endswith("record.json")); record=json.loads(members[path])
        if corruption=="predictions":members[record["checkpoints"]["best"]["prediction_path"]]=b"bad"
        elif corruption=="receipt":
            rp=record["checkpoints"]["best"]["receipt_path"]; value=json.loads(members[rp]); value["status"]="claimed"
            members[rp]=encoded(value);record["checkpoints"]["best"]["receipt_sha256"]=sha(members[rp])
        elif corruption=="inline":record["checkpoints"]["best"]["metrics"]["macro_f1"]=.123
        elif corruption=="fingerprint":record["run_fingerprint"]="f"*64
        else:record["payload_sha256"]="c"*64
        members[path]=encoded(record)
    z,d=archive(tmp_path,[evidence()],transform=corrupt)
    with pytest.raises(ValueError):aggregate(tmp_path,[z],d)


def test_mixed_gpu_neural_cpu_tree_label_and_same_model_device_error(tmp_path):
    z,d=archive(tmp_path,seed_block("B1",42))
    table=pd.read_csv(aggregate(tmp_path,[z],d)["paired_tests"])
    assert set(table.device_label)=={"mixed-device within gpu_v2; trees are device-independent"}
    assert set(table.reference_device)=={"cuda"}
    assert set(table[table.model.isin(["random_forest","xgboost"])].comparison_device)=={"cpu"}
    z,d=archive(tmp_path,[evidence(),evidence(seed=43,device="cpu")],"bad_devices.zip")
    with pytest.raises(ValueError,match="device"):aggregate(tmp_path,[z],d)


def test_smoke_cannot_write_canonical_report(tmp_path):
    z,d=archive(tmp_path,[evidence(campaign="gpu_v2_smoke")])
    with pytest.raises(ValueError,match="canonical"):
        aggregate_studies([z],d,tmp_path/"out",tmp_path/"REPORT_stats_v2.md",protocol_path=PROTOCOL,replicates=20)


def test_b3_reuses_b1_paper_without_duplicate_training_records(tmp_path):
    z,d=archive(tmp_path,seed_block("B3",42))
    paths=aggregate(tmp_path,[z],d)
    summary=pd.read_csv(paths["summaries"])
    assert set(summary[summary.study=="B3"].arm)=={"paper","fixed30","patience10_min15"}
    tests=pd.read_csv(paths["paired_tests"])
    arms=tests[(tests.study=="B3")&(tests.comparison_type=="arm")]
    assert len(arms)==16
    assert set(arms.comparison_arm)=={"paper"}
    assert set(arms.reference_arm)=={"fixed30","patience10_min15"}
    rows=pd.read_csv(paths["per_run"])
    assert set(rows[(rows.study=="B3")&(rows.arm=="paper")].source_study)=={"B1"}


def test_b5_incomplete_arms_never_select_lr(tmp_path):
    z,d=archive(tmp_path,[evidence(study="B5",model="std_cnn",arm="lr_0.0003",lr=.0003)])
    paths=aggregate(tmp_path,[z],d)
    table=pd.read_csv(paths["lr_selection"])
    assert table.selected_lr.isna().all()
    assert "PENDING" in table.iloc[0].status


def test_b5_pairs_only_seed_blocks_with_all_three_declared_arms(tmp_path):
    records=[evidence(seed,model="std_cnn",study="B5",arm=f"lr_{lr:g}",lr=lr)
             for seed in (42,43) for lr in (.0003,.001,.0015) if (seed,lr)!=(43,.001)]
    z,d=archive(tmp_path,records)
    paths=aggregate(tmp_path,[z],d)
    assert set(pd.read_csv(paths["paired_tests"]).seeds)=={42}
    assert pd.read_csv(paths["lr_selection"]).selected_lr.isna().all()


def test_b4_pairs_require_both_uncapped_models_and_their_b1_references(tmp_path):
    records=[evidence(seed,model,study=study,arm=arm) for seed in (42,43)
             for model in ("exlightnet","mlp") for study,arm in (("B4","uncapped60"),("B1","paper"))
             if (seed,model,study)!=(43,"mlp","B1")]
    z,d=archive(tmp_path,records)
    paths=aggregate(tmp_path,[z],d)
    assert set(pd.read_csv(paths["paired_tests"]).query('study == "B4"').seeds)=={42}
    missing=pd.read_csv(paths["completeness"]).query('study == "B4" and seed == 43').iloc[0]
    assert missing.status=="incomplete; excluded"
    assert json.loads(missing.missing_identities)[0]["study"]=="B1"


def test_noncomplete_record_ignored_and_counted(tmp_path):
    def pending(members):
        path=next(p for p in members if p.endswith("record.json")); value=json.loads(members[path]);value["status"]="running";members[path]=encoded(value)
    z,d=archive(tmp_path,[evidence()],transform=pending)
    loaded=read_study_archives([z],d)
    assert loaded["records"]==[]
    assert loaded["n_incomplete_records"]==1


def test_complete_b5_selects_by_validation_even_when_test_prefers_other_lr(tmp_path):
    records=[]
    for lr,val,wrong in [(.0003,.82,3),(.001,.9,4),(.0015,.85,0)]:
        for seed in range(42,52):
            r=evidence(seed,model="std_cnn",study="B5",arm=f"lr_{lr:g}",lr=lr,wrong=wrong)
            r["best_val_macro_f1"]=val;r["history"][0]["val_macro_f1"]=val
            records.append(r)
    z,d=archive(tmp_path,records)
    paths=aggregate(tmp_path,[z],d)
    selection=pd.read_csv(paths["lr_selection"])
    assert set(selection.selected_lr)=={.001}
    assert selection.set_index("lr").loc[.001,"selected"]
    assert not selection.set_index("lr").loc[.0015,"selected"]
    complete=pd.read_csv(paths["completeness"]).query('row_type == "study"').set_index("study")
    assert complete.loc["B5","status"]=="complete"


def test_b2_pairs_split_seeds_and_bootstraps_within_each_split(tmp_path):
    records=seed_block("B2",42)+seed_block("B2",43)
    records.extend(r for r in seed_block("B2",44) if r["task"]=="binary")
    z,d=archive(tmp_path,records)
    paths=aggregate(tmp_path,[z],d)
    paired=pd.read_csv(paths["paired_tests"])
    assert set(paired.n_pairs)=={2}
    flows=pd.read_csv(paths["paired_flow_bootstrap"])
    assert set(flows.pairing)=={"PAIRED"}
    assert set(flows.n_seeds)=={2}
    assert set(flows.seed_unit)=={"split_seed (train seed equals split seed)"}


def test_device_equivalence_has_side_by_side_ci_without_inference_labels(tmp_path):
    z,d=archive(tmp_path,seed_block("B1",42)+seed_block("B1",43)+[evidence(44)])
    directory=tmp_path/"cpu";directory.mkdir()
    records=[dict(task="binary",model="exlightnet",seed=s,macro_f1=.8+s/10000,
                  data_fingerprint="a"*64) for s in [42,43,44]]
    (directory/"results.json").write_text(json.dumps(records))
    (directory/"manifest.json").write_text(json.dumps({"mode":"full","seeds":list(range(42,47))}))
    paths=aggregate_studies([z],d,tmp_path/"out",tmp_path/"Report.md",directory/"results.json",protocol_path=PROTOCOL,replicates=25)
    table=pd.read_csv(paths["device_equivalence"])
    assert not set(table.columns)&{"p_value","p_holm","interpretation","pass","status"}
    assert set(table.reference_campaign)=={"cpu_v1"}
    assert set(table.comparison_campaign)=={"gpu_v2"}
    assert set(table.n_pairs)=={2}


def test_fixed30_completed_record_cannot_claim_early_stop(tmp_path):
    r=evidence(study="B3",arm="fixed30")
    r["history"]=r["history"][:2];r["epochs_run"]=2
    z,d=archive(tmp_path,[r])
    with pytest.raises(ValueError,match="fixed30"):
        aggregate(tmp_path,[z],d)


def test_corrupt_incremental_duplicate_artifact_not_hidden_by_good_copy(tmp_path):
    good,d=archive(tmp_path,[evidence()],"good.zip")
    def corrupt(members):
        path=next(p for p in members if p.endswith("predictions.npz"))
        members[path]=b"changed"
    bad,_=archive(tmp_path,[evidence()],"bad.zip",transform=corrupt)
    with pytest.raises(ValueError,match="artifact"):
        aggregate(tmp_path,[good,bad],d)


def test_b1_correlation_bootstrap_resamples_seed_pairs(tmp_path):
    records=[]
    for seed,length,wrong in [(42,7,3),(43,8,2),(44,9,1)]:
        block=seed_block("B1",seed)
        r=next(r for r in block if r["task"]=="binary" and r["model"]=="exlightnet")
        r["_prediction"]=evidence(wrong=wrong)["_prediction"]
        best=length-6
        r["history"]=[dict(epoch=i,train_loss=.2,val_macro_f1=.8 if i==best else .7,lr=.0015,seconds=.1) for i in range(1,length+1)]
        r["best_epoch"]=best
        r["epochs_run"]=length;records.extend(block)
    incomplete=set_history(evidence(45,wrong=5),[.7]*4+[.8]+[.7]*6)
    records.extend([incomplete,evidence(45,model="mlp")])
    z,d=archive(tmp_path,records)
    paths=aggregate(tmp_path,[z],d)
    table=pd.read_csv(paths["correlations"])
    assert table.iloc[0].n==3
    assert table.iloc[0].spearman_rho==pytest.approx(1)
    assert table.iloc[0].defined_replicates>0
    assert table.iloc[0].ci_low==pytest.approx(1)
    assert table.iloc[0].ci_high==pytest.approx(1)


def test_b1_merges_shards_before_admitting_only_complete_seed_blocks(tmp_path):
    records=seed_block("B1",42)+seed_block("B1",43)
    records=[r for r in records if not (r["seed"]==43 and r["task"]=="multiclass" and r["model"]=="xgboost")]
    neural,d=archive(tmp_path,[r for r in records if r["device"]=="cuda"],"neural.zip")
    trees,_=archive(tmp_path,[r for r in records if r["device"]=="cpu"],"trees.zip")
    for shard in (neural,trees):
        paths=aggregate(tmp_path,[shard],d)
        assert pd.read_csv(paths["paired_tests"]).empty
    paths=aggregate(tmp_path,[neural,trees],d)
    assert set(pd.read_csv(paths["paired_tests"]).seeds)=={42}
    assert set(pd.read_csv(paths["paired_flow_bootstrap"]).n_seeds)=={1}
    completeness=pd.read_csv(paths["completeness"],dtype={"eligible_seeds":"string"}).query('study == "B1"')
    summary=completeness.query('row_type == "study"').iloc[0]
    assert summary.eligible_seeds=="42"
    missing=completeness[completeness.seed==43].iloc[0]
    assert missing.status=="incomplete; excluded"
    assert missing.missing_count==1
    assert json.loads(missing.missing_identities)==[
        {"study":"B1","task":"multiclass","model":"xgboost","seed":43,"split_seed":42,"arm":"paper"}]
    summaries=pd.read_csv(paths["summaries"]).query('study == "B1" and task == "binary"')
    assert set(summaries.n)=={2}
    assert set(summaries.n_eligible)=={1}
    assert set(summaries.observation_scope)=={"partial; observed runs only; descriptive"}


def test_b3_excludes_seeds_missing_an_unrelated_reference_or_arm(tmp_path):
    records=sum((seed_block("B3",seed) for seed in [42,43,44]),[])
    records=[r for r in records if not (
        (r["seed"]==43 and r["study"]=="B1" and r["model"]=="std_cnn") or
        (r["seed"]==44 and r["arm"]=="patience10_min15" and r["model"]=="mlp"))]
    z,d=archive(tmp_path,records)
    paths=aggregate(tmp_path,[z],d)
    pairs=pd.read_csv(paths["paired_tests"]).query('study == "B3"')
    assert set(pairs.seeds)=={42}
    assert set(pd.read_csv(paths["paired_flow_bootstrap"]).query('study == "B3"').n_seeds)=={1}
    missing=pd.read_csv(paths["completeness"]).query('study == "B3" and seed in [43,44]').set_index("seed")
    assert set(missing.status)=={"incomplete; excluded"}
    assert set(missing.missing_count)=={1}
    assert json.loads(missing.loc[43,"missing_identities"])[0]["study"]=="B1"
    assert json.loads(missing.loc[44,"missing_identities"])[0]["arm"]=="patience10_min15"
    hypothesis=pd.read_csv(paths["early_stopping_hypothesis"])
    assert set(hypothesis.n_pairs)=={1}
    assert set(hypothesis.overall_decision)=={"PENDING"}


def test_b3_full_blocks_apply_primary_last_and_secondary_best_to_verified_checkpoints(tmp_path):
    records=[]
    for seed in range(42,57):
        for record in seed_block("B3",seed):
            if record["model"]=="exlightnet" and record["arm"]=="paper":
                record=evidence(seed,task="multiclass",wrong=3+(seed-42)%7,last_wrong=0)
            if record["model"]=="exlightnet" and record["arm"]=="fixed30":
                record=evidence(seed,study="B3",arm="fixed30",wrong=11,last_wrong=0)
            records.append(record)
    z,d=archive(tmp_path,records)
    paths=aggregate(tmp_path,[z],d)
    table=pd.read_csv(paths["early_stopping_hypothesis"]).set_index("endpoint_role")
    primary,secondary=table.loc["PRIMARY"],table.loc["SECONDARY"]
    assert primary.n_pairs==secondary.n_pairs==15
    assert primary.fixed30_mean==1
    assert primary.paper_mean==secondary.paper_mean<1
    assert primary.endpoint_decision=="conditions supported"
    assert secondary.endpoint_decision=="not supported"
    assert set(table.overall_decision)=={"inconclusive"}
    assert set(table.bootstrap_replicates)=={2000}  # CLI flow count is 50.
    assert set(table.b1_n)=={0}  # B3 references suffice for B3, not for B1 correlation.


def test_probe_projects_explicit_assumptions_and_determinism_verifies_logits(tmp_path):
    records=[evidence(study="probe",arm="probe"),evidence(study="determinism",arm="repeat1"),
             evidence(study="determinism",arm="repeat2",wrong=2)]
    z,d=archive(tmp_path,records)
    paths=aggregate(tmp_path,[z],d)
    probe=pd.read_csv(paths["probe"])
    assert set(probe.study)=={"B1","B2","B3","B4","B5"}
    assert set(probe.seconds_per_epoch)=={.1}
    assert probe.assumptions.str.contains("ignores trees").all()
    det=pd.read_csv(paths["determinism"])
    assert len(det)==2
    assert set(det.max_abs_logit_difference)=={1}


def test_saved_metrics_forged_consistently_still_must_match_predictions(tmp_path):
    def forged(members):
        recordpath=next(p for p in members if p.endswith("record.json")); record=json.loads(members[recordpath])
        item=record["checkpoints"]["best"]
        metrics=json.loads(members[item["metrics_path"]]);metrics["macro_f1"]=.123
        members[item["metrics_path"]]=encoded(metrics)
        item["metrics"]=metrics;item["metrics_sha256"]=sha(members[item["metrics_path"]])
        receipt=json.loads(members[item["receipt_path"]]);receipt["metrics_sha256"]=item["metrics_sha256"]
        members[item["receipt_path"]]=encoded(receipt);item["receipt_sha256"]=sha(members[item["receipt_path"]])
        members[recordpath]=encoded(record)
    z,d=archive(tmp_path,[evidence()],transform=forged)
    with pytest.raises(ValueError,match="metrics"):
        aggregate(tmp_path,[z],d)


def test_local_prepared_tampering_rejected(tmp_path):
    z,d=archive(tmp_path,[evidence()])
    (d/"split_42.npz").write_bytes(b"bad")
    with pytest.raises(ValueError,match="prepared"):
        aggregate(tmp_path,[z],d)


def set_history(record,scores):
    record["history"]=[dict(epoch=i,train_loss=.2,val_macro_f1=value,lr=.0015,seconds=.1) for i,value in enumerate(scores,1)]
    record["epochs_run"]=len(scores)
    record["best_epoch"]=int(np.argmax(scores))+1
    record["best_val_macro_f1"]=max(scores)
    return record


@pytest.mark.parametrize("study,arm",[("B1","paper"),("B2","paper"),("B4","uncapped60"),("B5","lr_0.0015")])
def test_premature_complete_ordinary_histories_are_rejected(study,arm):
    r=evidence(study=study,arm=arm)
    set_history(r,[.8,.7])
    with pytest.raises(ValueError,match="premature"):
        _validate_history(r)


def test_history_cannot_continue_past_earliest_legal_patience_stop():
    r=set_history(evidence(),[.8]+[.7]*7)
    with pytest.raises(ValueError,match="past"):
        _validate_history(r)
    # A late improvement cannot legitimize an epoch which should not run.
    r=set_history(evidence(),[.8]+[.7]*6+[.9])
    with pytest.raises(ValueError,match="past"):
        _validate_history(r)


@pytest.mark.parametrize("kind",["patience","cap","min15","fixed30","probe","smoke"])
def test_legal_stop_boundaries(kind):
    if kind=="patience":r=evidence()
    elif kind=="cap":r=set_history(evidence(),np.linspace(.1,.9,20))
    elif kind=="min15":r=evidence(study="B3",arm="patience10_min15")
    elif kind=="fixed30":r=evidence(study="B3",arm="fixed30")
    elif kind=="probe":r=evidence(study="probe",arm="probe")
    else:
        r=set_history(evidence(campaign="gpu_v2_smoke"),[.8,.7]);r["config"]["max_epochs"]=2
    _validate_history(r)


def test_probe_and_min15_boundary_premature_and_overrun():
    r=set_history(evidence(study="probe",arm="probe"),[.8,.7])
    with pytest.raises(ValueError,match="probe"):_validate_history(r)
    for length in [14,16]:
        r=set_history(evidence(study="B3",arm="patience10_min15"),[.8]+[.7]*(length-1))
        with pytest.raises(ValueError):_validate_history(r)


@pytest.mark.parametrize("field,value",[("gpu_name","GPU B"),("torch","other"),("cuda","other"),("cudnn","other"),("numpy","other")])
def test_cuda_hardware_or_runtime_mismatch_rejected(tmp_path,field,value):
    a,b=evidence(),evidence(seed=43)
    b["libraries"][field]=value
    z,d=archive(tmp_path,[a,b])
    with pytest.raises(ValueError,match="environment"):
        aggregate(tmp_path,[z],d)


def test_tree_environment_ignores_unused_gpu_but_checks_relevant_runtime(tmp_path):
    a,b=evidence(model="random_forest",device="cpu"),evidence(seed=43,model="random_forest",device="cpu")
    b["libraries"].update(gpu_name="GPU B",cuda="unused",cudnn="unused")
    z,d=archive(tmp_path,[a,b]);loaded=read_study_archives([z],d)
    assert len(loaded["records"])==2
    b["libraries"]["scikit-learn"]="different"
    z,d=archive(tmp_path,[a,b],"different.zip")
    with pytest.raises(ValueError,match="environment"):
        aggregate(tmp_path,[z],d)


def test_environment_provenance_and_sorted_binary_mapping(tmp_path):
    z,d=archive(tmp_path,[evidence(),evidence(seed=43)])
    loaded=read_study_archives([z],d)
    # Fixture JSON is sorted: attack serializes first, but class0 remains normal.
    assert list(loaded["records"][0]["checkpoints"]["best"]["metrics"]["per_class"])[0]=="attack"
    paths=aggregate(tmp_path,[z],d)
    table=pd.read_csv(paths["per_run"])
    assert set(table.gpu_name)=={"GPU A"}
    assert table.environment_id.nunique()==1
    manifest=json.loads(Path(paths["manifest"]).read_text())
    assert len(manifest["execution_environments"])==1


def test_cuda_record_requires_actual_gpu_identity(tmp_path):
    r=evidence();r["libraries"].pop("gpu_name")
    z,d=archive(tmp_path,[r])
    with pytest.raises(ValueError,match="environment"):
        aggregate(tmp_path,[z],d)
